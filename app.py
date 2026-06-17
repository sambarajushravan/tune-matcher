import streamlit as st
import librosa
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import io
import os
import re
import hashlib
import datetime
import pandas as pd
import time
import threading
import uuid

# --- APP CONFIG ---
st.set_page_config(page_title="Sataka Sankharavam Tune Matcher", page_icon="🎤")
st.title("🎤 Sataka Sankharavam Tune Matcher Challenge")
st.write("Match the tune and timing at 85% or higher to pass!")

# --- SERVER-SIDE SESSION TRACKING (single Streamlit process; best-effort on Community Cloud) ---
# Counts browser sessions active within ACTIVE_SESSION_WINDOW_SEC. When congested, Analyze
# is blocked so heavy librosa work doesn't pile up on the shared container.
MAX_CONCURRENT_SESSIONS = 40
INACTIVITY_LOGOUT_SEC = 600       # 10 minutes without interaction → auto log out
ACTIVE_SESSION_WINDOW_SEC = 300   # sessions seen in last 5 min count as "online now"

_active_lock = threading.Lock()
_active_sessions = {}  # session_id -> last_seen unix timestamp


def _session_id():
    if "_sid" not in st.session_state:
        st.session_state._sid = str(uuid.uuid4())
    return st.session_state._sid


def _prune_active_sessions(cutoff):
    stale = [k for k, t in _active_sessions.items() if t < cutoff]
    for k in stale:
        del _active_sessions[k]


def _touch_active_session():
    sid = _session_id()
    now = time.time()
    with _active_lock:
        _active_sessions[sid] = now
        _prune_active_sessions(now - INACTIVITY_LOGOUT_SEC)


def _unregister_active_session():
    sid = st.session_state.get("_sid")
    if not sid:
        return
    with _active_lock:
        _active_sessions.pop(sid, None)


def _active_session_count():
    now = time.time()
    cutoff = now - ACTIVE_SESSION_WINDOW_SEC
    with _active_lock:
        return sum(1 for t in _active_sessions.values() if t >= cutoff)


def _server_congested():
    return _active_session_count() >= MAX_CONCURRENT_SESSIONS


def _logout_user():
    """Clear participant session state and drop this browser from the active registry."""
    _unregister_active_session()
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.registration_id = None
    st.session_state.attempt_counts = {}
    for key in list(st.session_state.keys()):
        if (key.startswith("_audio_hash_") or key.startswith("_last_wrong_")
                or key in ("_active_song", "_rec_nonce")):
            st.session_state.pop(key, None)
    st.session_state.pop("completed_songs", None)
    st.session_state.pop("final_done", None)
    st.session_state.pop("final_score", None)
    st.session_state.pop("_last_activity", None)
    st.session_state.pop("_sid", None)


# --- GOOGLE SHEETS BACKEND (gspread, lightweight reads/writes) ---
# We talk to the sheet directly with gspread instead of rewriting the whole sheet on
# every save. This keeps memory/API usage low and avoids "two people qualify at once
# overwrites one row" (lost updates), which the old read-all/write-all pattern risked.
# NOTE: the sheet columns MUST stay in this exact left-to-right order (A..H), because
# row writes target the A:H range positionally.
SHEET_COLUMNS = ["User ID", "Registration ID", "Song", "Score",
                 "Status", "Last Attempt", "Voice ID", "Voice Print"]


@st.cache_resource(show_spinner=False)
def _get_worksheet():
    """Authorize a gspread worksheet handle from the service-account secrets.
    Cached as a resource so we authorize once per app process (not per rerun)."""
    cfg = dict(st.secrets["connections"]["gsheets"])
    spreadsheet_url = cfg.pop("spreadsheet")
    worksheet_name = cfg.pop("worksheet", "Sheet1")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(cfg, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(spreadsheet_url).worksheet(worksheet_name)


def _read_records():
    """Live read of the whole sheet as a list of header-keyed dicts (1 API call)."""
    return _get_worksheet().get_all_records()


def _get_user_progress(user_id):
    """One read of the sheet -> (qualified song->score dict, final_done, final_score)."""
    completed, final_done, final_score = {}, False, None
    for r in _read_records():
        if (str(r.get("User ID", "")).strip() == user_id
                and str(r.get("Status", "")).strip() == "QUALIFIED"):
            s = str(r.get("Song", "")).strip()
            try:
                sc = float(r.get("Score"))
            except (TypeError, ValueError):
                sc = None
            if s == FINAL_TEST_KEY:
                final_done = True
                if sc is not None:
                    final_score = sc
            elif s:
                if sc is not None:
                    completed[s] = max(completed.get(s, 0.0), sc)
                elif s not in completed:
                    completed[s] = None
    return completed, final_done, final_score


def _final_test_enabled():
    """Final Test (all-songs combined) is OFF unless [features] final_test is truthy
    in secrets. Lets you turn it on/off on demand without a code change."""
    try:
        val = st.secrets["features"]["final_test"]
    except (KeyError, FileNotFoundError):
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


# Scoring prioritizes pronunciation (articulation MFCC) + pacing.
TEMPO_TOLERANCE = 0.90
TEMPO_WEIGHT = 0.40
GOOD_MATCH_DIST = 1.30       # articulation MFCC anchor (coeffs 4–12)
PENALTY_SLOPE = 85.0
MIN_PRONUNCIATION_PCT = 88     # must sing words clearly — humming/mumbling blocked
MIN_ARTICULATION_RATIO = 0.68  # user syllable variation vs reference (hums are too flat)

# MFCC rows used for pass/fail (skip c0–c3: energy/broad spectrum — humming matches these).
_ARTICULATION_ROWS = slice(4, 13)
# Other poem must be clearly closer than selected (absolute + relative margin).
_WRONG_SONG_MARGIN = 0.05
_WRONG_SONG_REL_RATIO = 0.94
# If identity distances tie within this epsilon, trust the dropdown selection.
_IDENTITY_TIE_EPS = 0.025


def _dtw_norm_dist(X, Y):
    """Band-constrained DTW distance normalized by path length."""
    try:
        D, wp = librosa.sequence.dtw(
            X, Y, metric='euclidean', backtrack=True,
            global_constraints=True, band_rad=0.1,
        )
    except Exception:
        D, wp = librosa.sequence.dtw(X, Y, metric='euclidean', backtrack=True)
    pl = len(wp)
    return float(D[-1, -1] / pl) if pl > 0 else 100.0


def _song_identity_matrix(chroma, mfcc):
    """Chroma + full MFCC — discriminates WHICH padyam (lyrics/tune content).

    Chroma is weighted higher because different padyams follow different note patterns;
    full MFCC (not just articulation) captures lyric shape, not just vowel timbre."""
    return np.vstack([chroma * 2.0, mfcc])


def _detect_wrong_song(song_key, chroma_user, mfcc_user_norm, available_songs):
    """Return (is_wrong, best_match_name, selected_dist, best_dist).

    Compares the recording against every reference. Flags wrong poem only when
    another padyam is a clearly better match than the dropdown selection."""
    ident_user = _song_identity_matrix(chroma_user, mfcc_user_norm)
    dists = {}
    for name, path in available_songs.items():
        mtime, fsize = _ref_file_key(path)
        _, _, mfcc_r, chroma_r, _ = _reference_features(
            path, mtime, fsize, SONGS_CACHE_VERSION)
        dists[name] = _dtw_norm_dist(_song_identity_matrix(chroma_r, mfcc_r), ident_user)

    selected_dist = dists[song_key]
    best_name = min(dists, key=dists.get)
    best_dist = dists[best_name]
    wrong = False
    if best_name != song_key and abs(selected_dist - best_dist) > _IDENTITY_TIE_EPS:
        if (best_dist < selected_dist * _WRONG_SONG_REL_RATIO
                and best_dist + _WRONG_SONG_MARGIN < selected_dist):
            wrong = True
    return wrong, best_name, selected_dist, best_dist


# Bump when reference .wav files change (trim/replace) to bust stale Streamlit caches.
SONGS_CACHE_VERSION = 2


def _ref_file_key(path):
    """Cache-bust key from file identity (mtime + size)."""
    return os.path.getmtime(path), os.path.getsize(path)


@st.cache_data(show_spinner=False, max_entries=32)
def _reference_features(ref_path, file_mtime, file_size, cache_version):
    """Load + featurize a reference track ONCE and cache it across all sessions.
    Returns (duration_ref, hop_len, mfcc_ref, chroma_ref, f0_ref).

    file_mtime + file_size + cache_version are part of the cache key so trimmed or
    replaced .wav files invalidate stale entries on deploy."""
    y_ref, sr_ref = librosa.load(ref_path, sr=22050)
    duration_ref = librosa.get_duration(y=y_ref, sr=sr_ref)
    hop_len = 512 if duration_ref <= 90 else 2048

    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13, hop_length=hop_len)
    chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref, hop_length=hop_len)
    f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'), sr=sr_ref, hop_length=hop_len)
    f0_ref = np.nan_to_num(f0_ref)
    if np.std(f0_ref) > 0:
        f0_ref = (f0_ref - np.mean(f0_ref)) / (np.std(f0_ref) + 1e-6)
    f0_ref = f0_ref.reshape(1, -1)
    mfcc_ref = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-6)
    return duration_ref, hop_len, mfcc_ref, chroma_ref, f0_ref


def _load_reference(path):
    """Load cached reference features; pacing uses live file duration (matches top UI).

    If a stale cache entry slips through (cached duration != file duration), clear
    the cache once and reload."""
    mtime, fsize = _ref_file_key(path)
    cached_dur, hop_len, mfcc_ref, chroma_ref, f0_ref = _reference_features(
        path, mtime, fsize, SONGS_CACHE_VERSION)
    duration_ref = librosa.get_duration(path=path)
    if abs(cached_dur - duration_ref) > 0.5:
        _reference_features.clear()
        _, hop_len, mfcc_ref, chroma_ref, f0_ref = _reference_features(
            path, mtime, fsize, SONGS_CACHE_VERSION)
    return duration_ref, hop_len, mfcc_ref, chroma_ref, f0_ref


def _articulation_mfcc(mfcc):
    """Higher MFCC coefficients — sensitive to consonants/vowels; humming lacks these."""
    return mfcc[_ARTICULATION_ROWS]


def _articulation_ratio(mfcc_ref, mfcc_user):
    """Frame-to-frame MFCC variation vs reference. Humming/mumbling scores low."""
    ref_v = float(np.mean(np.std(mfcc_ref, axis=1)))
    user_v = float(np.mean(np.std(mfcc_user, axis=1)))
    return user_v / (ref_v + 1e-6)


def _qualified_label(name, score):
    """Dropdown / list label for a passed song, including score when known."""
    if score is None:
        return f"✅ {name}"
    return f"✅ {name} ({score:.1f}%)"


# --- PERFORMANCE FEEDBACK (user-facing coaching, separate from final score math) ---
_LAYER_ANCHORS = {
    "pronunciation": (1.15, 1.85),   # articulation MFCC (stricter — catches humming)
    "notes": (0.55, 1.10),
    "melody": (0.45, 0.95),
}


def _path_layer_distance(X, Y, wp, rows):
    """Average frame distance along the DTW path for one feature layer."""
    dists = []
    for r_idx, u_idx in wp:
        dists.append(float(np.linalg.norm(X[rows, r_idx] - Y[rows, u_idx])))
    return float(np.mean(dists)) if dists else 100.0


def _layer_to_pct(dist, good, bad):
    """Map a layer distance to a user-friendly 0-100% (higher = closer match)."""
    if dist <= good:
        return round(100.0 - (dist / good) * 8.0, 0)
    span = max(bad - good, 1e-6)
    return round(max(0.0, 92.0 - ((dist - good) / span) * 55.0), 0)


def _tempo_feedback(duration_ref, duration_user, time_ratio):
    """Return (status, user_message) for pacing."""
    if duration_ref <= 0:
        return "good", "Pacing matches the reference."
    delta_pct = (duration_user - duration_ref) / duration_ref * 100.0
    if time_ratio >= 0.85:
        return "good", (
            f"Pacing matches the reference well "
            f"({int(round(duration_user))}s sung vs {int(round(duration_ref))}s target)."
        )
    if duration_user > duration_ref * 1.05:
        return "slow", (
            f"You sang **slower** than the reference — about {abs(delta_pct):.0f}% longer "
            f"({int(round(duration_user))}s vs {int(round(duration_ref))}s). "
            f"Try keeping a steadier pace."
        )
    if duration_user < duration_ref * 0.95:
        return "fast", (
            f"You sang **faster** than the reference — about {abs(delta_pct):.0f}% shorter "
            f"({int(round(duration_user))}s vs {int(round(duration_ref))}s). "
            f"Slow down to match the reference."
        )
    return "ok", (
        f"Pacing is close ({int(round(duration_user))}s vs {int(round(duration_ref))}s), "
        f"but tightening rhythm may help your score."
    )


def _coaching_tips(pron_pct, notes_pct, melody_pct, tempo_status, score, clear_words):
    """Short, actionable suggestions — prioritizes pronunciation and pacing."""
    tips = []
    if not clear_words:
        tips.append("**Sing every word clearly** — humming or mumbling the tune alone won't qualify.")
    if tempo_status == "slow":
        tips.append("Practice with the reference playing — finish each line before the reference ends.")
    elif tempo_status == "fast":
        tips.append("Don't rush — sing each word clearly at the reference tempo.")
    elif tempo_status == "ok":
        tips.append("Your length is close; try matching the reference beat-for-beat.")
    if pron_pct < MIN_PRONUNCIATION_PCT:
        tips.append("Focus on **clear pronunciation** — listen to how each syllable is shaped in the reference.")
    elif pron_pct < 85:
        tips.append("Pronunciation needs work — sing along with the reference word-for-word, not just the melody.")
    if notes_pct < 60:
        tips.append("(Optional) Musical notes differ — for polish, hum the tune first, then sing the words.")
    if melody_pct < 60:
        tips.append("(Optional) Melody line differs — follow the reference's rise and fall for polish.")
    if not tips and score < 85:
        tips.append("Focus on **pronunciation and pacing** — those drive your score.")
    if not tips:
        tips.append("Great job — pronunciation and pacing look solid!")
    return tips


def _build_performance_feedback(mfcc_ref, mfcc_user, chroma_ref, chroma_user,
                                  f0_ref, f0_user, wp,
                                  duration_ref, duration_user, time_ratio, score,
                                  clear_words):
    """User-facing breakdown + coaching tips (Raw Dist kept out of main UI)."""
    pron_d = _path_layer_distance(_articulation_mfcc(mfcc_ref), _articulation_mfcc(mfcc_user),
                                  wp, slice(0, 9))
    chroma_d = _path_layer_distance(chroma_ref, chroma_user, wp, slice(0, chroma_ref.shape[0]))
    pitch_d = _path_layer_distance(f0_ref, f0_user, wp, slice(0, 1))

    pron_pct = _layer_to_pct(pron_d, *_LAYER_ANCHORS["pronunciation"])
    notes_pct = _layer_to_pct(chroma_d, *_LAYER_ANCHORS["notes"])
    melody_pct = _layer_to_pct(pitch_d, *_LAYER_ANCHORS["melody"])
    tempo_status, tempo_msg = _tempo_feedback(duration_ref, duration_user, time_ratio)
    tips = _coaching_tips(pron_pct, notes_pct, melody_pct, tempo_status, score, clear_words)

    return {
        "pronunciation_pct": pron_pct,
        "notes_pct": notes_pct,
        "melody_pct": melody_pct,
        "tempo_status": tempo_status,
        "tempo_msg": tempo_msg,
        "tips": tips,
    }

# --- NORMALIZATION HELPERS (for forgiving login matching) ---
def _norm_name(value):
    """Lowercase + collapse any run of whitespace to a single space + strip.
    So 'Sahitya   Malladi' == 'sahitya malladi'."""
    return " ".join(str(value).split()).lower()


def _norm_pwd(value):
    """Lowercase + remove all whitespace, so case and stray spaces don't matter."""
    return "".join(str(value).split()).lower()


@st.cache_data(ttl=300, show_spinner=False)
def _get_roster():
    """Cached list of (normalized_name, normalized_pwd, canonical_name, reg_id).
    Refreshes every 5 minutes and is shared across all sessions, so we don't re-read
    the whole sheet on every single login attempt. Trade-off: a participant added to
    the roster mid-event may take up to 5 minutes to be able to log in."""
    roster = []
    for r in _read_records():
        name = str(r.get("User ID", "")).strip()
        reg = str(r.get("Registration ID", "")).strip()
        if name and reg:
            roster.append((_norm_name(name), _norm_pwd(reg), name, reg))
    return roster


def authenticate(username, password):
    """Validate name + Registration ID against the cached roster.
    Returns (canonical_user_id, registration_id) on success, else None."""
    target_name = _norm_name(username)
    target_pwd = _norm_pwd(password)
    for norm_name, norm_pwd, canonical_name, reg_id in _get_roster():
        if norm_name == target_name and norm_pwd == target_pwd:
            return canonical_name, reg_id
    return None


# --- VOICE FINGERPRINT HELPERS (best-effort, for admin duplicate-singer review) ---
# NOTE: this is a lightweight timbre signature (MFCC mean+std), NOT biometric-grade
# speaker verification. It is content-dependent (everyone sings the same songs), so it
# is only reliable enough to be a REVIEW HINT — not proof. Voice IDs are scoped PER
# FAMILY (Registration ID): within one family the first singer is "Voice 1", a clearly
# different singer is "Voice 2", etc. If the same person sings under two names in the
# same family, both rows land on the same Voice ID, which the admin audit surfaces.
#
# Tuning: this threshold needs calibrating against real recordings. If genuinely
# different family members keep collapsing onto one Voice ID, raise it; if one person's
# takes keep splitting into many voices, lower it.
VOICE_MATCH_THRESHOLD = 0.95  # cosine similarity above which two prints are "same voice"


def _voice_signature(mfcc):
    """Compact, L2-normalized timbre signature from MFCC mean + std over time.

    Coefficient 0 (overall loudness/energy) is dropped on purpose: it is dominated by
    the SONG (which everyone sings identically) rather than the singer, and including it
    made every print point in nearly the same direction — collapsing everyone onto
    'Voice 1'. Dropping it lets the higher coefficients (vocal-tract timbre) actually
    discriminate between people."""
    if mfcc.shape[0] > 1:
        mfcc = mfcc[1:]  # drop c0 (energy) — keep the timbre coefficients
    sig = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig


def _voice_to_str(sig):
    return ",".join(f"{x:.4f}" for x in sig)


def _parse_voice(text):
    try:
        arr = np.array([float(x) for x in str(text).split(",") if x.strip() != ""])
        return arr if arr.size else None
    except Exception:
        return None


def _assign_voice_id(df, new_sig, reg_id):
    """Nearest-neighbour match against stored Voice Prints WITHIN THE SAME FAMILY.

    Comparison and 'Voice N' numbering are both scoped to the given Registration ID, so:
      - a family of 4 distinct singers tends to get Voice 1..Voice 4,
      - one person singing under two names in that family lands on a single Voice ID
        (the cheating signal the admin audit flags),
      - and different families never contaminate each other (every family starts at
        Voice 1, which is why the audit groups by Registration ID + Voice ID).

    Returns (voice_id, matched_user_id_or_None)."""
    target_reg = _norm_pwd(reg_id)
    best_sim, best_id, best_user = -1.0, None, None
    max_n = 0
    if "Voice ID" in df.columns and "Voice Print" in df.columns:
        for _, r in df.iterrows():
            if _norm_pwd(r.get("Registration ID", "")) != target_reg:
                continue  # only compare within the same family (Registration ID)
            vid = str(r.get("Voice ID", "")).strip()
            if vid:
                m = re.match(r"[Vv]oice\s+(\d+)", vid)
                if m:
                    max_n = max(max_n, int(m.group(1)))
            vp = _parse_voice(r.get("Voice Print", ""))
            if vp is not None and vp.shape == new_sig.shape:
                sim = float(np.dot(vp, new_sig))  # both L2-normalized => cosine similarity
                if sim > best_sim:
                    best_sim, best_id, best_user = sim, vid, str(r.get("User ID", "")).strip()
    if best_id and best_sim >= VOICE_MATCH_THRESHOLD:
        return best_id, best_user
    return f"Voice {max_n + 1}", None


def _save_qualification(user_id, reg_id, song_key, score, voice_sig,
                        is_final_test, final_key):
    """Upsert a qualifying result with a targeted write (update one row or append one
    row) instead of rewriting the whole sheet.

    Returns (saved, prev_score, songs_passed):
      - saved=True  -> row was written (new best / new song / placeholder filled)
      - saved=False -> kept the existing higher score; prev_score is that score
      - songs_passed -> distinct individual songs (final test excluded) now qualified
    Raises on any Sheets failure so the caller can show a friendly warning."""
    ws = _get_worksheet()
    records = ws.get_all_records()

    # Build a DataFrame purely so the existing voice-id helper can scan prior prints.
    df = pd.DataFrame(records)
    for col in SHEET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    voice_id, _matched_user = _assign_voice_id(df, voice_sig, reg_id)
    voice_str = _voice_to_str(voice_sig)

    def _row_values(reg):
        return [user_id, reg, song_key, score, "QUALIFIED", timestamp, voice_id, voice_str]

    # Find an existing (user, song) row, else a blank placeholder row for this user.
    existing_i, placeholder_i = None, None
    for i, r in enumerate(records):
        ru = str(r.get("User ID", "")).strip()
        rs = str(r.get("Song", "")).strip()
        if ru == user_id and rs == song_key:
            existing_i = i
            break
    if existing_i is None:
        for i, r in enumerate(records):
            ru = str(r.get("User ID", "")).strip()
            rs = str(r.get("Song", "")).strip()
            if ru == user_id and rs in ("", "nan", "None"):
                placeholder_i = i
                break

    saved = True
    prev_score = None
    if existing_i is not None:
        prev = records[existing_i]
        try:
            prev_score = float(prev.get("Score"))
        except (TypeError, ValueError):
            prev_score = None
        prev_status = str(prev.get("Status", "")).strip()
        if prev_status == "QUALIFIED" and prev_score is not None and score <= prev_score:
            saved = False  # existing best is higher; leave the row untouched
        else:
            reg = str(prev.get("Registration ID", "")).strip() or reg_id
            row_num = existing_i + 2  # +1 header row, +1 for 1-based indexing
            ws.update(range_name=f"A{row_num}:H{row_num}",
                      values=[_row_values(reg)], value_input_option="RAW")
    elif placeholder_i is not None:
        row_num = placeholder_i + 2
        ws.update(range_name=f"A{row_num}:H{row_num}",
                  values=[_row_values(reg_id)], value_input_option="RAW")
    else:
        ws.append_row(_row_values(reg_id), value_input_option="RAW")

    # Distinct individual songs (exclude the final test) this user has qualified for.
    qualified_songs = set()
    for r in records:
        if (str(r.get("User ID", "")).strip() == user_id
                and str(r.get("Status", "")).strip() == "QUALIFIED"):
            s = str(r.get("Song", "")).strip()
            if s and s != final_key:
                qualified_songs.add(s)
    if not is_final_test:
        qualified_songs.add(song_key)  # idempotent if it was already counted

    return saved, prev_score, len(qualified_songs)


# --- 1. LOGIN GATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.registration_id = None

if not st.session_state.authenticated:
    st.subheader("🔐 Participant Login")
    with st.form("login_form"):
        login_name = st.text_input("Name (as registered):")
        login_pwd = st.text_input("Registration ID:", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        try:
            result = authenticate(login_name, login_pwd)
        except Exception:
            st.error("Could not reach the registration list right now. Please try again in a moment.")
        else:
            if result:
                st.session_state.authenticated = True
                st.session_state.user_id = result[0]
                st.session_state.registration_id = result[1]
                st.session_state.attempt_counts = {}  # fresh attempt tally per login
                st.session_state.pop("completed_songs", None)
                st.session_state.pop("final_done", None)
                st.session_state.pop("final_score", None)
                st.session_state._sid = str(uuid.uuid4())
                st.session_state._last_activity = time.time()
                _touch_active_session()
                st.rerun()
            else:
                st.error("Name or Registration ID is incorrect. Please check and try again.")

# --- ADMIN REPORTING SECTION (always available, gated by its own password) ---
st.write("---")
with st.expander("📊 Admin Reports & Statistics (Internal Use Only)"):
    admin_password = st.text_input("Enter Admin Password:", type="password")

    if admin_password:
        # Fetch the password safely from Streamlit's secrets manager. Guard against a
        # missing [admin] secret or a Sheets read failure so the panel never hard-crashes.
        try:
            correct_password = st.secrets["admin"]["password"]
        except (KeyError, FileNotFoundError):
            st.error("Admin password is not configured. Add an [admin] password to the app secrets.")
            correct_password = None

        if correct_password is None:
            pass
        elif admin_password == correct_password:
            st.success("Access Granted! Fetching real-time reports...")

            try:
                admin_df = pd.DataFrame(_read_records())

                # Only completed passes count toward stats (skips blank roster rows).
                if "Status" in admin_df.columns:
                    passes_df = admin_df[admin_df["Status"] == "QUALIFIED"]
                else:
                    passes_df = admin_df.iloc[0:0]

                if not passes_df.empty:
                    total_passes = len(passes_df)
                    unique_singers = passes_df["User ID"].nunique()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Qualifications Logged", total_passes)
                    col2.metric("Total Unique Active Users", unique_singers)
                    col3.metric("Browsers Active Now (≈5 min)", _active_session_count())

                    st.subheader("🏆 Top Participant Progress")
                    leaderboard = passes_df["User ID"].value_counts().reset_index()
                    leaderboard.columns = ["User ID", "Songs Completed (Out of 18)"]
                    st.dataframe(leaderboard, use_container_width=True)

                    st.subheader("🎵 Completion Rates by Song")
                    song_counts = passes_df["Song"].value_counts()
                    st.bar_chart(song_counts)

                    # Voice Audit: within a family (Registration ID), flag a single
                    # Voice ID that qualified under more than one name — i.e. one person
                    # likely singing for relatives who share the Registration ID. Voice
                    # IDs are family-scoped, so we group by Registration ID + Voice ID
                    # ("Voice 1" exists in every family and is not itself suspicious).
                    # This is a REVIEW HINT, not proof — listen to the recordings before
                    # acting on it.
                    st.subheader("🕵️ Voice Audit (possible same singer within a family)")
                    if {"Voice ID", "Registration ID"}.issubset(passes_df.columns):
                        va = passes_df[passes_df["Voice ID"].astype(str).str.strip() != ""]
                        flagged_any = False
                        if not va.empty:
                            for (reg, vid), grp in va.groupby(["Registration ID", "Voice ID"]):
                                names = sorted(grp["User ID"].astype(str).str.strip().unique())
                                if len(names) > 1:
                                    flagged_any = True
                                    songs = sorted(grp["Song"].astype(str).str.strip().unique())
                                    st.error(
                                        f"Registration {reg} · {vid}: one voice qualified "
                                        f"under {len(names)} names → {', '.join(names)} "
                                        f"(songs: {', '.join(songs)})"
                                    )
                        if not flagged_any:
                            st.success("No within-family duplicate voices detected.")
                    else:
                        st.info("No voice fingerprints recorded yet.")

                    st.subheader("📋 Raw Activity Log")
                    log_cols = [c for c in passes_df.columns if c != "Voice Print"]
                    st.dataframe(passes_df[log_cols].sort_values(by="Last Attempt", ascending=False), use_container_width=True)
                else:
                    st.info("No qualifications logged yet. No stats to report.")
            except Exception as e:
                st.error("Could not load reports from Google Sheets right now. Please try again later.")
        else:
            st.error("Incorrect Password.")

# Inactivity timeout + heartbeat (before participant UI).
if st.session_state.authenticated:
    last = st.session_state.get("_last_activity")
    if last and time.time() - last > INACTIVITY_LOGOUT_SEC:
        _logout_user()
        st.warning(
            "You were logged out after **10 minutes** of inactivity. "
            "Please log in again. (Use **Log out** when switching to another family member.)"
        )
        st.rerun()
    st.session_state._last_activity = time.time()
    _touch_active_session()

# Stop here until the participant logs in (admin panel above still renders).
if not st.session_state.authenticated:
    st.stop()

# --- AUTHENTICATED PARTICIPANT FLOW ---
user_id = st.session_state.user_id
_active_now = _active_session_count()

header_col, logout_col = st.columns([3, 1])
header_col.success(f"Logged in as: {user_id}")
if logout_col.button("Log out"):
    _logout_user()
    st.rerun()

if _server_congested():
    st.warning(
        f"**High traffic right now** — about **{_active_now}** people are using the app. "
        f"Please **wait a few minutes** before pressing **Analyze**. "
        f"You can still listen to the reference and record your take. "
        f"This keeps scoring fast and reliable for everyone."
    )
else:
    st.caption(f"Active now: ~{_active_now} participant(s) (last few minutes).")

# --- DYNAMIC SONG LOADING ---
SONG_DIR = "songs"
# The final test is a single combined track of all 18 songs in sequence. It lives
# outside SONG_DIR so it is not listed as just another individual song.
FINAL_TEST_PATH = os.path.join("final", "all_songs.wav")
FINAL_TEST_LABEL = "🏆 FINAL TEST — Sing All 18 Songs in Sequence"
FINAL_TEST_KEY = "FINAL_TEST_ALL_SONGS"  # how the final test row is stored in the sheet

if os.path.exists(SONG_DIR):
    available_songs = {f.replace('.wav', ''): os.path.join(SONG_DIR, f)
                       for f in sorted(os.listdir(SONG_DIR)) if f.endswith('.wav')}
else:
    available_songs = {}
    st.error("Songs directory not found! Please check your GitHub folder structure.")

# Load this user's already-qualified songs once per session (one sheet read). It is
# then kept fresh in-session as they pass more songs, so we never re-read on reruns.
if "completed_songs" not in st.session_state:
    try:
        completed, final_done, final_score = _get_user_progress(user_id)
    except Exception:
        completed, final_done, final_score = {}, False, None
    st.session_state.completed_songs = completed
    st.session_state.final_done = final_done
    st.session_state.final_score = final_score

completed_songs = st.session_state.completed_songs  # dict: song_key -> score (or None)

# Progress summary + lists of what's done and what's left.
total_songs = len(available_songs)
done_here = [n for n in available_songs if n in completed_songs]
remaining = [n for n in available_songs if n not in completed_songs]
st.progress(len(done_here) / total_songs if total_songs else 0.0,
            text=f"Completed {len(done_here)} of {total_songs} songs")
if done_here:
    with st.expander(f"✅ Completed songs ({len(done_here)})"):
        for n in done_here:
            sc = completed_songs.get(n)
            if sc is not None:
                st.write(f"- {n} — **{sc:.1f}%**")
            else:
                st.write(f"- {n}")
if remaining:
    with st.expander(f"🎯 Remaining songs ({len(remaining)})"):
        st.write("\n".join(f"- {n}" for n in remaining))

# 2. Song Selection (the final test, if present, is always offered last). Completed
# songs are marked with ✅ but stay selectable in case they want to improve a score.
selected_song_path = None
is_final_test = False
song_key = None       # value stored in the sheet's "Song" column
display_name = None   # friendly name used in user-facing messages

label_to_name = {}
options = []
for name in available_songs:
    label = _qualified_label(name, completed_songs.get(name)) if name in completed_songs else name
    label_to_name[label] = name
    options.append(label)
# Final Test is only offered when its combined track exists AND it is enabled in
# secrets ([features] final_test = true). Off by default; enable on demand.
if os.path.exists(FINAL_TEST_PATH) and _final_test_enabled():
    if st.session_state.get("final_done"):
        final_label = _qualified_label(FINAL_TEST_LABEL, st.session_state.get("final_score"))
    else:
        final_label = FINAL_TEST_LABEL
    label_to_name[final_label] = FINAL_TEST_LABEL
    options.append(final_label)

if options:
    # Default to the first not-yet-completed song so they land on something to do.
    default_index = next((i for i, lbl in enumerate(options)
                          if not lbl.startswith("✅")), 0)
    selected_label = st.selectbox("Choose a song to practice:", options, index=default_index)
    selected_song_name = label_to_name[selected_label]
    if selected_song_name == FINAL_TEST_LABEL:
        is_final_test = True
        selected_song_path = FINAL_TEST_PATH
        song_key = FINAL_TEST_KEY
        display_name = "the Final Test (all 18 songs)"
        st.warning(
            "🏆 **Final Test:** sing all 18 songs one after another, in the same order "
            "as the reference, keeping pace with it. This is the full ~6 minute sequence — "
            "play the reference first to follow along."
        )
    else:
        selected_song_path = available_songs[selected_song_name]
        song_key = selected_song_name
        display_name = selected_song_name
    st.warning(f"📌 **Selected padyam:** {display_name} — sing **this** poem in your recording.")
    if song_key in completed_songs:
        sc = completed_songs[song_key]
        if sc is not None:
            st.info(f"You've already qualified for this song with **{sc:.1f}%**.")
        else:
            st.info("You've already qualified for this song.")
else:
    st.warning("No songs found in the /songs folder.")

if user_id and selected_song_path:
    st.audio(selected_song_path)

    # Show the target length so the singer knows roughly how long to sing (header read
    # only — instant, no full decode).
    try:
        _target_dur = librosa.get_duration(path=selected_song_path)
        st.caption(f"⏱️ Target length: about {int(round(_target_dur))} seconds — "
                   f"sing the whole song before pressing stop.")
    except Exception:
        pass

    # Reset the mic widget when the dropdown changes so an old recording cannot
    # be analyzed against a newly selected padyam.
    if st.session_state.get("_active_song") != song_key:
        st.session_state["_active_song"] = song_key
        st.session_state["_rec_nonce"] = st.session_state.get("_rec_nonce", 0) + 1
    rec_nonce = st.session_state["_rec_nonce"]

    st.markdown(
        """
        <style>
        div[data-testid="stAudioInput"] {
            border: 3px solid #e63946;
            border-radius: 14px;
            padding: 16px 14px 10px;
            background: linear-gradient(180deg, #fff5f5 0%, #ffffff 100%);
            box-shadow: 0 2px 10px rgba(230, 57, 70, 0.15);
        }
        div[data-testid="stAudioInput"] label p {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #b00020 !important;
        }
        /* Make the native record/stop button clearly red */
        div[data-testid="stAudioInput"] button {
            background-color: #e63946 !important;
            border: 2px solid #c1121f !important;
            color: #ffffff !important;
        }
        div[data-testid="stAudioInput"] button:hover {
            background-color: #c1121f !important;
            border-color: #9d0208 !important;
        }
        div[data-testid="stAudioInput"] button svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("### Step 2 — Record your singing")
        st.markdown(
            "In the **red box below**, tap the **microphone button** to start, "
            "sing the **whole padyam with clear words**, then tap **stop**. "
            "Nothing is analyzed until you press **Analyze**."
        )
        audio_value = st.audio_input("Tap the microphone button to start recording",
                                     key=f"recorder_{song_key}_{rec_nonce}")

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        do_analyze = audio_value is not None and st.button(
            "✅ Analyze my recording", key=f"analyze_{song_key}_{rec_nonce}")
    with rec_col2:
        if st.button("🔄 Discard & record again", key=f"reset_rec_{song_key}_{rec_nonce}"):
            st.session_state["_rec_nonce"] = rec_nonce + 1
            st.session_state.pop(f"_audio_hash_{song_key}", None)
            st.session_state.pop(f"_last_wrong_{song_key}", None)
            st.rerun()
    if audio_value is not None and not do_analyze:
        st.caption("Recorded. Press **Analyze** when ready — or **Discard & record again** "
                   "if you sang the wrong padyam or want a fresh take.")

    if do_analyze:
        _touch_active_session()
        if _server_congested():
            st.warning(
                f"**Too many people analyzing at once** (~{_active_session_count()} active). "
                f"Please wait **2–5 minutes** and try **Analyze** again. "
                f"Your recording is still here — no need to re-record."
            )
            st.stop()
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        hash_key = f"_audio_hash_{song_key}"
        if (st.session_state.get(hash_key) == audio_hash
                and st.session_state.get(f"_last_wrong_{song_key}")):
            st.warning(
                "This is the **same recording** as your last attempt. "
                "Tap **Discard & record again**, then sing the **selected padyam** before analyzing."
            )
            st.stop()
        st.session_state[hash_key] = audio_hash

        # Live, per-session attempt counter (no sheet writes, no API calls).
        attempt_counts = st.session_state.setdefault("attempt_counts", {})
        attempt_counts[song_key] = attempt_counts.get(song_key, 0) + 1
        attempt_no = attempt_counts[song_key]

        with st.spinner("Analyzing your pronunciation, timing, and tune..."):
            duration_ref, hop_len, mfcc_ref, chroma_ref, f0_ref = _load_reference(
                selected_song_path)

            try:
                y_user, sr_user = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as e:
                st.error("Audio decoding error. Please try recording again.")
                st.stop()

            # --- TIME CHECK (PACING) ---
            duration_user = librosa.get_duration(y=y_user, sr=sr_user)
            time_ratio = min(duration_ref, duration_user) / max(duration_ref, duration_user)

            # --- USER FEATURES (same hop as the cached reference) ---
            mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user, n_mfcc=13, hop_length=hop_len)
            # Layer B: Musical Notes (Chroma)
            chroma_user = librosa.feature.chroma_stft(y=y_user, sr=sr_user, hop_length=hop_len)
            # Layer C: Pitch Tracking (f0)
            f0_user, _, _ = librosa.pyin(y_user, fmin=librosa.note_to_hz('C2'),
                                         fmax=librosa.note_to_hz('C7'), sr=sr_user, hop_length=hop_len)
            f0_user = np.nan_to_num(f0_user)

            f0_user_norm = (f0_user - np.mean(f0_user)) / (np.std(f0_user) + 1e-6) if np.std(f0_user) > 0 else f0_user
            f0_user_norm = f0_user_norm.reshape(1, -1)
            mfcc_user_norm = (mfcc_user - np.mean(mfcc_user)) / (np.std(mfcc_user) + 1e-6)
            mfcc_ref_art = _articulation_mfcc(mfcc_ref)
            mfcc_user_art = _articulation_mfcc(mfcc_user_norm)
            articulation_ratio = _articulation_ratio(mfcc_ref, mfcc_user_norm)

            # --- 4. DTW ON ARTICULATION MFCC — pass/fail score driver ---
            try:
                D, wp = librosa.sequence.dtw(
                    X=mfcc_ref_art,
                    Y=mfcc_user_art,
                    metric='euclidean',
                    backtrack=True,
                    global_constraints=True,
                    band_rad=0.1
                )
            except Exception:
                try:
                    D, wp = librosa.sequence.dtw(
                        X=mfcc_ref_art,
                        Y=mfcc_user_art,
                        metric='euclidean',
                        backtrack=True
                    )
                except Exception:
                    st.error("Couldn't analyze this recording — it may be too short or "
                             "silent. Please record again.")
                    st.stop()

            final_accumulated_cost = D[-1, -1]
            path_length = len(wp)
            norm_dist = final_accumulated_cost / path_length if path_length > 0 else 100.0

            # --- Wrong-poem check: which padyam did they ACTUALLY sing? ---
            wrong_song = False
            identity_best = None
            identity_sel_d = identity_best_d = None
            if not is_final_test:
                wrong_song, identity_best, identity_sel_d, identity_best_d = _detect_wrong_song(
                    song_key, chroma_user, mfcc_user_norm, available_songs)
            identity_blocks_pass = (
                not is_final_test
                and identity_best != song_key
                and identity_best_d + _IDENTITY_TIE_EPS < identity_sel_d
            )

            # --- 5. FINAL SCORE: articulation (word clarity) + pacing to SELECTED ref ---
            # Note: this score alone does NOT prove the correct padyam — similar tunes can
            # score high against the wrong reference. wrong_song (above) enforces lyrics.
            if norm_dist <= GOOD_MATCH_DIST:
                base_score = 100.0 - ((norm_dist / GOOD_MATCH_DIST) * 5.0)
            else:
                base_score = max(0.0, 95.0 - ((norm_dist - GOOD_MATCH_DIST) * PENALTY_SLOPE))

            if time_ratio >= TEMPO_TOLERANCE:
                tempo_factor = 1.0
            else:
                tempo_factor = max(0.0, (time_ratio - 0.5) / (TEMPO_TOLERANCE - 0.5))
            tempo_blend = (1.0 - TEMPO_WEIGHT) + (TEMPO_WEIGHT * tempo_factor)
            final_score = base_score * tempo_blend

            # Safe NaN / Infinity boundary protection
            if np.isnan(final_score) or np.isinf(final_score):
                score = 0.0
            else:
                score = round(float(final_score), 2)

            if wrong_song:
                score = min(score, 35.0)
                st.session_state[f"_last_wrong_{song_key}"] = True
            else:
                st.session_state.pop(f"_last_wrong_{song_key}", None)

            if identity_blocks_pass and not wrong_song:
                score = min(score, 35.0)

            pron_d = _path_layer_distance(mfcc_ref_art, mfcc_user_art, wp, slice(0, 9))
            pron_pct = _layer_to_pct(pron_d, *_LAYER_ANCHORS["pronunciation"])
            clear_words = (pron_pct >= MIN_PRONUNCIATION_PCT
                           and articulation_ratio >= MIN_ARTICULATION_RATIO)

            feedback = _build_performance_feedback(
                mfcc_ref, mfcc_user_norm, chroma_ref, chroma_user, f0_ref, f0_user_norm, wp,
                duration_ref, duration_user, time_ratio, score, clear_words,
            )

            st.metric("Overall Match Score", f"{score}%")
            st.caption(f"🎙️ Attempt #{attempt_no} for this song (this session). "
                       f"Analyzed against: **{display_name}**")
            if wrong_song:
                st.error(
                    f"**Wrong padyam detected** — this recording does not match "
                    f"**{display_name}**. "
                    f"Please sing the **selected** padyam's lyrics and tap "
                    f"**Discard & record again**."
                )
            elif identity_blocks_pass:
                st.error(
                    f"**Wrong padyam detected** — this recording does not match "
                    f"**{display_name}**. "
                    f"Please sing the **selected** padyam's lyrics and tap "
                    f"**Discard & record again**."
                )

            st.subheader("How you did")
            st.caption(
                "Pass requires: **(1) correct padyam** (lyrics/content), "
                "**(2) clear pronunciation**, **(3) good pacing**. "
                "The score above measures pronunciation + pacing against the **selected** "
                "reference only — a different padyam can still score high if the tune is similar."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.progress(int(feedback["pronunciation_pct"]) / 100.0,
                            text=f"🗣️ Pronunciation: {int(feedback['pronunciation_pct'])}%")
                st.progress(int(feedback["notes_pct"]) / 100.0,
                            text=f"🎵 Musical notes: {int(feedback['notes_pct'])}%")
            with col2:
                st.progress(int(feedback["melody_pct"]) / 100.0,
                            text=f"🎼 Melody line: {int(feedback['melody_pct'])}%")
                tempo_icon = {"good": "✅", "ok": "⚠️", "slow": "🐢", "fast": "⚡"}.get(
                    feedback["tempo_status"], "⏱️")
                st.info(f"{tempo_icon} **Pacing:** {feedback['tempo_msg']}")

            st.markdown("**Tips to improve:**")
            for tip in feedback["tips"]:
                st.markdown(f"- {tip}")

            with st.expander("Technical details (for organizers)"):
                id_line = ""
                if identity_sel_d is not None:
                    id_line = (
                        f" | Identity: selected={round(identity_sel_d, 3)}"
                        f", best={round(identity_best_d, 3)} ({identity_best})"
                    )
                st.caption(
                    f"Raw Dist (articulation MFCC): {round(norm_dist, 2)} | "
                    f"Tempo Acc: {round(time_ratio * 100, 1)}% | "
                    f"Articulation ratio: {round(articulation_ratio, 2)} | "
                    f"Base score: {round(base_score, 1)} | Tempo blend: {round(tempo_blend, 2)} | "
                    f"Ref length: {duration_ref:.1f}s{id_line}"
                )

            # Rough timbre fingerprint of this recording (used only if they qualify)
            voice_sig = _voice_signature(mfcc_user)

        # --- GOOGLE SHEETS UPSERT LOGIC ---
        qualified = (score >= 85 and clear_words and not wrong_song
                     and not identity_blocks_pass)

        if wrong_song or identity_blocks_pass:
            pass  # error already shown above; score capped — cannot qualify
        elif score >= 85 and not clear_words:
            st.warning(
                f"Score: {score}% — but **clear pronunciation is required** to qualify "
                f"(need {MIN_PRONUNCIATION_PCT}%+ pronunciation). "
                f"**Sing every word** — humming or mumbling the tune alone won't pass."
            )
        elif qualified:
            st.balloons()
            st.success(f"🎉 PASS! You qualified for {display_name} with {score}%!")

            # If they've already passed this (per the progress loaded at login), do NOT
            # touch the sheet at all — no read, no write. Scores aren't improved once
            # passed, so repeat passes cost zero API calls. This is the main thing that
            # keeps us well under the rate limits during busy periods.
            if is_final_test:
                already_passed = bool(st.session_state.get("final_done"))
            else:
                already_passed = song_key in st.session_state.get("completed_songs", {})

            if already_passed:
                st.info("You've already qualified for this earlier — keeping your existing "
                        "result. (Scores aren't updated once you've passed.)")
            else:
                # Saving must never crash the app: if the Sheet is unreachable, the user
                # still sees their passing score.
                try:
                    saved, prev_score, songs_passed = _save_qualification(
                        user_id=user_id,
                        reg_id=st.session_state.registration_id,
                        song_key=song_key,
                        score=score,
                        voice_sig=voice_sig,
                        is_final_test=is_final_test,
                        final_key=FINAL_TEST_KEY,
                    )

                    # Keep in-session progress fresh so ✅ marks update without a re-read.
                    if is_final_test:
                        st.session_state.final_done = True
                        st.session_state.final_score = score
                    else:
                        updated = dict(st.session_state.get("completed_songs", {}))
                        updated[song_key] = score
                        st.session_state.completed_songs = updated

                    if is_final_test:
                        st.snow()
                        st.success("🏆 CONGRATULATIONS! You passed the FINAL TEST — "
                                   "all 18 songs sung in sequence!")
                    else:
                        st.success("Result saved!")
                        st.info(f"Progress: You have qualified for {songs_passed} out of 18 songs!")
                        if songs_passed == 18:
                            st.snow()
                            st.success("🏆 AMAZING! You have qualified for ALL 18 songs!")
                except Exception as e:
                    st.warning("Your score counts, but we couldn't reach the leaderboard right now. "
                               "(Check that the Google Sheet is shared with the service account email.)")
        else:
            st.error(f"Score: {score}%. You need 85% to qualify for {display_name}. Try again!")
