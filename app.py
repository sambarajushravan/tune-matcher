import streamlit as st
import librosa
import numpy as np
from audio_recorder_streamlit import audio_recorder
from streamlit_gsheets import GSheetsConnection
import io
import os
import re
import datetime
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Sataka Sankharavam Tune Matcher", page_icon="🎤")
st.title("🎤 Sataka Sankharavam Tune Matcher Challenge")
st.write("Match the tune and timing at 85% or higher to pass!")


# --- NORMALIZATION HELPERS (for forgiving login matching) ---
def _norm_name(value):
    """Lowercase + collapse any run of whitespace to a single space + strip.
    So 'Sahitya   Malladi' == 'sahitya malladi'."""
    return " ".join(str(value).split()).lower()


def _norm_pwd(value):
    """Lowercase + remove all whitespace, so case and stray spaces don't matter."""
    return "".join(str(value).split()).lower()


def authenticate(username, password):
    """Validate name + Registration ID against the roster in the sheet.
    Returns (canonical_user_id, registration_id) on success, else None."""
    conn = st.connection("gsheets", type=GSheetsConnection)
    roster = conn.read(ttl=0).dropna(how="all")
    if "User ID" not in roster.columns or "Registration ID" not in roster.columns:
        return None

    target_name = _norm_name(username)
    target_pwd = _norm_pwd(password)

    for _, row in roster.iterrows():
        reg = row.get("Registration ID")
        if reg is None or str(reg).strip() == "":
            continue  # skip rows without a Registration ID
        if _norm_name(row["User ID"]) == target_name and _norm_pwd(reg) == target_pwd:
            return str(row["User ID"]).strip(), str(reg).strip()
    return None


# --- VOICE FINGERPRINT HELPERS (best-effort, for admin duplicate-singer review) ---
# NOTE: this is a lightweight timbre signature (MFCC mean+std), NOT biometric-grade
# speaker verification. It is content-dependent, so treat it as an admin review flag.
VOICE_MATCH_THRESHOLD = 0.98  # cosine similarity above which two prints are "same voice"


def _voice_signature(mfcc):
    """Compact, L2-normalized timbre signature from MFCC mean + std over time."""
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


def _assign_voice_id(df, new_sig):
    """Nearest-neighbour match against stored Voice Prints.
    Returns (voice_id, matched_user_id_or_None). Reuses an existing Voice ID if a very
    similar print exists, otherwise mints the next 'Voice N' label."""
    best_sim, best_id, best_user = -1.0, None, None
    max_n = 0
    if "Voice ID" in df.columns and "Voice Print" in df.columns:
        for _, r in df.iterrows():
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
                conn = st.connection("gsheets", type=GSheetsConnection)
                admin_df = conn.read(ttl=0).dropna(how="all")

                # Only completed passes count toward stats (skips blank roster rows).
                if "Status" in admin_df.columns:
                    passes_df = admin_df[admin_df["Status"] == "QUALIFIED"]
                else:
                    passes_df = admin_df.iloc[0:0]

                if not passes_df.empty:
                    total_passes = len(passes_df)
                    unique_singers = passes_df["User ID"].nunique()

                    col1, col2 = st.columns(2)
                    col1.metric("Total Qualifications Logged", total_passes)
                    col2.metric("Total Unique Active Users", unique_singers)

                    st.subheader("🏆 Top Participant Progress")
                    leaderboard = passes_df["User ID"].value_counts().reset_index()
                    leaderboard.columns = ["User ID", "Songs Completed (Out of 18)"]
                    st.dataframe(leaderboard, use_container_width=True)

                    st.subheader("🎵 Completion Rates by Song")
                    song_counts = passes_df["Song"].value_counts()
                    st.bar_chart(song_counts)

                    # Voice Audit: flag a single voice qualifying under multiple usernames.
                    st.subheader("🕵️ Voice Audit (possible same singer across names)")
                    if "Voice ID" in passes_df.columns:
                        va = passes_df[passes_df["Voice ID"].astype(str).str.strip() != ""]
                        if not va.empty:
                            grouped = va.groupby("Voice ID")["User ID"].nunique()
                            flagged = grouped[grouped > 1]
                            if not flagged.empty:
                                for vid, n in flagged.items():
                                    users = sorted(va[va["Voice ID"] == vid]["User ID"].unique())
                                    st.error(f"{vid}: same voice qualified under {n} names → {', '.join(users)}")
                            else:
                                st.success("No duplicate voices detected across different users.")
                        else:
                            st.info("No voice fingerprints recorded yet.")
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

# Stop here until the participant logs in (admin panel above still renders).
if not st.session_state.authenticated:
    st.stop()

# --- AUTHENTICATED PARTICIPANT FLOW ---
user_id = st.session_state.user_id

header_col, logout_col = st.columns([3, 1])
header_col.success(f"Logged in as: {user_id}")
if logout_col.button("Log out"):
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.registration_id = None
    st.rerun()

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

# 2. Song Selection (the final test, if present, is always offered last)
selected_song_path = None
is_final_test = False
song_key = None       # value stored in the sheet's "Song" column
display_name = None   # friendly name used in user-facing messages

options = list(available_songs.keys())
if os.path.exists(FINAL_TEST_PATH):
    options.append(FINAL_TEST_LABEL)

if options:
    selected_song_name = st.selectbox("Choose a song to practice:", options)
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
else:
    st.warning("No songs found in the /songs folder.")

if user_id and selected_song_path:
    st.audio(selected_song_path)

    st.write("Click the mic and start singing!")
    audio_bytes = audio_recorder(text="Click to record", pause_threshold=2.0)

    if audio_bytes:
        with st.spinner("Analyzing your pronunciation, timing, and tune..."):
            # 1. Load Audio files securely
            y_ref, sr_ref = librosa.load(selected_song_path, sr=22050)

            try:
                y_user, sr_user = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as e:
                st.error("Audio decoding error. Please try recording again.")
                st.stop()

            # --- 2. TIME CHECK (STRICT TEMPO) ---
            duration_ref = librosa.get_duration(y=y_ref, sr=sr_ref)
            duration_user = librosa.get_duration(y=y_user, sr=sr_user)
            time_ratio = min(duration_ref, duration_user) / max(duration_ref, duration_user)

            # --- 3. PRONUNCIATION, TUNE, & MELODY MATCHING ---
            # A short song (~20s) uses a fine hop. The final test is ~6 minutes, and at
            # hop=512 the DTW cost matrix (frames x frames) would be ~2 GB and crash the
            # app. Use a coarser hop above ~90s to keep the matrix bounded.
            hop_len = 512 if max(duration_ref, duration_user) <= 90 else 2048

            # Layer A: Pronunciation (MFCC)
            mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13, hop_length=hop_len)
            mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user, n_mfcc=13, hop_length=hop_len)

            # Layer B: Musical Notes (Chroma)
            chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref, hop_length=hop_len)
            chroma_user = librosa.feature.chroma_stft(y=y_user, sr=sr_user, hop_length=hop_len)

            # Layer C: Pitch Tracking (f0) to stop completely different songs from matching
            f0_ref, _, _ = librosa.pyin(y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr_ref, hop_length=hop_len)
            f0_user, _, _ = librosa.pyin(y_user, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr_user, hop_length=hop_len)

            # Clean up pitch tracking NaNs (silence/unvoiced frames) safely
            f0_ref = np.nan_to_num(f0_ref)
            f0_user = np.nan_to_num(f0_user)

            # Normalize Pitch so Male vs Female doesn't fail (Tracks relative melody path)
            f0_ref_norm = (f0_ref - np.mean(f0_ref)) / (np.std(f0_ref) + 1e-6) if np.std(f0_ref) > 0 else f0_ref
            f0_user_norm = (f0_user - np.mean(f0_user)) / (np.std(f0_user) + 1e-6) if np.std(f0_user) > 0 else f0_user

            f0_ref_norm = f0_ref_norm.reshape(1, -1)
            f0_user_norm = f0_user_norm.reshape(1, -1)

            # Normalize MFCCs
            mfcc_ref_norm = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-6)
            mfcc_user_norm = (mfcc_user - np.mean(mfcc_user)) / (np.std(mfcc_user) + 1e-6)

            # Combine all 3 dimensions into a dense feature grid
            features_ref = np.vstack([mfcc_ref_norm, chroma_ref, f0_ref_norm])
            features_user = np.vstack([mfcc_user_norm, chroma_user, f0_user_norm])

            # --- 4. DYNAMIC TIME WARPING WITH TIMELINE CONSTRAINTS ---
            # band_rad is a FRACTION of the longer sequence (radius = int(band_rad * max(N, M))).
            # 0.1 == warp window of ~10% of the song length, which cuts off "cheat pathways"
            # that let a wrong song meander to a low cost. (band_rad=10 was a no-op: it made the
            # band 10x wider than the matrix, i.e. no constraint at all.)
            D, wp = librosa.sequence.dtw(
                X=features_ref,
                Y=features_user,
                metric='euclidean',
                backtrack=True,
                global_constraints=True,
                band_rad=0.1
            )

            final_accumulated_cost = D[-1, -1]
            path_length = len(wp)
            norm_dist = final_accumulated_cost / path_length if path_length > 0 else 100.0

            # --- 5. COMPUTE FINAL HYBRID SCORE (ANCHOR SCALING) ---
            # GOOD_MATCH_DIST = distance a correct rendition lands near (~1.90).
            # PENALTY_SLOPE   = how fast the score collapses once past the anchor.
            # Gentle slope: keeps correct attempts (whose raw dist varies a bit) in the 90s
            # while still dropping wrong songs (~2.15) below 50. Worst case for a wrong song
            # is 0 once its distance is large enough.
            GOOD_MATCH_DIST = 1.90
            PENALTY_SLOPE = 150.0

            if norm_dist <= GOOD_MATCH_DIST:
                base_score = 100.0 - ((norm_dist / GOOD_MATCH_DIST) * 5.0)
            else:
                base_score = max(0.0, 95.0 - ((norm_dist - GOOD_MATCH_DIST) * PENALTY_SLOPE))

            # Apply tempo accuracy multiplier
            final_score = base_score * time_ratio

            # Safe NaN / Infinity boundary protection
            if np.isnan(final_score) or np.isinf(final_score):
                score = 0.0
            else:
                score = round(float(final_score), 2)

            st.metric("Overall Match Score", f"{score}%")
            st.caption(f"Raw Dist: {round(norm_dist, 2)} | Tempo Acc: {round(time_ratio * 100, 1)}%")

            # Rough timbre fingerprint of this recording (used only if they qualify)
            voice_sig = _voice_signature(mfcc_user)

        # --- GOOGLE SHEETS UPSERT LOGIC ---
        if score >= 85:
            st.balloons()
            st.success(f"🎉 PASS! You qualified for {display_name} with {score}%!")

            # Saving to the leaderboard must never crash the app: if the Sheet is
            # unreachable/misconfigured, the user still sees their passing score.
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                expected_columns = ["User ID", "Registration ID", "Song", "Score",
                                    "Status", "Last Attempt", "Voice ID", "Voice Print"]

                df = conn.read(ttl=0)
                df = df.dropna(how="all")

                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = pd.Series(dtype="object")

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

                # Voice label for this recording (rough timbre match for admin auditing)
                voice_id, _matched_user = _assign_voice_id(df, voice_sig)
                voice_str = _voice_to_str(voice_sig)

                # One row per (User ID, Song). Update only the matching row.
                mask = (df["User ID"] == user_id) & (df["Song"] == song_key)

                saved = False
                if mask.any():
                    # Existing attempt for this song: keep the best score only.
                    prev_raw = df.loc[mask, "Score"].iloc[0]
                    try:
                        prev_score = float(prev_raw)
                    except (TypeError, ValueError):
                        prev_score = None
                    prev_status = str(df.loc[mask, "Status"].iloc[0]).strip()

                    if prev_status == "QUALIFIED" and prev_score is not None and score <= prev_score:
                        st.info(f"Your previous best for this song ({prev_score}%) is kept — "
                                f"this attempt ({score}%) wasn't higher.")
                    else:
                        df.loc[mask, ["Score", "Status", "Last Attempt", "Voice ID", "Voice Print"]] = \
                            [score, "QUALIFIED", timestamp, voice_id, voice_str]
                        saved = True
                else:
                    # No row yet for this exact song. Reuse the participant's blank roster
                    # placeholder row (name + Registration ID, but no Song filled in) if one
                    # exists, so we update it in place instead of leaving a duplicate empty
                    # row behind. Otherwise append a fresh row for this song.
                    song_blank = df["Song"].isna() | (
                        df["Song"].astype(str).str.strip().isin(["", "nan", "None"])
                    )
                    placeholder_mask = (df["User ID"] == user_id) & song_blank

                    if placeholder_mask.any():
                        idx = df[placeholder_mask].index[0]
                        df.loc[idx, ["Registration ID", "Song", "Score", "Status",
                                     "Last Attempt", "Voice ID", "Voice Print"]] = [
                            st.session_state.registration_id, song_key, score,
                            "QUALIFIED", timestamp, voice_id, voice_str,
                        ]
                    else:
                        new_data = {
                            "User ID": user_id,
                            "Registration ID": st.session_state.registration_id,
                            "Song": song_key,
                            "Score": score,
                            "Status": "QUALIFIED",
                            "Last Attempt": timestamp,
                            "Voice ID": voice_id,
                            "Voice Print": voice_str,
                        }
                        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                    saved = True

                if saved:
                    conn.update(data=df)
                    st.success("New best score saved!")

                user_progress = df[df["User ID"] == user_id]
                qualified = user_progress[user_progress["Status"] == "QUALIFIED"]
                # The final test is not one of the 18 songs, so exclude it from the tally.
                songs_passed = len(qualified[qualified["Song"] != FINAL_TEST_KEY])

                if is_final_test:
                    st.snow()
                    st.success("🏆 CONGRATULATIONS! You passed the FINAL TEST — "
                               "all 18 songs sung in sequence!")
                else:
                    st.info(f"Progress: You have qualified for {songs_passed} out of 18 songs!")
                    if songs_passed == 18:
                        st.snow()
                        st.success("🏆 AMAZING! You have qualified for ALL 18 songs!")
            except Exception as e:
                st.warning("Your score counts, but we couldn't reach the leaderboard right now. "
                           "(Check that the Google Sheet is shared with the service account email.)")
        else:
            st.error(f"Score: {score}%. You need 85% to qualify for {display_name}. Try again!")
