"""Pure scoring / identity / pacing logic for tune-matcher (no Streamlit deps).

Imported by app.py and exercised by tests/test_harness.py for regression checks.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import librosa
import numpy as np

# --- Scoring constants ---
TEMPO_OK_SEC = 2.0
TEMPO_MAX_SEC = 3.0
TEMPO_WEIGHT = 0.40
FINAL_TEST_TEMPO_OK_RATIO = 0.008
FINAL_TEST_TEMPO_MAX_RATIO = 0.012
GOOD_MATCH_DIST = 1.30
PENALTY_SLOPE = 85.0
MIN_PRONUNCIATION_PCT = 88
MIN_ARTICULATION_RATIO = 0.68
PASS_THRESHOLD = 85.0
WRONG_PADYAM_SCORE_CAP = 35.0

ARTICULATION_ROWS = slice(4, 13)
WRONG_SONG_MARGIN = 0.05
WRONG_SONG_REL_RATIO = 0.94
IDENTITY_TIE_EPS = 0.025

MAX_CONCURRENT_SESSIONS = 40
INACTIVITY_LOGOUT_SEC = 600
ACTIVE_SESSION_WINDOW_SEC = 300

SONGS_CACHE_VERSION = 2

RefLoader = Callable[[str], Tuple[np.ndarray, np.ndarray]]


def ref_file_key(path: str) -> Tuple[float, int]:
    return os.path.getmtime(path), os.path.getsize(path)


def dtw_norm_dist(X, Y) -> float:
    try:
        D, wp = librosa.sequence.dtw(
            X, Y, metric="euclidean", backtrack=True,
            global_constraints=True, band_rad=0.1,
        )
    except Exception:
        D, wp = librosa.sequence.dtw(X, Y, metric="euclidean", backtrack=True)
    pl = len(wp)
    return float(D[-1, -1] / pl) if pl > 0 else 100.0


def song_identity_matrix(chroma, mfcc):
    return np.vstack([chroma * 2.0, mfcc])


def articulation_mfcc(mfcc):
    return mfcc[ARTICULATION_ROWS]


def extract_reference_features(ref_path: str):
    """Load + featurize a reference track. Returns (duration, hop, mfcc, chroma, f0)."""
    y_ref, sr_ref = librosa.load(ref_path, sr=22050)
    duration_ref = librosa.get_duration(y=y_ref, sr=sr_ref)
    hop_len = 512 if duration_ref <= 90 else 2048

    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13, hop_length=hop_len)
    chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref, hop_length=hop_len)
    f0_ref, _, _ = librosa.pyin(
        y_ref,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr_ref,
        hop_length=hop_len,
    )
    f0_ref = np.nan_to_num(f0_ref)
    if np.std(f0_ref) > 0:
        f0_ref = (f0_ref - np.mean(f0_ref)) / (np.std(f0_ref) + 1e-6)
    f0_ref = f0_ref.reshape(1, -1)
    mfcc_ref = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-6)
    return duration_ref, hop_len, mfcc_ref, chroma_ref, f0_ref


def load_user_identity_features(path: str):
    """Chroma + normalized MFCC for wrong-poem identity (tests + analysis)."""
    y, sr = librosa.load(path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    hop_len = 512 if duration <= 90 else 2048
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_len)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_len)
    mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    return chroma, mfcc_norm, hop_len


def detect_wrong_song(
    song_key: str,
    chroma_user,
    mfcc_user_norm,
    available_songs: Dict[str, str],
    ref_loader: RefLoader,
) -> Tuple[bool, str, float, float]:
    """Return (is_wrong, best_name, selected_dist, best_dist)."""
    ident_user = song_identity_matrix(chroma_user, mfcc_user_norm)
    dists = {}
    for name, path in available_songs.items():
        mfcc_r, chroma_r = ref_loader(path)
        dists[name] = dtw_norm_dist(song_identity_matrix(chroma_r, mfcc_r), ident_user)

    selected_dist = dists[song_key]
    best_name = min(dists, key=dists.get)
    best_dist = dists[best_name]
    wrong = False
    if best_name != song_key and abs(selected_dist - best_dist) > IDENTITY_TIE_EPS:
        if (best_dist < selected_dist * WRONG_SONG_REL_RATIO
                and best_dist + WRONG_SONG_MARGIN < selected_dist):
            wrong = True
    return wrong, best_name, selected_dist, best_dist


def identity_blocks_pass(
    song_key: str,
    identity_best: str,
    identity_sel_d: float,
    identity_best_d: float,
    *,
    is_final_test: bool = False,
) -> bool:
    return (
        not is_final_test
        and identity_best != song_key
        and identity_best_d + IDENTITY_TIE_EPS < identity_sel_d
    )


def tempo_limits(duration_ref: float, is_final_test: bool) -> Tuple[float, float]:
    if is_final_test:
        ok = max(TEMPO_OK_SEC, duration_ref * FINAL_TEST_TEMPO_OK_RATIO)
        mx = max(TEMPO_MAX_SEC, duration_ref * FINAL_TEST_TEMPO_MAX_RATIO)
        return ok, max(mx, ok + 0.5)
    return TEMPO_OK_SEC, TEMPO_MAX_SEC


def tempo_factor(duration_ref: float, duration_user: float, is_final_test: bool):
    ok_sec, max_sec = tempo_limits(duration_ref, is_final_test)
    delta = abs(duration_user - duration_ref)
    if delta <= ok_sec:
        return 1.0, delta, ok_sec, max_sec
    if delta >= max_sec:
        return 0.0, delta, ok_sec, max_sec
    t = (delta - ok_sec) / (max_sec - ok_sec)
    return max(0.0, 1.0 - t), delta, ok_sec, max_sec


def compute_base_score(norm_dist: float) -> float:
    if norm_dist <= GOOD_MATCH_DIST:
        return 100.0 - ((norm_dist / GOOD_MATCH_DIST) * 5.0)
    return max(0.0, 95.0 - ((norm_dist - GOOD_MATCH_DIST) * PENALTY_SLOPE))


def compute_final_score(base_score: float, tempo_factor_val: float) -> Tuple[float, float]:
    tempo_blend = (1.0 - TEMPO_WEIGHT) + (TEMPO_WEIGHT * tempo_factor_val)
    final_score = base_score * tempo_blend
    if np.isnan(final_score) or np.isinf(final_score):
        return 0.0, tempo_blend
    return round(float(final_score), 2), tempo_blend


def apply_score_caps(
    score: float,
    *,
    wrong_song: bool,
    identity_blocks: bool,
) -> float:
    if wrong_song or identity_blocks:
        return min(score, WRONG_PADYAM_SCORE_CAP)
    return score


def would_qualify(
    score: float,
    *,
    clear_words: bool,
    wrong_song: bool,
    identity_blocks: bool,
) -> bool:
    return (
        score >= PASS_THRESHOLD
        and clear_words
        and not wrong_song
        and not identity_blocks
    )


# --- Session registry (pure; app holds the mutable dict) ---

def touch_active_session(
    sessions: dict,
    sid: str,
    now: float,
    *,
    inactivity_sec: float = INACTIVITY_LOGOUT_SEC,
) -> None:
    sessions[sid] = now
    cutoff = now - inactivity_sec
    stale = [k for k, t in sessions.items() if t < cutoff]
    for k in stale:
        del sessions[k]


def active_session_count(
    sessions: dict,
    now: float,
    *,
    window_sec: float = ACTIVE_SESSION_WINDOW_SEC,
) -> int:
    cutoff = now - window_sec
    return sum(1 for t in sessions.values() if t >= cutoff)


def server_congested(
    sessions: dict,
    now: float,
    *,
    max_concurrent: int = MAX_CONCURRENT_SESSIONS,
    window_sec: float = ACTIVE_SESSION_WINDOW_SEC,
) -> bool:
    return active_session_count(sessions, now, window_sec=window_sec) >= max_concurrent


def list_songs(song_dir: str) -> Dict[str, str]:
    if not os.path.isdir(song_dir):
        return {}
    return {
        f.replace(".wav", ""): os.path.join(song_dir, f)
        for f in sorted(os.listdir(song_dir))
        if f.endswith(".wav")
    }


def make_ref_loader(cache: Optional[dict] = None) -> RefLoader:
    """Build a ref loader with optional in-memory cache (for tests)."""
    store = cache if cache is not None else {}

    def loader(path: str):
        if path not in store:
            _, _, mfcc, chroma, _ = extract_reference_features(path)
            store[path] = (mfcc, chroma)
        return store[path]

    return loader
