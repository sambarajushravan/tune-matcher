import streamlit as st
import librosa
import numpy as np
from audio_recorder_streamlit import audio_recorder
from streamlit_gsheets import GSheetsConnection
import io
import os
import datetime
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Tune Matcher", page_icon="🎤")
st.title("🎤 Tune Matcher Challenge")
st.write("Match the tune and timing at 90% or higher to pass!")

HOP_LEN = 512


def stack_features(mfcc, chroma, f0):
    """Stack the 3 feature layers, truncating to the shortest frame count for safety."""
    n = min(mfcc.shape[1], chroma.shape[1], f0.shape[1])
    return np.vstack([mfcc[:, :n], chroma[:, :n], f0[:, :n]])


@st.cache_data(show_spinner=False)
def get_reference_features(path):
    """Compute and cache the reference song's feature matrix (it never changes)."""
    y_ref, sr_ref = librosa.load(path, sr=22050)
    duration_ref = librosa.get_duration(y=y_ref, sr=sr_ref)

    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13, hop_length=HOP_LEN)
    chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref, hop_length=HOP_LEN)
    f0_ref, _, _ = librosa.pyin(
        y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr_ref, hop_length=HOP_LEN,
    )
    f0_ref = np.nan_to_num(f0_ref)

    mfcc_ref_norm = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-6)
    f0_ref_norm = (f0_ref - np.mean(f0_ref)) / (np.std(f0_ref) + 1e-6) if np.std(f0_ref) > 0 else f0_ref
    f0_ref_norm = f0_ref_norm.reshape(1, -1)

    features_ref = stack_features(mfcc_ref_norm, chroma_ref, f0_ref_norm)
    return features_ref, duration_ref


# 1. User Info
user_id = st.text_input("Enter your Name or ID:", "")

# --- DYNAMIC SONG LOADING ---
SONG_DIR = "songs"

# This looks into the folder and finds all .wav files
if os.path.exists(SONG_DIR):
    # Get filenames, remove the '.wav' extension for the display name
    available_songs = {f.replace('.wav', ''): os.path.join(SONG_DIR, f)
                       for f in sorted(os.listdir(SONG_DIR)) if f.endswith('.wav')}
else:
    available_songs = {}
    st.error("Songs directory not found! Please check your GitHub folder structure.")

# 2. Song Selection (Now using the dynamic list)
selected_song_path = None
if available_songs:
    selected_song_name = st.selectbox("Choose a song to practice:", list(available_songs.keys()))
    selected_song_path = available_songs[selected_song_name]
else:
    st.warning("No songs found in the /songs folder.")

if user_id and selected_song_path:
    st.audio(selected_song_path) # Play the reference for them
    
    st.write("Click the mic and start singing!")
    audio_bytes = audio_recorder(text="Click to record", pause_threshold=2.0)

    if audio_bytes:
        with st.spinner("Analyzing your pronunciation, timing, and tune..."):
            # 1. Load the user's recording (reference features are cached separately)
            try:
                y_user, sr_user = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as e:
                st.error("Audio decoding error. Please try recording again.")
                st.stop()

            # Reference is computed once and cached (it never changes between attempts)
            features_ref, duration_ref = get_reference_features(selected_song_path)

            # --- 2. TIME CHECK (STRICT TEMPO) ---
            duration_user = librosa.get_duration(y=y_user, sr=sr_user)

            # If they finish more than 15% too fast or too slow, penalize them
            time_ratio = min(duration_ref, duration_user) / max(duration_ref, duration_user)

            # --- 3. USER FEATURES: PRONUNCIATION, TUNE, & MELODY ---
            # Layer A: Pronunciation (MFCC)
            mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user, n_mfcc=13, hop_length=HOP_LEN)
            # Layer B: Musical Notes (Chroma)
            chroma_user = librosa.feature.chroma_stft(y=y_user, sr=sr_user, hop_length=HOP_LEN)
            # Layer C: Pitch Tracking (f0) to stop completely different songs from matching
            f0_user, _, _ = librosa.pyin(y_user, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr_user, hop_length=HOP_LEN)
            f0_user = np.nan_to_num(f0_user)

            # Normalize so male vs female doesn't fail (tracks the change/shape, not absolute pitch)
            mfcc_user_norm = (mfcc_user - np.mean(mfcc_user)) / (np.std(mfcc_user) + 1e-6)
            f0_user_norm = (f0_user - np.mean(f0_user)) / (np.std(f0_user) + 1e-6) if np.std(f0_user) > 0 else f0_user
            f0_user_norm = f0_user_norm.reshape(1, -1)

            # GLUE ALL 3 LAYERS TOGETHER: Words + Notes + Melody Line
            features_user = stack_features(mfcc_user_norm, chroma_user, f0_user_norm)

            # --- 4. DYNAMIC TIME WARPING WITH STRICT COUNTERMEASURES ---
            D, wp = librosa.sequence.dtw(X=features_ref, Y=features_user, metric='euclidean', backtrack=True)

            final_accumulated_cost = D[-1, -1]
            path_length = len(wp)
            norm_dist = final_accumulated_cost / path_length if path_length > 0 else 100.0

            # --- 5. COMPUTE FINAL HYBRID SCORE ---
            # Adjusted strictness now that the feature vector has a melody weight anchor
            STRICTNESS_FACTOR = 16.0

            base_score = max(0.0, 100.0 - (norm_dist * STRICTNESS_FACTOR))
            final_score = base_score * time_ratio

            if np.isnan(final_score) or np.isinf(final_score):
                score = 0.0
            else:
                score = round(float(final_score), 2)

            st.metric("Overall Match Score", f"{score}%")
            st.caption(f"Raw Dist: {round(norm_dist, 2)} | Tempo Acc: {round(time_ratio * 100, 1)}%")

        # --- GOOGLE SHEETS UPSERT LOGIC (Per Song) ---
        if score >= 90:
            st.balloons()
            st.success(f"🎉 PASS! You qualified for {selected_song_name} with {score}%!")

            # Connect to Google Sheets (configured via .streamlit/secrets.toml)
            conn = st.connection("gsheets", type=GSheetsConnection)
            expected_columns = ["User ID", "Song", "Score", "Status", "Last Attempt"]

            # 1. Fetch current data from the sheet (ttl=0 => always read fresh)
            df = conn.read(ttl=0)
            df = df.dropna(how="all")
            # Make sure expected columns exist (handles a brand-new/empty sheet)
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")

            # 2. Prepare the record
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            new_data = {
                "User ID": user_id,
                "Song": selected_song_name,
                "Score": score,
                "Status": "QUALIFIED",
                "Last Attempt": timestamp,
            }

            # 3. Check if this specific user has already attempted THIS specific song
            # We look for a row where both User ID and Song match
            mask = (df["User ID"] == user_id) & (df["Song"] == selected_song_name)

            if mask.any():
                # Update only the score and date for that specific song row
                df.loc[mask, ["Score", "Last Attempt"]] = [score, timestamp]
            else:
                # User hasn't passed this song before, add a new row
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            # 4. Push the entire updated table back to Google Sheets
            conn.update(data=df)

            # 5. Show progress: How many songs out of 18 are done?
            user_progress = df[df["User ID"] == user_id]
            songs_passed = len(user_progress[user_progress["Status"] == "QUALIFIED"])
            st.info(f"Progress: You have qualified for {songs_passed} out of 18 songs!")

            if songs_passed == 18:
                st.snow()
                st.success("🏆 AMAZING! You have qualified for ALL 18 songs!")
        else:
            st.error(f"Score: {score}%. You need 90% to qualify for this song. Try again!")
