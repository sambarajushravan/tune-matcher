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
            # 1. Load Audio files
            y_ref, sr_ref = librosa.load(selected_song_path, sr=22050)

            try:
                y_user, sr_user = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as e:
                st.error("Audio decoding error. Please try recording again.")
                st.stop()

            # --- 2. TIME CHECK (STRICT TEMPO) ---
            duration_ref = librosa.get_duration(y=y_ref, sr=sr_ref)
            duration_user = librosa.get_duration(y=y_user, sr=sr_user)

            # If they finish more than 15% too fast or too slow, penalize them
            time_ratio = min(duration_ref, duration_user) / max(duration_ref, duration_user)

            # --- 3. PRONUNCIATION & TUNE MATCHING (MFCC + CHROMA) ---
            # MFCCs track vocal tract shape (pronunciation/lyrics)
            mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13)
            mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_user, n_mfcc=13)

            # Chroma tracks the musical notes, ignoring male/female octave differences
            chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr_ref)
            chroma_user = librosa.feature.chroma_stft(y=y_user, sr=sr_user)

            # THE FIX: Scale MFCCs down so they don't overpower the Chroma features
            # This locks both feature spaces into comparable bounds
            mfcc_ref_norm = (mfcc_ref - np.mean(mfcc_ref)) / (np.std(mfcc_ref) + 1e-6)
            mfcc_user_norm = (mfcc_user - np.mean(mfcc_user)) / (np.std(mfcc_user) + 1e-6)

            # Combine features into a comprehensive "voice print"
            features_ref = np.vstack([mfcc_ref_norm, chroma_ref])
            features_user = np.vstack([mfcc_user_norm, chroma_user])

            # --- 4. DYNAMIC TIME WARPING (TIMING PATTERN MATCH) ---
            # We specify metric='cosine' to evaluate the similarity of the shape,
            # which prevents clipping regardless of the volume or length!
            D, wp = librosa.sequence.dtw(X=features_ref, Y=features_user, metric='cosine', backtrack=True)

            # Grab the final accumulated cost at the top-right corner and normalize by path length
            final_accumulated_cost = D[-1, -1]
            path_length = len(wp)
            norm_dist = final_accumulated_cost / path_length if path_length > 0 else 1.0

            # --- 5. COMPUTE FINAL HYBRID SCORE ---
            # Cosine DTW distance natively ranges roughly from 0.0 to 1.0.
            # 0.0 distance means a flawless copy. 1.0 means complete mismatch.
            # We subtract it from 1 to flip it to an accuracy percentage (0.0 to 100.0)
            base_score = max(0.0, 100.0 * (1.0 - norm_dist))

            # Multiply by our time ratio to strictly punish pacing errors
            final_score = base_score * time_ratio
            score = round(float(final_score), 2)

            # Safety: cosine distance is undefined for silent/all-zero frames,
            # which can produce NaN. Treat that as a 0% match.
            if np.isnan(score):
                score = 0.0

            st.metric("Overall Match Score", f"{score}%")
            st.caption(f"Tempo Accuracy: {round(time_ratio * 100, 1)}% | Timing Target: {round(duration_ref, 1)}s")

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
