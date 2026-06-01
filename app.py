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
        with st.spinner("Analyzing your tune and timing..."):
            # 1. Load Audio
            # Load Reference Audio (.wav from your folder)
            y_ref, sr = librosa.load(selected_song_path, sr=22050)

            # --- SAFE IPHONE AUDIO LOADING ---
            try:
                # This safely converts the iPhone's microphone bytes on the fly
                y_user, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as e:
                st.error("Audio format error. Please try recording again or use Google Chrome.")
                st.stop()

            # 2. Extract Pitch (f0)
            # We define a human vocal range: C2 (~65Hz) to C7 (~2093Hz)
            f0_ref, voiced_flag_ref, voiced_probs_ref = librosa.pyin(
                y_ref, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_user, voiced_flag_user, voiced_probs_user = librosa.pyin(
                y_user, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )

            # 3. Clean Data (Handle silence/NaN)
            f0_ref = np.nan_to_num(f0_ref)
            f0_user = np.nan_to_num(f0_user)

            # 4. Dynamic Time Warping (DTW)
            # This aligns the user's recording to the reference timing
            # D = Distance matrix, wp = Warping path (the "alignment" map)
            D, wp = librosa.sequence.dtw(X=f0_ref, Y=f0_user, backtrack=True)

            # 5. Calculate Similarity
            # We look at the aligned points and find the average frequency difference
            diffs = []
            for ref_idx, user_idx in wp:
                if f0_ref[ref_idx] > 0 and f0_user[user_idx] > 0:
                    # Difference in Hz
                    diff = abs(f0_ref[ref_idx] - f0_user[user_idx])
                    diffs.append(diff)

            # 6. Generate the 0-100% Score
            if not diffs:
                score = 0.0
            else:
                avg_hz_diff = np.mean(diffs)
                # Formula: If avg diff is 0Hz = 100%.
                # We penalize roughly 1% per 2Hz of average error.
                score = max(0, 100 - (avg_hz_diff / 2.0))
                score = round(float(score), 2)

            # Now show the real result!
            st.metric("Tune Accuracy", f"{score}%")

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
