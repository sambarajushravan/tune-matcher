import streamlit as st
import librosa
import numpy as np
from audio_recorder_streamlit import audio_recorder
import io
import os

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
        with st.spinner("Calculating your score..."):
            # Load Reference and User Audio
            y_ref, sr = librosa.load(selected_song_path, sr=22050)
            
            # Convert bytes to librosa format
            buffer = io.BytesIO(audio_bytes)
            y_user, _ = librosa.load(buffer, sr=22050)

            # --- YOUR PITCH LOGIC ---
            # (Insert your f0_ref, f0_user, and DTW logic here)
            # Let's assume the result is 'score'
            score = 92.5 # Placeholder for your math result
            
            st.metric("Your Match Score", f"{score}%")

            if score >= 90:
                st.balloons()
                st.success(f"PASS! You matched {score}%.")
                # Add logic here to save to Google Sheets
            else:
                st.error(f"Try again! You need 90%, you got {score}%.")
