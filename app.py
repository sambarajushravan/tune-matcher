import streamlit as st
import librosa
import numpy as np
from audio_recorder_streamlit import audio_recorder
import io

# --- APP CONFIG ---
st.set_page_config(page_title="Tune Matcher", page_icon="🎤")
st.title("🎤 Tune Matcher Challenge")
st.write("Match the tune and timing at 90% or higher to pass!")

# 1. User Info
user_id = st.text_input("Enter your Name or ID:", "")

# 2. Song Selection
songs_list = {
    "Song 1: Happy Birthday": "songs/song1.wav",
    "Song 2: Imagine": "songs/song2.wav",
    # Add all 18 here...
}
selected_song = st.selectbox("Choose a song to practice:", list(songs_list.keys()))

if user_id:
    st.audio(songs_list[selected_song]) # Play the reference for them
    
    st.write("Click the mic and start singing!")
    audio_bytes = audio_recorder(text="Click to record", pause_threshold=2.0)

    if audio_bytes:
        with st.spinner("Calculating your score..."):
            # Load Reference and User Audio
            y_ref, sr = librosa.load(songs_list[selected_song], sr=22050)
            
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
