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
st.write("Match the tune and timing at 85% or higher to pass!")

# 1. User Info
user_id = st.text_input("Enter your Name or ID:", "")

# --- DYNAMIC SONG LOADING ---
SONG_DIR = "songs"

if os.path.exists(SONG_DIR):
    available_songs = {f.replace('.wav', ''): os.path.join(SONG_DIR, f)
                       for f in sorted(os.listdir(SONG_DIR)) if f.endswith('.wav')}
else:
    available_songs = {}
    st.error("Songs directory not found! Please check your GitHub folder structure.")

# 2. Song Selection
selected_song_path = None
if available_songs:
    selected_song_name = st.selectbox("Choose a song to practice:", list(available_songs.keys()))
    selected_song_path = available_songs[selected_song_name]
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
            hop_len = 512
            
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

        # --- GOOGLE SHEETS UPSERT LOGIC ---
        if score >= 85:
            st.balloons()
            st.success(f"🎉 PASS! You qualified for {selected_song_name} with {score}%!")

            # Saving to the leaderboard must never crash the app: if the Sheet is
            # unreachable/misconfigured, the user still sees their passing score.
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                expected_columns = ["User ID", "Song", "Score", "Status", "Last Attempt"]

                df = conn.read(ttl=0)
                df = df.dropna(how="all")

                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = pd.Series(dtype="object")

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                new_data = {
                    "User ID": user_id,
                    "Song": selected_song_name,
                    "Score": score,
                    "Status": "QUALIFIED",
                    "Last Attempt": timestamp,
                }

                mask = (df["User ID"] == user_id) & (df["Song"] == selected_song_name)

                if mask.any():
                    df.loc[mask, ["Score", "Last Attempt"]] = [score, timestamp]
                else:
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

                conn.update(data=df)

                user_progress = df[df["User ID"] == user_id]
                songs_passed = len(user_progress[user_progress["Status"] == "QUALIFIED"])
                st.info(f"Progress: You have qualified for {songs_passed} out of 18 songs!")

                if songs_passed == 18:
                    st.snow()
                    st.success("🏆 AMAZING! You have qualified for ALL 18 songs!")
            except Exception as e:
                st.warning("Your score counts, but we couldn't reach the leaderboard right now. "
                           "(Check that the Google Sheet is shared with the service account email.)")
        else:
            st.error(f"Score: {score}%. You need 85% to qualify for this song. Try again!")

# --- OPTIONAL: ADMIN REPORTING SECTION ---
st.write("---")
with st.expander("📊 Admin Reports & Statistics (Internal Use Only)"):
    admin_password = st.text_input("Enter Admin Password:", type="password")

    if admin_password:
        # THE SECURE FIX: Fetch the password safely from Streamlit's secrets manager.
        # Guard against a missing [admin] secret (e.g. not configured on Streamlit Cloud)
        # or a Sheets read failure so the admin panel never hard-crashes the app.
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
                # Connect and read fresh data from the sheet
                conn = st.connection("gsheets", type=GSheetsConnection)
                admin_df = conn.read(ttl=0).dropna(how="all")

                if not admin_df.empty:
                    # Metric Summary Cards
                    total_passes = len(admin_df)
                    unique_singers = admin_df["User ID"].nunique()

                    col1, col2 = st.columns(2)
                    col1.metric("Total Qualifications Logged", total_passes)
                    col2.metric("Total Unique Active Users", unique_singers)

                    # Leaderboard Chart
                    st.subheader("🏆 Top Participant Progress")
                    leaderboard = admin_df["User ID"].value_counts().reset_index()
                    leaderboard.columns = ["User ID", "Songs Completed (Out of 18)"]
                    st.dataframe(leaderboard, use_container_width=True)

                    # Song Difficulty Chart
                    st.subheader("🎵 Completion Rates by Song")
                    song_counts = admin_df["Song"].value_counts()
                    st.bar_chart(song_counts)

                    # Raw Data Viewer
                    st.subheader("📋 Raw Activity Log")
                    st.dataframe(admin_df.sort_values(by="Last Attempt", ascending=False), use_container_width=True)
                else:
                    st.info("The database sheet is currently empty. No stats to report yet!")
            except Exception as e:
                st.error("Could not load reports from Google Sheets right now. Please try again later.")
        else:
            st.error("Incorrect Password.")