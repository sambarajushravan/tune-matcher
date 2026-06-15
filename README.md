# 🎤 Sataka Sankharavam Tune Matcher Challenge
**An AI-powered vocal accuracy tester using Dynamic Time Warping (DTW), MFCC, Chroma, and Pitch Detection.**

Participants log in, sing along to reference tracks, and get a similarity score that
considers pronunciation, musical notes, melody, and timing. Passing scores are recorded to a
Google Sheets leaderboard, and an admin panel shows live statistics.

> For full implementation details (sheet schema, login rules, scoring, voice audit, admin
> panel, final test, scalability notes), see [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

---

## 🚀 Features
* **Hybrid Scoring:** Compares user audio to reference tracks using `librosa` — MFCC
  (pronunciation), Chroma (notes), and pYIN pitch (melody), aligned with band-constrained DTW.
* **Time-Aware Matching:** A tempo ratio penalizes singing too fast/slow without punishing a
  split-second late start.
* **85% Threshold Challenge:** Only scores of 85% or higher "Pass" for a song.
* **Participant Login:** Gated by Name + Registration ID against a Google Sheets roster
  (case- and whitespace-insensitive matching).
* **Progress Tracking:** Shows each user their completed vs. remaining songs on login.
* **Leaderboard & Upserts:** Targeted Google Sheets writes (one row per user+song); a passed
  song is saved once and not overwritten, keeping API usage low for large rosters.
* **Admin Panel:** Password-gated stats — totals, per-user leaderboard, per-song completion,
  and a voice audit.
* **Voice Audit (family-scoped):** A best-effort timbre fingerprint assigns per-family
  `Voice N` labels and flags one voice qualifying under multiple names within the same
  Registration ID (a review hint, not biometric proof).
* **Optional Final Test:** A combined all-songs-in-sequence track, toggleable via a secrets
  feature flag (off by default).

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Frontend:** [Streamlit](https://streamlit.io/) (native `st.audio_input` recorder)
* **Audio Engine:** [Librosa](https://librosa.org/) (MFCC, Chroma, pYIN, DTW)
* **Data Store:** Google Sheets (via `gspread` + `google-auth`)
* **Hosting:** Streamlit Community Cloud

---

## 📂 Project Structure
```text
├── app.py                # Main application logic
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud (ffmpeg)
├── PROJECT_SUMMARY.md     # Full documentation / future instructions
├── songs/                # 18 reference .wav files (vocal-only recommended)
├── final/                # Optional combined all-songs track for the Final Test
└── .streamlit/
    └── secrets.toml      # Google Sheets credentials, admin password, feature flags (local only)
```
