# 🎤 Tune Matcher AI
**An AI-powered vocal accuracy tester using Dynamic Time Warping (DTW) and Pitch Detection.**

This application allows users to test their singing accuracy against 18 reference tracks. Using the **pYIN algorithm** and **Dynamic Time Warping**, the app analyzes not just the notes (pitch) but also the timing (tempo) to provide a precision similarity score.

---

## 🚀 Features
* **Precision Scoring:** Compares user audio to reference tracks using `librosa` and `pYIN`.
* **Time-Aware Matching:** Uses Dynamic Time Warping (DTW) so users aren't penalized for starting a split-second late.
* **90% Threshold Challenge:** Gamified interface where only scores of 90% or higher "Pass."
* **Automated Leaderboard:** Integrated with Google Sheets to track the most recent scores for up to 2,500 users.
* **PWA Ready:** Can be added to an iPhone/Android home screen for an "app-like" experience.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Audio Engine:** [Librosa](https://librosa.org/) (Digital Signal Processing)
* **Data Store:** Google Sheets (via `st-gsheets-connection`)
* **Hosting:** Streamlit Community Cloud

---

## 📂 Project Structure
```text
├── app.py                # Main application logic
├── requirements.txt      # Python dependencies
├── songs/                # 18 Reference .wav files (vocal-only recommended)
└── .streamlit/
    └── secrets.toml      # Google Sheets API credentials (local only)
