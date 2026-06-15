# 🎤 Sataka Sankharavam Tune Matcher Challenge
**An AI-powered vocal accuracy tester for pronunciation and pacing.**

Participants log in, sing along to trimmed reference tracks, and get a match score driven
mainly by **pronunciation** and **pacing**. Passing scores are recorded to a Google Sheets
leaderboard; after each attempt the app shows coaching feedback (notes/melody bars are
guidance only). An admin panel shows live statistics.

> For full implementation details (sheet schema, login rules, scoring, trimming, voice audit,
> admin panel, final test, scalability notes), see [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

---

## 🚀 Features
* **Pronunciation + pacing scoring:** Pass/fail uses MFCC (pronunciation) aligned with
  band-constrained DTW, blended with pacing (recording length vs. reference). Musical notes
  and melody are shown as coaching bars only — they do not affect the score.
* **Coaching feedback:** After each attempt, shows pronunciation / notes / melody breakdown,
  a plain-language pacing message (too fast / too slow / good), and tips to improve.
* **Trimmed references:** Reference songs are trimmed to the vocal span (first word through
  closing "Sumathi"/"Vinura Vema"); trailing instrumental is removed so scores aren't
  penalized for silence/music the singer shouldn't match.
* **85% Threshold Challenge:** Only scores of 85% or higher "Pass" for a song.
* **Participant Login:** Gated by Name + Registration ID against a Google Sheets roster
  (case- and whitespace-insensitive matching).
* **Progress Tracking:** Shows each user their completed vs. remaining songs on login.
* **Leaderboard & Upserts:** Targeted Google Sheets writes (one row per user+song); a passed
  song is saved once and not overwritten, keeping API usage low for large rosters.
* **Admin Panel:** Password-gated stats — totals, per-user leaderboard, per-song completion,
  and a family-scoped voice audit.
* **Voice Audit (family-scoped):** Best-effort timbre fingerprint assigns per-family
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
├── trim_songs.py         # Trim instrumental intro/outro from reference .wav files
├── requirements.txt      # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud (ffmpeg)
├── PROJECT_SUMMARY.md    # Full documentation / future instructions
├── songs/                # 18 trimmed reference .wav files
├── final/                # Optional combined all-songs track for the Final Test
└── .streamlit/
    └── secrets.toml      # Google Sheets credentials, admin password, feature flags (local only)
```
