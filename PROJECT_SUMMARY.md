# Sataka Sankharavam Tune Matcher Challenge

A Streamlit web app where registered participants sing along to reference songs,
get an automatic match score (pronunciation + tune + timing), and have their
qualified songs tracked on a Google Sheets leaderboard. Includes a password-gated
admin panel with stats and a lightweight voice-audit flag for spotting possible
cheating.

---

## 1. What the app does

1. **Login gate** – A participant must enter their **Name** and **Registration ID**
   (their "password"). Both are validated against a roster stored in Google Sheets.
2. **Progress on login** – After login it reads their qualified songs once and shows a
   progress bar plus "Completed" / "Remaining" lists. Completed songs are marked with ✅
   in the dropdown, and the selection defaults to the first unfinished song.
3. **Pick a song** – Songs are loaded dynamically from the `songs/` folder (one `.wav`
   per song). There are 18 songs. An optional combined "Final Test" can be enabled (§6b).
4. **Record & analyze** – Uses Streamlit's native `st.audio_input`: the singer presses
   start/stop **manually** (so it never cuts off on a pause), then presses **Analyze**.
   Nothing is processed until Analyze is pressed, so they can re-record if they stopped
   early. A live, per-session **attempt counter** ("Attempt #N") is shown.
5. **Score** – The app compares the recording to the reference and produces an **Overall
   Match Score (%)**.
6. **Pass threshold = 85%** – At 85%+ the attempt is `QUALIFIED` and saved. **Once a
   participant has passed a song, scores are not updated** (re-singing a passed song does
   no sheet read/write at all — keeps API usage minimal).
7. **Progress tracking** – Shows "X out of 18 songs" qualified; celebrates at 18/18.
8. **Admin panel** – A password-protected expander shows qualification stats,
   a leaderboard, per-song completion chart, a voice-audit flag, and a raw activity log.

---

## 2. File / folder structure

```
tune-matcher/
├── app.py                  # The Streamlit app (main file)
├── download.py             # yt-dlp helper used to fetch the source YouTube audio
├── split_songs.py          # Splits the long source audio into 18 verse .wav files
├── requirements.txt        # Python dependencies (Streamlit Cloud installs these)
├── packages.txt            # System packages for Streamlit Cloud (ffmpeg)
├── .gitignore              # Ignores secrets, venvs, the summary PDF, etc.
├── PROJECT_SUMMARY.md      # This document
├── README.md               # Short project README
├── songs/                  # The 18 reference .wav files the app plays/compares
├── final/                  # Optional Final Test: all_songs.wav + concat_list.txt
└── .streamlit/
    └── secrets.toml        # Local secrets (NOT committed – gitignored)
```

> `secrets.toml`, the virtual environments (`.venv/`, `.venv_test/`), `test_gsheets.py`,
> and `tune_matcher_summary.pdf` are intentionally git-ignored.

---

## 3. How the score is calculated (`app.py`, sections 1–5)

Audio is loaded at `sr=22050` for both the reference and the user (with a `try/except`
so iPhone recordings that fail to decode show a friendly error instead of crashing).

**Reference features are cached** (`_reference_features`, `@st.cache_data`): each song's
MFCC/chroma/`pyin` is computed **once** and shared across all sessions, so only the user's
audio is processed per attempt. The analysis hop is chosen from the **reference** length
(`512` for short songs, `2048` above ~90s like the Final Test) which keeps it cacheable.
The DTW call is wrapped with a fallback to unconstrained DTW so a recording whose length
differs a lot from the reference can never crash it.

**Three feature layers** are extracted with a consistent per-song `hop_length`:

| Layer | Feature | Purpose |
|-------|---------|---------|
| A | MFCC (z-scored) | Pronunciation / words / vocal-tract timbre |
| B | Chroma | Musical notes, robust to male/female octave shifts |
| C | Pitch `f0` via `pyin` (z-scored) | Melody line / shape; stops a totally different song from matching |

The three layers are stacked into one feature matrix for the reference and the user.

**Dynamic Time Warping (DTW)** aligns the two sequences:
- `metric='euclidean'` (literal value match, stricter than cosine).
- `global_constraints=True, band_rad=0.1` applies a **Sakoe–Chiba band** so the warp
  window is ~10% of the song length. This blocks "cheat paths" that let a wrong song
  meander to a low cost.
  - Note: `band_rad` is a **fraction** of the longer sequence. `band_rad=10` would be a
    no-op (band 10× wider than the matrix). Use small fractions like `0.1`.

**Normalized distance:** `norm_dist = D[-1,-1] / len(path)`.

**Anchor-scaling score** (piecewise linear):
```python
GOOD_MATCH_DIST = 1.90   # where a correct rendition lands
PENALTY_SLOPE   = 150.0  # how fast score drops past the anchor

if norm_dist <= GOOD_MATCH_DIST:
    base_score = 100.0 - ((norm_dist / GOOD_MATCH_DIST) * 5.0)   # ~95–100
else:
    base_score = max(0.0, 95.0 - ((norm_dist - GOOD_MATCH_DIST) * PENALTY_SLOPE))

final_score = base_score * time_ratio   # tempo multiplier
```

**Tempo multiplier (`time_ratio`)** = `min(dur_ref, dur_user) / max(dur_ref, dur_user)`.
Singing much faster or slower drags the score down.

A NaN/Inf guard clamps the score to `0.0` if anything goes wrong.

### Tuning guide
- Correct renditions tend to land near `norm_dist ≈ 1.90` (score ~90s).
- Wrong songs land higher (`≈ 2.15+`) and fall below 50, down to 0.
- To make passing **harder**: lower `GOOD_MATCH_DIST` or raise `PENALTY_SLOPE`.
- To make passing **easier**: raise `GOOD_MATCH_DIST` or lower `PENALTY_SLOPE`.
- The pass threshold itself (`85`) appears in three places in `app.py`: the intro
  caption, the `if score >= 85` check, and the failure message. Change all three together.

---

## 4. Google Sheets integration

Connection uses **`gspread` + `google-auth`** directly (service account). The Sheet must
be **shared** with the service account's email (found in `secrets.toml`) or reads/writes
fail. A single authorized worksheet handle is cached with `@st.cache_resource`.

To keep memory/API usage low and avoid lost updates under concurrency, the app does
**targeted reads/writes** rather than rewriting the whole sheet:
- **Login** reads the roster via `@st.cache_data(ttl=300)` — loaded once and shared across
  sessions (so a participant added mid-event may take up to ~5 min to be able to log in).
- **Progress read** runs once per login (`_get_user_progress`) to populate the completed/
  remaining lists; it's then kept fresh in-session as the user passes more songs.
- **Saving a qualification** appends a single row (`append_row`) for a new song, or fills a
  blank placeholder/updates a single existing row's `A:H` range (`ws.update`) — never a
  full-sheet rewrite. This also prevents two simultaneous qualifiers from clobbering rows.
- **Already-passed songs do no sheet I/O** — if the participant already qualified for a song
  (per the progress loaded at login), re-singing it skips the read and the write entirely.
- **Admin panel** does a fresh full read (`get_all_records`) since it's low-frequency.

> Column order matters: row writes target `A:H` positionally, so the 8 columns must stay in
> the order listed below.

### Sheet columns (header row, exactly these names)
```
User ID | Registration ID | Song | Score | Status | Last Attempt | Voice ID | Voice Print
```

- **User ID** – Participant's name (used as the roster name AND the leaderboard key).
- **Registration ID** – The participant's "password". Pre-fill this column in the roster
  for everyone allowed to log in.
- **Song / Score / Status / Last Attempt** – Filled in when someone qualifies.
  `Status` is set to `QUALIFIED`.
- **Voice ID / Voice Print** – Auto-filled (see section 6). The app creates these columns
  automatically if missing, but it's fine to add them yourself.

### Login matching (forgiving)
- **Name**: case-insensitive and whitespace-insensitive (`"Sahitya   Malladi"` ==
  `"sahitya malladi"`). Extra spaces between first/last name don't matter.
- **Registration ID**: case-insensitive, all whitespace removed.

### Upsert rules
- One row per `(User ID, Song)`.
- **A song is saved only once** — the first qualifying pass. After that, scores are not
  updated (re-singing a passed song does nothing on the sheet).
- The first pass fills the participant's blank roster placeholder row if present, otherwise
  appends a new row.

---

## 5. Admin panel

- Lives in an expander labeled **"📊 Admin Reports & Statistics"** and is always rendered
  (even before participant login).
- Gated by `st.secrets["admin"]["password"]`. If that secret is missing, it shows a clear
  configuration message instead of crashing.
- Shows, for `QUALIFIED` rows only:
  - Total qualifications + unique active users.
  - Leaderboard (songs completed per user).
  - Bar chart of completion rate per song.
  - **Voice Audit** (section 6).
  - Raw activity log (Voice Print column hidden for readability).

---

## 6. Voice ID & Voice Print (anti-cheating heuristic)

> This is a **best-effort review flag, NOT biometric verification.** Only the
> Name + Registration ID are required to log in and qualify. The voice data exists
> purely so an admin can spot the same person qualifying under different names.

- **Voice Print** – A compact numeric "fingerprint" of the singer's timbre: the mean and
  standard deviation of the MFCCs, L2-normalized, stored as a comma-separated string.
- **Voice ID** – A human-readable label (`Voice 1`, `Voice 2`, …). When someone qualifies,
  the app compares their voice print to all stored prints using cosine similarity. If it's
  very similar (≥ `VOICE_MATCH_THRESHOLD = 0.98`) to an existing one, that **same Voice ID**
  is reused; otherwise a new `Voice N` label is minted.
- **Voice Audit** – In the admin panel, if one `Voice ID` is attached to **more than one**
  `User ID`, it's flagged in red as a possible same-singer-across-names case.

Caveats: the print is content-dependent (it reacts to which song/words were sung, not
just the voice), so similarity thresholds are approximate. Treat flags as "look into this",
not proof.

> **About the unreadable Voice Print column:** it is intentionally not human-readable — it
> is the feature vector the matching needs. Voice ID is *derived from* it, so you cannot
> drop Voice Print and keep Voice ID working. (An MD5/hash would only catch byte-identical
> re-uploads, not the same person singing a different take.) If the gibberish bothers you,
> just **hide that column in the Google Sheet UI** (right-click the column → Hide); the
> admin report already omits it.

---

## 6b. Final Test (all 18 songs in sequence) — optional, off by default

- A combined reference track (`final/all_songs.wav`, ~6 minutes) of all 18 songs in order
  is built by `final/concat_list.txt` (ffmpeg concat, mono/22050).
- When enabled, it appears as the **last** dropdown option: "🏆 FINAL TEST — Sing All 18
  Songs in Sequence". The participant must sing the whole sequence, in order, in time.
- It uses the same scoring pipeline, but with a **coarser analysis hop** (`hop_length=2048`
  above ~90s of audio) so the long DTW matrix stays within the ~1 GB Streamlit memory.
- Its result is stored under `Song = FINAL_TEST_ALL_SONGS` and is **excluded** from the
  "X out of 18" progress count.

**Enable / disable on demand (no code change):** it's controlled by a secret flag and is
OFF unless explicitly enabled. Add this to the app secrets (Streamlit Cloud → app →
Settings → Secrets, and/or local `.streamlit/secrets.toml`):

```toml
[features]
final_test = true
```

Set it to `false` (or remove the `[features]` block) to hide the option again. The change
takes effect on the next rerun — no redeploy needed. The option also only shows if the
`final/all_songs.wav` file is present in the repo.

> Note: the 85% threshold and scoring anchors were tuned on ~20s songs. The 6-minute test
> may land at a different raw distance, so you may want to try it and recalibrate the
> threshold/anchors specifically for the final test.

---

## 7. Preparing the songs

1. **Download source audio** (`download.py`): uses `yt-dlp` + `ffmpeg` to pull the audio
   from the source YouTube video into `songs/`.
2. **Split into verses** (`split_songs.py`): uses `ffmpeg` with hardcoded `(start_time,
   filename)` cut points (derived from the video captions) to produce the 18 individual
   `.wav` files. A verse ends when the lyric ends with "sumati" or "vinura vema".

To add/replace songs later: drop new `.wav` files into `songs/`. The selectbox lists them
automatically (sorted by filename; the display name is the filename without `.wav`).

---

## 8. Deployment (Streamlit Community Cloud)

- Push the repo to GitHub. Streamlit Cloud builds from `requirements.txt` (Python deps:
  `streamlit`, `librosa`, `numpy`, `soundfile`, `audioread`, `gspread`, `google-auth`) and
  `packages.txt` (system deps: `ffmpeg`). Recording uses the **native `st.audio_input`**
  (needs a recent Streamlit; `streamlit` is unpinned so Cloud installs a current version).
- **Secrets are NOT committed.** In the Streamlit Cloud app settings, paste the contents of
  your local `.streamlit/secrets.toml`, including:
  - `[connections.gsheets]` with the **real spreadsheet URL**, worksheet name, and the full
    service-account credentials block.
  - `[admin] password = "..."`.
- Common failure: `SpreadsheetNotFound` / 404 → the spreadsheet URL is wrong/placeholder, or
  the Sheet isn't shared with the service-account email.

### Local run
```bash
cd tune-matcher
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .streamlit/secrets.toml with real values
streamlit run app.py
```

---

## 8b. Scalability notes (free tier)

Designed for ~2000 participants completing songs **over time**, with light concurrency.

- **Cost:** Streamlit Community Cloud and the Google Sheets API are both free; neither can
  auto-charge (no billing attached). Exceeding limits yields throttling / `429` "busy"
  errors, never a bill.
- **Main bottleneck = the single free container** (~1 GB RAM, limited CPU, no autoscaling).
  Audio analysis (`pyin` + DTW) is CPU-heavy and largely serialized by the GIL, so a large
  *simultaneous* burst (many dozens at the same moment) can slow down or hit memory limits.
  Spread-out usage is fine.
- **Optimizations already in place to raise the ceiling:**
  - Reference features cached once per song (≈ halves per-attempt CPU).
  - Roster read cached 5 min and shared across sessions.
  - No sheet I/O for already-passed songs; first pass = one write.
  - Only a confirmed (Analyze-pressed) recording is ever processed.
- **If a large simultaneous crowd is expected**, the real fix is capacity, not code: a host
  that scales beyond one container (e.g., Streamlit in Snowflake or self-hosting behind
  autoscaling), and/or staggering participation (which the "songs over days" model allows).

---

## 9. Git / commit conventions

Commit author should be:
```
Phani Prasad Vinnakota Rajendra <vrppinvictaralabs@gmail.com>
```
Never commit `secrets.toml`, virtual environments, or the summary PDF (all gitignored).
