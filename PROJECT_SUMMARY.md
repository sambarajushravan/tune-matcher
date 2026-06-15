# Sataka Sankharavam Tune Matcher Challenge

A Streamlit web app where registered participants sing along to reference songs,
get an automatic match score (pronunciation + pacing), and have their qualified songs
tracked on a Google Sheets leaderboard. After each attempt the app shows coaching feedback
(notes/melody bars are guidance only). Includes a password-gated admin panel with stats
and a lightweight family-scoped voice-audit flag for spotting possible cheating.

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
   Match Score (%)**, driven by **pronunciation (MFCC)** and **pacing**. Musical notes and
   melody are shown as coaching bars only — they do not affect pass/fail.
6. **Coaching feedback** – After analysis, shows pronunciation / notes / melody progress
   bars, a plain-language pacing message (too fast / too slow / good), and tips to improve.
   Raw Dist and Tempo Acc are hidden in a "Technical details (for organizers)" expander.
7. **Pass threshold = 85%** – At 85%+ the attempt is `QUALIFIED` and saved. **Once a
   participant has passed a song, scores are not updated** (re-singing a passed song does
   no sheet read/write at all — keeps API usage minimal).
8. **Progress tracking** – Shows "X out of 18 songs" qualified; celebrates at 18/18.
9. **Admin panel** – A password-protected expander shows qualification stats,
   a leaderboard, per-song completion chart, a voice-audit flag, and a raw activity log.

---

## 2. File / folder structure

```
tune-matcher/
├── app.py                  # The Streamlit app (main file)
├── download.py             # yt-dlp helper used to fetch the source YouTube audio
├── split_songs.py          # Splits the long source audio into 18 verse .wav files
├── trim_songs.py           # Trims instrumental intro/outro from reference .wav files
├── requirements.txt        # Python dependencies (Streamlit Cloud installs these)
├── packages.txt            # System packages for Streamlit Cloud (ffmpeg)
├── .gitignore              # Ignores secrets, venvs, the summary PDF, etc.
├── PROJECT_SUMMARY.md      # This document
├── README.md               # Short project README
├── songs/                  # The 18 trimmed reference .wav files the app plays/compares
├── songs_backup/           # Auto-backup of originals before in-place trim (gitignored)
├── final/                  # Optional Final Test: all_songs.wav + concat_list.txt
└── .streamlit/
    └── secrets.toml        # Local secrets (NOT committed – gitignored)
```

> `secrets.toml`, the virtual environments (`.venv/`, `.venv_test/`, `.venv_trim/`),
> `songs_backup/`, `songs_trimmed/`, `test_gsheets.py`, generated PDFs (`*.pdf`), and
> `tune_matcher_summary.pdf` are intentionally git-ignored.

---

## 3. How the score is calculated (`app.py`)

Audio is loaded at `sr=22050` for both the reference and the user (with a `try/except`
so iPhone recordings that fail to decode show a friendly error instead of crashing).

**Reference features are cached** (`_reference_features`, `@st.cache_data`): each song's
MFCC/chroma/pitch is computed **once** and shared across all sessions. The cache key
includes the file's **modification time** so trimmed/replaced `.wav` files invalidate stale
entries (without this, pacing could still reference an old ~30s duration after trimming
song 18 to 18s).

The analysis hop is chosen from the **reference** length (`512` for short songs, `2048`
above ~90s like the Final Test). The DTW call has a fallback to unconstrained DTW so a
recording whose length differs a lot from the reference can never crash it.

### What drives pass/fail

| Component | Affects Overall Score? | How |
|-----------|------------------------|-----|
| **Pronunciation (MFCC)** | **Yes — main driver** | MFCC-only DTW with Sakoe–Chiba band |
| **Pacing / timing** | **Yes — up to 40% blend** | Recording length vs. reference length |
| **Musical notes (Chroma)** | No — coaching only | Shown as guidance bar after analysis |
| **Melody line (Pitch f0)** | No — coaching only | Shown as guidance bar after analysis |

Chroma and pitch are still extracted for the **coaching breakdown** ("How you did"), but
they are **not** stacked into the DTW that produces the Overall Match Score.

### Scoring pipeline

1. **Extract features** with a consistent per-song `hop_length`:
   - MFCC (z-scored) — pronunciation / words / vocal-tract timbre
   - Chroma — musical notes (coaching only)
   - Pitch `f0` via `pyin` (z-scored) — melody line (coaching only)

2. **DTW on MFCC only** aligns reference and user pronunciation:
   - `metric='euclidean'`
   - `global_constraints=True, band_rad=0.1` (Sakoe–Chiba band ~10% of song length)

3. **Normalized distance:** `norm_dist = D[-1,-1] / len(path)` (MFCC-only).

4. **Anchor-scaling base score** (piecewise linear):
```python
GOOD_MATCH_DIST = 1.45   # MFCC-only anchor (correct human rendition ~1.3–1.5)
PENALTY_SLOPE   = 80.0

if norm_dist <= GOOD_MATCH_DIST:
    base_score = 100.0 - ((norm_dist / GOOD_MATCH_DIST) * 5.0)   # ~95–100
else:
    base_score = max(0.0, 95.0 - ((norm_dist - GOOD_MATCH_DIST) * PENALTY_SLOPE))
```

5. **Pacing blend** (timing matters):
```python
TEMPO_TOLERANCE = 0.90   # no penalty within ~10% of reference length
TEMPO_WEIGHT    = 0.40   # pacing contributes up to 40% of the final blend

time_ratio = min(dur_ref, dur_user) / max(dur_ref, dur_user)
if time_ratio >= TEMPO_TOLERANCE:
    tempo_factor = 1.0
else:
    tempo_factor = max(0.0, (time_ratio - 0.5) / (TEMPO_TOLERANCE - 0.5))

tempo_blend = (1.0 - TEMPO_WEIGHT) + (TEMPO_WEIGHT * tempo_factor)
final_score = base_score * tempo_blend
```

A NaN/Inf guard clamps the score to `0.0` if anything goes wrong.

### Coaching feedback (separate from pass/fail)

After scoring, the app shows:
- Progress bars for pronunciation, musical notes, and melody (each mapped from per-layer
  DTW path distance along the MFCC alignment).
- A plain-language **pacing** message (too fast / too slow / good), using the **current**
  reference file duration.
- **Tips to improve** — prioritizes pronunciation and pacing; notes/melody tips only if
  very low and marked "(Optional)".
- **Technical details (for organizers)** expander: Raw Dist (MFCC), Tempo Acc, base score,
  tempo blend, and reference length in seconds.

### Tuning guide
- Correct renditions tend to land near `norm_dist ≈ 1.3–1.5` (MFCC-only) → score ~90–95.
- To make passing **harder**: lower `GOOD_MATCH_DIST` or raise `PENALTY_SLOPE`.
- To make passing **easier**: raise `GOOD_MATCH_DIST` or lower `PENALTY_SLOPE`.
- To weight **pacing** more: raise `TEMPO_WEIGHT` or lower `TEMPO_TOLERANCE`.
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

## 6. Voice ID & Voice Print (anti-cheating heuristic) — FAMILY-SCOPED

> This is a **best-effort review flag, NOT biometric verification.** Only the
> Name + Registration ID are required to log in and qualify. The voice data exists
> purely so an admin can spot **one person in a family singing for relatives**.

**Why family-scoped?** A Registration ID can be shared by a whole family (different names,
same ID). The goal is to catch one family member singing under another's name — *not* to
identify people across the whole event. Voice IDs are therefore numbered **per Registration
ID**: each family starts fresh at `Voice 1`.

- **Voice Print** – A compact numeric "fingerprint" of the singer's timbre: the mean and
  standard deviation of the MFCCs, L2-normalized, stored as a comma-separated string.
  **MFCC coefficient 0 (overall loudness/energy) is dropped on purpose** — it is dominated
  by the song (which everyone sings identically), and including it made every print look
  alike and collapse onto a single "Voice 1". Dropping it lets the timbre coefficients
  actually distinguish singers. (This changed the print length, so prints saved before this
  change won't match new ones — clear the `Voice ID`/`Voice Print` columns once when
  upgrading.)
- **Voice ID** – A label (`Voice 1`, `Voice 2`, …) **scoped to one family**. When someone
  qualifies, the app compares their print only to other prints **with the same Registration
  ID** using cosine similarity. If it's very similar (≥ `VOICE_MATCH_THRESHOLD = 0.95`) to an
  existing print in that family, the **same Voice ID** is reused; otherwise the next `Voice N`
  for that family is minted. So a family of 4 distinct singers tends to get `Voice 1`–`Voice 4`,
  while the same person singing under two names lands on a single Voice ID.
- **Voice Audit** – In the admin panel, results are grouped by **Registration ID + Voice ID**.
  If one `Voice ID` within a family is attached to **more than one** `User ID`, it's flagged in
  red (with the names and songs involved) as a possible same-singer-within-a-family case.
  `Voice 1` existing in every family is normal and is **not** flagged on its own.

Caveats: the print is content-dependent (it reacts to which song/words were sung, not just
the voice), so it is approximate — different relatives can occasionally merge, and one
person's takes can occasionally split. `VOICE_MATCH_THRESHOLD` may need tuning against real
recordings. Treat every flag as "listen to this", not proof.

> **About the unreadable Voice Print column:** it is intentionally not human-readable — it
> is the feature vector the matching needs. Voice ID is *derived from* it, so you cannot
> drop Voice Print and keep Voice ID working. (An MD5/hash would only catch byte-identical
> re-uploads, not the same person singing a different take.) If the gibberish bothers you,
> just **hide that column in the Google Sheet UI** (right-click the column → Hide); the
> admin report already omits it.

---

## 6b. Final Test (all 18 songs in sequence) — optional, off by default

- A combined reference track (`final/all_songs.wav`, ~5–6 minutes after trimming) of all
  18 trimmed songs in order is built by concatenating the files in `songs/` (or via
  `final/concat_list.txt` for ffmpeg concat, mono/22050).
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
3. **Trim instrumental intro/outro** (`trim_songs.py`): each reference should contain only
   the **vocal span** — from the first sung word through the closing "Sumathi"/"Vinura Vema".
   Trailing (and leading) instrumental is removed so the scorer isn't penalized for music
   the singer shouldn't match. Mid-song music between words is kept.

   ```bash
   python trim_songs.py              # writes trimmed copies to songs_trimmed/ for review
   python trim_songs.py --inplace    # overwrites songs/ (backs up to songs_backup/ first)
   ```

   Auto-detection uses vocal separation + an energy gate. If a song trims incorrectly,
   add manual `(start_sec, end_sec)` overrides in the `MANUAL` dict inside `trim_songs.py`
   (e.g. song 18 capped at 18.0s). After trimming, rebuild the Final Test track:
   `final/all_songs.wav` is concatenated from the trimmed `songs/` files.

To add/replace songs later: drop new `.wav` files into `songs/` (trim first if needed).
The selectbox lists them automatically (sorted by filename; the display name is the
filename without `.wav`). Replacing a file invalidates the reference cache automatically
(via file modification time).

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
