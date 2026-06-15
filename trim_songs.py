"""Trim leading/trailing instrumental from the reference padyams.

Every padyam starts after an instrumental intro and ends on the closing word
("Sumathi" / "Vinura Vema"). Anything BEFORE the first sung word and AFTER the last
sung word is instrumental we don't want the scorer to consider; music in the MIDDLE
is fine to keep.

This script detects the VOCAL span (first -> last singing) using a softmask vocal
separation + an energy gate, then cuts the ORIGINAL audio to that span. Cutting the
original (not the separated vocals) keeps full audio quality and preserves any
mid-song music between the first and last word.

Usage:
    python trim_songs.py            # auto-detect, write copies to songs_trimmed/
    python trim_songs.py --inplace  # overwrite songs/ (backs up to songs_backup/ first)

If auto-detection is off for a song, listen to songs_trimmed/<song>.wav, note the
real start/end seconds, add them to MANUAL below, and re-run. Manual entries skip
auto-detection:
    MANUAL = {"06_koorimi_gala_dinamulalo_sumati": (1.8, 17.4)}
"""
import os
import shutil
import argparse
import numpy as np
import librosa
import soundfile as sf

SRC_DIR = "songs"
OUT_DIR = "songs_trimmed"
SR = 22050
PAD_SEC = 0.30        # breathing room kept around the detected vocals
ENERGY_GATE = 0.06    # vocal RMS above this fraction of its peak counts as "singing"

# song_name (without .wav) -> (start_sec, end_sec). Overrides auto-detection.
MANUAL = {
    # Vocals ("...Vinura Vema") end ~18s; everything after is just music.
    "18_uppu_kappurambu_vemana": (0.0, 18.0),
}


def detect_vocal_span(y, sr):
    """Return (start_sec, end_sec) of the singing via softmask vocal separation."""
    S_full, _ = librosa.magphase(librosa.stft(y))
    # Accompaniment estimate: compare each frame to similar frames across the track.
    win = max(1, int(librosa.time_to_frames(2, sr=sr)))
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric="cosine", width=win)
    S_filter = np.minimum(S_full, S_filter)
    mask_v = librosa.util.softmask(S_full - S_filter, 10 * S_filter, power=2)
    S_vocal = mask_v * S_full

    rms = librosa.feature.rms(S=S_vocal)[0]
    dur = librosa.get_duration(y=y, sr=sr)
    if rms.size == 0 or rms.max() <= 0:
        return 0.0, dur
    voiced = np.where(rms > rms.max() * ENERGY_GATE)[0]
    if voiced.size == 0:
        return 0.0, dur
    start_t = max(0.0, librosa.frames_to_time(voiced[0], sr=sr) - PAD_SEC)
    end_t = librosa.frames_to_time(voiced[-1], sr=sr) + PAD_SEC
    return start_t, end_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite songs/ (backs up to songs_backup/ first)")
    args = ap.parse_args()

    out = SRC_DIR if args.inplace else OUT_DIR
    if args.inplace:
        if not os.path.exists("songs_backup"):
            shutil.copytree(SRC_DIR, "songs_backup")
            print("Backed up originals to songs_backup/\n")
    else:
        os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".wav"))
    print(f"{'song':45s} {'orig':>6s} {'start':>6s} {'end':>6s} {'new':>6s}")
    print("-" * 72)
    for f in files:
        name = f[:-4]
        y, sr = librosa.load(os.path.join(SRC_DIR, f), sr=SR)
        dur = librosa.get_duration(y=y, sr=sr)
        s, e = MANUAL[name] if name in MANUAL else detect_vocal_span(y, sr)
        s = max(0.0, min(s, dur))
        e = max(s, min(e, dur))
        sf.write(os.path.join(out, f), y[int(s * sr):int(e * sr)], sr)
        tag = " (manual)" if name in MANUAL else ""
        print(f"{name:45s} {dur:6.1f} {s:6.1f} {e:6.1f} {(e - s):6.1f}{tag}")
    print(f"\nDone -> {out}/   (listen, then add MANUAL overrides for any that look off)")


if __name__ == "__main__":
    main()
