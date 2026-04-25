"""prepare_springlab_mr.py — stream SPRINGLab/IndicTTS_Marathi into the
bol-tts-marathi training layout.

The dataset is studio-quality Marathi with two speakers split by `gender`
(0=female, 1=male). ~10,939 rows total, ~5 hours of audio. Compared to
Rasa+IV-R it scores higher on listening tests; we mix it in for v0.2/Stage 2.5
to anchor quality. Voicepacks for Mukta+Dnyanesh were already extracted from
this same source — adding it to training tightens those voices.

Output layout (relative to bol-tts-marathi/ repo root):
  dataset/audio/springlab_mr/<gender>_<idx:06d>.wav     24 kHz mono 16-bit
  training/springlab_mr.txt                              path|ipa|speaker
  training/springlab_mr_stats.json                       run summary

Speaker IDs: `springlab_female`, `springlab_male`. (Same convention as
prepare_rasa_mr.py's `marathi_female` / `marathi_male`.)

Usage:
  python3 scripts/data_prep/prepare_springlab_mr.py
  python3 scripts/data_prep/prepare_springlab_mr.py --max 100   # smoke test
  python3 scripts/data_prep/prepare_springlab_mr.py --dry-run

Resume: skips entries already present in the manifest by (gender, idx).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from misaki import espeak
from tqdm import tqdm

# ── Filter thresholds ────────────────────────────────────────────────────────
DURATION_MIN = 1.5       # seconds
DURATION_MAX = 15.0
TARGET_SR = 24_000
GENDER_TO_NAME = {0: "springlab_female", 1: "springlab_male"}

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scripts/data_prep/, parents[2] = repo root.
REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2]))
DST_AUDIO = REPO_ROOT / "dataset" / "audio" / "springlab_mr"
DST_MANIFEST = REPO_ROOT / "training" / "springlab_mr.txt"
DST_STATS = REPO_ROOT / "training" / "springlab_mr_stats.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=None, help="cap rows for smoke testing")
    ap.add_argument("--dry-run", action="store_true", help="phonemize + filter, but do not write anything")
    args = ap.parse_args()

    DST_AUDIO.mkdir(parents=True, exist_ok=True)
    DST_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read existing manifest entries by (gender, idx) -> wav_rel
    seen: set[tuple[str, int]] = set()
    if DST_MANIFEST.exists():
        with DST_MANIFEST.open() as f:
            for line in f:
                if not line.strip():
                    continue
                wav_rel = line.split("|", 1)[0]
                # springlab_mr/<gender>_<idx>.wav
                stem = Path(wav_rel).stem  # springlab_female_000123 etc.
                # Speaker is everything before the trailing _<digits>
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    seen.add((parts[0], int(parts[1])))
        print(f"resume: {len(seen)} existing entries; will skip duplicates")

    g2p = espeak.EspeakG2P(language="mr")
    print("loading dataset (streaming)…")
    ds = load_dataset("SPRINGLab/IndicTTS_Marathi", split="train", streaming=True)

    new_lines: list[str] = []
    counts: dict[str, int] = {n: 0 for n in GENDER_TO_NAME.values()}
    skipped = {"duration": 0, "phonemize": 0, "duplicate": 0}

    # We don't know per-gender index ahead of time; we'll use the dataset's
    # natural order and increment per gender.
    next_idx = {n: 0 for n in GENDER_TO_NAME.values()}
    # Bump next_idx past the highest (gender, idx) already seen.
    for spk, idx in seen:
        if spk in next_idx and idx >= next_idx[spk]:
            next_idx[spk] = idx + 1

    open_mode = "a" if DST_MANIFEST.exists() else "w"
    manifest_fh = None if args.dry_run else DST_MANIFEST.open(open_mode, encoding="utf-8")

    iterator = ds
    if args.max is not None:
        from itertools import islice
        iterator = islice(ds, args.max)

    pbar = tqdm(iterator, total=args.max if args.max else 10_939, desc="springlab")
    for row in pbar:
        gender = row.get("gender")
        if gender not in GENDER_TO_NAME:
            continue
        speaker = GENDER_TO_NAME[gender]

        # Decode audio. HF Audio feature gives {"array": np.ndarray, "sampling_rate": int}
        audio = row["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        dur = float(len(arr)) / sr
        if dur < DURATION_MIN or dur > DURATION_MAX:
            skipped["duration"] += 1
            continue

        if sr != TARGET_SR:
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

        # Phonemize
        try:
            ipa, _ = g2p(row["text"])
        except Exception:
            skipped["phonemize"] += 1
            continue
        if not ipa.strip():
            skipped["phonemize"] += 1
            continue

        idx = next_idx[speaker]
        if (speaker, idx) in seen:
            skipped["duplicate"] += 1
            next_idx[speaker] += 1
            continue
        next_idx[speaker] += 1

        wav_rel = f"springlab_mr/{speaker}_{idx:06d}.wav"
        if not args.dry_run:
            wav_path = DST_AUDIO / f"{speaker}_{idx:06d}.wav"
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")
            manifest_fh.write(f"{wav_rel}|{ipa}|{speaker}\n")
            manifest_fh.flush()

        counts[speaker] += 1
        new_lines.append(wav_rel)

    if manifest_fh:
        manifest_fh.close()

    stats = {
        "rows_added": sum(counts.values()),
        "per_speaker": counts,
        "skipped": skipped,
        "target_sr": TARGET_SR,
        "duration_window_sec": [DURATION_MIN, DURATION_MAX],
    }
    if not args.dry_run:
        with DST_STATS.open("w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
