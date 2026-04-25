#!/usr/bin/env bash
# prep_v0_2.sh — assemble the Stage 2.5 (v0.2) corpus.
#
# Combines Rasa (trimmed) + IndicVoices-R (filtered) + SPRINGLab (new) into the
# unified train_list.txt / val_list.txt. The trim_silence step on the Rasa
# audio is the data-prep fix for the trailing-silence bug that was causing
# Asha + Vivek to eat word-final phonemes (see docs/VOICEPACKS.md).
#
# Idempotent: each prep script is resume-aware. Trim step skips already-trimmed
# files. Re-run anytime the upstream scripts change.
#
# Run from anywhere; uses BOL_REPO env var, falls back to script-relative.

set -euo pipefail

REPO=${BOL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$REPO"

echo "── Stage 2.5 data prep ──────────────────────────────────────────────"
echo "repo: $REPO"

# 1. Rasa Marathi → dataset/audio/rasa/<wavs>
echo
echo "[1/5] prepare_rasa_mr.py"
python3 scripts/data_prep/prepare_rasa_mr.py

# 2. Trim leading + trailing silence on Rasa training audio. Done in-place
#    over a sibling _trimmed dir, then we point the manifest at the trimmed
#    dir (overwrites manifest paths from rasa/ → rasa_trimmed/).
echo
echo "[2/5] trim_silence on Rasa (50 ms pad each side)"
python3 scripts/trim_silence.py \
    --src-dir dataset/audio/rasa \
    --dst-dir dataset/audio/rasa_trimmed
sed -i.bak 's|^rasa/|rasa_trimmed/|' training/rasa_mr.txt
rm -f training/rasa_mr.txt.bak
echo "    rewrote training/rasa_mr.txt to point at rasa_trimmed/"

# 3. IndicVoices-R Marathi (filtered) → dataset/audio/indicvoices_r/<wavs>
echo
echo "[3/5] prepare_indicvoices_r_mr.py"
python3 scripts/data_prep/prepare_indicvoices_r_mr.py

# 4. SPRINGLab IndicTTS_Marathi → dataset/audio/springlab_mr/<wavs>  (new for v0.2)
echo
echo "[4/5] prepare_springlab_mr.py"
python3 scripts/data_prep/prepare_springlab_mr.py

# 5. Merge into train_list.txt + val_list.txt (95/5 stratified by speaker)
echo
echo "[5/5] merge_manifests.py"
python3 scripts/data_prep/merge_manifests.py

echo
echo "── done. Total combined corpus written to training/train_list.txt ──"
wc -l training/train_list.txt training/val_list.txt
