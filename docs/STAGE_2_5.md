# Stage 2.5 — v0.2 continuation training

After v0.1 (Stage 1 + Stage 2, ~34 h on a single A100 SXM 80 GB) we have a working Marathi Kokoro-82M with four named voices. Stage 2.5 is a *continuation* fine-tune — same model, same vocab, same architecture — but with:

- **Cleaner Rasa source** (trailing-silence trimmed; fixes the "Asha eats word-final phonemes" bug).
- **New SPRINGLab/IndicTTS_Marathi data** mixed in (~5 h studio Marathi, 2 speakers).
- **Tighter prosody** (10 more Stage-2 epochs at half the original learning rate).

It's *not* a Stage-1-and-Stage-2 redo. We init `train_second.py` directly from `epoch_2nd_00009.pth` — the alignment knowledge from v0.1 transfers, we just need to nudge the prosody and adversarial training onto the expanded distribution.

## Why Stage 2 only (not full re-train)

Stage 1 teaches alignment + acoustics. SPRINGLab is well-aligned Marathi from a *cleaner* recording environment, not a different language or different phonetic distribution — Stage 1's alignment knowledge transfers as-is. Stage 2's adversarial training is what actually adapts the model to the new acoustics, and it's the cheap half of the recipe (~25 h vs ~9 h for Stage 1). Skipping Stage 1 saves a third of the burn (~$15) without sacrificing quality on the data we care about.

## Data recipe

| Source | Utterances (approx) | Speakers | New for v0.2? |
|---|---|---|---|
| Rasa Marathi (trimmed) | 13,900 | 2 | trim_silence applied; same content as v0.1 |
| IndicVoices-R Marathi (filtered) | ~10,776 | 329 | unchanged |
| SPRINGLab/IndicTTS_Marathi | ~10,939 | 2 | **new** |
| **Combined** | **~35.6k** | **333** | ~50% bigger than v0.1 |

The new SPRINGLab data also gives us two more named voices (`mf_priya`, `mm_arjun` — TBD; voicepacks extracted post-training).

## Files added in v0.2

- [`scripts/data_prep/prepare_springlab_mr.py`](../scripts/data_prep/prepare_springlab_mr.py) — streams `SPRINGLab/IndicTTS_Marathi` and writes a manifest fragment + 24 kHz wavs.
- [`scripts/data_prep/prep_v0_2.sh`](../scripts/data_prep/prep_v0_2.sh) — single-command orchestrator: rasa → trim_silence → ivr → springlab → merge.
- [`configs/config_marathi_v0_2.yml`](../configs/config_marathi_v0_2.yml) — Stage 2.5 training config (`second_stage_load_pretrained: true`, lr halved, joint_epoch=0).
- [`scripts/data_prep/merge_manifests.py`](../scripts/data_prep/merge_manifests.py) — extended to read `springlab_mr.txt`.
- [`scripts/launch_training.sh`](../scripts/launch_training.sh) — `run_stage_2` now sniffs `second_stage_load_pretrained` and skips the `first_stage.pth` requirement when it's true.

## Launch protocol

```bash
# On the pod, after extracting the data + code bundles:

# 1. Stage v0.1 final ckpt where the v0.2 config expects it.
cp checkpoints/epoch_2nd_00009.pth bol_run/training/kokoro_mr_v0_1_final.pth

# 2. Assemble the expanded corpus (idempotent, resume-safe).
cd bol_run
bash scripts/data_prep/prep_v0_2.sh

# 3. Launch Stage 2.5 (Stage 2 only, init from v0.1 final).
cd ..
STAGE=2 CONFIG=../configs/config_marathi_v0_2.yml ./launch_training.sh \
    2>&1 | tee training_v0_2.log
```

Expected runtime: ~25 h on a single A100 SXM 80 GB at `bs=8` + `expandable_segments`. Final ckpt lands at `bol_run/StyleTTS2/logs/kokoro-marathi/epoch_2nd_00009.pth` (overwriting v0.1's, which is preserved as `kokoro_mr_v0_1_final.pth`).

## Post-training

Same as v0.1 — extract voicepacks against the new ckpt (now including `mf_priya` + `mm_arjun` from SPRINGLab speakers), convert to Kokoro format, ONNX export, push to model + ONNX repos + Space.

## Known caveats

- The lr-halving (1e-4 → 5e-5) is a hedge against catastrophic forgetting. If after a few epochs the loss plateaus visibly higher than v0.1 finished at, raise it back to 1e-4 — the model probably has more capacity to absorb than we're letting it.
- `joint_epoch: 0` means SLM joint training kicks in immediately. v0.1 used `joint_epoch: 3` to give the model a Stage-2 warmup; v0.2 doesn't need that warmup since it's resuming from an already-Stage-2-trained model.
- If you change tokenizer/vocab, Stage 2.5 won't help — those changes need a Stage 1 run.
