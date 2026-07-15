# training/

Source-controlled training artifacts needed to run or reproduce fine-tuning. Files NOT listed here (large checkpoints, speaker manifests derived from datasets) are `.gitignore`-d; your local copy will fill them in via the data-prep scripts.

## Files

| File | Purpose |
|---|---|
| `kokoro_symbols.py` | Marathi-patched Kokoro symbol table. ɭ (U+026D) at slot 144, replacing a PUA placeholder. Copy into `StyleTTS2/kokoro_symbols.py` before launching training (the launcher does this automatically). |
| `kokoro_tb_utils_mr.py` | Marathi test sentences + voicepack extraction for TensorBoard audio previews during training. Copy into `StyleTTS2/kokoro_tb_utils.py` before training. |
| `OOD_texts.txt` | ~2,000 Marathi out-of-domain sentences used by Stage 2 for adversarial prosody training. Can be regenerated via `scripts/data_prep/generate_ood_marathi.py`. |

## Files NOT in git (local only)

These are produced by data prep and live here during training:

| File | Produced by | Size |
|---|---|---|
| `rasa_mr.txt` | `scripts/data_prep/prepare_rasa_mr.py` | ~3 MB |
| `indicvoices_r_mr.txt` | `scripts/data_prep/prepare_indicvoices_r_mr.py` | ~3 MB |
| `indicvoices_r_mr_stats.json` | `scripts/data_prep/prepare_indicvoices_r_mr.py` | ~50 KB |
| `train_list.txt` | `scripts/data_prep/merge_manifests.py` | ~6 MB |
| `val_list.txt` | `scripts/data_prep/merge_manifests.py` | ~300 KB |
| `speaker_map.json` | `scripts/data_prep/fix_speaker_ids.py` | ~20 KB |
| `kokoro_base.pth` | `scripts/convert_kokoro_weights.py` | ~312 MB |
| `config.json` | Download from HuggingFace `hexgrad/Kokoro-82M` | ~150 KB |

## Why kokoro_symbols.py is here and not in StyleTTS2/

We overlay — not fork — the `StyleTTS2/` submodule. The launcher copies our `kokoro_symbols.py` on top of the submodule's version before each training run. This keeps the submodule pointer clean so semidark's upstream updates can be pulled without merge conflicts.

## Invariants (read before editing)

- `kokoro_symbols.py` must always keep the 178-slot layout. Adding a new phoneme means REPLACING an unused PUA placeholder (U+E000-U+F8FF range), never inserting a new slot. Re-numbering breaks every trained checkpoint.
- `configs/config_mr.json["vocab"]` must mirror every real-phoneme slot from `kokoro_symbols.py`. `scripts/diagnostics/diagnose_nan.py` verifies this.

## How to Resume Training if it Stops

If your training run is interrupted (e.g., Google Colab disconnects or the session times out), you can resume training from any saved epoch checkpoint:

1. Locate the latest checkpoint in your linked logs directory (e.g. `logs/kokoro-tamil/epoch_1st_00005.pth`).
2. Open your configuration file (e.g., `configs/config_tamil_ft.yml`).
3. Update the `pretrained_model` setting to point to the checkpoint:
   ```yaml
   pretrained_model: "logs/kokoro-tamil/epoch_1st_00005.pth"
   ```
4. Change the `load_only_params` setting to `false` (this ensures the optimizer state, learning rate schedulers, and current epoch counters are restored):
   ```yaml
   load_only_params: false
   ```
5. Relaunch training. Training will automatically pick up from the next epoch!

> [!WARNING]
> **Stage 2 Resume Warning**: Resuming Stage 2 in the middle of adversarial training is unstable in the upstream StyleTTS2 codebase due to optimizer mismatches and learning rate reset behaviors. If Stage 2 gets interrupted, it is recommended to restart Stage 2 from the beginning using your final Stage 1 checkpoint (`first_stage.pth`).

