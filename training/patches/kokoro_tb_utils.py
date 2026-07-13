import sys
import os

# Save original path
sys_path_save = list(sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

# Import original kokoro_tb_utils module
import kokoro_tb_utils

# Restore path
sys.path = sys_path_save

# Expose everything from the original kokoro_tb_utils module so downstream code is unbroken
globals().update({k: v for k, v in kokoro_tb_utils.__dict__.items() if not k.startswith('__')})

# Override extract_voicepack
def extract_voicepack(model, root_path, device, n_samples=200):
    # Check if we should read from parquet/SQL instead of WAV files
    is_parquet = False
    if isinstance(root_path, str) and ("*.parquet" in root_path or root_path.endswith(".parquet")):
        is_parquet = True

    if not is_parquet:
        # Delegate back to the original function
        return kokoro_tb_utils.extract_voicepack(model, root_path, device, n_samples)

    # SQL/parquet extraction
    import duckdb
    import soundfile as sf
    import torchaudio
    import torch
    import io
    import logging

    logger = logging.getLogger(__name__)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)
    mel_mean, mel_std = -4, 4

    audio_samples = []
    
    try:
        con = duckdb.connect()
        # Randomly fetch up to n_samples audio bytes
        rows = con.execute(f"SELECT audio.bytes FROM read_parquet('{root_path}') LIMIT {n_samples}").fetchall()
        for r in rows:
            if r[0] is not None:
                try:
                    data, file_sr = sf.read(io.BytesIO(r[0]), dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    waveform = torch.from_numpy(data).unsqueeze(0)
                    if file_sr != 24000:
                        waveform = torchaudio.functional.resample(waveform, file_sr, 24000)
                    audio_samples.append(waveform.to(device))
                except Exception as inner_e:
                    logger.warning(f"[SQL Wrapper] extract_voicepack: skipping parquet sample: {inner_e}")
        con.close()
    except Exception as e:
        logger.warning(f"[SQL Wrapper] extract_voicepack: failed to load audio from parquet: {e}")

    acoustic_styles = []
    prosodic_styles = []

    with torch.no_grad():
        for waveform in audio_samples:
            try:
                mel = mel_transform(waveform)
                mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std
                if mel.shape[-1] < 80:
                    continue
                mel_input = mel.unsqueeze(1)  # [1, 1, 80, T]
                acoustic_styles.append(model.style_encoder(mel_input).cpu())
                prosodic_styles.append(model.predictor_encoder(mel_input).cpu())
            except Exception as e:
                logger.warning(f"[SQL Wrapper] extract_voicepack: style encoding failed: {e}")

    if not acoustic_styles:
        logger.warning("[SQL Wrapper] extract_voicepack: no valid audio files processed")
        return None, 0.0, 0.0

    avg_acoustic = torch.cat(acoustic_styles, dim=0).mean(dim=0)  # [128]
    avg_prosodic = torch.cat(prosodic_styles, dim=0).mean(dim=0)  # [128]
    voicepack = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]

    acoustic_norm = avg_acoustic.norm().item()
    prosodic_norm = avg_prosodic.norm().item()

    return voicepack.to(device), acoustic_norm, prosodic_norm
