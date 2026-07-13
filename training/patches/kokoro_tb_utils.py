import sys
import os
import importlib.util

# Load original kokoro_tb_utils module using importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
original_tb_path = os.path.join(repo_root, "StyleTTS2", "kokoro_tb_utils.py")

spec = importlib.util.spec_from_file_location("tb_original", original_tb_path)
tb_original = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(repo_root, "StyleTTS2"))
spec.loader.exec_module(tb_original)
sys.path.pop(0)

# Expose everything from the original kokoro_tb_utils module so downstream code is unbroken
globals().update({k: v for k, v in tb_original.__dict__.items() if not k.startswith('__')})

# Override for Tamil test sentences
TEST_SENTENCES = [
    "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "தமிழ் மொழி மிகவும் பழமையான மற்றும் அழகான மொழி.",
    "நாளைக்கு மழை பெய்யும் என்று வானிலை அறிக்கை கூறுகிறது.",
    "புத்தகம் வாசிப்பது மனிதனின் அறிவை வளர்க்கும்.",
    "விருந்தினர்களை உபசரிப்பது தமிழர்களின் சிறந்த பண்பு."
]

def prepare_test_tokens(text_cleaner):
    """Convert Tamil test sentences to token ID lists via espeak G2P.

    Returns a list of (display_text, token_ids) tuples. Sentences that
    fail G2P or produce sequences longer than 510 tokens are skipped.
    """
    try:
        from misaki import espeak
        g2p = espeak.EspeakG2P(language="ta")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[SQL Wrapper] Could not load Tamil G2P for TensorBoard inference: {e}")
        return []

    result = []
    for text in TEST_SENTENCES:
        try:
            g2p_out = g2p(text)
            ipa = g2p_out[0] if isinstance(g2p_out, tuple) else g2p_out
            token_ids = text_cleaner(ipa)
            if not token_ids or len(token_ids) > 510:
                continue
            result.append((text, token_ids))
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f'[SQL Wrapper] G2P failed for test sentence "{text[:40]}": {e}')
    return result

# Override extract_voicepack
def extract_voicepack(model, root_path, device, n_samples=200):
    # Check if we should read from parquet/SQL instead of WAV files
    is_parquet = False
    if isinstance(root_path, str) and ("*.parquet" in root_path or root_path.endswith(".parquet")):
        is_parquet = True

    if not is_parquet:
        # Delegate back to the original function
        return tb_original.extract_voicepack(model, root_path, device, n_samples)

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
