import sys
import os
import importlib.util

# Load the original meldataset module using importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
original_meldataset_path = os.path.join(repo_root, "StyleTTS2", "meldataset.py")

spec = importlib.util.spec_from_file_location("meldataset_original", original_meldataset_path)
meldataset_original = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(repo_root, "StyleTTS2"))
spec.loader.exec_module(meldataset_original)
sys.path.pop(0)

# Expose everything from the original meldataset module so downstream code is unbroken
globals().update({k: v for k, v in meldataset_original.__dict__.items() if not k.startswith('__')})

# Override FilePathDataset subclass
class FilePathDataset(meldataset_original.FilePathDataset):
    def __init__(self, data_list, root_path, **kwargs):
        # Detect if we should use DuckDB SQL queries
        # The configuration root_path contains "*.parquet" if we are in SQL parquet mode
        self.use_sql = isinstance(root_path, str) and ("*.parquet" in root_path or root_path.endswith(".parquet"))
        
        super().__init__(data_list, root_path, **kwargs)
        
        if self.use_sql:
            print(f"[SQL Wrapper] Initialized SQL Dataset with parquet source: {root_path}")

    def _load_tensor(self, data):
        if not self.use_sql:
            return super()._load_tensor(data)

        # SQL-based lazy loading
        import duckdb
        import soundfile as sf
        import librosa
        import numpy as np
        import torch
        import io

        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)

        if not hasattr(self, "_con") or self._con is None:
            self._con = duckdb.connect()
            self._con.execute(f"CREATE VIEW dataset_view AS SELECT audio.bytes as audio_bytes, row_number() OVER () - 1 as row_idx FROM read_parquet('{self.root_path}') WHERE text IS NOT NULL")
        
        row_idx = int(wave_path)
        res = self._con.execute(f"SELECT audio_bytes FROM dataset_view WHERE row_idx = {row_idx}").fetchone()
        if res is None or res[0] is None:
            raise ValueError(f"[SQL Wrapper] Row index {row_idx} not found in parquet files at {self.root_path}")
        audio_bytes = res[0]
        
        wave, sr = sf.read(io.BytesIO(audio_bytes))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)

        return wave, text, speaker_id

# Monkey patch the original module so its internal functions (like build_dataloader)
# use our overridden FilePathDataset subclass!
meldataset_original.FilePathDataset = FilePathDataset
