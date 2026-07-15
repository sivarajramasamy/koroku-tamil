import sys
import os
import importlib.util

# Load original utils module using importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
original_utils_path = os.path.join(repo_root, "StyleTTS2", "utils.py")

spec = importlib.util.spec_from_file_location("utils_original", original_utils_path)
utils_original = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(repo_root, "StyleTTS2"))
spec.loader.exec_module(utils_original)
sys.path.pop(0)

# Expose everything from the original utils module so downstream code is unbroken
globals().update({k: v for k, v in utils_original.__dict__.items() if not k.startswith('__')})

# Override get_data_path_list
def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    # Detect if we should use DuckDB SQL queries
    is_sql_query = False
    for p in [train_path, val_path]:
        if p is not None:
            up_p = p.strip().upper()
            if up_p.startswith("SELECT") or up_p.startswith("WITH"):
                is_sql_query = True
                break

    if is_sql_query:
        import duckdb
        from misaki.espeak import EspeakG2P

        print("[SQL Wrapper] Initializing G2P phonemizer for Tamil ('ta')...")
        g2p = EspeakG2P(language="ta")

        # Connect to DuckDB
        con = duckdb.connect()

        # Execute training query
        print(f"[SQL Wrapper] Executing train query: {train_path}")
        train_rows = con.execute(train_path).fetchall()
        
        # Execute validation query
        print(f"[SQL Wrapper] Executing val query: {val_path}")
        val_rows = con.execute(val_path).fetchall()

        # Build speaker map from names to stringified integer indices dynamically
        all_rows = train_rows + val_rows
        unique_speakers = sorted(list(set(row[1] for row in all_rows)))
        speaker_map = {name: str(idx) for idx, name in enumerate(unique_speakers)}
        print(f"[SQL Wrapper] Found {len(unique_speakers)} speakers: {speaker_map}")

        print(f"[SQL Wrapper] Found {len(train_rows)} training rows. Phonemizing transcripts...")
        train_list = []
        for row in train_rows:
            # We expect the query to return: (text, speaker_id, idx)
            text, speaker_name, idx = row[0], row[1], row[2]
            try:
                ipa, _ = g2p(text)
            except Exception as e:
                print(f"[SQL Wrapper] G2P failed for text '{text}': {e}")
                ipa = text
            train_list.append(f"{idx}|{ipa}|{speaker_map[speaker_name]}")

        print(f"[SQL Wrapper] Found {len(val_rows)} validation rows. Phonemizing transcripts...")
        val_list = []
        for row in val_rows:
            text, speaker_name, idx = row[0], row[1], row[2]
            try:
                ipa, _ = g2p(text)
            except Exception as e:
                print(f"[SQL Wrapper] G2P failed for text '{text}': {e}")
                ipa = text
            val_list.append(f"{idx}|{ipa}|{speaker_map[speaker_name]}")

        con.close()
        return train_list, val_list

    # Otherwise fall back to original
    return utils_original.get_data_path_list(train_path, val_path)

# Monkey patch the original module so its internal functions use our overridden version
utils_original.get_data_path_list = get_data_path_list
