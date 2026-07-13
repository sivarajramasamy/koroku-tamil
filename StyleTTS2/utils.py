from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

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

        print("[SQL] Initializing G2P phonemizer for Tamil ('ta')...")
        g2p = EspeakG2P(language="ta")

        # Connect to DuckDB
        con = duckdb.connect()

        # Execute training query
        print(f"[SQL] Executing train query: {train_path}")
        train_rows = con.execute(train_path).fetchall()
        print(f"[SQL] Found {len(train_rows)} training rows. Phonemizing transcripts...")
        train_list = []
        for row in train_rows:
            # We expect the query to return: (text, speaker_id, idx)
            text, speaker_id, idx = row[0], row[1], row[2]
            try:
                ipa, _ = g2p(text)
            except Exception as e:
                print(f"[SQL] G2P failed for text '{text}': {e}")
                ipa = text
            train_list.append(f"{idx}|{ipa}|{speaker_id}")

        # Execute validation query
        print(f"[SQL] Executing val query: {val_path}")
        val_rows = con.execute(val_path).fetchall()
        print(f"[SQL] Found {len(val_rows)} validation rows. Phonemizing transcripts...")
        val_list = []
        for row in val_rows:
            text, speaker_id, idx = row[0], row[1], row[2]
            try:
                ipa, _ = g2p(text)
            except Exception as e:
                print(f"[SQL] G2P failed for text '{text}': {e}")
                ipa = text
            val_list.append(f"{idx}|{ipa}|{speaker_id}")

        con.close()
        return train_list, val_list

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    