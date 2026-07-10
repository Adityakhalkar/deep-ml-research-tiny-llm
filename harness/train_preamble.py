# === Deep-ML Research Harness (trusted preamble) ===
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

_rh_budget = json.loads("""__RH_BUDGET_JSON__""")
_rh_model_cfg = json.loads("""__RH_MODEL_CFG_JSON__""")
_rh_smoke = bool(__RH_SMOKE__)

torch.manual_seed(__RH_TRAIN_SEED__)
np.random.seed(__RH_TRAIN_SEED__)

_rh_device = "cuda" if torch.cuda.is_available() else "cpu"

# Hidden TRAIN split only (raw UTF-8 bytes) — val/test are not in this sandbox.
# Exposed to the tokenizer slot as `train_bytes`.
train_bytes = bytes(np.fromfile("/research_data/train.bin", dtype=np.uint8))
if _rh_smoke:
    # Fast test runs tokenize a slice so feedback stays snappy.
    train_bytes = train_bytes[:2_000_000]

cfg = SimpleNamespace(
    n_layer=_rh_model_cfg["n_layer"],
    n_head=_rh_model_cfg["n_head"],
    n_embd=_rh_model_cfg["n_embd"],
    block_size=_rh_model_cfg["block_size"],
    vocab_size=256,  # updated after the tokenizer slot runs: 256 + len(merges)
    dropout=_rh_model_cfg.get("dropout", 0.1),
    batch_size=_rh_model_cfg.get("batch_size", 64),
    learning_rate=_rh_model_cfg.get("learning_rate", 3e-4),
    max_steps=__RH_MAX_STEPS__,
    device=_rh_device,
)

def _rh_validate_merges(merges, vocab_cap):
    if not isinstance(merges, list):
        raise ValueError("merges must be a list of (id, id) pairs")
    if 256 + len(merges) > int(vocab_cap):
        raise ValueError(f"vocab too large: 256+{len(merges)} exceeds cap {vocab_cap}")
    clean = []
    for i, pr in enumerate(merges):
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            raise ValueError(f"merge {i} is not a pair")
        a, b = pr
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError(f"merge {i} ids must be ints")
        hi = 256 + i
        if not (0 <= a < hi and 0 <= b < hi):
            raise ValueError(f"merge {i} references id >= {hi}")
        clean.append((a, b))
    return clean


def _rh_encode(byts, merges):
    """Apply merges in order, greedy left-to-right per merge. Trusted."""
    arr = np.frombuffer(byts, dtype=np.uint8).astype(np.int32)
    for i, (a, b) in enumerate(merges):
        if len(arr) < 2:
            break
        m = (arr[:-1] == a) & (arr[1:] == b)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        keep = []
        last = -2
        for j in idx:
            if j > last + 1:
                keep.append(j)
                last = j
        keep = np.asarray(keep)
        arr[keep] = 256 + i
        arr = np.delete(arr, keep + 1)
    return arr.astype(np.int64)


def _rh_token_byte_lens(merges):
    lens = [1] * 256
    for a, b in merges:
        lens.append(lens[a] + lens[b])
    return np.asarray(lens, dtype=np.int64)

# === slot code follows (canonical + user-modified slots) ===
