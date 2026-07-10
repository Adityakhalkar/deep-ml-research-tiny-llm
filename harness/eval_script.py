# === Deep-ML Research Harness (trusted eval — no user code) ===
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F

_S = "__RH_SENTINEL__"
_cfg = json.loads("""__RH_EVAL_CFG_JSON__""")


def _fail(reason, **extra):
    print(_S + "INVALID " + json.dumps({"reason": reason, **extra}), flush=True)
    sys.exit(0)


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


# The tokenizer/config table came from the train sandbox — treat it as
# untrusted input and re-validate before use. The ENCODER is this script's
# own code. block_size rides along because the traced graph is
# shape-specialized at the submission's context length.
try:
    with open("/tmp/rh_tokenizer.json") as _f:
        _raw = json.load(_f)
    if isinstance(_raw, dict):
        _rh_merges = _rh_validate_merges([tuple(p) for p in _raw.get("merges", [])], _cfg["vocab_cap"])
        T = int(_raw.get("block_size", _cfg["block_size"]))
    else:  # legacy bare-list form
        _rh_merges = _rh_validate_merges([tuple(p) for p in _raw], _cfg["vocab_cap"])
        T = int(_cfg["block_size"])
    if not (16 <= T <= 1024):
        raise ValueError(f"block_size out of range: {T}")
except Exception as e:
    _fail("tokenizer_table_invalid", detail=str(e)[:500])

try:
    m = torch.jit.load("/tmp/rh_model.pt", map_location="cpu")
    m.eval()
except Exception as e:
    _fail("artifact_load_failed", detail=str(e)[:500])

params = sum(p.numel() for p in m.parameters())
if params > int(_cfg["max_params"]):
    _fail("param_cap_exceeded", params=params, max_params=int(_cfg["max_params"]))

V = 256 + len(_rh_merges)

# Causality check: two rows sharing a prefix, different suffixes. A causal
# model must produce identical logits over the shared prefix. (The traced
# graph is batch-size-specialized at B=2, which is also the eval batch.)
_g = torch.Generator().manual_seed(20260702)
_x1 = torch.randint(0, V, (T,), generator=_g)
_x2 = _x1.clone()
_half = T // 2
_x2[_half:] = torch.randint(0, V, (T - _half,), generator=_g)
_pair = torch.stack([_x1, _x2])
try:
    with torch.no_grad():
        _lg = m(_pair)
except Exception as e:
    _fail("forward_failed", detail=str(e)[:500])
if _lg.shape != (2, T, V):
    _fail("bad_output_shape", shape=list(_lg.shape), expected=[2, T, V])
if not torch.allclose(_lg[0, : _half - 1], _lg[1, : _half - 1], atol=1e-3, rtol=1e-3):
    _fail("causality_violation")

# --- bits per byte on the hidden splits ---
# The model predicts tokens; the score divides total NLL by the RAW BYTES
# those tokens cover, so scores are comparable across tokenizers. Windows
# are non-overlapping and evenly strided (deterministic, same rule for
# every submission).
_lens = _rh_token_byte_lens(_rh_merges)
_n_eval = int(_cfg.get("n_eval_windows", 256))

out = {"params": int(params), "vocab_size": int(V)}
for split in _cfg["splits"]:
    raw = bytes(np.fromfile(f"/research_data/{split}.bin", dtype=np.uint8))
    toks = _rh_encode(raw, _rh_merges)
    n = (len(toks) - 1) // T
    n -= n % 2  # traced graph is specialized at batch=2
    if n <= 0:
        _fail("split_too_small", split=split)
    stride = max(1, n // _n_eval)
    rows = list(range(0, n, stride))[:_n_eval]
    if len(rows) % 2 == 1:
        rows = rows[:-1]
    if not rows:
        _fail("split_too_small", split=split)
    xs = torch.from_numpy(np.stack([toks[r * T : r * T + T] for r in rows]))
    ys = torch.from_numpy(np.stack([toks[r * T + 1 : r * T + T + 1] for r in rows]))
    total_nll = 0.0
    total_bytes = 0
    with torch.no_grad():
        for i in range(0, len(rows), 2):
            logits = m(xs[i : i + 2])
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ys[i : i + 2].reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_bytes += int(_lens[ys[i : i + 2].numpy()].sum())
    out[f"{split}_loss"] = round(total_nll / math.log(2) / max(1, total_bytes), 6)

print(_S + "METRIC " + json.dumps(out), flush=True)
