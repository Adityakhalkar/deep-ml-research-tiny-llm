
# === Deep-ML Research Harness (trusted epilogue) ===

def _rh_err_line():
    """Deepest traceback line inside the stitched script — the service maps
    it back to the slot + line the user actually edited."""
    import traceback
    line = None
    for fr in traceback.extract_tb(sys.exc_info()[2]):
        if fr.filename.endswith("rh_train.py") or fr.filename == "<string>":
            line = fr.lineno
    return line


_rh_required = [
    ("configure_model", "config"), ("build_tokenizer", "tokenizer"),
    ("Embeddings", "embeddings"), ("Attention", "attention"), ("FFN", "ffn"),
    ("Norm", "norm"), ("build_model", "architecture"),
    ("configure_optimizer", "optimizer"),
    ("get_lr", "lr_schedule"), ("train_step", "train_step"),
]
for _rh_sym, _rh_slot in _rh_required:
    if _rh_sym not in globals():
        print(json.dumps({"rh_error": "missing_symbol", "symbol": _rh_sym, "slot": _rh_slot}), flush=True)
        sys.exit(3)

# --- config: the user shapes cfg; the harness re-asserts what it owns ---
try:
    cfg = configure_model(cfg) or cfg
    cfg.block_size = int(cfg.block_size)
    cfg.batch_size = max(1, int(cfg.batch_size))
    if not (16 <= cfg.block_size <= 1024):
        raise ValueError(f"block_size must be within [16, 1024], got {cfg.block_size}")
except Exception as _rh_e:
    print(json.dumps({"rh_error": "config_failed", "detail": str(_rh_e)[:800],
                      "line": _rh_err_line()}), flush=True)
    sys.exit(3)
# Budget and placement are not hyperparameters.
cfg.max_steps = __RH_MAX_STEPS__
cfg.device = _rh_device

# --- tokenizer: user code proposes a merges TABLE; trusted code applies it ---
try:
    _rh_merges = _rh_validate_merges(build_tokenizer(train_bytes), _rh_budget.get("vocab_cap", 2048))
except Exception as _rh_e:
    print(json.dumps({"rh_error": "tokenizer_failed", "detail": str(_rh_e)[:800],
                      "line": _rh_err_line()}), flush=True)
    sys.exit(3)

_rh_tok_t0 = time.monotonic()
_rh_train = torch.from_numpy(_rh_encode(train_bytes, _rh_merges))
cfg.vocab_size = 256 + len(_rh_merges)
if len(_rh_train) < cfg.block_size + 2:
    print(json.dumps({"rh_error": "tokenizer_failed",
                      "detail": "tokenized train stream shorter than one window"}), flush=True)
    sys.exit(3)
print(json.dumps({"rh_info": "tokenized", "vocab_size": cfg.vocab_size,
                  "train_tokens": int(len(_rh_train)),
                  "tokenize_seconds": round(time.monotonic() - _rh_tok_t0, 1)}), flush=True)

# Merges + block_size travel to the eval sandbox together: the traced graph
# is shape-specialized, so eval must window the hidden data at THIS length.
with open("/tmp/rh_tokenizer.json", "w") as _rh_f:
    json.dump({"merges": [[int(a), int(b)] for a, b in _rh_merges],
               "block_size": int(cfg.block_size)}, _rh_f)

_rh_gen = torch.Generator().manual_seed(__RH_TRAIN_SEED__)

def _rh_get_batch():
    ix = torch.randint(len(_rh_train) - cfg.block_size - 1, (cfg.batch_size,), generator=_rh_gen)
    x = torch.stack([_rh_train[i : i + cfg.block_size] for i in ix])
    y = torch.stack([_rh_train[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# The architecture slot owns the topology; the harness just calls it.
try:
    _rh_model = build_model(cfg)
    if not isinstance(_rh_model, nn.Module):
        raise TypeError("build_model(cfg) must return a torch.nn.Module")
    _rh_model = _rh_model.to(cfg.device)
except Exception as _rh_e:
    print(json.dumps({"rh_error": "build_failed", "detail": str(_rh_e)[:800],
                      "line": _rh_err_line()}), flush=True)
    sys.exit(3)

_rh_params = sum(p.numel() for p in _rh_model.parameters())
if _rh_params > int(_rh_budget["max_params"]):
    # Fast feedback only — the eval phase re-enforces this cap in trusted code.
    print(json.dumps({"rh_error": "param_cap_exceeded", "params": _rh_params,
                      "max_params": int(_rh_budget["max_params"])}), flush=True)
    sys.exit(3)

try:
    _rh_opt = configure_optimizer(_rh_model, cfg)
except Exception as _rh_e:
    print(json.dumps({"rh_error": "optimizer_failed", "detail": str(_rh_e)[:800],
                      "line": _rh_err_line()}), flush=True)
    sys.exit(3)

# --- the training loop: step cap + wall-clock guard, both trusted ---
_rh_model.train()
_rh_t0 = time.monotonic()
_rh_steps_run = 0
_rh_ema = None
_rh_log_every = __RH_LOG_EVERY__

for _rh_step in range(cfg.max_steps):
    if time.monotonic() - _rh_t0 > float(_rh_budget["wall_clock_seconds"]):
        break
    try:
        _rh_lr = float(get_lr(_rh_step, cfg))
        for _rh_g in _rh_opt.param_groups:
            _rh_g["lr"] = _rh_lr
        _rh_loss = float(train_step(_rh_model, _rh_get_batch(), _rh_opt, _rh_step))
    except Exception as _rh_e:
        print(json.dumps({"rh_error": "train_step_failed", "step": _rh_step,
                          "detail": str(_rh_e)[:800], "line": _rh_err_line()}), flush=True)
        sys.exit(3)
    _rh_ema = _rh_loss if _rh_ema is None else 0.9 * _rh_ema + 0.1 * _rh_loss
    _rh_steps_run += 1
    if _rh_step % _rh_log_every == 0 or _rh_step == cfg.max_steps - 1:
        print(json.dumps({"step": _rh_step, "train_loss": round(_rh_ema, 4),
                          "lr": round(_rh_lr, 6)}), flush=True)

_rh_wall = round(time.monotonic() - _rh_t0, 2)

# --- freeze to a self-contained TorchScript artifact for the trusted eval ---
_rh_model = _rh_model.to("cpu")
_rh_model.eval()
_rh_example = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
try:
    with torch.no_grad():
        _rh_traced = torch.jit.trace(_rh_model, _rh_example)
    _rh_traced.save("/tmp/rh_model.pt")
except Exception as _rh_e:
    print(json.dumps({"rh_error": "trace_failed", "detail": str(_rh_e)[:800]}), flush=True)
    sys.exit(3)

print("RH_TRAIN_SUMMARY " + json.dumps({
    "steps_run": _rh_steps_run,
    "wall_clock_seconds": _rh_wall,
    "params": _rh_params,
    "vocab_size": cfg.vocab_size,
    "final_train_loss": round(_rh_ema, 4) if _rh_ema is not None else None,
}), flush=True)
