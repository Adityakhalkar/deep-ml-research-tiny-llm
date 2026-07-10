# Slot: Config (v1, by moe chabot)
# `def configure_model(cfg) -> cfg` — the model's hyperparameters. Change anything: depth, width, heads, context length (`block_size`, up to 1024), batch size, base learning rate, dropout. The parameter cap, step cap and wall clock are the referee — a wider model trains fewer steps in the same budget, and that tradeoff is yours to explore. `vocab_size` (set by the tokenizer), `max_steps` and `device` are harness-owned and reset after this runs.

def configure_model(cfg):
    """The model's shape and training hyperparameters (vanilla nanoGPT).

    Everything here is a tradeoff against the fixed compute budget: deeper
    or wider means fewer steps before the wall clock; longer context means
    slower steps but more to attend to. The param cap is the hard ceiling.
    """
    cfg.n_layer = 4
    cfg.n_head = 6
    cfg.n_embd = 384
    cfg.block_size = 256      # context length (max 1024)
    cfg.dropout = 0.1
    cfg.batch_size = 64
    cfg.learning_rate = 3e-4
    return cfg

