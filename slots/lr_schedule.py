# Slot: LR schedule (v0, by Deep-ML)
# `def get_lr(step, cfg) -> float`. Called every step by the harness; the returned value is written into every param group before `train_step` runs. `cfg.max_steps` and `cfg.learning_rate` are available.

def get_lr(step, cfg):
    """Linear warmup then cosine decay to 10% of base lr."""
    warmup = 100
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

