# Slot: Embeddings (v1, by moe chabot)
# `class Embeddings(nn.Module)` — `__init__(cfg)`, `forward(idx)` where `idx` is a `(B, T)` LongTensor of token ids. Must return a `(B, T, cfg.n_embd)` float tensor. How tokens and positions become vectors is up to you (learned, sinusoidal, factorized…). Keep constant tensors in registered buffers rather than creating them with an explicit device in forward — the model is frozen with torch.jit.trace and must stay device-portable.

class Embeddings(nn.Module):
    """Learned token + positional embeddings.

    Improvements:
    - Checks sequence length instead of silently slicing incorrectly.
    - Supports `pos_offset`, useful for KV-cache generation.
    - Keeps position ids as a non-persistent buffer so checkpoints stay smaller.
    - Uses explicit shapes for easier debugging.
    """

    def __init__(self, cfg):
        super().__init__()

        self.vocab_size = cfg.vocab_size
        self.block_size = cfg.block_size
        self.n_embd = cfg.n_embd

        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        # Non-persistent because this can always be recreated from block_size.
        # It will still move with the module across devices.
        self.register_buffer(
            "pos_ids",
            torch.arange(cfg.block_size, dtype=torch.long),
            persistent=False,
        )

    def forward(self, idx, pos_offset: int = 0):
        """
        idx: LongTensor of shape (B, T)
        pos_offset: starting position, useful during autoregressive generation
        """

        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (B, T), got {tuple(idx.shape)}")

        B, T = idx.shape

        if pos_offset + T > self.block_size:
            raise ValueError(
                f"Sequence length {T} with pos_offset {pos_offset} exceeds "
                f"block_size {self.block_size}"
            )

        tok_emb = self.tok(idx)  # (B, T, C)

        pos_ids = self.pos_ids[pos_offset : pos_offset + T]
        pos_emb = self.pos(pos_ids).unsqueeze(0)  # (1, T, C)

        return self.drop(tok_emb + pos_emb)
