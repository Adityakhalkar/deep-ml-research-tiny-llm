# Slot: Architecture (v1, by moe chabot)
# `def build_model(cfg) -> nn.Module` — assembles the WHOLE model: `forward(idx)` takes `(B, T)` token ids and returns `(B, T, cfg.vocab_size)` logits. The default wires the other slots into a pre-norm transformer; override it to change the topology itself — parallel attention+FFN blocks, post-norm, weight tying, different residual patterns, early exit. You may call the other slot classes (Embeddings/Attention/FFN/Norm) or ignore them and build your own. Mutating `cfg` (e.g. `cfg.n_layer = 8`) is legal — the parameter cap and compute budget are the referee, not the config. Causality is verified on the hidden eval; the model must stay causal and trace-able (torch.jit.trace, so keep constant tensors in registered buffers).

class Block(nn.Module):
    """Pre-norm transformer block (vanilla nanoGPT wiring)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = Norm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = Norm(cfg.n_embd)
        self.ffn = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class GPT(nn.Module):
    """Embeddings -> N blocks -> final norm -> untied lm_head."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = Norm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embeddings(idx)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm_f(x))


def build_model(cfg):
    """The topology is yours: rewire blocks, tie weights, go parallel."""
    return GPT(cfg)

