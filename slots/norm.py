# Slot: Normalization (v0, by Deep-ML)
# `class Norm(nn.Module)` — `__init__(dim)`, `forward(x)` returns the same shape. Used for both pre-attention/pre-FFN norms and the final norm.

class Norm(nn.Module):
    """Standard LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)

