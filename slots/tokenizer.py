# Slot: Tokenizer (v0, by Deep-ML)
# `def build_tokenizer(train_bytes) -> list[(int, int)]` — return BPE-style merges over the byte alphabet. Each pair `(id_a, id_b)` mints a new token id (assigned sequentially from 256) representing the pair's concatenation. A TRUSTED encoder applies your merges (greedy left-to-right, in order) to tokenize both the train text and the hidden eval text — your code never runs near eval data, and the byte-level base makes every tokenizer lossless by construction. The metric is bits per byte, so better compression (shorter sequences → more text per window) directly improves the score. Vocab cap: 256 + len(merges) ≤ budget.vocab_cap. Return [] for pure byte-level.

def build_tokenizer(train_bytes):
    """Byte-level tokenizer: no merges (vocab = the 256 raw bytes).

    Return BPE-style merges [(id_a, id_b), ...] — each pair mints a new
    token id (256, 257, ...) for the pair's concatenation. Learning good
    merges from train_bytes (classic BPE counting, for example) shortens
    sequences, which lets the model read more text per window and directly
    improves bits-per-byte. This is the most under-optimized slot in the
    pipeline.
    """
    return []

