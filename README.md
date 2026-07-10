# Tiny-LLM: the crowd-trained language model

A tiny language model trained by a crowd on [Deep-ML Research](https://deep-ml.com/research/tiny-llm).

The model is defined entirely by **code slots**. Anyone can fork a slot, improve
it, and — if their candidate beats the canonical model on a hidden eval — their
code is merged automatically. This repo is a snapshot of the current canonical
model.

**Generation:** 1
**bits per byte:** 2.8500 (hidden test set, lower is better)

## Slots

| Slot | Version | Author |
|---|---|---|
| Config | v1 | moe chabot |
| Tokenizer | v0 | Deep-ML |
| Embeddings | v1 | moe chabot |
| Attention | v0 | Deep-ML |
| Feed-forward | v0 | Deep-ML |
| Normalization | v0 | Deep-ML |
| Architecture | v1 | moe chabot |
| Optimizer | v0 | Deep-ML |
| LR schedule | v0 | Deep-ML |
| Train step | v0 | Deep-ML |

## Layout

- `slots/` — the canonical code for every slot (this IS the model)
- `harness/` — the trusted harness that assembles the slots, owns the training
  loop/budget, and computes the metric on hidden data (published for auditability;
  `__RH_*__` placeholders are filled server-side per run)
- `RESEARCH_LOG.md` — every merged experiment, in order

Want your name in the table? Fork a slot at
[deep-ml.com/research/tiny-llm](https://deep-ml.com/research/tiny-llm).
