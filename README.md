# HRoPE v6 — Hierarchical Rotary Position Embedding

**nnU-Net-style hierarchical encoder for editable LLM inference.** Each sentence's meaning is encoded in isolation; inter-sentence and inter-paragraph relations live in upper-layer transformers as streams of summary tokens.

## Why

Standard LLMs entangle token meaning with document position, so editing one sentence in a 100k-token doc forces O(N²) recomputation. HRoPE v6 makes the representation of any sentence depend only on its own tokens, and routes document context through a separate sentence-summary stream that's cheap to update.

## Architecture

```
tokens ─► L0 (intra-sentence, RoPE base 10000)
            └─► attn-pool ─► L1 (sentence stream, RoPE base 1000)
                              └─► attn-pool ─► L2 (paragraph stream, RoPE base 100)
                                                 │
                                                 ▼
                       cross-attn broadcast ◄────┘
                              │
                              ▼
                          LM head
```

Three independent RoPE frequency bands eliminate the position-collision bug from earlier versions. Editing sentence *k* recomputes only that sentence at L0, one row of L1, optionally one row of L2, and a sentence-local broadcast — projected ~300× speedup on 100k-token docs.

## Files

| File | Purpose |
|---|---|
| `HRoPE_v6_Technical_Specification.html` | Full spec, errata table, math, training recipe, feasibility |
| `hrope_v6_reference.py` | Self-contained PyTorch reference (~700 lines) + `--smoke` test |
| `hrope_v6_train.py` | 4-stage curriculum: L0-only → +L1 → +L2 → stitch-robustness |

## Quickstart

```bash
pip install torch
python hrope_v6_reference.py --smoke   # verifies sentence-isolation invariant
python hrope_v6_train.py --stage 0 --steps 200 --save ckpt0.pt
python hrope_v6_train.py --stage 1 --resume ckpt0.pt --save ckpt1.pt
```

## Status

Reference compiles, smoke test passes (L0 invariance = 0.0 in fp32), all 4 training stages execute. Recommended next step: 350M ablation vs dense baseline before any larger run.

© 2026 Feng Zhou — Patent Pending.
