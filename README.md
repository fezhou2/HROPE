# HRoPE — Hierarchical Rotary Position Embedding

**A symmetric U-Net architecture for editable LLM inference.** Each sentence is encoded in isolation; document context flows through a separate hierarchy of summary tokens with additive skip connections — the nnU-Net pattern applied to text.

## Why

Standard LLMs entangle token meaning with document position, so editing one sentence in a 100k-token doc forces O(N²) recomputation. HRoPE makes a sentence's representation depend only on its own tokens; document context reaches the LM head through a separate, cheap-to-update summary stream.

## Architecture

```
ENCODE                          DECODE
embed
  ↓
L0_enc ──── skip ──────────►  L0_dec → LM head
  ↓ pool                        ↑ unpool + skip
L1_enc ──── skip ──────────►  L1_dec
  ↓ pool                        ↑ unpool + skip
L2_enc ──── bottleneck ───►   L2_dec
```

- **Three RoPE bands** (10k / 1k / 100), one per level — no cross-level collisions
- **Symmetric encoder/decoder** at every level, same transformer block throughout
- **Additive skips** preserve high-frequency token detail at the LM head
- **Parameter-free `unpool`** (indexed gather) for level-to-level upsampling

## Verified invariants (fp32, smoke test)

| Property | Measured |
|---|---|
| L0 encoder invariance under L1/L2 reordering | `0.0` ✓ |
| Decoder output varies under reordering | `> 0` ✓ |
| Zeroing skip changes decoder output | `> 0` ✓ |
| Non-edited sentences' skip unchanged after edit | `0.0` ✓ |

## Edit cost

Single-sentence edit in a 100k-token doc: **~10–25 ms** vs ~5,000 ms full recompute (~300× speedup); L1 self-attn over the ~5k-sentence summary stream dominates.

## Files

| File | Purpose |
|---|---|
| `HRoPE_v7_Technical_Specification.html` | Full architectural spec |
| `hrope_v7_reference.py` | Self-contained PyTorch reference + `--smoke` |
| `hrope_v7_train.py` | 4-stage curriculum (L0-only → +L1 → +L2 → stitch-robust) |

## Quickstart

```bash
pip install torch
python hrope_v7_reference.py --smoke
python hrope_v7_train.py --stage 0 --steps 200 --save ckpt0.pt
python hrope_v7_train.py --stage 1 --resume ckpt0.pt --save ckpt1.pt
```

## Status

Reference compiles, all four invariants verified, all training stages run end-to-end. Next step: 350M-parameter ablation vs a dense baseline before scaling to 1.5B.

© 2026 Feng Zhou — All Rights Reserved.
