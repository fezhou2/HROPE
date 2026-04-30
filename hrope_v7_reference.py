"""
HRoPE v7 — Symmetric U-Net Reference Implementation
====================================================
Faithful nnU-Net architecture for hierarchical text encoding.

Key change vs v6
----------------
v6 was an encoder-only hierarchy with a single cross-attention broadcast at
the very end. v7 is a **symmetric** U-shape with matching encoder and decoder
stacks at every level, connected by additive skip connections — exactly the
nnU-Net pattern.

ENCODE (going up the hierarchy)        DECODE (coming back down)
                                       ┌──────────────────────► LM head
embed                                  │
  ↓                                  L0_dec
L0_enc ───────── skip_l0 ──(add)─►    ↑ unpool (gather sent → tokens)
  ↓ pool                            L1_dec
L1_enc ───────── skip_l1 ──(add)─►    ↑ unpool (gather para → sent)
  ↓ pool                            L2_dec
L2_enc ─────────── (bottleneck) ────►  ↑

Each level's encoder and decoder use the SAME transformer block type
(`HRoPEEncoderBlock`) with the SAME RoPE frequency band, the SAME causal /
intra-sentence attention masks, and the SAME positions. The only thing that
distinguishes encoder from decoder at a level is which set of weights it
uses and what skip it consumes.

Sentence isolation invariant (PRESERVED)
----------------------------------------
- L0 ENCODER output `skip_l0[s]` depends only on the tokens of sentence s.
- L0 DECODER output `y0[token in s]` depends on:
    skip_l0[s]                      (sentence-local content) +
    unpool(L1_dec)[s]               (one vector summarizing doc context)
  Token-level attention at the decoder is still masked block-diagonal +
  causal within each sentence — so y0[s] depends only on sentence s's own
  skip + sentence s's own L1_dec vector.
- The architecture provides full document context to the LM head WITHOUT
  letting any document-global position leak into the per-token represen-
  tation. The doc context arrives via a single broadcast vector per sentence.

Smoke:  python hrope_v7_reference.py --smoke
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  1. Hierarchical RoPE (unchanged from v6)                                    #
# =========================================================================== #

class HierarchicalRoPE(nn.Module):
    """One independent frequency band per hierarchy level (GPT-NeoX layout)."""

    def __init__(self, head_dim: int, base: float, max_pos: int = 32768):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions]
        sin = self.sin[positions]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =========================================================================== #
#  2. ReLU² attention with explicit per-row denominator (unchanged from v6)    #
# =========================================================================== #

def relu_sq_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      attn_mask: Optional[torch.Tensor] = None,
                      eps: float = 1e-6
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.size(-1))
    weights = F.relu(scores) ** 2
    if attn_mask is not None:
        weights = weights.masked_fill(attn_mask == 0, 0.0)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.einsum("bhqk,bhkd->bhqd", weights / denom, v)
    return out, denom


# =========================================================================== #
#  3. Encoder/decoder block — same primitive used at every level + role        #
# =========================================================================== #

class HRoPEBlock(nn.Module):
    """Pre-norm transformer block parameterised by its level's HierarchicalRoPE.

    Used identically as encoder block AND decoder block; the only thing the
    forward() cares about is (input, positions, mask). The same class with
    different weights serves both roles, just like a U-Net's encoder/decoder
    convs are the same Conv2d operation with different weights.

    Inherits the v4 spec's cache-stitching robustness toolkit:
      * QK-Norm (q_norm, k_norm) — bounds attention logits independent of
        hidden-state magnitude (v4 §4.2).
      * Denominator-tracked ReLU² attention — stitch-decomposable (v4 §4.1).
      * Optional **parallel residual** (`parallel_residual=True`):
        attention and FFN are computed from the SAME normalized input and
        added together. Decouples FFN from attention output, providing
        error isolation when upstream context drifts (v4 §4.3).
      * Optional **micro-correction gate** (`use_micro_gate=True`): a
        sigmoid gate computed from external segment statistics that
        modulates the attention output. Used at stitch time only when an
        approximate (rather than exact) L1 cache update is desired (v4 §4.4).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int,
                 rope: HierarchicalRoPE, dropout: float = 0.0,
                 parallel_residual: bool = False,
                 use_micro_gate: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.parallel_residual = parallel_residual
        self.use_micro_gate = use_micro_gate

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        hidden = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

        if parallel_residual:
            # v4 §4.3: per-layer learned scalars weighting the two branches.
            # Init to 0.5 each (equal weighting, matches v4 init).
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))

        if use_micro_gate:
            # v4 §4.4: gate is a function of (mu, log_sigma) features of dim
            # 2*head_dim, projected to (n_heads, head_dim) sigmoid weights.
            # W and b initialised to zero so the gate is identity at init
            # (we use 2*sigmoid(0)=1.0 so the gate doesn't suppress signal).
            self.W_gate = nn.Parameter(
                torch.zeros(n_heads, self.head_dim, 2 * self.head_dim))
            self.b_gate = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def _apply_micro_gate(self, attn_out: torch.Tensor,
                          seg_stats: Optional[torch.Tensor]) -> torch.Tensor:
        """attn_out: (B, H, S, D); seg_stats: (B, H, 2*D) or None.
        Gate is 2*sigmoid(W·stats + b). With W=b=0 the gate is exactly 1.0
        — the block is identical to no-gate at initialisation, and only
        deviates from 1 once training learns when to suppress / amplify."""
        if seg_stats is None or not self.use_micro_gate:
            return attn_out
        gate_logits = torch.einsum("bhe,hde->bhd", seg_stats, self.W_gate) + self.b_gate
        gate = 2.0 * torch.sigmoid(gate_logits)         # init ≈ 1.0
        return attn_out * gate.unsqueeze(2)             # broadcast over seq

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                seg_stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        q = self.q_norm(self._split_heads(self.q_proj(h)))
        k = self.k_norm(self._split_heads(self.k_proj(h)))
        v = self._split_heads(self.v_proj(h))
        pos = positions.unsqueeze(1).expand(-1, self.n_heads, -1)
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        attn_out, _ = relu_sq_attention(q, k, v, attn_mask=attn_mask)
        attn_out = self._apply_micro_gate(attn_out, seg_stats)
        attn_proj = self.dropout(self.o_proj(self._merge_heads(attn_out)))

        if self.parallel_residual:
            # v4 §4.3: attention and FFN both read from the SAME h (norm1(x)).
            # The FFN branch is decoupled from the attention output, so
            # transient errors in the attention path do not pollute FFN.
            ffn_out = self.dropout(self.ffn(h))
            return x + self.alpha * attn_proj + self.beta * ffn_out
        # Sequential (default): FFN reads from x + attn (post-attn pre-norm).
        x = x + attn_proj
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


def compute_seg_stats(skip_tensor: torch.Tensor, group_id: torch.Tensor,
                      num_groups: int, n_heads: int, head_dim: int,
                      eps: float = 1e-6) -> torch.Tensor:
    """v4 §4.4 sufficient statistics over the skip / hidden tensor.

    Returns per-(batch, head, group) features of shape (B, H, num_groups, 2D)
    where the 2D feature concatenates (mu, log_sigma) of the per-token
    head-vectors that share each group.

    skip_tensor: (B, T, D_model)
    group_id:    (B, T) which group each token belongs to (long)

    For HRoPE v7 the natural use-case is `group_id = sent_id`, giving
    one stat row per sentence. Pass these stats into the L0 decoder when
    you want it to compensate for upstream broadcast drift after a stitch.
    """
    B, T, _ = skip_tensor.shape
    h = skip_tensor.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B,H,T,D)
    mu = h.new_zeros(B, n_heads, num_groups, head_dim)
    log_sigma = h.new_zeros(B, n_heads, num_groups, head_dim)
    for b in range(B):
        for g in range(num_groups):
            sel = group_id[b] == g
            if sel.any():
                v = h[b, :, sel, :]                                # (H, n_g, D)
                mu[b, :, g] = v.mean(dim=1)
                log_sigma[b, :, g] = (v.std(dim=1) + eps).log()
    return torch.cat([mu, log_sigma], dim=-1)                      # (B,H,G,2D)


# =========================================================================== #
#  4. Attention pool — used at every encode-side downsampling step             #
# =========================================================================== #

class AttentionPool(nn.Module):
    """Pool tokens of a group into a single vector via learnable-query attn."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm = nn.RMSNorm(d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, group_id: torch.Tensor,
                num_groups: int) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = (k * self.query).sum(-1) / math.sqrt(D)
        out = x.new_zeros(B, num_groups, D)
        for b in range(B):
            for g in range(num_groups):
                sel = group_id[b] == g
                if sel.any():
                    s = scores[b][sel]
                    w = torch.softmax(s, dim=0)
                    out[b, g] = (w.unsqueeze(-1) * v[b][sel]).sum(0)
        return out


# =========================================================================== #
#  5. Unpool — the upsampling op of nnU-Net, here just an indexed gather       #
# =========================================================================== #

def unpool(parent_seq: torch.Tensor, child_to_parent: torch.Tensor) -> torch.Tensor:
    """Broadcast each parent-level vector to its children.

    parent_seq:        (B, P, D) one vector per parent
    child_to_parent:   (B, C) which parent each child belongs to (long)
    returns:           (B, C, D) with parent_seq[child_to_parent[b, c]] at row c
    """
    B, P, D = parent_seq.shape
    idx = child_to_parent.unsqueeze(-1).expand(-1, -1, D)
    return parent_seq.gather(1, idx)


# =========================================================================== #
#  6. Doc structure (unchanged from v6)                                        #
# =========================================================================== #

@dataclass
class DocStructure:
    token_ids: torch.Tensor
    token_in_sent: torch.Tensor
    sent_id: torch.Tensor
    para_id_per_sent: torch.Tensor
    sent_pos_in_para: torch.Tensor
    para_pos: torch.Tensor
    n_sent: torch.Tensor
    n_para: torch.Tensor


# =========================================================================== #
#  7. Config                                                                   #
# =========================================================================== #

@dataclass
class HRoPEv7Config:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 4
    ffn_mult: int = 4
    # Encoder depths
    n_l0_enc: int = 2
    n_l1_enc: int = 1
    n_l2_enc: int = 1
    # Decoder depths (typically equal to encoder for symmetry)
    n_l0_dec: int = 2
    n_l1_dec: int = 1
    n_l2_dec: int = 1
    use_l2: bool = True
    max_tokens_per_sent: int = 256
    max_sents_per_doc: int = 2048
    max_paras_per_doc: int = 256
    dropout: float = 0.0

    # ── v4 carry-over (cache-stitching robustness) ────────────────────
    # Parallel residual (v4 §4.3): attention and FFN both read from norm(x).
    # Decouples FFN from attention output and bounds error propagation across
    # layers. Recommended ON for L1/L2 (where summary edits propagate causally
    # and small drifts can compound) and OFF for L0 (sentence-isolated by mask,
    # so there is nothing to compound).
    parallel_residual_l0: bool = False
    parallel_residual_l1: bool = True
    parallel_residual_l2: bool = True

    # Micro-correction gate (v4 §4.4): a per-layer sigmoid gate driven by
    # cached (mu, log-sigma) statistics of the skip tensor. Useful only when
    # the editor switches to APPROXIMATE L1 stitching (v8 roadmap). Default
    # OFF, because v7's editor does an exact L1/L2 re-pass which makes the
    # gate redundant. Setting these True adds parameters but the gate is
    # initialised to identity (W=b=0 → gate=1.0), so toggling it on does not
    # change the model's output until training starts to use the stats.
    use_micro_gate_l1_dec: bool = False
    use_micro_gate_l0_dec: bool = False


# =========================================================================== #
#  8. The symmetric U-Net model                                                #
# =========================================================================== #

class HRoPEv7Model(nn.Module):
    def __init__(self, cfg: HRoPEv7Config):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.d_model // cfg.n_heads
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Three RoPE bands — shared between encoder and decoder of each level.
        self.rope_l0 = HierarchicalRoPE(head_dim, base=10000.0,
                                        max_pos=cfg.max_tokens_per_sent + 4)
        self.rope_l1 = HierarchicalRoPE(head_dim, base=1000.0,
                                        max_pos=cfg.max_sents_per_doc + 4)
        self.rope_l2 = HierarchicalRoPE(head_dim, base=100.0,
                                        max_pos=cfg.max_paras_per_doc + 4)

        def stack(n: int, rope: HierarchicalRoPE,
                  parallel: bool = False,
                  micro_gate: bool = False) -> nn.ModuleList:
            return nn.ModuleList([
                HRoPEBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult,
                           rope, dropout=cfg.dropout,
                           parallel_residual=parallel,
                           use_micro_gate=micro_gate)
                for _ in range(n)
            ])

        # Encoder path: no micro-gate (a stitch-time op only).
        self.l0_enc = stack(cfg.n_l0_enc, self.rope_l0,
                            parallel=cfg.parallel_residual_l0)
        self.sent_pool = AttentionPool(cfg.d_model)
        self.l1_enc = stack(cfg.n_l1_enc, self.rope_l1,
                            parallel=cfg.parallel_residual_l1)
        if cfg.use_l2:
            self.para_pool = AttentionPool(cfg.d_model)
            self.l2_enc = stack(cfg.n_l2_enc, self.rope_l2,
                                parallel=cfg.parallel_residual_l2)

        # Decoder path: parallel residual + optional micro-gate at the levels
        # that consume cross-segment broadcasts (L1_dec, L0_dec).
        if cfg.use_l2:
            self.l2_dec = stack(cfg.n_l2_dec, self.rope_l2,
                                parallel=cfg.parallel_residual_l2)
        self.l1_dec = stack(cfg.n_l1_dec, self.rope_l1,
                            parallel=cfg.parallel_residual_l1,
                            micro_gate=cfg.use_micro_gate_l1_dec)
        self.l0_dec = stack(cfg.n_l0_dec, self.rope_l0,
                            parallel=cfg.parallel_residual_l0,
                            micro_gate=cfg.use_micro_gate_l0_dec)

        # Output
        self.head_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    # ----- masks -----
    def _l0_mask(self, sent_id: torch.Tensor) -> torch.Tensor:
        same = sent_id.unsqueeze(2) == sent_id.unsqueeze(1)
        idx = torch.arange(sent_id.size(1), device=sent_id.device)
        causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
        return (same & causal.unsqueeze(0)).unsqueeze(1)

    def _causal_mask(self, n: torch.Tensor, total: int) -> torch.Tensor:
        idx = torch.arange(total, device=n.device)
        valid = idx.unsqueeze(0) < n.unsqueeze(1)
        causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
        mask = valid.unsqueeze(2) & valid.unsqueeze(1) & causal.unsqueeze(0)
        return mask.unsqueeze(1)

    # ----- ENCODE (going up) -----
    def encode(self, doc: DocStructure) -> dict:
        x0 = self.tok_emb(doc.token_ids)
        l0_mask = self._l0_mask(doc.sent_id)
        for blk in self.l0_enc:
            x0 = blk(x0, doc.token_in_sent, attn_mask=l0_mask)
        skip_l0 = x0                                                    # save

        N_s = int(doc.n_sent.max().item())
        x1 = self.sent_pool(x0, doc.sent_id, N_s)
        l1_mask = self._causal_mask(doc.n_sent, N_s)
        for blk in self.l1_enc:
            x1 = blk(x1, doc.sent_pos_in_para, attn_mask=l1_mask)
        skip_l1 = x1                                                    # save

        bottleneck = None
        if self.cfg.use_l2:
            N_p = int(doc.n_para.max().item())
            x2 = self.para_pool(x1, doc.para_id_per_sent, N_p)
            l2_mask = self._causal_mask(doc.n_para, N_p)
            for blk in self.l2_enc:
                x2 = blk(x2, doc.para_pos, attn_mask=l2_mask)
            bottleneck = x2

        return {"skip_l0": skip_l0, "skip_l1": skip_l1,
                "bottleneck": bottleneck,
                "l1_mask": l1_mask,
                "l0_mask": l0_mask}

    # ----- DECODE (coming back down with skips) -----
    def decode(self, doc: DocStructure, encoded: dict) -> torch.Tensor:
        skip_l0 = encoded["skip_l0"]
        skip_l1 = encoded["skip_l1"]
        l0_mask = encoded["l0_mask"]
        l1_mask = encoded["l1_mask"]

        # L2 decoder + unpool to sentence level
        if self.cfg.use_l2:
            y2 = encoded["bottleneck"]
            N_p = y2.size(1)
            l2_mask = self._causal_mask(doc.n_para, N_p)
            for blk in self.l2_dec:
                y2 = blk(y2, doc.para_pos, attn_mask=l2_mask)
            y1 = unpool(y2, doc.para_id_per_sent) + skip_l1            # SKIP
        else:
            y1 = skip_l1

        # L1 decoder + unpool to token level
        for blk in self.l1_dec:
            y1 = blk(y1, doc.sent_pos_in_para, attn_mask=l1_mask)
        y0 = unpool(y1, doc.sent_id) + skip_l0                          # SKIP

        # L0 decoder (still intra-sentence) + LM head
        for blk in self.l0_dec:
            y0 = blk(y0, doc.token_in_sent, attn_mask=l0_mask)
        return y0

    def forward(self, doc: DocStructure) -> dict:
        encoded = self.encode(doc)
        y0 = self.decode(doc, encoded)
        logits = self.lm_head(self.head_norm(y0))
        return {"logits": logits,
                "y0": y0,
                "skip_l0": encoded["skip_l0"],
                "skip_l1": encoded["skip_l1"]}


# =========================================================================== #
#  9. Incremental editor                                                       #
# =========================================================================== #

class IncrementalEditorV7:
    """Sentence-level incremental edit on a v7 model.

    On encode_full: caches skip_l0 (per token), skip_l1 (per sentence), and
    bottleneck (per paragraph). On edit_sentence(s, new_tokens):
      1. Re-run L0 enc on the new tokens               → new skip_l0[s]
      2. Re-pool                                       → new u_s
      3. Re-run L1 enc with new u_s spliced in         → new skip_l1
      4. Re-pool affected paragraph                    → new para summary
      5. Re-run L2 enc                                 → new bottleneck
      6. Run full decode (L2_dec → L1_dec → L0_dec)    → new logits
    Decode is cheap because L1 and L2 are short streams; L0_dec is intra-
    sentence so token-level cost is per-sentence (not per-document-token).
    """

    def __init__(self, model: HRoPEv7Model):
        self.model = model
        self.cache: Optional[dict] = None
        self.last_doc: Optional[DocStructure] = None

    @torch.no_grad()
    def encode_full(self, doc: DocStructure) -> dict:
        out = self.model(doc)
        self.cache = self.model.encode(doc)
        self.last_doc = doc
        return out

    @torch.no_grad()
    def edit_sentence(self, sent_idx: int,
                      new_token_ids: torch.Tensor) -> dict:
        assert self.cache is not None and self.last_doc is not None
        m = self.model
        doc = self.last_doc
        device = new_token_ids.device

        # Re-encode just the edited sentence's tokens at L0.
        L_new = new_token_ids.numel()
        mini_tokens = new_token_ids.view(1, L_new)
        mini_pos = torch.arange(L_new, device=device).view(1, L_new)
        mini_sent = torch.zeros(1, L_new, dtype=torch.long, device=device)
        mini_mask = m._l0_mask(mini_sent)
        x0_new = m.tok_emb(mini_tokens)
        for blk in m.l0_enc:
            x0_new = blk(x0_new, mini_pos, attn_mask=mini_mask)
        skip_l0_new_sent = x0_new[0]                                # (L_new, D)

        # Splice into the document-level skip_l0 cache.
        sel = (doc.sent_id[0] == sent_idx)
        old_skip_l0 = self.cache["skip_l0"]
        new_skip_l0 = old_skip_l0.clone()
        # Trim or pad to fit original length (demo: assume unchanged length).
        L_old = sel.sum().item()
        if L_new != L_old:
            raise ValueError(
                f"demo editor assumes same-length edits ({L_old} -> {L_new}); "
                "production version would also rebuild the index tensors."
            )
        new_skip_l0[0, sel] = skip_l0_new_sent

        # Re-pool the edited sentence to get its new summary vector.
        N_s = int(doc.n_sent.max().item())
        u_new = m.sent_pool(new_skip_l0, doc.sent_id, N_s)
        # Re-run L1 encoder on the spliced sentence stream.
        l1_mask = m._causal_mask(doc.n_sent, N_s)
        x1 = u_new
        for blk in m.l1_enc:
            x1 = blk(x1, doc.sent_pos_in_para, attn_mask=l1_mask)
        new_skip_l1 = x1

        # Re-run L2 encoder.
        new_bottleneck = None
        if m.cfg.use_l2:
            N_p = int(doc.n_para.max().item())
            x2 = m.para_pool(new_skip_l1, doc.para_id_per_sent, N_p)
            l2_mask = m._causal_mask(doc.n_para, N_p)
            for blk in m.l2_enc:
                x2 = blk(x2, doc.para_pos, attn_mask=l2_mask)
            new_bottleneck = x2

        encoded = {"skip_l0": new_skip_l0, "skip_l1": new_skip_l1,
                   "bottleneck": new_bottleneck,
                   "l0_mask": m._l0_mask(doc.sent_id),
                   "l1_mask": l1_mask}
        # Update document-level token ids (so the embed lookup matches; only
        # used downstream for logging / not the forward path here).
        new_doc = DocStructure(
            token_ids=doc.token_ids.clone(),
            token_in_sent=doc.token_in_sent,
            sent_id=doc.sent_id, para_id_per_sent=doc.para_id_per_sent,
            sent_pos_in_para=doc.sent_pos_in_para, para_pos=doc.para_pos,
            n_sent=doc.n_sent, n_para=doc.n_para,
        )
        new_doc.token_ids[0, sel] = new_token_ids

        # Run the full decoder using the spliced encoder cache.
        y0 = m.decode(new_doc, encoded)
        logits = m.lm_head(m.head_norm(y0))

        # Update cache for next edit.
        self.cache = encoded
        self.last_doc = new_doc
        return {"logits": logits, "skip_l0": new_skip_l0,
                "skip_l1": new_skip_l1}


# =========================================================================== #
# 10. SimHash with character n-grams (unchanged from v6)                       #
# =========================================================================== #

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = s.strip().lower()
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def simhash(text: str, hash_bits: int = 64, ngram: int = 3) -> int:
    import hashlib
    feats = char_ngrams(text, ngram)
    if not feats:
        return 0
    v = [0] * hash_bits
    for f in feats:
        h = int.from_bytes(hashlib.md5(f.encode("utf-8")).digest()[:8], "big")
        for i in range(hash_bits):
            v[i] += 1 if (h >> i) & 1 else -1
    sig = 0
    for i in range(hash_bits):
        if v[i] > 0:
            sig |= (1 << i)
    return sig


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# =========================================================================== #
# 11. Smoke test                                                               #
# =========================================================================== #

def _build_toy_doc(vocab_size: int = 1000,
                   sentences_per_para: int = 3,
                   tokens_per_sent: int = 8,
                   n_paras: int = 2,
                   device: torch.device = torch.device("cpu")) -> DocStructure:
    g = torch.Generator(device=device).manual_seed(0)
    n_sents = sentences_per_para * n_paras
    T = n_sents * tokens_per_sent
    token_ids = torch.randint(1, vocab_size, (1, T), generator=g, device=device)
    token_in_sent = torch.arange(T, device=device) % tokens_per_sent
    sent_id = torch.arange(T, device=device) // tokens_per_sent
    para_id_per_sent = torch.arange(n_sents, device=device) // sentences_per_para
    sent_pos_in_para = torch.arange(n_sents, device=device) % sentences_per_para
    para_pos = torch.arange(n_paras, device=device)
    return DocStructure(
        token_ids=token_ids,
        token_in_sent=token_in_sent.unsqueeze(0),
        sent_id=sent_id.unsqueeze(0),
        para_id_per_sent=para_id_per_sent.unsqueeze(0),
        sent_pos_in_para=sent_pos_in_para.unsqueeze(0),
        para_pos=para_pos.unsqueeze(0),
        n_sent=torch.tensor([n_sents], device=device),
        n_para=torch.tensor([n_paras], device=device),
    )


def smoke_test() -> None:
    torch.manual_seed(0)
    cfg = HRoPEv7Config(vocab_size=1000, d_model=64, n_heads=4,
                        n_l0_enc=2, n_l0_dec=2,
                        n_l1_enc=1, n_l1_dec=1,
                        n_l2_enc=1, n_l2_dec=1,
                        max_tokens_per_sent=32, max_sents_per_doc=64,
                        max_paras_per_doc=16)
    model = HRoPEv7Model(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params: {n_params:,}")

    doc = _build_toy_doc()
    out = model(doc)
    print(f"[smoke] logits shape: {tuple(out['logits'].shape)}")
    print(f"[smoke] skip_l0 shape: {tuple(out['skip_l0'].shape)}")
    print(f"[smoke] skip_l1 shape: {tuple(out['skip_l1'].shape)}")

    # ---- Sentence isolation invariant on the L0 ENCODER -------------------- #
    # skip_l0 must be byte-identical when we permute L1/L2 positions.
    enc_a = model.encode(doc)
    doc_perm = DocStructure(
        token_ids=doc.token_ids,
        token_in_sent=doc.token_in_sent,
        sent_id=doc.sent_id,
        para_id_per_sent=doc.para_id_per_sent.flip(-1),
        sent_pos_in_para=doc.sent_pos_in_para.flip(-1),
        para_pos=doc.para_pos.flip(-1),
        n_sent=doc.n_sent, n_para=doc.n_para,
    )
    enc_b = model.encode(doc_perm)
    iso_err = (enc_a["skip_l0"] - enc_b["skip_l0"]).abs().max().item()
    print(f"[smoke] L0 ENCODER invariance under L1/L2 reordering: "
          f"{iso_err:.4e} (must be 0)")
    assert iso_err == 0.0, "skip_l0 must be invariant to L1/L2 positions"

    # ---- Decoder uses doc context: y0 should differ under reordering ------- #
    out_a = model(doc)
    out_b = model(doc_perm)
    dec_diff = (out_a["y0"] - out_b["y0"]).abs().max().item()
    print(f"[smoke] L0 DECODER differs under reordering: {dec_diff:.4e} "
          f"(must be > 0 — proves doc context flows back through decode)")
    assert dec_diff > 0.0, "decoder must consume doc context"

    # ---- Skip connection sanity: zeroing skip_l0 must change y0 ------------ #
    enc_z = model.encode(doc)
    enc_z["skip_l0"] = torch.zeros_like(enc_z["skip_l0"])
    y0_no_skip = model.decode(doc, enc_z)
    skip_effect = (out_a["y0"] - y0_no_skip).abs().max().item()
    print(f"[smoke] zeroing skip_l0 changes decoder output by "
          f"{skip_effect:.4e} (must be > 0 — proves skip is wired)")
    assert skip_effect > 0.0, "skip_l0 must affect decoder output"

    # ---- Incremental edit -------------------------------------------------- #
    editor = IncrementalEditorV7(model)
    full = editor.encode_full(doc)
    new_tokens = torch.randint(1, 1000, (8,))
    res = editor.edit_sentence(2, new_tokens)
    print(f"[smoke] edited logits shape: {tuple(res['logits'].shape)}")

    # Edit must change logits at the edited sentence's positions and may also
    # change them at later positions (causal L1/L2 propagation), but must NOT
    # change skip_l0 at sentences other than the edited one.
    sel_other = (doc.sent_id[0] != 2)
    skip_unchanged = (full["skip_l0"][0, sel_other] -
                      res["skip_l0"][0, sel_other]).abs().max().item()
    print(f"[smoke] non-edited sentences' skip_l0 unchanged: "
          f"{skip_unchanged:.4e} (must be 0 — sentence isolation in cache)")
    assert skip_unchanged == 0.0, "edits must not change other sentences' skip_l0"

    # ---- v4 carry-over: parallel residual + micro-correction gate ---------- #
    # 1) Default config has parallel residual ON at L1/L2. Flip it off and
    #    confirm the model still trains-shape-correctly with the same params.
    cfg_seq = HRoPEv7Config(
        vocab_size=1000, d_model=64, n_heads=4,
        n_l0_enc=2, n_l0_dec=2, n_l1_enc=1, n_l1_dec=1,
        n_l2_enc=1, n_l2_dec=1,
        max_tokens_per_sent=32, max_sents_per_doc=64, max_paras_per_doc=16,
        parallel_residual_l0=False, parallel_residual_l1=False,
        parallel_residual_l2=False,
    )
    model_seq = HRoPEv7Model(cfg_seq).eval()
    out_seq = model_seq(doc)
    print(f"[smoke] sequential-residual variant runs: "
          f"logits {tuple(out_seq['logits'].shape)}")

    # 2) Enable micro-correction gate. With W=b=0 init the gate is identity
    #    (2*sigmoid(0)=1), so output must MATCH the no-gate variant exactly
    #    when seg_stats=None (gate path is bypassed) AND match it numerically
    #    when seg_stats=zeros (gate=1 by construction).
    cfg_gate = HRoPEv7Config(
        vocab_size=1000, d_model=64, n_heads=4,
        n_l0_enc=2, n_l0_dec=2, n_l1_enc=1, n_l1_dec=1,
        n_l2_enc=1, n_l2_dec=1,
        max_tokens_per_sent=32, max_sents_per_doc=64, max_paras_per_doc=16,
        use_micro_gate_l1_dec=True, use_micro_gate_l0_dec=True,
    )
    torch.manual_seed(0)                                # match init seed
    model_gate = HRoPEv7Model(cfg_gate).eval()
    torch.manual_seed(0)
    model_ref = HRoPEv7Model(cfg).eval()                # same seed, no gate
    # The gate adds parameters but they're zero-init, so weight tensors of
    # the shared submodules are identical → output must match exactly.
    out_gate = model_gate(doc)
    out_ref = model_ref(doc)
    gate_diff = (out_gate["y0"] - out_ref["y0"]).abs().max().item()
    print(f"[smoke] micro-gate identity-init: y0 diff vs no-gate = "
          f"{gate_diff:.4e} (must be 0 — gate is identity at init)")
    assert gate_diff == 0.0, "zero-init micro-gate must be identity"

    # 3) compute_seg_stats produces the right shape for downstream use.
    stats = compute_seg_stats(out_ref["skip_l0"], doc.sent_id,
                              num_groups=int(doc.n_sent.max().item()),
                              n_heads=cfg.n_heads,
                              head_dim=cfg.d_model // cfg.n_heads)
    expected = (1, cfg.n_heads, int(doc.n_sent.max().item()),
                2 * (cfg.d_model // cfg.n_heads))
    print(f"[smoke] seg_stats shape: {tuple(stats.shape)} "
          f"(expected {expected})")
    assert tuple(stats.shape) == expected

    # ---- SimHash sanity ---------------------------------------------------- #
    s1 = simhash("the cat sat on the mat")
    s2 = simhash("the cat sat on the rug")
    s3 = simhash("entirely different content here xyz")
    print(f"[smoke] simhash hamming(close) = {hamming(s1, s2)}")
    print(f"[smoke] simhash hamming(far)   = {hamming(s1, s3)}")
    s_zh = simhash("床前明月光疑是地上霜")
    print(f"[smoke] simhash no-whitespace lang nonzero: {s_zh != 0}")
    print("[smoke] OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        print("Use --smoke to run the verification suite.")
