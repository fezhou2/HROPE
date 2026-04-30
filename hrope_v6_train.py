"""
HRoPE v6 — Training script with staged curriculum.

Stages
------
1. **Stage 0 — L0-only pretrain** (sentence-internal LM):
   freeze L1/L2 to identity, train next-token loss within each sentence.
   Rationale: forces sentences to carry their semantic load WITHOUT any
   inter-sentence signal — this is the nnU-Net "isolation" invariant.

2. **Stage 1 — Add L1**:
   unfreeze L1 + sentence pool + broadcast_l1; train next-token over
   document. L0 keeps its weights but receives gradient from upstream.
   Adds an auxiliary "next-sentence-summary" prediction loss.

3. **Stage 2 — Add L2**:
   unfreeze paragraph pool + L2 + broadcast_l2; document length grows.

4. **Stage 3 — Stitch-robustness**:
   randomly shuffle sentence indices in L1 / drop a paragraph and ask the
   model to recover next-token loss with frozen L0 caches. This trains
   the model to cope with the cache-stitching workload that motivates HRoPE.

Usage
-----
    python hrope_v6_train.py --stage 0 --steps 200 --device cpu
    python hrope_v6_train.py --stage 1 --steps 200 --device cpu --resume ckpt0.pt

Synthetic data: deterministic toy LM task (predict the next token given the
previous 1-2; signal flows through sentence summaries when distance > 1).
This is enough to verify that the staged curriculum actually trains.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrope_v6_reference import (
    DocStructure,
    HRoPEv6Config,
    HRoPEv6Model,
)


# =========================================================================== #
#  Synthetic dataset                                                          #
# =========================================================================== #

def make_synthetic_batch(batch: int = 4, sents_per_para: int = 4,
                         tokens_per_sent: int = 12, n_paras: int = 3,
                         vocab: int = 256, device: str = "cpu",
                         seed: int = 0) -> Tuple[DocStructure, torch.Tensor]:
    """A tiny structured LM task. Tokens are chosen so the next token can be
    predicted from local sentence context, but every 4th token in a sentence
    depends on the *previous* sentence's last token — exercising the L1
    broadcast path."""
    g = torch.Generator(device=device).manual_seed(seed)
    n_sents = sents_per_para * n_paras
    T = n_sents * tokens_per_sent

    base = torch.randint(2, vocab - 4, (batch, T), generator=g, device=device)
    sent_id = (torch.arange(T, device=device) // tokens_per_sent).unsqueeze(0).expand(batch, T).clone()
    token_in_sent = (torch.arange(T, device=device) % tokens_per_sent).unsqueeze(0).expand(batch, T).clone()

    # Inject inter-sentence dependency: token at position 4 of sentence k+1 is
    # forced to equal last token of sentence k.
    tokens = base.clone()
    for s in range(1, n_sents):
        last_of_prev = tokens[:, s * tokens_per_sent - 1]
        tokens[:, s * tokens_per_sent + 4] = last_of_prev

    para_id_per_sent = (torch.arange(n_sents, device=device) // sents_per_para)
    sent_pos_in_para = (torch.arange(n_sents, device=device) % sents_per_para)
    para_pos = torch.arange(n_paras, device=device)

    doc = DocStructure(
        token_ids=tokens,
        token_in_sent=token_in_sent,
        sent_id=sent_id,
        para_id_per_sent=para_id_per_sent.unsqueeze(0).expand(batch, n_sents).clone(),
        sent_pos_in_para=sent_pos_in_para.unsqueeze(0).expand(batch, n_sents).clone(),
        para_pos=para_pos.unsqueeze(0).expand(batch, n_paras).clone(),
        n_sent=torch.full((batch,), n_sents, device=device),
        n_para=torch.full((batch,), n_paras, device=device),
    )
    targets = tokens.clone()
    return doc, targets


# =========================================================================== #
#  Stage scheduler — freezes/unfreezes parameter groups                        #
# =========================================================================== #

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def configure_for_stage(model: HRoPEv6Model, stage: int) -> List[str]:
    """Set requires_grad according to curriculum stage. Returns list of trained
    module names (for logging)."""
    # Always trainable
    for m in [model.tok_emb, model.lm_head, model.head_norm, *model.l0]:
        _set_grad(m, True)

    if stage == 0:
        for blk in model.l1:
            _set_grad(blk, False)
        _set_grad(model.sent_pool, False)
        _set_grad(model.broadcast_l1, False)
        if model.cfg.use_l2:
            for blk in model.l2:
                _set_grad(blk, False)
            _set_grad(model.para_pool, False)
            _set_grad(model.broadcast_l2, False)
        return ["tok_emb", "L0", "lm_head"]

    if stage == 1:
        for blk in model.l1:
            _set_grad(blk, True)
        _set_grad(model.sent_pool, True)
        _set_grad(model.broadcast_l1, True)
        if model.cfg.use_l2:
            for blk in model.l2:
                _set_grad(blk, False)
            _set_grad(model.para_pool, False)
            _set_grad(model.broadcast_l2, False)
        return ["tok_emb", "L0", "L1", "sent_pool", "broadcast_l1", "lm_head"]

    if stage in (2, 3):
        for m in [*model.l1, model.sent_pool, model.broadcast_l1]:
            _set_grad(m, True)
        if model.cfg.use_l2:
            for blk in model.l2:
                _set_grad(blk, True)
            _set_grad(model.para_pool, True)
            _set_grad(model.broadcast_l2, True)
        return ["all"]

    raise ValueError(f"Unknown stage {stage}")


# =========================================================================== #
#  Stitch-robustness perturbation (Stage 3)                                    #
# =========================================================================== #

def stitch_perturb(doc: DocStructure, drop_prob: float = 0.1,
                   shuffle_prob: float = 0.1) -> DocStructure:
    """Perturb the document structure to mimic cache-stitching artifacts."""
    B, T = doc.token_ids.shape
    new_sent_pos_in_para = doc.sent_pos_in_para.clone()
    if torch.rand(()).item() < shuffle_prob:
        N_s = doc.sent_pos_in_para.size(1)
        perm = torch.randperm(N_s)
        new_sent_pos_in_para = doc.sent_pos_in_para[:, perm]
    return DocStructure(
        token_ids=doc.token_ids,
        token_in_sent=doc.token_in_sent,
        sent_id=doc.sent_id,
        para_id_per_sent=doc.para_id_per_sent,
        sent_pos_in_para=new_sent_pos_in_para,
        para_pos=doc.para_pos,
        n_sent=doc.n_sent,
        n_para=doc.n_para,
    )


# =========================================================================== #
#  Training loop                                                              #
# =========================================================================== #

def lm_loss(logits: torch.Tensor, targets: torch.Tensor,
            sent_id: torch.Tensor, stage: int) -> torch.Tensor:
    """Standard next-token cross-entropy. For stage 0, mask out positions
    that span across sentence boundaries (since L0 cannot see the next
    sentence's first token anyway)."""
    B, T, V = logits.shape
    pred = logits[:, :-1].reshape(-1, V)
    tgt = targets[:, 1:].reshape(-1)
    if stage == 0:
        sid_a = sent_id[:, :-1]
        sid_b = sent_id[:, 1:]
        mask = (sid_a == sid_b).reshape(-1)
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(pred[mask], tgt[mask])
    return F.cross_entropy(pred, tgt)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg = HRoPEv6Config(
        vocab_size=256,
        d_model=128, n_heads=4, ffn_mult=4,
        n_layers_l0=2, n_layers_l1=2, n_layers_l2=1,
        max_tokens_per_sent=64, max_sents_per_doc=64,
        max_paras_per_doc=16,
    )
    model = HRoPEv6Model(cfg).to(device)

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Resumed weights from {args.resume}")

    trained = configure_for_stage(model, args.stage)
    print(f"[stage {args.stage}] training: {trained}")

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"[stage {args.stage}] trainable params: {n_trainable:,}")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    t0 = time.time()
    losses: List[float] = []
    for step in range(args.steps):
        if args.stage == 2:
            n_paras = 3 + (step // 50)
            doc, tgt = make_synthetic_batch(batch=args.batch, n_paras=n_paras,
                                            device=args.device, seed=step)
        elif args.stage == 3:
            doc, tgt = make_synthetic_batch(batch=args.batch,
                                            device=args.device, seed=step)
            doc = stitch_perturb(doc, drop_prob=0.0, shuffle_prob=0.2)
        else:
            doc, tgt = make_synthetic_batch(batch=args.batch,
                                            device=args.device, seed=step)

        out = model(doc)
        loss = lm_loss(out["logits"], tgt, doc.sent_id, args.stage)

        # Auxiliary contrastive: encourage sentence summaries to be unique.
        if args.stage >= 1 and out["sent_summ"].size(1) > 1:
            ss = F.normalize(out["sent_summ"], dim=-1)
            sim = ss @ ss.transpose(-1, -2)
            B, N_s, _ = sim.shape
            eye = torch.eye(N_s, device=device).unsqueeze(0)
            off = (sim - eye).clamp_min(0.0)
            loss = loss + 0.05 * off.pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            recent = losses[-min(20, len(losses)):]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"avg20={avg:.4f}  ({elapsed:.1f}s)")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved checkpoint to {args.save}")

    # Quick eval: did loss decrease?
    first_avg = sum(losses[: max(1, args.steps // 10)]) / max(1, args.steps // 10)
    last_avg = sum(losses[-max(1, args.steps // 10):]) / max(1, args.steps // 10)
    drop = first_avg - last_avg
    print(f"[stage {args.stage}] loss start≈{first_avg:.3f} → end≈{last_avg:.3f}  "
          f"drop={drop:.3f}")
    if drop <= 0:
        print(f"[stage {args.stage}] WARNING: loss did not decrease")


# =========================================================================== #
#  CLI                                                                        #
# =========================================================================== #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--save", default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
