"""
HRoPE v7 — Training script with U-Net staged curriculum.

The v7 curriculum mirrors nnU-Net's standard practice: train deepest level
first (smallest data), then unfreeze upward. For us that means:

  Stage 0 — L0 enc + L0 dec only (sentence-internal LM).
            L1, L2, pools, decoder upper levels are inactive (skip_l1 = 0,
            unpool from L1 returns 0). This learns sentence-local
            representations and is the cheapest stage to converge.

  Stage 1 — Add L1 enc + L1 dec + sent_pool. Train doc-level next-token loss.

  Stage 2 — Add L2 enc + L2 dec + para_pool. Document length grows.

  Stage 3 — Stitch-robustness. Random sentence-stream perturbations within
            each paragraph; the U-Net's skips make L0 robust because the
            high-res sentence-local content reaches the head directly.

Usage:
    python hrope_v7_train.py --stage 0 --steps 200 --save ckpt0.pt
    python hrope_v7_train.py --stage 1 --resume ckpt0.pt --save ckpt1.pt
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrope_v7_reference import (
    DocStructure,
    HRoPEv7Config,
    HRoPEv7Model,
)


# =========================================================================== #
#  Synthetic dataset (same as v6 — exercises L1 broadcast path)                #
# =========================================================================== #

def make_synthetic_batch(batch: int = 4, sents_per_para: int = 4,
                         tokens_per_sent: int = 12, n_paras: int = 3,
                         vocab: int = 256, device: str = "cpu",
                         seed: int = 0) -> Tuple[DocStructure, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    n_sents = sents_per_para * n_paras
    T = n_sents * tokens_per_sent

    base = torch.randint(2, vocab - 4, (batch, T), generator=g, device=device)
    sent_id = (torch.arange(T, device=device) // tokens_per_sent).unsqueeze(0).expand(batch, T).clone()
    token_in_sent = (torch.arange(T, device=device) % tokens_per_sent).unsqueeze(0).expand(batch, T).clone()

    # Inter-sentence dependency: token at pos 4 of sentence k+1 = last token
    # of sentence k. This forces L1 to carry useful signal.
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
    return doc, tokens.clone()


# =========================================================================== #
#  Stage scheduler                                                            #
# =========================================================================== #

def _set_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def configure_for_stage(model: HRoPEv7Model, stage: int) -> List[str]:
    # Always trainable: token embedding + L0 enc/dec + LM head + head norm.
    for m in [model.tok_emb, model.head_norm, model.lm_head,
              *model.l0_enc, *model.l0_dec]:
        _set_grad(m, True)

    if stage == 0:
        # Freeze everything above L0.
        for m in [*model.l1_enc, *model.l1_dec, model.sent_pool]:
            _set_grad(m, False)
        if model.cfg.use_l2:
            for m in [*model.l2_enc, *model.l2_dec, model.para_pool]:
                _set_grad(m, False)
        return ["tok_emb", "L0_enc", "L0_dec", "lm_head"]

    if stage == 1:
        for m in [*model.l1_enc, *model.l1_dec, model.sent_pool]:
            _set_grad(m, True)
        if model.cfg.use_l2:
            for m in [*model.l2_enc, *model.l2_dec, model.para_pool]:
                _set_grad(m, False)
        return ["+ L1_enc", "L1_dec", "sent_pool"]

    if stage in (2, 3):
        for m in [*model.l1_enc, *model.l1_dec, model.sent_pool]:
            _set_grad(m, True)
        if model.cfg.use_l2:
            for m in [*model.l2_enc, *model.l2_dec, model.para_pool]:
                _set_grad(m, True)
        return ["all"]

    raise ValueError(f"Unknown stage {stage}")


# =========================================================================== #
#  Stage-0 surgery: zero-out the upper-level paths so L0 trains in isolation  #
# =========================================================================== #

def stage0_forward(model: HRoPEv7Model, doc: DocStructure) -> torch.Tensor:
    """Stage-0 forward: bypass L1/L2 entirely.
    skip_l1 = 0, unpool(skip_l1) = 0, so L0 dec sees only skip_l0.
    This trains L0 enc and L0 dec on pure intra-sentence next-token loss.
    """
    x0 = model.tok_emb(doc.token_ids)
    l0_mask = model._l0_mask(doc.sent_id)
    for blk in model.l0_enc:
        x0 = blk(x0, doc.token_in_sent, attn_mask=l0_mask)
    skip_l0 = x0
    # Skip the entire upper hierarchy: y0 = 0 + skip_l0
    y0 = skip_l0
    for blk in model.l0_dec:
        y0 = blk(y0, doc.token_in_sent, attn_mask=l0_mask)
    return model.lm_head(model.head_norm(y0))


def stitch_perturb(doc: DocStructure, shuffle_prob: float = 0.2) -> DocStructure:
    new_pos = doc.sent_pos_in_para.clone()
    if torch.rand(()).item() < shuffle_prob:
        N_s = doc.sent_pos_in_para.size(1)
        perm = torch.randperm(N_s)
        new_pos = doc.sent_pos_in_para[:, perm]
    return DocStructure(
        token_ids=doc.token_ids, token_in_sent=doc.token_in_sent,
        sent_id=doc.sent_id, para_id_per_sent=doc.para_id_per_sent,
        sent_pos_in_para=new_pos, para_pos=doc.para_pos,
        n_sent=doc.n_sent, n_para=doc.n_para,
    )


# =========================================================================== #
#  Loss                                                                       #
# =========================================================================== #

def lm_loss(logits: torch.Tensor, targets: torch.Tensor,
            sent_id: torch.Tensor, stage: int) -> torch.Tensor:
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


# =========================================================================== #
#  Training loop                                                              #
# =========================================================================== #

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg = HRoPEv7Config(
        vocab_size=256, d_model=128, n_heads=4, ffn_mult=4,
        n_l0_enc=2, n_l0_dec=2, n_l1_enc=1, n_l1_dec=1,
        n_l2_enc=1, n_l2_dec=1,
        max_tokens_per_sent=64, max_sents_per_doc=64, max_paras_per_doc=16,
    )
    model = HRoPEv7Model(cfg).to(device)

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"Resumed weights from {args.resume}")

    trained = configure_for_stage(model, args.stage)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage {args.stage}] training: {trained}  ({n_train:,} params)")

    params = [p for p in model.parameters() if p.requires_grad]
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
            doc = stitch_perturb(doc, shuffle_prob=0.2)
        else:
            doc, tgt = make_synthetic_batch(batch=args.batch,
                                            device=args.device, seed=step)

        if args.stage == 0:
            logits = stage0_forward(model, doc)
            sent_summ = None
        else:
            out = model(doc)
            logits = out["logits"]
            sent_summ = out["skip_l1"]

        loss = lm_loss(logits, tgt, doc.sent_id, args.stage)

        # Aux: encourage sentence summaries to be diverse (only when L1 active).
        if sent_summ is not None and sent_summ.size(1) > 1:
            ss = F.normalize(sent_summ, dim=-1)
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
            print(f"  step {step:4d}/{args.steps}  loss={loss.item():.4f}  "
                  f"avg20={avg:.4f}  ({time.time() - t0:.1f}s)")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved checkpoint to {args.save}")

    first = sum(losses[: max(1, args.steps // 10)]) / max(1, args.steps // 10)
    last = sum(losses[-max(1, args.steps // 10):]) / max(1, args.steps // 10)
    print(f"[stage {args.stage}] loss start≈{first:.3f} → end≈{last:.3f}  "
          f"drop={first - last:.3f}")


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
