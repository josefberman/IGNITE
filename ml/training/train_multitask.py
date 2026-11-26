"""Batched trainer for ``MultitaskPlanner`` (``train_planner --arch multitask``).

Same objective, masks, metrics, and early-stopping score as the legacy loop in
``train_planner``, computed on mini-batches of graphs instead of one pair at a
time:

* family cross-entropy over the parent's applicable families (+ gold),
* reagent cross-entropy given the true family, over that family's reagents
  applicable to the parent (+ gold),
* site loss: softmax over the reagent's candidate sites, every site giving the
  labeled child counts as correct; all-atom softmax when no candidate is
  labeled.

The legacy loop takes one optimizer step per ``accum_steps`` pairs with the
loss averaged over them, so ``--batch-size 16`` here is the same update as its
default ``--accum-steps 16``. Its per-pair family class weights cancel at batch
size 1, so the family loss here is unweighted too.

Validation picks the epoch (score = 0.25 family top-1 + site top-1 +
reagent top-1 given the true family, as in the legacy loop). The test split is
evaluated once, on the best epoch, only with ``--eval-test``.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.models.architectures import (
    MultitaskPlanner,
    _seg_max,
    _seg_sum,
    segment_logsumexp,
)
from ml.training.pair_cache import PairCache, ensure_pair_cache

NEG = -1e9


def selection_score(m: dict) -> float:
    return 0.25 * m["family_top1"] + m["site_top1"] + m["reagent_oracle_top1"]


def _family_mask(b, gold: torch.Tensor) -> torch.Tensor:
    allow = b["fam_allow"].clone()
    has = allow.any(dim=1)
    allow[torch.arange(len(gold)), gold] = True
    allow[~has] = True
    return allow


def _reagent_mask(b, fam_pool: torch.Tensor, fam: torch.Tensor, gold: torch.Tensor | None) -> torch.Tensor:
    pool = fam_pool[fam]
    matched = b["reag_match"]
    has_match = matched.any(dim=1, keepdim=True)
    allowed = torch.where(has_match, pool & matched, pool)
    empty = ~allowed.any(dim=1, keepdim=True)
    allowed = torch.where(empty, pool, allowed)
    if gold is not None:
        allowed = allowed.clone()
        rows = torch.arange(len(gold))
        allowed[rows, gold] = True
    none = ~allowed.any(dim=1, keepdim=True)
    return allowed | none


def _masked(logits, mask):
    return logits.masked_fill(~mask, NEG)


def _tie_rank(scores, pos, seg, n):
    """0-based rank of the best positive in each segment, ties broken by lower index.

    Rank 0 is exactly ``argmax`` (first index on ties) landing on a positive,
    which is how the legacy loop scores site top-1; symmetric atoms often tie.
    """
    inf = torch.full_like(scores, -float("inf"))
    best = _seg_max(torch.where(pos, scores, inf), seg, n)
    gidx = torch.arange(scores.numel())
    star = pos & (scores == best[seg])
    big = torch.full((n,), scores.numel(), dtype=torch.long)
    first_star = big.scatter_reduce(0, seg[star], gidx[star], reduce="amin", include_self=True)
    ahead = (scores > best[seg]) | ((scores == best[seg]) & (gidx < first_star[seg]))
    return torch.zeros(n, dtype=torch.long).index_add(0, seg, ahead.to(torch.long))


def _site_terms(b, atom_logits):
    """Candidate scores, per-pair site loss, and per-pair site rank (0 = top-1; -1 unlabeled)."""
    B = b["n"]
    batch = b["batch"]
    nc = b["n_cand"]
    cand_seg = b["cand_seg"]
    pos = b["cand_pos"]
    if nc > 0:
        cnt = _seg_sum(torch.ones_like(b["cand_atom_cand"], dtype=atom_logits.dtype), b["cand_atom_cand"], nc)
        tot = _seg_sum(atom_logits[b["cand_atom_idx"]], b["cand_atom_cand"], nc)
        scores = torch.where(cnt > 0, tot / cnt.clamp_min(1.0), torch.full_like(tot, NEG))
        lab = _seg_sum(pos.to(atom_logits.dtype), cand_seg, B) > 0
        lse_all = segment_logsumexp(scores, cand_seg, B)
        lse_pos = segment_logsumexp(scores[pos], cand_seg[pos], B)
        cand_loss = lse_all - lse_pos
    else:
        scores = atom_logits.new_zeros(0)
        lab = torch.zeros(B, dtype=torch.bool)
        cand_loss = atom_logits.new_zeros(B)
    tgt = torch.zeros_like(atom_logits)
    if b["site_idx"].numel():
        tgt[b["site_idx"]] = 1.0
    t_sum = _seg_sum(tgt, batch, B)
    has_t = t_sum > 0
    lse_atoms = segment_logsumexp(atom_logits, batch, B)
    atom_loss = lse_atoms - _seg_sum(tgt * atom_logits, batch, B) / t_sum.clamp_min(1.0)
    n_at = _seg_sum(torch.ones_like(atom_logits), batch, B)
    reg = 0.01 * _seg_sum(atom_logits ** 2, batch, B) / n_at.clamp_min(1.0)
    loss = torch.where(lab, cand_loss, torch.where(has_t, atom_loss, reg))

    with torch.no_grad():
        rank = torch.full((B,), -1, dtype=torch.long)
        if nc > 0 and bool(lab.any()):
            rank = torch.where(lab, _tie_rank(scores, pos, cand_seg, B), rank)
        fb = (~lab) & has_t
        if bool(fb.any()):
            rank = torch.where(fb, _tie_rank(atom_logits, tgt > 0, batch, B), rank)
    return loss, rank


def _forward(model: MultitaskPlanner, b, fam_pool, *, eval_mode: bool):
    gap = b["gap"]
    enc = model.encode_batch(b["x"], b["edge"], b["batch"], b["n"], gap)
    z_f, _ = enc["family"]
    z_r, emb_r = enc["reagent"]
    z_s, emb_s = enc["site"]
    fam, reag = b["fam"], b["reag"]
    if model.conditioning == "cascade":
        fam_h = model.family_hidden(z_f, gap)
        fam_logits = model.family_out(fam_h)
        reag_h = model.reagent_hidden(z_r, None, gap, emb_r, b["batch"], b["n"], fam_h=fam_h)
        reag_logits = model.reagent_out(reag_h)
        site_logits = model.site_logits(emb_s, z_s, gap, b["batch"], reag_h=reag_h)
    elif model.conditioning == "parallel":
        fam_logits = model.family_logits(z_f, gap)
        reag_logits = model.reagent_logits(z_r, None, gap, emb_r, b["batch"], b["n"])
        site_logits = model.site_logits(emb_s, z_s, gap, b["batch"])
    else:
        fam_logits = model.family_logits(z_f, gap)
        onehot = F.one_hot(fam, model.n_families).to(z_r.dtype)
        reag_logits = model.reagent_logits(z_r, onehot, gap, emb_r, b["batch"], b["n"])
        site_logits = model.site_logits(emb_s, z_s, gap, b["batch"], reag)
    fam_m = _masked(fam_logits, _family_mask(b, fam))
    reag_m = _masked(reag_logits, _reagent_mask(b, fam_pool, fam, reag))
    l_fam = F.cross_entropy(fam_m, fam, reduction="none")
    l_reag = F.cross_entropy(reag_m, reag, reduction="none")
    l_site, site_rank = _site_terms(b, site_logits)
    out = {"l_fam": l_fam, "l_reag": l_reag, "l_site": l_site}
    if eval_mode:
        pred_fam = fam_m.argmax(dim=1)
        top3 = fam_m.topk(min(3, fam_m.size(1)), dim=1).indices
        if model.conditioning in ("parallel", "cascade"):
            reag_pred_logits = reag_logits
        else:
            oh_pred = F.one_hot(pred_fam, model.n_families).to(z_r.dtype)
            reag_pred_logits = model.reagent_logits(z_r, oh_pred, gap, emb_r, b["batch"], b["n"])
        gold_if_right = torch.where(pred_fam == fam, reag, torch.full_like(reag, -1))
        pmask = _reagent_mask(b, fam_pool, pred_fam, None)
        right = gold_if_right >= 0
        pmask[right, reag[right]] = True
        pred_reag = _masked(reag_pred_logits, pmask).argmax(dim=1)
        if model.site_reagent_emb is not None:
            # Training loss stays on the gold reagent. This rank is the site
            # given the reagent the model just predicted, matching search.
            _, site_rank = _site_terms(
                b, model.site_logits(emb_s, z_s, gap, b["batch"], pred_reag)
            )
        out.update(
            fam_ok=pred_fam == fam,
            fam3_ok=(top3 == fam.unsqueeze(1)).any(dim=1),
            reag_ok=pred_reag == reag,
            reag_oracle_ok=reag_m.argmax(dim=1) == reag,
            site_rank=site_rank,
        )
    return out


def evaluate(model, cache: PairCache, idx: np.ndarray, fam_pool, batch_size: int, w_reag=1.0, w_site=1.0) -> dict:
    model.eval()
    acc = {k: 0 for k in ("fam", "fam3", "reag", "reag_oracle", "plan", "site", "site3", "site_n", "plan_all")}
    loss_sum = {"fam": 0.0, "reag": 0.0, "site": 0.0}
    n = 0
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = cache.batch(idx[s : s + batch_size])
            o = _forward(model, b, fam_pool, eval_mode=True)
            n += b["n"]
            for k in loss_sum:
                loss_sum[k] += float(o[f"l_{k}"].sum())
            plan = o["fam_ok"] & o["reag_ok"]
            acc["fam"] += int(o["fam_ok"].sum())
            acc["fam3"] += int(o["fam3_ok"].sum())
            acc["reag"] += int(o["reag_ok"].sum())
            acc["reag_oracle"] += int(o["reag_oracle_ok"].sum())
            acc["plan"] += int(plan.sum())
            lab = o["site_rank"] >= 0
            acc["site_n"] += int(lab.sum())
            acc["site"] += int((o["site_rank"] == 0).sum())
            acc["site3"] += int((lab & (o["site_rank"] < 3)).sum())
            acc["plan_all"] += int((plan & (o["site_rank"] == 0)).sum())
    sn = max(1, acc["site_n"])
    m = {
        "n": n,
        "loss": (loss_sum["fam"] + w_reag * loss_sum["reag"] + w_site * loss_sum["site"]) / max(1, n),
        "loss_family": loss_sum["fam"] / max(1, n),
        "loss_reagent": loss_sum["reag"] / max(1, n),
        "loss_site": loss_sum["site"] / max(1, n),
        "family_top1": acc["fam"] / max(1, n),
        "family_top3": acc["fam3"] / max(1, n),
        "reagent_top1": acc["reag"] / max(1, n),
        "reagent_oracle_top1": acc["reag_oracle"] / max(1, n),
        "plan_top1": acc["plan"] / max(1, n),
        "site_top1": acc["site"] / sn,
        "site_top3": acc["site3"] / sn,
        "plan_all_top1": acc["plan_all"] / sn,
        "site_labeled": acc["site_n"],
    }
    m["score"] = selection_score(m)
    return m


def _subsample(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap is None or cap < 0 or len(idx) <= cap:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, size=cap, replace=False))


def train_multitask(
    data_csv: Path,
    out_dir: Path,
    *,
    epochs: int = 30,
    patience: int = 7,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    batch_size: int = 16,
    seed: int = 42,
    hidden: int,
    dropout: float,
    bond_types: int,
    site_reagent: bool,
    shared_mp: int,
    branch_mp: int,
    shared_fc: int,
    tower_hidden: int | None,
    family_tower_layers: int,
    reagent_tower_layers: int,
    site_tower_layers: int,
    loss_reag_weight: float = 1.0,
    loss_site_weight: float = 1.0,
    loss_weighting: str = "fixed",
    split: str = "random_parent",
    split_fractions=(0.7, 0.1, 0.2),
    split_seed: int = 42,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    max_train: int = -1,
    max_val: int = -1,
    eval_test: bool = False,
    cache_path: Path | None = None,
    cache_workers: int = 8,
    threads: int = 0,
    max_hours: float = 0.0,
    conditioning: str = "hierarchical",
):
    if threads > 0:
        torch.set_num_threads(threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cache = PairCache(ensure_pair_cache(data_csv, cache_path, cache_workers))
    families, reagents = cache.families, cache.reagents
    fam_pool = torch.from_numpy(cache.fam_pool)
    if train_idx is not None:
        tr_full = np.asarray(train_idx)
        va_full = np.asarray(val_idx) if val_idx is not None else np.array([], dtype=np.int64)
        te_idx = np.asarray(test_idx) if test_idx is not None else np.array([], dtype=np.int64)
        split_mode = "explicit"
    else:
        codes = cache.split_codes(split, split_fractions, split_seed)
        all_idx = np.arange(cache.n_pairs)
        tr_full, va_full, te_idx = all_idx[codes == 0], all_idx[codes == 1], all_idx[codes == 2]
        split_mode = split
    tr_idx = _subsample(tr_full, max_train, seed)
    va_idx = _subsample(va_full, max_val, seed + 1)
    split_info = {
        "mode": split_mode,
        "fractions": list(split_fractions),
        "seed": split_seed,
        "n_pairs": int(cache.n_pairs),
        "n_train_split": int(len(tr_full)),
        "n_val_split": int(len(va_full)),
        "n_test_split": int(len(te_idx)),
        "n_train_used": int(len(tr_idx)),
        "n_val_used": int(len(va_idx)),
    }
    print(f"split {json.dumps(split_info)}", flush=True)

    model = MultitaskPlanner(
        n_families=len(families),
        n_reagents=len(reagents),
        hidden=hidden,
        shared_mp=shared_mp,
        branch_mp=branch_mp,
        shared_fc=shared_fc,
        tower_hidden=tower_hidden,
        family_tower_layers=family_tower_layers,
        reagent_tower_layers=reagent_tower_layers,
        site_tower_layers=site_tower_layers,
        dropout=dropout,
        bond_types=bond_types,
        site_reagent=site_reagent,
        conditioning=conditioning,
    )
    n_params = sum(p.numel() for p in model.parameters())
    params = list(model.parameters())
    log_vars = None
    if loss_weighting == "uncertainty":
        # Kendall et al. homoscedastic task weighting: sum_t exp(-s_t) L_t + s_t.
        log_vars = torch.nn.Parameter(torch.zeros(3))
        params.append(log_vars)
    elif loss_weighting != "fixed":
        raise ValueError(f"unknown loss weighting {loss_weighting!r}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    history = []
    best = None
    best_score = -1.0
    stale = 0
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(tr_idx)
        tot = 0.0
        n = 0
        t0 = time.time()
        for s in range(0, len(order), batch_size):
            b = cache.batch(order[s : s + batch_size])
            o = _forward(model, b, fam_pool, eval_mode=False)
            lf, lr_, ls = o["l_fam"].mean(), o["l_reag"].mean(), o["l_site"].mean()
            if log_vars is None:
                loss = lf + loss_reag_weight * lr_ + loss_site_weight * ls
            else:
                ls_all = torch.stack([lf, lr_, ls])
                loss = (torch.exp(-log_vars) * ls_all + log_vars).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            tot += float((lf + lr_ + ls).item()) * b["n"]
            n += b["n"]
            if (s // batch_size) % 2000 == 0:
                print(f"  epoch {epoch+1} {n}/{len(order)} loss={tot / max(1, n):.4f} {n / max(1e-9, time.time() - t0):.0f} pairs/s", flush=True)
        scheduler.step()
        train_time = time.time() - t0
        rec = {
            "epoch": epoch + 1,
            "train_loss": tot / max(1, n),
            "lr": scheduler.get_last_lr()[0],
            "train_seconds": train_time,
        }
        if log_vars is not None:
            rec["log_vars"] = [float(v) for v in log_vars.detach()]
        if len(va_idx) == 0:
            history.append(rec)
            print(f"epoch {epoch+1}: train={rec['train_loss']:.4f} ({train_time:.0f}s)", flush=True)
        else:
            vm = evaluate(model, cache, va_idx, fam_pool, 256, loss_reag_weight, loss_site_weight)
            rec.update({f"val_{k}": v for k, v in vm.items() if k != "n"})
            history.append(rec)
            print(
                f"epoch {epoch+1}: train={rec['train_loss']:.4f} val={vm['loss']:.4f} fam={vm['family_top1']:.4f} "
                f"fam3={vm['family_top3']:.4f} reag={vm['reagent_top1']:.4f} reag_oracle={vm['reagent_oracle_top1']:.4f} "
                f"site={vm['site_top1']:.4f} site3={vm['site_top3']:.4f} plan={vm['plan_top1']:.4f} "
                f"score={vm['score']:.4f} ({train_time:.0f}s)",
                flush=True,
            )
            if vm["score"] > best_score + 1e-4:
                best_score = vm["score"]
                best = {"state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1, "val": vm}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stop at epoch {epoch+1} (best score={best_score:.4f})", flush=True)
                    break
        if max_hours > 0 and (time.time() - t_start) / 3600 > max_hours:
            print(f"Stopping: time budget {max_hours} h reached", flush=True)
            break

    if best is None:
        best = {"state": copy.deepcopy(model.state_dict()), "epoch": len(history), "val": {"score": None, "family_top1": None}}
    model.load_state_dict(best["state"])
    test_metrics = None
    if eval_test and len(te_idx) > 0:
        test_metrics = evaluate(model, cache, te_idx, fam_pool, 256, loss_reag_weight, loss_site_weight)
        print(f"TEST {json.dumps(test_metrics)}", flush=True)

    arch_meta = {
        **model.arch_meta(),
        "encoder": "residual_mp_mean_max",
        "gap_fuse": "film",
        "site_head": "reagent_candidates" if model.site_reagent else "local_context",
        "reagent_local": "family_only",
        "vocab": "broad_family_specific_reagent",
        "applicability_mask": True,
        "family_conditioned_reagent": model.conditioning in ("hierarchical", "cascade"),
    }
    hparams = {
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "loss_reag_weight": loss_reag_weight,
        "loss_site_weight": loss_site_weight,
        "loss_weighting": loss_weighting,
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
    }
    bundle = {
        "families": families,
        "reagents": reagents,
        "family_to_reagent_idxs": {
            f: [int(i) for i in np.flatnonzero(cache.fam_pool[j])] for j, f in enumerate(families)
        },
        "history": history,
        "site_supervision": True,
        "reagent_supervision": True,
        **arch_meta,
        "planner_gap": "signed_normalized",
        "train_hparams": hparams,
        "split": split_info,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "best_val_family_top1": best["val"]["family_top1"],
        "test": test_metrics,
        "n_params": n_params,
    }
    torch.save({"planner": best["state"], "meta": bundle}, out_dir / "planner.pt")
    card = {
        "name": "ignite_modification_planner",
        "version": "0.11.0-multitask",
        "label_type": "weak_xgb_delta+reacting_atom_sites+reagent_variant",
        **{k: v for k, v in bundle.items() if k not in ("families", "reagents", "family_to_reagent_idxs")},
        "n_families": len(families),
        "n_reagents": len(reagents),
        "disclaimer": "Trained on synthetic weak labels; not experimental ground truth.",
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"Saved checkpoint to {out_dir / 'planner.pt'} (best epoch {best['epoch']})", flush=True)
    return card
