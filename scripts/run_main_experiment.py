#!/usr/bin/env python3
"""Reproduce Table 1: the full 10-dataset x 6-method MEHERAB benchmark.

This is a faithful, modular reproduction of the original pipeline's main
experiment loop (``notebooks/meherab_pipeline.py``, Cell 17), rebuilt on top
of the ``meherab`` package instead of notebook globals. Per dataset, it:

  1. extracts the unlabeled proxy set and computes the final-block baseline FDR,
  2. builds the task-conditional homophilic graph,
  3. runs the MDS-guided evolutionary search (Algorithm 1),
  4. evaluates all 6 methods (LP, Rand.RASS, LoRA, BnAdapter, CLIP-Adapter,
     MEHERAB) across 5 independent seeds,
  5. runs the MDS proxy-validation experiment (Spearman rho, Precision@k
     against MDS / NASWOT-adapted / SynFlow-adapted), and
  6. checkpoints results to disk after every dataset (resumable).

Usage
-----
    python scripts/run_main_experiment.py --output-dir outputs/

See docs/REPRODUCING.md for full setup instructions and expected runtime
(~3-4 minutes search + ~1-2 minutes evaluation per dataset on a single T4
GPU; ten datasets total).
"""
import argparse
import json
import logging
import os
import time
import traceback

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from meherab import MEHERABConfig, GLOBAL_SEED, EVAL_SEEDS, set_all_seeds, load_frozen_backbone
from meherab.data import load_all_datasets, extract_proxy, extract_split
from meherab.graph import fisher_discriminant_ratio, build_homophilic_graph
from meherab.rass import RASSFactory, FittedRASSTransform, build_meherab_features
from meherab.search import compute_mds, apply_rass_proxy, naswot_score, synflow_score, run_evo_search
from meherab.baselines import train_peft, train_clip_adapter
from meherab.eval import evaluate_with_probe, evaluate_fast, precision_at_k


def run_mds_correlation_experiment(backbone, device, cfg, ds_name, train_ds, test_ds, n_classes, n_candidates=None):
    """Validate the MDS proxy via Spearman rho and Precision@k against
    actual accuracy (paper Sec. 5.2, Fig. 3, Table 2).
    """
    n_candidates = n_candidates or cfg.n_corr_cands
    print(f"\n[Corr] {ds_name}  ({n_candidates} candidates) ...")
    set_all_seeds(GLOBAL_SEED)

    proxy_lf, proxy_lbl = extract_proxy(
        backbone, train_ds, n_classes, GLOBAL_SEED, device,
        cfg.base_proxy_n, cfg.proxy_max_n, cfg.proxy_per_class, cfg.proxy_batch,
    )
    pretrain_ref = np.stack([lf.numpy() for lf in proxy_lf.values()]).mean(0)
    fb_id = max(proxy_lf.keys())
    fb_nrm = StandardScaler().fit_transform(proxy_lf[fb_id].numpy())
    bfdr = fisher_discriminant_ratio(fb_nrm, proxy_lbl)

    hg = build_homophilic_graph(proxy_lf, proxy_lbl, cfg.n_clusters, cfg.hyperedge_delta, verbose=False)
    rass = RASSFactory(hg, cfg.sc_keep_ratios, cfg.adapter_ranks, cfg.n_ops, cfg.evo_mutation)

    tr_fin, tr_lbl, tr_blk = extract_split(backbone, train_ds, cfg.n_train, GLOBAL_SEED, device)
    te_fin, te_lbl, te_blk = extract_split(backbone, test_ds, cfg.n_test, GLOBAL_SEED, device)

    records = []
    set_all_seeds(GLOBAL_SEED)
    for i in range(n_candidates):
        cand = rass.random_candidate()
        adapted = apply_rass_proxy(cand.ops, proxy_lf, hg, cfg.adapter_scale)
        adapted_norm = StandardScaler().fit_transform(adapted)
        mds_v = compute_mds(cand, proxy_lf, pretrain_ref, proxy_lbl, hg, bfdr, cfg.mds_alpha)
        nas_v = naswot_score(adapted_norm)
        syn_v = synflow_score(adapted_norm, proxy_lbl)
        tf = FittedRASSTransform(cand.ops, hg, tr_blk, cfg.adapter_scale)
        Xtr = build_meherab_features(tf, tr_blk, tr_fin)
        Xte = build_meherab_features(tf, te_blk, te_fin)
        acc = evaluate_fast(Xtr, tr_lbl, Xte, te_lbl, cfg.pca_dim)
        records.append({"mds": mds_v, "naswot": nas_v, "synflow": syn_v, "acc": acc})
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{n_candidates}] MDS={mds_v:.3f}  acc={acc:.2f}%")

    accs = np.array([r["acc"] for r in records])
    out = {"n": n_candidates, "dataset": ds_name}
    for proxy in ["mds", "naswot", "synflow"]:
        scores = np.array([r[proxy] for r in records])
        rho, p = spearmanr(scores, accs)
        out[proxy] = {
            "rho": float(rho),
            "pval": float(p),
            "p_at_5": precision_at_k(scores, accs, 5),
            "p_at_10": precision_at_k(scores, accs, 10),
        }
    return out


def run_one_dataset(backbone, device, cfg, ds_name, train_ds, test_ds, n_cls):
    proxy_n = max(min(cfg.proxy_max_n, cfg.proxy_per_class * n_cls), cfg.base_proxy_n)
    print(f"\n[A] Proxy extraction  N={proxy_n} ...")
    proxy_lf, proxy_lbl = extract_proxy(
        backbone, train_ds, n_cls, GLOBAL_SEED, device,
        cfg.base_proxy_n, cfg.proxy_max_n, cfg.proxy_per_class, cfg.proxy_batch,
    )
    pretrain_ref = np.stack([lf.numpy() for lf in proxy_lf.values()]).mean(0)
    fb_id = max(proxy_lf.keys())
    fb_nrm = StandardScaler().fit_transform(proxy_lf[fb_id].numpy())
    bfdr = fisher_discriminant_ratio(fb_nrm, proxy_lbl)
    print(f"  Baseline FDR (final block): {bfdr:.4f}")

    print("\n[B] Building task-conditional homophilic graph ...")
    hg = build_homophilic_graph(proxy_lf, proxy_lbl, cfg.n_clusters, cfg.hyperedge_delta)
    rass = RASSFactory(hg, cfg.sc_keep_ratios, cfg.adapter_ranks, cfg.n_ops, cfg.evo_mutation)
    print(f"  Nodes={hg.n_nodes()}  Edges={hg.n_edges()}  Op-set={len(rass.all_ops)}")

    print("\n[C] Evolutionary search (MDS-guided) ...")
    best_cand, gen_best, gen_mean = run_evo_search(
        rass, proxy_lf, pretrain_ref, proxy_lbl, hg, bfdr,
        cfg.evo_pop, cfg.evo_gens, cfg.evo_elite, cfg.evo_tournament, cfg.mds_alpha,
    )

    print("\n[D] Multi-seed evaluation (5 seeds, 6 methods) ...")
    lp_a, rr_a, lora_a, ada_a, ca_a, mhb_a = [], [], [], [], [], []
    block_lp_vals = {}

    for seed in EVAL_SEEDS:
        set_all_seeds(seed)
        tr_fin, tr_lbl, tr_blk = extract_split(backbone, train_ds, cfg.n_train, seed, device)
        te_fin, te_lbl, te_blk = extract_split(backbone, test_ds, cfg.n_test, seed, device)

        lp = evaluate_with_probe(tr_fin, tr_lbl, te_fin, te_lbl, seed, cfg.pca_dim, cfg.probe_C_grid, cfg.eval_max_iter)
        lp_a.append(lp)

        rc = rass.random_candidate()
        rtf = FittedRASSTransform(rc.ops, hg, tr_blk, cfg.adapter_scale)
        rr = evaluate_with_probe(
            build_meherab_features(rtf, tr_blk, tr_fin), tr_lbl,
            build_meherab_features(rtf, te_blk, te_fin), te_lbl, seed, cfg.pca_dim, cfg.probe_C_grid, cfg.eval_max_iter,
        )
        rr_a.append(rr)

        mtf = FittedRASSTransform(best_cand.ops, hg, tr_blk, cfg.adapter_scale)
        mhb = evaluate_with_probe(
            build_meherab_features(mtf, tr_blk, tr_fin), tr_lbl,
            build_meherab_features(mtf, te_blk, te_fin), te_lbl, seed, cfg.pca_dim, cfg.probe_C_grid, cfg.eval_max_iter,
        )
        mhb_a.append(mhb)

        lora = train_peft(tr_fin, tr_lbl, te_fin, te_lbl, "lora", seed, cfg.lora_rank, cfg.adapter_bottle, cfg.peft_lr, cfg.peft_epochs, cfg.peft_batch)
        ada = train_peft(tr_fin, tr_lbl, te_fin, te_lbl, "adapter", seed, cfg.lora_rank, cfg.adapter_bottle, cfg.peft_lr, cfg.peft_epochs, cfg.peft_batch)
        ca = train_clip_adapter(tr_fin, tr_lbl, te_fin, te_lbl, seed, 64, 0.2, cfg.peft_lr, cfg.peft_epochs, cfg.peft_batch)
        lora_a.append(lora); ada_a.append(ada); ca_a.append(ca)

        print(f"  seed={seed:>6}  LP={lp:6.2f}  Rand={rr:6.2f}  LoRA={lora:6.2f}  Adptr={ada:6.2f}  CA={ca:6.2f}  MEHERAB={mhb:6.2f}")

        if seed == EVAL_SEEDS[0] and isinstance(tr_blk, dict) and tr_blk:
            for bid in sorted(tr_blk.keys()):
                block_lp_vals[bid] = evaluate_fast(tr_blk[bid], tr_lbl, te_blk[bid], te_lbl, cfg.pca_dim)

    delta = np.mean(mhb_a) - np.mean(lp_a)
    print(f"\n  MEHERAB: {np.mean(mhb_a):.2f}+/-{np.std(mhb_a):.2f}%   LP: {np.mean(lp_a):.2f}+/-{np.std(lp_a):.2f}%   Delta: {delta:+.2f}%")
    if delta < 0:
        print(f"  [NOTE] MEHERAB below LP on {ds_name} -- reported raw (no floor applied)")

    print(f"\n[E] MDS rank-correlation ({cfg.n_corr_cands} candidates) ...")
    corr = run_mds_correlation_experiment(backbone, device, cfg, ds_name, train_ds, test_ds, n_cls)

    best_blk_id = max(block_lp_vals, key=block_lp_vals.get) if block_lp_vals else fb_id
    return {
        "n_classes": n_cls,
        "lp": lp_a, "rr": rr_a, "lora": lora_a, "adapter": ada_a, "clip_adapter": ca_a, "meherab": mhb_a,
        "best_ops": [str(o) for o in best_cand.ops],
        "best_mds": float(best_cand.mds_score),
        "gen_best": gen_best, "gen_mean": gen_mean,
        "proxy_n": proxy_n, "baseline_fdr": float(bfdr),
        "block_lp_vals": {str(k): v for k, v in block_lp_vals.items()},
        "best_blk": int(best_blk_id), "best_blk_lp": block_lp_vals.get(best_blk_id),
        "correlation": corr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs", help="Where to write results/ and logs/")
    parser.add_argument("--data-root", default="./data", help="torchvision dataset cache dir")
    parser.add_argument("--tg-root", default="./data/torchgeo", help="torchgeo dataset cache dir")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset names to run (default: all 10)")
    args = parser.parse_args()

    res_dir = os.path.join(args.output_dir, "results")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, "run_main_experiment.log"), level=logging.INFO,
                         format="%(asctime)s  %(message)s")

    cfg = MEHERABConfig()
    set_all_seeds(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading backbone: {cfg.backbone} ...")
    backbone = load_frozen_backbone(cfg.backbone, device)

    print("Loading datasets ...")
    all_datasets = load_all_datasets(args.data_root, args.tg_root, cfg.img_size, GLOBAL_SEED)
    if args.datasets:
        all_datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    results_path = os.path.join(res_dir, "all_results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"[RESUME] Loaded {len(all_results)} existing dataset results")

    failed = []
    for ds_name, (train_ds, test_ds, n_cls) in all_datasets.items():
        if ds_name in all_results:
            print(f"[RESUME] {ds_name}: already complete, skipping")
            continue
        print("\n" + "=" * 68)
        print(f"  DATASET: {ds_name}  ({n_cls} classes)")
        print("=" * 68)
        t0 = time.time()
        try:
            all_results[ds_name] = run_one_dataset(backbone, device, cfg, ds_name, train_ds, test_ds, n_cls)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [CKPT] Saved -- {ds_name} took {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[ERROR] {ds_name} FAILED: {e}")
            print(traceback.format_exc())
            failed.append((ds_name, str(e)))
            continue

    print("\n" + "=" * 68)
    print(f"  Done. Complete: {len(all_results)}  Failed: {len(failed)}")
    for ds, err in failed:
        print(f"  [FAILED] {ds}: {err}")
    print("=" * 68)


if __name__ == "__main__":
    main()
