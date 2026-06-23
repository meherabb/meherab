<div align="center">

# MEHERAB

### Zero-Gradient Mid-layer Evolutionary Homophilic Exploration for Remote-sensing Adaptation with Frozen Backbones

**WACV 2027 · Applications Track · Anonymous Submission**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)

</div>

---

## Overview

Frozen vision transformers pretrained on ImageNet push discriminative information toward their final block — a structure that suits natural-image recognition but is often a poor match for the spectral and spatial regularities of remote-sensing imagery. **MEHERAB** asks a simple question: *is there a better feature already inside the frozen network, and can it be found without any gradient updates to the backbone?*

We answer this with the **FDR deficit ratio γ** — a ten-second, unlabeled-data diagnostic that predicts when a frozen mid-layer feature beats the standard final-block linear probe — and a **zero-gradient evolutionary search** that, when γ is high, discovers a 3-operation feature pathway beating both linear probing and gradient-based PEFT methods (LoRA, bottleneck adapters, CLIP-Adapter).

<p align="center">
  <img src="figures/fig2_main_results.png" width="850" alt="Main results across ten benchmarks">
</p>

## Headline results

| Dataset | γ (deficit ratio) | Linear Probe | MEHERAB | Δ vs. LP |
|---|---|---|---|---|
| EuroSAT | 2.32× | 92.52% | **95.74%** | **+3.22 pp** (p<0.001, wins all 6 baselines) |
| PatternNet | 2.46× | 95.24% | **97.66%** | **+2.42 pp** (p<0.001, wins all 6 baselines) |
| RESISC45 | 1.78× | 74.94% | **78.38%** | **+3.44 pp** (p<0.0125 vs. LP) |
| Aircraft (fine-grained, γ≈1) | 1.00× | 35.78% | 37.10% | −5.98 pp vs. LoRA — γ correctly predicts this failure |

At low label counts (n=50, EuroSAT), MEHERAB reaches 82.6% while LoRA *drops below* plain linear probing (75.2% vs. 76.4% LP) — fixed pathways discovered from unlabeled proxies don't overfit the way adapter parameters do.

Full results, all ten datasets, six baselines, three backbones, and a theoretical account (one approximation–estimation decomposition explaining all four outcome regimes) are in the paper.

## Method, in brief

1. **Task-conditional homophilic-graph construction** — cluster the 12 ViT-B/16 blocks via linear CKA similarity into 4 nodes; connect the two highest Fisher-Discriminant-Ratio nodes with a single edge (provably the only structure possible under near-unit intra-cluster CKA — Proposition 2).
2. **RASS operation search space** — three operation families (Adapter-Inject, Semantic-Compress, Cross-Fuse) over the selected node pair.
3. **Modality Discriminability Score (MDS)** — a zero-gradient proxy combining task alignment and manifold-drift penalty, used to rank candidates without ever touching labelled accuracy.
4. **Evolutionary search** — tournament selection over 300 MDS evaluations (≈3 minutes on a single T4 GPU), once per dataset, reused across all evaluation seeds.

See `docs/METHOD.md` *(added in the next repository update)* for the full walkthrough with equation references, and the paper for proofs.

## Repository contents (current)

```
meherab/
├── figures/        14 paper figures, PDF + PNG, ICLR/WACV camera-ready formatting
├── results/        all_results.json, figure_data.json, and 5 result tables (CSV)
├── notebooks/      meherab_pipeline.py -- the complete, de-identified experiment
│                   pipeline exactly as run (data loading through figure generation)
├── LICENSE
├── CITATION.cff
└── README.md
```

> **Note:** This repository is being assembled incrementally ahead of camera-ready submission. A modularized `src/meherab/` Python package, one-command reproduction scripts, configuration files, documentation, and tests are being added next. This README will be updated to reflect the final structure once that lands — nothing below this point should be treated as final until that update.

## Reproducing the results (current method)

The complete pipeline is a single, self-contained script that runs top-to-bottom on a Kaggle or Colab T4 GPU instance (15.6 GB VRAM):

```bash
python notebooks/meherab_pipeline.py
```

It performs, in order: dataset loading (10 benchmarks via `torchvision`/`torchgeo`), frozen ViT-B/16 feature extraction, homophilic-graph construction, evolutionary search, six-method evaluation across 5 seeds, MDS proxy validation, backbone-generalisation runs (DINOv2-B, ViT-S/16), few-shot evaluation, cross-domain transfer, ablations, and generation of every table and figure in the paper. All hyperparameters are centralized in the `MEHERABConfig` dataclass near the top of the file.

A more granular, script-per-experiment interface is coming in the next update.

## Results data

- `results/all_results.json` — raw per-seed accuracies for all 10 datasets × 6 methods, plus discovered RASS operations, MDS scores, and search dynamics.
- `results/figure_data.json` — fully serialized state used to regenerate every figure.
- `results/tables/` — camera-ready CSV exports of every paper table (main results, proxy validation, significance tests, discovered pathways, compute profile).

## Citation

```bibtex
@inproceedings{anonymous2027meherab,
  title     = {Zero-Gradient Mid-layer Evolutionary Homophilic Exploration
               for Remote-sensing Adaptation with Frozen Backbones},
  author    = {Anonymous},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer
               Vision (WACV)},
  year      = {2027},
  note      = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).
