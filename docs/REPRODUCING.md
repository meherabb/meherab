# Reproducing the Results

## Installation

```bash
git clone <this-repo>
cd meherab
pip install -e .
```

or with conda:

```bash
conda env create -f environment.yml
conda activate meherab
```

Requires Python >= 3.10. A CUDA GPU is strongly recommended (all paper
numbers were produced on a single Kaggle T4, 15.6 GB VRAM) but every
function will also run on CPU, just slower.

## Verifying the installation (no GPU, no dataset downloads, ~5 seconds)

The core mathematical claims (CKA self-similarity, FDR class-separation
sensitivity, the homophilic-graph single-edge fallback, RASS leakage-safety,
MDS determinism) are covered by a fast unit-test suite that needs nothing
but synthetic data:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 22 tests should pass in well under 10 seconds.

## Reproducing Table 1 (the full 10-dataset x 6-method benchmark)

```bash
python scripts/run_main_experiment.py --output-dir outputs/
```

This downloads all ten datasets on first run (five via `torchvision`, three
via `torchgeo`, EuroSAT and Caltech-101 via `torchvision` with a seeded
80/20 split -- see `docs/PAPER_MAP.md` for which loader handles which
dataset) and writes `outputs/results/all_results.json`, checkpointing after
every dataset so an interrupted run can simply be restarted (already-
finished datasets are skipped automatically).

**Expected runtime:** roughly 5-8 minutes per dataset on a T4 GPU (3-4 min
evolutionary search + 1-2 min five-seed evaluation + ~1 min MDS-correlation
validation), so 1-1.5 hours total for all ten datasets.

To run a subset (useful for a quick sanity check):

```bash
python scripts/run_main_experiment.py --output-dir outputs/ --datasets EuroSAT PatternNet
```

## Reproducing everything else (backbone generalisation, few-shot,
cross-domain transfer, ablations, every figure)

The modular `scripts/` entry point currently covers Table 1 end-to-end. The
remaining experiments (DINOv2-B / ViT-S/16 backbone generalisation, the
n=50 few-shot sweep, cross-domain operation transfer, the Food-101
ablations, and generation of all 14 figures) are still only available via
the complete original pipeline:

```bash
python notebooks/meherab_pipeline.py
```

This single script reproduces the entire paper end-to-end -- every table
and every figure -- and is the exact pipeline (de-identified for double-
blind review; see the file's docstring) used to produce every number in the
submission. It defaults to Kaggle-style paths (`/kaggle/working/...`);
change `FIG_DIR`, `LOG_DIR`, `RES_DIR`, `DATA_ROOT`, and `TG_ROOT` near the
top of the file if running elsewhere.

A fully modularized version of these remaining experiments (additional
`scripts/run_*.py` entry points mirroring `run_main_experiment.py`) is
planned but not yet part of this release.

## Verifying your run matches the paper

`results/all_results.json` and `results/tables/table1_main_results.csv` in
this repository are the exact outputs from the run reported in the paper.
After running the reproduction script yourself, the included
`results/all_results.json` is the reference to diff against -- per-seed
accuracies should match to within ordinary floating-point / library-version
noise (typically <0.1pp), since every random seed is fixed
(`meherab.config.EVAL_SEEDS`).

## Configuration

All hyperparameters are in `src/meherab/config.py::MEHERABConfig`
(mirrored in `configs/default.yaml`). To reproduce a result under a
different configuration, construct a `MEHERABConfig` with the fields you
want to change:

```python
from meherab import MEHERABConfig
cfg = MEHERABConfig(evo_gens=30, n_clusters=6)
```

See `docs/PAPER_MAP.md` for which field corresponds to which paper
equation/section, and Appendix F.6 (cluster-count ablation) / F.1 (alpha
sensitivity) in the paper for which fields the results are robust to.
