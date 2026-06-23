# Paper &harr; Code Map

Every equation, proposition, table, and figure in the paper, with the exact
file and function/class that implements or produces it. Use this to verify
any specific claim in under a minute.

## Equations and core method (Sec. 3)

| Paper item | Description | Implementation |
|---|---|---|
| Sec. 3.1, problem setting | Frozen ViT, final-block baseline `F_{L-1}` | `src/meherab/backbone.py::MEHERABBackbone` |
| Eq. 1 | Linear CKA similarity | `src/meherab/graph/cka.py::linear_cka` |
| Sec. 3.2, proxy set `N_prx` | Adaptive proxy sizing | `src/meherab/data/extraction.py::extract_proxy` |
| Eq. 2 | Fisher Discriminant Ratio `rho_k` | `src/meherab/graph/fdr.py::fisher_discriminant_ratio` |
| Eq. 3 | FDR deficit ratio `gamma` | `src/meherab/graph/fdr.py::fdr_deficit_ratio` |
| Sec. 3.2, node-pair selection | Homophilic-graph clustering + single-edge selection | `src/meherab/graph/homophilic_graph.py::build_homophilic_graph` |
| Eq. 4 | Adapter-Inject (AI) operation | `src/meherab/rass/transform.py::FittedRASSTransform` (`TType.AI` branch) |
| Sec. 3.3, Semantic-Compress (SC) | Top-kappa*d variance dims | `FittedRASSTransform` (`TType.SC` branch) |
| Sec. 3.3, Cross-Fuse (CF) | L2-normalize + average hyperedge endpoints | `FittedRASSTransform` (`TType.CF` branch) |
| Eq. 5 | MDS(c) = alpha*TA - (1-alpha)*MC | `src/meherab/search/mds.py::compute_mds` |
| Eq. 6 | TaskAlign | `compute_mds` (the `ta` variable) |
| Eq. 7 | ManifoldCollapse | `compute_mds` (the `mc` variable) |
| Algorithm 1 | Evolutionary search | `src/meherab/search/evolutionary.py::run_evo_search` |
| Sec. 3.6, Eq. 8 | Approximation-estimation decomposition | Theoretical; see paper Appendix G for proofs |
| Proposition 1 | FDR controls probe error | Paper Appendix G.1 (proof). Empirically exercised by `tests/test_fdr.py` |
| Proposition 2 | High homophily implies zero synergy -> universal single-edge graph | Paper Appendix G.2 (proof). Structural consequence implemented in `build_homophilic_graph`'s fallback branch |
| Proposition 3 | Estimation-error bound, small-n PEFT disadvantage | Paper Appendix G.3 (proof) |

## Baselines and evaluation (Sec. 4)

| Paper item | Implementation |
|---|---|
| LP (logistic regression, PCA(128), inner 3-fold CV) | `src/meherab/eval/probe.py::evaluate_with_probe` |
| Rand.RASS | One `RASSFactory.random_candidate()` per seed, evaluated with the same probe |
| LoRA (feature-space, rank 8) | `src/meherab/baselines/lora.py::LoRALayer` |
| BnAdapter (Houlsby, bottleneck 64) | `src/meherab/baselines/bottleneck_adapter.py::BnAdapter` |
| CLIP-Adapter (bottleneck 64, alpha=0.2) | `src/meherab/baselines/clip_adapter.py::CLIPAdapterLayer` |
| Shared PEFT training loop (Adam, cosine LR, 80 epochs) | `src/meherab/baselines/common.py::_train_peft_model` |
| Bonferroni paired t-tests (alpha=0.0125) | `src/meherab/eval/significance.py::run_significance_tests` |
| NASWOT-adapted / SynFlow-adapted proxies | `src/meherab/search/proxies.py` |
| Precision@k | `src/meherab/eval/proxy_validation.py::precision_at_k` |

## Datasets (Sec. 4, Appendix B)

| Dataset | Loader | Notes |
|---|---|---|
| Food-101, Oxford-Pets, DTD, Aircraft, Flowers102 | `src/meherab/data/datasets.py::load_all_datasets` | torchvision, official splits |
| EuroSAT | same | seeded 80/20 `random_split` (no official split) |
| Caltech-101 | same | seeded 80/20 `random_split` |
| RESISC45 | same | torchgeo built-in `split='train'/'test'` |
| PatternNet, UCMerced | same | torchgeo + `StratifiedShuffleSplit` (no built-in split; avoids class-sorted train/test mismatch) |

## Tables

| Paper table | Source data | Repo file |
|---|---|---|
| Table 1 (main results) | `all_results` per-seed accuracy lists | `results/tables/table1_main_results.csv` |
| Table 2 (proxy validation) | `corr_results` Spearman rho / P@k | `results/tables/table2_proxy_validation.csv` |
| Table 3 (backbone generalisation) | DINOv2-B / ViT-S/16 runs (notebook Cells 17b, 17e) | not included in `scripts/`; see `notebooks/meherab_pipeline.py` |
| Table A.1 (full pairwise significance) | `run_significance_tests` | `results/tables/table5_significance.csv` |
| Table B.2 / B.3 (dataset properties) | Static, from dataset metadata | paper only |
| Table D.6 (compute profile) | Per-dataset evolutionary-search wall-clock time | `results/tables/table3_compute_profile.csv` |
| Table E.7 (discovered pathways) | `best_ops` per dataset | `results/tables/table_appendixE7_pathways.csv` |

## Figures

| Paper figure | Repo file | Generating cell (notebook) |
|---|---|---|
| Fig. 1 (representational geometry) | `figures/fig1_representational_geometry.*` | Cell ~ "Figure 3: geometry" |
| Fig. 2 (main results) | `figures/fig2_main_results.*` | Cell "Figure 1: main results" |
| Fig. 3 (proxy validation) | `figures/fig3_proxy_validation.*` | Cell "Figure 2: proxy validation" |
| Fig. 4 (deficit ratio analysis) | `figures/fig4_deficit_ratio_analysis.*` | Cell "Figure 8: domain shift" |
| Fig. A.1 (significance heatmap) | `figures/figA1_significance_heatmap.*` | Cell "Figure 9" |
| Fig. C.2 (backbone comparison) | `figures/figC2_backbone_comparison.*` | Cell 17d |
| Fig. E.3 (discovered pathways) | `figures/figE3_discovered_pathways.*` | Cell 25 ("Figure 6: pathways") |
| Fig. F.4 (MDS/graph ablation) | `figures/figF4_mds_graph_ablation.*` | Cell 22a/22b |
| Fig. F.5 (FDR balance) | `figures/figF5_fdr_balance.*` | Cell 17c |
| Fig. F.6 (block FDR heatmap) | `figures/figF6_block_fdr_heatmap.*` | Cell "Figure 10" |
| Fig. F.7 (node selection pattern) | `figures/figF7_node_selection_pattern.*` | Cell "Figure 11" |
| Fig. F.8 (t-SNE, all 3 RS datasets) | `figures/figF8_tsne_remote_sensing.*` | Cell "Figure 12" |
| Fig. F.9 (search dynamics) | `figures/figF9_search_dynamics.*` | Cell 22a/22c |
| Fig. F.10 (operation analysis) | `figures/figF10_operation_analysis.*` | Cell 22d |

**Note:** one figure generated during development (`fig5_compute`, comparing
MEHERAB to NAS methods by GPU-hours/CO2) was **dropped** from the public
release because it never appears in the submitted paper and the comparison
mixes GPU architectures (V100 vs. T4) inconsistently. `table3_compute_profile.csv`
instead reports only the real, paper-consistent Appendix D numbers.

## Hyperparameters

Every numeric hyperparameter referenced anywhere above lives in exactly one
place: `src/meherab/config.py::MEHERABConfig` (mirrored in
`configs/default.yaml`). There are no hardcoded magic numbers elsewhere in
the package.
