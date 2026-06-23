# Method Walkthrough

This document walks through MEHERAB's four pieces in the order they execute,
matching paper Sec. 3. For formal proofs, see paper Appendix G; this is the
implementation-level explanation.

## 0. Setup

A frozen pretrained ViT `f` with `L=12` transformer blocks and embedding
dimension `d=768` (ViT-B/16). For each block `l`, we extract the CLS token
over `N` examples: `F_l in R^{N x d}`. The final-block matrix `F_{L-1}` is
the linear-probing baseline. **No gradient ever touches `f`,** anywhere in
the pipeline -- including during the PEFT baseline training, which only
updates a small adapter head on top of frozen features.

## 1. Task-conditional block-pair selection (`meherab.graph`)

**Why it's needed:** ImageNet pretraining pushes discriminative structure
toward the final block -- a good fit for natural images, often a poor fit
for remote-sensing imagery, where spectral/spatial regularities the network
never saw during pretraining may be better captured by an earlier block.

**What it does:**
1. Draw `N_prx = max(256, min(1024, 8C))` unlabeled proxy samples (`C` =
   class count) -- `meherab.data.extraction.extract_proxy`.
2. Compute the `L x L` linear-CKA similarity matrix across all blocks
   (`meherab.graph.cka.compute_cka_matrix`).
3. Average-linkage hierarchical clustering on `(1 - CKA)`, cut to `K=4`
   clusters -- `meherab.graph.homophilic_graph.build_homophilic_graph`.
4. Compute the per-node Fisher Discriminant Ratio under proxy labels
   (`meherab.graph.fdr.fisher_discriminant_ratio`).
5. Connect the two highest-FDR nodes with a single edge.

**The key finding (Proposition 2):** step 5 *never* produces a genuine
multi-node hyperedge on any of the ten benchmarks. This isn't a search
failure -- it's a direct algebraic consequence of near-unit intra-cluster
CKA: if `CKA(X, Y) = 1`, then `Y = XT` for some matrix `T`, so any linear
score on `[X || Y]` equals the same score on `X` alone. Concatenating two
near-identical representations adds no information a linear probe can use.
`build_homophilic_graph` tries every 2- and 3-node subset first (the
`hyperedge_delta` gain check) and only falls back to the single best pair
when no subset clears the threshold -- which is what happens, every time.

**The deficit ratio `gamma`** (`meherab.graph.fdr.fdr_deficit_ratio`) is the
ratio of the best node's FDR to the final block's FDR. `gamma > 1.5`
predicts MEHERAB is worth running; `gamma ~= 1` means PEFT is the better
choice (paper Sec. 6.1, Table 3).

## 2. RASS operations (`meherab.rass`)

Three operation families, defined over the selected node/edge
(`meherab.rass.operations.RASSFactory` enumerates all of them for a given
graph):

* **Adapter-Inject (AI)** -- project onto the top-`r` principal components
  of the node's proxy features and blend the reconstruction back in at a
  fixed scale `s=0.3`. Dominates empirically: present in all ten discovered
  pathways, 2+ operations in nine of them.
* **Semantic-Compress (SC)** -- keep the top `kappa*d` highest-variance
  feature dimensions. Only generated for multi-member cluster nodes.
* **Cross-Fuse (CF)** -- L2-normalize and average the two endpoints of a
  hyperedge. Appears in exactly one paper pathway (PatternNet) -- consistent
  with Proposition 2's near-zero cross-node synergy prediction.

A candidate is a 3-tuple of operations drawn without replacement.
`meherab.rass.transform.FittedRASSTransform` fits every operation's
parameters (PCA bases, variance-ranked indices) **once on training data**
and freezes them before applying to test data -- this is the mechanism that
prevents train/test leakage (verified directly in
`tests/test_rass_transform.py`).

MEHERAB features are `concat(F_{L-1}, RASS_output)` -- the final-block
features are always retained, so RASS only ever adds structure on top of
the LP baseline. No `max(rass, lp)` floor is applied anywhere: the reported
MEHERAB number is the raw value of this concatenation, even on the two
datasets (Aircraft, UCMerced near-ceiling) where it underperforms LP or
loses to LoRA.

## 3. Modality Discriminability Score (`meherab.search.mds`)

`MDS(c) = alpha * TaskAlign(c) - (1-alpha) * ManifoldCollapse(c)`, with
`alpha=0.5`. Both terms are computed **only on the unlabeled proxy set** --
no label leakage, no gradient, no backbone modification.

* **TaskAlign** = `tanh(FDR(adapted) / (FDR(final-block) + eps))` --
  saturates at 1 as the transformed features exceed final-block
  discriminability. The relative normalization (against the baseline FDR,
  not a hardcoded constant) is what makes MDS scores comparable across
  datasets with very different absolute FDR scales.
* **ManifoldCollapse** = `1 - CKA(pretrained_ref, adapted)` -- penalizes
  candidates that distort the feature manifold rather than exposing genuine
  structure already present in it.

## 4. Evolutionary search (`meherab.search.evolutionary`)

Algorithm 1: tournament-selection EA over `O^3` candidates, guided entirely
by MDS.

* Population 20, 15 generations -> 300 total MDS evaluations.
* Each generation: keep the top-5 elites, fill the rest via tournament
  selection (size 3) + crossover + single-op mutation (`p_m=0.3`).
* Wall-clock: 145.6-228.4s on a single T4 GPU. Runs once per dataset; the
  discovered candidate is reused across all 5 evaluation seeds without
  retraining.

**An honest caveat the paper makes explicit:** on EuroSAT and RESISC45,
Precision@5 is 0.00 -- MDS ranks the *global* distribution of candidates
reliably (Spearman `rho >= 0.5`, `p < 0.001`) but cannot reliably identify
the specific top-5 candidates on those two datasets. On those datasets, the
evolutionary loop functions closer to a structured random sampler than a
precision optimizer -- but the *routing gain* over LP (selecting the right
mid-layer node at all) remains real and significant either way. The paper's
generalizable contribution is the operation-set design and the
gamma-guided block routing, not the tournament dynamics themselves.

## 5. Evaluation (`meherab.eval`)

Every accuracy number: `StandardScaler -> PCA(128) -> LogisticRegression`,
with the regularization strength `C` selected by an inner 3-fold
`GridSearchCV` on training data only (`meherab.eval.probe.evaluate_with_probe`).
Five independent `StratifiedShuffleSplit` seeds give genuine replication
across data splits -- not a hyperparameter sweep relabelled as seeds.
Significance is assessed by paired t-tests with Bonferroni correction at
`alpha = 0.05/4 = 0.0125` for the four planned comparisons
(`meherab.eval.significance.run_significance_tests`).
