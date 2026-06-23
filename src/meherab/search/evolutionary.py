"""Algorithm 1: MEHERAB Evolutionary Search (paper Sec. 3.5).

Tournament-selection evolutionary algorithm guided entirely by MDS -- no
gradients, no training, no labelled accuracy evaluation at any point. This
is what makes MEHERAB's search "zero-gradient": every one of the
``evo_pop * evo_gens`` (20 x 15 = 300) candidate evaluations is a single
forward pass on the unlabeled proxy set.

Per-dataset wall-clock time is 145.6-228.4s on a single T4 GPU (paper
Appendix D); the discovered candidate is reused across all five evaluation
seeds without any retraining.

Extracted and refactored (explicit parameters instead of a module-level
``cfg`` global) from the original pipeline, Cell 13.
"""
import random
import time

import numpy as np

from .mds import compute_mds


def tournament_select(pop, k):
    return max(random.sample(pop, min(k, len(pop))), key=lambda c: c.mds_score)


def run_evo_search(
    rass_factory,
    layer_feats,
    pretrain_ref,
    labels,
    hg,
    baseline_fdr,
    evo_pop: int = 20,
    evo_gens: int = 15,
    evo_elite: int = 5,
    evo_tournament: int = 3,
    mds_alpha: float = 0.5,
    verbose: bool = True,
):
    """Run the MDS-guided evolutionary search (Algorithm 1).

    Returns
    -------
    (best_candidate, gen_best_history, gen_mean_history)
    """
    pop = [rass_factory.random_candidate() for _ in range(evo_pop)]
    for c in pop:
        c.mds_score = compute_mds(c, layer_feats, pretrain_ref, labels, hg, baseline_fdr, alpha=mds_alpha)

    gen_best, gen_mean = [], []
    t0 = time.time()

    if verbose:
        print(f'  {"Gen":>4}  {"Best MDS":>10}  {"Mean MDS":>10}  {"Time":>7}')
        print("  " + "-" * 38)

    for gen in range(evo_gens):
        pop.sort(key=lambda c: c.mds_score, reverse=True)
        gb = pop[0].mds_score
        gm = float(np.mean([c.mds_score for c in pop]))
        gen_best.append(gb)
        gen_mean.append(gm)

        if verbose:
            print(f"  {gen + 1:>4}  {gb:>10.4f}  {gm:>10.4f}  {time.time() - t0:>5.1f}s")

        elites = pop[:evo_elite]
        new_pop = list(elites)
        while len(new_pop) < evo_pop:
            p1 = tournament_select(pop, evo_tournament)
            p2 = tournament_select(pop, evo_tournament)
            child = rass_factory.crossover(p1, p2)
            child = rass_factory.mutate(child)
            child.mds_score = compute_mds(
                child, layer_feats, pretrain_ref, labels, hg, baseline_fdr, alpha=mds_alpha
            )
            new_pop.append(child)
        pop = new_pop

    pop.sort(key=lambda c: c.mds_score, reverse=True)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Best MDS  : {pop[0].mds_score:.4f}")
        print(f"  Best ops  : {[str(o) for o in pop[0].ops]}")
        print(f"  Time      : {elapsed:.1f}s  ({elapsed / 3600:.4f} GPU-hours)")

    return pop[0], gen_best, gen_mean
