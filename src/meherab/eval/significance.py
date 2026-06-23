"""Paired t-tests with Bonferroni correction (paper Sec. 4 "Evaluation",
Table 1 significance markers, Appendix A).

Four planned comparisons per dataset (MEHERAB vs. LP, Rand.RASS, LoRA,
BnAdapter) at Bonferroni-corrected alpha = 0.05/4 = 0.0125. CLIP-Adapter is
evaluated post-hoc at the uncorrected alpha=0.05 in the paper text, but is
included here as a fifth optional comparison for completeness.

Extracted and refactored (returns structured records instead of printing /
writing a CSV directly) from the original pipeline, Cell 24.
"""
from typing import Dict, List

import numpy as np
from scipy.stats import ttest_rel, t as t_dist

N_COMPARISONS = 4
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS  # 0.0125


def _significance_marker(p_val: float, bonferroni_alpha: float) -> str:
    if p_val < bonferroni_alpha / 3:
        return "***"
    if p_val < bonferroni_alpha:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


def confidence_interval(values: np.ndarray, level: float = 0.95):
    """Two-sided t-distribution confidence interval for the mean."""
    values = np.asarray(values)
    se = values.std(ddof=1) / np.sqrt(len(values))
    tc = t_dist.ppf(1 - (1 - level) / 2, len(values) - 1)
    return float(values.mean() - tc * se), float(values.mean() + tc * se)


def run_significance_tests(
    all_results: Dict[str, dict],
    baseline_keys=(("Linear Probe", "lp"), ("Random RASS", "rr"), ("LoRA", "lora"), ("Adapter", "adapter")),
    bonferroni_alpha: float = BONFERRONI_ALPHA,
) -> List[dict]:
    """Paired t-test of MEHERAB vs. each baseline, per dataset.

    Parameters
    ----------
    all_results : dict[str, dict]
        Same structure as ``results/all_results.json`` -- one entry per
        dataset, each containing per-seed accuracy lists under keys
        ``'meherab'``, ``'lp'``, ``'rr'``, ``'lora'``, ``'adapter'``.

    Returns
    -------
    list[dict]
        One row per (dataset, baseline) pair: Dataset, Baseline, MEHERAB
        mean, Baseline mean, Delta, t-statistic, p-value, significance
        marker. Mirrors ``results/tables/table5_significance.csv``.
    """
    rows = []
    for ds, res in all_results.items():
        mhb = np.array(res["meherab"])
        for bname, bkey in baseline_keys:
            base = np.array(res[bkey])
            delta = mhb.mean() - base.mean()
            t_s, p_val = ttest_rel(mhb, base)
            sig = _significance_marker(p_val, bonferroni_alpha)
            rows.append(
                {
                    "Dataset": ds,
                    "Baseline": bname,
                    "MEHERAB": round(float(mhb.mean()), 2),
                    "Baseline Mean": round(float(base.mean()), 2),
                    "Delta": round(float(delta), 2),
                    "t": round(float(t_s), 3),
                    "p": round(float(p_val), 5),
                    "Sig": sig,
                }
            )
    return rows
