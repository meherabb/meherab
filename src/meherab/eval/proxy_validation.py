"""Fast evaluation and Precision@k, used by the MDS proxy-validation
experiment (paper Sec. 5.2, Fig. 3, Table 2).

``evaluate_fast`` is a lower-fidelity, fixed-C variant of
``meherab.eval.probe.evaluate_with_probe`` (no inner CV) -- used here
purely for speed, since the proxy-validation experiment needs an accuracy
number for every one of ``n_corr_cands`` (50) random candidates per
dataset, and a 3-fold inner CV at that scale would dominate run time. The
headline Table 1 numbers always use the full ``evaluate_with_probe``.

Extracted verbatim from the original pipeline, Cell 16.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def evaluate_fast(X_train, y_train, X_test, y_test, pca_dim: int = 128, random_state: int = 42) -> float:
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    n_comp = min(pca_dim, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=random_state).fit(Xtr)
    Xtr_p = pca.transform(Xtr)
    Xte_p = pca.transform(Xte)
    clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=random_state).fit(
        Xtr_p, y_train
    )
    return float(accuracy_score(y_test, clf.predict(Xte_p)) * 100)


def precision_at_k(proxy_scores: np.ndarray, acc_scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k candidates by proxy score that are also top-k
    by actual accuracy (paper Sec. 3.5: P@5=0.00 on EuroSAT/RESISC45 despite
    high global Spearman rho -- MDS ranks well globally but cannot always
    pinpoint the specific top-tail candidates).
    """
    top_k_proxy = set(np.argsort(proxy_scores)[-k:])
    top_k_acc = set(np.argsort(acc_scores)[-k:])
    return len(top_k_proxy & top_k_acc) / k
