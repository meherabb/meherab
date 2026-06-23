"""The evaluation protocol used for every accuracy number in the paper
(Sec. 4): StandardScaler -> PCA(128) -> LogisticRegression, with the
regularization strength C selected by an inner 3-fold GridSearchCV on
training data only -- there is no leakage from the test set into model
selection, and no ``max(rass, lp)`` floor is applied anywhere: MEHERAB's
raw accuracy is what gets reported, even when it underperforms.

Extracted verbatim from the original pipeline, Cell 15.
"""
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def evaluate_with_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    pca_dim: int = 128,
    probe_C_grid=(0.1, 0.5, 1.0, 2.0, 5.0),
    eval_max_iter: int = 1000,
) -> float:
    """StandardScaler + PCA(pca_dim) + LogisticRegression(C selected by 3-fold CV).

    Returns top-1 test accuracy as a percentage. This is the function used
    to produce every LP / Rand.RASS / MEHERAB number in Table 1.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)

    n_comp = min(pca_dim, Xtr_s.shape[1] - 1, Xtr_s.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42).fit(Xtr_s)
    Xtr_p = pca.transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    # Inner CV for C selection -- training data only, no test leakage.
    clf = GridSearchCV(
        LogisticRegression(max_iter=eval_max_iter, random_state=seed, solver="lbfgs"),
        param_grid={"C": list(probe_C_grid)},
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    clf.fit(Xtr_p, y_train)
    preds = clf.predict(Xte_p)
    return float(accuracy_score(y_test, preds) * 100)
