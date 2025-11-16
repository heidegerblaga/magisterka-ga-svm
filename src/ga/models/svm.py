from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    average_precision_score, accuracy_score
)
import numpy as np
import time

@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    class_weight: str | dict | None = "balanced"
    probability: bool = True


def train_eval_svm(
    X_train, y_train, X_val, y_val,
    cfg: SVMConfig,
    metric: str = "roc_auc",
    return_dict: bool = False
):
    """Trenuje i ewaluje SVM, zwracając:
       - jedną metrykę (default)
       - lub pełne metryki + czasy jeśli return_dict=True
    """

    clf = SVC(
        kernel=cfg.kernel,
        C=cfg.C,
        gamma=cfg.gamma,
        class_weight=cfg.class_weight,
        probability=cfg.probability,
        random_state=0,
    )

    # --- czas trenowania ---
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # --- czas predykcji ---
    t1 = time.perf_counter()
    pred = clf.predict(X_val)
    score_time = time.perf_counter() - t1

    # --- proby dla AUC / PR-AUC ---
    try:
        proba = clf.predict_proba(X_val)[:, 1]
    except Exception:
        proba = None

    # --- metryki ---
    metrics = {
        "accuracy": accuracy_score(y_val, pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "f1": f1_score(y_val, pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_val, pred, average="macro"),
    }

    # AUC/PR-AUC (jeśli dostępne probabilities=True)
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_val, proba)
        metrics["pr_auc"] = average_precision_score(y_val, proba)
    else:
        metrics["roc_auc"] = metrics["balanced_accuracy"]
        metrics["pr_auc"] = metrics["balanced_accuracy"]

    chosen = metrics.get(metric, metrics["balanced_accuracy"])

    # --- tryb zwracania pełnych danych ---
    if return_dict:
        return {
            "metric": metrics.get(metric, metrics["balanced_accuracy"]),
            "all": metrics,
            "fit_time": fit_time,
            "score_time": score_time,
        }
    return chosen

    # --- tryb zwracania jednej metryki ---
    return metrics.get(metric, metrics["balanced_accuracy"])



from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def fit_svm_model(X_train, y_train, cfg: SVMConfig):
    """Trenuje SVM i zwraca gotowy, ustalony model (bez mierzenia metryk)."""
    clf = SVC(
        kernel=cfg.kernel,
        C=cfg.C,
        gamma=cfg.gamma,
        class_weight=cfg.class_weight,
        probability=cfg.probability,
        random_state=0,
    )
    clf.fit(X_train, y_train)
    return clf


def eval_fixed_svm(clf, X_val, y_val, metric: str = "balanced_accuracy") -> float:
    """Ewaluacja ustalonego SVM na podzbiorze walidacji."""
    pred = clf.predict(X_val)

    if metric == "accuracy":
        return accuracy_score(y_val, pred)
    elif metric == "f1":
        return f1_score(y_val, pred, average="binary", zero_division=0)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y_val, pred)
    else:
        # fallback – balanced_accuracy
        return balanced_accuracy_score(y_val, pred)
