from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    average_precision_score, accuracy_score
)
from ga.profiling import get_profile

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
    X_train, y_train,
    X_val,   y_val,
    cfg: SVMConfig,
    metric: str = "balanced_accuracy",
    return_dict: bool = False,
):
    """
    Trenuje SVM i ewaluuję na walidacji.
    Jeśli return_dict=True -> zwraca słownik z metrykami i czasami,
    inaczej zwraca tylko jedną wybraną metrykę.
    """
    profile = get_profile()

    # --- trening ---
    t0 = time.perf_counter()
    clf = SVC(
        kernel=cfg.kernel,
        C=cfg.C,
        gamma=cfg.gamma,
        class_weight=cfg.class_weight,
        probability=cfg.probability,
    )
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # --- ewaluacja ---
    t1 = time.perf_counter()
    pred = clf.predict(X_val)
    score_time = time.perf_counter() - t1

    # >>>>>> PROFILOWANIE <<<<<<
    if profile is not None:
        profile.n_train_eval += 1
        profile.val_points_evaluated += X_val.shape[0]

    # --- proby dla AUC / PR-AUC ---
    try:
        proba = clf.predict_proba(X_val)[:, 1]
    except Exception:
        proba = None

    # --- metryki klasyfikacji ---
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
        # fallback – żeby metryki były zawsze zdefiniowane
        metrics["roc_auc"] = metrics["balanced_accuracy"]
        metrics["pr_auc"] = metrics["balanced_accuracy"]

    chosen = metrics.get(metric, metrics["balanced_accuracy"])

    if return_dict:
        return {
            "metric": chosen,
            "all": metrics,
            "fit_time": fit_time,
            "score_time": score_time,
        }

    # --- tryb zwracania jednej metryki ---
    return chosen


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
    profile = get_profile()
    if profile is not None:
        profile.n_fixed_eval += 1
        profile.val_points_evaluated += X_val.shape[0]

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

def eval_from_predictions(y_true, y_pred, metric: str = "balanced_accuracy") -> float:
    """
    Ewaluacja metryki, gdy mamy już gotowe predykcje (bez dodatkowego clf.predict()).
    Użyteczne w Val-GA do cache'owania predykcji.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "f1":
        return f1_score(y_true, y_pred, average="binary", zero_division=0)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    else:
        return balanced_accuracy_score(y_true, y_pred)
