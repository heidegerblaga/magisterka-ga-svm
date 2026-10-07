from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    class_weight: str | dict | None = "balanced"
    probability: bool = True
    standardize: bool = True


def _metric_average(y_true) -> str:
    labels = np.unique(np.asarray(y_true))
    return "binary" if labels.size <= 2 else "macro"


def _safe_probability_scores(clf, X, y_true):
    try:
        proba = clf.predict_proba(X)
    except Exception:
        return None

    labels = np.unique(np.asarray(y_true))
    if labels.size <= 2 and proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    if labels.size > 2:
        return proba
    return None


def compute_prediction_metrics(y_true, y_pred, proba=None) -> dict[str, float]:
    average = _metric_average(y_true)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        if proba is not None:
            if np.asarray(proba).ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
                metrics["pr_auc"] = float(average_precision_score(y_true, proba))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
                )
                metrics["pr_auc"] = metrics["balanced_accuracy"]
        else:
            metrics["roc_auc"] = metrics["balanced_accuracy"]
            metrics["pr_auc"] = metrics["balanced_accuracy"]
    except Exception:
        metrics["roc_auc"] = metrics["balanced_accuracy"]
        metrics["pr_auc"] = metrics["balanced_accuracy"]

    return metrics


def _unwrap_svc(clf):
    if isinstance(clf, SVC):
        return clf
    if isinstance(clf, Pipeline):
        for _, step in reversed(clf.steps):
            if isinstance(step, SVC):
                return step
    return None


def _support_vector_count(clf) -> int:
    svc = _unwrap_svc(clf)
    if svc is None:
        return 0
    return int(np.sum(getattr(svc, "n_support_", 0)))


def fit_svm_model(X_train, y_train, cfg: SVMConfig, return_metadata: bool = False):
    """Trenuje SVM i opcjonalnie zwraca metadane treningu."""
    svc = SVC(
        kernel=cfg.kernel,
        C=cfg.C,
        gamma=cfg.gamma,
        class_weight=cfg.class_weight,
        probability=cfg.probability,
        random_state=0,
    )
    clf = (
        Pipeline([("scaler", StandardScaler()), ("svc", svc)])
        if cfg.standardize
        else svc
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    metadata = {
        "fit_time": float(fit_time),
        "num_support_vectors": _support_vector_count(clf),
    }

    if return_metadata:
        return clf, metadata
    return clf


_FAST_METRICS = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}


def evaluate_fitted_svm(
    clf,
    X_eval,
    y_eval,
    metric: str = "balanced_accuracy",
    all_metrics: bool = True,
) -> dict:
    """Liczy metryki dla już wytrenowanego modelu.

    `all_metrics=False` liczy wyłącznie metrykę `metric` (dla accuracy/balanced_accuracy),
    z identyczną wartością jak ścieżka pełna; używane w pętli GA, gdzie z wyniku
    potrzebna jest tylko wartość fitnessu.
    """
    t1 = time.perf_counter()
    pred = clf.predict(X_eval)
    score_time = time.perf_counter() - t1
    if not all_metrics and metric in _FAST_METRICS:
        value = float(_FAST_METRICS[metric](y_eval, pred))
        return {
            "metric": value,
            "all": {metric: value},
            "score_time": float(score_time),
            "num_support_vectors": _support_vector_count(clf),
        }
    proba = _safe_probability_scores(clf, X_eval, y_eval)
    metrics = compute_prediction_metrics(y_eval, pred, proba=proba)

    return {
        "metric": float(metrics.get(metric, metrics["balanced_accuracy"])),
        "all": metrics,
        "score_time": float(score_time),
        "num_support_vectors": _support_vector_count(clf),
    }


def train_eval_svm(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: SVMConfig,
    metric: str = "roc_auc",
    return_dict: bool = False,
    return_model: bool = False,
    all_metrics: bool = True,
):
    """Trenuje i ewaluje SVM, zwracając metrykę albo pełny raport."""
    clf, fit_meta = fit_svm_model(X_train, y_train, cfg, return_metadata=True)
    evaluation = evaluate_fitted_svm(clf, X_val, y_val, metric=metric, all_metrics=all_metrics)
    result = {
        "metric": evaluation["metric"],
        "all": evaluation["all"],
        "fit_time": fit_meta["fit_time"],
        "score_time": evaluation["score_time"],
        "num_support_vectors": fit_meta["num_support_vectors"],
    }

    if return_model:
        result["model"] = clf

    if return_dict:
        return result
    return result["metric"]


def eval_fixed_svm(clf, X_val, y_val, metric: str = "balanced_accuracy") -> float:
    """Ewaluacja ustalonego SVM na podzbiorze walidacji."""
    return float(evaluate_fitted_svm(clf, X_val, y_val, metric=metric)["metric"])
