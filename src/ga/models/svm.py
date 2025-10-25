from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, average_precision_score
import numpy as np

@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    class_weight: str | dict | None = "balanced"

def train_eval_svm(X_train, y_train, X_val, y_val, cfg: SVMConfig, metric: str = "roc_auc") -> float:
    # Binary labels handling (works also for multi-class with OvO in AUC if needed)
    clf = SVC(
        kernel=cfg.kernel, C=cfg.C, gamma=cfg.gamma,
        class_weight=cfg.class_weight, probability=True, random_state=0
    )
    clf.fit(X_train, y_train)
    # pick metric
    if metric == "roc_auc":
        # If single feature or degenerate, guard
        try:
            proba = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, proba)
        except Exception:
            pred = clf.predict(X_val)
            return balanced_accuracy_score(y_val, pred)
    elif metric == "f1":
        pred = clf.predict(X_val)
        return f1_score(y_val, pred, average="binary")  # adjust if multi-class
    elif metric == "balanced_accuracy":
        pred = clf.predict(X_val)
        return balanced_accuracy_score(y_val, pred)
    elif metric == "pr_auc":
        proba = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)
    else:
        pred = clf.predict(X_val)
        return balanced_accuracy_score(y_val, pred)
