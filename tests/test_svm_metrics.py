import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ga.models.svm import SVMConfig, compute_prediction_metrics, fit_svm_model, train_eval_svm


def test_compute_prediction_metrics_binary_precision_recall_f1():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = compute_prediction_metrics(y_true, y_pred)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 6) == round(2 / 3, 6)


def test_compute_prediction_metrics_multiclass_falls_back_to_macro():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])

    metrics = compute_prediction_metrics(y_true, y_pred)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_train_eval_svm_reports_num_support_vectors():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    X_val = X_train.copy()
    y_val = y_train.copy()

    result = train_eval_svm(
        X_train,
        y_train,
        X_val,
        y_val,
        SVMConfig(kernel="linear", C=1.0, gamma="scale"),
        metric="balanced_accuracy",
        return_dict=True,
    )
    clf, meta = fit_svm_model(X_train, y_train, SVMConfig(kernel="linear"), return_metadata=True)

    assert result["num_support_vectors"] > 0
    assert result["num_support_vectors"] == meta["num_support_vectors"]
