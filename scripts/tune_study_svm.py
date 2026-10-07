"""Dobór wspólnych hiperparametrów SVM (C, gamma) dla zbiorów badania.

Procedura: pełny SVM trenowany na CAŁYM Train, ocena balanced accuracy na V (pełny zbiór
walidacyjny). Wybór jest wykonywany raz, PRZED eksperymentami, i jest wspólny dla wszystkich
wariantów algorytmu — nie jest więc elementem pętli GA. Test nie jest używany.
Wynik: data/study/svm_params.json (z pełną siatką wyników).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1] / "data" / "study"
GRIDS = {
    "moons": ([1, 10, 100], [0.1, 0.5, 2, 8]),
    "circles": ([1, 10, 100], [0.1, 0.5, 2, 8]),
    "spiral": ([10, 100, 1000], [1, 4, 16, 64]),
    "classification2d": ([1, 10, 100], [0.05, 0.2, 1, 4]),
    "digits": ([1, 10, 100], [0.0005, 0.002, 0.008, 0.03]),
    "magic": ([1, 10, 100], [0.02, 0.05, 0.1, 0.3]),
}


def main() -> None:
    result = {}
    for name, (cs, gammas) in GRIDS.items():
        tr = pd.read_csv(ROOT / name / "train.csv")
        va = pd.read_csv(ROOT / name / "validation.csv")
        Xtr, ytr = tr.drop(columns="label").to_numpy(), tr["label"].to_numpy()
        Xva, yva = va.drop(columns="label").to_numpy(), va["label"].to_numpy()
        grid = []
        for C, g in itertools.product(cs, gammas):
            clf = SVC(kernel="rbf", C=C, gamma=g, class_weight="balanced", random_state=0).fit(Xtr, ytr)
            ba = float(balanced_accuracy_score(yva, clf.predict(Xva)))
            grid.append({"C": C, "gamma": g, "validation_balanced_accuracy": ba, "n_sv": int(clf.n_support_.sum())})
            print(name, C, g, round(ba, 4), int(clf.n_support_.sum()), flush=True)
        best = max(grid, key=lambda r: r["validation_balanced_accuracy"])
        result[name] = {"C": best["C"], "gamma": best["gamma"], "full_svm_val_ba": best["validation_balanced_accuracy"],
                        "full_svm_n_sv": best["n_sv"], "grid": grid}
    (ROOT / "svm_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in result.items():
        print("WYBRANO", k, v["C"], v["gamma"], round(v["full_svm_val_ba"], 4), v["full_svm_n_sv"])


if __name__ == "__main__":
    main()
