"""Generuje zbiory do badania V' wraz z deterministycznym podziałem Train / Validation / Test.

Zbiory 2D: Moons, Circles, Spiral (generator własny), Classification2D (sklearn make_classification).
Zbiory wielowymiarowe: Digits (64 cechy, 10 klas, sklearn) oraz MAGIC Gamma Telescope (UCI, 10 cech,
plik magic04.data z repozytorium).

Wszystkie podziały są stratyfikowane i deterministyczne (stałe ziarna zapisane w meta.json).
Skalowanie (StandardScaler) — tylko dla zbiorów wielowymiarowych, WYUCZONE WYŁĄCZNIE NA TRAIN
i zastosowane do Validation i Test.

Użycie:
    python scripts/generate_study_datasets.py            # wszystkie zbiory
    python scripts/generate_study_datasets.py moons      # wybrany zbiór
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "study"
MAGIC_RAW = REPO_ROOT / "data" / "benchmarks" / "magic_gamma" / "magic04.data"

GENERATOR_SEED = 2026
SPLIT_SEED = 2026
# (train, validation, test) jako ułamki całości
SPLIT_FRACTIONS = (0.40, 0.40, 0.20)
MAGIC_SUBSAMPLE = 8000


def make_spirals(n: int, noise: float, seed: int, turns: float = 2.0):
    """Dwuramienna spirala; ramię 1 jest obrotem ramienia 0 o 180°."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    sizes = [n0, n - n0]
    X, y = [], []
    for label, (m, phase) in enumerate(zip(sizes, (0.0, np.pi))):
        t = rng.uniform(0.1, 1.0, m) * turns * 2.0 * np.pi
        r = t / (turns * 2.0 * np.pi)
        pts = np.column_stack([r * np.cos(t + phase), r * np.sin(t + phase)])
        pts += rng.normal(0.0, noise, size=pts.shape)
        X.append(pts)
        y.append(np.full(m, label))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _split(X, y, seed: int = SPLIT_SEED):
    f_train, f_val, f_test = SPLIT_FRACTIONS
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=f_train, stratify=y, random_state=seed
    )
    rel_val = f_val / (f_val + f_test)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, train_size=rel_val, stratify=y_rest, random_state=seed + 1
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def _frame(X, y, columns):
    frame = pd.DataFrame(X, columns=columns)
    frame["label"] = np.asarray(y).astype(int)
    return frame


def _write(name: str, splits, columns, meta: dict, scaler: StandardScaler | None = None):
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_name, (X, y) in zip(("train", "validation", "test"), splits):
        _frame(X, y, columns).to_csv(out / f"{split_name}.csv", index=False)
        values, counts = np.unique(y, return_counts=True)
        summary[split_name] = {
            "rows": int(len(y)),
            "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "class_proportions": {str(int(v)): round(float(c) / len(y), 4) for v, c in zip(values, counts)},
        }
    payload = {
        "name": name,
        "features": len(columns),
        "generator_seed": GENERATOR_SEED,
        "split_seed": SPLIT_SEED,
        "split_fractions_train_val_test": list(SPLIT_FRACTIONS),
        "splits": summary,
        "scaler": None
        if scaler is None
        else {
            "type": "StandardScaler",
            "fitted_on": "train_only",
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        },
        **meta,
    }
    (out / "meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {summary['train']['rows']}/{summary['validation']['rows']}/{summary['test']['rows']} -> {out}")
    return payload


def gen_moons():
    X, y = make_moons(n_samples=12000, noise=0.20, random_state=GENERATOR_SEED)
    return _write(
        "moons", _split(X, y), ["x1", "x2"],
        {"generator": "sklearn.datasets.make_moons", "params": {"n_samples": 12000, "noise": 0.20}},
    )


def gen_circles():
    X, y = make_circles(n_samples=12000, noise=0.12, factor=0.5, random_state=GENERATOR_SEED)
    return _write(
        "circles", _split(X, y), ["x1", "x2"],
        {"generator": "sklearn.datasets.make_circles", "params": {"n_samples": 12000, "noise": 0.12, "factor": 0.5}},
    )


def gen_spiral():
    X, y = make_spirals(12000, noise=0.07, seed=GENERATOR_SEED, turns=2.0)
    return _write(
        "spiral", _split(X, y), ["x1", "x2"],
        {"generator": "scripts/generate_study_datasets.make_spirals", "params": {"n_samples": 12000, "noise": 0.07, "turns": 2.0}},
    )


def gen_classification2d():
    X, y = make_classification(
        n_samples=12000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
        n_classes=2, n_clusters_per_class=2, class_sep=1.2, flip_y=0.02,
        random_state=GENERATOR_SEED,
    )
    return _write(
        "classification2d", _split(X, y), ["x1", "x2"],
        {
            "generator": "sklearn.datasets.make_classification",
            "params": {"n_samples": 12000, "n_features": 2, "n_informative": 2, "n_redundant": 0,
                       "n_clusters_per_class": 2, "class_sep": 1.2, "flip_y": 0.02},
        },
    )


def _scaled_splits(X, y):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split(X, y)
    scaler = StandardScaler().fit(X_tr)  # tylko Train
    return (scaler.transform(X_tr), y_tr), (scaler.transform(X_va), y_va), (scaler.transform(X_te), y_te), scaler


def gen_digits():
    data = load_digits()
    X, y = data.data.astype(float), data.target
    tr, va, te, scaler = _scaled_splits(X, y)
    return _write(
        "digits", (tr, va, te), [f"p{i}" for i in range(X.shape[1])],
        {"generator": "sklearn.datasets.load_digits", "params": {"classes": 10}}, scaler,
    )


def gen_magic():
    frame = pd.read_csv(MAGIC_RAW, header=None)
    X = frame.iloc[:, :-1].to_numpy(dtype=float)
    y = (frame.iloc[:, -1] == "g").astype(int).to_numpy()  # g (gamma)=1, h (hadron)=0
    # Stratyfikowana podpróbka 8000 z 19020 obserwacji: pełny zbiór czyni pojedynczy przebieg
    # (K rosnące do ~2600/klasę) kosztownym ponad miarę dla serii 20 powtórzeń × kilka wariantów.
    X, _, y, _ = train_test_split(X, y, train_size=MAGIC_SUBSAMPLE, stratify=y, random_state=GENERATOR_SEED)
    tr, va, te, scaler = _scaled_splits(X, y)
    columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
    return _write(
        "magic", (tr, va, te), columns,
        {"generator": "UCI MAGIC Gamma Telescope (magic04.data)", "params": {"label": "g=1,h=0", "subsample": MAGIC_SUBSAMPLE, "source_rows": 19020}}, scaler,
    )


GENERATORS = {
    "moons": gen_moons,
    "circles": gen_circles,
    "spiral": gen_spiral,
    "classification2d": gen_classification2d,
    "digits": gen_digits,
    "magic": gen_magic,
}


def main(argv: list[str]) -> None:
    names = argv or list(GENERATORS)
    for name in names:
        GENERATORS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
