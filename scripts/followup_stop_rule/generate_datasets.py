"""Generuje podziały danych badania `followup_stop_rule`: data/followup_stop_rule/split_<seed>/<zbiór>/.

Pula obserwacji każdego zbioru jest STAŁA (ziarno generatora 2026); zmienia się wyłącznie podział
Train/Validation/Test (stratyfikowany 40/40/20, `split_seed` i `split_seed+1` jak w badaniu V').
Skalowanie StandardScaler (Digits, MAGIC) jest wyuczone wyłącznie na Train danego podziału.
Podział 2026 odtwarza dokładnie data/study (sprawdzane skrótami SHA-256 plików).

Użycie:
    python scripts/followup_stop_rule/generate_datasets.py                    # wszystkie podziały, zbiory główne
    python scripts/followup_stop_rule/generate_datasets.py --splits 2027 -d moons
    python scripts/followup_stop_rule/generate_datasets.py -d magic_full      # pełny MAGIC (19 020 obserwacji)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASETS, GENERATOR_SEED, LEGACY_DATA, MAIN_DATASETS, REPO_ROOT, SPLIT_SEEDS, dataset_dir,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_study_datasets import MAGIC_RAW, MAGIC_SUBSAMPLE, SPLIT_FRACTIONS, make_spirals  # noqa: E402


def pool(dataset: str):
    """Stała pula obserwacji (X, y, kolumny, meta) — niezależna od podziału."""
    if dataset == "moons":
        X, y = make_moons(n_samples=12000, noise=0.20, random_state=GENERATOR_SEED)
        return X, y, ["x1", "x2"], {"generator": "sklearn.datasets.make_moons", "params": {"n_samples": 12000, "noise": 0.20}}, False
    if dataset == "circles":
        X, y = make_circles(n_samples=12000, noise=0.12, factor=0.5, random_state=GENERATOR_SEED)
        return X, y, ["x1", "x2"], {"generator": "sklearn.datasets.make_circles",
                                    "params": {"n_samples": 12000, "noise": 0.12, "factor": 0.5}}, False
    if dataset == "spiral":
        X, y = make_spirals(12000, noise=0.07, seed=GENERATOR_SEED, turns=2.0)
        return X, y, ["x1", "x2"], {"generator": "scripts/generate_study_datasets.make_spirals",
                                    "params": {"n_samples": 12000, "noise": 0.07, "turns": 2.0}}, False
    if dataset == "classification2d":
        X, y = make_classification(
            n_samples=12000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
            n_classes=2, n_clusters_per_class=2, class_sep=1.2, flip_y=0.02, random_state=GENERATOR_SEED,
        )
        return X, y, ["x1", "x2"], {"generator": "sklearn.datasets.make_classification",
                                    "params": {"n_samples": 12000, "n_features": 2, "n_clusters_per_class": 2,
                                               "class_sep": 1.2, "flip_y": 0.02}}, False
    if dataset == "digits":
        data = load_digits()
        X, y = data.data.astype(float), data.target
        return X, y, [f"p{i}" for i in range(X.shape[1])], {"generator": "sklearn.datasets.load_digits",
                                                            "params": {"classes": 10}}, True
    if dataset in {"magic", "magic_full"}:
        frame = pd.read_csv(MAGIC_RAW, header=None)
        X = frame.iloc[:, :-1].to_numpy(dtype=float)
        y = (frame.iloc[:, -1] == "g").astype(int).to_numpy()
        meta = {"generator": "UCI MAGIC Gamma Telescope (magic04.data)", "params": {"label": "g=1,h=0", "source_rows": 19020}}
        if dataset == "magic":
            X, _, y, _ = train_test_split(X, y, train_size=MAGIC_SUBSAMPLE, stratify=y, random_state=GENERATOR_SEED)
            meta["params"]["subsample"] = MAGIC_SUBSAMPLE
        else:
            meta["params"]["subsample"] = None  # pełny zbiór
        cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
        return X, y, cols, meta, True
    raise KeyError(dataset)


def split_pool(X, y, seed: int):
    f_train, f_val, f_test = SPLIT_FRACTIONS
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=f_train, stratify=y, random_state=seed)
    rel_val = f_val / (f_val + f_test)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=rel_val, stratify=y_rest, random_state=seed + 1)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split(dataset: str, split: int) -> dict:
    X, y, cols, meta, scale = pool(dataset)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_pool(X, y, split)
    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_tr)  # tylko Train
        X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    out = dataset_dir(split, dataset)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (Xs, ys) in zip(("train", "validation", "test"), ((X_tr, y_tr), (X_va, y_va), (X_te, y_te))):
        frame = pd.DataFrame(Xs, columns=cols)
        frame["label"] = np.asarray(ys).astype(int)
        frame.to_csv(out / f"{name}.csv", index=False)
        values, counts = np.unique(ys, return_counts=True)
        summary[name] = {
            "rows": int(len(ys)),
            "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "class_proportions": {str(int(v)): round(float(c) / len(ys), 4) for v, c in zip(values, counts)},
        }
    payload = {
        "name": dataset, "split_seed": split, "generator_seed": GENERATOR_SEED, "features": len(cols),
        "split_fractions_train_val_test": list(SPLIT_FRACTIONS), "splits": summary,
        "pool_rows": int(len(y)),
        "scaler": None if scaler is None else {
            "type": "StandardScaler", "fitted_on": "train_only",
            "mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scaler.scale_],
        },
        "sha256": {n: sha256(out / f"{n}.csv") for n in ("train", "validation", "test")},
        **meta,
    }
    # test spójności z badaniem V' dla podziału bazowego
    legacy = LEGACY_DATA / dataset
    if split == 2026 and legacy.exists() and dataset != "magic_full":
        same = all(sha256(legacy / f"{n}.csv") == payload["sha256"][n] for n in ("train", "validation", "test"))
        payload["identical_to_legacy_data_study"] = bool(same)
    (out / "meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    flag = payload.get("identical_to_legacy_data_study")
    print(f"split_{split}/{dataset}: {summary['train']['rows']}/{summary['validation']['rows']}/{summary['test']['rows']}"
          + ("" if flag is None else f"  [identyczny z data/study: {flag}]"), flush=True)
    return payload


@click.command()
@click.option("--splits", "-s", multiple=True, type=int, help="Ziarna podziału (domyślnie wszystkie).")
@click.option("--datasets", "-d", multiple=True, help="Zbiory (domyślnie główne bez Szachownicy).")
def main(splits, datasets):
    splits = list(splits) or SPLIT_SEEDS
    datasets = list(datasets) or [d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]]
    for split in splits:
        for dataset in datasets:
            write_split(dataset, split)


if __name__ == "__main__":
    main()
