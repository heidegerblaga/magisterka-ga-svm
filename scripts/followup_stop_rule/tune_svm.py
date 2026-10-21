"""Dobór (C, gamma) SVM dla KAŻDEGO podziału osobno — ta sama procedura co w badaniu V' (tune_study_svm.py):
pełny SVM na całym Train danego podziału, balanced accuracy na V tego podziału, wybór raz PRZED eksperymentami,
wspólny dla wszystkich wariantów. Test nie jest używany. Wynik: data/followup_stop_rule/split_<seed>/svm_params.json.

Dla podziału 2026 wynik powinien odtworzyć data/study/svm_params.json (sprawdzane i raportowane).
"""
from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, LEGACY_DATA, MAIN_DATASETS, SPLIT_SEEDS, dataset_dir, split_dir  # noqa: E402

GRIDS = {  # identyczne siatki jak w scripts/tune_study_svm.py
    "moons": ([1, 10, 100], [0.1, 0.5, 2, 8]),
    "circles": ([1, 10, 100], [0.1, 0.5, 2, 8]),
    "spiral": ([10, 100, 1000], [1, 4, 16, 64]),
    "classification2d": ([1, 10, 100], [0.05, 0.2, 1, 4]),
    "digits": ([1, 10, 100], [0.0005, 0.002, 0.008, 0.03]),
    "magic": ([1, 10, 100], [0.02, 0.05, 0.1, 0.3]),
}


def _fit_one(args):
    split, dataset, C, gamma = args
    d = dataset_dir(split, dataset)
    tr, va = pd.read_csv(d / "train.csv"), pd.read_csv(d / "validation.csv")
    clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", random_state=0).fit(
        tr.drop(columns="label").to_numpy(), tr["label"].to_numpy()
    )
    ba = float(balanced_accuracy_score(va["label"].to_numpy(), clf.predict(va.drop(columns="label").to_numpy())))
    return split, dataset, {"C": C, "gamma": gamma, "validation_balanced_accuracy": ba, "n_sv": int(clf.n_support_.sum())}


@click.command()
@click.option("--splits", "-s", multiple=True, type=int)
@click.option("--datasets", "-d", multiple=True)
@click.option("--workers", "-w", default=8, show_default=True)
def main(splits, datasets, workers):
    splits = list(splits) or SPLIT_SEEDS
    datasets = list(datasets) or [d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]]
    jobs = []
    for split in splits:
        for dataset in datasets:
            grid_key = DATASETS[dataset].get("svm_from", dataset)
            cs, gammas = GRIDS[grid_key]
            jobs += [(split, dataset, C, g) for C, g in itertools.product(cs, gammas)]
    results: dict[tuple[int, str], list[dict]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, dataset, row in pool.map(_fit_one, jobs):
            results.setdefault((split, dataset), []).append(row)
    legacy = json.loads((LEGACY_DATA / "svm_params.json").read_text(encoding="utf-8"))
    for split in splits:
        out_path = split_dir(split) / "svm_params.json"
        payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        for dataset in datasets:
            grid = results[(split, dataset)]
            best = max(grid, key=lambda r: r["validation_balanced_accuracy"])
            entry = {"C": best["C"], "gamma": best["gamma"], "full_svm_val_ba": best["validation_balanced_accuracy"],
                     "full_svm_n_sv": best["n_sv"], "grid": grid}
            if split == 2026 and dataset in legacy:
                entry["identical_to_legacy_choice"] = (best["C"], best["gamma"]) == (legacy[dataset]["C"], legacy[dataset]["gamma"])
            payload[dataset] = entry
            print(f"split_{split}/{dataset}: C={best['C']} gamma={best['gamma']} V-BA={best['validation_balanced_accuracy']:.4f}"
                  + (f"  [zgodne z data/study: {entry['identical_to_legacy_choice']}]" if "identical_to_legacy_choice" in entry else ""),
                  flush=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
