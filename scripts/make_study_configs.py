"""Generuje szablony konfiguracji badania V' (configs/study/<zbiór>/<wariant>.yaml) i specyfikację badania.

Warianty doboru V' (wszystkie: ten sam algorytm T', te same parametry GA, ta sama polityka K):
  evo_mem      — Val-GA (ewolucyjny dobór), V' pamięta wcześniejsze elementy      [wariant główny]
  evo_nomem    — Val-GA, bez pamiętania (V' regenerowane przy tym samym harmonogramie rozmiaru)
  rand_fixed   — losowe stałe V' o rozmiarze = średni końcowy rozmiar V' z evo_mem   [baza losowa]
  rand_growth  — losowy dobór nowych elementów, ten sam harmonogram rozszerzeń, z pamięcią
  heur_mem     — heurystyka „błędy modelu, potem mały margines" (pełne skanowanie puli), z pamięcią

Użycie:  python scripts/make_study_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "study"
STUDY_PARAMS = REPO_ROOT / "data" / "study" / "svm_params.json"

SEEDS = list(range(42, 62))  # 20 seedów, identyczne dla wszystkich wariantów i zbiorów

VARIANTS = {
    "evo_mem": {"selection": "evolutionary", "memory": True, "mode": "growing"},
    "evo_nomem": {"selection": "evolutionary", "memory": False, "mode": "growing"},
    "rand_fixed": {"selection": "random", "memory": True, "mode": "fixed", "fixed_size": None},
    "rand_growth": {"selection": "random", "memory": True, "mode": "growing"},
    "heur_mem": {"selection": "heuristic", "memory": True, "mode": "growing"},
}

GA = {
    "population_size": 30,
    "generations": 30,
    "tournament_k": 3,
    "crossover_rate": 0.8,
    "mutation_rate": 0.05,
    "elitism": 2,
    "early_stopping": {"patience": 3, "min_delta": 0.0001},
}
VAL_GA = {
    "population_size": 30,
    "generations": 20,
    "tournament_k": 3,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "elitism": 2,
}

# dataset -> (nazwa wyświetlana, ścieżki danych, header, svm, k_start)
DATASETS = {
    "chessboard": {
        "label": "Szachownica",
        "dir": "data/raw",
        "header": False,
        "svm": {"C": 10.0, "gamma": 0.002},
        "k_start": 10,
    },
    "moons": {"label": "Moons", "dir": "data/study/moons", "header": True, "k_start": 10},
    "circles": {"label": "Circles", "dir": "data/study/circles", "header": True, "k_start": 10},
    "spiral": {"label": "Spiral", "dir": "data/study/spiral", "header": True, "k_start": 10},
    "classification2d": {
        "label": "Classification 2D", "dir": "data/study/classification2d", "header": True, "k_start": 10,
    },
    "digits": {"label": "Digits (64 cechy)", "dir": "data/study/digits", "header": True, "k_start": 5},
    "magic": {"label": "MAGIC Gamma (UCI)", "dir": "data/study/magic", "header": True, "k_start": 10},
}


def build_config(dataset: str, variant: str, svm_params: dict) -> dict:
    ds = DATASETS[dataset]
    svm = ds.get("svm") or {"C": svm_params[dataset]["C"], "gamma": svm_params[dataset]["gamma"]}
    vprime = dict(VARIANTS[variant])
    return {
        "seed": 42,
        "experiment_name": f"study_{dataset}_{variant}",
        "dataset_name": ds["label"],
        "algorithm_mode": "dagasvm_phase",
        "artifacts_root": f"artifacts/study/{dataset}/{variant}",
        "data": {
            "train_csv": f"{ds['dir']}/train.csv",
            "validation_csv": f"{ds['dir']}/validation.csv",
            "test_csv": f"{ds['dir']}/test.csv",
            "target": "label",
            "stratify": True,
            "header": bool(ds["header"]),
            "target_candidates": ["target", "y", "class"],
            "max_rows": 0,
        },
        "svm": {
            "kernel": "rbf",
            "C": float(svm["C"]),
            "gamma": float(svm["gamma"]),
            "class_weight": "balanced",
            "probability": False,
            "standardize": False,
        },
        "fitness": {"metric": "balanced_accuracy", "class_imbalance_penalty": 0.0},
        "ga": dict(GA, early_stopping=dict(GA["early_stopping"])),
        "daga": {
            "k_start": ds["k_start"],
            "TSV": 0.25,
            "Ts": 3,
            "phase_min_delta": 0.0005,
            "sv_ratio_stability_epsilon": 0.02,
            "sv_ratio_saturation_margin": 0.15,
        },
        "vprime": vprime,
        "val_ga": dict(VAL_GA),
        "observation": {"full_validation": True, "test": True},
    }


def main() -> None:
    svm_params = json.loads(STUDY_PARAMS.read_text(encoding="utf-8")) if STUDY_PARAMS.exists() else {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            cfg = build_config(dataset, variant, svm_params)
            out = CONFIG_ROOT / dataset / f"{variant}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    spec = {
        "seeds": SEEDS,
        "datasets": {k: {"label": v["label"]} for k, v in DATASETS.items()},
        "variants": VARIANTS,
        "fixed_size_source_variant": "evo_mem",
    }
    (CONFIG_ROOT / "study_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"zapisano {len(DATASETS) * len(VARIANTS)} szablonów w {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
