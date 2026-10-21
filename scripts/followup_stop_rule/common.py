"""Wspólna specyfikacja badania `followup_stop_rule` (jedno źródło prawdy dla wszystkich skryptów).

Badanie jest ODDZIELONE od badania V' (data/study, artifacts/study, reports/vprime_study): własne dane
(data/followup_stop_rule/split_<seed>/), konfiguracje (configs/followup_stop_rule/), artefakty
(artifacts/followup_stop_rule/split_<seed>/), tabele/wykresy (reports/followup_stop_rule/) i raport PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

STUDY = "followup_stop_rule"
DATA_ROOT = REPO_ROOT / "data" / STUDY
CONFIG_ROOT = REPO_ROOT / "configs" / STUDY
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / STUDY
REPORT_ROOT = REPO_ROOT / "reports" / STUDY
TABLES = REPORT_ROOT / "tables"
FIGS = REPORT_ROOT / "figures"
OUT_PDF = REPO_ROOT / "output" / "pdf" / "raport_followup_regula_stopu.pdf"

# --- poprzednie badanie V' (tylko do odczytu; służy jako stan „przed zmianą reguły”) ---
LEGACY_ARTIFACTS = REPO_ROOT / "artifacts" / "study"
LEGACY_DATA = REPO_ROOT / "data" / "study"

# --- podziały danych i seedy algorytmu ---
BASE_SPLIT = 2026                       # = podział z badania V' (data/study); stan „przed” jest porównywalny
SPLIT_SEEDS = [2026, 2027, 2028, 2029]  # BASE_SPLIT + 3 dodatkowe
GENERATOR_SEED = 2026                   # pula obserwacji (generatory/MAGIC) jest stała; zmienia się tylko PODZIAŁ
ALGO_SEEDS_BASE = list(range(42, 62))   # 20 seedów algorytmu na podziale bazowym
ALGO_SEEDS_EXTRA = list(range(42, 47))  # 5 wspólnych seedów na podziałach dodatkowych (układ zrównoważony 4×5)
ALGO_SEEDS_BUDGET_CHESS = list(range(42, 52))  # Szachownica, serie z budżetem: 10 seedów (koszt)
ALGO_SEEDS_PILOT = list(range(42, 45))         # pilotaż pełnego MAGIC

MAIN_VARIANTS = ["evo_mem", "evo_nomem", "rand_growth", "heur_mem", "rand_fixed"]
BUDGET_VARIANTS = ["evo_mem_bud", "heur_mem_bud", "rand_growth_bud"]
BUDGET_FACTOR = 10.0   # budżet doboru: podpula <= 10 × (liczba wybieranych elementów V'), wspólna dla strategii

VARIANTS = {
    "evo_mem": {"selection": "evolutionary", "memory": True, "mode": "growing"},
    "evo_nomem": {"selection": "evolutionary", "memory": False, "mode": "growing"},
    "rand_fixed": {"selection": "random", "memory": True, "mode": "fixed", "fixed_size": None},
    "rand_growth": {"selection": "random", "memory": True, "mode": "growing"},
    "heur_mem": {"selection": "heuristic", "memory": True, "mode": "growing"},
    "evo_mem_bud": {"selection": "evolutionary", "memory": True, "mode": "growing", "pool_budget_factor": BUDGET_FACTOR},
    "heur_mem_bud": {"selection": "heuristic", "memory": True, "mode": "growing", "pool_budget_factor": BUDGET_FACTOR},
    "rand_growth_bud": {"selection": "random", "memory": True, "mode": "growing", "pool_budget_factor": BUDGET_FACTOR},
}
VARIANT_LABELS = {
    "evo_mem": "ewolucyjny, z pamięcią",
    "evo_nomem": "ewolucyjny, bez pamięci",
    "rand_growth": "losowy rosnący",
    "heur_mem": "heurystyka błędów, z pamięcią",
    "rand_fixed": "losowy stały (uprzywilejowany)",
    "evo_mem_bud": "ewolucyjny, z pamięcią, budżet ×10",
    "heur_mem_bud": "heurystyka, z pamięcią, budżet ×10",
    "rand_growth_bud": "losowy rosnący, budżet ×10",
}
VARIANT_SHORT = {
    "evo_mem": "evo_mem", "evo_nomem": "evo_nomem", "rand_growth": "rand_growth", "heur_mem": "heur_mem",
    "rand_fixed": "rand_fixed*", "evo_mem_bud": "evo_mem_bud", "heur_mem_bud": "heur_mem_bud",
    "rand_growth_bud": "rand_growth_bud",
}
# wariant z serii budżetowej -> jego odpowiednik bez budżetu (ten sam dobór, pełna pula)
BUDGET_TWIN = {"evo_mem_bud": "evo_mem", "heur_mem_bud": "heur_mem", "rand_growth_bud": "rand_growth"}

GA = {
    "population_size": 30, "generations": 30, "tournament_k": 3, "crossover_rate": 0.8,
    "mutation_rate": 0.05, "elitism": 2, "early_stopping": {"patience": 3, "min_delta": 0.0001},
}
VAL_GA = {
    "population_size": 30, "generations": 20, "tournament_k": 3, "crossover_rate": 0.8,
    "mutation_rate": 0.1, "elitism": 2,
}

# nazwa -> etykieta, katalog danych (względem podziału), header, K_start, parametry reguły stopu
#   min_vprime_per_class: 50 dla problemów binarnych; 20 dla Digits (10 klas, V ma ≈72 próbki/klasę — wykonalność)
#   perfect_recheck_min_add_per_class: „blok” ponownego sprawdzenia = ta sama liczba próbek/klasę co minimum
DATASETS = {
    "chessboard": {"label": "Szachownica", "kind": "file", "header": False, "k_start": 10, "svm": {"C": 10.0, "gamma": 0.002},
                   "min_vp": 50, "split_dependent": False},
    "moons": {"label": "Moons", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50, "split_dependent": True},
    "circles": {"label": "Circles", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50, "split_dependent": True},
    "spiral": {"label": "Spiral", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50, "split_dependent": True},
    "classification2d": {"label": "Classification 2D", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50,
                         "split_dependent": True},
    "digits": {"label": "Digits (64 cechy)", "kind": "gen", "header": True, "k_start": 5, "min_vp": 20, "split_dependent": True},
    "magic": {"label": "MAGIC (podpróbka 8000)", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50,
              "split_dependent": True},
    "magic_full": {"label": "MAGIC (pełny, 19 020)", "kind": "gen", "header": True, "k_start": 10, "min_vp": 50,
                   "split_dependent": True, "svm_from": "magic"},
}
MAIN_DATASETS = ["chessboard", "moons", "circles", "spiral", "classification2d", "digits", "magic"]
EXTRA_SPLIT_DATASETS = [d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]]
D2 = ["chessboard", "moons", "circles", "spiral", "classification2d"]
HD = ["digits", "magic"]
RECHECK_EXPANSIONS = 1
PERFECT_STOP_CATEGORY_LABELS = {
    "perfect_ba": "BA(V′)=1,0",
    "no_k_growth_signal": "brak sygnału wzrostu K",
    "k_limit": "limit K",
    "no_candidates": "brak kandydatów",
    "max_cycles": "limit cykli",
}


def split_dir(split: int) -> Path:
    return DATA_ROOT / f"split_{split}"


def dataset_dir(split: int, dataset: str) -> Path:
    return split_dir(split) / dataset


def series_root(split: int, dataset: str, variant: str) -> Path:
    return ARTIFACT_ROOT / f"split_{split}" / dataset / variant


def seeds_for(split: int, dataset: str, variant: str) -> list[int]:
    """Seedy algorytmu zaplanowane dla serii (split × zbiór × wariant)."""
    if dataset == "magic_full":
        return list(ALGO_SEEDS_PILOT)
    if variant in BUDGET_VARIANTS:
        if split != BASE_SPLIT:
            return []
        return list(ALGO_SEEDS_BUDGET_CHESS) if dataset == "chessboard" else list(ALGO_SEEDS_BASE)
    if split == BASE_SPLIT:
        return list(ALGO_SEEDS_BASE)
    return list(ALGO_SEEDS_EXTRA) if dataset in EXTRA_SPLIT_DATASETS else []


def planned_series(include_pilot: bool = False) -> list[tuple[int, str, str, list[int]]]:
    """Pełna lista zaplanowanych serii (split, zbiór, wariant, seedy)."""
    out = []
    datasets = MAIN_DATASETS + (["magic_full"] if include_pilot else [])
    for split in SPLIT_SEEDS:
        for dataset in datasets:
            variants = MAIN_VARIANTS + (BUDGET_VARIANTS if split == BASE_SPLIT and dataset != "magic_full" else [])
            if dataset == "magic_full":
                variants = ["evo_mem", "rand_growth", "heur_mem"]
            for variant in variants:
                seeds = seeds_for(split, dataset, variant)
                if seeds and (split == BASE_SPLIT or dataset in EXTRA_SPLIT_DATASETS):
                    out.append((split, dataset, variant, seeds))
    return out


# ------------------------------------------------------------------------------------------
# konfiguracje przebiegów
# ------------------------------------------------------------------------------------------
def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def svm_params(split: int, dataset: str) -> dict:
    """(C, gamma) WSPÓLNE dla wszystkich podziałów: wybrane raz na podziale bazowym (jak w badaniu V').

    Powód: ponowne strojenie na każdym podziale zmieniało cały reżim obliczeniowy (np. Classification 2D, podział 2029:
    gamma 4 zamiast 1 => K końcowe ~555 zamiast ~130) i mieszałoby wpływ podziału danych z niestabilnością strojenia.
    Wyniki strojenia per podział (tune_svm.py) są zachowane jako tabela diagnostyczna.
    """
    import json

    ds = DATASETS[dataset]
    if ds.get("svm"):
        return dict(ds["svm"])
    payload = json.loads((split_dir(BASE_SPLIT) / "svm_params.json").read_text(encoding="utf-8"))
    return {"C": float(payload[dataset]["C"]), "gamma": float(payload[dataset]["gamma"])}


def stop_rule_params(dataset: str) -> dict:
    """Parametry rozszerzonej reguły stopu dla zbioru (wartości i uzasadnienie: patrz raport, sekcja 2)."""
    min_vp = int(DATASETS[dataset]["min_vp"])
    return {
        "min_vprime_per_class": min_vp,
        "perfect_recheck_expansions": RECHECK_EXPANSIONS,
        "perfect_recheck_min_add_per_class": min_vp,
    }


def build_config(split: int, dataset: str, variant: str, seed: int = 42, legacy_rule: bool = False) -> dict:
    """Pełna konfiguracja przebiegu. `legacy_rule=True` zeruje nowe parametry (reguła historyczna)."""
    ds = DATASETS[dataset]
    svm = svm_params(split, dataset)
    if ds["kind"] == "file":
        ddir = "data/raw"
    else:
        ddir = rel(dataset_dir(split, dataset))
    rule = {"min_vprime_per_class": 0, "perfect_recheck_expansions": 0, "perfect_recheck_min_add_per_class": 0} \
        if legacy_rule else stop_rule_params(dataset)
    ga = dict(GA, early_stopping=dict(GA["early_stopping"]))
    return {
        "seed": int(seed),
        "experiment_name": f"{STUDY}_s{split}_{dataset}_{variant}_seed{seed}",
        "dataset_name": ds["label"],
        "algorithm_mode": "dagasvm_phase",
        "artifacts_root": rel(series_root(split, dataset, variant)),
        "data": {
            "train_csv": f"{ddir}/train.csv",
            "validation_csv": f"{ddir}/validation.csv",
            "test_csv": f"{ddir}/test.csv",
            "target": "label",
            "stratify": True,
            "header": bool(ds["header"]),
            "target_candidates": ["target", "y", "class"],
            "max_rows": 0,
        },
        "svm": {
            "kernel": "rbf", "C": float(svm["C"]), "gamma": float(svm["gamma"]),
            "class_weight": "balanced", "probability": False, "standardize": False,
        },
        "fitness": {"metric": "balanced_accuracy", "class_imbalance_penalty": 0.0},
        "ga": ga,
        "daga": {
            "k_start": ds["k_start"], "TSV": 0.25, "Ts": 3, "phase_min_delta": 0.0005,
            "sv_ratio_stability_epsilon": 0.02, "sv_ratio_saturation_margin": 0.15,
            **rule,
        },
        "vprime": dict(VARIANTS[variant]),
        "val_ga": dict(VAL_GA),
        "observation": {"full_validation": True, "test": True},
        "followup": {"study": STUDY, "split_seed": int(split), "dataset": dataset, "variant": variant,
                     "stop_rule": "legacy" if legacy_rule else "extended"},
    }
