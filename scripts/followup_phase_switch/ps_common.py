"""Wspólna specyfikacja badania `followup_phase_switch` (automat T' <-> V', ``algorithm_mode: dagasvm_phase_switch``).

Badanie jest ODDZIELONE od `followup_stop_rule`: własne konfiguracje (configs/followup_phase_switch/), artefakty
(artifacts/followup_phase_switch/<seria>/<zbiór>/<wariant>/), tabele (reports/followup_phase_switch/). Dane i (C, γ)
są wspólne z `followup_stop_rule` (podział bazowy 2026; Szachownica: data/raw) i tylko odczytywane.

Seria = nazwany zestaw nadpisań sekcji ``phase_switch`` (np. inny limit |V'|); zapisywany w series_spec.json serii.

WERSJE. v1 (smoke z 2026-10-14; serie bez prefiksu ``v2_``) pozostaje nietknięta: jej artefakty, tabele
(reports/followup_phase_switch/tables/), ``RAPORT_SMOKE.md`` i szablony konfiguracji. v2 (serie ``v2_*``) ma
poprawioną semantykę próby V' (T' dostaje szansę po zwiększeniu K), ``vprime_difficulty_metric=error_rate``,
globalny ``max_vprime_expansions`` i liczniki kosztu; jej tabele trafiają do reports/followup_phase_switch/v2/,
a szablony do configs/followup_phase_switch/v2/. Parametry v1 odtwarza ``PHASE_SWITCH_DEFAULTS_V1`` (+ nadpisania
``retry_resolution=first_step``, ``vprime_difficulty_metric=one_minus_metric``).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fsr_common", REPO_ROOT / "scripts" / "followup_stop_rule" / "common.py")
fsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fsr)

STUDY = "followup_phase_switch"
CONFIG_ROOT = REPO_ROOT / "configs" / STUDY
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / STUDY
REPORT_ROOT = REPO_ROOT / "reports" / STUDY
TABLES = REPORT_ROOT / "tables"                      # v1 (nie nadpisywać)
V2_PREFIX = "v2_"
CONFIG_ROOT_V2 = CONFIG_ROOT / "v2"
REPORT_ROOT_V2 = REPORT_ROOT / "v2"
TABLES_V2 = REPORT_ROOT_V2 / "tables"
SPLIT = fsr.BASE_SPLIT

VARIANTS = {
    "evo_switch": {"selection": "evolutionary", "memory": True, "mode": "growing"},
    "rand_switch": {"selection": "random", "memory": True, "mode": "growing"},
    # losowy dobór ΔV' przy DOKŁADNIE tym samym harmonogramie (fazy, K, |ΔV'|) co evo_switch z tym samym seedem
    "rand_replay": {"selection": "random", "memory": True, "mode": "growing"},
    "heur_switch": {"selection": "heuristic", "memory": True, "mode": "growing"},
}
VARIANT_LABELS = {
    "evo_switch": "ewolucyjny dobór ΔV'",
    "rand_switch": "losowy dobór ΔV', własny automat (te same reguły, własny harmonogram)",
    "rand_replay": "losowy dobór ΔV', harmonogram (fazy, K, |ΔV'|) odtworzony z evo_switch",
    "heur_switch": "heurystyka błędów ΔV'",
}

# wartości domyślne automatu v1 (smoke 2026-10-14) — tylko do odtwarzania; NIE używane przez nowe serie
PHASE_SWITCH_DEFAULTS_V1 = {
    "vprime_increment_per_class": 10,
    "max_vprime_fraction": 0.25,
    "max_vprime_rows": None,
    "vprime_difficulty_min_delta": 0.01,
    "vprime_difficulty_patience": 2,
    "vprime_difficulty_window": 2,
    "vprime_min_difficulty": 0.0,
    "max_vprime_expansions_per_phase": 10,
    "max_no_improvement_k_increases": 3,
    "fitness_min_delta": 0.0005,
    "allow_one_vprime_retry_after_t_stall": True,
    "k_growth_requires_sv_signal": True,
    "warm_start_tprime": True,
    "ga_stop_at_perfect": True,
    "representative": "best_on_final_vprime",
    "max_cycles": 300,
}
# wartości domyślne automatu v2 (sekcja ``phase_switch``); seria może je nadpisać
PHASE_SWITCH_DEFAULTS = {
    "vprime_increment_per_class": 10,
    "max_vprime_fraction": 0.25,
    "max_vprime_rows": None,
    "max_vprime_expansions": None,
    "vprime_difficulty_metric": "error_rate",
    "vprime_difficulty_min_delta": 0.01,
    "vprime_difficulty_patience": 2,
    "vprime_difficulty_window": 2,
    "vprime_min_difficulty": 0.0,
    "max_vprime_expansions_per_phase": 10,
    "max_no_improvement_k_increases": 3,
    "fitness_min_delta": 0.0005,
    "allow_one_vprime_retry_after_t_stall": True,
    "retry_resolution": "after_k_increases",
    "retry_reference": "same_vprime",
    "k_growth_requires_sv_signal": True,
    "warm_start_tprime": True,
    "ga_stop_at_perfect": True,
    "representative": "best_on_final_vprime",
    "max_cycles": 300,
}


def series_root(series: str, dataset: str, variant: str) -> Path:
    return ARTIFACT_ROOT / series / dataset / variant


REPLAY_SOURCE = {"rand_replay": "evo_switch"}


def reference_cycles(series: str, dataset: str, variant: str, seed: int) -> Path:
    """cycles.csv ukończonego przebiegu referencyjnego (najnowszego, gdy było kilka) dla danego seeda."""
    import json

    found = []
    for path in series_root(series, dataset, variant).glob("runs/*/metrics/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if int(summary["seed"]) == int(seed):
            found.append((summary["created_at"], path.parent / "cycles.csv"))
    if not found:
        raise FileNotFoundError(f"brak przebiegu referencyjnego {series}/{dataset}/{variant}/seed{seed}")
    return sorted(found)[-1][1]


def build_config(series: str, dataset: str, variant: str, seed: int, overrides: dict | None = None,
                 split: int = SPLIT) -> dict:
    """Konfiguracja przebiegu: dane, (C, γ), GA i Val-GA jak w followup_stop_rule; automat z PHASE_SWITCH_DEFAULTS."""
    base = fsr.build_config(split, dataset, "evo_mem", seed=seed)
    ps = dict(PHASE_SWITCH_DEFAULTS)
    ps.update(overrides or {})
    if variant in REPLAY_SOURCE:
        ps["schedule_replay"] = str(reference_cycles(series, dataset, REPLAY_SOURCE[variant], seed))
    unknown = set(ps) - set(PHASE_SWITCH_DEFAULTS) - {"schedule_replay"}
    if unknown:
        raise ValueError(f"nieznane klucze phase_switch: {sorted(unknown)}")
    daga = {k: v for k, v in base["daga"].items()
            if k in ("k_start", "TSV", "Ts", "phase_min_delta")}  # reguła min. V' NIE jest używana
    return {
        "seed": int(seed),
        "experiment_name": f"{STUDY}_{series}_{dataset}_{variant}_seed{seed}",
        "dataset_name": base["dataset_name"],
        "algorithm_mode": "dagasvm_phase_switch",
        "artifacts_root": str(series_root(series, dataset, variant).relative_to(REPO_ROOT)),
        "data": base["data"],
        "svm": base["svm"],
        "fitness": {"metric": "balanced_accuracy", "tprime_metric": "balanced_accuracy",
                    "vprime_metric": "balanced_accuracy", "class_imbalance_penalty": 0.0},
        "ga": base["ga"],
        "daga": daga,
        "phase_switch": ps,
        "vprime": dict(VARIANTS[variant]),
        "val_ga": base["val_ga"],
        "observation": {"full_validation": True, "test": True},
        "followup": {"study": STUDY, "series": series, "split_seed": int(split), "dataset": dataset, "variant": variant,
                     "phase_switch_overrides": dict(overrides or {})},
    }
