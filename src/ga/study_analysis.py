"""Agregacja wyników badania V' — wyłącznie z metryk per-run (summary.json + cycles.csv).

Główne źródło danych to pliki per-run (``artifacts/study/<zbiór>/<wariant>/runs/*/metrics``).
Pliki rankingowe sweepów (ranking.csv) NIE są używane: patrz ``audit_legacy_ranking``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPO_ROOT / "artifacts" / "study"
SPEC_PATH = REPO_ROOT / "configs" / "study" / "study_spec.yaml"

VARIANT_LABELS = {
    "evo_mem": "ewolucyjny V', z pamięcią",
    "evo_nomem": "ewolucyjny V', bez pamięci",
    "rand_fixed": "losowy V' (stały rozmiar)",
    "rand_growth": "losowy V' (rosnący)",
    "heur_mem": "heurystyka błędów, z pamięcią",
}

# kolumny „sterujące" vs „obserwacyjne" — używane w raportowaniu
CONTROL_METRICS = ["vprime_ba"]
OBSERVATION_METRICS = ["test_ba", "fullv_ba"]


def load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _series_dir(dataset: str, variant: str, root: Path = STUDY_ROOT) -> Path:
    return root / dataset / variant


def _run_row(summary: dict, cycles: pd.DataFrame, job: dict | None) -> dict:
    rep = summary["representative_model"]
    gb = summary["global_best"]
    ps = summary.get("phase_study") or {}
    cost = ps.get("cost") or {}
    test = gb.get("metrics_test") or {}
    full = gb.get("metrics_full_validation") or {}
    vcur = gb["metrics_val"]
    expand = cycles[cycles["decision"] == "expand_validation_only"]
    changes = expand[expand["vprime_new_elements"].fillna(0) > 0] if "vprime_new_elements" in expand else expand
    train_rows = cycles["train_rows"] if "train_rows" in cycles else None
    evals = cycles["train_ga_svm_evaluations"]
    row = {
        "seed": int(summary["seed"]),
        "run_id": summary["run_id"],
        "status": "completed",
        "job_status": None if job is None else job.get("status"),
        "stop_decision": ps.get("stop_decision") or cycles["decision"].iloc[-1],
        "cycles": int(len(cycles)),
        "final_k": int(gb["k_per_class"]),
        "k_after_last": int(cycles["k_after"].iloc[-1]) if "k_after" in cycles else None,
        "k_increases": int((cycles["decision"] == "increase_k").sum()),
        "max_k_increases_in_cycle": int(cycles["k_increases_this_cycle"].max())
        if "k_increases_this_cycle" in cycles
        else None,
        "vprime_expansions": int(len(expand)),
        "vprime_changes": int(len(changes)),
        "vprime_size": int(rep["validation_rows"]),
        "training_rows": int(rep["training_rows"]),
        "num_support_vectors": int(vcur["num_support_vectors"]),
        # --- metryki sterujące (V') ---
        "vprime_ba": float(vcur["metric"]),
        # --- metryki obserwacyjne ---
        "test_ba": float(test["metric"]) if test else np.nan,
        "test_acc": float(test["all"]["accuracy"]) if test else np.nan,
        "test_f1": float(test["all"].get("f1_macro", test["all"].get("f1"))) if test else np.nan,
        "fullv_ba": float(full["metric"]) if full else np.nan,
        "fullv_acc": float(full["all"]["accuracy"]) if full else np.nan,
        "fullv_f1": float(full["all"].get("f1_macro", full["all"].get("f1"))) if full else np.nan,
        # --- koszt ---
        "duration_sec": float(summary["experiment_duration_sec"]),
        "svm_fits_total": int(evals.sum()),
        "fit_rows_total": float((evals * train_rows).sum()) if train_rows is not None else np.nan,
        "control_model_predictions": cost.get("control_model_predictions"),
        "val_ga_population_evaluations": cost.get("val_ga_population_evaluations"),
        "observation_rows_scored": cost.get("observation_full_validation_rows_scored"),
        # --- trudność przyrostów V' ---
        "mean_increment_error_rate": float(cycles["increment_error_rate"].mean())
        if "increment_error_rate" in cycles and cycles["increment_error_rate"].notna().any()
        else np.nan,
        "mean_pool_error_rate": float(cycles["candidate_pool_error_rate"].mean())
        if "candidate_pool_error_rate" in cycles and cycles["candidate_pool_error_rate"].notna().any()
        else np.nan,
        "forced_growth_after_patience": int(cycles["forced_growth_after_patience"].astype(bool).sum())
        if "forced_growth_after_patience" in cycles
        else 0,
    }
    return row


def load_series(dataset: str, variant: str, root: Path = STUDY_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Zwraca (per_run, per_cycle, audyt) dla serii; ostatni przebieg na seed, duplikaty są raportowane."""
    series = _series_dir(dataset, variant, root)
    runs, cycle_frames, audit = [], [], {"duplicates": [], "failed_jobs": [], "missing_summary": []}
    jobs = {}
    for jp in (series / "jobs").glob("seed_*.json"):
        j = json.loads(jp.read_text(encoding="utf-8"))
        jobs[int(j["seed"])] = j
        if j.get("status") != "completed":
            audit["failed_jobs"].append({"seed": int(j["seed"]), "status": j.get("status")})
    by_seed: dict[int, list[Path]] = {}
    for sp in sorted((series / "runs").glob("*/metrics/summary.json")):
        seed = int(json.loads(sp.read_text(encoding="utf-8"))["seed"])
        by_seed.setdefault(seed, []).append(sp)
    for seed, paths in sorted(by_seed.items()):
        if len(paths) > 1:
            audit["duplicates"].append({"seed": seed, "n": len(paths)})
        sp = paths[-1]
        summary = json.loads(sp.read_text(encoding="utf-8"))
        cycles = pd.read_csv(sp.parent / "cycles.csv")
        row = _run_row(summary, cycles, jobs.get(seed))
        runs.append(row)
        cyc = cycles.copy()
        cyc.insert(0, "seed", seed)
        cycle_frames.append(cyc)
    per_run = pd.DataFrame(runs)
    per_cycle = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    if not per_run.empty:
        per_run.insert(0, "variant", variant)
        per_run.insert(0, "dataset", dataset)
        per_cycle.insert(0, "variant", variant)
        per_cycle.insert(0, "dataset", dataset)
    return per_run, per_cycle, audit


def validate_series(per_run: pd.DataFrame, expected_seeds: list[int]) -> dict:
    """Kontrola kompletności i spójności serii: seedy, statusy, niezmienność inwariantów."""
    if per_run.empty:
        return {"complete": False, "n_runs": 0, "missing_seeds": list(expected_seeds), "issues": ["brak przebiegów"]}
    seeds = sorted(per_run["seed"].tolist())
    missing = [s for s in expected_seeds if s not in seeds]
    extra = [s for s in seeds if s not in expected_seeds]
    issues = []
    if missing:
        issues.append(f"brakujące seedy: {missing}")
    if extra:
        issues.append(f"nadmiarowe seedy: {extra}")
    if (per_run["max_k_increases_in_cycle"].dropna() > 1).any():
        issues.append("wzrost K więcej niż raz w cyklu")
    if per_run["job_status"].notna().any() and (per_run["job_status"].dropna() != "completed").any():
        issues.append("przebiegi o statusie innym niż completed")
    return {
        "complete": not missing and not extra and not issues,
        "n_runs": int(len(per_run)),
        "missing_seeds": missing,
        "extra_seeds": extra,
        "issues": issues,
    }


def describe(values) -> dict:
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return {k: np.nan for k in ("n", "mean", "sd", "min", "median", "max", "cv")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {
        "n": int(len(v)),
        "mean": mean,
        "sd": sd,
        "min": float(v.min()),
        "median": float(v.median()),
        "max": float(v.max()),
        "cv": float(sd / mean) if mean else np.nan,
    }


def paired_comparison(a: pd.Series, b: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """Porównanie sparowane po seedach (a − b): średnia różnic, bootstrapowy 95% CI, Wilcoxon, dz."""
    from scipy import stats

    a, b = a.align(b, join="inner")
    d = (a - b).dropna()
    n = len(d)
    if n < 2:
        return {"n": n}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d.to_numpy(), size=(n_boot, n), replace=True).mean(axis=1)
    sd = float(d.std(ddof=1))
    try:
        w = stats.wilcoxon(d.to_numpy(), zero_method="wilcox")
        p_w = float(w.pvalue)
    except ValueError:  # same wartości
        p_w = float("nan")
    return {
        "n": int(n),
        "mean_diff": float(d.mean()),
        "sd_diff": sd,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "wilcoxon_p": p_w,
        "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "a_better_count": int((d > 0).sum()),
        "b_better_count": int((d < 0).sum()),
    }


def audit_legacy_ranking(sweep_dir: Path) -> dict:
    """Weryfikuje wiarygodność ranking.csv starego sweepa (Szachownica, 20 seedów) względem metryk per-run."""
    ranking = pd.read_csv(sweep_dir / "ranking.csv")
    rep = pd.read_csv(sweep_dir / "representative_metrics.csv")
    out = {
        "ranking_rows": int(len(ranking)),
        "ranking_unique_num_support_vectors": int(ranking["num_support_vectors"].nunique()),
        "ranking_unique_full_validation_f1": int(ranking["full_validation_f1"].round(12).nunique()),
        "ranking_unique_test_f1": int(ranking["test_f1"].round(12).nunique()),
        "per_run_unique_num_support_vectors": int(rep["num_support_vectors"].nunique()),
        "per_run_unique_full_validation_f1": int(rep["full_validation_f1"].round(12).nunique()),
        "ranking_num_support_vectors_value": int(ranking["num_support_vectors"].mode().iloc[0]),
        "per_run_num_support_vectors_mean": float(rep["num_support_vectors"].mean()),
        "ranking_full_validation_f1_mean": float(ranking["full_validation_f1"].mean()),
        "per_run_full_validation_f1_mean": float(rep["full_validation_f1"].mean()),
    }
    out["diagnosis"] = (
        "ranking.csv zawiera metryki SVM wytrenowanego na pełnym zbiorze treningowym (stała liczba SV, "
        "niemal stałe F1), a nie reprezentatywnego T' — build_report_row pomijał tryb dagasvm_phase. "
        "Wykluczony z analizy; źródłem są metryki per-run."
    )
    return out
