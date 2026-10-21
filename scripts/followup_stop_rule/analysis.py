"""Ładowanie i agregacja przebiegów badania `followup_stop_rule` oraz stanu „przed” z badania V' (data/study).

Wszystkie liczby w raporcie pochodzą z plików per-run (summary.json + cycles.csv); żadne zbiorcze ranking.csv nie jest używane.
Wspólne definicje (te same dla przebiegów starych i nowych):
  * stop przez BA(V')=1,0        : stop_category == "perfect_ba" (kod decyzji stop_perfect_on_current_val),
  * minimum osiągnięte w cyklu c : najmniejsza liczność klasy w V' na POCZĄTKU cyklu >= wymagane minimum zbioru,
  * rozszerzenie po minimum      : cykl z decyzją expand_validation_only, w którym minimum było już osiągnięte,
  * stop PRZEDWCZESNY (post hoc) : stop przez BA(V')=1,0 oraz (minimum nieosiągnięte LUB brak rozszerzenia po minimum);
                                   dla rand_fixed (V' stałe, brak rozszerzeń z definicji) tylko pierwszy warunek.
Dla przebiegów starej reguły jest to oszacowanie „z dołu” tego, co nowa reguła by zablokowała.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ARTIFACT_ROOT, BUDGET_TWIN, DATASETS, LEGACY_ARTIFACTS, RECHECK_EXPANSIONS, series_root,
)

STOP_CODE_TO_CATEGORY = {
    "stop_perfect_on_current_val": "perfect_ba",
    "stop_no_growth_signal": "no_k_growth_signal",
    "stop_no_k_growth": "k_limit",
    "stop_k_class_cap": "k_limit",
    "stop_no_validation_candidates": "no_candidates",
    "stop_no_new_vprime_elements": "no_candidates",
    "stop_max_cycles": "max_cycles",
}


def _min_class_count(class_distribution: dict, n_classes: int) -> int:
    counts = [int(v) for v in class_distribution.values()]
    if len(counts) < n_classes:
        return 0
    return min(counts)


def _run_row(summary: dict, cycles: pd.DataFrame, job: dict | None, dataset: str, variant: str, legacy: bool) -> dict:
    rep, gb = summary["representative_model"], summary["global_best"]
    ps = summary.get("phase_study") or {}
    cost = ps.get("cost") or {}
    test, full, vcur = gb.get("metrics_test") or {}, gb.get("metrics_full_validation") or {}, gb["metrics_val"]
    expand = cycles[cycles["decision"] == "expand_validation_only"]
    evals = cycles["train_ga_svm_evaluations"]
    n_classes = len(summary["cycles"][0]["current_val"]["class_distribution"])
    min_required = int(DATASETS[dataset]["min_vp"]) if dataset in DATASETS else 0
    stop_decision = ps.get("stop_decision") or cycles["decision"].iloc[-1]
    stop_category = ps.get("stop_category") or STOP_CODE_TO_CATEGORY.get(stop_decision, "other")

    # --- minimalny rozmiar V' i rozszerzenia po jego osiągnięciu (definicje wspólne dla starej i nowej reguły) ---
    min_counts = [_min_class_count(c["current_val"]["class_distribution"], n_classes) for c in summary["cycles"]]
    reached = [m >= min_required for m in min_counts]
    expansions_after_min = int(sum(
        1 for c, r in zip(summary["cycles"], reached)
        if r and c["decision"] == "expand_validation_only" and (c["val_ga"].get("global_val_rows_added") or 0) > 0
    ))
    perfect_stop = stop_category == "perfect_ba"
    fixed = (ps.get("vprime") or {}).get("mode") == "fixed"
    premature = bool(perfect_stop and (not reached[-1] or (not fixed and expansions_after_min < RECHECK_EXPANSIONS)))
    first_reached = next((i + 1 for i, r in enumerate(reached) if r), None)

    hist = [c["val_ga"]["history_summary"] for c in summary["cycles"]]
    sel_predict = cost.get("selection_predict_rows", cost.get("control_model_predictions", 0))
    if legacy:  # starsze summary: control_model_predictions = wiersze przekazane modelowi przy wyborze (predict)
        sel_decision = int(cost.get("control_model_predictions", 0)) if variant.startswith("heur") else 0
    else:
        sel_decision = cost.get("selection_decision_rows", 0)
    nominal_rows = sum((h.get("val_ga_nominal_subset_rows_scored") or 0) for h in hist)
    cand_total = sum((h.get("n_candidates") or 0) for h in hist)
    scored_total = sum((h.get("n_scored_pool", h.get("n_candidates")) or 0) for h in hist)
    sr = ps.get("stop_rule") or {}
    row = {
        "seed": int(summary["seed"]), "run_id": summary["run_id"], "job_status": None if job is None else job.get("status"),
        "stop_decision": stop_decision, "stop_category": stop_category,
        "perfect_ba_stop": bool(perfect_stop), "premature_stop": premature,
        "cycles": int(len(cycles)), "final_k": int(gb["k_per_class"]), "k_after_last": int(cycles["k_after"].iloc[-1]),
        "k_increases": int((cycles["decision"] == "increase_k").sum()),
        "max_k_increases_in_cycle": int(cycles["k_increases_this_cycle"].max()),
        "vprime_expansions": int(len(expand)), "vprime_size": int(rep["validation_rows"]),
        "training_rows": int(rep["training_rows"]), "num_support_vectors": int(vcur["num_support_vectors"]),
        "vprime_ba": float(vcur["metric"]),
        "test_ba": float(test["metric"]) if test else np.nan,
        "test_acc": float(test["all"]["accuracy"]) if test else np.nan,
        "fullv_ba": float(full["metric"]) if full else np.nan,
        "duration_sec": float(summary["experiment_duration_sec"]),
        "min_class_count_final": int(min_counts[-1]), "min_reached_at_stop": bool(reached[-1]),
        "min_first_reached_cycle": first_reached, "expansions_after_min": expansions_after_min,
        "perfect_cycles": int(cycles["perfect_on_current_val"].astype(bool).sum()),
        "gate_pending_cycles": int(sr.get("perfect_gate_pending_cycles", 0)),
        "stops_blocked": int(sr.get("perfect_stops_blocked_total", 0)),
        "rechecks_total": int(sr.get("perfect_rechecks_total", 0)),
        # --- koszt: SVM w GA dla T' ---
        "svm_fits_total": int(evals.sum()),
        "tga_fit_rows": float((evals * cycles["train_rows"]).sum()),
        "tga_vprime_rows_scored": float((evals * cycles["val_rows_before"]).sum()),
        # --- koszt: dobór przyrostu V' (faktyczne wiersze przekazane modelowi) ---
        "selection_predict_rows": int(sel_predict), "selection_decision_rows": int(sel_decision),
        "val_ga_nominal_evals": int(cost.get("val_ga_population_evaluations", 0) or 0),
        "val_ga_nominal_subset_rows": int(nominal_rows),
        "candidates_total": int(cand_total), "scored_pool_total": int(scored_total),
        "increment_rows_total": int(sum(c["val_ga"]["adv_selected_rows"] for c in summary["cycles"])),
        "n_expansions_executed": int(sum(1 for c in summary["cycles"] if c["val_ga"]["adv_selected_rows"] > 0)),
        "diagnostic_scored_rows": int(cost.get("diagnostic_scored_rows", cost.get("diagnostic_predictions", 0)) or 0),
        "mean_increment_error_rate": float(cycles["increment_error_rate"].mean())
        if "increment_error_rate" in cycles and cycles["increment_error_rate"].notna().any() else np.nan,
        "mean_pool_error_rate": float(cycles["candidate_pool_error_rate"].mean())
        if "candidate_pool_error_rate" in cycles and cycles["candidate_pool_error_rate"].notna().any() else np.nan,
    }
    return row


def load_series(root: Path, dataset: str, variant: str, legacy: bool = False):
    """(per_run, per_cycle, audyt) dla serii; ostatni przebieg na seed, duplikaty/awarie raportowane."""
    runs, frames = [], []
    audit = {"duplicates": [], "failed_jobs": [], "stale_running": []}
    jobs = {}
    for jp in (root / "jobs").glob("seed_*.json"):
        j = json.loads(jp.read_text(encoding="utf-8"))
        jobs[int(j["seed"])] = j
        if j.get("status") == "failed":
            audit["failed_jobs"].append(int(j["seed"]))
        elif j.get("status") == "running":
            audit["stale_running"].append(int(j["seed"]))
    by_seed: dict[int, list[Path]] = {}
    for sp in sorted((root / "runs").glob("*/metrics/summary.json")):
        try:
            seed = int(json.loads(sp.read_text(encoding="utf-8"))["seed"])
        except Exception:
            continue
        by_seed.setdefault(seed, []).append(sp)
    for seed, paths in sorted(by_seed.items()):
        if len(paths) > 1:
            audit["duplicates"].append({"seed": seed, "n": len(paths)})
        sp = paths[-1]
        if not (sp.parent / "cycles.csv").exists():
            continue
        summary = json.loads(sp.read_text(encoding="utf-8"))
        cycles = pd.read_csv(sp.parent / "cycles.csv")
        runs.append(_run_row(summary, cycles, jobs.get(seed), dataset, variant, legacy))
        cyc = cycles.copy()
        cyc.insert(0, "seed", seed)
        frames.append(cyc)
    per_run = pd.DataFrame(runs)
    per_cycle = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return per_run, per_cycle, audit


def describe(values) -> dict:
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return {k: np.nan for k in ("n", "mean", "sd", "min", "median", "max", "cv", "sem")}
    mean, sd = float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return {"n": int(len(v)), "mean": mean, "sd": sd, "min": float(v.min()), "median": float(v.median()),
            "max": float(v.max()), "cv": float(sd / mean) if mean else np.nan,
            "sem": float(sd / np.sqrt(len(v))) if len(v) > 1 else np.nan}


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(pd.Series(values, dtype=float).dropna())
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_comparison(a: pd.Series, b: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """Różnice sparowane po seedach (a − b): średnia, bootstrapowy 95% CI, Wilcoxon, dz, liczby seedów na korzyść."""
    from scipy import stats

    a, b = a.align(b, join="inner")
    d = (a - b).dropna()
    n = len(d)
    if n < 2:
        return {"n": n}
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    sd = float(d.std(ddof=1))
    try:
        p_w = float(stats.wilcoxon(d.to_numpy(), zero_method="wilcox").pvalue)
    except ValueError:
        p_w = float("nan")
    return {"n": int(n), "mean_diff": float(d.mean()), "sd_diff": sd, "ci95_low": lo, "ci95_high": hi,
            "wilcoxon_p": p_w, "dz": float(d.mean() / sd) if sd > 0 else float("nan"),
            "a_better_count": int((d > 0).sum()), "b_better_count": int((d < 0).sum())}


def variance_components(frame: pd.DataFrame, value: str, group: str = "split", unit: str = "seed") -> dict:
    """Jednokierunkowa ANOVA z efektem losowym (układ zrównoważony): zmienność między podziałami vs między seedami.

    Seed jest zagnieżdżony w podziale (ten sam numer seedu na różnych podziałach to inne losowanie niezależne od podziału).
    Zwraca składniki wariancji σ²_split i σ²_seed oraz udział podziału (ICC). Zwraca NaN dla układu niezrównoważonego.
    """
    g = frame.dropna(subset=[value]).groupby(group)[value]
    sizes = g.size()
    if sizes.nunique() != 1 or len(sizes) < 2 or sizes.iloc[0] < 2:
        return {"n_groups": int(len(sizes)), "n_per_group": int(sizes.iloc[0]) if len(sizes) else 0,
                "var_split": np.nan, "var_seed": np.nan, "sd_split": np.nan, "sd_seed": np.nan, "icc": np.nan,
                "sd_total": np.nan}
    k, m = len(sizes), int(sizes.iloc[0])
    means = g.mean()
    ms_within = float(g.var(ddof=1).mean())
    ms_between = float(m * ((means - means.mean()) ** 2).sum() / (k - 1))
    var_split = max(0.0, (ms_between - ms_within) / m)
    total = var_split + ms_within
    return {"n_groups": int(k), "n_per_group": m, "var_split": var_split, "var_seed": ms_within,
            "sd_split": float(np.sqrt(var_split)), "sd_seed": float(np.sqrt(ms_within)),
            "icc": float(var_split / total) if total > 0 else np.nan, "sd_total": float(np.sqrt(total))}
