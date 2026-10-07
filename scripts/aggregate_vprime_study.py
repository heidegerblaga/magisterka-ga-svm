"""Agreguje wyniki badania V' z plików per-run i zapisuje tabele do reports/vprime_study/tables/.

Źródłem są wyłącznie metryki per-run (summary.json + cycles.csv). Stary ranking.csv jest audytowany
(patrz ga.study_analysis.audit_legacy_ranking) i NIE wchodzi do analizy.

Użycie:  python scripts/aggregate_vprime_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ga import study_analysis as sa  # noqa: E402

TABLES = REPO_ROOT / "reports" / "vprime_study" / "tables"
LEGACY_SWEEP = REPO_ROOT / "artifacts" / "sweeps" / "20260922_204113_256002_dagasvm_phase_evolutionary_seeds20"

SUMMARY_METRICS = [
    "vprime_ba", "test_ba", "test_acc", "test_f1", "fullv_ba", "fullv_acc", "fullv_f1",
    "cycles", "final_k", "vprime_size", "vprime_changes", "vprime_expansions", "k_increases",
    "num_support_vectors", "training_rows", "duration_sec", "svm_fits_total", "fit_rows_total",
    "control_model_predictions", "mean_increment_error_rate", "mean_pool_error_rate",
]
PAIRED_METRICS = [
    "test_ba", "fullv_ba", "vprime_ba", "cycles", "final_k", "vprime_size", "vprime_changes",
    "duration_sec", "svm_fits_total", "fit_rows_total", "num_support_vectors",
]
PAIRS = [("evo_mem", "rand_fixed"), ("evo_mem", "rand_growth"), ("evo_mem", "heur_mem"), ("evo_mem", "evo_nomem")]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    spec = sa.load_spec()
    seeds = spec["seeds"]
    per_run_frames, per_cycle_frames, validations = [], [], []
    for dataset in spec["datasets"]:
        for variant in spec["variants"]:
            per_run, per_cycle, audit = sa.load_series(dataset, variant)
            if per_run.empty and not (sa.STUDY_ROOT / dataset / variant).exists():
                continue
            check = sa.validate_series(per_run, seeds)
            validations.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n_runs": check["n_runs"],
                    "complete_20_seeds": check["complete"],
                    "missing_seeds": json.dumps(check.get("missing_seeds", [])),
                    "issues": "; ".join(check["issues"]),
                    "failed_jobs": json.dumps(audit["failed_jobs"]),
                    "duplicate_runs": json.dumps(audit["duplicates"]),
                }
            )
            if not per_run.empty:
                per_run_frames.append(per_run)
                per_cycle_frames.append(per_cycle)
    per_run_all = pd.concat(per_run_frames, ignore_index=True)
    per_cycle_all = pd.concat(per_cycle_frames, ignore_index=True)
    per_run_all.to_csv(TABLES / "per_run.csv", index=False)
    per_cycle_all.to_csv(TABLES / "per_cycle.csv", index=False)
    pd.DataFrame(validations).to_csv(TABLES / "series_validation.csv", index=False)

    rows = []
    for (dataset, variant), grp in per_run_all.groupby(["dataset", "variant"]):
        for metric in SUMMARY_METRICS:
            if metric in grp:
                rows.append({"dataset": dataset, "variant": variant, "metric": metric, **sa.describe(grp[metric])})
    pd.DataFrame(rows).to_csv(TABLES / "summary_by_series.csv", index=False)

    paired = []
    for dataset, grp in per_run_all.groupby("dataset"):
        for a_name, b_name in PAIRS:
            a = grp[grp["variant"] == a_name].set_index("seed")
            b = grp[grp["variant"] == b_name].set_index("seed")
            if a.empty or b.empty:
                continue
            for metric in PAIRED_METRICS:
                res = sa.paired_comparison(a[metric], b[metric])
                paired.append({"dataset": dataset, "a": a_name, "b": b_name, "metric": metric, **res})
    pd.DataFrame(paired).to_csv(TABLES / "paired_comparisons.csv", index=False)

    # --- skuteczność wyszukiwania Val-GA: najlepszy fitness populacji startowej vs końcowy ---
    search_rows = []
    for (dataset, variant), grp in per_run_all.groupby(["dataset", "variant"]):
        if variant not in ("evo_mem", "evo_nomem"):
            continue
        for _, run in grp.iterrows():
            metrics_dir = sa.STUDY_ROOT / dataset / variant / "runs" / run.run_id / "metrics"
            for path in sorted(metrics_dir.glob("cycle_*_val_ga_history.json")):
                h = json.loads(path.read_text(encoding="utf-8"))
                hist = h.get("val_ga_history")
                if not hist:
                    continue
                search_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": int(run.seed),
                        "cycle": int(path.stem.split("_")[1]),
                        "n_candidates": h.get("n_candidates"),
                        "n_select": h.get("n_select_target"),
                        "ga_initial_best_fitness": hist[0],
                        "ga_final_best_fitness": max(hist),
                        "ga_generations": len(hist) - 1,
                        "pool_error_rate": h.get("candidate_pool_error_rate"),
                        "selected_error_rate": h.get("selected_error_rate"),
                        "errors_available": h.get("candidate_pool_errors_available"),
                        "selected_errors": h.get("selected_errors"),
                        "nominal_subset_rows_scored": h.get("val_ga_nominal_subset_rows_scored"),
                    }
                )
    pd.DataFrame(search_rows).to_csv(TABLES / "val_ga_search.csv", index=False)

    # --- audyt starego rankingu + regresja względem starej serii (Szachownica, evo_mem) ---
    if (LEGACY_SWEEP / "ranking.csv").exists():
        audit = sa.audit_legacy_ranking(LEGACY_SWEEP)
        (TABLES / "legacy_ranking_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        legacy = pd.read_csv(LEGACY_SWEEP / "representative_metrics.csv").set_index("seed")
        new = per_run_all[(per_run_all["dataset"] == "chessboard") & (per_run_all["variant"] == "evo_mem")].set_index("seed")
        common = sorted(set(legacy.index) & set(new.index))
        if common:
            cmp = pd.DataFrame(
                {
                    "legacy_cycles": legacy.loc[common, "cycles"],
                    "new_cycles": new.loc[common, "cycles"],
                    "legacy_final_k": legacy.loc[common, "final_k"],
                    "new_final_k": new.loc[common, "final_k"],
                    "legacy_vprime_size": legacy.loc[common, "validation_rows"],
                    "new_vprime_size": new.loc[common, "vprime_size"],
                    "legacy_test_ba": legacy.loc[common, "test_balanced_accuracy"],
                    "new_test_ba": new.loc[common, "test_ba"],
                    "legacy_duration_sec": legacy.loc[common, "duration_sec"],
                    "new_duration_sec": new.loc[common, "duration_sec"],
                }
            )
            cmp["identical_trajectory"] = (
                (cmp.legacy_cycles == cmp.new_cycles)
                & (cmp.legacy_final_k == cmp.new_final_k)
                & (cmp.legacy_vprime_size == cmp.new_vprime_size)
                & ((cmp.legacy_test_ba - cmp.new_test_ba).abs() < 1e-12)
            )
            cmp.to_csv(TABLES / "regression_vs_legacy_series.csv")
    print(f"zapisano tabele w {TABLES}")
    print(pd.DataFrame(validations).to_string(index=False))


if __name__ == "__main__":
    main()
