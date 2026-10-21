"""Zbiera wyniki `followup_phase_switch` do tabel.

    --version v1 (domyślnie)  serie BEZ prefiksu v2_  -> reports/followup_phase_switch/tables/     (smoke 2026-10-14)
    --version v2              serie z prefiksem v2_    -> reports/followup_phase_switch/v2/tables/  (nie nadpisuje v1)

    per_run.csv         jeden wiersz = przebieg (seria, zbiór, wariant, seed): stop, |V'|, |V|, |V'|/|V|, fazy, K, BA
    per_step.csv        wszystkie kroki automatu (cycles.csv) z kluczami przebiegu
    summary.csv         średnie/odch. std. per (seria, zbiór, wariant)
    paired.csv          evo_switch − rand_replay (ten sam harmonogram |V'|) i evo_switch − rand_switch, te same seedy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import ARTIFACT_ROOT, TABLES, TABLES_V2, V2_PREFIX  # noqa: E402


def collect(version: str = "v1") -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, steps = [], []
    for summary_path in sorted(ARTIFACT_ROOT.glob("*/*/*/runs/*/metrics/summary.json")):
        series, dataset, variant = summary_path.parts[-7:-4]
        if series.startswith(V2_PREFIX) != (version == "v2"):
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        ps = s["phase_study"]
        gb = s["global_best"]
        t = (gb.get("metrics_test") or {})
        fv = (gb.get("metrics_full_validation") or {})
        cyc = pd.read_csv(summary_path.parent / "cycles.csv")
        tsteps = cyc[cyc.phase == "optimize_T_prime"]
        vsteps = cyc[cyc.phase == "optimize_V_prime"]
        cfg_path = summary_path.parent / "config_snapshot.json"
        split = int((json.loads(cfg_path.read_text(encoding="utf-8")).get("followup") or {}).get("split_seed", 2026)) if cfg_path.exists() else 2026
        key = {"series": series, "dataset": dataset, "variant": variant, "seed": int(s["seed"])}
        runs.append({
            **key,
            "split": split,
            "run_id": s["run_id"],
            "created_at": s["created_at"],
            "stop_decision": ps["stop_decision"],
            "stop_category": ps["stop_category"],
            "steps": ps["steps"],
            "tprime_steps": ps["tprime_steps"],
            "vprime_steps": ps["vprime_steps"],
            "vprime_phases": ps["vprime_phases"],
            "vprime_phases_mastered": ps["vprime_phases_mastered"],
            "vprime_retries": ps["vprime_retries"],
            "perfect_tprime_steps": ps["perfect_tprime_steps"],
            "k_increases": ps["k_increases_total"],
            "final_k": ps["final_k"],
            "vprime_rows": ps["vprime_rows"],
            "full_validation_rows": ps["full_validation_rows"],
            "vprime_fraction": ps["vprime_fraction"],
            "vprime_cap_rows": ps["vprime_cap_rows"],
            "vprime_cap_reached": ps["vprime_cap_reached"],
            "rep_cycle": gb["cycle"],
            "rep_k": gb["k_per_class"],
            "rep_vprime_rows": gb["validation_rows"],
            "rep_vprime_ba": (gb.get("metrics_val") or {}).get("metric"),
            "test_ba": t.get("metric"),
            "fullv_ba": fv.get("metric"),
            "last_tprime_vprime_ba": float(tsteps.tprime_fitness.iloc[-1]) if len(tsteps) else np.nan,
            "last_tprime_test_ba": float(tsteps.observed_test_metric.iloc[-1]) if len(tsteps) else np.nan,
            "baseline_full_svm_test_ba": ((s.get("baseline") or {}).get("test") or {}).get("metric"),
            "mean_delta_difficulty": ps["mean_delta_difficulty"],
            "mean_delta_error_rate": ps["mean_delta_error_rate"],
            "vprime_phase_end_reasons": json.dumps(ps["vprime_phase_end_reasons"], sort_keys=True),
            "train_ga_svm_evaluations": ps["cost"]["train_ga_svm_evaluations"],
            "selection_predict_rows": ps["cost"]["selection_predict_rows"],
            "delta_eval_rows": ps["cost"]["delta_eval_rows"],
            "duration_sec": s["experiment_duration_sec"],
            # --- v2 (brak w v1) ---
            "vprime_expansions": ps.get("vprime_expansions"),
            "vprime_limit_reason": ps.get("vprime_limit_reason"),
            "retry_started": (ps.get("retry_outcomes") or {}).get("started"),
            "retry_resolved_with_progress": (ps.get("retry_outcomes") or {}).get("resolved_with_progress"),
            "retry_failed_checks": (ps.get("retry_outcomes") or {}).get("failed_checks"),
            "last_tprime_fitness": float(tsteps.tprime_fitness.iloc[-1]) if len(tsteps) else np.nan,
            "delta_ba_mean": float(vsteps.delta_metric.mean()) if len(vsteps) else np.nan,
            **{f"cost_{k}": v for k, v in ((ps["cost"].get("total")) or {}).items()},
            **{f"cost_tprime_{k}": v for k, v in ((ps["cost"].get("tprime_phase")) or {}).items()},
            **{f"cost_vprime_{k}": v for k, v in ((ps["cost"].get("vprime_phase")) or {}).items()},
            "cost_observation_rows": ps["cost"].get("observation_rows"),
        })
        cyc.insert(0, "seed", key["seed"])
        cyc.insert(0, "split", split)
        for k in ("variant", "dataset", "series"):
            cyc.insert(0, k, key[k])
        cyc.insert(0, "run_id", s["run_id"])
        steps.append(cyc)
    runs_df = pd.DataFrame(runs)
    steps_df = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()
    if len(runs_df):
        # powtórzony przebieg tego samego klucza (np. ponowne uruchomienie joba): zostaje najnowszy
        keep = runs_df.sort_values("created_at").drop_duplicates(["series", "dataset", "variant", "seed"], keep="last")
        dropped = len(runs_df) - len(keep)
        if dropped:
            print(f"UWAGA: pominięto {dropped} powtórzonych przebiegów (zostaje najnowszy dla klucza seria/zbiór/wariant/seed)")
        runs_df = keep.sort_values(["series", "dataset", "variant", "seed"]).reset_index(drop=True)
        steps_df = steps_df[steps_df.run_id.isin(set(runs_df.run_id))].reset_index(drop=True)
    return runs_df, steps_df


SUMMARY_METRICS = ["test_ba", "fullv_ba", "rep_vprime_ba", "vprime_rows", "vprime_fraction", "final_k", "steps",
                   "tprime_steps", "vprime_steps", "vprime_phases", "vprime_retries", "mean_delta_difficulty",
                   "train_ga_svm_evaluations", "duration_sec"]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["series", "dataset", "variant"])
    out = g[SUMMARY_METRICS].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "n", g.size())
    out["cap_reached_share"] = g.vprime_cap_reached.mean()
    out["stops"] = g.stop_decision.agg(lambda s: json.dumps(s.value_counts().to_dict(), sort_keys=True))
    return out.reset_index()


PAIRS = [("evo_switch", "rand_replay"), ("evo_switch", "rand_switch")]


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    """a − b na tych samych seedach; para główna evo_switch − rand_replay (identyczny harmonogram |V'|)."""
    rows = []
    for (series, dataset), grp in runs.groupby(["series", "dataset"]):
        for a, b in PAIRS:
            e = grp[grp.variant == a].set_index("seed")
            r = grp[grp.variant == b].set_index("seed")
            common = sorted(set(e.index) & set(r.index))
            if not common:
                continue
            for metric in ("test_ba", "fullv_ba", "vprime_rows", "vprime_fraction", "steps", "final_k",
                           "mean_delta_difficulty"):
                d = (e.loc[common, metric] - r.loc[common, metric]).astype(float)
                rows.append({"series": series, "dataset": dataset, "a": a, "b": b, "metric": metric, "n": len(common),
                             "mean_a": float(e.loc[common, metric].mean()), "mean_b": float(r.loc[common, metric].mean()),
                             "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                             "a_larger": int((d > 0).sum()), "b_larger": int((d < 0).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v1")
    version = ap.parse_args().version
    out_dir = TABLES_V2 if version == "v2" else TABLES
    runs, steps = collect(version)
    if runs.empty:
        raise SystemExit(f"brak przebiegów dla wersji {version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "per_run.csv", index=False)
    steps.to_csv(out_dir / "per_step.csv", index=False)
    summarize(runs).to_csv(out_dir / "summary.csv", index=False)
    paired(runs).to_csv(out_dir / "paired.csv", index=False)
    print(f"[{version}] przebiegów: {len(runs)}; kroków: {len(steps)} -> {out_dir}")


if __name__ == "__main__":
    main()
