"""Agreguje wyniki badania `followup_stop_rule` do tabel CSV w reports/followup_stop_rule/tables/.

Źródło: wyłącznie pliki per-run (summary.json + cycles.csv) serii nowych (artifacts/followup_stop_rule) i — jako stan „przed zmianą
reguły stopu” — serii badania V' (artifacts/study, podział bazowy 2026, te same seedy 42–61).

    python scripts/followup_stop_rule/aggregate.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bootstrap_ci, describe, load_series, paired_comparison, variance_components  # noqa: E402
from common import (  # noqa: E402
    ALGO_SEEDS_BASE, ALGO_SEEDS_EXTRA, BASE_SPLIT, BUDGET_TWIN, DATASETS, LEGACY_ARTIFACTS, MAIN_DATASETS,
    MAIN_VARIANTS, SPLIT_SEEDS, TABLES, VARIANTS, planned_series, series_root, split_dir,
)

SUMMARY_METRICS = [
    "test_ba", "fullv_ba", "vprime_ba", "cycles", "final_k", "vprime_size", "training_rows", "num_support_vectors",
    "vprime_expansions", "k_increases", "perfect_ba_stop", "premature_stop", "svm_fits_total", "tga_fit_rows",
    "tga_vprime_rows_scored", "selection_predict_rows", "selection_decision_rows", "val_ga_nominal_evals",
    "val_ga_nominal_subset_rows", "candidates_total", "scored_pool_total", "increment_rows_total", "n_expansions_executed",
    "mean_increment_error_rate", "mean_pool_error_rate", "duration_sec", "perfect_cycles", "gate_pending_cycles",
]
PAIRED_METRICS = [
    "test_ba", "fullv_ba", "vprime_ba", "cycles", "final_k", "vprime_size", "svm_fits_total", "tga_vprime_rows_scored",
    "selection_predict_rows", "selection_decision_rows", "increment_rows_total", "mean_increment_error_rate",
    "num_support_vectors", "perfect_ba_stop",
]
VARIANT_PAIRS = [
    ("evo_mem", "rand_growth"), ("evo_mem", "heur_mem"), ("evo_mem", "rand_fixed"), ("evo_mem", "evo_nomem"),
    ("heur_mem", "rand_growth"), ("evo_mem_bud", "rand_growth_bud"), ("evo_mem_bud", "heur_mem_bud"),
    ("heur_mem_bud", "rand_growth_bud"),
]
CYCLE_COLUMNS = [
    "seed", "split", "dataset", "variant", "rule", "cycle", "k_per_class", "k_after", "decision", "val_rows_before", "val_rows_after",
    "val_rows_added", "perfect_on_current_val", "train_ga_best_val_metric", "train_ga_test_metric",
    "train_ga_best_full_validation_metric", "num_support_vectors", "increment_error_rate", "candidate_pool_error_rate",
    "train_rows", "perfect_gate_pending", "perfect_stop_blocked", "perfect_stop_blocked_reason",
    "min_vprime_size_reached", "perfect_rechecks_done", "vprime_min_class_count", "stop_category",
]


def _tag(frame: pd.DataFrame, **cols) -> pd.DataFrame:
    for i, (k, v) in enumerate(cols.items()):
        frame.insert(i, k, v)
    return frame


def load_all():
    runs, cycles, status = [], [], []
    for split, dataset, variant, seeds in planned_series(include_pilot=True):
        root = series_root(split, dataset, variant)
        per_run, per_cycle, audit = load_series(root, dataset, variant)
        done = sorted(per_run["seed"].tolist()) if not per_run.empty else []
        ok = [s for s in done if s in seeds]
        status.append({
            "split": split, "dataset": dataset, "variant": variant, "planned": len(seeds), "completed": len(ok),
            "missing_seeds": json.dumps([s for s in seeds if s not in done]),
            "extra_seeds": json.dumps([s for s in done if s not in seeds]),
            "failed_jobs": json.dumps(audit["failed_jobs"]), "running_or_stale": json.dumps(audit["stale_running"]),
            "duplicates": json.dumps(audit["duplicates"]), "complete": len(ok) == len(seeds) and not audit["failed_jobs"],
            "pilot": dataset == "magic_full",
        })
        if not per_run.empty:
            runs.append(_tag(per_run, split=split, dataset=dataset, variant=variant, rule="extended"))
            cycles.append(_tag(per_cycle, split=split, dataset=dataset, variant=variant, rule="extended"))
    legacy_runs, legacy_cycles = [], []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            per_run, per_cycle, _ = load_series(LEGACY_ARTIFACTS / dataset / variant, dataset, variant, legacy=True)
            if not per_run.empty:
                legacy_runs.append(_tag(per_run, split=BASE_SPLIT, dataset=dataset, variant=variant, rule="legacy"))
                legacy_cycles.append(_tag(per_cycle, split=BASE_SPLIT, dataset=dataset, variant=variant, rule="legacy"))
    cat = lambda fs: pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()
    return cat(runs), cat(cycles), pd.DataFrame(status), cat(legacy_runs), cat(legacy_cycles)


def invariants(per_run: pd.DataFrame, per_cycle: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name, violations, total):
        rows.append({"invariant": name, "violations": int(violations), "checked": int(total)})

    add("wzrost K więcej niż raz w cyklu", (per_cycle.k_increases_this_cycle > 1).sum(), len(per_cycle))
    add("K wzrosło w cyklu z BA(V')=1,0", ((per_cycle.perfect_on_current_val.astype(bool)) & (per_cycle.k_after > per_cycle.k_per_class)).sum(), len(per_cycle))
    perfect = per_run[per_run.perfect_ba_stop & (per_run.variant != "rand_fixed")]
    last = per_cycle.sort_values("cycle").groupby(["split", "dataset", "variant", "seed"]).tail(1).set_index(["split", "dataset", "variant", "seed"])
    keyed = perfect.set_index(["split", "dataset", "variant", "seed"])
    joined = keyed.join(last[["min_vprime_size_reached", "perfect_rechecks_done", "val_rows_before"]], rsuffix="_c")
    add("stop BA=1,0 przy V' poniżej minimum", (~joined.min_vprime_size_reached.astype(bool)).sum(), len(joined))
    add("stop BA=1,0 bez obowiązkowego ponownego sprawdzenia", (joined.perfect_rechecks_done < 1).sum(), len(joined))
    add("przedwczesny stop (def. post hoc) w przebiegach nowej reguły", per_run.premature_stop.sum(), len(per_run))
    add("przebiegi o statusie innym niż completed", (per_run.job_status.fillna("completed") != "completed").sum(), len(per_run))
    add("przebiegi zakończone limitem cykli (bezpiecznik)", (per_run.stop_category == "max_cycles").sum(), len(per_run))
    return pd.DataFrame(rows)


def config_consistency() -> pd.DataFrame:
    """Czy w obrębie (podział × zbiór × seed) warianty mają identyczne dane, SVM, GA T', politykę K i regułę stopu."""
    rows = []
    for split in SPLIT_SEEDS:
        for dataset in MAIN_DATASETS + ["magic_full"]:
            cfgs = {}
            for variant in VARIANTS:
                for p in sorted((series_root(split, dataset, variant) / "configs").glob("seed_*.yaml")):
                    import yaml

                    c = yaml.safe_load(p.read_text(encoding="utf-8"))
                    seed = int(c["seed"])
                    key = {"data": c["data"], "svm": c["svm"], "ga": c["ga"], "daga": c["daga"], "val_ga": c["val_ga"],
                           "observation": c["observation"], "fitness": c["fitness"]}
                    cfgs.setdefault(seed, {})[variant] = json.dumps(key, sort_keys=True)
            for seed, by_variant in cfgs.items():
                if len(by_variant) > 1:
                    rows.append({"split": split, "dataset": dataset, "seed": seed, "variants": len(by_variant),
                                 "identical": len(set(by_variant.values())) == 1})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (frame.groupby(["split", "dataset"]).agg(seeds=("seed", "count"), variants_max=("variants", "max"),
                                                    all_identical=("identical", "all")).reset_index())


def split_hash_table() -> pd.DataFrame:
    rows = []
    for split in SPLIT_SEEDS:
        for dataset in MAIN_DATASETS + ["magic_full"]:
            meta = split_dir(split) / dataset / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text(encoding="utf-8"))
                rows.append({"split": split, "dataset": dataset, "train": m["splits"]["train"]["rows"],
                             "validation": m["splits"]["validation"]["rows"], "test": m["splits"]["test"]["rows"],
                             "features": m["features"], "pool_rows": m.get("pool_rows"),
                             "identical_to_legacy": m.get("identical_to_legacy_data_study"),
                             "scaler_fitted_on": (m.get("scaler") or {}).get("fitted_on")})
    return pd.DataFrame(rows)


def svm_params_table() -> pd.DataFrame:
    """Diagnostyka (C, gamma): wybór per podział vs parametry wspólne (z podziału bazowego) na siatce danego podziału."""
    rows = []
    base = json.loads((split_dir(BASE_SPLIT) / "svm_params.json").read_text(encoding="utf-8"))
    for split in SPLIT_SEEDS:
        path = split_dir(split) / "svm_params.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dataset, entry in payload.items():
            if dataset not in base or dataset == "magic_full":
                continue
            shared = next(g for g in entry["grid"] if g["C"] == base[dataset]["C"] and g["gamma"] == base[dataset]["gamma"])
            rows.append({"split": split, "dataset": dataset, "C_split_best": entry["C"], "gamma_split_best": entry["gamma"],
                         "ba_split_best": entry["full_svm_val_ba"], "C_shared": base[dataset]["C"],
                         "gamma_shared": base[dataset]["gamma"], "ba_shared_params": shared["validation_balanced_accuracy"],
                         "ba_loss_shared": entry["full_svm_val_ba"] - shared["validation_balanced_accuracy"]})
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    per_run, per_cycle, status, leg_run, leg_cycle = load_all()
    per_run.to_csv(TABLES / "per_run.csv", index=False)
    leg_run.to_csv(TABLES / "per_run_legacy.csv", index=False)
    keep = lambda df: df[[c for c in CYCLE_COLUMNS if c in df.columns]]
    keep(per_cycle).to_csv(TABLES / "per_cycle.csv", index=False)
    keep(leg_cycle).to_csv(TABLES / "per_cycle_legacy.csv", index=False)
    status.to_csv(TABLES / "series_status.csv", index=False)
    svm_params_table().to_csv(TABLES / "svm_params_by_split.csv", index=False)
    split_hash_table().to_csv(TABLES / "split_datasets.csv", index=False)
    config_consistency().to_csv(TABLES / "config_consistency.csv", index=False)
    if per_run.empty:
        print("brak ukończonych przebiegów")
        return
    invariants(per_run, per_cycle).to_csv(TABLES / "invariants.csv", index=False)

    # ---------------- statystyki serii (nowa reguła; także stara jako osobny plik) ----------------
    def summarize(frame, keys):
        rows = []
        for key, grp in frame.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            for metric in SUMMARY_METRICS:
                if metric in grp:
                    rows.append({**dict(zip(keys, key)), "metric": metric, **describe(grp[metric].astype(float))})
        return pd.DataFrame(rows)

    summarize(per_run, ["split", "dataset", "variant"]).to_csv(TABLES / "summary_by_series.csv", index=False)
    summarize(leg_run, ["split", "dataset", "variant"]).to_csv(TABLES / "summary_by_series_legacy.csv", index=False)

    # ---------------- przyczyny zakończenia, przed i po ----------------
    both = pd.concat([leg_run, per_run[per_run.split == BASE_SPLIT]], ignore_index=True)
    both = both[both.variant.isin(MAIN_VARIANTS)]
    cat = (both.groupby(["rule", "dataset", "variant", "stop_category"]).size().rename("n").reset_index())
    cat.to_csv(TABLES / "stop_categories.csv", index=False)
    rows = []
    for (dataset, variant), grp in both.groupby(["dataset", "variant"]):
        old, new = grp[grp.rule == "legacy"], grp[grp.rule == "extended"]
        rows.append({
            "dataset": dataset, "variant": variant, "n_old": len(old), "n_new": len(new),
            "ba1_stops_old": int(old.perfect_ba_stop.sum()), "ba1_stops_new": int(new.perfect_ba_stop.sum()),
            "premature_old": int(old.premature_stop.sum()), "premature_new": int(new.premature_stop.sum()),
            "median_vprime_at_ba1_stop_old": float(old[old.perfect_ba_stop].vprime_size.median()) if old.perfect_ba_stop.any() else np.nan,
            "median_vprime_at_ba1_stop_new": float(new[new.perfect_ba_stop].vprime_size.median()) if new.perfect_ba_stop.any() else np.nan,
        })
    pd.DataFrame(rows).to_csv(TABLES / "stop_before_after.csv", index=False)

    # ---------------- przed vs po (te same seedy, ten sam podział) ----------------
    paired_ba = []
    for (dataset, variant), grp in both.groupby(["dataset", "variant"]):
        old = grp[grp.rule == "legacy"].set_index("seed")
        new = grp[grp.rule == "extended"].set_index("seed")
        if len(new) < 3:
            continue
        for metric in PAIRED_METRICS:
            paired_ba.append({"dataset": dataset, "variant": variant, "metric": metric,
                              **paired_comparison(new[metric].astype(float), old[metric].astype(float))})
        # podzbiór przebiegów, które w starej serii zakończyły się przedwcześnie
        prem = old.index[old.premature_stop & old.index.isin(new.index)]
        if len(prem) >= 1:
            paired_ba.append({"dataset": dataset, "variant": variant, "metric": "test_ba|przedwczesne_w_starej",
                              "n": len(prem), "mean_diff": float((new.loc[prem, "test_ba"] - old.loc[prem, "test_ba"]).mean())})
    pd.DataFrame(paired_ba).to_csv(TABLES / "paired_before_after.csv", index=False)

    # ---------------- porównania wariantów (podział bazowy: 20 par; podziały: 5 par) ----------------
    rows = []
    for (split, dataset), grp in per_run.groupby(["split", "dataset"]):
        for a_name, b_name in VARIANT_PAIRS:
            a = grp[grp.variant == a_name].set_index("seed")
            b = grp[grp.variant == b_name].set_index("seed")
            if len(a) < 3 or len(b) < 3:
                continue
            for metric in PAIRED_METRICS:
                rows.append({"split": split, "dataset": dataset, "a": a_name, "b": b_name, "metric": metric,
                             **paired_comparison(a[metric].astype(float), b[metric].astype(float))})
    pd.DataFrame(rows).to_csv(TABLES / "paired_variants.csv", index=False)

    # ---------------- porównania wariantów ze starej reguły (dla odniesienia) ----------------
    rows = []
    for dataset, grp in leg_run.groupby("dataset"):
        for a_name, b_name in VARIANT_PAIRS[:4]:
            a = grp[grp.variant == a_name].set_index("seed")
            b = grp[grp.variant == b_name].set_index("seed")
            if len(a) < 3 or len(b) < 3:
                continue
            for metric in ("test_ba", "cycles", "final_k", "vprime_size", "svm_fits_total"):
                rows.append({"dataset": dataset, "a": a_name, "b": b_name, "metric": metric,
                             **paired_comparison(a[metric].astype(float), b[metric].astype(float))})
    pd.DataFrame(rows).to_csv(TABLES / "paired_variants_legacy.csv", index=False)

    # ---------------- zmienność: seedy algorytmu vs podział danych ----------------
    common = per_run[per_run.seed.isin(ALGO_SEEDS_EXTRA) & per_run.dataset.isin([d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]])
                     & per_run.variant.isin(MAIN_VARIANTS)]
    vc_rows, diff_rows = [], []
    for (dataset, variant), grp in common.groupby(["dataset", "variant"]):
        if grp.split.nunique() < 2:
            continue
        for metric in ("test_ba", "cycles", "final_k", "vprime_size", "svm_fits_total"):
            vc_rows.append({"dataset": dataset, "variant": variant, "metric": metric,
                            **variance_components(grp, metric),
                            "mean_by_split": json.dumps({int(s): round(float(v), 5) for s, v in grp.groupby("split")[metric].mean().items()})})
    pd.DataFrame(vc_rows).to_csv(TABLES / "variance_components.csv", index=False)
    # różnice wariantów: efekt podziału vs efekt seedu (hierarchiczny bootstrap po podziałach i seedach)
    rng = np.random.default_rng(0)
    for dataset in [d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]]:
        grp = per_run[(per_run.dataset == dataset) & per_run.seed.isin(ALGO_SEEDS_EXTRA)]
        for a_name, b_name in VARIANT_PAIRS[:4] + VARIANT_PAIRS[5:]:
            for metric in ("test_ba", "cycles", "svm_fits_total", "final_k"):
                a = grp[grp.variant == a_name].set_index(["split", "seed"])[metric]
                b = grp[grp.variant == b_name].set_index(["split", "seed"])[metric]
                d = (a - b).dropna()
                if d.empty or d.index.get_level_values("split").nunique() < 2:
                    continue
                per_split = d.groupby(level="split").mean()
                per_split_n = d.groupby(level="split").size()
                splits = per_split.index.to_numpy()
                table = {s: d.xs(s, level="split").to_numpy() for s in splits}
                boots = np.empty(10000)
                for i in range(10000):
                    pick = rng.choice(splits, size=len(splits), replace=True)
                    boots[i] = np.mean([rng.choice(table[s], size=len(table[s]), replace=True).mean() for s in pick])
                diff_rows.append({
                    "dataset": dataset, "a": a_name, "b": b_name, "metric": metric, "n_splits": int(len(splits)),
                    "n_seeds_per_split": int(per_split_n.min()), "mean_diff": float(per_split.mean()),
                    "sd_of_split_means": float(per_split.std(ddof=1)) if len(per_split) > 1 else np.nan,
                    "sd_within_split_diff": float(np.mean([table[s].std(ddof=1) for s in splits])),
                    "ci95_hier_low": float(np.percentile(boots, 2.5)), "ci95_hier_high": float(np.percentile(boots, 97.5)),
                    "splits_a_better": int((per_split > 0).sum()), "splits_b_better": int((per_split < 0).sum()),
                    "per_split_mean_diff": json.dumps({int(s): round(float(v), 5) for s, v in per_split.items()}),
                })
    pd.DataFrame(diff_rows).to_csv(TABLES / "variant_diff_across_splits.csv", index=False)

    # ---------------- pamięć V': rozrzut jakości z pamięcią i bez ----------------
    from scipy import stats

    rows = []
    frames = [("legacy", leg_run[leg_run.split == BASE_SPLIT]), ("extended", per_run[per_run.split == BASE_SPLIT])]
    for rule, frame in frames:
        for dataset in MAIN_DATASETS:
            a = frame[(frame.dataset == dataset) & (frame.variant == "evo_mem")]
            b = frame[(frame.dataset == dataset) & (frame.variant == "evo_nomem")]
            if len(a) < 5 or len(b) < 5:
                continue
            for metric in ("test_ba", "cycles", "final_k", "vprime_size"):
                x, y = a[metric].astype(float).to_numpy(), b[metric].astype(float).to_numpy()
                boots = []
                rng2 = np.random.default_rng(1)
                for _ in range(5000):
                    xs, ys = rng2.choice(x, len(x)), rng2.choice(y, len(y))
                    if xs.std(ddof=1) > 0:
                        boots.append(ys.std(ddof=1) / xs.std(ddof=1))
                try:
                    p_bf = float(stats.levene(x, y, center="median").pvalue)
                except ValueError:
                    p_bf = np.nan
                rows.append({"rule": rule, "dataset": dataset, "metric": metric, "n_mem": len(x), "n_nomem": len(y),
                             "mean_mem": float(x.mean()), "mean_nomem": float(y.mean()), "sd_mem": float(x.std(ddof=1)),
                             "sd_nomem": float(y.std(ddof=1)),
                             "sd_ratio_nomem_over_mem": float(y.std(ddof=1) / x.std(ddof=1)) if x.std(ddof=1) > 0 else np.nan,
                             "sd_ratio_ci_low": float(np.percentile(boots, 2.5)) if boots else np.nan,
                             "sd_ratio_ci_high": float(np.percentile(boots, 97.5)) if boots else np.nan,
                             "brown_forsythe_p": p_bf,
                             "iqr_mem": float(np.subtract(*np.percentile(x, [75, 25]))),
                             "iqr_nomem": float(np.subtract(*np.percentile(y, [75, 25])))})
    # rozrzut wewnątrz podziałów (zagregowany po podziałach, 5 wspólnych seedów)
    for dataset in [d for d in MAIN_DATASETS if DATASETS[d]["split_dependent"]]:
        a = per_run[(per_run.dataset == dataset) & (per_run.variant == "evo_mem") & per_run.seed.isin(ALGO_SEEDS_EXTRA)]
        b = per_run[(per_run.dataset == dataset) & (per_run.variant == "evo_nomem") & per_run.seed.isin(ALGO_SEEDS_EXTRA)]
        if a.split.nunique() < 2 or b.split.nunique() < 2:
            continue
        sd_a = np.sqrt(a.groupby("split").test_ba.var(ddof=1).mean())
        sd_b = np.sqrt(b.groupby("split").test_ba.var(ddof=1).mean())
        rows.append({"rule": "extended_within_splits", "dataset": dataset, "metric": "test_ba", "n_mem": len(a), "n_nomem": len(b),
                     "mean_mem": float(a.test_ba.mean()), "mean_nomem": float(b.test_ba.mean()), "sd_mem": float(sd_a),
                     "sd_nomem": float(sd_b), "sd_ratio_nomem_over_mem": float(sd_b / sd_a) if sd_a > 0 else np.nan})
    pd.DataFrame(rows).to_csv(TABLES / "memory_spread.csv", index=False)

    # ---------------- budżet doboru V': seria z limitem vs pełna pula ----------------
    rows = []
    base = per_run[per_run.split == BASE_SPLIT]
    for dataset in MAIN_DATASETS:
        for bud, twin in BUDGET_TWIN.items():
            a = base[(base.dataset == dataset) & (base.variant == bud)].set_index("seed")
            b = base[(base.dataset == dataset) & (base.variant == twin)].set_index("seed")
            if len(a) < 3:
                continue
            seeds = sorted(set(a.index) & set(b.index))
            row = {"dataset": dataset, "variant": bud, "twin": twin, "n": len(seeds)}
            for metric in ("test_ba", "cycles", "final_k", "vprime_size", "svm_fits_total", "selection_predict_rows",
                           "selection_decision_rows", "tga_vprime_rows_scored", "scored_pool_total", "candidates_total",
                           "increment_rows_total"):
                row[f"{metric}_bud"] = float(a.loc[seeds, metric].mean())
                row[f"{metric}_full"] = float(b.loc[seeds, metric].mean())
            res = paired_comparison(a.loc[seeds, "test_ba"], b.loc[seeds, "test_ba"])
            row.update({"test_ba_diff": res.get("mean_diff"), "test_ba_ci_low": res.get("ci95_low"),
                        "test_ba_ci_high": res.get("ci95_high"), "test_ba_wilcoxon_p": res.get("wilcoxon_p")})
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES / "budget_comparison.csv", index=False)

    # ---------------- retrospektywne liczniki ----------------
    print(status.groupby("complete").size().to_string())
    inc = status[~status.complete & ~status.pilot]
    print(f"serie kompletne: {int(status.complete.sum())}/{len(status)} (w tym pilotaż); niekompletne w planie głównym: {len(inc)}")
    print(f"zapisano tabele w {TABLES}")


if __name__ == "__main__":
    main()
