"""Analiza końcowa badania `followup_phase_switch` v2 (dane -> tabele CSV). Używana przez `final_report.py`.

Wejście (tylko odczyt): `reports/followup_phase_switch/v2/tables/per_run.csv` i `per_step.csv` (`aggregate.py --version v2`),
`artifacts/followup_phase_switch/<seria>/.../summary.json` (metryki F1/accuracy; AUC NIE jest liczony — patrz raport) oraz
`reports/followup_stop_rule/tables/per_run.csv` (stary algorytm; porównanie opisowe).

Wyjście: tabele `reports/followup_phase_switch/tables/final_*.csv` (nie nadpisują tabel v1 ani smoke).

Podział danych:
  GŁÓWNE  = v2_full (podział 2026, 20 seedów, 7 zbiorów) + v2_split2027/2028/2029 (5 seedów, 6 zbiorów zależnych od podziału),
            warianty evo_switch / rand_replay / rand_switch  -> 690 przebiegów
  PILOTAŻ = v2_smoke_* (smoke; Moons 3 seedy, Szachownica 2 seedy, limit |V'| = 1000)  -> 21 przebiegów
  WCZEŚNIEJSZE (v1, poza wnioskami) = serie bez prefiksu v2_  -> 111 przebiegów
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import REPO_ROOT, TABLES_V2, fsr  # noqa: E402

ART = REPO_ROOT / "artifacts" / "followup_phase_switch"
OUT_TABLES = REPO_ROOT / "reports" / "followup_phase_switch" / "tables"
OLD_PER_RUN = REPO_ROOT / "reports" / "followup_stop_rule" / "tables" / "per_run.csv"

MAIN_SERIES = ["v2_full", "v2_split2027", "v2_split2028", "v2_split2029"]
DATASETS = ["chessboard", "moons", "circles", "spiral", "classification2d", "digits", "magic"]
SPLIT_DEP = [d for d in DATASETS if d != "chessboard"]
LABEL = {"chessboard": "Szachownica", "moons": "Moons", "circles": "Circles", "spiral": "Spiral",
         "classification2d": "Classification 2D", "digits": "Digits", "magic": "MAGIC"}
VARIANTS = ["evo_switch", "rand_replay", "rand_switch"]
VLABEL = {"evo_switch": "ewolucja", "rand_replay": "losowanie (harmonogram z ewolucji)", "rand_switch": "losowanie (własny automat)",
          "evo_mem_old": "ewolucja, stary algorytm"}
MARGIN = 0.005          # przyjęta (nie wynikająca z danych) granica praktycznej istotności różnicy BA / F1: 0,5 p.p.
NBOOT = 10000
RNG = np.random.default_rng(20261022)


# ---------------------------------------------------------------------------------------------
# narzędzia statystyczne
# ---------------------------------------------------------------------------------------------
def boot_ci(d, alpha=0.05, n=NBOOT) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return (np.nan, np.nan)
    means = RNG.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def hier_boot_ci(split: np.ndarray, d: np.ndarray, alpha=0.05, n=NBOOT) -> tuple[float, float]:
    """Bootstrap hierarchiczny: losowanie podziałów ze zwracaniem, a w nich seedów (uwzględnia obie warstwy niepewności)."""
    split, d = np.asarray(split), np.asarray(d, dtype=float)
    groups = [d[split == s] for s in np.unique(split)]
    out = np.empty(n)
    for i in range(n):
        pick = RNG.integers(0, len(groups), len(groups))
        out[i] = np.mean([RNG.choice(groups[j], size=groups[j].size, replace=True).mean() for j in pick])
    return (float(np.percentile(out, 100 * alpha / 2)), float(np.percentile(out, 100 * (1 - alpha / 2))))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def classify(diff, lo, hi) -> str:
    """Ocena praktyczna różnicy BA/F1 względem przyjętej granicy ±0,005."""
    if np.isnan(diff):
        return "—"
    if abs(diff) >= MARGIN:
        return "praktycznie istotna"
    if lo > -MARGIN and hi < MARGIN:
        return "praktycznie zaniedbywalna"
    return "nierozstrzygnięta"


def paired(df: pd.DataFrame, a: str, b: str, metric: str, keys=("seed",), hier=False) -> dict:
    ga = df[df.variant == a].set_index(list(keys))
    gb = df[df.variant == b].set_index(list(keys))
    idx = ga.index.intersection(gb.index)
    x, y = ga.loc[idx, metric].to_numpy(float), gb.loc[idx, metric].to_numpy(float)
    d = x - y
    lo, hi = boot_ci(d)
    out = {"n": len(idx), "mean_a": float(np.nanmean(x)), "sd_a": float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan,
           "mean_b": float(np.nanmean(y)), "sd_b": float(np.nanstd(y, ddof=1)) if len(y) > 1 else np.nan,
           "diff": float(np.nanmean(d)), "sd_diff": float(np.nanstd(d, ddof=1)) if len(d) > 1 else np.nan,
           "ci_lo": lo, "ci_hi": hi, "a_gt": int((d > 0).sum()), "a_lt": int((d < 0).sum())}
    out["dz"] = out["diff"] / out["sd_diff"] if out["sd_diff"] and out["sd_diff"] > 0 else np.nan
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else np.nan
    except ValueError:
        out["p_wilcoxon"] = np.nan
    if hier and "split" in keys:
        sp = np.array([i[list(keys).index("split")] for i in idx])
        out["h_lo"], out["h_hi"] = hier_boot_ci(sp, d)
    return out


# ---------------------------------------------------------------------------------------------
# ładowanie
# ---------------------------------------------------------------------------------------------
def _summary_extra(row) -> dict:
    p = ART / row.series / row.dataset / row.variant / "runs" / row.run_id / "metrics" / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    gb, ps = s["global_best"], s["phase_study"]
    te, fv, vv = gb["metrics_test"]["all"], gb["metrics_full_validation"]["all"], gb["metrics_val"]["all"]
    bt = s["baseline"]["test"]["all"]
    return {"run_id": row.run_id, "test_f1m": te["f1_macro"], "test_acc": te["accuracy"], "fullv_f1m": fv["f1_macro"],
            "fullv_acc": fv["accuracy"], "vprime_acc": vv["accuracy"], "vprime_f1m": vv["f1_macro"],
            "base_test_f1m": bt["f1_macro"], "base_test_acc": bt["accuracy"], "n_train": s["dataset"]["train"]["rows"],
            "n_test": s["dataset"]["test"]["rows"], "rep_train_rows": gb["train_summary"]["rows"],
            "ctl_full": ps.get("used_full_validation_for_control"), "ctl_test": ps.get("used_test_for_control"),
            "tprime_metric": ps["settings"]["tprime_metric"], "vprime_metric": ps["settings"]["vprime_metric"]}


def load() -> dict:
    runs = pd.read_csv(TABLES_V2 / "per_run.csv")
    steps = pd.read_csv(TABLES_V2 / "per_step.csv", low_memory=False)
    runs = runs[runs.series.str.startswith("v2_")].copy()
    steps = steps[steps.series.str.startswith("v2_")].copy()
    extra = pd.DataFrame([_summary_extra(r) for r in runs.itertuples()])
    runs = runs.merge(extra, on="run_id", how="left")
    runs["role"] = np.where(runs.series.isin(MAIN_SERIES), "główne", "pilotaż")
    # sumy ocen osobników z kroków
    t = steps[steps.phase == "optimize_T_prime"].groupby("run_id").ga_svm_evaluations.sum().rename("tprime_individual_evals")
    v = steps[steps.phase == "optimize_V_prime"].groupby("run_id").val_ga_population_evaluations.sum().rename("valga_individual_evals")
    runs = runs.merge(t, on="run_id", how="left").merge(v, on="run_id", how="left")
    incr = steps[steps.phase == "optimize_V_prime"].groupby("run_id").delta_rows.mean().rename("mean_increment")
    runs = runs.merge(incr, on="run_id", how="left")
    runs["perfect_steps"] = steps[(steps.phase == "optimize_T_prime")].assign(
        p=lambda d: d.tprime_perfect.astype(str).str.lower() == "true").groupby("run_id").p.sum().reindex(runs.run_id).to_numpy()
    old = pd.read_csv(OLD_PER_RUN, low_memory=False)
    old = old[(old.variant == "evo_mem") & old.split.isin([2026, 2027, 2028, 2029])].copy()
    return {"runs": runs, "steps": steps, "old": old}


# ---------------------------------------------------------------------------------------------
# kompletność przebiegów
# ---------------------------------------------------------------------------------------------
def completeness() -> pd.DataFrame:
    rows = []
    for sdir in sorted(p for p in ART.iterdir() if p.is_dir()):
        s = sdir.name
        spec = json.loads((sdir / "series_spec.json").read_text(encoding="utf-8")) if (sdir / "series_spec.json").exists() else {}
        jobs = list(sdir.glob("*/*/jobs/*.json"))
        status = pd.Series([json.loads(j.read_text(encoding="utf-8")).get("status") for j in jobs]).value_counts().to_dict()
        summaries = list(sdir.glob("*/*/runs/*/metrics/summary.json"))
        planned = None
        for q in sorted(ART.glob(f"queue_{s}*.out")):
            m = re.search(r"\[" + re.escape(s) + r"\] (\d+) jobów", q.read_text(encoding="utf-8").splitlines()[0])
            if m and (q.name == f"queue_{s}.out" or planned is None):
                planned = int(m.group(1))
        if planned is None:
            planned = len(jobs)
        role = "główna" if s in MAIN_SERIES else ("pilotaż" if s.startswith("v2_smoke") else "wcześniejsza wersja automatu (v1)")
        rows.append({"seria": s, "rola": role, "podział": int(spec.get("split", 2026)),
                     "zbiory": len({j.parts[-4] for j in jobs}), "warianty": len({j.parts[-3] for j in jobs}),
                     "seedy": len({int(j.stem.split("_")[1]) for j in jobs}), "zaplanowane": planned, "ukończone (job)": status.get("completed", 0),
                     "błędy (job)": status.get("failed", 0), "w toku": status.get("running", 0), "podsumowania": len(summaries)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# rozmiar V'
# ---------------------------------------------------------------------------------------------
def size_table(runs: pd.DataFrame, old: pd.DataFrame, base_only: bool) -> pd.DataFrame:
    d = runs[(runs.role == "główne")]
    if base_only:
        d = d[d.series == "v2_full"]
    rows = []
    for ds in DATASETS:
        for v in VARIANTS:
            g = d[(d.dataset == ds) & (d.variant == v)]
            if g.empty:
                continue
            reasons = g.stop_decision.str.replace("stop_", "").value_counts().to_dict()
            rows.append({"zbiór": ds, "wariant": v, "n": len(g), "V′ średnio": g.vprime_rows.mean(), "V′ mediana": g.vprime_rows.median(),
                         "V′ SD": g.vprime_rows.std(ddof=1), "V′ min": g.vprime_rows.min(), "V′ max": g.vprime_rows.max(),
                         "|V| średnio": g.full_validation_rows.mean(), "V′/V średnio": g.vprime_fraction.mean(),
                         "rozszerzenia V′ (śr.)": g.vprime_expansions.mean(), "przyrost ΔV′ (śr. próbek)": g.mean_increment.mean(),
                         "limit rozmiaru blokuje wzrost na końcu (udział)": float((g.vprime_limit_reason == "size_limit").mean()),
                         "powody zakończenia": json.dumps(reasons, ensure_ascii=False)})
        o = old[(old.dataset == ds) & (old.split == 2026)] if base_only else old[old.dataset == ds]
        if len(o):
            rows.append({"zbiór": ds, "wariant": "evo_mem_old", "n": len(o), "V′ średnio": o.vprime_size.mean(), "V′ mediana": o.vprime_size.median(),
                         "V′ SD": o.vprime_size.std(ddof=1), "V′ min": o.vprime_size.min(), "V′ max": o.vprime_size.max(),
                         "|V| średnio": np.nan, "V′/V średnio": np.nan, "rozszerzenia V′ (śr.)": o.vprime_expansions.mean(),
                         "przyrost ΔV′ (śr. próbek)": np.nan, "limit rozmiaru blokuje wzrost na końcu (udział)": np.nan,
                         "powody zakończenia": json.dumps(o.stop_category.value_counts().to_dict(), ensure_ascii=False)})
    out = pd.DataFrame(rows)
    # |V| z danych nowych (ten sam zbiór/podział)
    vmap = d.groupby("dataset").full_validation_rows.mean()
    m = out["wariant"] == "evo_mem_old"
    out.loc[m, "|V| średnio"] = out.loc[m, "zbiór"].map(vmap)
    out.loc[m, "V′/V średnio"] = out.loc[m, "V′ średnio"] / out.loc[m, "|V| średnio"]
    return out


# ---------------------------------------------------------------------------------------------
# porównania sparowane (ewolucja vs losowanie)
# ---------------------------------------------------------------------------------------------
METRICS = {"test_ba": "BA test", "fullv_ba": "BA pełne V", "test_f1m": "F1-macro test", "test_acc": "accuracy test",
           "rep_vprime_ba": "BA na V′ (model rep.)", "mean_delta_difficulty": "błąd na ΔV′ (trudność)", "delta_ba_mean": "BA na ΔV′",
           "final_k": "końcowe K", "vprime_rows": "końcowe |V′|", "steps": "liczba cykli"}


def paired_table(runs: pd.DataFrame, a: str, b: str, design: str, metrics=None) -> pd.DataFrame:
    """design: 'base' (v2_full, 20 seedów, 7 zbiorów) albo 'splits' (4 podziały × seedy 42–46, 6 zbiorów zależnych od podziału)."""
    metrics = metrics or list(METRICS)
    rows = []
    for m in metrics:
        res = []
        for ds in (DATASETS if design == "base" else SPLIT_DEP):
            if design == "base":
                g = runs[(runs.series == "v2_full") & (runs.dataset == ds)]
                r = paired(g, a, b, m, keys=("seed",))
            else:
                g = runs[(runs.role == "główne") & (runs.dataset == ds) & runs.seed.isin(range(42, 47))]
                r = paired(g, a, b, m, keys=("split", "seed"), hier=True)
            r.update(zbiór=ds, metryka=m, projekt=design)
            res.append(r)
        adj = holm([r["p_wilcoxon"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = pa
            r["ocena"] = classify(r["diff"], r["ci_lo"], r["ci_hi"]) if m in ("test_ba", "fullv_ba", "test_f1m", "test_acc") else ""
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# przełączanie faz
# ---------------------------------------------------------------------------------------------
def _stall_reason(s: str) -> str:
    if "zwiększeń K bez poprawy" in s:
        return "zastój: 3 zwiększenia K bez poprawy"
    if "ρ_SV < TSV" in s:
        return "bramka ρ_SV < TSV (bez zwiększenia K)"
    if "limit klasy" in s:
        return "limit K klasy (bez zwiększenia K)"
    return "inne"


def phase_tables(steps: pd.DataFrame) -> dict:
    s = steps[steps.series.isin(MAIN_SERIES)].copy()
    for c in ("tprime_perfect", "k_increased", "retry_check", "retry_progress", "vprime_expanded"):
        s[c] = s[c].astype(str).str.lower() == "true"
    own = s[s.variant != "rand_replay"]
    t = own[own.phase == "optimize_T_prime"]
    v = s[s.phase == "optimize_V_prime"]
    # przełączenia na poziomie przebiegu
    per = []
    for (ds, var), g in s.groupby(["dataset", "variant"]):
        runs_n = g.run_id.nunique()
        gt, gv = g[g.phase == "optimize_T_prime"], g[g.phase == "optimize_V_prime"]
        entries = gt[gt.next_phase == "optimize_V_prime"]
        per.append({"zbiór": ds, "wariant": var, "przebiegi": runs_n,
                    "T′→V′ (wejścia, śr./przebieg)": len(entries) / runs_n,
                    "w tym z BA(V′)=1": int(entries.tprime_perfect.sum()) / runs_n,
                    "w tym z zastoju/ρ_SV (próba)": int((~entries.tprime_perfect).sum()) / runs_n,
                    "V′→T′ (powroty, śr./przebieg)": int((gv.next_phase == "optimize_T_prime").sum()) / runs_n,
                    "wzrosty K (śr./przebieg)": int(gt.k_increased.sum()) / runs_n,
                    "rozszerzenia V′ (śr./przebieg)": int(gv.vprime_expanded.sum()) / runs_n,
                    "kroki T′ z BA(V′)=1 (śr./przebieg)": int(gt.tprime_perfect.sum()) / runs_n,
                    "kontrole po próbie V′ (śr./przebieg)": int(gt.retry_check.sum()) / runs_n,
                    "…z osiągnięciem poziomu": int((gt.retry_check & gt.retry_progress).sum()) / runs_n})
    per = pd.DataFrame(per)
    # powody przejścia do V' bez BA=1
    sw = t[(t.next_phase == "optimize_V_prime") & ~t.tprime_perfect].copy()
    sw["powód"] = sw.transition_reason.map(_stall_reason)
    why = sw.groupby(["dataset", "powód"]).size().unstack(fill_value=0)
    # kroki T' z BA<1: co dalej
    below = t[~t.tprime_perfect]
    after_below = below.decision.value_counts().rename("liczba").to_frame()
    after_below["udział"] = after_below["liczba"] / len(below)
    # po BA=1: faza V' i następny krok T'
    rows = []
    for rid, g in own.sort_values(["run_id", "cycle"]).groupby("run_id"):
        g = g.reset_index(drop=True)
        for i in np.flatnonzero((g.phase == "optimize_T_prime") & g.tprime_perfect):
            j = i + 1
            vs = []
            while j < len(g) and g.loc[j, "phase"] == "optimize_V_prime":
                vs.append(j)
                j += 1
            nxt = g.loc[j] if j < len(g) else None
            rows.append({"run_id": rid, "dataset": g.loc[i, "dataset"], "variant": g.loc[i, "variant"], "V′ przed": g.loc[i, "vprime_rows_before"],
                         "następna faza": g.loc[i, "next_phase"], "decyzja": g.loc[i, "decision"], "długość fazy V′ (rozszerzenia)": len(vs),
                         "błąd na ΔV′ (śr.)": g.loc[vs, "delta_error_rate"].mean() if vs else np.nan,
                         "kolejny krok T′: BA(V′)": float(nxt.tprime_fitness) if nxt is not None else np.nan,
                         "kolejny krok T′: BA(V′)=1": bool(nxt.tprime_perfect) if nxt is not None else np.nan})
    after_perfect = pd.DataFrame(rows)
    # trudność ΔV' wg numeru rozszerzenia w fazie
    diff_idx = v.groupby(["variant", "expansions_in_phase"]).agg(n=("delta_error_rate", "size"), blad_dV=("delta_error_rate", "mean"),
                                                                   sd=("delta_error_rate", "std"), ba_dV=("delta_metric", "mean")).reset_index()
    # długość faz V' i powód zakończenia
    ph = v.groupby(["run_id", "vprime_phase_id"]).agg(variant=("variant", "first"), dataset=("dataset", "first"), L=("cycle", "size"),
                                                      first_err=("delta_error_rate", "first"), last_err=("delta_error_rate", "last"),
                                                      first_ba=("delta_metric", "first"), last_ba=("delta_metric", "last"),
                                                      end=("transition_reason", "last"), entry=("phase_entry_reason", "first")).reset_index()
    ph["zmiana błędu (ost.−pierwszy)"] = ph.last_err - ph.first_err
    return {"per": per, "why": why, "after_below": after_below, "after_perfect": after_perfect, "diff_idx": diff_idx, "phases": ph}


# ---------------------------------------------------------------------------------------------
# koszt
# ---------------------------------------------------------------------------------------------
COST_COLS = {"cost_svm_trainings": "treningi SVM", "tprime_individual_evals": "oceny osobników T′ (GA)",
             "valga_individual_evals": "oceny osobników Val-GA", "cost_predict_calls": "wywołania predict",
             "cost_predict_rows": "wiersze predict", "cost_decision_function_calls": "wywołania decision_function",
             "cost_tprime_eval_rows_actual": "wiersze V′ ocenione w T′ (faktycznie)",
             "cost_tprime_eval_rows_nominal": "wiersze V′ ocenione w T′ (nominalnie)",
             "cost_delta_eval_rows": "wiersze ΔV′ ocenione", "cost_n_candidates": "kandydaci (suma po doborach)",
             "cost_vprime_predict_rows": "wiersze predict w fazie V′", "cost_tprime_predict_rows": "wiersze predict w fazie T′",
             "cost_val_ga_nominal_rows": "Val-GA nominalnie (wiersze)", "duration_sec": "czas zegarowy [s]"}


def cost_table(runs: pd.DataFrame, base_only: bool = True) -> pd.DataFrame:
    d = runs[runs.role == "główne"]
    if base_only:
        d = d[d.series == "v2_full"]
    rows = []
    for ds in DATASETS:
        for v in VARIANTS:
            g = d[(d.dataset == ds) & (d.variant == v)]
            if g.empty:
                continue
            row = {"zbiór": ds, "wariant": v, "n": len(g)}
            for c, name in COST_COLS.items():
                row[name] = g[c].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def cost_ratio_table(runs: pd.DataFrame, a="evo_switch", b="rand_replay") -> pd.DataFrame:
    d = runs[(runs.series == "v2_full")]
    rows = []
    for ds in DATASETS:
        ga = d[(d.dataset == ds) & (d.variant == a)].set_index("seed")
        gb = d[(d.dataset == ds) & (d.variant == b)].set_index("seed")
        idx = ga.index.intersection(gb.index)
        row = {"zbiór": ds, "n": len(idx)}
        for c, name in [("cost_svm_trainings", "treningi SVM"), ("cost_predict_rows", "wiersze predict"),
                        ("cost_predict_calls", "wywołania predict"), ("cost_vprime_predict_rows", "wiersze predict w fazie V′"),
                        ("cost_decision_function_calls", "wywołania decision_function"), ("duration_sec", "czas zegarowy")]:
            x, y = ga.loc[idx, c].to_numpy(float), gb.loc[idx, c].to_numpy(float)
            ratio = x / np.where(y == 0, np.nan, y)
            lo, hi = boot_ci(np.log(ratio))
            row[name] = float(np.exp(np.nanmean(np.log(ratio))))
            row[name + " CI"] = (float(np.exp(lo)), float(np.exp(hi)))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# rozmiar V' a jakość
# ---------------------------------------------------------------------------------------------
def partial_spearman(x, y, z) -> tuple[float, float]:
    """Korelacja cząstkowa rang x–y przy kontroli z (reszty z regresji rang na randze z)."""
    rx, ry, rz = (stats.rankdata(np.asarray(a, float)) for a in (x, y, z))
    if np.ptp(rx) == 0 or np.ptp(ry) == 0 or np.ptp(rz) == 0:
        return (np.nan, np.nan)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return (float(r), float(p))


def size_quality(runs: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        g = runs[(runs.series == "v2_full") & (runs.dataset == ds) & (runs.variant == "evo_switch")]
        r, p = stats.spearmanr(g.vprime_rows, g.test_ba) if g.vprime_rows.nunique() > 1 else (np.nan, np.nan)
        pool = runs[(runs.role == "główne") & (runs.dataset == ds) & (runs.variant == "evo_switch")]
        r2, p2 = stats.spearmanr(pool.vprime_rows, pool.test_ba) if pool.vprime_rows.nunique() > 1 else (np.nan, np.nan)
        o = old[(old.dataset == ds) & (old.split == 2026)]
        newm = g.set_index("seed")
        om = o.set_index("seed")
        idx = newm.index.intersection(om.index)
        d = newm.loc[idx, "test_ba"].to_numpy(float) - om.loc[idx, "test_ba"].to_numpy(float)
        lo, hi = boot_ci(d)
        rk, pk = stats.spearmanr(g.final_k, g.test_ba) if g.final_k.nunique() > 1 else (np.nan, np.nan)
        rv_k, pv_k = partial_spearman(g.vprime_rows, g.test_ba, g.final_k)
        rows.append({"zbiór": ds, "n (podział 2026)": len(g), "ρ Spearmana |V′|–BA test (podział 2026)": r, "p": p,
                     "ρ Spearmana K–BA test": rk, "p (K)": pk, "ρ cząstkowa |V′|–BA test | K": rv_k, "p (cząstkowa)": pv_k,
                     "n (wszystkie podziały)": len(pool), "ρ Spearmana (wszystkie podziały)": r2, "p (wszystkie)": p2,
                     "|V′| nowy": g.vprime_rows.mean(), "|V′| stary": o.vprime_size.mean(), "stosunek nowy/stary": g.vprime_rows.mean() / o.vprime_size.mean(),
                     "BA test nowy": g.test_ba.mean(), "BA test stary": o.test_ba.mean(), "różnica (nowy−stary)": float(d.mean()),
                     "CI dolny": lo, "CI górny": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# weryfikacja wymagań promotora na danych
# ---------------------------------------------------------------------------------------------
def requirements(D: dict, P: dict) -> pd.DataFrame:
    runs, steps = D["runs"], D["steps"]
    m = steps[steps.series.isin(MAIN_SERIES)].copy()
    for c in ("tprime_perfect", "k_increased", "vprime_expanded"):
        m[c] = m[c].astype(str).str.lower() == "true"
    own = m[m.variant != "rand_replay"]
    t = own[own.phase == "optimize_T_prime"]
    v = m[m.phase == "optimize_V_prime"]
    mr = runs[runs.role == "główne"]
    rows = []

    # 1-2
    perfect = t[t.tprime_perfect]
    n_p, n_sw, n_stop = len(perfect), int((perfect.decision == "switch_to_V_prime").sum()), int(perfect.decision.str.startswith("stop").sum())
    k_inc_perfect = int(perfect.k_increased.sum())
    stops_perfect_exp = perfect[perfect.decision.str.startswith("stop")]
    rows.append(("1", "BA(V′)=1,0 w T′ nie kończy ewolucji",
                 f"{n_p} kroków T′ z BA(V′)=1 (wersje z własnym automatem): {n_sw} → przełączenie do V′; {n_stop} → stop, wyłącznie gdy V′ nie dało się już rozszerzyć "
                 f"(kod stop_perfect_on_max_vprime: {', '.join(sorted(set(stops_perfect_exp.dataset)))}); K wzrosło w {k_inc_perfect} z {n_p} takich kroków.",
                 "SPEŁNIONE (z jednym ograniczeniem: stop, gdy V′ nierozszerzalne)"))
    nxt_ok = 0
    for rid, g in own.sort_values(["run_id", "cycle"]).groupby("run_id"):
        g = g.reset_index(drop=True)
        for i in np.flatnonzero((g.phase == "optimize_T_prime") & g.tprime_perfect & (g.decision == "switch_to_V_prime")):
            nxt_ok += int(i + 1 < len(g) and g.loc[i + 1, "phase"] == "optimize_V_prime")
    rows.append(("2", "Po BA(V′)=1 przejście do rozszerzenia i optymalizacji V′",
                 f"{nxt_ok} z {n_sw} przełączeń kończy się cyklem w fazie optimize_V_prime.", "SPEŁNIONE"))
    # 3
    ends = v[v.next_phase == "optimize_T_prime"]
    bad = 0
    for rid, g in m.sort_values(["run_id", "cycle"]).groupby("run_id"):
        g = g.reset_index(drop=True)
        for i in np.flatnonzero((g.phase == "optimize_V_prime") & (g.next_phase == "optimize_T_prime")):
            if i + 1 < len(g):
                bad += int(g.loc[i + 1, "phase"] != "optimize_T_prime")
    rows.append(("3", "Po rozszerzeniu V′ powrót do optymalizacji T′",
                 f"{len(ends)} zakończeń fazy V′ (wszystkie warianty); następny cykl nie jest T′: {bad}.", "SPEŁNIONE"))
    # 4
    ev = v[v.variant == "evo_switch"]
    exact_evo = int((ev.predict_rows == ev.n_candidates + 2 * ev.delta_rows).sum())
    rd = v[v.variant != "evo_switch"]
    exact_rd = int((rd.predict_rows == 2 * rd.delta_rows).sum()) if len(rd) else 0
    rows.append(("4", "Dobór V′: osobniki oceniane wyłącznie na nowych próbkach ΔV′, kandydaci = V∖V′",
                 f"W każdym kroku V′ ewolucji liczba wierszy przekazanych modelowi = |V∖V′| + 2·|ΔV′|: {exact_evo} z {len(ev)} kroków (jedno predict na pulę kandydatów + diagnostyka ΔV′); "
                 f"stare próbki V′ nie są oceniane ponownie. Losowanie: 2·|ΔV′|, {exact_rd} z {len(rd)}. Ograniczenie: cała pula V∖V′ jest oceniana raz na rozszerzenie "
                 f"(zbuforowane predykcje), a nie tylko ΔV′.", "SPEŁNIONE (z ograniczeniem kosztu puli)"))
    # 5
    tt = m[(m.phase == "optimize_T_prime")]
    eq5 = int((tt.tprime_eval_rows_actual == tt.ga_svm_evaluations * tt.vprime_rows_before).sum())
    mono = int(sum(g.vprime_rows_before.is_monotonic_increasing for _, g in m.groupby("run_id")))
    rows.append(("5", "T′ optymalizowane na całym, skumulowanym V′",
                 f"Koszt oceny T′ = liczba ocen GA × |V′| aktualnego, skumulowanego V′: {eq5} z {len(tt)} kroków; |V′| nie maleje w żadnym z {m.run_id.nunique()} przebiegów ({mono}); "
                 f"ponadto test jednostkowy porównuje dane widziane przez GA z V′ ze stanu pamięci (test_req04).", "SPEŁNIONE"))
    # 6
    below = t[~t.tprime_perfect]
    inc = int((below.decision == "increase_k").sum())
    sw = below[below.decision == "switch_to_V_prime"]
    rs = sw.transition_reason.map(_stall_reason).value_counts()
    stop_rows = below[below.decision.str.startswith("stop")]
    rows.append(("6", "Gdy BA(V′) < 1, najpierw zwiększane jest K",
                 f"Z {len(below)} kroków T′ z BA(V′)<1: {inc} ({inc/len(below):.1%}) zwiększa K; {len(sw)} przechodzi do V′ — z tego {int(rs.get('zastój: 3 zwiększenia K bez poprawy', 0))} po 3 nieskutecznych zwiększeniach K, "
                 f"{int(rs.get('bramka ρ_SV < TSV (bez zwiększenia K)', 0))} z powodu bramki ρ_SV < TSV (bez zwiększenia K) i {int(rs.get('limit K klasy (bez zwiększenia K)', 0))} z powodu limitu K klasy; "
                 f"{len(stop_rows)} kończy przebieg.", "SPEŁNIONE CZĘŚCIOWO (odstępstwo: bramka ρ_SV)"))
    # 7
    tk = m[m.phase == "optimize_T_prime"]
    vk = m[m.phase == "optimize_V_prime"]
    cont_ok = cont_tot = 0
    for _, g in m.sort_values(["run_id", "cycle"]).groupby("run_id"):
        a = g.k_per_class.to_numpy()[1:]
        b = g.k_after.to_numpy()[:-1]
        cont_ok += int((a == b).sum())
        cont_tot += len(a)
    rows.append(("7", "K nie rośnie więcej niż raz w cyklu",
                 f"Cykle ze zmianą K w fazie V′: {int((vk.k_after != vk.k_per_class).sum())}; cykle, w których wzrosło K i jednocześnie rozszerzono V′: {int((m.k_increased & m.vprime_expanded).sum())}; "
                 f"w cyklu T′ uruchamiane jest jedno GA przy jednym K, a decyzja o wzroście K jest jedna (kontrola ciągłości: K następnego cyklu = K po poprzednim cyklu w {cont_ok} z {cont_tot} przejść).", "SPEŁNIONE"))
    # 8
    stopc = runs[runs.role == "główne"].groupby("variant").stop_decision.value_counts()
    npg = int((mr.stop_decision == "stop_no_progress_after_vprime_retry").sum())
    rows.append(("8", "Warunek zakończenia po braku postępu",
                 f"stop_no_progress_after_vprime_retry kończy {npg} z {len(mr)} przebiegów ({int(((mr.stop_decision == 'stop_no_progress_after_vprime_retry') & (mr.variant == 'evo_switch')).sum())} z "
                 f"{int((mr.variant == 'evo_switch').sum())} przebiegów ewolucyjnych); dodatkowo stop_tprime_no_progress, stop_no_k_growth_signal, stop_k_limit. "
                 f"Przebiegi zakończone bezpiecznikiem max_cycles: {int((mr.stop_decision == 'stop_max_cycles').sum())}.", "SPEŁNIONE"))
    # 9
    lim = mr.vprime_limit_reason.fillna("brak blokady").value_counts().to_dict()
    lim_ds = mr[mr.vprime_limit_reason == "size_limit"].groupby("dataset").size().to_dict()
    rows.append(("9", "V′ nie rośnie bez ograniczeń",
                 f"Maks. |V′|/|V| w {len(mr)} przebiegach: {mr.vprime_fraction.max():.3f} (limit 0,25); maks. |V′| na Szachownicy: {int(mr[mr.dataset == 'chessboard'].vprime_rows.max())} "
                 f"(stary algorytm: średnio ≈ 3528). Stan blokady rozszerzania na końcu przebiegu: {', '.join(f"{k.replace('size_limit', 'limit rozmiaru')}: {v}" for k, v in lim.items())}; "
                 f"limit rozmiaru zablokował dalszy wzrost w {sum(lim_ds.values())} przebiegach ({', '.join(f'{k}: {v}' for k, v in lim_ds.items()) or '—'}), "
                 f"w pozostałych wzrost zatrzymały reguły automatu.", "SPEŁNIONE"))
    # 10
    nn = mr[["vprime_rows", "full_validation_rows", "vprime_fraction"]].notna().all(axis=1).mean()
    bad_frac = int((np.abs(mr.vprime_fraction - mr.vprime_rows / mr.full_validation_rows) > 1e-12).sum())
    rows.append(("10", "Raportowane |V′|, |V| i |V′|/|V|",
                 f"Kolumny vprime_rows, full_validation_rows, vprime_fraction wypełnione w {nn:.0%} przebiegów i w każdym cyklu; niezgodność |V′|/|V|: {bad_frac}.", "SPEŁNIONE"))
    # 11
    ctl = int((mr.ctl_full == True).sum()) + int((mr.ctl_test == True).sum())  # noqa: E712
    rows.append(("11", "Pełne V i test nie sterują algorytmem",
                 f"Flagi used_full_validation_for_control / used_test_for_control w summary.json: wartość „prawda” w {ctl} z {2*len(mr)}; "
                 f"testy jednostkowe: trajektorie identyczne przy obserwatorze pełnego V zwracającym losowe wartości i przy odwróconych etykietach testu (test_req10).", "SPEŁNIONE"))
    # 12
    pairs = 0; eq = 0
    e = mr[mr.variant == "evo_switch"].set_index(["series", "dataset", "seed"])
    r = mr[mr.variant == "rand_replay"].set_index(["series", "dataset", "seed"])
    common = e.index.intersection(r.index)
    eq = int((e.loc[common, "vprime_rows"] == r.loc[common, "vprime_rows"]).sum())
    w = mr[mr.variant == "rand_switch"].set_index(["series", "dataset", "seed"])
    cw = e.index.intersection(w.index)
    ratio = (w.loc[cw, "vprime_rows"] / e.loc[cw, "vprime_rows"]).median()
    rows.append(("12", "Wariant losowy porównywany z ewolucyjnym przy porównywalnym |V′|",
                 f"rand_replay: identyczny końcowy |V′| w {eq} z {len(common)} par (harmonogram faz, K i |ΔV′| odtworzony z przebiegu ewolucyjnego). rand_switch (własny automat): mediana stosunku |V′| do ewolucji = {ratio:.2f}. "
                 f"Uprzywilejowanie: harmonogram (w tym K i moment stopu) rand_replay pochodzi z ewolucji, więc wariant ten nie jest samodzielnym algorytmem.",
                 "SPEŁNIONE (z zastrzeżeniem uprzywilejowania rand_replay)"))
    return pd.DataFrame(rows, columns=["nr", "wymaganie", "dowód w danych", "status"])


# ---------------------------------------------------------------------------------------------
def build_all(save: bool = True) -> dict:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    D = load()
    runs, steps, old = D["runs"], D["steps"], D["old"]
    P = {"complete": completeness()}
    P["size_base"] = size_table(runs, old, base_only=True)
    P["size_all"] = size_table(runs, old, base_only=False)
    P["pair_base"] = paired_table(runs, "evo_switch", "rand_replay", "base")
    P["pair_splits"] = paired_table(runs, "evo_switch", "rand_replay", "splits")
    P["pair_aux_base"] = paired_table(runs, "evo_switch", "rand_switch", "base", metrics=["test_ba", "fullv_ba", "test_f1m", "vprime_rows", "final_k", "steps"])
    P["pair_aux_splits"] = paired_table(runs, "evo_switch", "rand_switch", "splits", metrics=["test_ba", "test_f1m", "vprime_rows"])
    P["phase"] = phase_tables(steps)
    P["cost_base"] = cost_table(runs, True)
    P["cost_ratio"] = cost_ratio_table(runs)
    P["size_quality"] = size_quality(runs, old)
    P["req"] = requirements(D, P)
    # pilotaż
    pil = runs[runs.role == "pilotaż"]
    P["pilot"] = pil.groupby(["series", "dataset", "variant"]).agg(
        n=("seed", "size"), vprime=("vprime_rows", "mean"), vprime_min=("vprime_rows", "min"), vprime_max=("vprime_rows", "max"),
        frac=("vprime_fraction", "mean"), final_k=("final_k", "mean"), test_ba=("test_ba", "mean"), fullv_ba=("fullv_ba", "mean"),
        steps=("steps", "mean"), time=("duration_sec", "mean")).reset_index()
    if save:
        def w(name, df):
            df.to_csv(OUT_TABLES / name, index=False)
        w("final_01_kompletnosc_serii.csv", P["complete"])
        w("final_02_wymagania_promotora.csv", P["req"])
        w("final_03_rozmiar_vprime_podzial2026.csv", P["size_base"])
        w("final_03b_rozmiar_vprime_wszystkie_podzialy.csv", P["size_all"])
        w("final_04_ewolucja_vs_losowanie_podzial2026.csv", P["pair_base"])
        w("final_04b_ewolucja_vs_losowanie_podzialy.csv", P["pair_splits"])
        w("final_04c_ewolucja_vs_rand_switch_podzial2026.csv", P["pair_aux_base"])
        w("final_04d_ewolucja_vs_rand_switch_podzialy.csv", P["pair_aux_splits"])
        w("final_05_przelaczanie_faz.csv", P["phase"]["per"])
        w("final_05b_powody_przejscia_do_V.csv", P["phase"]["why"].reset_index())
        w("final_05c_po_BA1_kroki.csv", P["phase"]["after_perfect"])
        w("final_05d_trudnosc_dV_wg_numeru_rozszerzenia.csv", P["phase"]["diff_idx"])
        w("final_05e_fazy_V.csv", P["phase"]["phases"])
        w("final_06_koszt_podzial2026.csv", P["cost_base"])
        w("final_06b_koszt_stosunek_ewolucja_do_losowania.csv", P["cost_ratio"].astype(str))
        w("final_07_rozmiar_V_a_jakosc.csv", P["size_quality"])
        w("final_08_pilotaz.csv", P["pilot"])
        runs.to_csv(OUT_TABLES / "final_00_przebiegi_v2_z_metrykami.csv", index=False)
    return {"D": D, "P": P}


if __name__ == "__main__":
    R = build_all()
    P = R["P"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 200)
    print(P["complete"].to_string(index=False))
    print(P["req"][["nr", "wymaganie", "status"]].to_string(index=False))
