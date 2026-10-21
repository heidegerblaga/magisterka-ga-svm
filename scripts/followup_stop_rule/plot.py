"""Wykresy badania `followup_stop_rule` (opisy po polsku). Dane: reports/followup_stop_rule/tables/*.csv.

    python scripts/followup_stop_rule/plot.py        # -> reports/followup_stop_rule/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import bootstrap_ci  # noqa: E402
from common import (  # noqa: E402
    ALGO_SEEDS_EXTRA, BASE_SPLIT, BUDGET_TWIN, DATASETS, FIGS, MAIN_DATASETS, MAIN_VARIANTS, SPLIT_SEEDS, TABLES,
    VARIANT_LABELS,
)

ORDER = ["evo_mem", "evo_nomem", "heur_mem", "rand_growth", "rand_fixed"]
COLORS = {"evo_mem": "#2a78d6", "evo_nomem": "#4a3aa7", "heur_mem": "#1baf7a", "rand_growth": "#eda100", "rand_fixed": "#d9542a",
          "evo_mem_bud": "#7fb2ea", "heur_mem_bud": "#7fd8b5", "rand_growth_bud": "#f6cf6a"}
MARKERS = {"evo_mem": "o", "evo_nomem": "D", "heur_mem": "v", "rand_growth": "^", "rand_fixed": "s",
           "evo_mem_bud": "o", "heur_mem_bud": "v", "rand_growth_bud": "^"}
SHORT = {"evo_mem": "evo_mem", "evo_nomem": "evo_nomem", "heur_mem": "heur_mem", "rand_growth": "rand_growth",
         "rand_fixed": "rand_fixed*", "evo_mem_bud": "evo_mem\nbud ×10", "heur_mem_bud": "heur_mem\nbud ×10",
         "rand_growth_bud": "rand_growth\nbud ×10"}
LABEL = {d: DATASETS[d]["label"] for d in DATASETS}
LABEL["magic"] = "MAGIC (8000)"
LABEL["digits"] = "Digits"
CAT_COLORS = {"perfect_ba": "#d9542a", "no_k_growth_signal": "#2a78d6", "k_limit": "#6b7280", "no_candidates": "#eda100",
              "max_cycles": "#0b0b0b"}
CAT_LABELS = {"perfect_ba": "BA(V′) = 1,0", "no_k_growth_signal": "brak sygnału wzrostu K", "k_limit": "limit K",
              "no_candidates": "brak kandydatów", "max_cycles": "limit cykli"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.4, "xtick.labelsize": 7, "ytick.labelsize": 7,
    })


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("zapisano", name, flush=True)


def _t(name):
    try:
        return pd.read_csv(TABLES / name)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _load():
    t = _t
    return {
        "new": t("per_run.csv"), "old": t("per_run_legacy.csv"), "cyc_new": t("per_cycle.csv"), "cyc_old": t("per_cycle_legacy.csv"),
        "status": t("series_status.csv"),
    }


def _datasets_present(frame):
    return [d for d in MAIN_DATASETS if d in set(frame.dataset)]


def _grid(n_panels, ncols=4, panel_w=4.1, panel_h=2.9):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows))
    return fig, np.atleast_1d(axes).ravel()


# ------------------------------------------------------------------------------------------
# 1–4. metryki per seed: małe wielokrotności (nowa reguła = punkty; stara reguła = pusty romb)
# ------------------------------------------------------------------------------------------
def fig_metric_panels(d, metric, ylabel, name, title, log=False, base_only=True):
    new = d["new"][d["new"].split == BASE_SPLIT]
    old = d["old"]
    datasets = _datasets_present(new)
    fig, axes = _grid(len(datasets) + 1)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, datasets):
        for i, v in enumerate(ORDER):
            vals = new[(new.dataset == ds) & (new.variant == v)][metric].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=7, color=COLORS[v], alpha=0.55, zorder=2, linewidths=0)
            lo, hi = bootstrap_ci(vals)
            ax.vlines(i, lo, hi, color=INK, lw=1.4, zorder=4)
            ax.hlines(vals.mean(), i - 0.27, i + 0.27, color=INK, lw=2.0, zorder=5)
            ov = old[(old.dataset == ds) & (old.variant == v)][metric].dropna().to_numpy(float)
            if len(ov):
                ax.scatter([i + 0.36], [ov.mean()], marker="D", s=20, facecolors="white", edgecolors="#7a7a74", linewidths=1.1, zorder=6)
        ax.set_xticks(range(len(ORDER)), [SHORT[v].replace("_", "\n", 1) for v in ORDER], fontsize=6.3)
        ax.set_xlim(-0.6, len(ORDER) - 0.3)
        ax.set_title(f"{LABEL[ds]}", fontsize=8.5)
        if log:
            ax.set_yscale("log")
    for ax in axes[:1]:
        ax.set_ylabel(ylabel)
    leg = axes[len(datasets)]
    leg.axis("off")
    leg.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#888", markersize=4.5, label="pojedynczy seed (nowa reguła)"),
        Line2D([], [], color=INK, lw=2, label="średnia (nowa reguła)"),
        Line2D([], [], color=INK, lw=1.4, label="95% CI średniej (bootstrap)"),
        Line2D([], [], marker="D", color="none", markerfacecolor="white", markeredgecolor="#7a7a74", markersize=5,
               label="średnia — stara reguła (to samo seedy)"),
    ], loc="center left", fontsize=7.5)
    for ax in axes[len(datasets) + 1:]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=10)
    fig.tight_layout()
    _save(fig, name)


# ------------------------------------------------------------------------------------------
# 5. przyczyny zakończenia przed/po + rozmiar V' przy stopie BA=1,0
# ------------------------------------------------------------------------------------------
def fig_stops(d):
    new = d["new"][(d["new"].split == BASE_SPLIT) & d["new"].variant.isin(ORDER)]
    old = d["old"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    cats = list(CAT_COLORS)
    width = 0.36
    for i, v in enumerate(ORDER):
        for j, (rule, frame) in enumerate((("stara", old), ("nowa", new))):
            sub = frame[frame.variant == v]
            if sub.empty:
                continue
            bottom = 0.0
            for c in cats:
                share = float((sub.stop_category == c).mean()) * 100
                ax.bar(i + (j - 0.5) * (width + 0.02), share, width, bottom=bottom, color=CAT_COLORS[c],
                       alpha=0.55 if rule == "stara" else 1.0, edgecolor="white", linewidth=0.4)
                bottom += share
            ax.text(i + (j - 0.5) * (width + 0.02), 101, "stara" if rule == "stara" else "nowa", ha="center", va="bottom", fontsize=5.8,
                    color=INK2)
    ax.set_xticks(range(len(ORDER)), [SHORT[v] for v in ORDER], fontsize=7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% przebiegów (7 zbiorów × 20 seedów = 140)")
    ax.set_title("Przyczyna zakończenia przebiegu: stara (blada) vs nowa (pełna) reguła stopu")
    ax.legend(handles=[Patch(color=CAT_COLORS[c], label=CAT_LABELS[c]) for c in cats], loc="lower center", ncol=3, fontsize=6.8,
              bbox_to_anchor=(0.5, -0.33))
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    rng = np.random.default_rng(1)
    pos = 0
    ticks = []
    for v in ["rand_growth", "evo_nomem", "heur_mem", "evo_mem"]:
        for rule, frame, color in (("stara", old, "#9a9a94"), ("nowa", new, COLORS[v])):
            sub = frame[(frame.variant == v) & frame.perfect_ba_stop]
            ax.scatter(pos + rng.uniform(-0.16, 0.16, len(sub)), sub.vprime_size, s=11, color=color, alpha=0.8, linewidths=0)
            ax.text(pos, 1.5, f"n={len(sub)}", ha="center", fontsize=6.3, color=INK2)
            ticks.append((pos, f"{SHORT[v].split(chr(10))[0]}\n{rule}"))
            pos += 1
        pos += 0.5
    ax.axhline(100, color="#d9542a", ls="--", lw=0.9)
    ax.text(pos - 0.6, 108, "minimum dla problemu binarnego (2×50)", fontsize=6.3, color="#d9542a", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1.2, 6000)
    ax.set_xticks([p for p, _ in ticks], [t.replace("\n", " — ") for _, t in ticks], fontsize=6.3, rotation=40, ha="right")
    ax.set_ylabel("|V′| w chwili stopu BA(V′) = 1,0")
    ax.set_title("Rozmiar V′ przy zatrzymaniu przez BA(V′) = 1,0")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_stops_before_after.png")


# ------------------------------------------------------------------------------------------
# 6. przykładowe trajektorie: stara vs nowa reguła
# ------------------------------------------------------------------------------------------
def fig_trajectory(d, dataset="moons", variant="rand_growth", seed=42):
    co, cn = d["cyc_old"], d["cyc_new"]
    o = co[(co.dataset == dataset) & (co.variant == variant) & (co.seed == seed)].sort_values("cycle")
    n = cn[(cn.split == BASE_SPLIT) & (cn.dataset == dataset) & (cn.variant == variant) & (cn.seed == seed)].sort_values("cycle")
    if o.empty or n.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.4), sharex="col")
    for col, (title, c) in enumerate((("stara reguła stopu", o), ("nowa reguła stopu (min 50/klasę + 1 ponowne sprawdzenie)", n))):
        x = c["cycle"].to_numpy()
        ax = axes[0, col]
        ax.step(x, c["val_rows_before"], where="mid", color=COLORS[variant], label="|V′| na początku cyklu")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.step(x, c["k_per_class"], where="mid", color="#52514e", ls="--", lw=1.1, label="K (próbek na klasę w T′)")
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.set_title(f"{LABEL[dataset]}, {VARIANT_LABELS[variant]}, seed {seed}\n{title}", fontsize=8.3)
        if col == 0:
            ax.set_ylabel("|V′| (skala log)")
        if col == 1:
            ax2.set_ylabel("K")
        ax = axes[1, col]
        ax.plot(x, c["train_ga_best_val_metric"], color="#d9542a", marker="o", ms=3, label="BA na V′ (steruje)")
        ax.plot(x, c["train_ga_test_metric"], color="#1baf7a", marker="s", ms=2.5, lw=1.1, label="BA test (obserwacja)")
        last = c.iloc[-1]
        ax.axvline(last["cycle"], color=INK, lw=0.9)
        ax.annotate("stop", (last["cycle"], ax.get_ylim()[0]), xytext=(-3, 4), textcoords="offset points", ha="right", fontsize=7)
        if "perfect_gate_pending" in c.columns:
            pend = c[c.perfect_gate_pending.fillna("").astype(str).str.len() > 0]
            ax.scatter(pend["cycle"], pend["train_ga_best_val_metric"], s=70, facecolors="none", edgecolors="#2a78d6", linewidths=1.3,
                       label="BA = 1,0, stop zablokowany", zorder=5)
        ax.set_xlabel("numer cyklu T′")
        if col == 0:
            ax.set_ylabel("balanced accuracy")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(*[sum(z, []) for z in zip(axes[0, 1].get_legend_handles_labels(), axes[0, 1].figure.axes[3].get_legend_handles_labels())],
                      loc="center right", fontsize=7)
    axes[1, 1].legend(loc="lower right", fontsize=7)
    axes[1, 0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    _save(fig, f"fig_trajectory_{dataset}_{variant}.png")


# ------------------------------------------------------------------------------------------
# 7. koszt
# ------------------------------------------------------------------------------------------
def fig_cost(d):
    new = d["new"][d["new"].split == BASE_SPLIT]
    datasets = _datasets_present(new)
    panels = [
        ("svm_fits_total", "treningi SVM w GA dla T′ (na przebieg)", None),
        ("tga_vprime_rows_scored", "wiersze V′ klasyfikowane w GA dla T′ (Σ ocen × |V′|)", None),
        ("selection_predict_rows", "wiersze przekazane modelowi przy wyborze przyrostu V′ (predict + decision_function)", "sel"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.2), sharex=True)
    x = np.arange(len(datasets))
    w = 0.16
    for ax, (metric, ylabel, kind) in zip(axes, panels):
        for i, v in enumerate(ORDER):
            vals, bottoms = [], []
            for ds in datasets:
                g = new[(new.dataset == ds) & (new.variant == v)]
                vals.append(float(g[metric].mean()) if len(g) else np.nan)
                bottoms.append(float(g["selection_decision_rows"].mean()) if (len(g) and kind) else 0.0)
            xs = x + (i - 2) * w
            if kind == "sel":
                pred = np.array(vals) - np.array(bottoms) * 0  # predict
                dec = np.array(bottoms)
                ax.bar(xs, np.where(np.isfinite(pred), pred, 0) + dec, w, color=COLORS[v], alpha=0.35, label=None)
                ax.bar(xs, np.where(np.isfinite(pred), pred, 0), w, color=COLORS[v], label=SHORT[v] if metric == panels[0][0] else None)
                zero = [j for j, p in enumerate(pred) if np.isfinite(p) and p == 0 and dec[j] == 0]
                ax.scatter(xs[zero], [1.4] * len(zero), marker="_", s=30, color=COLORS[v])
                if v.startswith("evo"):
                    nominal = [float(new[(new.dataset == ds) & (new.variant == v)]["val_ga_nominal_subset_rows"].mean()) for ds in datasets]
                    ax.scatter(xs, nominal, marker="x", s=22, color=INK, zorder=5, linewidths=1.0)
            else:
                ax.bar(xs, vals, w, color=COLORS[v], label=SHORT[v] if metric == panels[0][0] else None)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=7.2)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=7.5)
    axes[2].legend(handles=[Patch(facecolor="#888", label="predict (słupek pełny) + decision_function (jasna nakładka; tylko heur_mem)"),
                            Line2D([], [], marker="x", color=INK, ls="none", label="NOMINALNA liczba wierszy ocenianych w Val-GA (bez cache; NIE wykonana)"),
                            Line2D([], [], marker="_", color="#888", ls="none", markersize=7, label="zero wywołań modelu (dobór losowy)")],
                   loc="upper right", fontsize=6.6, bbox_to_anchor=(1.0, 1.0))
    axes[2].set_xticks(x, [LABEL[ds] for ds in datasets], fontsize=7.3)
    fig.tight_layout()
    _save(fig, "fig_cost.png")


# ------------------------------------------------------------------------------------------
# 8. porównania sparowane po seedach: różnice (forest) i wykresy par
# ------------------------------------------------------------------------------------------
PAIRS_FOREST = [("evo_mem", "rand_growth"), ("evo_mem", "heur_mem"), ("evo_mem", "rand_fixed"), ("evo_mem", "evo_nomem")]


def fig_forest(d):
    pv = _t("paired_variants.csv")
    if pv.empty:
        return
    pv = pv[pv.split == BASE_SPLIT]
    datasets = _datasets_present(d["new"])
    panels = [("test_ba", "Δ BA na teście (a − b)", 1.0), ("cycles", "Δ liczby cykli T′ (a − b)", 1.0),
              ("svm_fits_total", "Δ treningów SVM w GA dla T′ (a − b)", 1.0), ("mean_increment_error_rate", "Δ trudności przyrostu V′ (odsetek błędów, a − b)", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.6), sharey=True)
    pair_colors = {"rand_growth": COLORS["rand_growth"], "heur_mem": COLORS["heur_mem"], "rand_fixed": COLORS["rand_fixed"], "evo_nomem": COLORS["evo_nomem"]}
    for ax, (metric, title, _) in zip(axes, panels):
        for i, ds in enumerate(datasets):
            for j, (a, b) in enumerate(PAIRS_FOREST):
                row = pv[(pv.dataset == ds) & (pv.a == a) & (pv.b == b) & (pv.metric == metric)]
                if row.empty or pd.isna(row.iloc[0].get("mean_diff")):
                    continue
                r = row.iloc[0]
                y = i + (j - 1.5) * 0.17
                ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci95_low], [r.ci95_high - r.mean_diff]], fmt=MARKERS[b] if b in MARKERS else "o",
                            ms=3.8, color=pair_colors[b], lw=1.1, capsize=0)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(datasets)), [LABEL[ds] for ds in datasets])
        ax.invert_yaxis()
        ax.set_xlabel("średnia różnica i 95% CI (bootstrap, 20 par)", fontsize=7)
        if metric in ("cycles", "svm_fits_total"):
            ax.set_xscale("symlog", linthresh=10 if metric == "cycles" else 1000)
    fig.legend(handles=[Line2D([], [], marker=MARKERS[b], color=pair_colors[b], ls="none", label=f"evo_mem vs {SHORT[b].replace('*', '')}") for _, b in PAIRS_FOREST],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_paired_forest.png")


def fig_pairs_scatter(d, a="evo_mem", b="rand_growth", metric="test_ba", name="fig_pairs_evo_vs_randgrowth.png"):
    new = d["new"][d["new"].split == BASE_SPLIT]
    datasets = _datasets_present(new)
    fig, axes = _grid(len(datasets) + 1, panel_w=3.7, panel_h=3.2)
    for ax, ds in zip(axes, datasets):
        A = new[(new.dataset == ds) & (new.variant == a)].set_index("seed")[metric]
        B = new[(new.dataset == ds) & (new.variant == b)].set_index("seed")[metric]
        seeds = sorted(set(A.index) & set(B.index))
        if not seeds:
            continue
        ax.scatter(A.loc[seeds], B.loc[seeds], s=14, color=COLORS[b], alpha=0.85, edgecolors=INK2, linewidths=0.3)
        lo, hi = min(A.loc[seeds].min(), B.loc[seeds].min()), max(A.loc[seeds].max(), B.loc[seeds].max())
        pad = 0.04 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK, lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        diff = (A.loc[seeds] - B.loc[seeds])
        ax.set_title(f"{LABEL[ds]}\nśr. Δ = {diff.mean():+.4f}; {int((diff > 0).sum())}/{int((diff < 0).sum())} seedów", fontsize=8)
        ax.set_xlabel(f"{SHORT[a]}", fontsize=7)
        ax.set_ylabel(f"{SHORT[b]}", fontsize=7)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    axes[len(datasets)].text(0.0, 0.5, "Każdy punkt = jeden seed (ten sam seed i podział\nw obu wariantach). Powyżej przekątnej:\nlepszy wariant z osi pionowej.",
                             fontsize=8, va="center")
    fig.tight_layout()
    _save(fig, name)


# ------------------------------------------------------------------------------------------
# 9. trudność V' a jakość
# ------------------------------------------------------------------------------------------
def fig_hardness(d):
    new = d["new"][d["new"].split == BASE_SPLIT]
    datasets = _datasets_present(new)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    variants = ["evo_mem", "heur_mem", "rand_growth", "rand_fixed"]
    x = np.arange(len(datasets))
    w = 0.2
    specs = [("mean_pool_error_rate", "błędy modelu w CAŁEJ puli kandydatów V∖V′"), ("mean_increment_error_rate", "błędy modelu w DODANYCH elementach V′"),
             ("vprime_ba", "BA na końcowym V′ (metryka sterująca)")]
    for ax, (metric, title) in zip(axes, specs):
        for i, v in enumerate(variants):
            vals = [float(new[(new.dataset == ds) & (new.variant == v)][metric].mean()) if len(new[(new.dataset == ds) & (new.variant == v)]) else np.nan
                    for ds in datasets]
            ax.bar(x + (i - 1.5) * w, vals, w, color=COLORS[v], label=SHORT[v])
        ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("średnia po seedach")
    axes[2].set_ylim(0.9, 1.005)
    axes[0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_hardness.png")


# ------------------------------------------------------------------------------------------
# 10. pamięć V'
# ------------------------------------------------------------------------------------------
def fig_memory(d):
    ms = _t("memory_spread.csv")
    if ms.empty:
        return
    ms = ms[ms.metric == "test_ba"]
    datasets = [ds for ds in MAIN_DATASETS if ds in set(ms.dataset)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.19
    for k, (rule, label, alpha) in enumerate((("legacy", "stara reguła", 0.4), ("extended", "nowa reguła", 1.0))):
        sub = ms[ms.rule == rule].set_index("dataset")
        for i, (col, v) in enumerate((("sd_mem", "evo_mem"), ("sd_nomem", "evo_nomem"))):
            vals = [float(sub.loc[ds, col]) if ds in sub.index else np.nan for ds in datasets]
            ax.bar(x + (k * 2 + i - 1.5) * w, vals, w, color=COLORS[v], alpha=alpha,
                   label=f"{SHORT[v]} — {label}")
    ax.set_yscale("log")
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=20, ha="right")
    ax.set_ylabel("odch. std. BA na teście (20 seedów, podział bazowy)")
    ax.set_title("Rozrzut jakości testowej: z pamięcią vs bez pamięci V′")
    ax.legend(ncol=2, fontsize=6.8, loc="upper left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    sub = ms[ms.rule == "extended"].set_index("dataset")
    for i, ds in enumerate(datasets):
        if ds not in sub.index:
            continue
        r = sub.loc[ds]
        ax.errorbar(r.sd_ratio_nomem_over_mem, i, xerr=[[r.sd_ratio_nomem_over_mem - r.sd_ratio_ci_low], [r.sd_ratio_ci_high - r.sd_ratio_nomem_over_mem]],
                    fmt="o", color=COLORS["evo_nomem"], ms=4, lw=1.1)
        old = ms[(ms.rule == "legacy") & (ms.dataset == ds)]
        if len(old):
            ax.scatter([float(old.iloc[0].sd_ratio_nomem_over_mem)], [i + 0.18], marker="D", s=16, facecolors="white", edgecolors="#7a7a74", zorder=4)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(datasets)), [LABEL[ds] for ds in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("stosunek odch. std.: bez pamięci / z pamięcią (nowa reguła; 95% CI bootstrap)\n○ nowa, ◇ stara reguła; > 1 = pamięć zmniejsza rozrzut", fontsize=7)
    ax.set_title("Czy pamięć V′ zmniejsza rozrzut?")
    fig.tight_layout()
    _save(fig, "fig_memory.png")


# ------------------------------------------------------------------------------------------
# 11. podziały danych
# ------------------------------------------------------------------------------------------
def fig_splits(d):
    new = d["new"]
    common = new[new.seed.isin(ALGO_SEEDS_EXTRA) & new.variant.isin(ORDER)]
    datasets = [ds for ds in MAIN_DATASETS if DATASETS[ds]["split_dependent"] and common[common.dataset == ds].split.nunique() >= 2]
    if not datasets:
        return
    fig, axes = _grid(len(datasets), ncols=3, panel_w=4.0, panel_h=2.9)
    for ax, ds in zip(axes, datasets):
        for v in ORDER:
            g = common[(common.dataset == ds) & (common.variant == v)]
            if g.empty:
                continue
            st = g.groupby("split").test_ba.agg(["mean", "std"])
            off = (ORDER.index(v) - 2) * 0.05
            xs = np.arange(len(st)) + off
            ax.errorbar(xs, st["mean"], yerr=st["std"], marker=MARKERS[v], ms=4, color=COLORS[v], lw=1.0, capsize=0, label=SHORT[v])
        splits = sorted(common[common.dataset == ds].split.unique())
        ax.set_xticks(range(len(splits)), [str(s) for s in splits])
        ax.set_xlabel("ziarno podziału danych")
        ax.set_title(LABEL[ds], fontsize=8.5)
    axes[0].set_ylabel("BA na teście: średnia ± odch. std. (5 seedów)")
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Wynik testowy na czterech podziałach danych (seedy 42–46; punkty przesunięte poziomo dla czytelności)", y=1.0, fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig_splits_test_ba.png")

    vc = _t("variance_components.csv")
    if vc.empty:
        return
    vc = vc[vc.metric == "test_ba"]
    if vc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6))
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.15
    for i, v in enumerate(ORDER):
        sub = vc[vc.variant == v].set_index("dataset")
        ax.bar(x + (i - 2) * w, [float(sub.loc[ds, "sd_split"]) if ds in sub.index else np.nan for ds in datasets], w, color=COLORS[v], label=SHORT[v])
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=20, ha="right")
    ax.set_ylabel("składnik: odch. std. między podziałami (σ_split)")
    ax.set_title("Zmienność wynikająca z PODZIAŁU danych (test BA)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for i, v in enumerate(ORDER):
        sub = vc[vc.variant == v].set_index("dataset")
        ax.bar(x + (i - 2) * w, [float(sub.loc[ds, "sd_seed"]) if ds in sub.index else np.nan for ds in datasets], w, color=COLORS[v])
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=20, ha="right")
    ax.set_ylabel("składnik: odch. std. między seedami w obrębie podziału (σ_seed)")
    ax.set_title("Zmienność wynikająca z SEEDÓW algorytmu (test BA)")
    ax.grid(axis="x", visible=False)
    ymax = max(ax.get_ylim()[1], axes[0].get_ylim()[1])
    for a in axes:
        a.set_ylim(0, ymax)
    fig.tight_layout()
    _save(fig, "fig_variance_components.png")


def fig_split_diff(d):
    vd = _t("variant_diff_across_splits.csv")
    if vd.empty:
        return
    vd = vd[vd.metric == "test_ba"]
    if vd.empty:
        return
    datasets = [ds for ds in MAIN_DATASETS if ds in set(vd.dataset)]
    pairs = [("evo_mem", "rand_growth"), ("evo_mem", "heur_mem"), ("evo_mem", "rand_fixed"), ("evo_mem", "evo_nomem")]
    pair_color = {"rand_growth": COLORS["rand_growth"], "heur_mem": COLORS["heur_mem"], "rand_fixed": COLORS["rand_fixed"], "evo_nomem": COLORS["evo_nomem"]}
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for i, ds in enumerate(datasets):
        for j, (a, b) in enumerate(pairs):
            r = vd[(vd.dataset == ds) & (vd.a == a) & (vd.b == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (j - 1.5) * 0.18
            per = json.loads(r.per_split_mean_diff)
            ax.scatter(list(per.values()), [y] * len(per), s=12, color=pair_color[b], alpha=0.45, linewidths=0)
            ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci95_hier_low], [r.ci95_hier_high - r.mean_diff]], fmt="o", ms=4.2, color=pair_color[b], lw=1.2, capsize=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(datasets)), [LABEL[ds] for ds in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("Δ BA na teście (evo_mem − wariant): kropka = średnia z podziałów, blady punkt = średnia na jednym podziale,\nwąs = 95% CI z bootstrapu hierarchicznego (podziały, a w nich seedy)", fontsize=7.3)
    ax.legend(handles=[Line2D([], [], marker="o", color=pair_color[b], ls="none", label=f"vs {SHORT[b].replace('*', '')}") for _, b in pairs], loc="lower right", fontsize=7)
    ax.set_title("Różnice między wariantami przy uwzględnieniu zmienności podziałów i seedów")
    fig.tight_layout()
    _save(fig, "fig_variant_diff_splits.png")


# ------------------------------------------------------------------------------------------
# 12. budżet
# ------------------------------------------------------------------------------------------
def fig_budget(d):
    bc = _t("budget_comparison.csv")
    if bc.empty:
        return
    datasets = [ds for ds in MAIN_DATASETS if ds in set(bc.dataset)]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    x = np.arange(len(datasets))
    w = 0.26
    twins = list(BUDGET_TWIN)
    ax = axes[0]
    for i, v in enumerate(twins):
        sub = bc[bc.variant == v].set_index("dataset")
        vals = [float(sub.loc[ds, "test_ba_diff"]) if ds in sub.index else np.nan for ds in datasets]
        lo = [float(sub.loc[ds, "test_ba_ci_low"]) if ds in sub.index else np.nan for ds in datasets]
        hi = [float(sub.loc[ds, "test_ba_ci_high"]) if ds in sub.index else np.nan for ds in datasets]
        ax.errorbar(x + (i - 1) * w * 0.7, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt=MARKERS[v], color=COLORS[v.replace("_bud", "")],
                    ms=4.5, lw=1.1, capsize=0, label=SHORT[v.replace("_bud", "")])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Δ BA na teście: budżet ×10 − pełna pula")
    ax.set_title("Jakość: budżet vs pełna pula (95% CI, pary po seedach)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    for i, v in enumerate(["evo_mem_bud", "heur_mem_bud"]):
        sub = bc[bc.variant == v].set_index("dataset")
        ratio = [float((sub.loc[ds, "selection_predict_rows_bud"] + sub.loc[ds, "selection_decision_rows_bud"]) /
                       max(1.0, sub.loc[ds, "selection_predict_rows_full"] + sub.loc[ds, "selection_decision_rows_full"])) if ds in sub.index else np.nan for ds in datasets]
        ax.bar(x + (i - 0.5) * 0.35, ratio, 0.35, color=COLORS[v.replace("_bud", "")], label=SHORT[v.replace("_bud", "")])
    ax.set_yscale("log")
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("wiersze oceniane modelem: budżet / pełna pula")
    ax.set_title("Koszt doboru przyrostu V′ (faktyczne wywołania modelu)")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    for i, v in enumerate(twins):
        sub = bc[bc.variant == v].set_index("dataset")
        ax.bar(x + (i - 1) * 0.26, [float(sub.loc[ds, "svm_fits_total_bud"] / sub.loc[ds, "svm_fits_total_full"]) if ds in sub.index else np.nan for ds in datasets], 0.26,
               color=COLORS[v.replace("_bud", "")], label=SHORT[v.replace("_bud", "")])
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xticks(x, [LABEL[ds] for ds in datasets], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("treningi SVM w GA dla T′: budżet / pełna pula")
    ax.set_title("Koszt GA dla T′ (zmienia się przez długość przebiegu)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_budget.png")


def main():
    _style()
    d = _load()
    new = d["new"]
    if new.empty:
        print("brak danych")
        return
    fig_metric_panels(d, "test_ba", "balanced accuracy na teście", "fig_quality_test_ba.png", "Jakość testowa (BA) — podział bazowy, 20 seedów, nowa reguła stopu")
    fig_metric_panels(d, "cycles", "liczba cykli T′", "fig_cycles.png", "Liczba cykli T′ — podział bazowy, 20 seedów", log=True)
    fig_metric_panels(d, "final_k", "końcowe K (próbek na klasę)", "fig_final_k.png", "Końcowe K — podział bazowy, 20 seedów")
    fig_metric_panels(d, "vprime_size", "końcowy |V′|", "fig_final_vprime.png", "Końcowy rozmiar V′ — podział bazowy, 20 seedów", log=True)
    fig_stops(d)
    fig_trajectory(d, "moons", "rand_growth", 42)
    fig_cost(d)
    fig_forest(d)
    fig_pairs_scatter(d)
    fig_hardness(d)
    fig_memory(d)
    fig_splits(d)
    fig_split_diff(d)
    fig_budget(d)


if __name__ == "__main__":
    main()
