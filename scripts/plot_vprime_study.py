"""Wizualizacje badania V' (opisy po polsku). Dane: reports/vprime_study/tables/*.csv oraz artefakty przebiegów.

Użycie:  python scripts/plot_vprime_study.py
Wynik:   reports/vprime_study/figures/*.png
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
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
from ga.models.svm import SVMConfig, fit_svm_model  # noqa: E402
from ga.study_analysis import STUDY_ROOT, VARIANT_LABELS  # noqa: E402

TABLES = REPO_ROOT / "reports" / "vprime_study" / "tables"
FIGS = REPO_ROOT / "reports" / "vprime_study" / "figures"

DATASET_LABELS = {
    "chessboard": "Szachownica", "moons": "Moons", "circles": "Circles", "spiral": "Spiral",
    "classification2d": "Classification 2D", "digits": "Digits (64 cechy)", "magic": "MAGIC (UCI)",
}
# kolory zależne od ROLI (wariantu), nie od rangi; wszystkie serie dodatkowo mają markery/etykiety
COLORS = {
    "evo_mem": "#2a78d6", "evo_nomem": "#4a3aa7", "rand_fixed": "#eb6834",
    "rand_growth": "#eda100", "heur_mem": "#1baf7a",
}
MARKERS = {"evo_mem": "o", "evo_nomem": "D", "rand_fixed": "s", "rand_growth": "^", "heur_mem": "v"}
DECISION_COLORS = {
    "expand_validation_only": "#2a78d6", "increase_k": "#eb6834",
    "stop": "#0b0b0b",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
            "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
            "legend.fontsize": 8, "lines.linewidth": 1.6,
        }
    )


def _save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("zapisano", path.name)
    return path


def _load():
    return pd.read_csv(TABLES / "per_run.csv"), pd.read_csv(TABLES / "per_cycle.csv")


def _decision_class(decision: str) -> str:
    return decision if decision in DECISION_COLORS else "stop"


# ------------------------------------------------------------------------------------------
# 1. przebieg algorytmu w czasie: krótki / typowy / najdłuższy seed
# ------------------------------------------------------------------------------------------
def fig_trajectory(per_run, per_cycle, dataset: str, variant: str = "evo_mem"):
    runs = per_run[(per_run.dataset == dataset) & (per_run.variant == variant)].sort_values("cycles")
    if runs.empty:
        return None
    picks = [
        ("najkrótszy przebieg", runs.iloc[0]),
        ("typowy przebieg (mediana cykli)", runs.iloc[(runs["cycles"] - runs["cycles"].median()).abs().argmin()]),
        ("najdłuższy przebieg", runs.iloc[-1]),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 10.5), sharex="col")
    for col, (title, run) in enumerate(picks):
        cyc = per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == variant) & (per_cycle.seed == run.seed)]
        x = cyc["cycle"].to_numpy()
        ax = axes[0, col]
        ax.step(x, cyc["val_rows_before"], where="mid", color=COLORS["evo_mem"], label="|V'| na początku cyklu")
        exp = cyc[cyc.decision == "expand_validation_only"]
        ax.scatter(exp["cycle"], exp["val_rows_after"], s=14, color=COLORS["evo_mem"], zorder=3,
                   label="rozszerzenie V' (rozmiar po)")
        ax.set_title(f"{title}\nseed {int(run.seed)}, {int(run.cycles)} cykli", fontsize=9)
        if col == 0:
            ax.set_ylabel("rozmiar V' (liczba próbek)")
            ax.legend(loc="upper left")
        ax = axes[1, col]
        ax.step(x, cyc["k_per_class"], where="mid", color=COLORS["rand_fixed"], label="K (próbek na klasę w T')")
        inc = cyc[cyc.decision == "increase_k"]
        ax.scatter(inc["cycle"], inc["k_after"], s=16, marker="^", color=COLORS["rand_fixed"], zorder=3,
                   label="wzrost K (maks. raz na cykl)")
        if col == 0:
            ax.set_ylabel("K (próbek na klasę)")
            ax.legend(loc="upper left")
        ax = axes[2, col]
        ax.plot(x, cyc["train_ga_best_val_metric"], color=COLORS["rand_fixed"], lw=1.2, ls="--",
                label="V' (steruje algorytmem)")
        ax.plot(x, cyc["train_ga_best_full_validation_metric"], color=COLORS["evo_mem"], label="pełne V (obserwacja)")
        ax.plot(x, cyc["train_ga_test_metric"], color=COLORS["heur_mem"], label="test (obserwacja)")
        last = cyc.iloc[-1]
        ax.axvline(last["cycle"], color=INK, lw=0.8)
        ax.annotate(f"stop: {last['decision'].replace('stop_', '')}", (last["cycle"], ax.get_ylim()[1]),
                    xytext=(-4, -10), textcoords="offset points", ha="right", fontsize=7, color=INK2)
        ax.set_xlabel("numer cyklu T'")
        if col == 0:
            ax.set_ylabel("balanced accuracy")
            ax.legend(loc="lower right")
        ax = axes[3, col]
        ax.step(x, cyc["num_support_vectors"], where="mid", color=COLORS["heur_mem"],
                label="liczba SV")
        ax.set_xlabel("numer cyklu T'")
        if col == 0:
            ax.set_ylabel("wektory podpierające (SV)")
            ax.legend(loc="upper left")
    fig.suptitle(
        f"Przebieg algorytmu w czasie — {DATASET_LABELS[dataset]}, wariant: {VARIANT_LABELS[variant]}", y=1.0, fontsize=11
    )
    fig.tight_layout()
    return _save(fig, f"fig_trajectory_{dataset}_{variant}.png")


# ------------------------------------------------------------------------------------------
# 2. oś decyzji dla wszystkich seedów
# ------------------------------------------------------------------------------------------
def fig_decisions(per_run, per_cycle, dataset: str, variant: str = "evo_mem"):
    cyc = per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == variant)]
    if cyc.empty:
        return None
    seeds = sorted(cyc.seed.unique())
    max_c = int(cyc.cycle.max())
    codes = {"expand_validation_only": 1, "increase_k": 2, "stop": 3}
    grid = np.zeros((len(seeds), max_c))
    for i, s in enumerate(seeds):
        sub = cyc[cyc.seed == s]
        for _, r in sub.iterrows():
            grid[i, int(r.cycle) - 1] = codes[_decision_class(r.decision)]
    cmap = ListedColormap(["#ffffff", DECISION_COLORS["expand_validation_only"], DECISION_COLORS["increase_k"], DECISION_COLORS["stop"]])
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(seeds) + 1.6))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_yticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_xticks(np.arange(0, max_c, 5), [str(c) for c in range(1, max_c + 1, 5)])
    ax.grid(False)
    ax.set_xlabel("numer cyklu T'")
    ax.set_ylabel("seed")
    ax.set_title(f"Decyzje algorytmu w kolejnych cyklach — {DATASET_LABELS[dataset]}, {VARIANT_LABELS[variant]}")
    ax.legend(
        handles=[Patch(color=DECISION_COLORS["expand_validation_only"], label="rozszerzenie V' (K bez zmian)"),
                 Patch(color=DECISION_COLORS["increase_k"], label="wzrost K"),
                 Patch(color=DECISION_COLORS["stop"], label="zakończenie")],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
    )
    return _save(fig, f"fig_decisions_{dataset}_{variant}.png")


# ------------------------------------------------------------------------------------------
# 3. trudność elementów dodawanych do V'
# ------------------------------------------------------------------------------------------
def fig_increment_difficulty(per_cycle, dataset: str):
    variants = [v for v in ("evo_mem", "heur_mem", "rand_growth") if
                not per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == v)].empty]
    if not variants:
        return None
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, variants):
        cyc = per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == v) & per_cycle.increment_error_rate.notna()]
        g = cyc.groupby("cycle")
        stat = g.agg(inc=("increment_error_rate", "mean"), pool=("candidate_pool_error_rate", "mean"), n=("seed", "count"))
        stat = stat[stat.n >= 5]
        ax.plot(stat.index, stat.inc, color=COLORS[v], marker=MARKERS[v], ms=3, label="błędy w dodanych elementach")
        ax.plot(stat.index, stat.pool, color=INK2, ls="--", label="błędy w całej puli kandydatów V∖V'")
        ax.set_title(VARIANT_LABELS[v])
        ax.set_xlabel("numer cyklu T' (średnia po seedach, n ≥ 5)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="center right")
    axes[0].set_ylabel("odsetek błędnie klasyfikowanych")
    fig.suptitle(f"Trudność elementów wybieranych do V' — {DATASET_LABELS[dataset]}", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, f"fig_increment_difficulty_{dataset}.png")


# ------------------------------------------------------------------------------------------
# 4/5. porównania sparowane po seedach
# ------------------------------------------------------------------------------------------
PAIR_PANELS = [
    ("test_ba", "test: balanced accuracy"),
    ("fullv_ba", "pełne V (obserwacja): balanced accuracy"),
    ("cycles", "liczba cykli T'"),
    ("final_k", "końcowe K"),
    ("vprime_size", "końcowy rozmiar V'"),
    ("duration_sec", "czas wykonania [s]"),
]


def fig_paired(per_run, dataset: str, a: str, b: str, name: str):
    A = per_run[(per_run.dataset == dataset) & (per_run.variant == a)].set_index("seed")
    B = per_run[(per_run.dataset == dataset) & (per_run.variant == b)].set_index("seed")
    seeds = sorted(set(A.index) & set(B.index))
    if len(seeds) < 3:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for ax, (col, title) in zip(axes.ravel(), PAIR_PANELS):
        for s in seeds:
            ax.plot([0, 1], [A.loc[s, col], B.loc[s, col]], color="#b8b7b0", lw=0.7, zorder=1)
        for k, (grp, v) in enumerate(((A, a), (B, b))):
            vals = grp.loc[seeds, col]
            ax.scatter(np.full(len(seeds), k) + np.linspace(-0.04, 0.04, len(seeds)), vals, s=14,
                       color=COLORS[v], marker=MARKERS[v], zorder=3)
            ax.hlines(vals.mean(), k - 0.17, k + 0.17, color=INK, lw=2, zorder=4)
        ax.set_xticks([0, 1], [VARIANT_LABELS[a], VARIANT_LABELS[b]], fontsize=7.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: {VARIANT_LABELS[a]} vs {VARIANT_LABELS[b]} — {len(seeds)} identycznych seedów "
        "(linie łączą ten sam seed, gruba kreska = średnia)", y=1.0, fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def fig_vprime_size_by_variant(per_cycle, dataset: str):
    variants = [v for v in ("evo_mem", "evo_nomem", "heur_mem", "rand_growth", "rand_fixed")
                if not per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == v)].empty]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for v in variants:
        cyc = per_cycle[(per_cycle.dataset == dataset) & (per_cycle.variant == v)]
        stat = cyc.groupby("cycle").agg(vp=("val_rows_after", "median"), k=("k_after", "median"), n=("seed", "count"))
        stat = stat[stat.n >= max(3, int(0.25 * cyc.seed.nunique()))]
        axes[0].plot(stat.index, stat.vp, color=COLORS[v], marker=MARKERS[v], ms=2.5, label=VARIANT_LABELS[v])
        axes[1].plot(stat.index, stat.k, color=COLORS[v], marker=MARKERS[v], ms=2.5, label=VARIANT_LABELS[v])
    axes[0].set_ylabel("rozmiar V' (mediana po seedach)")
    axes[1].set_ylabel("K po decyzji cyklu (mediana)")
    for ax in axes:
        ax.set_xlabel("numer cyklu T' (cykle z ≥ 25% seedów)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Wzrost V' i K w kolejnych cyklach — {DATASET_LABELS[dataset]}", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, f"fig_vprime_k_growth_{dataset}.png")


# ------------------------------------------------------------------------------------------
# 6. przegląd zbiorów
# ------------------------------------------------------------------------------------------
def fig_overview(per_run):
    datasets = [d for d in DATASET_LABELS if d in set(per_run.dataset)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(datasets) * 5 / 2 + 1.8))
    panels = [("test_ba", "test: balanced accuracy"), ("cycles", "liczba cykli T'"), ("final_k", "końcowe K")]
    for ax, (col, title) in zip(axes, panels):
        for i, d in enumerate(datasets):
            vs = [v for v in COLORS if not per_run[(per_run.dataset == d) & (per_run.variant == v)].empty]
            for j, v in enumerate(vs):
                vals = per_run[(per_run.dataset == d) & (per_run.variant == v)][col]
                y = i + (j - (len(vs) - 1) / 2) * 0.15
                ax.errorbar(vals.mean(), y, xerr=vals.std(ddof=1) if len(vals) > 1 else 0, color=COLORS[v],
                            marker=MARKERS[v], ms=4.5, lw=1.2, capsize=0,
                            label=f"{VARIANT_LABELS[v]}" if i == 0 else None)
        ax.set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("średnia ± odch. std. po seedach")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.18), ncol=5)
    fig.tight_layout()
    return _save(fig, "fig_overview_datasets.png")


# ------------------------------------------------------------------------------------------
# 7. dane 2D i granice decyzyjne
# ------------------------------------------------------------------------------------------
def _load_xy(path: str, header: bool):
    df = pd.read_csv(REPO_ROOT / path, header=0 if header else None)
    y = df.iloc[:, -1].to_numpy()
    return df.iloc[:, :-1].to_numpy(dtype=float), y


def _representative_run(per_run, dataset, variant):
    runs = per_run[(per_run.dataset == dataset) & (per_run.variant == variant)]
    if runs.empty:
        return None
    row = runs.iloc[(runs["cycles"] - runs["cycles"].median()).abs().argmin()]
    return STUDY_ROOT / dataset / variant / "runs" / row.run_id, row


def fig_2d(per_run, dataset: str, variants=("evo_mem", "rand_fixed")):
    present = [v for v in variants if _representative_run(per_run, dataset, v)]
    if not present:
        return None
    first_dir, _ = _representative_run(per_run, dataset, present[0])
    cfg = json.loads((first_dir / "metrics" / "config_snapshot.json").read_text(encoding="utf-8"))
    d = cfg["data"]
    Xtr, ytr = _load_xy(d["train_csv"], d["header"])
    Xva, yva = _load_xy(d["validation_csv"], d["header"])
    svm = SVMConfig(kernel="rbf", C=cfg["svm"]["C"], gamma=cfg["svm"]["gamma"], probability=False, standardize=False)
    lo, hi = np.vstack([Xtr, Xva]).min(0), np.vstack([Xtr, Xva]).max(0)
    pad = 0.05 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 260), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 260))
    cls_colors = ["#2a78d6", "#eb6834"]
    cmap = ListedColormap(["#cfe0f5", "#f9d8c9"])
    fig, axes = plt.subplots(1, 1 + len(present), figsize=(4.3 * (1 + len(present)), 4.4))
    ax = axes[0]
    for c in (0, 1):
        ax.scatter(Xtr[ytr == c, 0], Xtr[ytr == c, 1], s=2, alpha=0.35, color=cls_colors[c], label=f"klasa {c} (Train)")
    ax.set_title(f"{DATASET_LABELS[dataset]} — zbiór Train")
    ax.legend(markerscale=4, loc="upper right")
    for ax, v in zip(axes[1:], present):
        run_dir, row = _representative_run(per_run, dataset, v)
        mask = np.load(run_dir / "checkpoints" / "global_best_train_mask.npy")
        ids = np.load(run_dir / "checkpoints" / "final_vprime_ids.npy")
        clf = fit_svm_model(Xtr[mask], ytr[mask], svm)
        zz = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.9)
        ax.contour(gx, gy, zz, levels=[0.5], colors=INK, linewidths=1.0)
        ax.scatter(Xva[ids, 0], Xva[ids, 1], s=3, color="#7a7a74", alpha=0.55, label=f"V' ({len(ids)})")
        ax.scatter(Xtr[mask, 0], Xtr[mask, 1], s=8, facecolor="none", edgecolor=INK, linewidth=0.5,
                   label=f"T' ({int(mask.sum())})")
        ax.set_title(f"{VARIANT_LABELS[v]}\nseed {int(row.seed)}, test BA = {row.test_ba:.3f}", fontsize=9)
        ax.legend(loc="upper right", markerscale=2)
    for ax in axes:
        ax.set_xlabel("x₁")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    axes[0].set_ylabel("x₂")
    fig.tight_layout()
    return _save(fig, f"fig_2d_{dataset}.png")


def main() -> None:
    _style()
    per_run, per_cycle = _load()
    for dataset in sorted(set(per_run.dataset)):
        fig_trajectory(per_run, per_cycle, dataset, "evo_mem")
        fig_decisions(per_run, per_cycle, dataset, "evo_mem")
        fig_increment_difficulty(per_cycle, dataset)
        fig_vprime_size_by_variant(per_cycle, dataset)
        fig_paired(per_run, dataset, "evo_mem", "rand_fixed", f"fig_paired_evo_vs_randfixed_{dataset}.png")
        fig_paired(per_run, dataset, "evo_mem", "evo_nomem", f"fig_paired_memory_{dataset}.png")
        if dataset in {"moons", "circles", "spiral", "classification2d", "chessboard"}:
            fig_2d(per_run, dataset)
    fig_overview(per_run)


if __name__ == "__main__":
    main()
