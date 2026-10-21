"""Wykresy raportu końcowego `followup_phase_switch` v2 -> reports/followup_phase_switch/figures/*.png.

Paleta (walidowana skryptem `validate_palette.js`, kolejność stała): ewolucja = niebieski #2a78d6, losowanie z harmonogramem ewolucji =
pomarańczowy #eb6834, losowanie z własnym automatem = aqua #1baf7a, stary algorytm = szary. Kolor aqua ma kontrast < 3:1, więc zawsze
towarzyszą mu inny kształt markera, legenda i liczby w tabelach.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_analysis as A  # noqa: E402
from ps_common import REPO_ROOT  # noqa: E402

FIGS = REPO_ROOT / "reports" / "followup_phase_switch" / "figures"
C_EVO, C_REP, C_SW, C_OLD = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
C_BAND = "#e9eef6"
INK, INK2, GRID = "#1b1b1a", "#52514e", "#e4e4df"
STOP_COL = {"no_progress_after_vprime_retry": "#2a78d6", "no_k_growth_signal": "#eb6834", "k_limit": "#1baf7a",
            "tprime_no_progress": "#eda100", "perfect_on_max_vprime": "#e87ba4"}
STOP_LABEL = {"no_progress_after_vprime_retry": "brak postępu po próbie V′", "no_k_growth_signal": "ρ_SV < TSV (brak sygnału wzrostu K)",
              "k_limit": "limit K (wszystkie próbki klasy)", "tprime_no_progress": "zastój T′, próba V′ już wykonana",
              "perfect_on_max_vprime": "BA(V′)=1, V′ nierozszerzalne"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 8.8, "axes.titleweight": "bold", "axes.labelsize": 8.2,
        "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.6, "ytick.labelsize": 7.8, "text.color": INK,
        "axes.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 7.8, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.dpi": 100, "savefig.dpi": 220, "lines.linewidth": 1.2,
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wykres:", name)


# ------------------------------------------------------------------------------------------------
def fig_phase_machine() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.set_xlim(-0.3, 10)
    ax.set_ylim(-0.4, 7.2)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.3, color=INK2)

    box(0.3, 3.0, 3.4, 1.5, "optimize_T_prime", "GA dla T′ przy stałym K,\nocena na całym skumulowanym V′", "#eaf1fb", C_EVO)
    box(6.3, 3.0, 3.4, 1.5, "optimize_V_prime", "dobór ΔV′ z puli V∖V′ (Val-GA),\nmodel T′ ustalony; ocena ΔV′", "#fdeee8", C_REP)
    box(3.6, 0.3, 2.8, 1.0, "stop", "kod zakończenia", "#f1f1ee", C_OLD)
    ax.annotate("", xy=(0.3, 3.75), xytext=(-0.25, 3.75), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    ax.text(-0.25, 3.45, "start", fontsize=7.2, color=INK2, ha="left", va="center")
    ax.text(0.3, 2.62, "start: K = K_start", fontsize=6.9, color=INK2, ha="left", va="center")

    def arrow(p, q, rad, text, tx, ty, ha="center", color=INK2):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=11, color=color, lw=1.1))
        ax.text(tx, ty, text, fontsize=7.0, color=INK, ha=ha, va="center", linespacing=1.25)

    arrow((3.7, 4.2), (6.3, 4.2), -0.18, "BA(V′) = 1 (opanowanie V′)\nlub zastój T′ (3 × K↑ bez poprawy,\nρ_SV < TSV, limit K): próba V′", 5.0, 5.3)
    arrow((6.3, 3.3), (3.7, 3.3), -0.18, "trudność ΔV′ przestała rosnąć (2 przyrosty),\nlimit |V′|, brak kandydatów lub limit rozszerzeń", 5.0, 2.55)
    arrow((1.2, 4.55), (2.8, 4.55), -1.7, "BA(V′) < 1, brak zastoju:\nK ↑ (najwyżej raz na cykl)", 2.0, 6.55)
    arrow((7.2, 4.55), (8.8, 4.55), -1.7, "trudność ΔV′ nadal rośnie:\nkolejne rozszerzenie", 8.0, 6.55)
    arrow((2.5, 3.0), (3.9, 1.15), 0.10, "po próbie V′ brak postępu\n(także po K ↑); V′ nierozszerzalne\nlub bezpiecznik max_cycles", 0.1, 0.75, ha="left")
    ax.text(5.0, -0.3, "Pełne V i test nie uczestniczą w żadnym przejściu — tylko obserwacja po decyzji.", ha="center", fontsize=7.2, color=INK2, style="italic")
    save(fig, "rys01_maszyna_faz.png")


# ------------------------------------------------------------------------------------------------
def fig_size(R: dict) -> None:
    runs, old = R["D"]["runs"], R["D"]["old"]
    base = runs[(runs.series == "v2_full")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    ds_rev = A.DATASETS[::-1]
    specs = [("evo_switch", C_EVO, "o", -0.24, "ewolucja"), ("rand_switch", C_SW, "s", 0.0, "losowanie (własny automat)"),
             ("old", C_OLD, "D", 0.24, "stary algorytm (ewolucja)")]
    for k, ax in enumerate(axes):
        for v, col, mk, off, lab in specs:
            for i, ds in enumerate(ds_rev):
                if v == "old":
                    g = old[(old.dataset == ds) & (old.split == 2026)]
                    x = g.vprime_size.to_numpy(float)
                    if k == 1:
                        x = x / base[base.dataset == ds].full_validation_rows.iloc[0]
                else:
                    g = base[(base.dataset == ds) & (base.variant == v)]
                    x = (g.vprime_rows if k == 0 else g.vprime_fraction).to_numpy(float)
                m, sd = x.mean(), x.std(ddof=1)
                jit = np.random.default_rng(i).uniform(-0.06, 0.06, len(x))
                ax.scatter(x, np.full(len(x), i + off) + jit, s=5, color=col, alpha=0.28, linewidths=0, zorder=2)
                ax.plot([max(m - sd, 1e-4), m + sd], [i + off] * 2, color=col, lw=1.3, zorder=3)
                ax.scatter([m], [i + off], s=26, color=col, marker=mk, edgecolor="white", linewidths=0.6, zorder=4,
                           label=lab if i == 0 and k == 0 else None)
        ax.set_yticks(range(len(ds_rev)))
        ax.set_yticklabels([A.LABEL[d] for d in ds_rev])
        ax.grid(axis="y", visible=False)
    axes[0].set_xscale("log")
    axes[0].set_xticks([100, 200, 500, 1000, 2000, 4000])
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].minorticks_off()
    axes[0].set_xlabel("końcowe |V′| (próbki; skala logarytmiczna)")
    axes[1].set_xlabel("końcowy udział |V′| / |V|")
    axes[1].axvline(0.25, color="#b4533a", lw=0.9, ls=(0, (4, 3)))
    axes[1].text(0.252, len(ds_rev) - 0.45, "limit 0,25", fontsize=7, color="#b4533a", va="center")
    axes[0].set_title("A. Rozmiar końcowy V′")
    axes[1].set_title("B. Udział V′ w pełnym V")
    axes[1].set_xlim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.095, "punkt = średnia, odcinek = ±1 SD, drobne punkty = pojedyncze przebiegi (podział 2026, 20 seedów). "
             "Losowanie z harmonogramem ewolucji ma z konstrukcji ten sam |V′| co ewolucja.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "rys02_rozmiar_vprime.png")


def fig_quality_vs_size(R: dict) -> None:
    runs, old = R["D"]["runs"], R["D"]["old"]
    base = runs[(runs.series == "v2_full") & (runs.variant == "evo_switch")]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1))
    for ax, ds in zip(axes.ravel(), A.DATASETS):
        g = base[base.dataset == ds]
        o = old[(old.dataset == ds) & (old.split == 2026)]
        ax.scatter(o.vprime_size, o.test_ba, s=11, color=C_OLD, marker="D", alpha=0.75, linewidths=0, label="stary algorytm")
        ax.scatter(g.vprime_rows, g.test_ba, s=13, color=C_EVO, alpha=0.85, linewidths=0, label="nowy (ewolucja)")
        ax.set_xscale("log")
        ax.set_title(A.LABEL[ds], fontsize=8.2)
        ax.tick_params(labelsize=6.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        lo_, hi_ = min(g.vprime_rows.min(), o.vprime_size.min()), max(g.vprime_rows.max(), o.vprime_size.max())
        ticks, last = [], None
        for t_ in (100, 200, 300, 500, 1000, 2000, 3000, 5000):
            if lo_ / 1.5 < t_ < hi_ * 1.5 and (last is None or t_ / last >= 1.9):
                ticks.append(t_)
                last = t_
        ax.set_xticks(ticks)
        ax.minorticks_off()
    axes[1, 3].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].legend(h, l, loc="center", fontsize=7.6)
    fig.supxlabel("końcowe |V′| (skala logarytmiczna)", fontsize=8, y=0.01)
    fig.supylabel("BA na teście (obserwacja)", fontsize=8, x=0.005)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    save(fig, "rys03_jakosc_vs_rozmiar_vprime.png")


# ------------------------------------------------------------------------------------------------
def fig_trajectories(R: dict) -> None:
    steps = R["D"]["steps"]
    runs = R["D"]["runs"]
    sel = [("moons", "Moons"), ("chessboard", "Szachownica"), ("spiral", "Spiral (trudniejszy zbiór)")]
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.2), gridspec_kw={"height_ratios": [2.3, 1.2, 1.2]}, sharex="col")
    for col, (ds, title) in enumerate(sel):
        rid = runs[(runs.series == "v2_full") & (runs.dataset == ds) & (runs.variant == "evo_switch") & (runs.seed == 42)].run_id.iloc[0]
        g = steps[steps.run_id == rid].sort_values("cycle").reset_index(drop=True)
        g["perfect"] = g.tprime_perfect.astype(str).str.lower() == "true"
        g["kinc"] = g.k_increased.astype(str).str.lower() == "true"
        a1, a2, a3 = axes[:, col]
        # pasma faz V'
        spans, start = [], None
        for i, ph in enumerate(g.phase):
            if ph == "optimize_V_prime" and start is None:
                start = g.cycle[i]
            if ph != "optimize_V_prime" and start is not None:
                spans.append((start - 0.5, g.cycle[i - 1] + 0.5))
                start = None
        if start is not None:
            spans.append((start - 0.5, g.cycle.iloc[-1] + 0.5))
        for ax in (a1, a2, a3):
            for lo, hi in spans:
                ax.axvspan(lo, hi, color=C_BAND, lw=0, zorder=0)
            ax.axvline(g.cycle.iloc[-1], color="#c23b3b", lw=0.9, ls=(0, (3, 2)))
        t = g[g.phase == "optimize_T_prime"]
        v = g[g.phase == "optimize_V_prime"]
        a1.plot(t.cycle, t.tprime_fitness, color=C_EVO, lw=1.0, marker="o", ms=2.6, zorder=3, label="BA(V′), krok T′")
        pf = t[t.perfect]
        a1.scatter(pf.cycle, pf.tprime_fitness, marker="*", s=34, color=INK, zorder=4, label="BA(V′) = 1")
        a1.scatter(v.cycle, v.delta_metric, marker="D", s=7, facecolor="white", edgecolor=C_OLD, linewidths=0.8, zorder=3, label="BA(ΔV′), krok V′")
        a1.set_title(title, fontsize=8.6)
        a1.set_ylim(0.25 if ds == "spiral" else 0.3, 1.03)
        a2.step(g.cycle, g.vprime_rows_after, where="post", color=INK2, lw=1.1)
        a3.step(g.cycle, g.k_after, where="post", color=INK2, lw=1.1)
        kk = g[g.kinc]
        a3.scatter(kk.cycle, kk.k_after, marker="^", s=9, color=C_EVO, zorder=4, label="zwiększenie K")
        a1.text(g.cycle.iloc[-1] - 0.8, 0.32 if ds != "spiral" else 0.27, str(g.stop_decision.iloc[-1]).replace("stop_", ""), color="#c23b3b", fontsize=6.3,
                ha="right", va="bottom", rotation=0)
        a3.set_xlabel("cykl (krok automatu)")
        if col == 0:
            a1.set_ylabel("BA")
            a2.set_ylabel("|V′|")
            a3.set_ylabel("K (próbki/klasę)")
        if col == 0:
            h, l = a1.get_legend_handles_labels()
            h3, l3 = a3.get_legend_handles_labels()
            fig.legend(h + h3, l + l3, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.025))
    fig.text(0.5, -0.062, "Jasne tło = faza optimize_V_prime; czerwona linia = stop (powód przy linii). Podział 2026, seed 42, wariant ewolucyjny.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "rys04_trajektorie_faz.png")


# ------------------------------------------------------------------------------------------------
def fig_forest(R: dict) -> None:
    P = R["P"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5), sharey=True)
    panels = [("A. BA test, podział 2026", P["pair_base"], "test_ba", A.DATASETS, False),
              ("B. BA test, 4 podziały", P["pair_splits"], "test_ba", A.SPLIT_DEP, True),
              ("C. BA pełne V, podział 2026", P["pair_base"], "fullv_ba", A.DATASETS, False)]
    ds_rev = A.DATASETS[::-1]
    for ax, (title, df, metric, dss, hier) in zip(axes, panels):
        ax.axvspan(-A.MARGIN * 100, A.MARGIN * 100, color="#eef1f5", lw=0, zorder=0)
        ax.axvline(0, color="#8a8a85", lw=0.8, zorder=1)
        for i, ds in enumerate(ds_rev):
            if ds not in dss:
                continue
            r = df[(df.metryka == metric) & (df["zbiór"] == ds)].iloc[0]
            lo, hi = (r.h_lo, r.h_hi) if hier else (r.ci_lo, r.ci_hi)
            ax.plot([lo * 100, hi * 100], [i, i], color=C_EVO, lw=1.6, zorder=3)
            ax.scatter([r["diff"] * 100], [i], s=26, color=C_EVO, edgecolor="white", linewidths=0.6, zorder=4)
        ax.set_yticks(range(len(ds_rev)))
        ax.set_yticklabels([A.LABEL[d] for d in ds_rev])
        ax.set_title(title, fontsize=8.0)
        ax.set_xlabel("ewolucja − losowanie [p.p.]")
        ax.set_xlim(-0.95, 0.95)
        ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.045, "punkt = średnia różnica sparowana, odcinek = 95% CI (B: bootstrap hierarchiczny); szare pasmo = ±0,5 p.p. (przyjęta granica praktycznej istotności). "
             "A, C: n = 20 seedów; B: 4 podziały × 5 seedów. Losowanie z identycznym harmonogramem.", ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout()
    save(fig, "rys05_roznice_ewolucja_losowanie.png")


def fig_difficulty(R: dict) -> None:
    steps = R["D"]["steps"]
    m = steps[steps.series.isin(A.MAIN_SERIES) & (steps.phase == "optimize_V_prime")]
    ph = R["P"]["phase"]["phases"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for v, col, mk, lab in [("evo_switch", C_EVO, "o", "ewolucja"), ("rand_replay", C_REP, "^", "losowanie (harmonogram z ewolucji)"),
                            ("rand_switch", C_SW, "s", "losowanie (własny automat)")]:
        g = m[m.variant == v].groupby("expansions_in_phase").delta_error_rate.agg(["mean", "std", "size"]).reset_index()
        g = g[g["size"] >= 50]
        se = g["std"] / np.sqrt(g["size"])
        ax.fill_between(g.expansions_in_phase, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se, color=col, alpha=0.16, lw=0)
        ax.plot(g.expansions_in_phase, g["mean"], color=col, marker=mk, ms=3.6, lw=1.3, label=lab)
    ax.set_xlabel("numer rozszerzenia w fazie V′")
    ax.set_ylabel("błąd klasyfikacji na ΔV′ (trudność)")
    ax.set_title("A. Trudność kolejnych przyrostów")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 0.42)
    ax2 = axes[1]
    e = ph[ph.variant == "evo_switch"].L.value_counts().sort_index()
    ax2.bar(e.index, e.values, color=C_EVO, width=0.7)
    for x, y in zip(e.index, e.values):
        ax2.text(x, y + 8, str(y), ha="center", fontsize=6.8, color=INK2)
    ax2.set_xlabel("liczba rozszerzeń w jednej fazie V′")
    ax2.set_ylabel("liczba faz")
    ax2.set_title("B. Długość fazy V′ (ewolucja)")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "rys06_trudnosc_dV_i_dlugosc_fazy.png")


def fig_cost(R: dict) -> None:
    runs = R["D"]["runs"]
    cr = A.cost_ratio_table(runs)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ds_rev = A.DATASETS[::-1]
    specs = [("treningi SVM", C_EVO, "o", -0.22), ("wiersze predict", C_REP, "s", 0.0), ("czas zegarowy", C_SW, "^", 0.22)]
    for name, col, mk, off in specs:
        for i, ds in enumerate(ds_rev):
            r = cr[cr["zbiór"] == ds].iloc[0]
            lo, hi = r[name + " CI"]
            ax.plot([lo, hi], [i + off] * 2, color=col, lw=1.5, zorder=3)
            ax.scatter([r[name]], [i + off], s=26, color=col, marker=mk, edgecolor="white", linewidths=0.6, zorder=4, label=name if i == 0 else None)
    ax.axvline(1, color="#8a8a85", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 1.25, 1.5, 2, 2.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.minorticks_off()
    ax.set_yticks(range(len(ds_rev)))
    ax.set_yticklabels([A.LABEL[d] for d in ds_rev])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("koszt ewolucji / koszt losowania przy identycznym harmonogramie (średnia geometryczna z par po seedach)")
    ax.legend(loc="lower right", ncol=1)
    fig.text(0.5, -0.03, "odcinek = 95% CI (bootstrap z logarytmu stosunku, 20 par na zbiór, podział 2026). Czas zegarowy pochodzi z równoległych uruchomień "
             "na jednej maszynie — miara pomocnicza.", ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout()
    save(fig, "rys07_koszt_ewolucja_do_losowania.png")


def fig_stops(R: dict) -> None:
    runs = R["D"]["runs"]
    m = runs[runs.role == "główne"]
    rows = []
    for ds in A.DATASETS[::-1]:
        for v in ("rand_switch", "evo_switch"):
            g = m[(m.dataset == ds) & (m.variant == v)]
            vc = g.stop_decision.str.replace("stop_", "").value_counts()
            rows.append((ds, v, len(g), vc))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ylab, used = [], set()
    for i, (ds, v, n, vc) in enumerate(rows):
        left = 0.0
        for key in STOP_COL:
            c = int(vc.get(key, 0))
            if c == 0:
                continue
            w = c / n
            ax.barh(i, w, left=left, color=STOP_COL[key], edgecolor="white", linewidth=1.5, height=0.68)
            used.add(key)
            if w >= 0.075:
                ax.text(left + w / 2, i, str(c), ha="center", va="center", fontsize=6.8, color=INK, fontweight="bold")
            left += w
        ylab.append(f"{A.LABEL[ds]} · {'ewolucja' if v == 'evo_switch' else 'losowanie (własny)'} (n={n})")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("udział przebiegów")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    keys = [k for k in STOP_COL if k in used]
    fig.legend([Patch(facecolor=STOP_COL[k], edgecolor="white") for k in keys], [STOP_LABEL[k] for k in keys], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.115))
    fig.text(0.5, -0.19, "liczby w segmentach = liczba przebiegów (segmenty < 7,5% bez liczby). Wszystkie serie główne (podziały 2026–2029). "
             "rand_replay kończy się z konstrukcji razem z przebiegiem ewolucyjnym.", ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout()
    save(fig, "rys08_powody_zakonczenia.png")


def main() -> None:
    style()
    R = A.build_all(save=False)
    fig_phase_machine()
    fig_size(R)
    fig_quality_vs_size(R)
    fig_trajectories(R)
    fig_forest(R)
    fig_difficulty(R)
    fig_cost(R)
    fig_stops(R)


if __name__ == "__main__":
    main()
