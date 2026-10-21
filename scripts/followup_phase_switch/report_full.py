"""Raport pełnego badania v2 (seria `v2_full`) -> reports/followup_phase_switch/v2/RAPORT_PELNE_V2.md.

Wszystkie liczby pochodzą z tabel `aggregate.py --version v2` (per_run.csv, per_step.csv) oraz — dla porównania ze
starym algorytmem — z `reports/followup_phase_switch/../followup_stop_rule/tables/per_run.csv` (tylko odczyt).
Nie nadpisuje `RAPORT_V2.md` (smoke) ani raportów v1.

    python scripts/followup_phase_switch/report_full.py --pytest "140 passed"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import REPO_ROOT, REPORT_ROOT_V2, TABLES_V2, fsr  # noqa: E402
from report_v2 import as_bool, cost_table, entry_stats, f, invariants, results_table, table  # noqa: E402

OUT = REPORT_ROOT_V2 / "RAPORT_PELNE_V2.md"
OLD_PER_RUN = REPO_ROOT / "reports" / "followup_stop_rule" / "tables" / "per_run.csv"
SERIES = "v2_full"
RNG = np.random.default_rng(20261020)


def boot_ci(d: np.ndarray, n: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return (np.nan, np.nan)
    means = RNG.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_ci_table(runs: pd.DataFrame, a: str, b: str, metrics: list[str]) -> tuple[str, pd.DataFrame]:
    rows = []
    for ds, g in runs.groupby("dataset"):
        ga = g[g.variant == a].set_index("seed")
        gb = g[g.variant == b].set_index("seed")
        common = sorted(set(ga.index) & set(gb.index))
        for m in metrics:
            d = (ga.loc[common, m] - gb.loc[common, m]).astype(float).to_numpy()
            lo, hi = boot_ci(d)
            blo, bhi = boot_ci(d, alpha=0.05 / runs.dataset.nunique())
            rows.append({"zbiór": ds, "metryka": m, "n": len(common), a: f(ga.loc[common, m].mean()), b: f(gb.loc[common, m].mean()),
                         "różnica": f(np.nanmean(d)), "95% CI": f"[{f(lo)}; {f(hi)}]",
                         "CI wyklucza 0": "tak" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "nie",
                         "CI po korekcie Bonferroniego": "tak" if (not np.isnan(blo) and (blo > 0 or bhi < 0)) else "nie",
                         f"{a} większe/mniejsze": f"{int((d > 0).sum())}/{int((d < 0).sum())}"})
    df = pd.DataFrame(rows)
    return table(df), df


def old_vs_new(runs: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    old = pd.read_csv(OLD_PER_RUN, low_memory=False)
    old = old[(old.split == fsr.BASE_SPLIT) & (old.variant == "evo_mem")].set_index(["dataset", "seed"])
    new = runs[runs.variant == "evo_switch"].set_index(["dataset", "seed"])
    rows = []
    for ds in sorted(new.index.get_level_values(0).unique()):
        seeds = sorted(set(new.loc[ds].index) & set(old.loc[ds].index)) if ds in old.index.get_level_values(0) else []
        if not seeds:
            continue
        n_ = new.loc[ds].loc[seeds]
        o_ = old.loc[ds].loc[seeds]
        d = (n_.test_ba - o_.test_ba).to_numpy(dtype=float)
        lo, hi = boot_ci(d)
        rows.append({"zbiór": ds, "n": len(seeds),
                     "\\|V′\\| stary (śr.)": f(o_.vprime_size.mean(), 0), "\\|V′\\| nowy (śr.)": f(n_.vprime_rows.mean(), 0),
                     "stosunek nowy/stary": f(n_.vprime_rows.mean() / o_.vprime_size.mean(), 2),
                     "K końc. stary": f(o_.final_k.mean(), 0), "K końc. nowy": f(n_.final_k.mean(), 0),
                     "BA test stary": f(o_.test_ba.mean()), "BA test nowy": f(n_.test_ba.mean()),
                     "różnica (nowy − stary)": f(np.nanmean(d)), "95% CI": f"[{f(lo)}; {f(hi)}]",
                     "CI wyklucza 0": "tak" if (lo > 0 or hi < 0) else "nie"})
    df = pd.DataFrame(rows)
    return table(df), df


def stop_table(runs: pd.DataFrame) -> str:
    rows = []
    for (ds, v), g in runs.groupby(["dataset", "variant"]):
        vc = g.stop_decision.value_counts()
        rows.append({"zbiór": ds, "wariant": v, "n": len(g), **{k.replace("stop_", ""): int(vc.get(k, 0)) for k in
                    ["stop_no_progress_after_vprime_retry", "stop_no_k_growth_signal", "stop_k_limit", "stop_tprime_no_progress",
                     "stop_perfect_on_max_vprime", "stop_replay_end", "stop_max_cycles"]}})
    return table(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest", default="(nie podano)")
    args = ap.parse_args()
    runs_all = pd.read_csv(TABLES_V2 / "per_run.csv")
    steps_all = pd.read_csv(TABLES_V2 / "per_step.csv", low_memory=False)
    runs = runs_all[runs_all.series == SERIES].copy()
    steps = steps_all[steps_all.series == SERIES].copy()
    inv = invariants(steps, runs)
    ent = entry_stats(steps)
    base_ba = runs.groupby("dataset").baseline_full_svm_test_ba.first()

    main_tab, main_df = paired_ci_table(runs, "evo_switch", "rand_replay",
                                        ["test_ba", "fullv_ba", "vprime_rows", "mean_delta_difficulty"])
    aux_tab, _ = paired_ci_table(runs, "evo_switch", "rand_switch", ["test_ba", "vprime_rows"])
    old_tab, old_df = old_vs_new(runs)
    n_sig_main = int(((main_df.metryka == "test_ba") & (main_df["CI wyklucza 0"] == "tak")).sum())
    n_sig_bonf = int(((main_df.metryka == "test_ba") & (main_df["CI po korekcie Bonferroniego"] == "tak")).sum())
    sig_rows = main_df[(main_df.metryka == "test_ba") & (main_df["CI wyklucza 0"] == "tak")]
    sig_txt = "; ".join(f"{r['zbiór']} ({r['różnica']})" for _, r in sig_rows.iterrows()) or "brak"
    n_ds = runs.dataset.nunique()
    own = runs[runs.variant != "rand_replay"]
    frac_stats = runs.groupby("dataset").vprime_fraction.agg(["mean", "max"])
    not_max_cycles = inv["max_cycles_stops"] == 0

    t = steps[(steps.phase == "optimize_T_prime") & (steps.variant != "rand_replay")]
    sw = t[t.next_phase == "optimize_V_prime"]
    mastered = int(as_bool(sw.tprime_perfect).sum())
    retry_checks = t[as_bool(t.retry_check)]
    retry_ok = int(as_bool(retry_checks.retry_progress).sum())

    md = f"""# followup_phase_switch v2 — pełne badanie (seria `{SERIES}`)

Raport generowany przez `scripts/followup_phase_switch/report_full.py` z tabel `reports/followup_phase_switch/v2/tables/`.
Zakres: podział bazowy 2026 (jak w badaniu `followup_stop_rule`), {n_ds} zbiorów × 20 seedów (42–61) × 3 warianty
(`evo_switch`, `rand_replay` — główne porównanie, `rand_switch` — pomocniczo) = **{len(runs)} przebiegów**, ustawienia domyślne automatu
(`RAPORT_V2.md` §3), `max_vprime_rows: null`, `k_growth_requires_sv_signal: true`. Dodatkowe podziały danych (2027–2029) **nie** były uruchamiane.
Cykli (kroków automatu) w serii: {inv['steps']}. Opis automatu i warunków: `RAPORT_V2.md` §2; audyt: `AUDYT_v2.md`.
`pytest`: {args.pytest}. Wyniki v1, smoke v2 i stary tryb `dagasvm_phase` są nietknięte.

Trudność przyrostu ΔV′ i jakość modelu są osobnymi właściwościami; raport nie wnioskuje z jednej o drugiej. BA na teście i na pełnym V to wyłącznie obserwacje.
`rand_replay` ma identyczny z `evo_switch` (ten sam seed) harmonogram faz, K i \\|ΔV′\\| — różni się tylko wyborem próbek do ΔV′, więc \\|V′\\|, K i liczba cykli są
z konstrukcji równe; `rand_switch` stosuje te same reguły przy własnym harmonogramie, więc jego \\|V′\\| bywa mniejszy.

## 1. Wyniki per zbiór i wariant (n = 20 seedów w każdej komórce)

Pełny SVM (cały zbiór treningowy) ma BA test: {', '.join(f'{d} {f(v)}' for d, v in base_ba.items())}.

{results_table(runs, [SERIES])}

Rozmiar V′ względem V (wszystkie warianty): {', '.join(f"{d} śr. {f(r['mean'], 3)}, maks. {f(r['max'], 3)}" for d, r in frac_stats.iterrows())}.
Maks. \\|V′\\| na Szachownicy = {inv['max_vprime_chess']} (stary algorytm: ≈ 3600).

## 2. Ewolucja a losowanie przy tym samym harmonogramie (evo_switch − rand_replay)

Różnice sparowane po seedach (n = 20), 95% CI z bootstrapu (10 000 prób). Różnica liczby próbek/K/cykli jest zerowa z konstrukcji.

{main_tab}

Przedział ufności różnicy BA test wyklucza zero w **{n_sig_main}** z {n_ds} zbiorów: {sig_txt}; po korekcie Bonferroniego na {n_ds} zbiorów (przedziały
{f(100 * (1 - 0.05 / n_ds), 1)} %) — w **{n_sig_bonf}**. Przedziały w tabeli są liczone osobno dla każdej metryki i zbioru, bez korekty na wielokrotność porównań
(tabela ma {len(main_df)} wierszy), więc pojedyncze „tak” przy BA pełnego V (obserwacja) nie jest mocnym dowodem. Poza zbiorami wskazanymi wyżej nie ma podstaw,
by twierdzić o przewadze któregokolwiek doboru. Ewolucja buduje przyrosty o wyższym odsetku błędów modelu (wiersz `mean_delta_difficulty`) — to jest własność przyrostu, a nie dowód lepszego modelu.

Pomocniczo, evo_switch − rand_switch (różne harmonogramy; różnica miesza efekt doboru z efektem rozmiaru V′):

{aux_tab}

## 3. Czy mniejsze V′ pogarsza jakość? Porównanie ze starym algorytmem (evo_mem, ten sam podział i seedy)

Stary: `dagasvm_phase`, reguła rozszerzona, `followup_stop_rule`. Nowy: `evo_switch` (v2). To porównanie opisowe (różne algorytmy przełączania faz i K), nie izoluje wpływu rozmiaru V′.

{old_tab}

## 4. Skąd pochodzą wejścia do fazy V′ i jak kończą się próby

Wejścia do fazy V′ (warianty z własnym automatem): **{len(sw)}**, z czego z opanowania V′ (BA=1): **{mastered}** ({f(mastered / max(len(sw), 1) * 100, 1)} %),
z zastoju T′ (próba): {len(sw) - mastered}. Kontrole po próbie V′: {len(retry_checks)}, z czego poziom odniesienia osiągnięty: {retry_ok}.

{table(ent)}

### Powody zakończenia

{stop_table(runs)}

## 5. Koszt obliczeń (średnia na przebieg)

{cost_table(runs, [SERIES])}

## 6. Niezmienniki na wszystkich {inv['steps']} cyklach

| sprawdzenie | wynik |
|---|---|
| cykle V′ bez poprzedzającej decyzji przełączenia (lub kontynuacji fazy V′) | {inv['v_entry_bad']} |
| faza V′ zakończona decyzją „do T′”, a następny cykl nie jest T′ | {inv['back_to_t_bad']} |
| `prev_phase` niezgodne z poprzednim cyklem | {inv['prev_phase_bad']} |
| cykle, w których jednocześnie wzrosło K i rozszerzono V′ | {inv['k_and_v_same_step']} |
| zmiana K w cyklu V′ / zmiana V′ w cyklu T′ | {inv['k_change_in_v_step']} / {inv['v_changed_in_t_step']} |
| cykl V′, w którym V′ urosło inaczej niż o jedno ΔV′ | {inv['v_step_not_one_expansion']} |
| BA(V′)=1 i wzrost K w tym samym cyklu (warianty z własnym automatem) | {inv['perfect_increases_k_own']} |
| j.w. w `rand_replay` (K narzucone harmonogramem evo) | {inv['perfect_increases_k_replay']} (informacyjnie) |
| \\|V′\\| powyżej limitu / `vprime_fraction` ≠ \\|V′\\|/\\|V\\| | {inv['over_cap']} / {inv['frac_bad']} |
| więcej niż jedna próba V′ po zastoju przy tym samym K | {inv['dup_retry_per_k']} |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | {inv['max_cycles_stops']} |
| pary `rand_replay` z identycznym końcowym \\|V′\\| co `evo_switch` | {inv['replay_equal_size']} / {inv['replay_pairs']} |
| kontrole po próbie V′ bez osiągnięcia poziomu: dalej zwiększenie K / stop | {inv['retry_fail_then_k_increase']} / {inv['retry_fail_then_stop']} (z {inv['retry_fail_checks']}) |

Zachowanie zgodne ze specyfikacją na wszystkich cyklach: {'TAK' if (inv['v_entry_bad'] + inv['back_to_t_bad'] + inv['prev_phase_bad'] + inv['k_and_v_same_step'] + inv['k_change_in_v_step'] + inv['v_changed_in_t_step'] + inv['v_step_not_one_expansion'] + inv['perfect_increases_k_own'] + inv['over_cap'] + inv['frac_bad'] + inv['dup_retry_per_k']) == 0 and not_max_cycles else 'NIE — patrz tabela'}.

## 7. Ograniczenia

* Jeden podział danych (2026); wariancja między podziałami nie była badana.
* Wartości domyślne automatu nie były strojone; wrażliwość (przyrost ΔV′, progi trudności, `max_no_improvement_k_increases`) nie była badana.
* `k_growth_requires_sv_signal: true` (bramka ρ_SV) wyznacza zakończenie w {int((runs.stop_decision.isin(['stop_no_k_growth_signal'])).sum())} przebiegach i ogranicza K; wpływ bramki na jakość był sprawdzony tylko na Szachownicy (smoke, `RAPORT_V2.md`).
* Digits: limit 0,25·\\|V\\| ({f(runs[runs.dataset == 'digits'].vprime_cap_rows.iloc[0], 0)} próbek) jest praktycznie osiągany i K dochodzi do limitu klasy (`stop_k_limit`).
* Skład klasowy ΔV′ przy doborze ewolucyjnym i losowym nie jest wymuszany (zob. `RAPORT_V2.md` §9).
* Koszt doboru ewolucyjnego obejmuje ocenę całej puli kandydatów raz na rozszerzenie.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"zapisano {OUT} ({len(md)} znaków)")


if __name__ == "__main__":
    main()
