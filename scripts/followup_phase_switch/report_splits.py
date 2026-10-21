"""Raport podziałów dodatkowych v2 (serie `v2_split2027..2029` + podział bazowy 2026 z serii `v2_full`, seedy 42–46)
-> reports/followup_phase_switch/v2/RAPORT_PODZIALY_V2.md.

Układ zrównoważony jak w `followup_stop_rule`: 4 podziały × 5 seedów (42–46) × 6 zbiorów zależnych od podziału (Szachownica czyta stałe
`data/raw`, więc nie bierze udziału). Liczby pochodzą z `reports/followup_phase_switch/v2/tables/` (`aggregate.py --version v2`) oraz,
dla porównania ze starym algorytmem, z `reports/followup_stop_rule/tables/per_run.csv` (tylko odczyt).

    python scripts/followup_phase_switch/report_splits.py --pytest "140 passed"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_common import REPORT_ROOT_V2, TABLES_V2  # noqa: E402
from report_full import OLD_PER_RUN, boot_ci  # noqa: E402
from report_v2 import f, invariants, table  # noqa: E402

OUT = REPORT_ROOT_V2 / "RAPORT_PODZIALY_V2.md"
SPLITS = [2026, 2027, 2028, 2029]
SEEDS = list(range(42, 47))
VARIANTS = ["evo_switch", "rand_replay", "rand_switch"]


def per_split_table(runs: pd.DataFrame) -> str:
    rows = []
    for ds, g in runs.groupby("dataset"):
        for sp in SPLITS:
            gs = g[g.split == sp]
            e, r, w = (gs[gs.variant == v] for v in VARIANTS)
            rows.append({"zbiór": ds, "podział": sp, "n": len(e), "\\|V\\|": int(gs.full_validation_rows.iloc[0]),
                         "\\|V′\\| evo (śr.)": f(e.vprime_rows.mean(), 0), "\\|V′\\|/\\|V\\|": f(e.vprime_fraction.mean(), 3),
                         "K końc. evo": f(e.final_k.mean(), 0),
                         "BA test evo": f(e.test_ba.mean()), "BA test rand_replay": f(r.test_ba.mean()),
                         "BA test rand_switch": f(w.test_ba.mean()),
                         "evo − replay": f((e.set_index("seed").test_ba - r.set_index("seed").test_ba).mean()),
                         "pełny SVM": f(gs.baseline_full_svm_test_ba.mean())})
    return table(pd.DataFrame(rows))


def pooled_pairs(runs: pd.DataFrame, a: str, b: str, metric: str) -> pd.DataFrame:
    rows = []
    nds = runs.dataset.nunique()
    for ds, g in runs.groupby("dataset"):
        ga = g[g.variant == a].set_index(["split", "seed"])
        gb = g[g.variant == b].set_index(["split", "seed"])
        common = ga.index.intersection(gb.index)
        d = (ga.loc[common, metric] - gb.loc[common, metric]).to_numpy(dtype=float)
        lo, hi = boot_ci(d)
        blo, bhi = boot_ci(d, alpha=0.05 / nds)
        rows.append({"zbiór": ds, "n": len(common), a: f(ga.loc[common, metric].mean()), b: f(gb.loc[common, metric].mean()),
                     "różnica": f(d.mean()), "95% CI": f"[{f(lo)}; {f(hi)}]",
                     "CI wyklucza 0": "tak" if (lo > 0 or hi < 0) else "nie",
                     "po korekcie Bonferroniego": "tak" if (blo > 0 or bhi < 0) else "nie",
                     f"{a} większe/mniejsze": f"{int((d > 0).sum())}/{int((d < 0).sum())}"})
    return pd.DataFrame(rows)


def variance_table(runs: pd.DataFrame) -> str:
    """Odch. std. między podziałami (średnie podziałów) wobec odch. std. w obrębie podziału (po seedach), osobno dla wariantów."""
    rows = []
    for ds, g in runs.groupby("dataset"):
        row = {"zbiór": ds}
        for v in VARIANTS:
            gv = g[g.variant == v]
            means = gv.groupby("split").test_ba.mean()
            within = gv.groupby("split").test_ba.std(ddof=1).mean()
            row[f"{v}: między podziałami"] = f(means.std(ddof=1))
            row[f"{v}: w obrębie podziału"] = f(within)
        rows.append(row)
    return table(pd.DataFrame(rows))


def old_vs_new(runs: pd.DataFrame) -> str:
    old = pd.read_csv(OLD_PER_RUN, low_memory=False)
    old = old[(old.variant == "evo_mem") & old.split.isin(SPLITS)].set_index(["dataset", "split", "seed"])
    new = runs[runs.variant == "evo_switch"].set_index(["dataset", "split", "seed"])
    rows = []
    nds = runs.dataset.nunique()
    for ds in sorted(new.index.get_level_values(0).unique()):
        idx = new.loc[ds].index.intersection(old.loc[ds].index)
        n_, o_ = new.loc[ds].loc[idx], old.loc[ds].loc[idx]
        d = (n_.test_ba - o_.test_ba).to_numpy(dtype=float)
        lo, hi = boot_ci(d)
        blo, bhi = boot_ci(d, alpha=0.05 / nds)
        rows.append({"zbiór": ds, "n": len(idx), "\\|V′\\| stary": f(o_.vprime_size.mean(), 0), "\\|V′\\| nowy": f(n_.vprime_rows.mean(), 0),
                     "stosunek": f(n_.vprime_rows.mean() / o_.vprime_size.mean(), 2),
                     "BA test stary": f(o_.test_ba.mean()), "BA test nowy": f(n_.test_ba.mean()),
                     "różnica": f(d.mean()), "95% CI": f"[{f(lo)}; {f(hi)}]", "CI wyklucza 0": "tak" if (lo > 0 or hi < 0) else "nie",
                     "po korekcie Bonferroniego": "tak" if (blo > 0 or bhi < 0) else "nie"})
    return table(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest", default="(nie podano)")
    args = ap.parse_args()
    runs_all = pd.read_csv(TABLES_V2 / "per_run.csv")
    steps_all = pd.read_csv(TABLES_V2 / "per_step.csv", low_memory=False)
    extra = runs_all[runs_all.series.isin([f"v2_split{s}" for s in SPLITS[1:]])]
    base = runs_all[(runs_all.series == "v2_full") & runs_all.seed.isin(SEEDS) & runs_all.dataset.isin(extra.dataset.unique())]
    runs = pd.concat([base, extra], ignore_index=True)
    keep_ids = set(runs.run_id)
    steps = steps_all[steps_all.run_id.isin(keep_ids)] if "run_id" in steps_all else steps_all[steps_all.series.isin(runs.series.unique())]
    inv = invariants(extra_steps := steps_all[steps_all.run_id.isin(set(extra.run_id))], extra)
    n_new = len(extra)
    main_df = pooled_pairs(runs, "evo_switch", "rand_replay", "test_ba")
    full_df = pooled_pairs(runs, "evo_switch", "rand_replay", "fullv_ba")
    aux_df = pooled_pairs(runs, "evo_switch", "rand_switch", "test_ba")
    sig = main_df[main_df["CI wyklucza 0"] == "tak"]
    sig_b = main_df[main_df["po korekcie Bonferroniego"] == "tak"]
    stops = extra.groupby(["variant"]).stop_decision.value_counts().unstack(0).fillna(0).astype(int)
    stop_tab = table(stops.reset_index().rename(columns={"stop_decision": "powód zakończenia"}))
    complete = {(d, v): int(((runs.dataset == d) & (runs.variant == v)).sum()) for d in runs.dataset.unique() for v in VARIANTS}
    ok_cells = all(n == len(SPLITS) * len(SEEDS) for n in complete.values())
    viol = (inv["v_entry_bad"] + inv["back_to_t_bad"] + inv["prev_phase_bad"] + inv["k_and_v_same_step"] + inv["k_change_in_v_step"]
            + inv["v_changed_in_t_step"] + inv["v_step_not_one_expansion"] + inv["perfect_increases_k_own"] + inv["over_cap"]
            + inv["frac_bad"] + inv["dup_retry_per_k"])

    md = f"""# followup_phase_switch v2 — podziały dodatkowe 2027–2029

Raport generowany przez `scripts/followup_phase_switch/report_splits.py`. Uzupełnia `RAPORT_PELNE_V2.md` (podział bazowy 2026, 20 seedów).
Zakres nowych przebiegów: podziały 2027, 2028, 2029 × {runs.dataset.nunique()} zbiorów zależnych od podziału ({', '.join(sorted(runs.dataset.unique()))}) × seedy 42–46 ×
3 warianty (`evo_switch`, `rand_replay`, `rand_switch`) = **{n_new} przebiegów**; ustawienia domyślne automatu, parametry SVM (C, γ) osobne dla każdego podziału
(jak w `followup_stop_rule`). Szachownica nie bierze udziału (stały plik `data/raw`, nie zależy od podziału). Analizowany układ zrównoważony: 4 podziały (2026 z serii
`v2_full`, tylko seedy 42–46) × 5 seedów = 20 obserwacji na zbiór i wariant ({'komplet' if ok_cells else 'UWAGA: brakuje przebiegów'}).
`pytest`: {args.pytest}. Poprzednie wyniki (v1, smoke, `v2_full`) i stary tryb są nietknięte.

BA na teście i na pełnym V to wyłącznie obserwacje. Trudność przyrostu ΔV′ i jakość modelu są osobnymi właściwościami; raport nie wnioskuje z jednej o drugiej.

## 1. Wyniki per zbiór i podział

{per_split_table(runs)}

## 2. Ewolucja a losowanie przy tym samym harmonogramie (evo_switch − rand_replay), 4 podziały × 5 seedów

Różnica sparowana po parach (podział, seed), n = 20 na zbiór; 95% CI z bootstrapu. Obserwacje w obrębie jednego podziału nie są w pełni niezależne,
więc przedziały są orientacyjne. Korekta Bonferroniego na {runs.dataset.nunique()} zbiorów (osobno dla BA test).

BA test:

{table(main_df)}

BA pełnego V (obserwacja):

{table(full_df)}

CI różnicy BA test wyklucza zero w {len(sig)} z {len(main_df)} zbiorów{': ' + ', '.join(sig['zbiór']) if len(sig) else ''}; po korekcie Bonferroniego w {len(sig_b)}{': ' + ', '.join(sig_b['zbiór']) if len(sig_b) else ''}.

Pomocniczo, evo_switch − rand_switch (różne harmonogramy, więc różnica miesza efekt doboru z efektem rozmiaru V′):

{table(aux_df)}

## 3. Wpływ podziału danych na poziom BA

Odch. std. BA test między średnimi podziałów (4 wartości) wobec średniego odch. std. po seedach w obrębie podziału:

{variance_table(runs)}

## 4. Stary algorytm (evo_mem) a nowy (evo_switch), te same podziały i seedy

Porównanie opisowe (różne automaty faz i polityka rozszerzania V′).

{old_vs_new(runs)}

## 5. Powody zakończenia (nowe {n_new} przebiegów)

{stop_tab}

## 6. Niezmienniki na wszystkich {inv['steps']} cyklach nowych przebiegów

| sprawdzenie | wynik |
|---|---|
| cykle V′ bez poprzedzającej decyzji przełączenia (lub kontynuacji fazy V′) | {inv['v_entry_bad']} |
| faza V′ zakończona decyzją „do T′”, a następny cykl nie jest T′ | {inv['back_to_t_bad']} |
| `prev_phase` niezgodne z poprzednim cyklem | {inv['prev_phase_bad']} |
| jednoczesne zwiększenie K i rozszerzenie V′ w jednym cyklu | {inv['k_and_v_same_step']} |
| zmiana K w cyklu V′ / zmiana V′ w cyklu T′ / V′ urosło inaczej niż o jedno ΔV′ | {inv['k_change_in_v_step']} / {inv['v_changed_in_t_step']} / {inv['v_step_not_one_expansion']} |
| BA(V′)=1 i wzrost K w tym samym cyklu (warianty z własnym automatem) | {inv['perfect_increases_k_own']} |
| \\|V′\\| powyżej limitu / `vprime_fraction` ≠ \\|V′\\|/\\|V\\| | {inv['over_cap']} / {inv['frac_bad']} |
| więcej niż jedna próba V′ po zastoju przy tym samym K | {inv['dup_retry_per_k']} |
| przebiegi zakończone bezpiecznikiem `stop_max_cycles` | {inv['max_cycles_stops']} |
| pary `rand_replay` z identycznym końcowym \\|V′\\| co `evo_switch` | {inv['replay_equal_size']} / {inv['replay_pairs']} |

Zachowanie zgodne ze specyfikacją na wszystkich cyklach: {'TAK' if viol == 0 and inv['max_cycles_stops'] == 0 else 'NIE — patrz tabela'}.

## 7. Ograniczenia

* Tylko 5 seedów na podział; różnice rzędu 0,001 BA są poniżej rozdzielczości tego układu.
* Szachownica nie była badana na podziałach dodatkowych (jej dane nie zależą od podziału).
* Wartości domyślne automatu nie były strojone; bramka ρ_SV jest włączona (`RAPORT_V2.md`).
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"zapisano {OUT} ({len(md)} znaków)")


if __name__ == "__main__":
    main()
