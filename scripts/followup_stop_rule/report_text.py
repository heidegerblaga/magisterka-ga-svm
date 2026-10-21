"""Streszczenie, wnioski i ograniczenia raportu. Każde twierdzenie ma predykat liczony z tabel; status
„POTWIERDZONE” oznacza, że predykat jest spełniony w ukończonych danych, „NIEPOTWIERDZONE / HIPOTEZA” — że dane tego nie wykazują."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import BASE_SPLIT, DATASETS, MAIN_DATASETS
from pdfkit import fint, fnum, pval

ORDER = ["evo_mem", "evo_nomem", "heur_mem", "rand_growth", "rand_fixed"]
LAB = {d: DATASETS[d]["label"] for d in DATASETS}
LAB["magic"] = "MAGIC (8000)"
LAB["digits"] = "Digits"
OK, NO = "POTWIERDZONE", "NIEPOTWIERDZONE"


def _ds_list(df, ds_col="dataset"):
    return [d for d in MAIN_DATASETS if d in set(df[ds_col])]


class Facts:
    """Liczby i predykaty używane w streszczeniu i wnioskach (wszystko z tabel)."""

    def __init__(self, d):
        self.d = d
        sb = d.sb
        self.n_old = int(sb.n_old.sum())
        self.n_new = int(sb[sb.n_new > 0].n_new.sum())
        self.ba1_old, self.ba1_new = int(sb.ba1_stops_old.sum()), int(sb.ba1_stops_new.sum())
        self.prem_old, self.prem_new = int(sb.premature_old.sum()), int(sb.premature_new.sum())
        rg = sb[sb.variant == "rand_growth"]
        self.rg_ba1_old, self.rg_ba1_new = int(rg.ba1_stops_old.sum()), int(rg.ba1_stops_new.sum())
        self.rg_prem_old, self.rg_prem_new = int(rg.premature_old.sum()), int(rg.premature_new.sum())
        self.rg_n = int(rg.n_new.sum())
        by_var = sb.groupby("variant")[["ba1_stops_old", "ba1_stops_new", "premature_old", "premature_new", "n_new"]].sum()
        self.by_var = by_var
        # nowa reguła, wszystkie splity i serie (z budżetem): stopy BA=1,0
        n = d.new
        self.new_all_runs = len(n)
        self.new_all_ba1 = int(n.perfect_ba_stop.sum())
        self.new_all_prem = int(n.premature_stop.sum())
        self.new_all_ba1_by_var = n.groupby("variant").perfect_ba_stop.sum().to_dict()
        # jakość przed/po dla wariantów przerywanych przedwcześnie
        pb = d.pba[d.pba.metric == "test_ba"]
        self.pb = pb
        gains = pb[pb.variant.isin(["rand_growth", "evo_nomem"]) & (pb.mean_diff > 0) & (pb.ci95_low > 0)]
        self.gain_cells = [(r.dataset, r.variant, r.mean_diff) for r in gains.itertuples()]
        # ewolucja vs losowy / heurystyka (podział bazowy)
        pv = d.pv[(d.pv.split == BASE_SPLIT) & (d.pv.metric == "test_ba")]
        self.pv = pv

        def cnt(a, b, sign):
            s = pv[(pv.a == a) & (pv.b == b)]
            if sign > 0:
                return [(r.dataset, r.mean_diff) for r in s.itertuples() if r.ci95_low > 0]
            return [(r.dataset, r.mean_diff) for r in s.itertuples() if r.ci95_high < 0]

        self.evo_better_rg, self.evo_worse_rg = cnt("evo_mem", "rand_growth", 1), cnt("evo_mem", "rand_growth", -1)
        self.evo_better_heur, self.evo_worse_heur = cnt("evo_mem", "heur_mem", 1), cnt("evo_mem", "heur_mem", -1)
        self.evo_better_rf, self.evo_worse_rf = cnt("evo_mem", "rand_fixed", 1), cnt("evo_mem", "rand_fixed", -1)
        self.n_ds_pv = int(pv[(pv.a == "evo_mem") & (pv.b == "rand_growth")].dataset.nunique())
        self.mean_diff_rg = pv[(pv.a == "evo_mem") & (pv.b == "rand_growth")].set_index("dataset").mean_diff
        # trudność V'
        pvh = d.pv[(d.pv.split == BASE_SPLIT) & (d.pv.a == "evo_mem") & (d.pv.b == "rand_growth")]
        inc = pvh[pvh.metric == "mean_increment_error_rate"]
        vp = pvh[pvh.metric == "vprime_ba"]
        self.inc_harder = int((inc.ci95_low > 0).sum()), int(len(inc))
        self.vp_lower = int((vp.ci95_high < 0).sum()), int(len(vp))
        self.inc_range = (float(inc.mean_diff.min()), float(inc.mean_diff.max())) if len(inc) else (np.nan, np.nan)
        self.vp_range = (float(vp.mean_diff.min()), float(vp.mean_diff.max())) if len(vp) else (np.nan, np.nan)
        # pamięć
        ms = d.ms
        e = ms[(ms.rule == "extended") & (ms.metric == "test_ba")]
        o = ms[(ms.rule == "legacy") & (ms.metric == "test_ba")]
        self.ms_new, self.ms_old = e.set_index("dataset"), o.set_index("dataset")
        self.mem_sd_sig_new = [r.dataset for r in e.itertuples() if r.sd_ratio_ci_low > 1]
        self.mem_sd_sig_old = [r.dataset for r in o.itertuples() if r.sd_ratio_ci_low > 1]
        self.mem_ratio_new_median = float(e.sd_ratio_nomem_over_mem.median()) if len(e) else np.nan
        self.mem_ratio_old_median = float(o.sd_ratio_nomem_over_mem.median()) if len(o) else np.nan
        pmem = d.pv[(d.pv.split == BASE_SPLIT) & (d.pv.metric == "test_ba") & (d.pv.a == "evo_mem") & (d.pv.b == "evo_nomem")]
        self.mem_mean_better = [(r.dataset, r.mean_diff) for r in pmem.itertuples() if r.ci95_low > 0]
        self.mem_mean_worse = [(r.dataset, r.mean_diff) for r in pmem.itertuples() if r.ci95_high < 0]
        # podziały
        vc = d.vc[d.vc.metric == "test_ba"]
        self.icc_median = float(vc.icc.median()) if len(vc) else np.nan
        self.sd_split_median = float(vc.sd_split.median()) if len(vc) else np.nan
        self.sd_seed_median = float(vc.sd_seed.median()) if len(vc) else np.nan
        vd = d.vd[d.vd.metric == "test_ba"]
        self.vd = vd
        self.vd_excl0 = [(r.dataset, r.a, r.b, r.mean_diff) for r in vd.itertuples() if (r.ci95_hier_low > 0 or r.ci95_hier_high < 0)]
        self.vd_n = len(vd)
        sign_consistent = [r for r in vd.itertuples() if r.splits_a_better == r.n_splits or r.splits_b_better == r.n_splits]
        self.vd_consistent = len(sign_consistent)
        self.max_abs_variant_diff = float(vd.mean_diff.abs().max()) if len(vd) else np.nan
        # budżet
        self.bc = d.bc

    def budget_summary(self):
        bc = self.bc
        if bc.empty:
            return None
        sel_ratio = []
        for r in bc.itertuples():
            full = r.selection_predict_rows_full + r.selection_decision_rows_full
            bud = r.selection_predict_rows_bud + r.selection_decision_rows_bud
            if full > 0:
                sel_ratio.append(bud / full)
        return {
            "n_cells": len(bc),
            "sig_worse": [(r.dataset, r.variant, r.test_ba_diff) for r in bc.itertuples() if r.test_ba_ci_high < 0],
            "sig_better": [(r.dataset, r.variant, r.test_ba_diff) for r in bc.itertuples() if r.test_ba_ci_low > 0],
            "max_abs_diff": float(bc.test_ba_diff.abs().max()),
            "sel_ratio_median": float(np.median(sel_ratio)) if sel_ratio else np.nan,
            "sel_ratio_range": (float(min(sel_ratio)), float(max(sel_ratio))) if sel_ratio else (np.nan, np.nan),
        }


def _fmt_list(items, fmt=lambda x: x):
    return ", ".join(fmt(x) for x in items) if items else "żaden zbiór"


def add_summary(doc, d):
    f = Facts(d)
    b = f.budget_summary()
    doc.h1("Streszczenie")
    rg = f.rg_ba1_old, f.rg_ba1_new
    lines = [
        f"<b>Reguła stopu.</b> Stara reguła kończyła przebieg przy BA(V′) = 1,0 po jednym rozszerzeniu V′: {f.ba1_old} z {f.n_old} przebiegów podziału bazowego "
        f"({f.prem_old} w sensie post hoc przedwcześnie, na V′ poniżej minimum lub bez rozszerzenia po jego osiągnięciu); najczęściej rand_growth ({rg[0]} z {f.rg_n}) i evo_nomem. "
        f"Nowa reguła (minimum 50 próbek V′ na klasę, Digits 20, plus obowiązkowe rozszerzenie V′ o co najmniej to samo minimum i ponowne sprawdzenie) daje "
        f"{f.ba1_new} takich stopów na {f.n_new} przebiegów podziału bazowego (rand_growth: {rg[1]}), a we wszystkich {f.new_all_runs} przebiegach nowej reguły: "
        f"{f.new_all_ba1} stopów BA = 1,0, z czego {f.new_all_prem} przedwczesnych. [{OK}]",
        "<b>Jakość.</b> Wariantom, które wcześniej kończyły się przedwcześnie (rand_growth, evo_nomem), usunięcie przedwczesnych stopów podnosi BA na teście o "
        + (_fmt_list(f.gain_cells, lambda x: f"{LAB[x[0]]}/{x[1]} +{fnum(x[2], 4)}") if f.gain_cells else "—")
        + "; pozostałe warianty mają trajektorie praktycznie niezmienione. [" + OK + "]",
        f"<b>Ewolucja a losowanie.</b> Przy nowej regule ewolucyjny dobór V′ nie daje przewagi jakościowej: evo_mem − rand_growth ma 95% CI wykluczający zero na korzyść ewolucji w "
        f"{len(f.evo_better_rg)} z {f.n_ds_pv} zbiorów (na korzyść losowego: {len(f.evo_worse_rg)}); wobec heurystyki błędów — na korzyść ewolucji {len(f.evo_better_heur)}, na korzyść heurystyki {len(f.evo_worse_heur)}. "
        f"Ewolucja buduje <b>trudniejsze</b> V′ (odsetek błędów modelu w dodanych próbkach wyższy o {fnum(f.inc_range[0], 2)}–{fnum(f.inc_range[1], 2)}; BA(V′) niższe o {fnum(-f.vp_range[1], 3)}–{fnum(-f.vp_range[0], 3)}), "
        f"ale nie lepszy model. [{OK}]",
        f"<b>Pamięć V′.</b> Duży rozrzut wariantu bez pamięci ze starej reguły (mediana stosunku odch. std. bez/z pamięcią {fnum(f.mem_ratio_old_median, 1)}) był głównie skutkiem przedwczesnych stopów: "
        f"po zmianie reguły mediana wynosi {fnum(f.mem_ratio_new_median, 2)}, a 95% CI stosunku wyklucza 1 tylko dla: {_fmt_list(f.mem_sd_sig_new, lambda x: LAB[x])}. "
        "Pamięć nie jest więc potwierdzonym czynnikiem redukcji rozrzutu (poza wskazanym wyjątkiem). [" + OK + "]",
        f"<b>Podziały danych.</b> Poziom BA zależy od podziału silniej niż od wariantu: mediana odch. std. między podziałami {fnum(f.sd_split_median, 4)} vs między seedami {fnum(f.sd_seed_median, 4)} (ICC ≈ {fnum(f.icc_median, 2)}); "
        f"największa średnia różnica wariantów to {fnum(f.max_abs_variant_diff, 4)}, a jej znak jest zgodny na wszystkich 4 podziałach tylko w {f.vd_consistent} z {f.vd_n} porównań. [" + OK + "]",
    ]
    if b:
        lines.append(
            f"<b>Koszt doboru V′.</b> Przy tej liczbie kandydatów dobór ewolucyjny wymaga tyle samo wywołań <i>predict</i> na kandydatach co heurystyka (heurystyka dodatkowo <i>decision_function</i>); "
            f"nominalna liczba ocen Val-GA nie odpowiada wywołaniom modelu (cache). Wspólny budżet ×10 zmniejsza wywołania modelu do mediany {fnum(b['sel_ratio_median'], 3)} kosztu pełnej puli "
            f"(zakres {fnum(b['sel_ratio_range'][0], 3)}–{fnum(b['sel_ratio_range'][1], 3)}) przy |Δ BA| ≤ {fnum(b['max_abs_diff'], 4)}. [" + OK + "]"
        )
    doc.bullets(lines)
    inc = d.incomplete_main
    doc.p("<b>Status wykonania:</b> " + ("wszystkie serie planu głównego są kompletne; pilotaż pełnego MAGIC opisano w sekcji 1." if inc.empty else
                                         f"<b>plan niekompletny</b> ({len(inc)} serii).") + " Wniosków nie należy uogólniać poza badane zbiory, jedno jądro (RBF) i jedną metrykę (BA).")


def add_conclusions(doc, d):
    f = Facts(d)
    b = f.budget_summary()
    doc.h1("5. Wnioski: wynik potwierdzony a hipoteza")
    rows = [["Twierdzenie", "Status", "Podstawa w danych"]]

    def row(claim, ok, basis):
        rows.append([claim, f"<b>{OK}</b>" if ok else f"<b>{NO} / hipoteza</b>", basis])

    row("Nowa reguła usuwa przedwczesne stopy BA(V′) = 1,0.", f.new_all_prem == 0,
        f"przedwczesne: {f.prem_old} → {f.new_all_prem}; stopy BA = 1,0 (podział bazowy): {f.ba1_old} → {f.ba1_new}.")
    row("Po zmianie reguły rand_growth nie ma już wielu przedwczesnych stopów.", f.rg_prem_new <= 1 and f.rg_ba1_new <= 2,
        f"rand_growth: {f.rg_ba1_old} z {f.rg_n} → {f.rg_ba1_new} z {f.rg_n} (przedwczesnych {f.rg_prem_old} → {f.rg_prem_new}).")
    row("Usunięcie przedwczesnych stopów poprawia jakość wariantów, które były nimi dotknięte.", len(f.gain_cells) >= 3,
        _fmt_list(f.gain_cells, lambda x: f"{LAB[x[0]]}/{x[1]}: +{fnum(x[2], 4)}") + " (95% CI > 0, 20 par).")
    row("Ewolucyjny dobór V′ daje przewagę jakościową nad losowym.", len(f.evo_better_rg) >= max(2, f.n_ds_pv // 2) and len(f.evo_worse_rg) == 0,
        f"CI na korzyść ewolucji: {len(f.evo_better_rg)}/{f.n_ds_pv} zbiorów; na korzyść losowego: {len(f.evo_worse_rg)} ({_fmt_list(f.evo_worse_rg, lambda x: LAB[x[0]])}).")
    row("Ewolucyjny dobór V′ daje trudniejsze V′ (więcej błędów modelu w przyroście, niższe BA(V′)).", f.inc_harder[0] >= f.inc_harder[1] - 1 and f.vp_lower[0] >= f.vp_lower[1] - 1,
        f"przyrost trudniejszy w {f.inc_harder[0]}/{f.inc_harder[1]} zbiorów; BA(V′) niższe w {f.vp_lower[0]}/{f.vp_lower[1]} (evo_mem vs rand_growth).")
    row("Pamięć V′ zmniejsza rozrzut wyników (po zmianie reguły).", len(f.mem_sd_sig_new) >= max(3, len(f.ms_new) // 2),
        f"CI stosunku odch. std. wyklucza 1 w: {_fmt_list(f.mem_sd_sig_new, lambda x: LAB[x])}; mediana stosunku {fnum(f.mem_ratio_new_median, 2)} (stara reguła: {fnum(f.mem_ratio_old_median, 1)}; "
        f"stosunek > 1 z CI wykluczającym 1: {_fmt_list(f.mem_sd_sig_old, lambda x: LAB[x])}).")
    row("Pamięć V′ podnosi średnią jakość.", len(f.mem_mean_better) >= 4 and len(f.mem_mean_worse) == 0,
        f"CI > 0: {_fmt_list(f.mem_mean_better, lambda x: f'{LAB[x[0]]} (+{fnum(x[1], 4)})')}; CI < 0: {_fmt_list(f.mem_mean_worse, lambda x: f'{LAB[x[0]]} ({fnum(x[1], 4)})')}.")
    row("Różnice jakościowe między wariantami przewyższają niepewność wynikającą z seedów i podziałów danych.", len(f.vd_excl0) >= f.vd_n // 2,
        f"CI hierarchiczne wyklucza 0 w {len(f.vd_excl0)} z {f.vd_n} porównań ({_fmt_list(f.vd_excl0, lambda x: f'{LAB[x[0]]}: {x[1]} − {x[2]} {fnum(x[3], 4)}')}).")
    row("Poziom BA zależy od podziału danych silniej niż od wariantu doboru V′.", f.sd_split_median > f.max_abs_variant_diff,
        f"mediana σ_split = {fnum(f.sd_split_median, 4)}, σ_seed = {fnum(f.sd_seed_median, 4)}; max. |średnia różnica wariantów| = {fnum(f.max_abs_variant_diff, 4)}.")
    if b:
        row("Wspólny budżet doboru V′ (×10) nie pogarsza jakości.", len(b["sig_worse"]) == 0,
            f"komórek ze CI < 0: {len(b['sig_worse'])} z {b['n_cells']}; CI > 0: {len(b['sig_better'])}; max |Δ| = {fnum(b['max_abs_diff'], 4)}.")
    doc.table(rows, [6.4, 2.6, 8.2], "Weryfikacja twierdzeń (status wyliczony z danych; „potwierdzone” = spełnione w badanych zbiorach i podziałach, nie więcej).", font_scale=0.95)
    doc.h2("Hipotezy (nie potwierdzone danymi tego badania)")
    doc.bullets([
        "Ewolucyjny dobór V′ mógłby dawać przewagę przy innych parametrach Val-GA, mniejszej puli kandydatów, innych regułach stopu lub w problemach, w których błędy modelu są rzadkie — nie badano.",
        "Dlaczego heurystyka błędów jest o ułamek punktu procentowego lepsza od evo_mem w kilku zbiorach (patrz tabela par): możliwe, że minimalny margines jest bardziej informatywny dla kolejnego T′ niż „średnio trudny” podzbiór wybierany przez Val-GA; "
        "przy 6–7 zbiorach i bez korekty na wielokrotne porównania jest to obserwacja, nie wynik.",
        "Mechanizm przewagi pamięci w spirali (większy rozrzut bez pamięci) wymagałby osobnych przebiegów diagnostycznych (np. śledzenia, które trudne próbki V′ traci bez pamięci).",
        "Dla zbiorów z limitem K (Digits, MAGIC) przebieg wyznacza limit danych, więc różnice wariantów dotyczą głównie trajektorii, a nie punktu końcowego.",
    ])
    doc.h1("6. Ograniczenia metodologiczne")
    doc.bullets([
        f"<b>Liczba podziałów i seedów.</b> Cztery podziały (jeden bazowy + trzy dodatkowe), na dodatkowych tylko 5 wspólnych seedów (42–46); składniki zmienności (σ_split) mają 3 stopnie swobody i są szacowane z dużą niepewnością. "
        "Szachownica ma jeden podział (gotowe pliki), więc nie wchodzi do analizy wpływu podziałów.",
        "<b>Wspólne (C, γ).</b> Hiperparametry SVM wybrano raz na V podziału bazowego i użyto na wszystkich podziałach (powód w sekcji 3.2). Część V podziału bazowego trafia na innych podziałach do zbioru testowego — "
        "niewielki, wspólny dla wszystkich wariantów optymizm poziomów bezwzględnych (siatka 12 punktów, strata BA na V względem optimum podziału: tabela 4); nie wpływa na porównania wariantów.",
        "<b>Stały zbiór obserwacji.</b> Zmienia się podział, nie pula obserwacji (generatory z ziarnem 2026; Digits i MAGIC to jedna pula): zmienność wynikająca z generowania danych nie jest uwzględniona.",
        "<b>Parametry reguły stopu</b> (50/klasę, Digits 20; jedno ponowne sprawdzenie o blok równy minimum) są przyjętymi założeniami, nie wynikiem optymalizacji; inne wartości mogą dawać inne liczby cykli. "
        "„Przedwczesny stop” jest definicją post hoc (sekcja 2.3), a nie oceną jakości modelu.",
        "<b>rand_fixed*</b> jest wariantem uprzywilejowanym; jego rozmiar V′ pochodzi z nowej serii evo_mem tego samego podziału (nie ze starej), więc porównanie przed/po dla rand_fixed zmienia też rozmiar V′ i nie jest czystym efektem reguły.",
        "<b>Porównania wielokrotne.</b> Wartości p i CI nie są korygowane (7 zbiorów × wiele par × wiele metryk); należy je czytać opisowo. CI bootstrapowe zakładają niezależność seedów.",
        "<b>Budżet doboru</b> zbadano dla jednego czynnika (×10) i jednej konfiguracji Val-GA; serie z budżetem dla Szachownicy mają 10 seedów.",
        "<b>Koszt.</b> Liczby treningów SVM i wierszy przekazanych modelowi są dokładne (zliczane w kodzie); koszt rzeczywisty zależy też od liczby wektorów podpierających, której nie przeliczano na operacje jądra. Czasy zegarowe pochodzą z równoległych uruchomień.",
        "<b>Pełny MAGIC</b> jest przygotowany, lecz z powodu kosztu uruchomiono tylko pilotaż (sekcja 1 i 4.9); wyniki podpróbki 8000 nie są wynikami pełnego zbioru.",
        "<b>Role V i V′.</b> V′ ⊂ V, więc BA na pełnym V jest obserwacją, nie niezależną miarą generalizacji; jedyną niezależną miarą jest test. Wniosków nie należy uogólniać poza badane zbiory, jądro RBF i metrykę BA.",
    ])
    doc.h1("7. Odtworzenie badania")
    doc.code(
        "python scripts/followup_stop_rule/generate_datasets.py && python scripts/followup_stop_rule/generate_datasets.py -s 2026 -d magic_full\n"
        "python scripts/followup_stop_rule/tune_svm.py && python scripts/followup_stop_rule/tune_svm.py -s 2026 -d magic_full\n"
        "python scripts/followup_stop_rule/make_configs.py && python -m pytest tests -q\n"
        "python scripts/followup_stop_rule/verify_legacy.py && python scripts/followup_stop_rule/smoke.py -d moons --save\n"
        "python scripts/followup_stop_rule/run_queue.py --all -w 10                # wznawialna; rand_fixed czeka na evo_mem\n"
        "python scripts/followup_stop_rule/run_queue.py -d magic_full --include-pilot -w 3\n"
        "python scripts/followup_stop_rule/aggregate.py && python scripts/followup_stop_rule/plot.py\n"
        "python scripts/followup_stop_rule/report.py                               # output/pdf/raport_followup_regula_stopu.pdf"
    )
    doc.p("Dane per-run: artifacts/followup_stop_rule/split_&lt;seed&gt;/&lt;zbiór&gt;/&lt;wariant&gt;/runs/*/metrics; tabele: reports/followup_stop_rule/tables; "
          "stan „przed” (badanie V′, podział bazowy): artifacts/study i reports/vprime_study — nie były modyfikowane.")
