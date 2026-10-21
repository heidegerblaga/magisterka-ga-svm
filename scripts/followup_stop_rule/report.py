"""Generator jedynego raportu PDF badania `followup_stop_rule` (po polsku).

Wszystkie liczby pochodzą z tabel reports/followup_stop_rule/tables/*.csv (zbudowanych z metryk per-run przez aggregate.py),
a rysunki z reports/followup_stop_rule/figures/*.png (plot.py). Raport KOŃCOWY powstaje tylko wtedy, gdy wszystkie serie planu
głównego są kompletne; w przeciwnym razie generator przerywa pracę (można wymusić raport roboczy flagą --allow-incomplete,
który jest wtedy wyraźnie oznaczony jako NIEKOMPLETNY).

    python scripts/followup_stop_rule/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALGO_SEEDS_BASE, ALGO_SEEDS_EXTRA, BASE_SPLIT, BUDGET_FACTOR, BUDGET_TWIN, DATASETS, D2, FIGS, HD, MAIN_DATASETS,
    OUT_PDF, SPLIT_SEEDS, TABLES, VARIANT_LABELS, stop_rule_params,
)
from pdfkit import Doc, fint, fnum, pm, pval  # noqa: E402

ORDER = ["evo_mem", "evo_nomem", "heur_mem", "rand_growth", "rand_fixed"]
BUD = ["evo_mem_bud", "heur_mem_bud", "rand_growth_bud"]
SHORT = {"evo_mem": "evo_mem", "evo_nomem": "evo_nomem", "heur_mem": "heur_mem", "rand_growth": "rand_growth",
         "rand_fixed": "rand_fixed*", "evo_mem_bud": "evo_mem_bud", "heur_mem_bud": "heur_mem_bud", "rand_growth_bud": "rand_growth_bud"}
LAB = {d: DATASETS[d]["label"] for d in DATASETS}
LAB["magic"] = "MAGIC (8000)"
LAB["digits"] = "Digits"
CAT = {"perfect_ba": "BA(V′)=1,0", "no_k_growth_signal": "brak sygnału wzrostu K", "k_limit": "limit K",
       "no_candidates": "brak kandydatów", "max_cycles": "limit cykli"}


class Data:
    def __init__(self):
        t = lambda n: pd.read_csv(TABLES / n)
        self.new, self.old = t("per_run.csv"), t("per_run_legacy.csv")
        self.status = t("series_status.csv")
        self.sb = t("stop_before_after.csv")
        self.pba = t("paired_before_after.csv")
        self.pv = t("paired_variants.csv")
        self.pvl = t("paired_variants_legacy.csv")
        self.vc = t("variance_components.csv")
        self.vd = t("variant_diff_across_splits.csv")
        self.ms = t("memory_spread.csv")
        self.bc = t("budget_comparison.csv")
        self.inv = t("invariants.csv")
        self.cons = t("config_consistency.csv")
        self.svm = t("svm_params_by_split.csv")
        self.splits = t("split_datasets.csv")
        self.legacy_repro = t("legacy_reproduction.csv") if (TABLES / "legacy_reproduction.csv").exists() else None
        self.smoke = json.loads((TABLES / "smoke_results.json").read_text()) if (TABLES / "smoke_results.json").exists() else None
        self.base = self.new[self.new.split == BASE_SPLIT]

    # --- dostęp ---
    def grp(self, dataset, variant, split=BASE_SPLIT):
        return self.new[(self.new.dataset == dataset) & (self.new.variant == variant) & (self.new.split == split)]

    def old_grp(self, dataset, variant):
        return self.old[(self.old.dataset == dataset) & (self.old.variant == variant)]

    def mean(self, dataset, variant, metric, split=BASE_SPLIT):
        g = self.grp(dataset, variant, split)
        return float(g[metric].mean()) if len(g) else np.nan

    def sd(self, dataset, variant, metric, split=BASE_SPLIT):
        g = self.grp(dataset, variant, split)
        return float(g[metric].std(ddof=1)) if len(g) > 1 else np.nan

    def pair(self, dataset, a, b, metric, split=BASE_SPLIT):
        r = self.pv[(self.pv.dataset == dataset) & (self.pv.a == a) & (self.pv.b == b) & (self.pv.metric == metric) & (self.pv.split == split)]
        return r.iloc[0] if len(r) else None

    @property
    def incomplete_main(self):
        s = self.status
        return s[(~s.complete) & (~s.pilot)]

    @property
    def pilot(self):
        return self.status[self.status.pilot]


# ------------------------------------------------------------------------------------------
# sekcje
# ------------------------------------------------------------------------------------------
def sec_title(doc, d, draft):
    doc.p("DAGASVM: reguła stopu BA(V′) = 1,0 a dobór podzbioru walidacyjnego V′", "Title2")
    sub = ("Badanie uzupełniające <i>followup_stop_rule</i> — nowa reguła stopu, 5 wariantów doboru V′ × 20 seedów, "
           "porównanie kosztu doboru V′, osobna analiza podziałów danych")
    if draft:
        sub += "<br/><b>WERSJA ROBOCZA — NIEKOMPLETNE SERIE (nie jest to raport końcowy)</b>"
    doc.p(sub, "Sub")


def sec_status(doc, d):
    st = d.status
    done = int(st[~st.pilot].completed.sum())
    planned = int(st[~st.pilot].planned.sum())
    n_series = int((~st.pilot).sum())
    n_complete = int(st[~st.pilot].complete.sum())
    rows = [["Zbiór", f"podział {BASE_SPLIT}: 5 wariantów × 20 seedów", "podziały 2027, 2028, 2029: 5 wariantów × 5 seedów (każdy)",
             "serie z budżetem ×10 (podział bazowy)"]]

    def cnt(ds, splits, variants):
        s = st[(st.dataset == ds) & st.split.isin(splits) & st.variant.isin(variants)]
        if s.empty:
            return "—"
        return f"{int(s.completed.sum())}/{int(s.planned.sum())}"

    for ds in MAIN_DATASETS:
        rows.append([LAB[ds], cnt(ds, [BASE_SPLIT], ORDER), cnt(ds, SPLIT_SEEDS[1:], ORDER), cnt(ds, [BASE_SPLIT], BUD)])
    doc.h1("1. Zakres i status wykonania")
    doc.p(
        f"Wykonano <b>{done} z {planned}</b> zaplanowanych przebiegów planu głównego ({n_complete} z {n_series} serii kompletnych). "
        "Przebieg = jeden seed algorytmu na jednym podziale danych dla jednego wariantu. Wszystkie przebiegi mają status <i>completed</i>, "
        "bez przebiegów nieudanych ani przerwanych bezpiecznikiem limitu cykli (tabela niezmienników w sekcji 3.4)."
        if not len(d.incomplete_main) else
        f"<b>Uwaga: plan główny jest niekompletny</b> — ukończono {done} z {planned} przebiegów ({n_complete} z {n_series} serii)."
    )
    doc.table(rows, [3.0, 4.6, 5.6, 4.0], "Liczba ukończonych / zaplanowanych przebiegów (ukończone = status completed i komplet seedów serii). "
              "Szachownica: gotowe pliki Train/V/Test z repozytorium — jeden podział (bez ponownego dzielenia); serie z budżetem dla Szachownicy: 10 seedów (koszt).")
    pil = d.pilot
    if len(pil):
        p_done, p_plan = int(pil.completed.sum()), int(pil.planned.sum())
        doc.p(
            f"<b>Pełny MAGIC (19 020 obserwacji; Train/V/Test = 7608/7608/3804)</b> — przygotowany jako zbiór <i>magic_full</i> (dane, (C, γ), konfiguracje). "
            f"Pilotaż (evo_mem, rand_growth, heur_mem × 3 seedy) ukończono w {p_done}/{p_plan} przebiegach; "
            + ("wyniki pilotażu są podane w sekcji 4.8, nie wchodzą do żadnych porównań z podpróbką 8000."
               if p_done else "pełny MAGIC jest zbyt kosztowny na ten etap — konfiguracja jest gotowa (<i>configs/followup_stop_rule/magic_full/</i>), "
               "ale serie nie zostały uruchomione i raport nie zawiera wyników dla tego zbioru.")
        )


def sec_rule(doc, d):
    doc.h1("2. Reguła stopu BA(V′) = 1,0: stara i nowa")
    doc.p(
        "Po każdym cyklu GA dla T′ (przy stałym K) algorytm liczy BA(V′) na aktualnym podzbiorze walidacyjnym V′ i podejmuje <b>jedną</b> decyzję: "
        "rozszerzyć V′ (K bez zmian), zwiększyć K (najwyżej raz na cykl) albo zakończyć. O decyzji decydują wyłącznie V′, T′, ρ<sub>SV</sub> i dostępność kandydatów; "
        "pełne V jest pulą kandydatów i źródłem obserwacji, a test — wyłącznie obserwacją."
    )
    doc.h2("2.1. Stara reguła (badanie V′)")
    doc.code(
        "jesli BA(V') >= 1,0:\n"
        "    jesli V' mozna rozszerzyc  I  V' nie bylo jeszcze utwardzone przy biezacym K:  rozszerz V' (K bez zmian)\n"
        "    w przeciwnym razie:  STOP (stop_perfect_on_current_val)         # BA = 1,0 NIGDY nie zwieksza K"
    )
    doc.p("Nie ma żadnego warunku na rozmiar V′: wystarczy jedno rozszerzenie (często o 1 próbkę na klasę), a potem kolejny wynik 100% na małym V′ kończy przebieg.")
    doc.h2("2.2. Nowa reguła (parametry <i>daga.min_vprime_per_class</i>, <i>daga.perfect_recheck_expansions</i>)")
    doc.code(
        "jesli BA(V') >= 1,0:\n"
        "    W1 := min po klasach |V' ∩ klasa| >= m          m = min_vprime_per_class (nie wiecej niz najmniejsza klasa w V)\n"
        "    W2 := r >= R                                    r = rozszerzenia wymuszone ta regula, wykonane PO osiagnieciu W1\n"
        "                                                    przy biezacym K;  R = perfect_recheck_expansions\n"
        "    jesli W1 i W2 (i stara regula zatrzymalaby):   STOP (BA = 1,0)\n"
        "    jesli nie W1 lub nie W2, V' mozna rozszerzyc:  rozszerz V', K bez zmian\n"
        "          nie W1: doloz brakujaca liczbe prob/klase jednym krokiem;   nie W2: doloz >= perfect_recheck_min_add_per_class/klase, r := r + 1\n"
        "    jesli nie W1 lub nie W2, brak kandydatow:      STOP (stop_no_validation_candidates), K bez zmian\n"
        "r i znacznik „utwardzenia” zeruja sie przy wzroscie K;  V' stale (rand_fixed): warunki nie dotycza (V' nie moze rosnac)."
    )
    doc.p(
        "Przy wartościach domyślnych (<i>0, 0</i>) reguła jest tożsama ze starą: tabela prawdy 256 kombinacji argumentów <i>decide_cycle</i> jest porównywana z dosłowną kopią starej funkcji (test), "
        "a <b>24 stare przebiegi</b> (Moons, Classification 2D, Circles, Digits; wszystkie strategie, z pamięcią i bez) uruchomione nowym kodem ze starymi konfiguracjami są identyczne co do każdej kolumny "
        "<i>cycles.csv</i> (39 kolumn) i końcowych metryk"
        + (f" ({int(d.legacy_repro.identical.sum())}/{len(d.legacy_repro)})." if d.legacy_repro is not None else ".")
    )
    rows = [["Zbiór", "m (min/klasę)", "minimalne |V′|", "R", "dołożenie przy ponownym sprawdzeniu", "|V′| startowe", "uzasadnienie"]]
    for ds in MAIN_DATASETS:
        p = stop_rule_params(ds)
        n_cls = 10 if ds == "digits" else 2
        why = ("V ma 69–73 próbki/klasę: 20/klasę ≈ 28% V (50 byłoby ≈ 70% V); przy 0 błędów górne 95% ograniczenie błędu klasy ≈ 13,9%"
               if ds == "digits" else "0 błędów na 50 próbkach klasy ⇒ górne 95% ograniczenie błędu klasy ≈ 5,8%; ≥ 5× większe niż V′ startowe")
        rows.append([LAB[ds], p["min_vprime_per_class"], p["min_vprime_per_class"] * n_cls, p["perfect_recheck_expansions"],
                     f"≥ {p['perfect_recheck_min_add_per_class']}/klasę", f"{DATASETS[ds]['k_start']}/klasę", why if ds in ("digits", "moons") else "jak wyżej"])
    doc.table(rows, [2.8, 1.5, 1.6, 0.7, 2.6, 1.7, 6.3],
              "Wartości parametrów w badaniu. Wartości są jawnie przyjętymi założeniami projektowymi (nie były dobierane na wynikach testowych); "
              "„dołożenie przy ponownym sprawdzeniu” = parametr <i>perfect_recheck_min_add_per_class</i> (trzeci, opcjonalny, domyślnie 0): ponowne sprawdzenie na V′ powiększonym o 1 próbkę na klasę byłoby tylko formalne.")
    doc.h2("2.3. Niezmienniki i testy")
    doc.bullets([
        "BA(V′) = 1,0 <b>nigdy</b> nie zwiększa K; K rośnie najwyżej raz na cykl (asercje w kodzie + testy na wszystkich kombinacjach flag oraz w pętli).",
        "Stop przez BA(V′) = 1,0 jest niemożliwy przy V′ mniejszym niż minimum (test: łatwy zbiór, V′ startowe 4/klasę — stara reguła kończy na |V′| = 14, nowa dopiero po osiągnięciu minimum i ponownym sprawdzeniu).",
        "Po osiągnięciu minimum wymagane jest ≥ R rozszerzeń i kolejny cykl; R = 2 przedłuża przebieg o kolejne rozszerzenie (test).",
        "Brak kandydatów: blokada nie zapętla się — przebieg kończy się jako <i>stop_no_validation_candidates</i> bez wzrostu K (test z minimum większym niż cały V).",
        "Metryka pełnego V nie steruje przebiegiem: identyczne trajektorie z włączoną, wyłączoną i „zatrutą” (losową) obserwacją, dla wszystkich strategii i nowej reguły (test).",
        "Dodatkowe informacje: <i>cycles.csv</i> i <i>summary.json</i> zawierają wymagane minimum, osiągnięcie minimum, liczbę ponownych sprawdzeń, powód blokady stopu i kategorię zakończenia "
        "(BA = 1,0 / brak sygnału wzrostu K / limit K / brak kandydatów / limit cykli) oraz rozbicie kosztu doboru V′.",
    ])
    doc.p(
        "<b>Definicja „przedwczesnego” stopu</b> (jednakowa dla przebiegów starych i nowych, liczona post hoc z zapisanych V′): stop przez BA(V′) = 1,0, gdy "
        "(a) najmniejsza klasa w V′ miała mniej niż <i>m</i> próbek albo (b) nie wykonano żadnego rozszerzenia V′ po osiągnięciu minimum. "
        "To oszacowanie z dołu tego, co nowa reguła blokuje; nie jest dowodem, że wynik był „zły” — jest to stop oparty na zbyt małej próbce V′."
    )


def sec_method(doc, d):
    doc.h1("3. Układ badania")
    rows = [["Wariant", "Dobór nowych elementów V′", "Pamięć", "Rozmiar V′", "Wywołania modelu przy wyborze"]]
    rows += [
        ["evo_mem", "ewolucyjny (Val-GA: pop. 30, 20 pokoleń; fitness = 1 − BA ustalonego modelu na podzbiorze)", "tak", "rośnie", "1 przejście predict po puli"],
        ["evo_nomem", "j.w., ale V′ regenerowane z V∖V′ (ten sam harmonogram rozmiaru)", "nie", "rośnie", "1 przejście predict po puli"],
        ["heur_mem", "błędy bieżącego modelu, potem najmniejszy |margines| (osobno w klasach)", "tak", "rośnie", "predict + decision_function po puli"],
        ["rand_growth", "losowo z V∖V′ (ten sam harmonogram rozszerzeń)", "tak", "rośnie", "brak"],
        ["<b>rand_fixed*</b>", "jedno losowe, stratyfikowane V′ o rozmiarze = średni końcowy |V′| z serii evo_mem <b>tego samego podziału i zbioru</b>", "—", "stały",
         "brak"],
    ]
    doc.h1("3.1 Warianty")
    doc.table(rows, [2.3, 8.2, 1.2, 1.5, 4.1], "Pięć wariantów głównych. <b>rand_fixed*</b> jest wariantem <b>uprzywilejowanym</b>: jego rozmiar V′ jest znany dzięki wcześniejszym przebiegom ewolucyjnym "
              "(w praktyce nie jest znany z góry); nie podlega warunkom nowej reguły (V′ nie rośnie), a jego rozmiar spełnia minimum z konstrukcji.")
    doc.p(
        f"Wszystkie warianty mają identyczne: seedy (42–61 na podziale bazowym {BASE_SPLIT}; 42–46 na podziałach dodatkowych), podziały Train/V/Test, parametry SVM (RBF, class_weight = balanced, (C, γ) wspólne dla zbioru), "
        "parametry GA dla T′ (pop. 30, ≤ 30 pokoleń, turniej 3, p<sub>c</sub> = 0,8, p<sub>m</sub> = 0,05, elitaryzm 2, stop po 3 pokoleniach bez poprawy), politykę K (K<sub>start</sub> 10, Digits 5; TSV = 0,25; Ts = 3) "
        "i <b>nową regułę stopu</b>. Zgodność konfiguracji wariantów w obrębie (podział × zbiór × seed) sprawdza agregator "
        + (f"({int(d.cons.all_identical.sum())}/{len(d.cons)} grup zgodnych)." if len(d.cons) else ".")
    )
    doc.h2("3.2 Podziały danych i seedy")
    ds_list = [x for x in MAIN_DATASETS if DATASETS[x]["split_dependent"]]
    doc.p(
        f"Pula obserwacji każdego zbioru jest stała (ziarno generatora 2026); zmienia się wyłącznie <b>podział</b> (stratyfikowany 40/40/20, skalowanie StandardScaler wyuczone tylko na Train) — "
        f"ziarna podziału {', '.join(map(str, SPLIT_SEEDS))}. Podział {BASE_SPLIT} jest bajt w bajt identyczny z <i>data/study</i> (skróty SHA-256), więc stan „przed zmianą reguły” (700 przebiegów badania V′) jest "
        "porównywalny przebieg po przebiegu. Szachownica pochodzi z gotowych plików i ma jeden podział. (C, γ) wybrano raz na podziale bazowym i użyto na wszystkich podziałach: ponowne strojenie na podziale 2029 "
        "zmieniło np. γ Classification 2D z 1 na 4 i sprowadziło koszt przebiegu o rząd wielkości, mieszając wpływ podziału z niestabilnością strojenia (diagnostyka: tabela 4)."
    )
    s = d.svm
    rows = [["Zbiór", "(C, γ) wspólne"] + [f"podział {sp}: optimum (BA na V) / strata wspólnych" for sp in SPLIT_SEEDS[1:]]]
    for ds in ds_list:
        sub = s[s.dataset == ds].set_index("split")
        if sub.empty:
            continue
        r0 = sub.loc[BASE_SPLIT]
        row = [LAB[ds], f"({fnum(r0.C_shared, 0)}; {fnum(r0.gamma_shared, 4)})"]
        for sp in SPLIT_SEEDS[1:]:
            if sp in sub.index:
                r = sub.loc[sp]
                row.append(f"({fnum(r.C_split_best, 0)}; {fnum(r.gamma_split_best, 4)}) {fnum(r.ba_split_best, 4)} / −{fnum(r.ba_loss_shared, 4)}")
            else:
                row.append("—")
        rows.append(row)
    doc.table(rows, [2.5, 2.6, 4.1, 4.1, 4.1], "Diagnostyka strojenia: optimum siatki na V każdego podziału oraz strata BA na V przy zastosowaniu wspólnych parametrów (z podziału bazowego). "
              "Małe straty oznaczają płaskie optimum — wspólne parametry są prawie optymalne na każdym podziale.")
    doc.h2("3.3 Koszt: co jest liczone")
    doc.bullets([
        "<b>Treningi SVM w GA dla T′</b> — faktyczna liczba dopasowań SVM (populacja × (1 + pokolenia) w każdym cyklu; bez cache). Osobno: wiersze V′ klasyfikowane przy ocenie osobników (Σ ocen × |V′|) — rośnie z |V′|.",
        "<b>Wiersze przekazane modelowi przy wyborze przyrostu V′</b> (faktyczne wywołania): <i>predict</i> po puli kandydatów (evo, heur) i <i>decision_function</i> po puli (tylko heur). "
        "Fitness Val-GA korzysta z predykcji zbuforowanych raz dla puli, więc <b>nominalna</b> liczba ocen podzbiorów Val-GA (30 × 21 ocen × |przyrost|) <b>nie jest</b> liczbą wywołań <i>clf.predict</i>; podajemy ją osobno jako wartość nominalną.",
        "Liczba kandydatów (|V∖V′| przy każdym rozszerzeniu), rozmiar przyrostu oraz wiersze diagnostyczne (obliczane tylko do raportu) są zapisywane osobno w <i>cycles.csv</i>.",
        f"<b>Wspólny budżet doboru</b> (serie <i>_bud</i>, podział bazowy): przy każdym rozszerzeniu strategia widzi wyłącznie losową, stratyfikowaną klasowo podpulę o rozmiarze ≤ {BUDGET_FACTOR:g} × (liczba wybieranych elementów). "
        "Podpula zależy od seedu i etykiet, nie od strategii ani modelu; obowiązuje identycznie dla ewolucji, heurystyki i losowania (dla losowania nie zmienia rozkładu wyboru, więc służy jako kontrola).",
        "Czasy zegarowe pochodzą z równoległych uruchomień (10 procesów, jedna maszyna) i nie są podstawą porównań kosztu.",
    ])


def sec_integrity(doc, d):
    doc.h2("3.4 Kontrola poprawności")
    inv = d.inv
    rows = [["Niezmiennik (liczony z cycles.csv wszystkich przebiegów nowej reguły)", "naruszenia", "sprawdzono"]]
    for _, r in inv.iterrows():
        rows.append([r.invariant, int(r.violations), int(r.checked)])
    doc.table(rows, [11.5, 2.5, 3.0], "Niezmienniki reguły stopu i polityki K. Oczekiwana liczba naruszeń: 0 (przedwczesne stopy w sensie post hoc oraz limit cykli też).")


def stop_cell(d, ds, v):
    so = d.sb[(d.sb.dataset == ds) & (d.sb.variant == v)]
    if so.empty:
        return "—"
    r = so.iloc[0]
    return f"{int(r.ba1_stops_old)}→{int(r.ba1_stops_new)}" if r.n_new else f"{int(r.ba1_stops_old)}→—"


def sec_results_stops(doc, d):
    doc.h1("4. Wyniki")
    doc.h2("4.1 Stopy przez BA(V′) = 1,0: przed i po zmianie reguły")
    rows = [["Zbiór"] + [SHORT[v] for v in ORDER]]
    tot = {v: [0, 0, 0, 0, 0] for v in ORDER}
    for ds in MAIN_DATASETS:
        row = [LAB[ds]]
        for v in ORDER:
            so = d.sb[(d.sb.dataset == ds) & (d.sb.variant == v)]
            if so.empty or not so.iloc[0].n_new:
                row.append("—")
                continue
            r = so.iloc[0]
            row.append(f"{int(r.ba1_stops_old)} ({int(r.premature_old)}) → {int(r.ba1_stops_new)} ({int(r.premature_new)})")
            t = tot[v]
            t[0] += int(r.ba1_stops_old); t[1] += int(r.premature_old); t[2] += int(r.ba1_stops_new); t[3] += int(r.premature_new); t[4] += int(r.n_new)
        rows.append(row)
    rows.append(["<b>Razem</b>"] + [f"<b>{tot[v][0]} ({tot[v][1]}) → {tot[v][2]} ({tot[v][3]})</b> z {tot[v][4]}" for v in ORDER])
    doc.table(rows, [3.0] + [2.8] * 5,
              "Liczba przebiegów zakończonych przez BA(V′) = 1,0 przy starej → nowej regule (w nawiasie: z tego przedwczesnych w sensie post hoc z sekcji 2.3). "
              "Podział bazowy, te same seedy 42–61, 20 przebiegów na komórkę.", bold_rows=(len(rows) - 1,))
    doc.figure("fig_stops_before_after.png", "Po lewej: przyczyny zakończenia przebiegów (% z 140 przebiegów na wariant; blade słupki = stara reguła, pełne = nowa). "
               "Po prawej: |V′| w chwili stopu BA(V′) = 1,0 (każdy punkt = jeden przebieg; linia przerywana = minimum 2×50).", 17.0)
    doc.figure("fig_trajectory_moons_rand_growth.png", "Przykład (Moons, rand_growth, seed 42): stara reguła kończy po dwóch cyklach na V′ ≈ 36; nowa dobija V′ do minimum, wykonuje ponowne sprawdzenie "
               "(niebieskie kółka = BA = 1,0 ze zablokowanym stopem) i dopiero potem przebieg biegnie dalej według sygnału wzrostu K.", 16.0)


def metric_table(d, metric, digits, title_col, sd_digits=None):
    rows = [["Zbiór"] + [SHORT[v] for v in ORDER]]
    for ds in MAIN_DATASETS:
        row = [LAB[ds]]
        for v in ORDER:
            g = d.grp(ds, v)
            row.append("—" if g.empty else pm(g[metric].mean(), g[metric].std(ddof=1), digits))
        rows.append(row)
    return rows


def sec_results_quality(doc, d):
    doc.h2("4.2 Jakość testowa, liczba cykli, końcowe K i |V′|")
    doc.table(metric_table(d, "test_ba", 4, "test BA"), [3.0] + [2.8] * 5,
              "BA na zbiorze testowym (obserwacja; średnia ± odch. std. po 20 seedach, nowa reguła, podział bazowy).")
    doc.figure("fig_quality_test_ba.png", "BA na teście dla każdego seedu (punkty), średnia i 95% CI (bootstrap); pusty romb = średnia przy starej regule (te same seedy).", 17.2)
    doc.table(metric_table(d, "cycles", 1, "cykle"), [3.0] + [2.8] * 5, "Liczba cykli T′ (średnia ± odch. std.).")
    doc.table(metric_table(d, "final_k", 0, "K"), [3.0] + [2.8] * 5, "Końcowe K (próbek na klasę w T′).")
    doc.table(metric_table(d, "vprime_size", 0, "V′"), [3.0] + [2.8] * 5, "Końcowy rozmiar V′ (liczba próbek).")
    doc.figure("fig_cycles.png", "Liczba cykli T′ (skala logarytmiczna).", 16.0)
    doc.figure("fig_final_k.png", "Końcowe K.", 16.0)
    doc.figure("fig_final_vprime.png", "Końcowy rozmiar V′ (skala logarytmiczna).", 16.0)


def sec_before_after_quality(doc, d):
    doc.h2("4.3 Zmiana reguły: wpływ na jakość i koszt (pary po seedach, nowa − stara)")
    rows = [["Zbiór", "wariant", "Δ test BA (95% CI)", "Δ cykli", "Δ treningów SVM w GA", "Δ |V′| końcowe", "seedy: lepiej / gorzej (test BA)"]]
    pb = d.pba
    for ds in MAIN_DATASETS:
        for v in ORDER:
            sub = pb[(pb.dataset == ds) & (pb.variant == v)].set_index("metric")
            if "test_ba" not in sub.index or pd.isna(sub.loc["test_ba"].get("mean_diff")):
                continue
            t = sub.loc["test_ba"]
            changed = int(d.sb[(d.sb.dataset == ds) & (d.sb.variant == v)].iloc[0].ba1_stops_old) > 0
            if not changed and abs(t.mean_diff) < 1e-12 and ds not in ("moons", "circles", "classification2d"):
                continue
            rows.append([LAB[ds], SHORT[v], f"{fnum(t.mean_diff, 4)} [{fnum(t.ci95_low, 4)}; {fnum(t.ci95_high, 4)}]",
                         fnum(sub.loc["cycles"].mean_diff, 1), fint(sub.loc["svm_fits_total"].mean_diff), fnum(sub.loc["vprime_size"].mean_diff, 0),
                         f"{int(t.a_better_count)} / {int(t.b_better_count)}"])
    doc.table(rows, [2.6, 2.3, 4.4, 1.5, 2.6, 1.9, 2.3],
              "Różnice nowa − stara reguła dla tych samych seedów i podziału (pominięto komórki bez żadnej zmiany: tam trajektorie są identyczne). Wartości dodatnie Δ test BA = nowa reguła lepsza.", font_scale=0.95)


def sec_results_cost(doc, d):
    doc.h2("4.4 Koszt: GA dla T′ oraz dobór przyrostu V′")
    rows = [["Zbiór"] + [SHORT[v] for v in ORDER]]
    for ds in MAIN_DATASETS:
        rows.append([LAB[ds]] + [fint(d.mean(ds, v, "svm_fits_total")) if len(d.grp(ds, v)) else "—" for v in ORDER])
    doc.table(rows, [3.0] + [2.8] * 5, "Treningi SVM w GA wybierającym T′ (średnia na przebieg; faktyczne dopasowania SVM).")
    rows = [["Zbiór", "wariant", "kandydaci łącznie (Σ|V∖V′|)", "przyrost łącznie", "wiersze predict (faktyczne)", "wiersze decision_function", "NOMINALNE wiersze Val-GA (nie wykonane)",
             "wiersze V′ w GA dla T′"]]
    for ds in MAIN_DATASETS:
        for v in ("evo_mem", "evo_nomem", "heur_mem", "rand_growth"):
            g = d.grp(ds, v)
            if g.empty:
                continue
            rows.append([LAB[ds], SHORT[v], fint(g.candidates_total.mean()), fint(g.increment_rows_total.mean()), fint(g.selection_predict_rows.mean()),
                         fint(g.selection_decision_rows.mean()), fint(g.val_ga_nominal_subset_rows.mean()) if v.startswith("evo") else "—",
                         fint(g.tga_vprime_rows_scored.mean())])
    doc.table(rows, [2.3, 2.1, 2.2, 1.9, 2.1, 2.0, 2.6, 2.4],
              "Koszt doboru przyrostu V′ na przebieg (średnie): kandydaci i przyrost, faktyczne wiersze przekazane modelowi, nominalna liczba wierszy ocenianych w Val-GA (bez cache — wartość teoretyczna) oraz, dla porównania, "
              "wiersze V′ klasyfikowane w GA dla T′. Dobór losowy nie wywołuje modelu.", font_scale=0.92)
    doc.figure("fig_cost.png", "Koszt w skali logarytmicznej: (góra) treningi SVM w GA dla T′; (środek) wiersze V′ klasyfikowane w GA dla T′; (dół) wiersze przekazane modelowi przy wyborze przyrostu V′ "
               "(krzyżyki = nominalna liczba wierszy ocenianych w Val-GA, której nie wykonano dzięki cache predykcji).", 16.5)


def sec_budget(doc, d):
    doc.h2("4.5 Wspólny budżet doboru V′ (serie _bud)")
    bc = d.bc
    if bc.empty:
        doc.p("Serie z budżetem nie zostały ukończone.")
        return
    rows = [["Zbiór", "wariant", "n", "test BA: budżet", "test BA: pełna pula", "Δ (95% CI)", "wiersze modelu: budżet / pełna pula", "treningi SVM w GA: budżet / pełna pula"]]
    for ds in MAIN_DATASETS:
        for v in BUD:
            r = bc[(bc.dataset == ds) & (bc.variant == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            sel_b = r.selection_predict_rows_bud + r.selection_decision_rows_bud
            sel_f = r.selection_predict_rows_full + r.selection_decision_rows_full
            rows.append([LAB[ds], SHORT[v], int(r.n), fnum(r.test_ba_bud, 4), fnum(r.test_ba_full, 4),
                         f"{fnum(r.test_ba_diff, 4)} [{fnum(r.test_ba_ci_low, 4)}; {fnum(r.test_ba_ci_high, 4)}]",
                         f"{fint(sel_b)} / {fint(sel_f)}" if sel_f > 0 else "0 / 0 (losowy)", f"{fint(r.svm_fits_total_bud)} / {fint(r.svm_fits_total_full)}"])
    doc.table(rows, [2.2, 2.6, 0.7, 1.8, 1.9, 3.5, 3.2, 3.2],
              f"Budżet ×{BUDGET_FACTOR:g} (podpula ≤ {BUDGET_FACTOR:g} × przyrost) wobec pełnej puli: ten sam seed, ten sam podział. Dla Szachownicy 10 seedów, dla pozostałych 20.", font_scale=0.92)
    doc.figure("fig_budget.png", "Wpływ budżetu: różnica jakości testowej, stosunek wierszy ocenianych przez model i stosunek treningów SVM w GA dla T′.", 17.0)


def sec_memory(doc, d):
    doc.h2("4.6 Pamięć V′")
    ms = d.ms
    rows = [["Zbiór", "odch. std. BA test: z pamięcią / bez (nowa)", "stosunek bez/z (95% CI)", "Brown–Forsythe p", "stara reguła: odch. std. z / bez (stosunek)", "stop BA = 1,0 (stara → nowa): z pamięcią | bez"]]
    for ds in MAIN_DATASETS:
        n = ms[(ms.rule == "extended") & (ms.dataset == ds) & (ms.metric == "test_ba")]
        o = ms[(ms.rule == "legacy") & (ms.dataset == ds) & (ms.metric == "test_ba")]
        if n.empty:
            continue
        n, o_ = n.iloc[0], (o.iloc[0] if len(o) else None)
        rows.append([LAB[ds], f"{fnum(n.sd_mem, 4)} / {fnum(n.sd_nomem, 4)}", f"{fnum(n.sd_ratio_nomem_over_mem, 2)} [{fnum(n.sd_ratio_ci_low, 2)}; {fnum(n.sd_ratio_ci_high, 2)}]",
                     pval(n.brown_forsythe_p), "—" if o_ is None else f"{fnum(o_.sd_mem, 4)} / {fnum(o_.sd_nomem, 4)} ({fnum(o_.sd_ratio_nomem_over_mem, 2)})",
                     f"{stop_cell(d, ds, 'evo_mem')} | {stop_cell(d, ds, 'evo_nomem')}"])
    doc.table(rows, [2.3, 3.4, 3.2, 1.8, 3.6, 3.0], "Rozrzut jakości testowej z pamięcią i bez pamięci V′ (20 seedów, podział bazowy; stosunek > 1 = większy rozrzut bez pamięci).", font_scale=0.95)
    doc.figure("fig_memory.png", "Odchylenie standardowe BA na teście (skala log) i jego stosunek bez/z pamięcią; ○ nowa reguła, ◇ stara reguła.", 16.5)


def sec_random_and_evo(doc, d):
    doc.h2("4.7 Warianty losowe, ewolucja a trudność V′")
    pairs = [("evo_mem", "rand_growth"), ("evo_mem", "heur_mem"), ("evo_mem", "rand_fixed"), ("evo_mem", "evo_nomem")]
    rows = [["Zbiór", "para (a vs b)", "Δ test BA (a − b) [95% CI]", "p Wilcoxon", "a lepszy / b lepszy", "Δ cykli", "Δ trudności przyrostu"]]
    for ds in MAIN_DATASETS:
        for a, b in pairs:
            p = d.pair(ds, a, b, "test_ba")
            if p is None or pd.isna(p.get("mean_diff")):
                continue
            c = d.pair(ds, a, b, "cycles")
            h = d.pair(ds, a, b, "mean_increment_error_rate")
            rows.append([LAB[ds], f"{SHORT[a]} vs {SHORT[b]}", f"{fnum(p.mean_diff, 4)} [{fnum(p.ci95_low, 4)}; {fnum(p.ci95_high, 4)}]", pval(p.wilcoxon_p),
                         f"{int(p.a_better_count)} / {int(p.b_better_count)}", fnum(c.mean_diff, 1) if c is not None else "—",
                         fnum(h.mean_diff, 3) if (h is not None and not pd.isna(h.get('mean_diff'))) else "—"])
    doc.table(rows, [2.3, 3.6, 4.0, 1.5, 2.0, 1.5, 2.3], "Porównania sparowane po seedach (podział bazowy, 20 par). Wartości p Wilcoxona bez korekty na wielokrotne porównania — opisowe.", font_scale=0.92)
    doc.figure("fig_paired_forest.png", "Różnice sparowane evo_mem − wariant: jakość testowa, liczba cykli, treningi SVM w GA dla T′ i trudność przyrostu V′ (średnia i 95% CI bootstrap).", 17.2)
    doc.figure("fig_pairs_evo_vs_randgrowth.png", "Pary po seedach: evo_mem (oś pozioma) vs rand_growth (oś pionowa), BA na teście.", 15.5)
    doc.figure("fig_hardness.png", "Trudność V′: błędy modelu w całej puli kandydatów, w dodanych elementach i BA na końcowym V′ (metryka sterująca).", 17.0)


def sec_splits(doc, d):
    doc.h2("4.8 Wpływ podziału danych vs seedów algorytmu")
    vc = d.vc[d.vc.metric == "test_ba"]
    rows = [["Zbiór", "wariant", "σ_split (podział)", "σ_seed (seedy w podziale)", "udział podziału (ICC)", "średnie po podziałach (2026 / 2027 / 2028 / 2029)"]]
    for ds in [x for x in MAIN_DATASETS if DATASETS[x]["split_dependent"]]:
        for v in ORDER:
            r = vc[(vc.dataset == ds) & (vc.variant == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            means = json.loads(r.mean_by_split)
            rows.append([LAB[ds], SHORT[v], fnum(r.sd_split, 4), fnum(r.sd_seed, 4), fnum(r.icc, 2), " / ".join(fnum(means.get(str(s)), 4) for s in SPLIT_SEEDS)])
    doc.table(rows, [2.3, 2.4, 2.2, 2.7, 2.3, 5.7], "Rozkład zmienności BA na teście (4 podziały × 5 wspólnych seedów 42–46, ANOVA z efektem losowym; seed zagnieżdżony w podziale). ICC = σ²_split / (σ²_split + σ²_seed).", font_scale=0.92)
    doc.figure("fig_splits_test_ba.png", "BA na teście na czterech podziałach: wszystkie warianty przesuwają się razem wraz z podziałem.", 16.5)
    doc.figure("fig_variance_components.png", "Składniki zmienności: między podziałami (po lewej) i między seedami w obrębie podziału (po prawej), ta sama skala osi.", 16.5)
    doc.figure("fig_variant_diff_splits.png", "Różnice wariantów w teście przy uwzględnieniu obu źródeł niepewności (bootstrap hierarchiczny).", 13.5)


def sec_pilot(doc, d):
    pil = d.pilot
    if pil.empty or not int(pil.completed.sum()):
        return
    doc.h2("4.9 Pilotaż: pełny MAGIC (19 020 obserwacji)")
    sub = d.new[d.new.dataset == "magic_full"]
    rows = [["wariant", "n seedów", "test BA", "cykle", "końcowe K", "końcowe |V′|", "stop", "czas [godz., zegarowo]"]]
    for v in ("evo_mem", "rand_growth", "heur_mem"):
        g = sub[sub.variant == v]
        if g.empty:
            continue
        rows.append([SHORT[v], len(g), pm(g.test_ba.mean(), g.test_ba.std(ddof=1) if len(g) > 1 else np.nan, 4), fnum(g.cycles.mean(), 1), fnum(g.final_k.mean(), 0),
                     fnum(g.vprime_size.mean(), 0), ", ".join(sorted(set(g.stop_category))), fnum(g.duration_sec.mean() / 3600, 2)])
    doc.table(rows, [2.4, 1.4, 3.4, 1.4, 1.8, 2.0, 2.4, 2.6], "Pilotaż pełnego MAGIC (3 seedy; bez wniosków porównawczych — za mało przebiegów).")


# ------------------------------------------------------------------------------------------
@click.command()
@click.option("--allow-incomplete", is_flag=True, help="Raport roboczy mimo niekompletnych serii (oznaczony jako NIEKOMPLETNY).")
@click.option("--out", default=None)
def main(allow_incomplete, out):
    d = Data()
    draft = len(d.incomplete_main) > 0
    if draft and not allow_incomplete:
        details = "; ".join(f"split{r.split}/{r.dataset}/{r.variant}: {r.completed}/{r.planned}" for r in d.incomplete_main.itertuples())
        raise SystemExit(f"Nie wygenerowano raportu końcowego: niekompletne serie ({len(d.incomplete_main)}): {details[:1500]}")
    out_pdf = Path(out) if out else OUT_PDF
    doc = Doc(FIGS, out_pdf, "DAGASVM — reguła stopu BA(V′)=1,0 i dobór V′ (badanie uzupełniające)")
    sec_title(doc, d, draft)
    # narracja i wnioski dołączane przez report_text (po analizie wyników)
    from report_text import add_summary, add_conclusions  # noqa: E402

    add_summary(doc, d)
    sec_status(doc, d)
    sec_rule(doc, d)
    sec_method(doc, d)
    sec_integrity(doc, d)
    sec_results_stops(doc, d)
    sec_results_quality(doc, d)
    sec_before_after_quality(doc, d)
    sec_results_cost(doc, d)
    sec_budget(doc, d)
    sec_memory(doc, d)
    sec_random_and_evo(doc, d)
    sec_splits(doc, d)
    sec_pilot(doc, d)
    add_conclusions(doc, d)
    doc.build()


if __name__ == "__main__":
    main()
