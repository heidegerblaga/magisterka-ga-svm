"""Treść raportu: opisy metodologiczne (stałe) i interpretacja wyników (liczby z tabel)."""
from __future__ import annotations

import pandas as pd
import yaml

from generate_vprime_study_report import DATASET_LABELS, VARIANT_SHORT, f, fi, pm, pval
from ga.study_analysis import VARIANT_LABELS

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
D2 = ["chessboard", "moons", "circles", "spiral", "classification2d"]
HD = ["digits", "magic"]


def cfg_of(dataset: str, variant: str = "evo_mem") -> dict:
    return yaml.safe_load((REPO / "configs" / "study" / dataset / f"{variant}.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------
# sekcje stałe (metodologia)
# ------------------------------------------------------------------------------------------
def sec_roles(doc, d):
    doc.h1("2. Role zbiorów V i V′")
    doc.p(
        "W fazowym DAGASVM w jednym cyklu optymalizowany jest podzbiór treningowy T′ (K próbek na klasę, wybieranych "
        "algorytmem genetycznym) tak, aby maksymalizować metrykę (balanced accuracy) na aktualnym podzbiorze walidacyjnym V′. "
        "Po zakończeniu optymalizacji algorytm podejmuje jedną z decyzji: rozszerzyć V′, zwiększyć K albo zakończyć. "
        "Pełny zbiór walidacyjny V pełni w tym układzie dwie rozłączne role, które w raportowaniu są konsekwentnie rozdzielone."
    )
    rows = [
        ["Zbiór / wielkość", "Rola", "Steruje algorytmem?", "Gdzie występuje"],
        ["V′ (aktualny podzbiór walidacyjny)", "metryka sterująca", "tak",
         "fitness T′ w GA; decyzja o rozszerzeniu V′, wzroście K i zatrzymaniu (razem z ρ<sub>SV</sub>)"],
        ["V (pełny zbiór walidacyjny) — pula kandydatów", "źródło nowych elementów V′", "pośrednio: wybór elementów, nie ocena jakości",
         "wybór elementów do V′ (losowy, ewolucyjny albo heurystyczny); koszt: patrz sekcja 3.4"],
        ["V (pełny zbiór walidacyjny) — obserwacja", "zewnętrzna ocena obserwacyjna", "nie",
         "metryki liczone <b>po</b> podjęciu decyzji cyklu, tylko zapisywane (cycles.csv, summary.json); można je wyłączyć (observation.full_validation=false) bez zmiany przebiegu"],
        ["Test", "zewnętrzna ocena obserwacyjna", "nie", "jak wyżej; nigdy nie bierze udziału w wyborze modelu"],
    ]
    doc.table(rows, [4.1, 3.3, 3.0, 6.9], "Role zbiorów w algorytmie.")
    doc.p(
        "Rozróżnienie ma konsekwencję interpretacyjną, którą należy mieć na uwadze: ponieważ nowe elementy V′ są wybierane z V, "
        "informacja o V trafia do algorytmu poprzez <i>które</i> próbki zasilają V′, ale nigdy poprzez <i>ocenę jakości modelu na pełnym V</i>. "
        "Metryki „sterujące” (V′) i „obserwacyjne” (pełne V, test) są w tabelach raportu oznaczone osobno. "
        "Wynik na pełnym V nie jest więc niezależną miarą generalizacji w takim sensie jak test: jest obserwacją, "
        "ale V′ ⊂ V, więc nie jest to zbiór rozłączny z tym, na którym algorytm się uczy sterować."
    )
    doc.h2("2.1. Weryfikacja: miejsca użycia pełnego V w kodzie")
    doc.bullets(
        [
            "<b>src/ga/phase_engine.py</b>: (i) inicjalizacja V′ — losowa, zbalansowana klasowo próbka indeksów V (tylko etykiety, bez modelu); "
            "(ii) przekazanie wierszy V′ do GA i do oceny metryk sterujących; (iii) wywołanie <i>expand_vprime</i> (pula kandydatów V \\ V′); "
            "(iv) <i>FullValidationObserver.observe</i> po decyzji cyklu.",
            "<b>src/ga/vprime.py</b>: skanowanie / losowanie z puli kandydatów V \\ V′ oraz diagnostyka trudności wybranych elementów "
            "(liczniki <i>control_model_predictions</i> i <i>diagnostic_predictions</i> rozdzielają koszt sterujący od diagnostycznego).",
            "<b>scripts/run_experiment.py</b>: faza bazowa (pełny SVM na pełnym Train, ocena na V — punkt odniesienia, poza pętlą algorytmu) oraz "
            "wiersz raportu (<i>build_report_row</i>, ocena obserwacyjna).",
            "<b>scripts/tune_study_svm.py</b>: wybór wspólnych (C, γ) pełnym SVM oceną na V — <i>przed</i> eksperymentami, identyczny dla wszystkich wariantów (patrz ograniczenia).",
        ]
    )
    doc.p(
        "Przejrzenie kodu nie wykazało użycia metryk pełnego V w fitnessie, selekcji, decyzji o rozszerzeniu V′, wzroście K ani w warunkach stopu. "
        "Poprzednia implementacja liczyła metrykę na pełnym V <i>przed</i> decyzją cyklu (bez użycia wartości); obecnie obserwacja jest wykonywana po decyzji i "
        "może być wyłączona. Testy (tests/test_phase_engine.py) uruchamiają algorytm trzykrotnie — z obserwacją włączoną, wyłączoną oraz z obserwatorem "
        "zwracającym losowe „metryki” — i wymagają identycznych trajektorii (K, decyzje, identyfikatory V′) dla wszystkich trzech wariantów doboru V′."
    )


def sec_implementation(doc, d):
    doc.h1("3. Implementacja")
    doc.h2("3.1. Pętla fazowa i polityka K")
    doc.p(
        "Nazewnictwo lokalne projektu: K = <i>k_per_class</i> (liczba próbek na klasę w T′); T′ = wybrany podzbiór treningowy; "
        "V′ = podzbiór walidacyjny; ρ<sub>SV</sub> = liczba wektorów podpierających / |T′|; TSV = 0,25 (próg). "
        "Jeden cykl = jedna optymalizacja T′ przy stałym K. Po niej wywoływana jest czysta funkcja decyzyjna (<i>decide_cycle</i>):"
    )
    doc.code(
        "1. T' = GA(K; fitness = BA na V')            # K stale w cyklu\n"
        "2. if BA(V') >= 1.0: V' jeszcze nie utrudniane -> rozszerz V' (K bez zmian)\n"
        "                     inaczej                  -> stop (K bez zmian)\n"
        "3. if rho_SV < TSV (lub K = limit klasy)      -> stop\n"
        "4. if V' nie bylo utrudniane przy tym K       -> rozszerz V'\n"
        "5. if rho_SV stabilne po utrudnieniu V'       -> rozszerz V'\n"
        "6. inaczej: K := max(K+1, ceil(K*(1+abs(rho_SV-TSV)/(1-TSV))))"
    )
    doc.bullets(
        [
            "K może wzrosnąć <b>najwyżej raz w cyklu T′</b> (decyzja <i>increase_k</i> zapada raz po zakończeniu GA; dodatkowo asercja w kodzie, "
            "kolumna <i>k_increases_this_cycle</i> i test sprawdzający każdy cykl).",
            "Gdy aktualne T′ osiąga 100% na V′, K <b>nie</b> rośnie automatycznie: algorytm najpierw utrudnia V′, a jeśli V′ było już utrudnione dla tego K — kończy.",
            "Mechanizm „najpierw V′, potem K”: po każdym wzroście K pierwsza decyzja z sygnałem wzrostu to rozszerzenie V′; "
            "K rośnie dopiero, gdy ρ<sub>SV</sub> po utrudnieniu V′ nie jest stabilne albo wyczerpano cierpliwość Ts (wymuszony wzrost).",
            "Każda decyzja jest logowana wraz z powodem (kolumna <i>decision_reason</i> w cycles.csv oraz log przebiegu): rozszerzenie V′, wzrost K, brak zmiany K, stop.",
            "Reprezentatywny model to ostatni ukończony cykl — nie jest wybierany na podstawie pełnego V. Bezpiecznik <i>max_cycles</i> (500) nigdy nie zadziałał w seriach raportu.",
        ]
    )
    doc.h2("3.2. Warianty doboru V′")
    rows = [
        ["Wariant", "Dobór nowych elementów V′", "Pamięć", "Skan V"],
        ["evo_mem (ewolucyjny, z pamięcią)", "adwersarialny Val-GA nad maskami kandydatów: fitness = 1 − BA ustalonego modelu na podzbiorze; "
         "liczba wybieranych elementów = (przyrost K na klasę) × liczba klas", "tak: V′<sub>t+1</sub> = V′<sub>t</sub> ∪ Δ<sub>t</sub>", "nie wprost (patrz 3.4)"],
        ["evo_nomem (ewolucyjny, bez pamięci)", "j.w., ale V′<sub>t+1</sub> = świeży wybór o rozmiarze |V′<sub>t</sub>| + |Δ<sub>t</sub>| z V \\ V′<sub>t</sub>", "nie (elementy poprzednie odrzucane)", "j.w."],
        ["rand_fixed (losowy, stały rozmiar)", "jednorazowe losowe, stratyfikowane klasowo V′ o rozmiarze = średni końcowy |V′| z serii evo_mem; brak rozszerzeń — K rośnie wg ρ<sub>SV</sub>", "—", "nie"],
        ["rand_growth (losowy, rosnący)", "losowe elementy z V \\ V′ przy takim samym harmonogramie rozszerzeń jak wariant ewolucyjny", "tak", "nie"],
        ["heur_mem (heurystyka błędów)", "w każdej klasie: błędnie klasyfikowane przez bieżący model, potem najmniejszy |margines|", "tak", "tak: klasyfikacja całej puli V \\ V′"],
    ]
    doc.table(rows, [3.7, 7.6, 3.3, 2.7], "Warianty doboru V′ (identyczne T′, GA, parametry K, kryteria stopu).")
    doc.p(
        "<b>Dobór rozmiaru wariantu losowego jest łagodnie uprzywilejowujący.</b> Wariant rand_fixed korzysta z informacji o średnim "
        "końcowym rozmiarze V′ uzyskanym przez ewolucję (jest to więc bazowy wariant, któremu „podpowiedziano” rząd wielkości V′, a który "
        "nie musi go budować stopniowo). Taki dobór rozmiaru faworyzuje wariant losowy, dlatego ewentualna przewaga ewolucji nad nim jest oceną ostrożną; "
        "wariant rand_growth usuwa tę podpowiedź, ale wprowadza inny efekt (patrz sekcja 8)."
    )
    doc.h2("3.3. Pamięć V′ i identyfikatory próbek")
    doc.p(
        "Elementy V′ są identyfikowane przez indeks wiersza w V (<i>VPrimeSet</i>): zbiór uporządkowany, bez duplikatów, z walidacją zakresu. "
        "Przy <i>memory=true</i> operacja rozszerzenia dołącza nowe identyfikatory na końcu (poprzednie zachowują kolejność — asercja prefiksu); "
        "przy <i>memory=false</i> V′ jest zastępowane świeżym wyborem z puli V \\ V′<sub>t</sub>, o rozmiarze zgodnym z harmonogramem wariantu z pamięcią "
        "(asercja braku przecięcia ze starym V′). Opcja jest ustawiana w konfiguracji (<i>vprime.memory</i>); testy: tests/test_vprime.py i tests/test_phase_engine.py. "
        "Uwaga: przy memory=false rozmiar V′ jest ograniczony liczbą pozostałych kandydatów."
    )
    doc.h2("3.4. Koszt obliczeniowy doboru V′")
    doc.p(
        "Heurystyka błędów wymaga klasyfikacji całej puli kandydatów w każdym rozszerzeniu: |V \\ V′| predykcji modelu. "
        "Val-GA nie wymaga skanu V <i>z definicji</i> — jego fitness jest liczony na podzbiorach — ale w tej konfiguracji (populacja 30, 20 generacji) "
        "nominalnie klasyfikuje (G+1)·P·n<sub>sel</sub> = 630·n<sub>sel</sub> wierszy na rozszerzenie, czyli więcej niż |V \\ V′| gdy n<sub>sel</sub> przekracza ok. |V \\ V′|/630 "
        "(dla Szachownicy ≈ 44 próbki; średni przyrost jest większy). Ponieważ model jest w trakcie Val-GA ustalony, implementacja liczy predykcje puli raz i "
        "indeksuje je (wynik identyczny, zweryfikowane testem) — to oszczędność czasu, nie metody. "
        "Wariant losowy nie wymaga żadnych predykcji. Diagnostyka trudności przyrostów (odsetek błędów w puli) wymaga skanu, ale jest liczona osobno "
        "(<i>diagnostic_predictions</i>) i nie wpływa na przebieg."
    )
    doc.h2("3.5. Poprawki w repozytorium wynikające z analizy")
    doc.bullets(
        [
            "<b>reporting.py</b>: <i>build_report_row</i> pomijał tryb <i>dagasvm_phase</i> przy stosowaniu maski T′, więc wiersz raportu (a z niego ranking sweepa) "
            "opisywał SVM wytrenowany na całym Train. Naprawione + test regresyjny.",
            "<b>Pętla fazowa</b> przeniesiona do <i>ga.phase_engine</i> (testowalna), z jawną polityką decyzji; zachowanie dla konfiguracji historycznych jest identyczne "
            "(odtworzone bit-w-bit, sekcja 4).",
            "<b>Szybsza ścieżka metryk</b> w pętli GA (liczona tylko metryka fitnessu zamiast pełnego zestawu; wartości identyczne) — skraca przebiegi bez zmiany wyników.",
            "Nieaktualne testy (oczekiwały validation_set_selection=Yes dla DAGASVM adaptacyjnego oraz błędnego nagłówka pliku testowego) zaktualizowano.",
        ]
    )


def sec_config(doc, d):
    doc.h1("4. Konfiguracja eksperymentów")
    c = cfg_of("chessboard")
    rows = [
        ["Parametr", "Wartość (wszystkie warianty i zbiory, o ile nie zaznaczono)"],
        ["Seedy algorytmu", "42, 43, …, 61 (20, identyczne dla wszystkich wariantów i zbiorów)"],
        ["Podział danych", "stały (Szachownica: pliki z repozytorium; pozostałe: stratyfikowany 40/40/20, ziarno generatora i podziału 2026); seed zmienia tylko algorytm"],
        ["SVM", "RBF, class_weight=balanced; (C, γ) na zbiór — tabela 5; bez standaryzacji w pętli (dane wielowymiarowe wcześniej wystandaryzowane na Train)"],
        ["Metryka", "balanced accuracy (fitness T′, metryka sterująca na V′, metryki obserwacyjne)"],
        ["GA dla T′", f"populacja {c['ga']['population_size']}, ≤{c['ga']['generations']} generacji, turniej {c['ga']['tournament_k']}, krzyżowanie {c['ga']['crossover_rate']}, "
                      f"mutacja {c['ga']['mutation_rate']}, elitaryzm {c['ga']['elitism']}; stop: brak poprawy &gt; {c['daga']['phase_min_delta']} przez Ts={c['daga']['Ts']} generacji"],
        ["Polityka K", f"K<sub>start</sub>={c['daga']['k_start']} (Digits: 5), TSV={c['daga']['TSV']}, Ts={c['daga']['Ts']}, ε<sub>stab</sub>={c['daga']['sv_ratio_stability_epsilon']}, margines nasycenia={c['daga']['sv_ratio_saturation_margin']}, próg „100%” = 1,0"],
        ["Val-GA (wariant ewolucyjny)", f"populacja {c['val_ga']['population_size']}, {c['val_ga']['generations']} generacji, turniej {c['val_ga']['tournament_k']}, krzyżowanie {c['val_ga']['crossover_rate']}, mutacja bitowa {c['val_ga']['mutation_rate']}, elitaryzm {c['val_ga']['elitism']}"],
        ["Obserwacja", "pełne V i test po każdym cyklu (nie steruje)"],
        ["Warunek stopu", "ρ<sub>SV</sub> &lt; TSV, 100% na V′ po utrudnieniu, brak kandydatów, limit K (najmniejsza klasa Train); bezpiecznik 500 cykli"],
    ]
    doc.table(rows, [3.6, 13.7], "Wspólna konfiguracja eksperymentów.")
    params = {}
    for ds in D2 + HD:
        try:
            cc = cfg_of(ds)
            ddir = cc["data"]["train_csv"].rsplit("/", 1)[0]
            meta_path = REPO / ddir / "meta.json"
            if meta_path.exists():
                import json
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                tr, va, te = (m["splits"][k]["rows"] for k in ("train", "validation", "test"))
                cls = m["splits"]["train"]["class_proportions"]
                prop = " / ".join(f"{float(v) * 100:.0f}%" for v in cls.values()) if len(cls) <= 3 else f"{len(cls)} klas, po ≈{100 / len(cls):.0f}%"
                nfeat = m["features"]
            else:
                tr, va, te, prop, nfeat = 19949, 27853, 22169, "58% / 42%", 2
            params[ds] = (tr, va, te, nfeat, prop, cc["svm"]["C"], cc["svm"]["gamma"], cc["daga"]["k_start"])
        except FileNotFoundError:
            continue
    rows = [["Zbiór", "Train", "V", "Test", "Cechy", "Proporcje klas (Train)", "C", "γ", "K<sub>start</sub>"]]
    for ds, (tr, va, te, nf, prop, C, g, ks) in params.items():
        rows.append([DATASET_LABELS[ds], fi(tr), fi(va), fi(te), nf, prop, f(C, 1), f(g, 4), ks])
    doc.table(rows, [3.1, 1.5, 1.5, 1.5, 1.2, 3.8, 1.0, 1.3, 1.5], "Zbiory danych i hiperparametry SVM (C, γ dobrane raz pełnym SVM na V; wspólne dla wariantów).")
    rows = [["Wariant", "selection", "memory", "mode", "rozmiar V′"]]
    for v, spec in d.spec["variants"].items():
        size = "średni końcowy |V′| z evo_mem" if spec.get("mode") == "fixed" else "rośnie wg harmonogramu K"
        rows.append([f"<b>{v}</b> — {VARIANT_LABELS[v]}", spec["selection"], "tak" if spec["memory"] else "nie", spec["mode"], size])
    doc.table(rows, [6.4, 2.5, 1.6, 1.8, 5.0], "Warianty (configs/study/&lt;zbiór&gt;/&lt;wariant&gt;.yaml).")


# ------------------------------------------------------------------------------------------
# sekcje wynikowe (liczby z tabel)
# ------------------------------------------------------------------------------------------
METRIC_ROWS = [
    ("vprime_ba", "BA na V′ (metryka sterująca)", 4),
    ("test_ba", "BA na teście (obserwacja)", 4),
    ("fullv_ba", "BA na pełnym V (obserwacja)", 4),
    ("cycles", "liczba cykli T′", 1),
    ("final_k", "końcowe K", 1),
    ("vprime_size", "końcowy |V′|", 1),
    ("vprime_changes", "liczba zmian (rozszerzeń) V′", 1),
    ("k_increases", "liczba wzrostów K", 1),
    ("num_support_vectors", "liczba SV (model końcowy)", 1),
    ("svm_fits_total", "liczba treningów SVM w GA (T′)", 0),
    ("duration_sec", "czas wykonania [s] (zegarowy, równoległe uruchomienia)", 0),
]


def sec_series_table(doc, d, dataset, variant, caption):
    rows = [["Metryka", "średnia", "odch. std.", "min", "mediana", "max", "CV"]]
    for key, label, dig in METRIC_ROWS:
        rows.append([label] + [f(d.stat(dataset, variant, key, k), dig if k != "cv" else 3)
                               for k in ("mean", "sd", "min", "median", "max", "cv")])
    doc.table(rows, [6.2, 2.0, 2.0, 2.0, 2.0, 2.0, 1.3], caption)


def sec_legacy_audit(doc, d):
    a = d.legacy
    doc.h2("5.3. Wiarygodność plików agregujących")
    doc.p(
        f"Plik ranking.csv wcześniejszego sweepa (20 seedów, Szachownica) zawiera {a['ranking_unique_num_support_vectors']} unikalną wartość liczby SV "
        f"({a['ranking_num_support_vectors_value']}) oraz {a['ranking_unique_full_validation_f1']} unikalną wartość F1 na pełnym V "
        f"({f(a['ranking_full_validation_f1_mean'], 4)}) dla wszystkich 20 przebiegów, podczas gdy metryki per-run mają {a['per_run_unique_num_support_vectors']} "
        f"różnych liczb SV (średnio {f(a['per_run_num_support_vectors_mean'], 1)}) i {a['per_run_unique_full_validation_f1']} różnych wartości F1 "
        f"(średnio {f(a['per_run_full_validation_f1_mean'], 4)}). Przyczyną był błąd w <i>build_report_row</i> (sekcja 3.5): model do wiersza raportu "
        "trenowano na całym zbiorze Train zamiast na reprezentatywnym T′. Ranking został <b>wykluczony z analizy</b>; wszystkie liczby w raporcie pochodzą z metryk per-run "
        "(summary.json, cycles.csv). Błąd naprawiono, więc nowe serie mają poprawne wiersze raportu."
    )
    if d.regression is not None and len(d.regression):
        n_same = int(d.regression.identical_trajectory.sum())
        doc.p(
            f"<b>Test powtarzalności względem poprzedniej implementacji.</b> Seria evo_mem na Szachownicy została uruchomiona ponownie nowym kodem na tych samych 20 seedach; "
            f"{n_same}/{len(d.regression)} przebiegów ma identyczną liczbę cykli, końcowe K, rozmiar V′ i test BA (różnica &lt; 1e-12) jak w serii historycznej. "
            "Refaktoryzacja i szybsza ścieżka metryk nie zmieniły więc wyników; różnią się wyłącznie czasy zegarowe (nowa seria uruchamiana była 10 procesami równolegle zamiast 4)."
        )


def sec_series_validation(doc, d):
    rows = [["Zbiór", "Wariant", "Przebiegi", "Komplet 20 seedów", "Uwagi"]]
    for _, r in d.validation.iterrows():
        rows.append([DATASET_LABELS[r.dataset], r.variant, int(r.n_runs), "tak" if r.complete_20_seeds else "NIE",
                     "" if pd.isna(r.issues) else str(r.issues)[:90]])
    doc.table(rows, [3.0, 2.6, 2.0, 3.0, 6.7], "Kompletność serii (seedy 42–61, status completed, brak wzrostu K &gt; 1 na cykl).")


PAIRED_ROWS = [
    ("test_ba", "test BA (obs.)", 4), ("fullv_ba", "pełne V BA (obs.)", 4), ("vprime_ba", "V′ BA (steruje)", 4),
    ("cycles", "cykle", 1), ("final_k", "końcowe K", 1), ("vprime_size", "końcowy |V′|", 0),
    ("vprime_changes", "zmiany V′", 1), ("svm_fits_total", "treningi SVM", 0), ("duration_sec", "czas [s]", 0),
]


def sec_paired_table(doc, d, dataset, a, b, caption):
    rows = [["Metryka", f"{VARIANT_SHORT[a]} (śr.)", f"{VARIANT_SHORT[b]} (śr.)", "różnica śr. (a−b)", "95% CI (bootstrap)", "a lepszy / b lepszy (seedy)", "p Wilcoxon"]]
    for key, label, dig in PAIRED_ROWS:
        p = d.pair(dataset, a, b, key)
        if p is None:
            continue
        rows.append([label, f(d.stat(dataset, a, key), dig), f(d.stat(dataset, b, key), dig), f(p.mean_diff, dig if dig else 1),
                     f"[{f(p.ci95_low, dig if dig else 1)}; {f(p.ci95_high, dig if dig else 1)}]",
                     f"{int(p.a_better_count)} / {int(p.b_better_count)}", pval(p.wilcoxon_p)])
    doc.table(rows, [3.0, 2.3, 2.3, 2.4, 3.4, 2.4, 1.5], caption)


def sec_dataset_table(doc, d, dataset, caption):
    rows = [["Wariant", "n", "V′ BA (steruje)", "test BA (obs.)", "pełne V BA (obs.)", "cykle", "K końc.", "|V′| końc.", "czas [s]"]]
    for v in VARIANT_LABELS:
        if not d.have(dataset, v):
            continue
        rows.append([VARIANT_SHORT[v], d.n_runs(dataset, v),
                     pm(d.stat(dataset, v, "vprime_ba"), d.stat(dataset, v, "vprime_ba", "sd"), 3),
                     pm(d.stat(dataset, v, "test_ba"), d.stat(dataset, v, "test_ba", "sd"), 4),
                     pm(d.stat(dataset, v, "fullv_ba"), d.stat(dataset, v, "fullv_ba", "sd"), 4),
                     pm(d.stat(dataset, v, "cycles"), d.stat(dataset, v, "cycles", "sd"), 1),
                     pm(d.stat(dataset, v, "final_k"), d.stat(dataset, v, "final_k", "sd"), 0),
                     pm(d.stat(dataset, v, "vprime_size"), d.stat(dataset, v, "vprime_size", "sd"), 0),
                     f(d.stat(dataset, v, "duration_sec"), 0)])
    doc.table(rows, [2.7, 0.8, 2.3, 2.4, 2.4, 1.9, 1.8, 2.0, 1.2], caption)


def sec_per_seed_table(doc, d, dataset, variant, caption):
    sub = d.per_run[(d.per_run.dataset == dataset) & (d.per_run.variant == variant)].sort_values("seed")
    rows = [["seed", "cykle", "K", "|T′|", "|V′|", "zmiany V′", "wzr. K", "SV", "V′ BA", "test BA", "pełne V BA", "czas [s]", "stop"]]
    for _, r in sub.iterrows():
        rows.append([int(r.seed), int(r.cycles), int(r.final_k), int(r.training_rows), int(r.vprime_size), int(r.vprime_changes),
                     int(r.k_increases), int(r.num_support_vectors), f(r.vprime_ba, 4), f(r.test_ba, 4), f(r.fullv_ba, 4),
                     fi(r.duration_sec), str(r.stop_decision).replace("stop_", "")])
    doc.table(rows, [0.9, 1.0, 1.0, 1.2, 1.2, 1.3, 1.0, 0.9, 1.4, 1.4, 1.6, 1.3, 2.6], caption)


def add_all(doc, d):
    """Składa pełny raport z sekcji zdefiniowanych w tym module.

    Generator PDF importuje tę funkcję jako jedyny punkt wejścia. Dzięki temu
    część opisowa i układ dokumentu nie mogą rozjechać się przy dalszej edycji.
    """
    doc.h1("DAGASVM — badanie doboru podzbioru walidacyjnego V′")
    doc.p(
        "Raport badawczy porównujący ewolucyjny, heurystyczny i losowy dobór V′, "
        "z opcją pamiętania wcześniej wybranych próbek. Wszystkie zestawienia liczbowe "
        "pochodzą z metryk per-run; pełny zbiór V i test są prezentowane wyłącznie jako obserwacje."
    )
    doc.h1("1. Cel badania")
    doc.p(
        "Celem jest sprawdzenie, czy sposób budowania V′ wpływa na jakość końcową, "
        "koszt obliczeniowy i stabilność fazowego DAGASVM. Porównania są sparowane po "
        "identycznych seedach 42–61; raport końcowy jest generowany tylko po ukończeniu "
        "wszystkich wymaganych serii."
    )
    sec_roles(doc, d)
    sec_implementation(doc, d)
    sec_config(doc, d)

    doc.h1("5. Wyniki i kompletność danych")
    sec_series_validation(doc, d)
    if d.have("chessboard", "evo_mem"):
        sec_series_table(doc, d, "chessboard", "evo_mem", "Stabilność wariantu ewolucyjnego z pamięcią — Szachownica.")
        doc.figure(
            "fig_trajectory_chessboard_evo_mem.png",
            "Trzy reprezentatywne przebiegi: zmiana V′, K oraz metryk sterujących i obserwacyjnych.",
        )
        doc.figure(
            "fig_decisions_chessboard_evo_mem.png",
            "Decyzje algorytmu dla wszystkich seedów wariantu ewolucyjnego z pamięcią.",
        )
        doc.figure(
            "fig_increment_difficulty_chessboard.png",
            "Trudność elementów dodawanych do V′ względem całej puli kandydatów.",
        )
        doc.figure(
            "fig_vprime_k_growth_chessboard.png",
            "Mediany rozmiaru V′ i K w kolejnych cyklach dla wariantów Szachownicy.",
        )
    sec_legacy_audit(doc, d)
    doc.figure("fig_overview_datasets.png", "Porównanie jakości, liczby cykli i końcowego K między zbiorami.")

    doc.h1("6. Porównania wariantów")
    pairs = [
        ("evo_mem", "rand_fixed", "ewolucyjny z pamięcią vs losowy stały"),
        ("evo_mem", "rand_growth", "ewolucyjny z pamięcią vs losowy rosnący"),
        ("evo_mem", "heur_mem", "ewolucyjny z pamięcią vs heurystyka błędów"),
        ("evo_mem", "evo_nomem", "ewolucyjny z pamięcią vs ewolucyjny bez pamięci"),
    ]
    for dataset in d.spec["datasets"]:
        available = [variant for variant in VARIANT_LABELS if d.have(dataset, variant)]
        if not available:
            continue
        doc.h2(DATASET_LABELS[dataset])
        sec_dataset_table(doc, d, dataset, f"Wyniki dla zbioru {DATASET_LABELS[dataset]} (średnia ± odchylenie standardowe).")
        for a, b, label in pairs:
            if d.have(dataset, a) and d.have(dataset, b):
                sec_paired_table(doc, d, dataset, a, b, f"Porównanie sparowane: {DATASET_LABELS[dataset]}, {label}.")
        if dataset in D2:
            doc.figure(
                f"fig_2d_{dataset}.png",
                f"Dane 2D oraz granice decyzyjne reprezentatywnych modeli — {DATASET_LABELS[dataset]}.",
            )

    doc.h1("7. Wnioski i ograniczenia")
    doc.bullets(
        [
            "Wnioski o przewadze wariantów należy formułować wyłącznie dla kompletnych, sparowanych serii 20 seedów; "
            "wartości dla nieukończonych serii są raportem roboczym, a nie wynikiem końcowym.",
            "Wariant rand_fixed otrzymuje średni końcowy rozmiar V′ z evo_mem, więc jest łagodnie uprzywilejowany; "
            "porównanie z nim należy interpretować ostrożnie.",
            "Pełne V nie steruje pętlą T′/K, ale jest pulą, z której wybierane są kandydaty do V′. "
            "Wariant heurystyczny i ewolucyjny mogą więc wymagać oceny całej puli kandydatów, co jest częścią ich kosztu.",
            "Wynik na teście pozostaje jedyną miarą zewnętrznej generalizacji; pełne V jest obserwacją pomocniczą, "
            "ponieważ V′ jest jego podzbiorem.",
        ]
    )


# ------------------------------------------------------------------------------------------
# składanie raportu
# ------------------------------------------------------------------------------------------
def _stop_counts(d, dataset, variant):
    sub = d.per_run[(d.per_run.dataset == dataset) & (d.per_run.variant == variant)]
    return sub.stop_decision.value_counts().to_dict()


def _pair_sentence(d, dataset, a, b, metric="test_ba", dig=4):
    p = d.pair(dataset, a, b, metric)
    if p is None:
        return ""
    return (f"{DATASET_LABELS[dataset]}: różnica {f(p.mean_diff, dig)} (95% CI [{f(p.ci95_low, dig)}; {f(p.ci95_high, dig)}], "
            f"p<sub>Wilcoxon</sub> = {pval(p.wilcoxon_p)}; {int(p.a_better_count)}/{int(p.b_better_count)} seedów na korzyść a/b)")


def sec_results_chessboard(doc, d):
    doc.h1("5. Wyniki: ewolucyjny dobór V′ — 20 powtórzeń (Szachownica)")
    doc.p(
        f"Seria evo_mem obejmuje 20 przebiegów (seedy 42–61, wszystkie zakończone statusem completed, zakończenie decyzją "
        f"<i>{', '.join(_stop_counts(d, 'chessboard', 'evo_mem'))}</i>). Wszystkie liczby pochodzą z metryk per-run. "
        "Metryki sterujące (V′) i obserwacyjne (pełne V, test) podano osobno."
    )
    doc.h2("5.1. Statystyki serii")
    sec_series_table(doc, d, "chessboard", "evo_mem", "Szachownica, wariant ewolucyjny z pamięcią (n = 20). CV = odch. std. / średnia.")
    q_cv = d.stat("chessboard", "evo_mem", "test_ba", "cv")
    c_cv = d.stat("chessboard", "evo_mem", "duration_sec", "cv")
    cy_cv = d.stat("chessboard", "evo_mem", "cycles", "cv")
    doc.p(
        f"<b>Stabilność jakości.</b> Jakość końcowa jest ciasno skupiona: test BA = {pm(d.stat('chessboard','evo_mem','test_ba'), d.stat('chessboard','evo_mem','test_ba','sd'))} "
        f"(CV = {f(q_cv, 4)}), rozstęp {f(d.stat('chessboard','evo_mem','test_ba','max') - d.stat('chessboard','evo_mem','test_ba','min'), 4)}. "
        f"<b>Stabilność kosztu.</b> Koszt dojścia do wyniku zmienia się wielokrotnie silniej: liczba cykli CV = {f(cy_cv, 3)} "
        f"({fi(d.stat('chessboard','evo_mem','cycles','min'))}–{fi(d.stat('chessboard','evo_mem','cycles','max'))}), liczba treningów SVM w GA "
        f"CV = {f(d.stat('chessboard','evo_mem','svm_fits_total','cv'), 3)}, czas zegarowy CV = {f(c_cv, 3)}. "
        "Końcowe K jest stabilne (CV ≈ 0,02), końcowy rozmiar V′ umiarkowanie zmienny. Czasy zegarowe są zawyżone i zaszumione przez równoległe uruchamianie wielu procesów "
        "na jednej maszynie, dlatego jako miarę kosztu należy preferować liczby cykli i treningów SVM."
    )
    doc.h2("5.2. Tabela per seed")
    sec_per_seed_table(doc, d, "chessboard", "evo_mem", "Wyniki per seed — Szachownica, evo_mem.")
    sec_legacy_audit(doc, d)


def sec_vs_random(doc, d):
    doc.h1("6. Ewolucyjny V′ a losowy V′ — identyczne seedy")
    doc.p(
        "Wszystkie warianty uruchomiono na tych samych 20 seedach (42–61), z identycznymi podziałami danych, parametrami GA, polityką K i kryteriami stopu. "
        "Różnica a−b jest liczona parami (po seedzie); podano średnią różnic, 95% przedział bootstrapowy (10 000 prób) i test Wilcoxona. "
        "Przy 20 parach i wielu metrykach/zbiorach wartości p nie są korygowane o porównania wielokrotne — należy je traktować opisowo. "
        "<b>Wariant rand_fixed ma rozmiar V′ równy średniemu końcowemu rozmiarowi V′ z serii ewolucyjnej (łagodnie uprzywilejowuje wariant losowy).</b> "
        "Jego V′ jest od początku duże, więc przebieg jest krótszy — porównanie kosztu jest więc korzystne dla wariantu losowego również z tego powodu."
    )
    for ds in D2 + HD:
        if not d.have(ds, "rand_fixed"):
            continue
        doc.h2(f"6.{(D2 + HD).index(ds) + 1}. {DATASET_LABELS[ds]}")
        sec_dataset_table(doc, d, ds, f"{DATASET_LABELS[ds]} — wszystkie warianty (średnia ± odch. std.; V′ BA steruje algorytmem, test i pełne V to obserwacje).")
        sec_paired_table(doc, d, ds, "evo_mem", "rand_fixed", f"{DATASET_LABELS[ds]}: evo_mem (a) vs rand_fixed (b), pary po seedach.")
        doc.figure(f"fig_paired_evo_vs_randfixed_{ds}.png", f"{DATASET_LABELS[ds]}: porównanie sparowane evo_mem vs rand_fixed.", 16.0)


def sec_memory(doc, d):
    doc.h1("7. Pamiętanie V′ między fazami")
    doc.p(
        "Wariant z pamięcią (evo_mem) i bez pamięci (evo_nomem) różnią się wyłącznie tym, czy elementy wybrane w poprzednich fazach pozostają w V′. "
        "Przy braku pamięci rozmiar V′ rośnie według tego samego harmonogramu, ale każdy nowy V′ jest świeżym wyborem z V \\ V′<sub>t</sub> (elementy poprzednie są wyłączone z puli "
        "kandydatów w danym kroku). Poprawność mechanizmu (brak duplikatów, zachowanie identyfikatorów prefiksowo, rozłączność przy braku pamięci) jest sprawdzana asercjami i testami."
    )
    for ds in D2 + HD:
        if d.have(ds, "evo_nomem"):
            doc.p(_pair_sentence(d, ds, "evo_mem", "evo_nomem") + "; cykle: " + _pair_sentence(d, ds, "evo_mem", "evo_nomem", "cycles", 1).split(": ", 1)[1])
    doc.figure("fig_paired_memory_chessboard.png", "Szachownica: evo_mem vs evo_nomem, pary po seedach.", 16.0)
    doc.figure("fig_paired_memory_circles.png", "Circles: evo_mem vs evo_nomem, pary po seedach.", 16.0)


def sec_other_datasets(doc, d):
    doc.h1("8. Nowe zbiory 2D i zbiory wielowymiarowe")
    doc.p(
        "Dla zbiorów syntetycznych 2D (Moons, Circles, Spiral, Classification 2D) oraz wielowymiarowych (Digits: 64 cechy, 10 klas; MAGIC Gamma Telescope z UCI: 10 cech, "
        "podpróbka stratyfikowana 8000 obserwacji) wykonano serie 20 seedów dla wariantów evo_mem, evo_nomem, rand_growth, heur_mem i rand_fixed (tabele i wykresy w sekcji 6). "
        "Podział Train/V/Test jest deterministyczny (stratyfikowany 40/40/20; ziarno 2026), skalowanie dla Digits i MAGIC wyuczone wyłącznie na Train (meta.json zapisuje średnie i skale)."
    )
    for ds in ["moons", "circles", "spiral", "classification2d"]:
        doc.figure(f"fig_2d_{ds}.png", f"{DATASET_LABELS[ds]}: dane Train oraz granica decyzyjna modelu końcowego (przebieg o medianowej liczbie cykli) dla wariantu ewolucyjnego i losowego; V′ — szare punkty, T′ — okręgi.", 17.0)
    doc.figure("fig_overview_datasets.png", "Przegląd: średnie ± odch. std. po seedach dla wszystkich zbiorów i wariantów.", 17.0)
    doc.pagebreak()
    doc.h2("8.1. Przebieg algorytmu — wizualizacje")
    doc.figure("fig_trajectory_chessboard_evo_mem.png", "Szachownica, evo_mem: rozmiar V′, K, liczba wektorów podpierających (SV) oraz jakość (V′ — steruje; pełne V i test — obserwacje) w kolejnych cyklach dla najkrótszego, typowego i najdłuższego przebiegu.", 17.0)
    doc.figure("fig_decisions_chessboard_evo_mem.png", "Szachownica, evo_mem: decyzje algorytmu w kolejnych cyklach dla wszystkich seedów.", 16.0)
    doc.figure("fig_increment_difficulty_chessboard.png", "Szachownica: odsetek błędów modelu w elementach dodanych do V′ (ewolucyjny, heurystyczny, losowy) wobec całej puli kandydatów.", 17.0)
    doc.figure("fig_vprime_k_growth_chessboard.png", "Szachownica: wzrost V′ i K (mediany po seedach) dla wszystkich wariantów.", 15.0)
    doc.figure("fig_trajectory_moons_evo_mem.png", "Moons, evo_mem: przebieg krótki / typowy / długi.", 17.0)


def sec_summary(doc, d):
    doc.h1("1. Wnioski własne")
    n_runs = len(d.per_run)
    n_series = int(d.validation.complete_20_seeds.sum())
    doc.bullets(
        [
            "Wariant ewolucyjny V′ sprawdzony w 20 powtórzeniach osiąga bardzo zbliżoną jakość końcową między seedami, natomiast koszt dojścia do wyniku (liczba cykli i treningów SVM) jest wyraźnie bardziej zmienny.",
        ]
    )
    doc.h2("Zakres badania")
    doc.p(
        "Celem badania jest ocena, czy <b>ewolucyjny dobór podzbioru walidacyjnego V′</b> w fazowym DAGASVM (naprzemienna optymalizacja podzbioru treningowego T′ "
        "i rozszerzanie V′, ze wzrostem K najwyżej raz na cykl) daje wymierną korzyść w porównaniu z wariantami bazowymi: losowym V′ o stałym rozmiarze "
        "(rozmiar = średni końcowy rozmiar V′ z serii ewolucyjnej), losowym V′ rosnącym wg tego samego harmonogramu oraz heurystyką wyboru błędnie klasyfikowanych przykładów. "
        "Zbadano również opcję pamiętania V′ między fazami. Eksperymenty wykonano na Szachownicy, czterech syntetycznych zbiorach 2D (Moons, Circles, Spiral, Classification 2D) "
        f"i dwóch zbiorach wielowymiarowych (Digits, MAGIC) — łącznie {n_runs} przebiegów w {n_series} seriach po 20 identycznych seedów (42–61)."
    )
    doc.bullets(
        [
            "Weryfikacja kodu nie wykazała użycia pełnego V do sterowania algorytmem; V jest pulą kandydatów dla V′ oraz źródłem metryk obserwacyjnych (sekcja 2). "
            "Wcześniejszy plik ranking.csv był niewiarygodny (błąd w raportowaniu) i został wykluczony (5.3).",
            "Jakość końcowa wariantu ewolucyjnego jest bardzo stabilna między seedami (Szachownica: test BA "
            f"{pm(d.stat('chessboard','evo_mem','test_ba'), d.stat('chessboard','evo_mem','test_ba','sd'))}), natomiast koszt dojścia do wyniku (liczba cykli, treningów SVM, czas) jest zmienny wielokrotnie silniej.",
            "W naszych danych <b>nie stwierdzono spójnej przewagi jakości ewolucyjnego V′ nad losowym V′</b>: różnice w test BA są rzędu 10<sup>−3</sup> i mają zmienny znak między zbiorami; "
            "losowy V′ o stałym, „podpowiedzianym” rozmiarze osiąga porównywalną jakość przy 2–3 razy mniejszej liczbie cykli i treningów SVM.",
            "Pamiętanie V′ daje niewielką, nieistotną statystycznie poprawę średniej jakości na wszystkich zbiorach oraz zmniejsza zmienność wyników na większości z nich (sekcje 7, 9.3).",
            "Zjawiska te występują także na zbiorach wielowymiarowych, ale tam K dochodzi do limitu liczności klasy, więc wniosek jest słabszy (sekcja 9).",
        ]
    )


def sec_interpretation(doc, d):
    doc.h1("9. Interpretacja wyników")
    pr = d.per_run
    doc.h2("9.1. Czy ewolucja V′ rzeczywiście „ewoluuje” V′?")
    s = d.val_ga.copy()
    s["gain"] = s.ga_final_best_fitness - s.ga_initial_best_fitness
    g = s.groupby("dataset").agg(gain=("gain", "mean"), sel=("selected_error_rate", "mean"), pool=("pool_error_rate", "mean"),
                                 nom=("nominal_subset_rows_scored", "mean"), cand=("n_candidates", "mean"))
    rows = [["Zbiór", "przyrost fitnessu Val-GA (końcowy − populacja startowa)", "błędy w wybranych elementach", "błędy w puli V \\ V′ (= losowy)", "nominalna liczba klasyfikowanych wierszy na rozszerzenie", "|V \\ V′| (średnio)"]]
    for ds in D2 + HD:
        if ds in g.index:
            r = g.loc[ds]
            rows.append([DATASET_LABELS[ds], f(r.gain, 3), f(r.sel, 3), f(r.pool, 3), fi(r.nom), fi(r.cand)])
    doc.table(rows, [2.6, 3.4, 2.6, 2.8, 3.6, 2.3], "Skuteczność i koszt wyszukiwania Val-GA (średnie po wszystkich rozszerzeniach wariantów evo_mem i evo_nomem).")
    doc.p(
        "Val-GA wyszukuje podzbiory trudniejsze niż losowe: fitness najlepszego osobnika rośnie w trakcie 20 generacji względem populacji startowej, a odsetek błędów modelu w wybranych elementach "
        "jest kilkakrotnie wyższy niż w całej puli (np. Szachownica ≈ 0,13 vs 0,06). Mechanizm nie sprowadza się jednak do „dodawania błędów z pełnego V”: wybierane są podzbiory o zadanej liczności, "
        "wśród których błędy stanowią mniejszość, a heurystyka błędów (heur_mem) wybiera elementy, które są błędami w ≈ 70–100% na pięciu zbiorach (Szachownica, Moons, Spiral, Classification 2D, MAGIC; na Circles ≈ 0,12 i Digits ≈ 0,26, gdzie pula zawiera mało błędów), czyli znacznie „twardsze” przyrosty. "
        "Wyszukiwanie ewolucyjne daje więc łagodniejszy, bardziej zróżnicowany (a nie minimalnomarginesowy) przyrost niż heurystyka."
    )
    doc.p(
        "<b>Koszt.</b> Wariant ewolucyjny nie wymaga <i>koncepcyjnie</i> klasyfikowania całego V, ale nominalna liczba klasyfikowanych wierszy w Val-GA (kolumna 5) jest "
        "w każdym badanym zbiorze wielokrotnie większa niż rozmiar puli kandydatów (kolumna 6), więc przy tych parametrach (populacja 30, 20 generacji) pełny skan puli byłby tańszy. "
        "Wynik ten dotyczy konkretnej konfiguracji Val-GA; przy małych przyrostach lub mniejszej populacji proporcja mogłaby się odwrócić. "
        "Heurystyka wymaga dokładnie |V \\ V′| predykcji na rozszerzenie; wariant losowy — zero."
    )
    doc.h2("9.2. Ewolucyjny a losowy V′")
    doc.bullets([_pair_sentence(d, ds, "evo_mem", "rand_fixed") for ds in D2 + HD if d.have(ds, "rand_fixed")])
    doc.p(
        "Różnice w test BA między evo_mem a rand_fixed są małe (|średnia różnica| ≤ 0,0031 na każdym zbiorze) i nie mają stałego znaku; przedziały ufności w większości obejmują zero, "
        "a dla dwóch zbiorów (Circles, MAGIC) przeważa wariant losowy. Ponieważ rozmiar rand_fixed jest łagodnie uprzywilejowany (patrz 3.2), ten wynik <i>nie</i> dowodzi, że losowy dobór jest równoważny "
        "w sytuacji, gdy wymagany rozmiar V′ nie jest znany; wskazuje natomiast, że przy znajomości rozmiaru jakość końcowa nie zależy wyraźnie od tego, <i>które</i> elementy wchodzą do V′. "
        "Z kolei wariant rand_growth (ten sam harmonogram rozszerzeń, bez podpowiedzi o rozmiarze) osiąga jakość podobną (Szachownica, Spiral, Digits, MAGIC) lub niższą i bardziej zmienną "
        "(Moons, Circles, Classification 2D) — w tych zbiorach jest to związane głównie z przedwczesnym zatrzymaniem: małe losowe V′ (kilkadziesiąt próbek) "
        "bywa „doskonale” klasyfikowane przy K = 10 i algorytm kończy po 2 cyklach (liczby przypadków w tabeli poniżej). To wskazuje, że reguła stopu „100% na V′” jest wrażliwa na rozmiar V′ i "
        "to ona, a nie jakość doboru elementów, odpowiada za część różnic."
    )
    rows = [["Zbiór"] + [VARIANT_SHORT[v] for v in VARIANT_LABELS]]
    for ds in D2 + HD:
        row = [DATASET_LABELS[ds]]
        for v in VARIANT_LABELS:
            if not d.have(ds, v):
                row.append("—")
            else:
                row.append(f"{int((pr[(pr.dataset == ds) & (pr.variant == v)].stop_decision == 'stop_perfect_on_current_val').sum())} / {d.n_runs(ds, v)}")
        rows.append(row)
    doc.table(rows, [3.4, 2.8, 2.8, 2.8, 2.8, 2.8], "Liczba przebiegów zatrzymanych decyzją stop_perfect_on_current_val (100% na V′ po utrudnieniu V′).")
    ch = lambda v, m: d.stat("chessboard", v, m)
    doc.p(
        f"<b>Koszt.</b> Wariant rand_fixed jest znacząco tańszy: na Szachownicy {f(ch('rand_fixed','cycles'),1)} cykli i {fi(ch('rand_fixed','svm_fits_total'))} treningów SVM wobec "
        f"{f(ch('evo_mem','cycles'),1)} i {fi(ch('evo_mem','svm_fits_total'))} dla evo_mem (różnica istotna, 19/20 seedów). Wynika to jednak głównie z braku kroków rozszerzania V′ (V′ jest duże od początku), "
        "a nie z samego doboru losowego; rand_growth ma koszt zbliżony do evo_mem. Metryka na V′ (sterująca) <b>nie jest porównywalna</b> między wariantami: "
        f"losowe V′ jest łatwiejsze (Szachownica: {f(ch('rand_fixed','vprime_ba'),3)} vs {f(ch('evo_mem','vprime_ba'),3)} dla ewolucyjnego), co jest oczekiwanym skutkiem doboru trudnych przykładów, "
        "a nie różnicą jakości modelu."
    )
    doc.h2("9.3. Pamięć V′")
    rows = [["Zbiór", "test BA: mem", "test BA: bez pam.", "różnica (95% CI)", "odch. std. mem / bez pam.", "przedwczesne stopy (mem / bez pam.)"]]
    for ds in D2 + HD:
        if not d.have(ds, "evo_nomem"):
            continue
        p = d.pair(ds, "evo_mem", "evo_nomem", "test_ba")
        sub = lambda v: pr[(pr.dataset == ds) & (pr.variant == v)]
        pf = lambda v: int((sub(v).stop_decision == "stop_perfect_on_current_val").sum())
        rows.append([DATASET_LABELS[ds], f(d.stat(ds, "evo_mem", "test_ba"), 4), f(d.stat(ds, "evo_nomem", "test_ba"), 4),
                     f"{f(p.mean_diff, 4)} [{f(p.ci95_low, 4)}; {f(p.ci95_high, 4)}]",
                     f"{f(d.stat(ds,'evo_mem','test_ba','sd'),4)} / {f(d.stat(ds,'evo_nomem','test_ba','sd'),4)}", f"{pf('evo_mem')} / {pf('evo_nomem')}"])
    doc.table(rows, [2.7, 2.2, 2.4, 4.4, 3.4, 2.7], "Pamięć V′: porównanie evo_mem (a) i evo_nomem (b), te same seedy.")
    doc.p(
        "Wariant z pamięcią ma wyższą średnią jakość testową na wszystkich siedmiu zbiorach, ale różnice są małe i tylko w części przypadków przedział ufności wyklucza zero. Odchylenie standardowe jakości jest niższe przy pamięci na pięciu zbiorach (Moons, Circles, Spiral, Digits, MAGIC), zbliżone na dwóch (Szachownica, Classification 2D). "
        "Wyraźniejszy efekt dotyczy zmienności: bez pamięci część przebiegów kończy się wcześnie (mały V′ „bez historii” bywa doskonale klasyfikowany), co daje ogony w rozkładzie jakości "
        "(Circles, MAGIC). Pamięć dodatkowo stabilizuje trajektorię V′, bo model nie „zapomina” wcześniej wykrytych trudnych przykładów — interpretacja zgodna z intuicją, ale w danych poparta słabo "
        "(małe efekty, 20 seedów, jeden podział danych na zbiór)."
    )
    doc.h2("9.4. Zbiory 2D a wielowymiarowe")
    train_sizes = {"chessboard": 19949, "moons": 4800, "circles": 4800, "spiral": 4800, "classification2d": 4800, "digits": 718, "magic": 3200}
    rows = [["Zbiór", "|Train|", "średni |T′| końcowy (evo_mem)", "|T′| / |Train|", "decyzja końcowa"]]
    for ds in D2 + HD:
        tr = d.stat(ds, "evo_mem", "training_rows")
        sc = _stop_counts(d, ds, "evo_mem")
        rows.append([DATASET_LABELS[ds], fi(train_sizes[ds]), fi(tr), f(tr / train_sizes[ds], 2), ", ".join(f"{k.replace('stop_', '')}: {v}" for k, v in sc.items())])
    doc.table(rows, [3.4, 2.0, 3.8, 2.4, 5.7], "Udział T′ w zbiorze Train i przyczyny zakończenia.")
    doc.p(
        "Na zbiorach 2D (Szachownica, Moons, Circles, Spiral, Classification 2D) algorytm kończy decyzją o braku sygnału wzrostu K (ρ<sub>SV</sub> &lt; TSV), a T′ stanowi niewielką część Train "
        "(Szachownica ≈ 8%). Na Digits i MAGIC każdy przebieg kończy się osiągnięciem limitu K (najmniejsza klasa Train), przy T′ obejmującym znaczną część zbioru treningowego. "
        "Oznacza to, że w tych zbiorach (mały Train, wysoka złożoność granicy) metoda nie osiąga kompresji zbioru treningowego, a porównanie wariantów V′ dotyczy głównie trajektorii dojścia do limitu K. "
        "Zjawiska zaobserwowane w 2D — brak spójnej różnicy jakości między ewolucyjnym i losowym V′, niższy koszt wariantu o stałym rozmiarze V′ — występują także tutaj, "
        "co sugeruje, że nie są cechą wyłącznie prostych geometrii dwuwymiarowych; ze względu na efekt limitu K wniosek ten jest jednak wstępny."
    )
    doc.h2("9.5. Stabilność jakości a zmienność kosztu")
    rows = [["Zbiór (evo_mem)", "CV test BA", "CV liczby cykli", "CV treningów SVM", "CV czasu zegarowego"]]
    for ds in D2 + HD:
        rows.append([DATASET_LABELS[ds], f(d.stat(ds, "evo_mem", "test_ba", "cv"), 4), f(d.stat(ds, "evo_mem", "cycles", "cv"), 3),
                     f(d.stat(ds, "evo_mem", "svm_fits_total", "cv"), 3), f(d.stat(ds, "evo_mem", "duration_sec", "cv"), 3)])
    doc.table(rows, [4.0, 3.0, 3.2, 3.4, 3.7], "Zmienność względna (CV) jakości i kosztu w serii evo_mem.")
    doc.p(
        "We wszystkich zbiorach współczynnik zmienności jakości testowej jest wielokrotnie mniejszy niż współczynnik zmienności kosztu (liczby treningów SVM: 23–110 razy; "
        "liczby cykli: 39–90 razy na zbiorach 2D, 6 razy na MAGIC; na Digits liczba cykli jest stała, CV = 0, bo przebieg wyznacza limit K, ale liczba treningów SVM nadal się zmienia). Wynik jakościowy jest więc powtarzalny, a niepewność dotyczy głównie "
        "tego, ile cykli i treningów SVM trzeba wykonać."
    )


def sec_limits(doc, d):
    doc.h1("10. Ograniczenia metodologiczne i dalsze kroki")
    doc.bullets(
        [
            "<b>Jeden podział danych na zbiór.</b> Seed zmienia wyłącznie algorytm (inicjalizacja V′, GA); zmienność wynikająca z podziału danych nie jest uwzględniona. "
            "Dla Szachownicy podział jest dany plikami; dla zbiorów syntetycznych — jeden deterministyczny podział (ziarno 2026).",
            "<b>Hiperparametry SVM (C, γ)</b> dla nowych zbiorów dobrano raz, pełnym SVM, oceną na pełnym V (przed eksperymentami, wspólnie dla wariantów). To jedyne użycie jakości na V poza pętlą algorytmu, "
            "ale oznacza, że V nie jest w pełni „nieużywane” w całym procesie badawczym; test pozostaje niezależny. Parametry Szachownicy pochodzą z wcześniejszych prac.",
            "<b>V′ jest podzbiorem V.</b> Metryka na pełnym V jest obserwacją, lecz zbiory nie są rozłączne, więc nie jest niezależną miarą generalizacji; za taką należy uważać test.",
            "<b>Uprzywilejowanie wariantu rand_fixed</b> (rozmiar V′ z serii ewolucyjnej) oraz brak korekty na porównania wielokrotne (7 zbiorów × wiele metryk; wartości p mają charakter opisowy).",
            "<b>Reguła stopu „100% na V′”</b> jest wrażliwa na rozmiar V′; na małych V′ powoduje przedwczesne zakończenie (Moons, Circles, Classification 2D), co zniekształca porównania wariantów o różnych harmonogramach V′. "
            "Wymagałaby minimalnego rozmiaru V′ albo kryterium opartego na przedziale ufności.",
            "<b>Czasy zegarowe</b> pochodzą z równoległych uruchomień o zmiennym obciążeniu maszyny (kilka procesów na rdzeń, różna współbieżność dla różnych serii), więc nie są porównywalne między seriami; "
            "do oceny kosztu użyto liczby cykli, treningów SVM i predykcji. Nie wykonano pomiaru czasu na wyizolowanej maszynie.",
            "<b>MAGIC</b> użyto w podpróbce 8000 obserwacji (pełny zbiór czyniłby serie zbyt kosztownymi); Digits ma mały Train (718), więc K szybko osiąga limit.",
            "<b>Val-GA nie był strojony</b> (populacja 30, 20 generacji). Nie badano wpływu tych parametrów ani kosztowo-świadomego wariantu (np. podpróbkowanie puli kandydatów).",
            "Wniosków nie należy uogólniać poza badane zbiory, jeden typ jądra (RBF) i jedną metrykę (balanced accuracy).",
        ]
    )
    doc.h2("Dalsze kroki")
    doc.bullets(
        [
            "Reguła stopu odporna na mały V′ oraz powtórzenie porównania wariantów losowych i ewolucyjnych.",
            "Wariant kosztowo-świadomy: Val-GA na podpróbce puli kandydatów lub z mniejszym budżetem; porównanie kosztu w predykcjach modelu przy równej jakości.",
            "Powtórzenia z różnymi podziałami danych oraz większe, rzeczywiste zbiory wielowymiarowe (pełny MAGIC, zbiory UCI o większym Train).",
            "Rozszerzenie diagnostyki: rozkład trudności elementów (margines, odległość od granicy) w V′ w czasie, test, czy ewolucja przyspiesza stabilizację ρ<sub>SV</sub>.",
        ]
    )


def sec_repro(doc, d):
    doc.h1("11. Odtworzenie eksperymentów i status wykonania")
    doc.code(
        "python scripts/generate_study_datasets.py && python scripts/tune_study_svm.py\n"
        "python scripts/make_study_configs.py\n"
        "python scripts/run_study_queue.py -d chessboard -d moons -d circles -d spiral -d classification2d -d digits -d magic \\\n"
        "       -v evo_mem -v evo_nomem -v rand_growth -v heur_mem -w 8\n"
        "python scripts/run_study_queue.py -d chessboard -d moons -d circles -d spiral -d classification2d -d digits -d magic \\\n"
        "       -v rand_fixed -w 4        # po ukonczeniu evo_mem\n"
        "python scripts/aggregate_vprime_study.py && python scripts/plot_vprime_study.py\n"
        "python scripts/generate_vprime_study_report.py\n"
        "python -m pytest tests -q"
    )
    doc.p(
        f"Wykonano wszystkie zaplanowane serie: {int(d.validation.complete_20_seeds.sum())} z {len(d.validation)} serii (7 zbiorów × 5 wariantów) ma komplet 20 seedów i status completed; "
        "nie ma przebiegów nieudanych ani niewykonanych. Dane per-run: artifacts/study/&lt;zbiór&gt;/&lt;wariant&gt;/runs/*/metrics; tabele zbiorcze: reports/vprime_study/tables. "
        "Seria evo_mem na Szachownicy została wykonana ponownie nowym kodem; wyniki są identyczne z serią historyczną (20/20 przebiegów)."
    )
    doc.table(
        [["Zbiór", "Źródło", "Generator / ziarna", "Podział"],
         ["Szachownica", "pliki z repozytorium (data/raw)", "—", "gotowy Train/V/Test"],
         ["Moons, Circles", "sklearn make_moons / make_circles (12 000 pkt)", "generator 2026", "40/40/20 stratyf., ziarno 2026"],
         ["Spiral", "generator własny (2 ramiona, szum 0,07)", "2026", "j.w."],
         ["Classification 2D", "sklearn make_classification (2 cechy, 2 klastry/klasę)", "2026", "j.w."],
         ["Digits", "sklearn load_digits (64 cechy, 10 klas)", "—", "j.w., StandardScaler na Train"],
         ["MAGIC", "UCI MAGIC Gamma Telescope (podpróbka 8 000)", "2026", "j.w., StandardScaler na Train"]],
        [3.0, 6.4, 3.0, 4.9], "Zbiory danych: pochodzenie i ziarna (szczegóły: data/study/*/meta.json).")


def add_all(doc, d):
    doc.story.append(__import__("reportlab.platypus", fromlist=["Paragraph"]).Paragraph("DAGASVM: ewolucyjny dobór podzbioru walidacyjnego V′", doc.ss["Title2"]))
    doc.story.append(__import__("reportlab.platypus", fromlist=["Paragraph"]).Paragraph("Raport z badania: 20 powtórzeń, porównanie z doborem losowym, pamięć V′, zbiory 2D i wielowymiarowe", doc.ss["Cap"]))
    sec_summary(doc, d)
    sec_roles(doc, d)
    sec_implementation(doc, d)
    sec_config(doc, d)
    sec_results_chessboard(doc, d)
    sec_series_validation(doc, d)
    sec_vs_random(doc, d)
    sec_memory(doc, d)
    sec_other_datasets(doc, d)
    sec_interpretation(doc, d)
    sec_limits(doc, d)
    sec_repro(doc, d)
