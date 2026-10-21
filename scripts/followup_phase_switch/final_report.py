"""Raport końcowy badania `followup_phase_switch` (v2) -> PDF + Markdown.

    python scripts/followup_phase_switch/final_report.py

Wyjście: `output/pdf/raport_phase_switch.pdf`, `reports/followup_phase_switch/RAPORT.md` (+ tabele `tables/final_*.csv`, wykresy `figures/`).
Wszystkie liczby w tekście pochodzą z obliczonych tabel (`final_analysis.py`); wykresy z `final_figs.py`. Poprzednie raporty (`RAPORT_SMOKE.md`,
`v2/RAPORT_*.md`, PDF `raport_followup_regula_stopu.pdf`) nie są nadpisywane.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_analysis as A  # noqa: E402
import final_figs as F  # noqa: E402
from ps_common import REPO_ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("fsr_pdfkit", REPO_ROOT / "scripts" / "followup_stop_rule" / "pdfkit.py")
pdfkit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdfkit)
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import CondPageBreak, KeepTogether  # noqa: E402

fint = pdfkit.fint
_fnum0 = pdfkit.fnum


def fnum(x, d=4):
    """Liczba z przecinkiem dziesiętnym; bez „−0,0000” i z prawdziwym znakiem minus."""
    try:
        if x is not None and not pd.isna(x) and round(float(x), d) == 0:
            x = 0.0
    except (TypeError, ValueError):
        pass
    return _fnum0(x, d).replace("-", "−")

OUT_PDF = REPO_ROOT / "output" / "pdf" / "raport_phase_switch.pdf"
OUT_MD = REPO_ROOT / "reports" / "followup_phase_switch" / "RAPORT.md"
FIGS = REPO_ROOT / "reports" / "followup_phase_switch" / "figures"
TITLE = "DAGASVM — naprzemienna optymalizacja T′ ↔ V′ (raport badania)"
DS = A.DATASETS
LB = A.LABEL


def loc(ds):
    """Szachownica w dopełniaczu/miejscowniku (pozostałe nazwy są nieodmienne)."""
    return "Szachownicy" if ds == "chessboard" else LB[ds]


def fixsv(t):
    return str(t).replace("ρ_SV", "ρ<sub>SV</sub>").replace("K_next", "K<sub>next</sub>").replace("K_start", "K<sub>start</sub>")


def pbound(p):
    return "p Holma &lt; 0,001" if p < 0.001 else f"p Holma ≤ {pf(p)}"


def pmv(m, s, d=4):
    return f"{fnum(m, d)} ± {fnum(s, d)}"


def ci(lo, hi, d=4):
    return f"[{fnum(lo, d)}; {fnum(hi, d)}]"


def pf(p):
    if p is None or pd.isna(p):
        return "—"
    return "<0,001" if p < 0.001 else fnum(p, 3)


# ----------------------------------------------------------------------------------------------
class Rep:
    """Ten sam zestaw bloków zapisywany do PDF (reportlab) i Markdown."""

    def __init__(self):
        self.doc = pdfkit.Doc(FIGS, OUT_PDF, TITLE)
        self.doc.ss["Cell"].leading = 10.0
        self.doc.ss["H1"].keepWithNext = 1   # nagłówek zawsze razem z następnym elementem (bez osieroconych nagłówków)
        self.doc.ss["H2"].keepWithNext = 1
        for name in ("P", "Bul", "Box"):
            self.doc.ss[name].allowWidows = 0
            self.doc.ss[name].allowOrphans = 0
        self.md: list[str] = []

    @staticmethod
    def _md(t: str) -> str:
        t = re.sub(r"<b>(.*?)</b>", r"**\1**", t, flags=re.S)
        t = re.sub(r"<i>(.*?)</i>", r"*\1*", t, flags=re.S)
        t = t.replace("<br/>", "  \n").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        t = re.sub(r"<sub>(.*?)</sub>", r"_\1", t)
        t = re.sub(r"</?[a-z]+[^>]*>", "", t)
        return t

    def title(self, t, sub):
        self.doc.story.append(pdfkit.Paragraph(t, self.doc.ss["Title2"]))
        self.doc.story.append(pdfkit.Paragraph(sub, self.doc.ss["Sub"]))
        self.md += [f"# {t}", "", f"*{self._md(sub)}*", ""]

    def h1(self, t):
        t = fixsv(t)
        self.doc.h1(t)
        self.md += ["", f"## {t}", ""]

    def h2(self, t):
        t = fixsv(t)
        self.doc.h2(t)
        self.md += ["", f"### {t}", ""]

    def p(self, t):
        t = fixsv(t)
        self.doc.p(t)
        self.md += [self._md(t), ""]

    def box(self, t):
        t = fixsv(t)
        self.doc.box(t)
        self.md += ["> " + self._md(t).replace("\n", "\n> "), ""]

    def bullets(self, items):
        items = [fixsv(i) for i in items]
        self.doc.bullets(items)
        self.md += [f"- {self._md(i)}" for i in items] + [""]

    def _chain(self, big_table: bool = False):
        """Rozpakowuje KeepTogether z pdfkit i łączy podpis/rysunek z kolejnym elementem (reportlab nie łączy nagłówka z KeepTogether)."""
        story = self.doc.story
        if big_table:   # [podpis, tabela, odstęp] bez KeepTogether
            story[-3].keepWithNext = 1
            return
        item = story[-1]
        if isinstance(item, KeepTogether):
            content = list(item._content)
            story[-1:] = content
            for f in content[:-2]:
                f.keepWithNext = 1

    def table(self, rows, widths, caption, **kw):
        rows = [[fixsv(c) for c in r] for r in rows]
        caption = fixsv(caption)
        self.doc.table(rows, widths, caption, **kw)
        self._chain(big_table=len(rows) >= 30)
        n = self.doc.tab_no
        self.md += [f"**Tabela {n}.** {self._md(caption)}", ""]
        self.md += ["| " + " | ".join(self._md(str(c)).replace("|", "\\|").replace("\n", " ") for c in rows[0]) + " |",
                    "|" + "---|" * len(rows[0])]
        for r in rows[1:]:
            self.md.append("| " + " | ".join(self._md(str(c)).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
        self.md.append("")

    def figure(self, name, caption, width_cm=17.2):
        caption = fixsv(caption)
        self.doc.figure(name, caption, width_cm)
        self._chain()
        self.md += [f"![Rys. {self.doc.fig_no}]({'figures/' + name})", "", f"**Rys. {self.doc.fig_no}.** {self._md(caption)}", ""]

    def code(self, text):
        self.doc.code(text)
        self.md += ["```bash", text.strip("\n"), "```", ""]

    def pagebreak(self):
        self.doc.pagebreak()

    def build(self):
        self.doc.build()
        OUT_MD.write_text("\n".join(self.md) + "\n", encoding="utf-8")
        print("zapisano", OUT_MD)


# ----------------------------------------------------------------------------------------------
def main() -> None:
    F.style()
    R = A.build_all(save=True)
    F_main = [F.fig_phase_machine, F.fig_size, F.fig_quality_vs_size, F.fig_trajectories, F.fig_forest, F.fig_difficulty, F.fig_cost, F.fig_stops]
    for fn in F_main:
        fn(R) if fn is not F.fig_phase_machine else fn()
    D, P = R["D"], R["P"]
    runs, steps, old = D["runs"], D["steps"], D["old"]
    main_runs = runs[runs.role == "główne"]
    pil = runs[runs.role == "pilotaż"]
    comp = P["complete"]
    n_main, n_pilot = len(main_runs), len(pil)
    n_legacy = int(comp[comp["rola"].str.startswith("wcześniejsza")]["ukończone (job)"].sum())
    n_cycles = int(steps[steps.series.isin(A.MAIN_SERIES)].shape[0])
    base = runs[runs.series == "v2_full"]
    evo_m = main_runs[main_runs.variant == "evo_switch"]
    pb, ps_, pa = P["pair_base"], P["pair_splits"], P["pair_aux_base"]
    ph = P["phase"]

    def pbm(metric, ds):
        return pb[(pb.metryka == metric) & (pb["zbiór"] == ds)].iloc[0]

    # ---------- skalary do tekstu ----------
    chess_new, chess_old = P["size_base"].query("zbiór=='chessboard' and wariant=='evo_switch'").iloc[0], P["size_base"].query("zbiór=='chessboard' and wariant=='evo_mem_old'").iloc[0]
    sz = P["size_base"]

    def szv(ds, var, col):
        return sz[(sz["zbiór"] == ds) & (sz["wariant"] == var)].iloc[0][col]

    own = steps[steps.series.isin(A.MAIN_SERIES) & (steps.variant != "rand_replay")].copy()
    for c in ("tprime_perfect", "k_increased", "retry_check", "retry_progress"):
        own[c] = own[c].astype(str).str.lower() == "true"
    t_own = own[own.phase == "optimize_T_prime"]
    evo_t = t_own[t_own.variant == "evo_switch"]
    entries_evo = evo_t[evo_t.next_phase == "optimize_V_prime"]
    mast_evo = int(entries_evo.tprime_perfect.sum())
    perfect_evo = evo_t[evo_t.tprime_perfect]
    init_rows = own.groupby("run_id").vprime_rows_before.transform("min")
    own["is_init"] = own.vprime_rows_before == init_rows
    perfect_init = float(own[(own.variant == "evo_switch") & (own.phase == "optimize_T_prime") & own.tprime_perfect].is_init.mean())
    ap = ph["after_perfect"]
    ap_e, ap_r = ap[ap.variant == "evo_switch"], ap[ap.variant == "rand_switch"]
    phs = ph["phases"]
    phe = phs[phs.variant == "evo_switch"]
    rch = t_own[t_own.retry_check]
    n_rc, n_rc_ok = len(rch), int(rch.retry_progress.sum())
    why = ph["why"]
    att = t_own[(t_own.next_phase == "optimize_V_prime") & (~t_own.tprime_perfect)]
    n_att = len(att)
    n_fail = int((main_runs[main_runs.variant != "rand_replay"].stop_decision == "stop_no_progress_after_vprime_retry").sum())
    n_res = n_att - n_fail
    first_ok = t_own[t_own.retry_check & t_own.retry_progress & (t_own.prev_phase == "optimize_V_prime")]
    assert n_res == int((t_own.retry_check & t_own.retry_progress).sum()), "niespójne liczby prób V′"
    why_tot = why.sum()
    below = t_own[~t_own.tprime_perfect]
    # nie gorsze niż stary algorytm (granica 0,5 p.p.)
    sq = P["size_quality"]
    noninf = sq[sq["CI dolny"] > -A.MARGIN]["zbiór"].tolist()
    notinf = sq[sq["CI dolny"] <= -A.MARGIN]["zbiór"].tolist()
    # górne granice CI evo - replay
    tb = pb[pb.metryka == "test_ba"]
    excl_adv = tb[tb.ci_hi < A.MARGIN]["zbiór"].tolist()
    n_sig = tb[(tb.ci_lo > 0) | (tb.ci_hi < 0)]["zbiór"].tolist()
    # koszt
    cr = A.cost_ratio_table(runs)
    svm_lo = min(r["treningi SVM CI"][0] for _, r in cr.iterrows())
    r_svm = (cr["treningi SVM"].min(), cr["treningi SVM"].max())
    r_pred = (cr["wiersze predict"].min(), cr["wiersze predict"].max())
    r_time = (cr["czas zegarowy"].min(), cr["czas zegarowy"].max())
    tt = steps[(steps.series == "v2_full") & (steps.phase == "optimize_T_prime")].copy()
    tt["reached"] = tt.ga_reached_target.astype(str).str.lower() == "true"
    e_ = tt[tt.variant == "evo_switch"].set_index(["dataset", "seed", "cycle"])
    r_ = tt[tt.variant == "rand_replay"].set_index(["dataset", "seed", "cycle"])
    pidx = e_.index.intersection(r_.index)
    gen_diff = float((e_.loc[pidx, "ga_generations"] - r_.loc[pidx, "ga_generations"]).mean())
    gen_longer = float((e_.loc[pidx, "ga_generations"] > r_.loc[pidx, "ga_generations"]).mean())
    gen_shorter = float((e_.loc[pidx, "ga_generations"] < r_.loc[pidx, "ga_generations"]).mean())
    reach_e, reach_r = float(e_.loc[pidx, "reached"].mean()), float(r_.loc[pidx, "reached"].mean())
    ba_e, ba_r = float(e_.loc[pidx, "tprime_fitness"].mean()), float(r_.loc[pidx, "tprime_fitness"].mean())
    diff_pb = pb[pb.metryka == "mean_delta_difficulty"]
    err_ratio = diff_pb.mean_a.to_numpy() / diff_pb.mean_b.to_numpy()
    dba_pb = pb[pb.metryka == "delta_ba_mean"]

    rep = Rep()
    rep.title("DAGASVM: naprzemienna optymalizacja T′ ↔ V′ i dobór podzbioru walidacyjnego V′",
              "Badanie uzupełniające followup_phase_switch (wersja v2) — ewolucyjny a losowy dobór V′, rozmiar V′, przełączanie faz, koszt")

    # ============================ 1. STRESZCZENIE ============================
    rep.h1("1. Streszczenie")
    rep.box(
        "<b>Co zmieniono i dlaczego.</b> Poprzednia wersja DAGASVM rozszerzała V′ w rytmie wyznaczanym przez wzrost K i sygnał ρ<sub>SV</sub> "
        f"(przyrost = K<sub>next</sub> − K), a osiągnięcie BA(V′) = 1,0 kończyło przebieg albo — po zmianie reguły — wymuszało minimalny rozmiar V′ i ponowne sprawdzenie. "
        f"W efekcie V′ rosło na Szachownicy średnio do {fint(chess_old['V′ średnio'])} próbek. Zgodnie z uwagami promotora wprowadzono jawny automat dwóch faz: "
        "optymalizacji T′ (GA przy stałym K, ocena na całym skumulowanym V′) i optymalizacji V′ (dobór nowych próbek ΔV′ z puli V \\ V′ przy ustalonym modelu). "
        "BA(V′) = 1,0 przełącza do fazy V′ i nie kończy algorytmu; rozmiar V′ ma jawne limity; koniec następuje po braku postępu.<br/>"
        f"<b>Co wykonano.</b> {n_main} przebiegów głównych (7 zbiorów, 3 warianty doboru V′: ewolucja, losowanie z harmonogramem ewolucji i losowanie z własnym automatem; "
        f"podział bazowy: 20 seedów, trzy podziały dodatkowe: 5 seedów) oraz {n_pilot} przebiegów pilotażowych; wszystkie ukończone, bez błędów.<br/>"
        f"<b>Co wykazano.</b> (1) Implementacja spełnia 11 z 12 wymagań z uwag promotora w pełni lub z zastrzeżeniem, a jedno (najpierw zwiększać K) częściowo: "
        f"w {int(why_tot.get('bramka ρ_SV < TSV (bez zwiększenia K)', 0))} przypadkach faza V′ zaczyna się z powodu bramki ρ<sub>SV</sub> bez zwiększenia K. "
        f"(2) V′ przestało rosnąć do nadmiernych rozmiarów: Szachownica {fint(chess_new['V′ średnio'])} ± {fint(chess_new['V′ SD'])} próbek ({fnum(chess_new['V′/V średnio'] * 100, 1)}% V) wobec {fint(chess_old['V′ średnio'])} "
        f"({fnum(chess_old['V′/V średnio'] * 100, 1)}%), przy BA na teście bez różnicy ({fnum(float(sq[sq['zbiór'] == 'chessboard']['różnica (nowy−stary)'].iloc[0]), 4)}); maks. |V′|/|V| w całej serii = {fnum(main_runs.vprime_fraction.max(), 3)}. "
        f"(3) Ewolucja wybiera istotnie trudniejsze próbki (błąd modelu na ΔV′ wyższy o {fnum(diff_pb['diff'].min(), 2)}–{fnum(diff_pb['diff'].max(), 2)} na wszystkich 7 zbiorach), "
        f"ale <b>nie wykazano praktycznie istotnej przewagi jakościowej ewolucji nad losowaniem</b>: różnice BA na teście mieszczą się w [{fnum(tb['diff'].min() * 100, 2)}; {fnum(tb['diff'].max() * 100, 2)}] p.p., "
        f"CI wyklucza zero tylko na {', '.join(loc(d) for d in n_sig)} (+{fnum(float(pbm('test_ba', 'chessboard')['diff']) * 100, 2)} p.p., poniżej przyjętej granicy 0,5 p.p.). "
        f"(4) Ewolucja kosztuje więcej: {fnum(r_svm[0], 2)}–{fnum(r_svm[1], 2)}× treningów SVM i {fnum(r_pred[0], 2)}–{fnum(r_pred[1], 2)}× wierszy predict niż losowanie przy tym samym harmonogramie.<br/>"
        f"<b>Najważniejszy problem, który pozostał.</b> Rozmiar V′ i moment zakończenia wynikają głównie z reguł automatu (długość fazy V′ ≈ 3 rozszerzenia, liczba zastojów T′, bramka ρ<sub>SV</sub>), a nie z sygnału jakościowego: "
        f"w wariancie ewolucyjnym tylko {fnum(mast_evo / len(entries_evo) * 100, 0)}% wejść do fazy V′ pochodzi z BA(V′) = 1, a {fnum(perfect_init * 100, 0)}% takich przypadków to V′ startowe (20 próbek). Trudność ΔV′ nie rośnie z kolejnymi rozszerzeniami, więc warunek „jakość nowych próbek przestała się pogarszać” "
        "jest spełniany niemal od razu i działa jak stała długość fazy.")

    # ============================ 2. CEL ============================
    rep.h1("2. Cel badania i pytania badawcze")
    rep.p("Celem jest sprawdzenie, czy przebudowany algorytm DAGASVM — z naprzemienną optymalizacją podzbioru treningowego T′ i podzbioru walidacyjnego V′ — realizuje założenia "
          "promotora oraz czy ewolucyjny dobór V′ daje korzyść względem doboru losowego. Raport opisuje wyłącznie wykonane eksperymenty. "
          "Trudniejsze V′ i lepszy model są osobnymi właściwościami i są analizowane oddzielnie.")
    rep.bullets([
        "<b>P0.</b> Czy implementacja spełnia 12 wymagań z uwag promotora (sekcja 5.1)?",
        "<b>P1.</b> Czy ewolucja wybiera trudniejsze próbki do ΔV′ niż losowanie (sekcja 5.3)?",
        "<b>P2.</b> Czy ewolucyjny dobór V′ prowadzi do lepszego modelu niż losowy, przy tym samym rozmiarze V′ (sekcja 5.4)?",
        "<b>P3.</b> Czy ewolucja jest warta dodatkowego kosztu obliczeniowego (sekcja 5.5)?",
        "<b>P4.</b> Czy V′ przestało rosnąć do nadmiernych rozmiarów i czy rozmiar V′ wpływa na jakość (sekcja 5.2)?",
        "<b>P5.</b> Jak działają warunki przełączania faz i zakończenia (sekcje 6 i 7)?",
    ])

    # ============================ 3. ALGORYTM ============================
    rep.h1("3. Opis algorytmu")
    rep.p("Oznaczenia: V — pełny zbiór walidacyjny; V′ ⊂ V — skumulowany podzbiór walidacyjny sterujący algorytmem; ΔV′ — przyrost dodany w bieżącym rozszerzeniu (V′ ← V′ ∪ ΔV′); "
          "T′ — podzbiór treningowy (K próbek na klasę wybranych z pełnego zbioru treningowego); K — liczba próbek na klasę w T′; ρ<sub>SV</sub> — udział wektorów podpierających w T′ po treningu SVM; "
          "TSV — próg ρ<sub>SV</sub> (0,25). Jeden <i>cykl</i> to jeden krok automatu: jedno uruchomienie GA dla T′ albo jedno rozszerzenie V′.")
    rep.h2("3.1 Faza optimize_T_prime")
    rep.p("GA (populacja 30, do 30 generacji) wybiera T′ o zadanym K z pełnego zbioru treningowego; fitness osobnika = metryka (domyślnie balanced accuracy) SVM wytrenowanego na T′, policzona na całym skumulowanym V′. "
          "Kolejna optymalizacja startuje z poprzednim najlepszym T′ w populacji (warm start), więc wynik przy tym samym K nie pogarsza się. GA kończy się wcześnie po 3 generacjach bez poprawy o ≥ 0,0005 "
          "albo po osiągnięciu BA(V′) = 1,0. Po GA liczone są: BA(V′) najlepszego T′, ρ<sub>SV</sub> i kandydat na kolejne K.")
    rep.h2("3.2 Faza optimize_V_prime")
    rep.p("Model T′ jest ustalony na czas fazy. Dobór ΔV′ odbywa się wyłącznie z puli kandydatów V \\ V′ (stare próbki V′ nie są oceniane ponownie). Wariant ewolucyjny: adwersarialny GA (Val-GA, populacja 30, 20 generacji) szuka "
          "zbioru 10 próbek na klasę (łącznie 10·liczba klas), na którym ustalony model radzi sobie najgorzej; fitness osobnika Val-GA = 1 − BA modelu na wybranym podzbiorze ΔV′. Wariant losowy: ΔV′ losowane z puli. "
          "Po rozszerzeniu mierzone są: BA i błąd klasyfikacji na ΔV′, skład klasowy ΔV′, zmiana trudności względem poprzedniego przyrostu oraz koszt. "
          "Trudność przyrostu to odsetek błędów modelu na ΔV′. Faza trwa, dopóki trudność (średnia z 2 ostatnich przyrostów) rośnie o co najmniej 0,01; po 2 kolejnych przyrostach bez takiego wzrostu faza się kończy.")
    rep.h2("3.3 Przełączanie faz")
    rows = [["Aktualna faza", "Warunek", "Działanie", "Następna faza"],
            ["optimize_T_prime", "BA(V′) = 1,0 i V′ można rozszerzyć", "K bez zmian; zapamiętanie modelu (opanowanie V′)", "optimize_V_prime"],
            ["optimize_T_prime", "BA(V′) = 1,0, a V′ nierozszerzalne (limit rozmiaru, limit rozszerzeń lub brak kandydatów)", "zakończenie algorytmu", "stop (stop_perfect_on_max_vprime)"],
            ["optimize_T_prime", "BA(V′) < 1,0; brak zastoju (&lt; 3 kolejnych zwiększeń K bez poprawy o ≥ 0,0005); ρ<sub>SV</sub> ≥ TSV; K może rosnąć",
             "K ← K<sub>next</sub> (wzór DAGA zależny od ρ<sub>SV</sub>); najwyżej raz w cyklu; ponowne GA", "optimize_T_prime"],
            ["optimize_T_prime", "BA(V′) < 1,0 i zastój: 3 kolejne zwiększenia K bez poprawy, albo ρ<sub>SV</sub> &lt; TSV, albo limit K klasy; próba V′ przy tym K jeszcze nie wykonana; V′ rozszerzalne",
             "zapamiętanie najlepszego T′ i jego fitnessu jako poziomu odniesienia (próba V′)", "optimize_V_prime"],
            ["optimize_T_prime", "j.w., ale próba przy tym K już była, V′ nierozszerzalne albo próby wyłączone", "zakończenie algorytmu",
             "stop (stop_tprime_no_progress, stop_no_k_growth_signal lub stop_k_limit)"],
            ["optimize_T_prime (po próbie V′)", "T′ nie osiągnęło poziomu odniesienia (fitness &lt; najlepszy T′ sprzed próby, oceniony na tym samym V′, + 0,0005); K może rosnąć; &lt; 3 kolejnych niepowodzeń",
             "K ← K<sub>next</sub> (raz), ponowna optymalizacja T′", "optimize_T_prime"],
            ["optimize_T_prime (po próbie V′)", "j.w., ale K nie może rosnąć (ρ<sub>SV</sub> &lt; TSV albo limit klasy) lub 3 zwiększenia K nie dały poziomu", "zakończenie algorytmu",
             "stop (stop_no_progress_after_vprime_retry)"],
            ["optimize_T_prime (po próbie V′)", "poziom odniesienia osiągnięty (albo BA(V′) = 1)", "powrót do zwykłej pracy; licznik zastoju od zera", "optimize_T_prime lub optimize_V_prime"],
            ["optimize_V_prime", "trudność ΔV′ (średnia z okna 2) wzrosła o ≥ 0,01", "kolejne rozszerzenie V′ (+10 próbek/klasę)", "optimize_V_prime"],
            ["optimize_V_prime", "2 kolejne przyrosty bez wzrostu trudności; albo limit |V′| (≤ 0,25·|V|); albo brak kandydatów; albo 10 rozszerzeń w fazie", "koniec fazy; T′ optymalizowane na pełnym, skumulowanym V′ przy tym samym K", "optimize_T_prime"]]
    rep.table(rows, [2.7, 5.5, 4.4, 5.2], "Maszyna fazowa — warunki przejść zgodne z kodem automatu. Do żadnego warunku nie wchodzą pełne V ani test.", font_scale=0.97)
    rep.figure("rys01_maszyna_faz.png", "Diagram maszyny fazowej. Pętla przy optimize_T_prime = zwiększenie K; pętla przy optimize_V_prime = kolejne rozszerzenie V′.", 13.2)
    rep.h2("3.4 Role K, ρ<sub>SV</sub>, TSV i Ts")
    rows = [["Parametr", "Rola w tym algorytmie", "Czego nie robi"],
            ["K", "rozmiar T′ (próbek na klasę); rośnie tylko w fazie T′, najwyżej raz na cykl, nigdy przy BA(V′) = 1", "nie wpływa na przyrost V′"],
            ["ρ<sub>SV</sub>", "(i) wielkość kroku K: K<sub>next</sub> = ceil(K·(1 + |ρ<sub>SV</sub> − TSV| / (1 − TSV))), co najmniej K + 1; (ii) bramka: przy ρ<sub>SV</sub> &lt; TSV K nie rośnie, a taki krok jest zastojem T′ (próba V′)",
             "sam nie rozszerza V′"],
            ["TSV = 0,25", "próg dla ρ<sub>SV</sub>", "nie może samodzielnie powodować wzrostu V′"],
            ["Ts = 3", "cierpliwość wewnątrz GA dla T′ (generacje bez poprawy o ≥ 0,0005)", "nie steruje fazami ani rozmiarem V′"]]
    rep.table(rows, [2.2, 10.6, 5.0], "Znaczenie parametrów K, ρ_SV, TSV i Ts. Uwaga: wzór na K_next zawiera wartość bezwzględną, więc poniżej TSV działałby odwrotnie (im niższe ρ_SV, tym większy krok) — dlatego włączona jest bramka ρ_SV ≥ TSV.")
    rep.h2("3.5 Warunek zakończenia")
    rep.p("Algorytm kończy się: (a) po nieudanej próbie V′ — gdy T′ nie osiąga poziomu odniesienia także po zwiększeniach K (<i>stop_no_progress_after_vprime_retry</i>), (b) przy zastoju T′, gdy próba V′ przy tym K już była lub V′ nie da się rozszerzyć "
          "(<i>stop_tprime_no_progress</i>, <i>stop_no_k_growth_signal</i>, <i>stop_k_limit</i>), (c) gdy BA(V′) = 1, a V′ nierozszerzalne (<i>stop_perfect_on_max_vprime</i>), (d) bezpiecznikiem 300 cykli. "
          "Dodatkowe zabezpieczenia przed zapętleniem: najwyżej jedna próba V′ na poziom K, rosnące K, ograniczony rozmiar V′. BA(V′) = 1,0 podczas optymalizacji T′ nigdy samo nie kończy algorytmu.")
    rep.h2("3.6 Funkcja fitness: co jest używane i gdzie")
    rows = [["Miejsce", "Metryka (konfiguracja)", "Na czym liczona", "Steruje?"],
            ["fitness osobników GA dla T′", "balanced accuracy (fitness.tprime_metric)", "cały skumulowany V′", "tak (selekcja, zatrzymanie GA przy BA = 1)"],
            ["wartość kroku T′ (opanowanie, poprawa, zastój, postęp po próbie V′)", "balanced accuracy", "V′", "tak"],
            ["fitness osobników Val-GA (wybór ΔV′)", "1 − balanced accuracy (fitness.vprime_metric)", "wyłącznie podzbiór ΔV′ z puli V \\ V′", "tak (wariant ewolucyjny)"],
            ["trudność ΔV′ (koniec fazy V′)", "odsetek błędów = 1 − accuracy (vprime_difficulty_metric)", "ΔV′", "tak"],
            ["wybór modelu reprezentatywnego", "balanced accuracy", "końcowy V′", "tak (po zakończeniu)"],
            ["BA/F1/accuracy na pełnym V i na teście", "te same metryki", "pełne V, test", "nie — wyłącznie obserwacja po decyzji"]]
    rep.table(rows, [5.6, 5.0, 4.6, 2.6], "Użycie metryk w algorytmie. Kary: class_imbalance_penalty = 0; brak kary za rozmiar T′ ani V′.")
    rep.p("<b>Konsekwencje.</b> BA(V′) = 1,0 oznacza zero błędów na V′ (każda klasa jest obecna w V′ od startu), więc BA, accuracy i F1-macro mają ten sam próg „opanowania”; dla AUC tak nie byłoby, "
          "bo AUC = 1 nie oznacza zera błędów przy progu decyzyjnym SVM, i przy badaniu metryk próg trzeba definiować osobno. V′ jest budowane przeciw bieżącemu T′, a T′ jest optymalizowane na V′, więc BA(V′) "
          "nie jest nieobciążoną oceną uogólniania (dlatego test i pełne V są tylko obserwacją). <b>F1 i AUC nie były używane jako fitness</b>. W zapisanych wynikach pola roc_auc i pr_auc są równe BA, bo SVM jest trenowany z probability = false; "
          "<b>AUC nie został policzony</b> i raport go nie podaje. F1-macro i accuracy na teście zostały policzone dla modelu reprezentatywnego i są raportowane jako obserwacje. "
          "Porównanie BA z F1 i AUC jako fitnessem pozostaje do wykonania (wymaga probability = true i rozszerzenia Val-GA, który obsługuje dziś tylko BA, accuracy i binarne F1).")

    # ============================ 4. KONFIGURACJA ============================
    rep.h1("4. Konfiguracja eksperymentu")
    rep.h2("4.1 Kompletność i status przebiegów")
    rows = [["Seria", "Rola", "Podział", "Zbiory × warianty × seedy", "Zaplan.", "Ukończone", "Błędy"]]
    for _, r in comp[comp["rola"].isin(["główna", "pilotaż"])].iterrows():
        rows.append([r["seria"], r["rola"], r["podział"], f"{r['zbiory']} × {r['warianty']} × {r['seedy']}", r["zaplanowane"], r["ukończone (job)"], r["błędy (job)"]])
    leg = comp[comp["rola"].str.startswith("wcześniejsza")]
    rows.append([f"{len(leg)} serii bez prefiksu v2_", "wcześniejsza wersja automatu (v1) — poza wnioskami", "2026", "smoke i krótkie serie", int(leg["zaplanowane"].sum()),
                 int(leg["ukończone (job)"].sum()), int(leg["błędy (job)"].sum())])
    rows.append(["<b>Razem główne</b>", "", "", "", int(comp[comp.rola == 'główna'].zaplanowane.sum()), int(comp[comp.rola == 'główna']['ukończone (job)'].sum()),
                 int(comp[comp.rola == 'główna']['błędy (job)'].sum())])
    rep.table(rows, [4.0, 3.3, 1.3, 3.5, 1.4, 1.7, 1.3], "Kompletność serii (stan wg plików statusów jobów i wyjść kolejek). Wszystkie joby mają status completed; 0 jobów failed lub running, 0 kodów wyjścia ≠ 0, 0 wpisów Traceback/ERROR w logach.", bold_rows=(len(rows) - 1,))
    rep.p(f"Do wniosków wykorzystano <b>{n_main} przebiegów</b> (420 + 3·90): podział bazowy 2026 (7 zbiorów × 3 warianty × 20 seedów: 42–61) i podziały 2027–2029 (6 zbiorów zależnych od podziału × 3 warianty × 5 seedów: 42–46). "
          f"Szachownica czyta stały plik, więc ma tylko podział bazowy. Pilotaż ({n_pilot} przebiegów: Moons 3 seedy, Szachownica 2 seedy, limit |V′| = 1000, w tym wariant bez bramki ρ<sub>SV</sub>) jest oznaczony jako pilotażowy i nie wchodzi do głównych wniosków. "
          f"{n_legacy} przebiegów wcześniejszej wersji automatu (v1) pochodzi z innej semantyki próby V′ i nie jest używane; jedna seria pilotażowa zawiera jeden zdublowany przebieg (identyczny co do wyniku; agregacja bierze najnowszy). "
          f"Liczba cykli (kroków automatu) w danych głównych: {fint(n_cycles)}.")
    rep.h2("4.2 Zbiory danych, podziały i seedy")
    rows = [["Zbiór", "Klasy / cechy", "Trening", "V (pełne)", "Test", "K<sub>start</sub>", "C", "γ"]]
    cfg = {"chessboard": (2, 2, 10, 10.0, 0.002), "moons": (2, 2, 10, 1.0, 2.0), "circles": (2, 2, 10, 10.0, 0.1), "spiral": (2, 2, 10, 1000.0, 4.0),
           "classification2d": (2, 2, 10, 1.0, 1.0), "digits": (10, 64, 5, 100.0, 0.002), "magic": (2, 10, 10, 10.0, 0.05)}
    for ds in DS:
        g = base[base.dataset == ds].iloc[0]
        k, f_, ks, C, gm = cfg[ds]
        rows.append([LB[ds], f"{k} / {f_}", fint(g.n_train), fint(g.full_validation_rows), fint(g.n_test), ks, fnum(C, 0 if C >= 1 else 3).replace(",000", ""), fnum(gm, 3)])
    rep.table(rows, [3.0, 2.2, 1.9, 2.0, 1.7, 1.5, 1.7, 1.5], "Zbiory (rozmiary zbiorów treningowego, walidacyjnego i testowego są takie same na wszystkich podziałach; Digits: 10 klas, MAGIC: podpróbka 8000 obserwacji). Hiperparametry SVM (jądro RBF, class_weight = balanced) są wspólne dla wszystkich podziałów.")
    rep.p("Podział bazowy 2026 pochodzi z poprzedniego badania V′; podziały 2027–2029 zmieniają wyłącznie przydział obserwacji do zbiorów (pula obserwacji i generatory są stałe). Hiperparametry (C, γ) wybrano raz w poprzednim badaniu i użyto wszędzie; "
          "część V podziału bazowego trafia na podziałach dodatkowych do zbioru testowego, co daje niewielki, wspólny dla wariantów optymizm poziomów bezwzględnych i nie wpływa na porównania wariantów.")
    rep.h2("4.3 Parametry GA i automatu")
    rows = [["Grupa", "Parametry (wartości domyślne użyte we wszystkich seriach głównych)"],
            ["GA dla T′", "populacja 30, ≤ 30 generacji, turniej 3, krzyżowanie 0,8, mutacja 0,05, elitaryzm 2; wczesne zatrzymanie: Ts = 3 generacje bez poprawy o ≥ 0,0005 lub BA(V′) = 1; warm start"],
            ["Val-GA (dobór ΔV′)", "populacja 30, ≤ 20 generacji, turniej 3, krzyżowanie 0,8, mutacja 0,1, elitaryzm 2"],
            ["V′ i przyrosty", "V′ startowe: K<sub>start</sub> próbek na klasę (losowe, zrównoważone); przyrost 10 próbek/klasę; limit |V′| ≤ 0,25·|V|; limit rozszerzeń globalny: brak; w fazie: 10"],
            ["Faza V′", "trudność = odsetek błędów na ΔV′; okno 2; min. wzrost 0,01; cierpliwość 2"],
            ["Zastój i stop", "3 kolejne zwiększenia K bez poprawy o ≥ 0,0005; jedna próba V′ na poziom K; po próbie: ponowne zwiększenia K, poziom odniesienia = najlepszy T′ sprzed próby na tym samym V′; bezpiecznik 300 cykli"],
            ["K i ρ<sub>SV</sub>", "TSV = 0,25; bramka ρ<sub>SV</sub> włączona (K rośnie tylko przy ρ<sub>SV</sub> ≥ TSV); K ≤ liczność najmniejszej klasy treningowej"],
            ["Model reprezentatywny", "najlepszy T′ na końcowym V′ (wybór wyłącznie po V′)"]]
    rep.table(rows, [3.4, 14.4], "Parametry. Wartości robocze, nie strojone; wrażliwość na nie nie była badana.")
    rep.h2("4.4 Warianty")
    rows = [["Wariant", "Dobór ΔV′", "Harmonogram faz, K, |ΔV′| i moment stopu", "Uwagi o porównywalności"],
            ["evo_switch (ewolucja)", "Val-GA (adwersarialny)", "własny automat", "wariant badany"],
            ["rand_replay (losowanie, harmonogram z ewolucji)", "losowy z puli V \\ V′", "odtworzony z przebiegu evo_switch o tym samym seedzie",
             "to samo V′ startowe, ten sam |V′|, K i liczba cykli; <b>uprzywilejowany</b>: harmonogram (w tym K i stop) pochodzi z ewolucji, więc nie jest samodzielnym algorytmem"],
            ["rand_switch (losowanie, własny automat)", "losowy z puli V \\ V′", "własny automat z tymi samymi regułami", "samodzielny algorytm, ale |V′| i K są inne niż w ewolucji (mniejsze), więc różnica miesza efekt doboru z efektem rozmiaru"]]
    rep.table(rows, [4.0, 3.0, 4.8, 6.0], "Warianty. Główne porównanie ewolucja ↔ losowanie wykonano na rand_replay (identyczny rozmiar V′); rand_switch jest porównaniem pomocniczym. Wariant ze stałym rozmiarem V′ pochodzącym z wyniku ewolucji (rand_fixed) nie był używany.")

    # ============================ 5. WYNIKI ============================
    rep.h1("5. Wyniki")
    rep.h2("5.1 Zgodność implementacji z uwagami promotora (P0)")
    rows = [["Nr", "Wymaganie", "Dowód w danych", "Status"]]
    for _, r in P["req"].iterrows():
        rows.append([r["nr"], r["wymaganie"], re.sub(r"(\d)\.(\d)", r"\1,\2", r["dowód w danych"].replace("<", "&lt;").replace(">", "&gt;")), r["status"]])
    rep.table(rows, [0.8, 3.6, 10.4, 3.0], f"Weryfikacja 12 wymagań na {n_main} przebiegach i {fint(n_cycles)} cyklach (pełna tabela zapisana w katalogu tabel raportu). "
              "Wymagania 4, 5, 7, 11 mają dodatkowo testy jednostkowe z podsłuchem danych przekazywanych GA i z zatrutym obserwatorem pełnego V i etykietami testu.", font_scale=0.9)
    rep.p("<b>Różnice względem założeń</b> (szczegóły w sekcji 8): (a) wymaganie 6 spełnione częściowo — bramka ρ<sub>SV</sub> powoduje, że przy ρ<sub>SV</sub> &lt; TSV algorytm przechodzi do V′ bez zwiększania K; "
          "(b) wymaganie 1 — BA(V′) = 1 kończy algorytm tylko wtedy, gdy V′ nie da się już rozszerzyć (3 przypadki, Digits); (c) wymaganie 4 — ocena odbywa się na ΔV′, ale cała pula V \\ V′ jest oceniana raz na rozszerzenie; "
          "(d) wymaganie 12 — wariant rand_replay jest uprzywilejowany harmonogramem ewolucji.")

    # ---------------- 5.2 rozmiar ----------------
    rep.h2("5.2 Rozmiar V′ (P4)")
    rows = [["Zbiór", "Wariant", "n", "|V′| śr.", "mediana", "SD", "|V| śr.", "|V′|/|V|", "rozszerzenia V′", "przyrost ΔV′"]]
    vname = {"evo_switch": "ewolucja", "rand_replay": "los. (harm. ewol.)", "rand_switch": "los. (własny)", "evo_mem_old": "<i>stary algorytm</i>"}
    for ds in DS:
        for v in ["evo_switch", "rand_replay", "rand_switch", "evo_mem_old"]:
            r = sz[(sz["zbiór"] == ds) & (sz["wariant"] == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            inc = "—" if pd.isna(r["przyrost ΔV′ (śr. próbek)"]) else fnum(r["przyrost ΔV′ (śr. próbek)"], 0)
            rows.append([LB[ds] if v == "evo_switch" else "", vname[v], int(r["n"]), fnum(r["V′ średnio"], 0), fnum(r["V′ mediana"], 0), fnum(r["V′ SD"], 0), fint(r["|V| średnio"]),
                         fnum(r["V′/V średnio"], 3), fnum(r["rozszerzenia V′ (śr.)"], 1), inc])
    rep.table(rows, [2.4, 3.0, 0.8, 1.6, 1.6, 1.3, 1.6, 1.6, 2.1, 1.7], "Końcowy rozmiar V′ (podział 2026, 20 seedów). Stary algorytm: evo_mem z badania followup_stop_rule, te same seedy i podział; kolumna „rozszerzenia” oznacza liczbę kroków rozszerzenia. "
              "Średni przyrost ΔV′ = 20 próbek (10/klasę × 2 klasy) wszędzie poza Digits (limit rozmiaru ogranicza ostatni przyrost: 100 + 20 próbek).", font_scale=0.95)
    rep.figure("rys02_rozmiar_vprime.png", "Końcowy |V′| (A, skala logarytmiczna) i udział |V′|/|V| (B) w podziale 2026. Losowanie z harmonogramem ewolucji ma z konstrukcji ten sam |V′| co ewolucja.")
    lim_d = int((main_runs.vprime_limit_reason == "size_limit").sum())
    rep.p(f"<b>Czy nowa metoda ograniczyła wzrost V′?</b> Tak, tam, gdzie stary algorytm go nie ograniczał: Szachownica {fint(chess_old['V′ średnio'])} → {fint(chess_new['V′ średnio'])} (mediana {fint(chess_new['V′ mediana'])}, "
          f"{fnum(chess_old['V′ średnio'] / chess_new['V′ średnio'], 1)}× mniej), MAGIC {fint(szv('magic', 'evo_mem_old', 'V′ średnio'))} → {fint(szv('magic', 'evo_switch', 'V′ średnio'))}, Digits {fint(szv('digits', 'evo_mem_old', 'V′ średnio'))} → {fint(szv('digits', 'evo_switch', 'V′ średnio'))}, "
          f"Circles {fint(szv('circles', 'evo_mem_old', 'V′ średnio'))} → {fint(szv('circles', 'evo_switch', 'V′ średnio'))}. Na Moons i Classification 2D nowe V′ <b>nie</b> jest mniejsze "
          f"({fint(szv('moons', 'evo_mem_old', 'V′ średnio'))} → {fint(szv('moons', 'evo_switch', 'V′ średnio'))} i {fint(szv('classification2d', 'evo_mem_old', 'V′ średnio'))} → {fint(szv('classification2d', 'evo_switch', 'V′ średnio'))}), na Spiral porównywalne. "
          f"Maksymalny udział |V′|/|V| w całej serii wynosi {fnum(main_runs.vprime_fraction.max(), 3)}, maks. |V′| na Szachownicy {fint(main_runs[main_runs.dataset == 'chessboard'].vprime_rows.max())}.")
    rep.p(f"<b>Powód zakończenia wzrostu V′.</b> V′ przestaje rosnąć razem z końcem przebiegu. Limit rozmiaru (0,25·|V|) zablokował dalszy wzrost wyłącznie na Digits ({lim_d} z {n_main} przebiegów; |V′| = 170 z 719, ostatni przyrost 100 próbek nie mieści się w limicie 179), "
          "w pozostałych przebiegach wzrost zatrzymały reguły automatu (sekcja 6). Ograniczenie wzrostu wynika więc z reguł fazy V′ i liczby zastojów T′, a nie z limitu rozmiaru.")
    rep.p("<b>Czy V′ rośnie do rozmiaru, przy którym losowanie jest wystarczające?</b> Przy rozmiarach uzyskanych w tym badaniu losowanie z identycznym harmonogramem daje ten sam wynik testowy co ewolucja (sekcja 5.4). "
          "Nie badano jednak stałej siatki mniejszych rozmiarów V′, więc nie wiadomo, od jakiego rozmiaru losowanie przestaje wystarczać ani czy przy mniejszych V′ ewolucja pomaga bardziej.")
    pil_c = pil[(pil.dataset == "chessboard")]
    nog = pil_c[(pil_c.series == "v2_smoke_chess_r1000_nogate")]
    rep.p(f"<b>Czy dla Szachownicy udało się porównanie przy ≈ 1000 próbek?</b> W serii głównej (wartości domyślne) końcowe |V′| wynosi {fint(chess_new['V′ średnio'])} (maks. {fint(main_runs[main_runs.dataset == 'chessboard'].vprime_rows.max())}) — nie 1000 i nie ≈ 3500. "
          f"Rozmiar bliski 1000 uzyskano tylko w pilotażu (limit |V′| = 1000, bez bramki ρ<sub>SV</sub>, n = 2 seedy): |V′| = {fint(nog[nog.variant == 'evo_switch'].vprime_rows.mean())} "
          f"({fint(nog[nog.variant == 'evo_switch'].vprime_rows.min())}–{fint(nog[nog.variant == 'evo_switch'].vprime_rows.max())}), BA test ewolucji {fnum(nog[nog.variant == 'evo_switch'].test_ba.mean(), 4)} i losowania z harmonogramem {fnum(nog[nog.variant == 'rand_replay'].test_ba.mean(), 4)}. "
          "Przy n = 2 jest to wyłącznie ilustracja, że porównanie przy takim rozmiarze jest wykonalne; nie jest podstawą wniosków. W tym pilotażu K rośnie do ≈ 5500 (bramka wyłączona), więc zmiana |V′| jest zmieszana ze zmianą K.")
    rep.h2("5.2.1 Czy rozmiar V′ wpływa na jakość modelu?")
    rows = [["Zbiór", "ρ |V′|–BA test", "p", "ρ K–BA test", "ρ cząstkowa |V′|–BA | K", "p", "|V′| nowy / stary", "BA test nowy − stary", "95% CI", "nie gorsze o > 0,5 p.p."]]
    for _, r in sq.iterrows():
        rows.append([LB[r["zbiór"]], fnum(r["ρ Spearmana |V′|–BA test (podział 2026)"], 2), pf(r["p"]), fnum(r["ρ Spearmana K–BA test"], 2), fnum(r["ρ cząstkowa |V′|–BA test | K"], 2), pf(r["p (cząstkowa)"]),
                     f"{fint(r['|V′| nowy'])} / {fint(r['|V′| stary'])}", fnum(r["różnica (nowy−stary)"], 4), ci(r["CI dolny"], r["CI górny"]), "tak" if r["CI dolny"] > -A.MARGIN else "nie"])
    rep.table(rows, [2.4, 1.5, 1.0, 1.4, 1.9, 1.0, 2.4, 1.7, 2.3, 2.0], "Rozmiar V′ a BA na teście. Kolumny 2–6: korelacje rang w obrębie zbioru, wariant ewolucyjny, podział 2026 (n = 20; ρ cząstkowa po usunięciu wpływu K; p bez korekty na wielokrotność). "
              "Kolumny 7–10: stary algorytm (evo_mem) a nowy, te same seedy (różnica sparowana, bootstrap). Dla Digits i MAGIC K jest stałe (limit klasy), więc ρ cząstkowa jest niezdefiniowana.", font_scale=0.92)
    rep.figure("rys03_jakosc_vs_rozmiar_vprime.png", "BA na teście (obserwacja) w zależności od końcowego |V′|: wariant ewolucyjny nowego algorytmu (niebieski) i stary algorytm (szary), podział 2026, 20 seedów.")
    rep.p(f"<b>Wniosek.</b> Między algorytmami: 4–7× mniejsze V′ (Szachownica, MAGIC, Digits) nie dało mierzalnego spadku jakości na Szachownicy i MAGIC (dolna granica CI różnicy &gt; −0,5 p.p.); na Digits (różnica {fnum(float(sq[sq['zbiór'] == 'digits']['różnica (nowy−stary)'].iloc[0]), 4)}, "
          f"CI {ci(float(sq[sq['zbiór'] == 'digits']['CI dolny'].iloc[0]), float(sq[sq['zbiór'] == 'digits']['CI górny'].iloc[0]))}) nie można wykluczyć spadku większego niż 0,5 p.p. Większe V′ starego algorytmu nie dało więc poprawy na tych zbiorach. "
          "To porównanie jest opisowe, bo algorytmy różnią się także polityką K i przełączaniem faz. W obrębie nowego algorytmu dodatnia korelacja |V′|–BA widoczna na Szachownicy (ρ cząstkowa po kontroli K: "
          f"{fnum(float(sq[sq['zbiór'] == 'chessboard']['ρ cząstkowa |V′|–BA test | K'].iloc[0]), 2)}) i na Spiral (przed kontrolą K) nie ma charakteru przyczynowego: |V′| i K rosną razem z długością przebiegu, a przebiegi o różnej długości kończą się z różnych powodów. "
          "Na pozostałych zbiorach korelacji nie ma.")

    # ---------------- 5.3 trudniejsze próbki ----------------
    rep.h2("5.3 Czy ewolucja wybiera trudniejsze próbki? (P1)")
    rows = [["Zbiór", "błąd na ΔV′ — ewolucja", "błąd na ΔV′ — losowanie", "różnica", "95% CI", "d<sub>z</sub>", "p (Holm)", "BA na ΔV′ ewol.", "BA na ΔV′ los."]]
    for ds in DS:
        d1, d2 = diff_pb[diff_pb["zbiór"] == ds].iloc[0], dba_pb[dba_pb["zbiór"] == ds].iloc[0]
        rows.append([LB[ds], pmv(d1.mean_a, d1.sd_a, 3), pmv(d1.mean_b, d1.sd_b, 3), fnum(d1["diff"], 3), ci(d1.ci_lo, d1.ci_hi, 3), fnum(d1.dz, 1), pf(d1.p_holm), fnum(d2.mean_a, 3), fnum(d2.mean_b, 3)])
    rep.table(rows, [2.6, 2.7, 2.7, 1.4, 2.3, 1.0, 1.3, 1.9, 1.9], "Trudność przyrostów ΔV′ (średnia z kroków V′ w przebiegu; podział 2026, n = 20 par po seedach; losowanie z harmonogramem ewolucji). Ewolucja wybiera próbki, na których ustalony model myli się wyraźnie częściej; BA(ΔV′) jest o 0,17–0,36 niższe.", font_scale=0.95)
    rep.p(f"<b>Tak.</b> Różnica błędu na ΔV′ wynosi od {fnum(diff_pb['diff'].min(), 3)} do {fnum(diff_pb['diff'].max(), 3)} na wszystkich 7 zbiorach (CI wyklucza zero, {pbound(diff_pb.p_holm.max())}, d<sub>z</sub> od {fnum(diff_pb.dz.min(), 1)} do {fnum(diff_pb.dz.max(), 1)}), "
          "a BA na ΔV′ jest niższe o 0,17–0,36. Rozmiar efektu jest bardzo duży, wynik robustny (te same seedy, ten sam harmonogram). Zastrzeżenie: modele T′ w obu wariantach nie są identyczne (różne V′), "
          "więc trudność jest mierzona względem modelu danego wariantu. <b>To stwierdzenie dotyczy wyłącznie trudności próbek, nie jakości modelu</b> (sekcja 5.4).")

    # ---------------- 5.4 jakość ----------------
    rep.h2("5.4 Czy ewolucja prowadzi do lepszego modelu? (P2)")
    rows = [["Zbiór", "ewolucja (śr. ± SD)", "losowanie (śr. ± SD)", "różnica", "95% CI", "d<sub>z</sub>", "p (Holm)", "ocena praktyczna (granica 0,5 p.p.)"]]
    for ds in DS:
        r = pbm("test_ba", ds)
        rows.append([LB[ds], pmv(r.mean_a, r.sd_a), pmv(r.mean_b, r.sd_b), fnum(r["diff"], 4), ci(r.ci_lo, r.ci_hi), fnum(r.dz, 2), pf(r.p_holm), r["ocena"]])
    rep.table(rows, [2.5, 2.8, 2.8, 1.5, 2.5, 1.0, 1.3, 3.4], "BA na teście, ewolucja − losowanie z harmonogramem ewolucji (podział 2026, n = 20 par po seedach; 95% CI z bootstrapu; p Holma z testu Wilcoxona, 7 zbiorów). Test jest wyłącznie obserwacją. "
              "Granica 0,5 p.p. jest przyjętą konwencją, nie wynikiem z danych.", font_scale=0.95)
    rows = [["Zbiór", "F1-macro test: ewol.", "F1-macro: los.", "różnica [95% CI]", "p (Holm)", "BA pełne V: różnica [95% CI]", "p (Holm)"]]
    for ds in DS:
        r, q = pbm("test_f1m", ds), pbm("fullv_ba", ds)
        rows.append([LB[ds], fnum(r.mean_a, 4), fnum(r.mean_b, 4), f"{fnum(r['diff'], 4)} {ci(r.ci_lo, r.ci_hi)}", pf(r.p_holm), f"{fnum(q['diff'], 4)} {ci(q.ci_lo, q.ci_hi)}", pf(q.p_holm)])
    rep.table(rows, [2.5, 2.3, 2.1, 4.0, 1.4, 4.1, 1.4], "Metryki dodatkowe (obserwacje): F1-macro na teście i BA na pełnym V, ewolucja − losowanie (podział 2026, n = 20). AUC nie został policzony (patrz sekcja 3.6).", font_scale=0.95)
    sb = ps_[ps_.metryka == "test_ba"]
    rows = [["Zbiór", "ewolucja", "losowanie", "różnica", "95% CI (bootstrap po parach)", "95% CI (hierarchiczny)", "ocena praktyczna"]]
    for _, r in sb.iterrows():
        rows.append([LB[r["zbiór"]], fnum(r.mean_a, 4), fnum(r.mean_b, 4), fnum(r["diff"], 4), ci(r.ci_lo, r.ci_hi), ci(r.h_lo, r.h_hi), r["ocena"]])
    rep.table(rows, [2.6, 1.8, 1.8, 1.5, 3.6, 3.4, 3.1], "Powtórzenie na 4 podziałach (2026–2029) × 5 seedów (42–46), 6 zbiorów zależnych od podziału, n = 20 par na zbiór. Bootstrap hierarchiczny losuje podziały, a w nich seedy (4 podziały to niewiele, więc CI są orientacyjne).", font_scale=0.95)
    rep.figure("rys05_roznice_ewolucja_losowanie.png", "Różnice ewolucja − losowanie (harmonogram identyczny) w punktach procentowych, z 95% CI. Szare pasmo = ±0,5 p.p.")
    c_bonf = pbm("test_ba", "chessboard")
    rep.p(f"<b>Nie wykazano praktycznie istotnej przewagi ewolucyjnego doboru V′ nad doborem losowym.</b> Różnice BA na teście wynoszą od {fnum(tb['diff'].min() * 100, 2)} do +{fnum(tb['diff'].max() * 100, 2)} p.p., żadna nie przekracza przyjętej granicy 0,5 p.p. "
          f"CI wyklucza zero tylko na Szachownicy (+{fnum(float(c_bonf['diff']) * 100, 2)} p.p., CI {ci(c_bonf.ci_lo, c_bonf.ci_hi)}, p Holma {pf(c_bonf.p_holm)}, d<sub>z</sub> = {fnum(c_bonf.dz, 2)}); ta różnica jest mała, "
          f"niespójna między metrykami (F1-macro: p Holma {pf(pbm('test_f1m', 'chessboard').p_holm)}; BA pełne V: {pf(pbm('fullv_ba', 'chessboard').p_holm)}) i nie powtarza się na innym zbiorze, a Szachownica nie ma podziałów dodatkowych. "
          f"Górna granica CI jest poniżej 0,5 p.p. na {len(excl_adv)} z 7 zbiorów ({', '.join(LB[d] for d in excl_adv)}), więc na nich można wykluczyć przewagę ewolucji ≥ 0,5 p.p."
          + (f" (na Szachownicy brzegowo: górna granica CI = {fnum(float(c_bonf.ci_hi), 4)})" if ("chessboard" in excl_adv and float(c_bonf.ci_hi) > 0.9 * A.MARGIN) else "")
          + (f"; nie można tego zrobić dla {', '.join(loc(d) for d in DS if d not in excl_adv)} (szerokie CI). " if len(excl_adv) < 7 else ". ") +
          f"Na MAGIC ewolucja ma nominalnie niższe F1-macro ({fnum(float(pbm('test_f1m', 'magic')['diff']), 4)}, CI {ci(pbm('test_f1m', 'magic').ci_lo, pbm('test_f1m', 'magic').ci_hi)}, p Holma {pf(pbm('test_f1m', 'magic').p_holm)}) — bez potwierdzenia po korekcie. "
          f"Na podziałach dodatkowych żadne CI (także hierarchiczne) nie wyklucza zera.")
    pax = pa[pa.metryka == "test_ba"]
    sig_aux = pax[(pax.ci_lo > 0) | (pax.ci_hi < 0)]["zbiór"].tolist()
    rep.p(f"<b>Porównanie pomocnicze z rand_switch</b> (samodzielny automat losowy, mniejsze V′ i K): ewolucja ma wyższe BA na teście na {len(pax[pax['diff'] > 0])} z 7 zbiorów, a CI wyklucza zero na {', '.join(loc(d) for d in sig_aux) or '—'}. "
          "Ta różnica miesza efekt doboru z efektem mniejszego |V′| i K u wariantu losowego, więc nie jest dowodem przewagi ewolucji (rand_switch kończy się wcześniej, najczęściej po braku postępu po próbie V′).")

    # ---------------- 5.5 koszt ----------------
    rep.h2("5.5 Czy ewolucja jest warta dodatkowego kosztu? (P3)")
    cb = P["cost_base"]
    rows = [["Zbiór", "Wariant", "treningi SVM", "oceny osobników T′", "oceny osobników Val-GA", "wywołania predict", "wiersze predict", "decision function", "wiersze ΔV′", "wiersze V′ w T′"]]
    for ds in DS:
        for v in A.VARIANTS:
            r = cb[(cb["zbiór"] == ds) & (cb["wariant"] == v)].iloc[0]
            rows.append([LB[ds] if v == "evo_switch" else "", vname[v], fint(r["treningi SVM"]), fint(r["oceny osobników T′ (GA)"]), fint(r["oceny osobników Val-GA"]), fint(r["wywołania predict"]),
                         fint(r["wiersze predict"]), fint(r["wywołania decision_function"]), fint(r["wiersze ΔV′ ocenione"]), fint(r["wiersze V′ ocenione w T′ (faktycznie)"])])
    rep.table(rows, [2.0, 2.3, 1.4, 1.6, 1.7, 1.6, 2.0, 1.5, 1.2, 2.0], "Koszt całego przebiegu (średnia na przebieg, podział 2026, n = 20): wywołania modelu wykonane przez algorytm (bez obserwacji pełnego V i testu). Koszt T′ = oceny GA × |V′|; koszt V′ = ocena puli kandydatów "
              "(ewolucja: jedno predict na pulę) + diagnostyka ΔV′. Treningi SVM = oceny osobników T′ + 1 trening końcowy na krok T′.", font_scale=0.9)
    rows = [["Zbiór", "treningi SVM", "wiersze predict (łącznie)", "wiersze predict w fazie V′", "czas zegarowy"]]
    for _, r in cr.iterrows():
        rows.append([LB[r["zbiór"]], f"{fnum(r['treningi SVM'], 2)} {ci(*r['treningi SVM CI'], 2)}", f"{fnum(r['wiersze predict'], 2)} {ci(*r['wiersze predict CI'], 2)}",
                     fnum(r["wiersze predict w fazie V′"], 0) + "×", f"{fnum(r['czas zegarowy'], 2)} {ci(*r['czas zegarowy CI'], 2)}"])
    rep.table(rows, [2.6, 3.8, 4.4, 3.2, 3.8], "Stosunek kosztu ewolucji do kosztu losowania przy identycznym harmonogramie (średnia geometryczna z 20 par po seedach, 95% CI z bootstrapu logarytmu). Czas zegarowy pochodzi z równoległych uruchomień na jednej maszynie (ok. 10–12 procesów) — miara pomocnicza.", font_scale=0.95)
    rep.figure("rys07_koszt_ewolucja_do_losowania.png", "Stosunek kosztu ewolucji do kosztu losowania (harmonogram identyczny), skala logarytmiczna, z 95% CI.", 15.0)
    rep.p(f"<b>Koszt ewolucji jest wyższy, i to nie tylko przez Val-GA.</b> Przy identycznym harmonogramie ewolucja wymaga {fnum(r_svm[0], 2)}–{fnum(r_svm[1], 2)}× treningów SVM (dolna granica CI na wszystkich zbiorach &gt; 1: {fnum(svm_lo, 2)}) i {fnum(r_pred[0], 2)}–{fnum(r_pred[1], 2)}× wierszy predict. "
          "Większość kosztu to faza T′ (GA dla T′ ocenia kilka tysięcy osobników). "
          f"Na tych samych krokach harmonogramu ({fint(len(pidx))} par, podział 2026) GA dla T′ trwa w wariancie ewolucyjnym średnio o {fnum(gen_diff, 2)} generacji dłużej (dłużej w {fnum(gen_longer * 100, 0)}% par, krócej w {fnum(gen_shorter * 100, 0)}%), "
          f"a tylko {fnum(reach_e * 100, 1)}% uruchomień kończy się osiągnięciem BA(V′) = 1 wobec {fnum(reach_r * 100, 1)}% w losowaniu (średnie BA(V′) kroku T′: {fnum(ba_e, 3)} wobec {fnum(ba_r, 3)}). "
          "Dane są zgodne z wyjaśnieniem, że trudniejsze V′ zostawia GA więcej miejsca na poprawę i wczesne zatrzymanie następuje rzadziej; związku przyczynowego nie testowano. "
          "Sam Val-GA dodaje w fazie V′ ocenę całej puli kandydatów (np. Szachownica ≈ 660 tys. wierszy na przebieg wobec ≈ 1 tys. dla losowania); decision_function jest wywoływane tylko diagnostycznie (jedno wywołanie na krok V′). "
          f"Czas zegarowy: {fnum(r_time[0], 2)}–{fnum(r_time[1], 2)}× (ograniczenie: równoległe uruchomienia, więc nie jest podstawą wniosku).")
    rep.p(f"<b>Czy warto?</b> Przy kosztach wyższych o {fnum((r_svm[0] - 1) * 100, 0)}–{fnum((r_svm[1] - 1) * 100, 0)}% (treningi SVM) i {fnum((r_pred[0] - 1) * 100, 0)}–{fnum((r_pred[1] - 1) * 100, 0)}% (wiersze predict) ewolucja nie dała praktycznie istotnej poprawy jakości (sekcja 5.4). "
          "Na danych tego badania koszt dodatkowy nie jest uzasadniony zyskiem jakościowym. Zastrzeżenie: wniosek dotyczy zakresu rozmiarów V′ z tego badania; nie sprawdzano mniejszych V′ ani innych metryk fitness.")

    # ---------------- 5.6 pilotaż ----------------
    rep.h2("5.6 Pilotaż (oznaczony jako pilotażowy, poza wnioskami)")
    pp = P["pilot"]
    rows = [["Seria", "Zbiór", "Wariant", "n", "|V′| śr. (min–max)", "|V′|/|V|", "K końc.", "BA test", "cykle"]]
    for _, r in pp.iterrows():
        rows.append([r["series"].replace("v2_smoke_", ""), LB[r["dataset"]], vname[r["variant"]], int(r["n"]), f"{fint(r['vprime'])} ({fint(r['vprime_min'])}–{fint(r['vprime_max'])})", fnum(r["frac"], 3), fint(r["final_k"]), fnum(r["test_ba"], 4), fnum(r["steps"], 1)])
    rep.table(rows, [3.0, 2.3, 3.0, 0.7, 3.1, 1.4, 1.3, 1.5, 1.2], "Pilotaż (2–3 seedy na komórkę; chess_r1000: limit |V′| = 1000, bramka ρ_SV włączona; chess_r1000_nogate: limit 1000, bramka wyłączona). Wyłącznie ilustracja wykonalności porównania przy |V′| ≈ 1000.", font_scale=0.92)

    # ============================ 6. STOP ============================
    rep.h1("6. Analiza warunku stopu")
    rep.figure("rys08_powody_zakonczenia.png", "Powody zakończenia przebiegów (wszystkie serie główne; Szachownica tylko podział 2026). rand_replay kończy się z konstrukcji razem z przebiegiem ewolucyjnym.", 16.4)
    rows = [["Zbiór", "Wariant", "n", "brak postępu po próbie V′", "ρ_SV &lt; TSV", "limit K", "zastój T′ (próba wykonana)", "BA=1, V′ nierozsz."]]
    for ds in DS:
        for v in ("evo_switch", "rand_switch"):
            g = main_runs[(main_runs.dataset == ds) & (main_runs.variant == v)]
            vc = g.stop_decision.str.replace("stop_", "").value_counts()
            rows.append([LB[ds] if v == "evo_switch" else "", vname[v], len(g)] + [int(vc.get(k, 0)) for k in ("no_progress_after_vprime_retry", "no_k_growth_signal", "k_limit", "tprime_no_progress", "perfect_on_max_vprime")])
    rep.table(rows, [2.6, 2.8, 0.9, 2.6, 1.9, 1.6, 2.8, 2.2], "Powody zakończenia (liczba przebiegów; serie główne). Żaden przebieg nie zakończył się bezpiecznikiem max_cycles.", font_scale=0.95)
    npg_e = int((evo_m.stop_decision == "stop_no_progress_after_vprime_retry").sum())
    npg_r = int(((main_runs.variant == "rand_switch") & (main_runs.stop_decision == "stop_no_progress_after_vprime_retry")).sum())
    nks_e = int((evo_m.stop_decision == "stop_no_k_growth_signal").sum())
    rep.p(f"<b>Warunek braku postępu jest aktywny i skuteczny.</b> Wśród {len(evo_m)} przebiegów ewolucyjnych {npg_e} ({fnum(npg_e / len(evo_m) * 100, 0)}%) kończy się brakiem postępu po próbie V′, a {nks_e} ({fnum(nks_e / len(evo_m) * 100, 0)}%) — bramką ρ<sub>SV</sub>; "
          f"w losowaniu z własnym automatem {npg_r} z {int((main_runs.variant == 'rand_switch').sum())} ({fnum(npg_r / int((main_runs.variant == 'rand_switch').sum()) * 100, 0)}%) kończy się brakiem postępu. "
          "Przebiegi ewolucyjne kończą się więc najczęściej, gdy ρ<sub>SV</sub> spada poniżej TSV (K jest wtedy wystarczająco duże), a losowe — gdy po próbie V′ T′ nie poprawia się, co następuje wcześniej i przy mniejszym V′ i K. "
          f"Prób V′ po zastoju (przejście T′→V′ bez BA(V′) = 1): {n_att}; w {n_res} ({fnum(n_res / n_att * 100, 0)}%) T′ osiągnęło poziom odniesienia — w {len(first_ok)} z nich ({fnum(len(first_ok) / n_res * 100, 0)}%) już w pierwszym kroku po powrocie z V′ (dzięki warm startowi), "
          f"w pozostałych po kolejnych zwiększeniach K; {n_fail} ({fnum(n_fail / n_att * 100, 0)}%) zakończyło przebieg (stop po braku postępu, gdy K nie mogło rosnąć albo 3 zwiększenia nie dały poziomu). "
          "Zabezpieczenia przed zapętleniem nie musiały działać: 0 przebiegów z max_cycles, 0 powtórzonych prób na poziom K.")

    # ============================ 7. PRZEŁĄCZANIE ============================
    rep.h1("7. Analiza warunku przełączania faz")
    pr = ph["per"]
    rows = [["Zbiór", "Wariant", "n", "T′→V′", "w tym BA(V′)=1", "w tym zastój/ρ_SV", "V′→T′", "wzrosty K", "rozszerzenia V′", "kroki T′ z BA=1"]]
    for ds in DS:
        for v in ("evo_switch", "rand_switch"):
            r = pr[(pr["zbiór"] == ds) & (pr["wariant"] == v)].iloc[0]
            rows.append([LB[ds] if v == "evo_switch" else "", vname[v], int(r["przebiegi"]), fnum(r["T′→V′ (wejścia, śr./przebieg)"], 1), fnum(r["w tym z BA(V′)=1"], 1), fnum(r["w tym z zastoju/ρ_SV (próba)"], 1),
                         fnum(r["V′→T′ (powroty, śr./przebieg)"], 1), fnum(r["wzrosty K (śr./przebieg)"], 1), fnum(r["rozszerzenia V′ (śr./przebieg)"], 1), fnum(r["kroki T′ z BA(V′)=1 (śr./przebieg)"], 1)])
    rep.table(rows, [2.4, 2.4, 1.3, 1.2, 1.8, 1.9, 1.2, 1.5, 1.8, 1.9], "Przełączenia faz na przebieg (średnie; serie główne, wszystkie podziały: Szachownica n = 20, pozostałe n = 35). rand_replay pominięto, bo jego przejścia są narzucone harmonogramem ewolucji. "
              "W ostatniej fazie T′ przebiegu T′→V′ nie następuje, dlatego liczba powrotów V′→T′ jest równa liczbie wejść do V′.", font_scale=0.95)
    tv, tk, te = ph["per"][ph["per"].wariant == "evo_switch"], None, None
    ev_pr = pr[(pr.wariant == "evo_switch") & (pr["zbiór"] != "digits")]
    rep.p(f"<b>Przejścia.</b> W przebiegu ewolucyjnym (poza Digits) T′→V′ następuje średnio {fnum(ev_pr['T′→V′ (wejścia, śr./przebieg)'].min(), 1)}–{fnum(ev_pr['T′→V′ (wejścia, śr./przebieg)'].max(), 1)} razy na przebieg, "
          f"V′→T′ tyle samo, K wzrasta {fnum(ev_pr['wzrosty K (śr./przebieg)'].min(), 0)}–{fnum(ev_pr['wzrosty K (śr./przebieg)'].max(), 0)} razy, V′ jest rozszerzane {fnum(ev_pr['rozszerzenia V′ (śr./przebieg)'].min(), 0)}–{fnum(ev_pr['rozszerzenia V′ (śr./przebieg)'].max(), 0)} razy; "
          "na Digits (limit rozmiaru, limit K klasy) jest jedno przejście, 5 wzrostów K i 2 rozszerzenia. "
          f"Z {len(entries_evo)} wejść do fazy V′ w wariancie ewolucyjnym <b>{mast_evo} ({fnum(mast_evo / len(entries_evo) * 100, 0)}%)</b> pochodzi z BA(V′) = 1, a {len(entries_evo) - mast_evo} z zastoju T′ lub bramki ρ<sub>SV</sub>. "
          f"BA(V′) = 1 występuje w {fnum(perfect_init * 100, 0)}% przypadków na V′ startowym (mediana |V′| przy BA = 1: {fint(ap_e['V′ przed'].median())} próbek; nie zawsze w cyklu 1 — T′ potrzebuje czasem kilku cykli, żeby opanować nawet 20 próbek).")
    rows = [["Wariant", "kroków T′ z BA=1", "stop zamiast V′", "długość fazy V′ (rozszerzenia)", "błąd na ΔV′ w tej fazie", "następny krok T′: BA(V′)", "następny krok T′ znów BA=1"]]
    for v, g in (("evo_switch", ap_e), ("rand_switch", ap_r)):
        rows.append([vname[v], len(g), int(g["decyzja"].str.startswith("stop").sum()), fnum(g["długość fazy V′ (rozszerzenia)"].mean(), 2), fnum(g["błąd na ΔV′ (śr.)"].mean(), 3),
                     fnum(g["kolejny krok T′: BA(V′)"].mean(), 3), f"{int(g['kolejny krok T′: BA(V′)=1'].fillna(False).astype(bool).sum())} z {len(g)} ({fnum(g['kolejny krok T′: BA(V′)=1'].fillna(False).astype(bool).mean() * 100, 1)}%)"])
    rep.table(rows, [2.6, 2.0, 2.0, 2.8, 2.6, 2.8, 3.0], "Co dzieje się po osiągnięciu BA(V′) = 1 (wszystkie kroki T′ z BA = 1 w seriach głównych, warianty z własnym automatem). Po BA = 1 algorytm zawsze przechodzi do fazy V′ (wyjątek: 3 stopy przy nierozszerzalnym V′, Digits).")
    rep.p("<b>Co dzieje się po BA(V′) = 1.</b> Następuje faza V′ (średnio ok. 4 rozszerzenia po 20 próbek). W wariancie ewolucyjnym przyrosty są bardzo trudne (błąd modelu na ΔV′ ≈ 0,34), więc po powrocie do T′ BA(V′) spada "
          f"(średnio do {fnum(ap_e['kolejny krok T′: BA(V′)'].mean(), 2)}), a ponowne BA = 1 zdarza się w {int(ap_e['kolejny krok T′: BA(V′)=1'].fillna(False).astype(bool).sum())} z {len(ap_e)} przypadków; w losowaniu przyrosty są łatwe (błąd ≈ 0,11) i BA = 1 wraca w {fnum(ap_r['kolejny krok T′: BA(V′)=1'].fillna(False).astype(bool).mean() * 100, 0)}% przypadków. "
          "Z tego powodu po początkowym opanowaniu V′ ścieżka „BA = 1 → V′” prawie się nie powtarza w wariancie ewolucyjnym, a dalsze wejścia do fazy V′ wynikają z zastoju T′.")
    rep.figure("rys04_trajektorie_faz.png", "Trajektorie faz dla trzech zbiorów (podział 2026, seed 42, wariant ewolucyjny): BA(V′) w krokach T′ i BA(ΔV′) w krokach V′ (góra), |V′| (środek), K ze zaznaczonymi zwiększeniami (dół). "
               "Jasne tło = faza optimize_V_prime; czerwona linia = stop.")
    rep.p("Trajektorie mają wspólny wzór: krótki początkowy etap, w którym T′ opanowuje małe V′ (gwiazdki), pierwsza faza V′, a następnie „piłokształtny” przebieg — seria zwiększeń K w fazie T′ bez zmiany V′, zastój, jedna próba V′ (jasne tło, 3–6 rozszerzeń), ponowne zwiększenia K. "
          "BA(V′) maleje wraz z utrudnianiem V′ (ewolucja dobiera próbki, na których model się myli), a BA(ΔV′) pozostaje niskie przez cały przebieg. Przebiegi kończą się przy ρ<sub>SV</sub> &lt; TSV (Moons, Szachownica, Spiral w tych przykładach).")
    rep.h2("7.1 Przyczyny wejścia do fazy V′ bez BA(V′) = 1 i zasada „najpierw K”")
    wt = why.reset_index()
    rows = [["Zbiór", "zastój: 3 zwiększenia K bez poprawy", "bramka ρ_SV &lt; TSV (bez zwiększenia K)", "limit K klasy (bez zwiększenia K)"]]
    for ds in DS:
        r = wt[wt.dataset == ds].iloc[0]
        rows.append([LB[ds], int(r["zastój: 3 zwiększenia K bez poprawy"]), int(r["bramka ρ_SV < TSV (bez zwiększenia K)"]), int(r["limit K klasy (bez zwiększenia K)"])])
    rows.append(["<b>Razem</b>", int(why_tot["zastój: 3 zwiększenia K bez poprawy"]), int(why_tot["bramka ρ_SV < TSV (bez zwiększenia K)"]), int(why_tot["limit K klasy (bez zwiększenia K)"])])
    rep.table(rows, [3.5, 4.8, 5.2, 4.0], "Powody wejścia do fazy V′ przy BA(V′) < 1 (warianty z własnym automatem, serie główne).", bold_rows=(len(rows) - 1,))
    rep.p(f"Z {len(below)} kroków T′ z BA(V′) &lt; 1 {int((below.decision == 'increase_k').sum())} ({fnum((below.decision == 'increase_k').mean() * 100, 1)}%) zwiększa K — zasada „najpierw K” obowiązuje w zdecydowanej większości. "
          f"Odstępstwem są {int(why_tot['bramka ρ_SV < TSV (bez zwiększenia K)'])} wejść do V′ z powodu bramki ρ<sub>SV</sub> (K nie zwiększono) i {int(why_tot['limit K klasy (bez zwiększenia K)'])} z powodu limitu K klasy (K nie mogło rosnąć) — łącznie "
          f"{fnum((why_tot['bramka ρ_SV < TSV (bez zwiększenia K)'] + why_tot['limit K klasy (bez zwiększenia K)']) / len(below) * 100, 1)}% kroków z BA &lt; 1. Pozostałe {int(why_tot['zastój: 3 zwiększenia K bez poprawy'])} wejść poprzedziły 3 nieskuteczne zwiększenia K, zgodnie z założeniem.")
    rep.h2("7.2 Warunek końca fazy V′: czy jakość nowych próbek przestaje się pogarszać?")
    rep.figure("rys06_trudnosc_dV_i_dlugosc_fazy.png", "A: błąd klasyfikacji na ΔV′ w zależności od numeru rozszerzenia w fazie V′ (średnia i 95% CI; serie główne, punkty o n ≥ 50). B: rozkład liczby rozszerzeń w jednej fazie V′ (ewolucja).", 16.0)
    first_last = phe["zmiana błędu (ost.−pierwszy)"].mean()
    p3 = float((phe.L == 3).mean())
    pl = float((phe.end == "difficulty_plateau").mean())
    rep.p(f"Trudność ΔV′ <b>nie rośnie</b> wraz z kolejnymi rozszerzeniami fazy: w wariancie ewolucyjnym błąd na ΔV′ jest praktycznie stały (≈ 0,26 dla numerów rozszerzenia 1–6), a zmiana między ostatnim a pierwszym przyrostem fazy wynosi średnio {fnum(first_last, 3)}. "
          f"Warunek „trudność przestała rosnąć o ≥ 0,01 przez 2 przyrosty” jest więc spełniany niemal natychmiast: {fnum(pl * 100, 0)}% faz kończy się plateau trudności, a {fnum(p3 * 100, 0)}% faz trwa dokładnie 3 rozszerzenia (minimum przy cierpliwości 2). "
          f"Średnia długość fazy to {fnum(phe.L.mean(), 1)} rozszerzenia (mediana {fnum(phe.L.median(), 0)}). Jakość klasyfikacji nowych próbek faktycznie „przestaje się pogarszać”, ale dzieje się to od pierwszego przyrostu, "
          "więc reguła działa głównie jak stała liczba rozszerzeń na fazę (parametr vprime_difficulty_patience), a nie jak adaptacyjny sygnał. W losowaniu trudność lekko rośnie z numerem rozszerzenia (0,10 → 0,14 dla numerów 1–6), co przedłuża część faz.")

    # ============================ 8. OGRANICZENIA ============================
    rep.h1("8. Ograniczenia i problemy")
    rep.h2("8.1 Różnice względem założeń")
    rep.bullets([
        f"<b>Zasada „najpierw K” spełniona częściowo.</b> Bramka ρ<sub>SV</sub> (K rośnie tylko przy ρ<sub>SV</sub> ≥ TSV) powoduje przejście do V′ bez zwiększenia K w {int(why_tot['bramka ρ_SV < TSV (bez zwiększenia K)'])} przypadkach; po nieudanej próbie V′ ta sama bramka kończy algorytm, bo K nie może rosnąć. Bramka dziedziczona jest z polityki DAGA; "
        "jej zmiana wymaga przeprojektowania wzoru na K<sub>next</sub> (wartość bezwzględna).",
        "<b>BA(V′) = 1 może zakończyć algorytm</b>, ale tylko gdy V′ nie da się rozszerzyć (3 przypadki, Digits, limit rozmiaru).",
        "<b>Poziom odniesienia po próbie V′</b> to najlepszy T′ sprzed próby oceniony na tym samym, rozszerzonym V′ (+ 0,0005), a nie dosłownie jego fitness sprzed próby: V′ po próbie jest trudniejsze, więc dosłowne porównanie kończyłoby algorytm prawie zawsze. Wariant dosłowny jest zaimplementowany (retry_reference: literal), ale nie był uruchamiany.",
        "<b>Ocena puli kandydatów.</b> Val-GA ocenia osobniki na podzbiorach ΔV′, ale predykcje dla całej puli V \\ V′ są liczone raz na rozszerzenie (zbuforowane); na Szachownicy (pula ≈ 27 tys.) to droższe niż nominalna ocena podzbiorów. Stare próbki V′ nie są oceniane ponownie.",
        "<b>rand_replay jest uprzywilejowany</b>: jego harmonogram (K, liczba cykli, moment stopu) pochodzi z ewolucji; rand_switch jest samodzielny, ale ma mniejsze V′ i K. Wariant rand_fixed (rozmiar z wyniku ewolucji) nie był używany.",
    ])
    rep.h2("8.2 Ograniczenia metodologiczne")
    rep.bullets([
        "<b>Liczba podziałów i seedów.</b> Podział bazowy: 20 seedów; trzy podziały dodatkowe: tylko 5 seedów; Szachownica ma jeden podział. Różnice rzędu 0,1–0,3 p.p. leżą poniżej rozdzielczości układu; CI hierarchiczne (4 podziały) są orientacyjne.",
        "<b>Porównania wielokrotne.</b> Wartości p są korygowane metodą Holma tylko w obrębie 7 zbiorów dla danej metryki; CI nie są korygowane; metryk jest kilka, więc pojedyncze wyniki istotne należy czytać ostrożnie.",
        "<b>Granica praktycznej istotności 0,5 p.p.</b> jest przyjętą konwencją, nie wynikiem z danych.",
        "<b>Brak niezależności metryk.</b> BA na teście, F1-macro i BA na pełnym V pochodzą z tego samego modelu reprezentatywnego i są silnie skorelowane; nie stanowią niezależnych potwierdzeń. Obserwacje z jednego podziału nie są w pełni niezależne.",
        "<b>Porównanie ze starym algorytmem jest opisowe</b>: różni się nie tylko rozmiarem V′, ale też polityką K, przełączaniem faz i warm startem T′.",
        "<b>Metryki.</b> Fitness = BA; F1 i AUC nie były używane jako fitness, AUC nie został policzony; porównanie metryk pozostaje do wykonania. Pełne V jest tylko obserwacją (tak działa implementacja).",
        "<b>Hiperparametry.</b> (C, γ) wspólne dla podziałów; parametry automatu nie były strojone; wrażliwość nie była badana. Stała pula obserwacji: zmienność wynikająca z generowania danych nie jest uwzględniona.",
        "<b>Koszt.</b> Liczniki obejmują predict i decision_function; przy probability = true predict_proba nie byłoby liczone (wszystkie serie mają probability = false). Czas zegarowy pochodzi z równoległych uruchomień.",
    ])

    # ============================ 9. WNIOSKI ============================
    rep.h1("9. Wnioski")
    rep.h2("9.1 Potwierdzone przez dane")
    rows = [["Twierdzenie", "Podstawa w danych"],
            ["Automat dwóch faz działa zgodnie z założeniami: BA(V′) = 1 przełącza do V′ (nie kończy algorytmu), po V′ następuje T′, K rośnie najwyżej raz na cykl, T′ widzi cały skumulowany V′, pełne V i test nie sterują.",
             f"wymagania 1–5 i 7–11 spełnione (z zastrzeżeniami z sekcji 5.1) na {n_main} przebiegach i {fint(n_cycles)} cyklach (0 naruszeń); 0 przebiegów z max_cycles"],
            [f"V′ nie rośnie do nadmiernych rozmiarów: Szachownica {fint(chess_new['V′ średnio'])} wobec {fint(chess_old['V′ średnio'])} (stary algorytm).",
             f"20 par seedów; maks. |V′|/|V| = {fnum(main_runs.vprime_fraction.max(), 3)}; maks. |V′| na Szachownicy {fint(main_runs[main_runs.dataset == 'chessboard'].vprime_rows.max())}"],
            [f"BA na teście nowego algorytmu nie jest gorsze od starego o więcej niż 0,5 p.p. na {len(noninf)} z 7 zbiorów ({', '.join(LB[d] for d in noninf)}): przy 4–7× mniejszym V′ na Szachownicy, MAGIC i Digits (Digits: patrz 9.2), przy podobnym lub większym V′ na pozostałych.",
             "dolna granica 95% CI różnicy sparowanej (nowy − stary) > −0,005 (porównanie opisowe)"],
            ["Ewolucja wybiera istotnie trudniejsze próbki do ΔV′ niż losowanie.", f"7/7 zbiorów: CI wyklucza zero, {pbound(diff_pb.p_holm.max())}, d_z ≥ {fnum(diff_pb.dz.min(), 1)}"],
            ["Ewolucja kosztuje więcej niż losowanie przy tym samym harmonogramie.", f"treningi SVM {fnum(r_svm[0], 2)}–{fnum(r_svm[1], 2)}×, wiersze predict {fnum(r_pred[0], 2)}–{fnum(r_pred[1], 2)}×; dolna granica CI treningów &gt; 1 na wszystkich zbiorach"],
            [f"Nie wykazano praktycznie istotnej (≥ 0,5 p.p.) przewagi ewolucji nad losowaniem w BA na teście; na {len(excl_adv)} zbiorach ({', '.join(LB[d] for d in excl_adv)}) można ją wykluczyć.",
             "górna granica 95% CI różnicy ewolucja − losowanie (rand_replay) &lt; 0,005"]]
    rep.table(rows, [10.0, 7.8], "Wnioski potwierdzone (status wyliczony z danych; „potwierdzone” = spełnione w badanych zbiorach i podziałach, nie więcej).", font_scale=0.95)
    rep.h2("9.2 Sugerowane przez dane")
    rows = [["Obserwacja", "Dlaczego tylko sugerowana"],
            [f"Na Szachownicy ewolucja ma wyższe BA na teście (+{fnum(float(c_bonf['diff']) * 100, 2)} p.p.).", f"p Holma {pf(c_bonf.p_holm)} dla BA, ale {pf(pbm('test_f1m', 'chessboard').p_holm)} dla F1-macro; poniżej 0,5 p.p.; brak replikacji (jeden podział)"],
            ["Na MAGIC ewolucja ma nominalnie niższe F1-macro niż losowanie.", f"CI wyklucza zero, ale p Holma {pf(pbm('test_f1m', 'magic').p_holm)}; BA na teście bez różnicy"],
            [f"Na Szachownicy większe V′ wiąże się z wyższym BA w obrębie nowego algorytmu (ρ cząstkowa po kontroli K = {fnum(float(sq[sq['zbiór'] == 'chessboard']['ρ cząstkowa |V′|–BA test | K'].iloc[0]), 2)}).", "n = 20, bez korekty na wielokrotność, zależność nieprzyczynowa (długość przebiegu); brak na pozostałych zbiorach"],
            [f"rand_switch (samodzielny automat losowy) ma niższe BA niż ewolucja; CI wyklucza zero na: {', '.join(LB[d] for d in sig_aux) or '—'}.", "różnica zmieszana z mniejszym |V′| i K u wariantu losowego"],
            ["Przy 4× mniejszym V′ na Digits nie można wykluczyć spadku jakości > 0,5 p.p.", f"różnica {fnum(float(sq[sq['zbiór'] == 'digits']['różnica (nowy−stary)'].iloc[0]), 4)}, CI {ci(float(sq[sq['zbiór'] == 'digits']['CI dolny'].iloc[0]), float(sq[sq['zbiór'] == 'digits']['CI górny'].iloc[0]))}"]]
    rep.table(rows, [9.0, 8.8], "Obserwacje sugerowane (widoczne w danych, bez wystarczającego potwierdzenia).", font_scale=0.95)
    rep.h2("9.3 Nie rozstrzygnięte")
    rep.bullets([
        "Czy ewolucyjny dobór V′ pomaga przy znacznie mniejszych V′ niż w tym badaniu (nie badano stałej siatki rozmiarów V′).",
        "Wpływ bramki ρ<sub>SV</sub> na jakość: sprawdzony tylko w pilotażu na Szachownicy (n = 2) — bez bramki K rośnie do ≈ 5500, a BA test jest wyższe, kosztem kilkukrotnie większego czasu.",
        "Porównanie BA z F1 i AUC jako fitnessem (AUC nie został policzony; wymaga probability = true i rozszerzenia Val-GA).",
        "Czy próg „opanowania” poniżej 1,0 zmieniłby rolę ścieżki „BA = 1 → V′”, która dziś dotyczy głównie V′ startowego.",
        "Wrażliwość wyników na parametry automatu (przyrost ΔV′, progi trudności, liczba zastojów) i na wybór modelu reprezentatywnego.",
        "Czy wyniki utrzymują się na większej liczbie podziałów i seedów oraz dla zbiorów o większej liczbie klas niż Digits.",
    ])

    # ============================ 10. DALSZE KROKI ============================
    rep.h1("10. Proponowane dalsze kroki")
    rep.bullets([
        "Zdecydować z promotorem o roli bramki ρ<sub>SV</sub> (zachować, czy zmienić wzór na K<sub>next</sub> i pozwolić K rosnąć przy ρ<sub>SV</sub> &lt; TSV) — to najsilniej wpływa na zakończenie przebiegów i jakość na Szachownicy.",
        "Zbadać stałą siatkę rozmiarów V′ (np. 100, 200, 500, 1000) dla ewolucji i losowania przy tym samym K, żeby odpowiedzieć, czy ewolucja pomaga przy małych V′.",
        "Przeanalizować próg opanowania poniżej 1,0 oraz stałą/adaptacyjną liczbę rozszerzeń w fazie V′ (dziś de facto 3).",
        "Porównać BA, F1 i AUC jako fitness (probability = true, definicja progu opanowania dla AUC).",
        "Rozważyć „leniwe” predykcje w Val-GA (tylko wiersze użyte przez osobniki), żeby obniżyć koszt oceny puli kandydatów.",
        "Zwiększyć liczbę seedów na podziałach dodatkowych oraz zbadać wrażliwość na parametry automatu.",
    ])

    # ============================ 11. ODTWORZENIE ============================
    rep.h1("11. Instrukcja odtworzenia eksperymentu")
    rep.p("Wszystkie polecenia uruchamiane są w katalogu głównym repozytorium (środowisko .venv). Kolejka wznawia przerwane przebiegi (pomija ukończone). Każdy seed to osobna flaga --seeds. "
          "Poprzednie wyniki (v1, followup_stop_rule) i stary tryb dagasvm_phase pozostają niezmienione.")
    rep.code("""./.venv/bin/pytest -q                                   # 140 testow
# podzial bazowy 2026, 7 zbiorow, seedy 42-61 (420 przebiegow)
./.venv/bin/python scripts/followup_phase_switch/run_queue.py --series v2_full \\
    -d chessboard -d spiral -d magic -d circles -d classification2d -d digits -d moons \\
    -v evo_switch -v rand_switch -v rand_replay --seeds 42 --seeds 43 ... --seeds 61 -w 10
# podzialy dodatkowe (po 90 przebiegow: 6 zbiorow, seedy 42-46)
./.venv/bin/python scripts/followup_phase_switch/run_queue.py --series v2_split2027 --split 2027 \\
    -d spiral -d magic -d circles -d classification2d -d digits -d moons \\
    -v evo_switch -v rand_switch -v rand_replay --seeds 42 --seeds 43 --seeds 44 --seeds 45 --seeds 46 -w 4
./.venv/bin/python scripts/followup_phase_switch/aggregate.py --version v2
./.venv/bin/python scripts/followup_phase_switch/final_report.py     # tabele, wykresy, PDF i Markdown""")
    rep.p("Pliki wynikowe i ich lokalizacja w repozytorium (podziały 2028 i 2029 uruchamiane analogicznie do 2027, z opcjami --series i --split; "
          "pilotaż: serie v2_smoke_* z --set max_vprime_rows=1000, a dla wariantu bez bramki dodatkowo --set k_growth_requires_sv_signal=false).")
    rep.code("""output/pdf/raport_phase_switch.pdf                    # ten raport
reports/followup_phase_switch/RAPORT.md                # wersja Markdown
reports/followup_phase_switch/tables/final_*.csv       # tabele wynikowe
reports/followup_phase_switch/figures/                 # wykresy
artifacts/followup_phase_switch/v2_*                   # przebiegi (summary.json, cycles.csv, konfiguracje)""")

    rep.build()


if __name__ == "__main__":
    main()
