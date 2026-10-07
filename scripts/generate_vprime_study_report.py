"""Generator raportu PDF z badania doboru V' (po polsku). Wszystkie liczby pochodzą z tabel
reports/vprime_study/tables/*.csv (zbudowanych przez scripts/aggregate_vprime_study.py z metryk per-run),
a wykresy z reports/vprime_study/figures/*.png (scripts/plot_vprime_study.py).

Użycie:  python scripts/generate_vprime_study_report.py
Wynik:   output/pdf/raport_vprime_badanie.pdf
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
from ga.study_analysis import VARIANT_LABELS, load_spec  # noqa: E402

TABLES = REPO_ROOT / "reports" / "vprime_study" / "tables"
FIGS = REPO_ROOT / "reports" / "vprime_study" / "figures"
OUT_PDF = REPO_ROOT / "output" / "pdf" / "raport_vprime_badanie.pdf"

DATASET_LABELS = {
    "chessboard": "Szachownica", "moons": "Moons", "circles": "Circles", "spiral": "Spiral",
    "classification2d": "Classification 2D", "digits": "Digits", "magic": "MAGIC",
}
VARIANT_SHORT = {
    "evo_mem": "ewol. + pamięć", "evo_nomem": "ewol. bez pamięci", "rand_fixed": "losowy stały",
    "rand_growth": "losowy rosnący", "heur_mem": "heurystyka błędów",
}


# ------------------------------------------------------------------------------------------
# dane
# ------------------------------------------------------------------------------------------
class Data:
    def __init__(self):
        self.per_run = pd.read_csv(TABLES / "per_run.csv")
        self.per_cycle = pd.read_csv(TABLES / "per_cycle.csv")
        self.summary = pd.read_csv(TABLES / "summary_by_series.csv")
        self.paired = pd.read_csv(TABLES / "paired_comparisons.csv")
        self.validation = pd.read_csv(TABLES / "series_validation.csv")
        self.val_ga = pd.read_csv(TABLES / "val_ga_search.csv")
        self.legacy = json.loads((TABLES / "legacy_ranking_audit.json").read_text(encoding="utf-8"))
        reg = TABLES / "regression_vs_legacy_series.csv"
        self.regression = pd.read_csv(reg) if reg.exists() else None
        self.spec = load_spec()

    def stat(self, dataset, variant, metric, key="mean"):
        row = self.summary[(self.summary.dataset == dataset) & (self.summary.variant == variant) & (self.summary.metric == metric)]
        return float(row[key].iloc[0]) if len(row) else float("nan")

    def pair(self, dataset, a, b, metric):
        row = self.paired[(self.paired.dataset == dataset) & (self.paired.a == a) & (self.paired.b == b) & (self.paired.metric == metric)]
        return row.iloc[0] if len(row) else None

    def n_runs(self, dataset, variant):
        return int(((self.per_run.dataset == dataset) & (self.per_run.variant == variant)).sum())

    def have(self, dataset, variant):
        return self.n_runs(dataset, variant) > 0

    def incomplete_series(self):
        """Zwraca serie, dla których nie ma kompletu 20 powtórzeń."""
        complete = self.validation["complete_20_seeds"].astype(str).str.lower().eq("true")
        return self.validation.loc[~complete, ["dataset", "variant", "n_runs", "issues"]]


def f(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def fi(x):
    return "—" if x is None or pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def pm(mean, sd, d=4):
    return f"{f(mean, d)} ± {f(sd, d)}"


def pval(p):
    if p is None or pd.isna(p):
        return "—"
    return "<0,001" if p < 0.001 else f(p, 3)


# ------------------------------------------------------------------------------------------
# style PDF
# ------------------------------------------------------------------------------------------
def register_fonts():
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Body", str(base / "Arial Unicode.ttf")))
    pdfmetrics.registerFont(TTFont("BodyBold", str(base / "Arial Bold.ttf")))
    pdfmetrics.registerFont(TTFont("BodyItalic", str(base / "Arial Italic.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")


def make_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Title2", parent=ss["Title"], fontName="BodyBold", fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=8))
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontName="BodyBold", fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=6))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontName="BodyBold", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4))
    ss.add(ParagraphStyle("P", parent=ss["BodyText"], fontName="Body", fontSize=9.4, leading=13, spaceAfter=5, alignment=4))
    ss.add(ParagraphStyle("Bul", parent=ss["P"], leftIndent=14, bulletIndent=3, spaceAfter=3))
    ss.add(ParagraphStyle("Cap", parent=ss["P"], fontSize=8, leading=10.5, textColor=colors.HexColor("#4B5563"), alignment=0))
    ss.add(ParagraphStyle("Cell", parent=ss["P"], fontSize=7.8, leading=9.6, spaceAfter=0, alignment=0))
    ss.add(ParagraphStyle("CellB", parent=ss["Cell"], fontName="BodyBold"))
    ss.add(ParagraphStyle("CellH", parent=ss["Cell"], fontName="BodyBold", textColor=colors.white))
    ss.add(ParagraphStyle("Mono", parent=ss["P"], fontName="Courier", fontSize=7.8, leading=10, alignment=0, leftIndent=6))
    return ss


class Doc:
    def __init__(self):
        register_fonts()
        self.ss = make_styles()
        self.story = []
        self.fig_no = 0
        self.tab_no = 0

    # --- elementy ---
    def h1(self, text):
        self.story.append(Paragraph(text, self.ss["H1"]))

    def h2(self, text):
        self.story.append(Paragraph(text, self.ss["H2"]))

    def p(self, text):
        self.story.append(Paragraph(text, self.ss["P"]))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(it, self.ss["Bul"], bulletText="•"))
            self.story.append(Spacer(1, 2))

    def code(self, text):
        for line in text.strip("\n").split("\n"):
            self.story.append(Paragraph(line.replace(" ", "&nbsp;") or "&nbsp;", self.ss["Mono"]))
        self.story.append(Spacer(1, 4))

    def table(self, rows, widths, caption=None, header=True, align_right_from=1, zebra=True):
        self.tab_no += 1
        data = []
        for i, row in enumerate(rows):
            style = self.ss["CellH"] if (header and i == 0) else self.ss["Cell"]
            data.append([Paragraph(str(c), style) for c in row])
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0, hAlign="LEFT")
        st = [
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4D7DC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")))
        if zebra:
            for r in range(1, len(rows)):
                if r % 2 == 0:
                    st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F4F5F7")))
        t.setStyle(TableStyle(st))
        block = []
        if caption:
            block.append(Paragraph(f"<b>Tabela {self.tab_no}.</b> {caption}", self.ss["Cap"]))
        block.append(t)
        block.append(Spacer(1, 6))
        self.story.append(KeepTogether(block) if len(rows) < 28 else block[0])
        if len(rows) >= 28:
            self.story.extend(block[1:])

    def figure(self, name, caption, width_cm=17.0):
        path = FIGS / name
        if not path.exists():
            self.story.append(Paragraph(f"[brak wykresu {name}]", self.ss["Cap"]))
            return
        from PIL import Image as PILImage

        w, h = PILImage.open(path).size
        self.fig_no += 1
        img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
        self.story.append(KeepTogether([img, Paragraph(f"<b>Rys. {self.fig_no}.</b> {caption}", self.ss["Cap"]), Spacer(1, 6)]))

    def pagebreak(self):
        self.story.append(PageBreak())

    def build(self, title):
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Body", 7.5)
            canvas.setFillColor(colors.HexColor("#59636E"))
            canvas.drawString(1.6 * cm, 1.0 * cm, title)
            canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"strona {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                                topMargin=1.5 * cm, bottomMargin=1.7 * cm, title=title, author="Paweł Pluta")
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)
        print("zapisano", OUT_PDF)


if __name__ == "__main__":
    from report_text import add_all  # noqa: E402  (treść raportu; liczby z tabel)

    parser = argparse.ArgumentParser(description="Generuje raport PDF z badania V'.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="wygeneruj raport roboczy mimo brakujących seedów; nie używaj do raportu końcowego",
    )
    args = parser.parse_args()
    data = Data()
    incomplete = data.incomplete_series()
    if not incomplete.empty and not args.allow_incomplete:
        details = "; ".join(
            f"{row.dataset}/{row.variant}: {int(row.n_runs)}/20"
            for row in incomplete.itertuples(index=False)
        )
        raise SystemExit(
            "Nie wygenerowano końcowego raportu: niekompletne serie ("
            f"{details}). Po zakończeniu kolejki uruchom agregację i generator ponownie. "
            "Do raportu roboczego użyj jawnie --allow-incomplete."
        )
    doc = Doc()
    add_all(doc, data)
    doc.build("DAGASVM — dobór podzbioru walidacyjnego V' (raport badawczy)")
