"""Minimalny szkielet PDF (reportlab) dla raportu `followup_stop_rule`: style, tabele z opisem, rysunki z opisem."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def fnum(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def fint(x):
    return "—" if x is None or pd.isna(x) else f"{x:,.0f}".replace(",", " ")


def pm(mean, sd, d=4):
    return f"{fnum(mean, d)} ± {fnum(sd, d)}"


def pval(p):
    if p is None or pd.isna(p):
        return "—"
    return "<0,001" if p < 0.001 else fnum(p, 3)


class Doc:
    def __init__(self, figs_dir: Path, out_pdf: Path, title: str):
        pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "Arial Unicode.ttf")))
        pdfmetrics.registerFont(TTFont("BodyBold", str(FONT_DIR / "Arial Bold.ttf")))
        pdfmetrics.registerFont(TTFont("BodyItalic", str(FONT_DIR / "Arial Italic.ttf")))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")
        ss = getSampleStyleSheet()
        ss.add(ParagraphStyle("Title2", parent=ss["Title"], fontName="BodyBold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4))
        ss.add(ParagraphStyle("Sub", parent=ss["BodyText"], fontName="Body", fontSize=9.2, leading=12, alignment=TA_CENTER,
                              textColor=colors.HexColor("#4B5563"), spaceAfter=8))
        ss.add(ParagraphStyle("H1", parent=ss["Heading1"], fontName="BodyBold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=5))
        ss.add(ParagraphStyle("H2", parent=ss["Heading2"], fontName="BodyBold", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3))
        ss.add(ParagraphStyle("P", parent=ss["BodyText"], fontName="Body", fontSize=9, leading=12.3, spaceAfter=4.5, alignment=4))
        ss.add(ParagraphStyle("Bul", parent=ss["P"], leftIndent=13, bulletIndent=2, spaceAfter=2.5))
        ss.add(ParagraphStyle("Cap", parent=ss["P"], fontSize=7.8, leading=10, textColor=colors.HexColor("#374151"), alignment=0, spaceAfter=2))
        ss.add(ParagraphStyle("Cell", parent=ss["P"], fontSize=7.4, leading=9.0, spaceAfter=0, alignment=0))
        ss.add(ParagraphStyle("CellH", parent=ss["Cell"], fontName="BodyBold", textColor=colors.white))
        ss.add(ParagraphStyle("Mono", parent=ss["P"], fontName="Courier", fontSize=7.3, leading=9.2, alignment=0, leftIndent=6, spaceAfter=0))
        ss.add(ParagraphStyle("Box", parent=ss["P"], fontSize=8.8, leading=12, backColor=colors.HexColor("#F3F6FA"), borderPadding=5,
                              borderColor=colors.HexColor("#C9D3E0"), borderWidth=0.5, spaceBefore=3, spaceAfter=8))
        self.ss, self.story, self.fig_no, self.tab_no = ss, [], 0, 0
        self.figs_dir, self.out_pdf, self.title = figs_dir, out_pdf, title

    # --- elementy ---
    def h1(self, t):
        self.story.append(Paragraph(t, self.ss["H1"]))

    def h2(self, t):
        self.story.append(Paragraph(t, self.ss["H2"]))

    def p(self, t, style="P"):
        self.story.append(Paragraph(t, self.ss[style]))

    def box(self, t):
        self.story.append(Paragraph(t, self.ss["Box"]))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(it, self.ss["Bul"], bulletText="•"))

    def code(self, text):
        for line in text.strip("\n").split("\n"):
            self.story.append(Paragraph(line.replace(" ", "&nbsp;").replace("<", "&lt;") or "&nbsp;", self.ss["Mono"]))
        self.story.append(Spacer(1, 5))

    def spacer(self, h=4):
        self.story.append(Spacer(1, h))

    def pagebreak(self):
        self.story.append(PageBreak())

    def table(self, rows, widths, caption, header=True, zebra=True, bold_rows=(), font_scale=1.0):
        self.tab_no += 1
        cell, cellh = self.ss["Cell"], self.ss["CellH"]
        if font_scale != 1.0:
            cell = ParagraphStyle("c2", parent=cell, fontSize=cell.fontSize * font_scale, leading=cell.leading * font_scale)
            cellh = ParagraphStyle("c3", parent=cellh, fontSize=cellh.fontSize * font_scale, leading=cellh.leading * font_scale)
        data = []
        for i, row in enumerate(rows):
            style = cellh if (header and i == 0) else cell
            data.append([Paragraph(f"<b>{c}</b>" if i in bold_rows else str(c), style) for c in row])
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0, hAlign="LEFT")
        st = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4D7DC")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")))
        if zebra:
            st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F4F5F7")) for r in range(2, len(rows), 2)]
        t.setStyle(TableStyle(st))
        cap = Paragraph(f"<b>Tabela {self.tab_no}.</b> {caption}", self.ss["Cap"])
        if len(rows) < 30:
            self.story.append(KeepTogether([cap, t, Spacer(1, 6)]))
        else:
            self.story.extend([cap, t, Spacer(1, 6)])

    def figure(self, name, caption, width_cm=17.2):
        path = self.figs_dir / name
        if not path.exists():
            raise FileNotFoundError(f"brak wykresu {path} — raport nie może odwoływać się do nieistniejących rysunków")
        w, h = PILImage.open(path).size
        self.fig_no += 1
        img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
        self.story.append(KeepTogether([img, Paragraph(f"<b>Rys. {self.fig_no}.</b> {caption}", self.ss["Cap"]), Spacer(1, 6)]))

    def build(self):
        self.out_pdf.parent.mkdir(parents=True, exist_ok=True)

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("Body", 7.3)
            canvas.setFillColor(colors.HexColor("#59636E"))
            canvas.drawString(1.6 * cm, 1.0 * cm, self.title)
            canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"strona {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(str(self.out_pdf), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                                topMargin=1.4 * cm, bottomMargin=1.7 * cm, title=self.title, author="Paweł Pluta")
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)
        print("zapisano", self.out_pdf)
