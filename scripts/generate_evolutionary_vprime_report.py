from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = REPO_ROOT / "artifacts" / "sweeps" / "20260922_204113_256002_dagasvm_phase_evolutionary_seeds20"
INPUT_CSV = SWEEP_ROOT / "representative_metrics.csv"
OUT_DIR = REPO_ROOT / "output" / "pdf"
TMP_DIR = REPO_ROOT / "tmp" / "pdfs" / "evolutionary_vprime"
OUT_PDF = OUT_DIR / "raport_ewolucja_vprime_20_powtorzen.pdf"


def _register_fonts() -> tuple[str, str]:
    regular = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial.ttf"
    pdfmetrics.registerFont(TTFont("ReportRegular", regular))
    pdfmetrics.registerFont(TTFont("ReportBold", bold))
    return "ReportRegular", "ReportBold"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _summary(df: pd.DataFrame, column: str) -> dict[str, float]:
    values = df[column].astype(float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _save_stats_csv(df: pd.DataFrame) -> Path:
    fields = [
        "metric",
        "mean",
        "sd",
        "min",
        "max",
    ]
    rows = []
    for column in [
        "test_balanced_accuracy",
        "full_validation_balanced_accuracy",
        "current_v_balanced_accuracy",
        "cycles",
        "final_k",
        "training_rows",
        "validation_rows",
        "num_support_vectors",
        "duration_sec",
        "expand_validation_only",
        "increase_k",
        "forced_growth_after_patience",
        "stable_since_hardening",
    ]:
        item = _summary(df, column)
        rows.append({"metric": column, **item})

    path = SWEEP_ROOT / "representative_metrics_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_quality(df: pd.DataFrame) -> Path:
    path = TMP_DIR / "quality_by_seed.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=180)
    ax.plot(df["seed"], df["test_balanced_accuracy"], marker="o", label="test balanced accuracy", linewidth=2)
    ax.plot(
        df["seed"],
        df["full_validation_balanced_accuracy"],
        marker="s",
        label="full validation balanced accuracy",
        linewidth=2,
    )
    ax.axhline(df["test_balanced_accuracy"].mean(), color="#1f77b4", alpha=0.25, linestyle="--")
    ax.axhline(df["full_validation_balanced_accuracy"].mean(), color="#ff7f0e", alpha=0.25, linestyle="--")
    ax.set_title("Jakość końcowa w 20 powtórzeniach")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.962, 0.982)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cost(df: pd.DataFrame) -> Path:
    path = TMP_DIR / "cost_and_size_by_seed.png"
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    ax2 = ax1.twinx()
    ax1.bar(df["seed"], df["cycles"], color="#4C78A8", alpha=0.65, label="liczba cykli")
    ax2.plot(df["seed"], df["validation_rows"], color="#F58518", marker="o", linewidth=2, label="|V'|")
    ax2.plot(df["seed"], df["training_rows"], color="#54A24B", marker="s", linewidth=2, label="|T'|")
    ax1.set_title("Koszt przebiegu i końcowe rozmiary T' oraz V'")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Liczba cykli")
    ax2.set_ylabel("Liczba próbek")
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_decisions(df: pd.DataFrame) -> Path:
    path = TMP_DIR / "decisions_by_seed.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    seeds = df["seed"].astype(str)
    bottom = [0] * len(df)
    for column, color, label in [
        ("expand_validation_only", "#72B7B2", "rozszerzenie V' bez wzrostu K"),
        ("increase_k", "#E45756", "zwiększenie K"),
        ("forced_growth_after_patience", "#B279A2", "wzrost wymuszony po Ts"),
    ]:
        values = df[column].astype(float)
        ax.bar(seeds, values, bottom=bottom, color=color, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_title("Decyzje fazowe w kolejnych runach")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Liczba decyzji")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_scatter(df: pd.DataFrame) -> Path:
    path = TMP_DIR / "cost_quality_scatter.png"
    fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=180)
    sizes = 28 + (df["validation_rows"] - df["validation_rows"].min()) / (
        df["validation_rows"].max() - df["validation_rows"].min()
    ) * 140
    scatter = ax.scatter(
        df["cycles"],
        df["test_balanced_accuracy"],
        s=sizes,
        c=df["final_k"],
        cmap="viridis",
        alpha=0.82,
        edgecolor="#263238",
        linewidth=0.5,
    )
    for _, row in df.iterrows():
        ax.annotate(str(int(row["seed"])), (row["cycles"], row["test_balanced_accuracy"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Relacja kosztu do jakości")
    ax.set_xlabel("Liczba cykli")
    ax.set_ylabel("Test balanced accuracy")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("K końcowe")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _table(data: list[list[str]], widths: list[float], header: bool = True) -> Table:
    table = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "ReportRegular"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4D7DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "ReportBold"),
            ]
        )
    table.setStyle(TableStyle(style))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("ReportRegular", 8)
    canvas.setFillColor(colors.HexColor("#59636E"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "DAGASVM - ewolucyjny dobór V' - 20 powtórzeń")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"strona {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    regular, bold = _register_fonts()
    df = pd.read_csv(INPUT_CSV).sort_values("seed")
    _save_stats_csv(df)

    charts = [_plot_quality(df), _plot_cost(df), _plot_decisions(df), _plot_scatter(df)]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitlePL", parent=styles["Title"], fontName=bold, fontSize=21, leading=26, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="HeadingPL", parent=styles["Heading1"], fontName=bold, fontSize=14, leading=18, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="Heading2PL", parent=styles["Heading2"], fontName=bold, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyPL", parent=styles["BodyText"], fontName=regular, fontSize=9.5, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallPL", parent=styles["BodyText"], fontName=regular, fontSize=8.2, leading=11, spaceAfter=4, textColor=colors.HexColor("#4B5563")))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="Raport - ewolucja V prime",
        author="Pawel Pluta",
    )

    story = []
    story.append(Paragraph("Raport z badania ewolucyjnego doboru V'", styles["TitlePL"]))
    story.append(Paragraph("DAGASVM phase, Chessboard, 20 powtórzeń dla seedów 42-61", styles["SmallPL"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Uwagi/obserwacje własne autora bez użycia LLM'a", styles["HeadingPL"]))
    own_notes = [
        "Wariant ewolucyjny V' został sprawdzony nie tylko pojedynczo, ale na serii 20 powtórzeń. Wyniki jakościowe są dość stabilne: model zwykle dochodzi do bardzo podobnego poziomu na teście i pełnej walidacji.",
        "O ile jakość końcowa zmienia się stosunkowo niewiele, o tyle koszt dojścia do tego wyniku jest bardziej zmienny. Niektóre seedy kończą po 44 cyklach, a najdłuższy potrzebuje 81 cykli. Metoda jest więc stabilna jakościowo, ale kosztowo potrafi być mniej przewidywalna.",
        "Widać też, że zastosowany mechanizm 'najpierw utrudnij V', potem ewentualnie zwiększ K' faktycznie ogranicza automatyczny wzrost K. Decyzji o rozszerzaniu V' jest dużo więcej niż decyzji o wzroście K.",
        "Na ten moment traktowałbym ewolucyjny dobór V' jako sensowny wariant do pracy: nie daje jednego przypadkowego wyniku, tylko powtarzalne rezultaty. Naturalną kontynuacją eksperymentów byłaby próba na tych samych seedach dla wariantu heurystycznego.",
    ]
    for note in own_notes:
        story.append(Paragraph(note, styles["BodyPL"]))

    story.append(Paragraph("Skrót wyników", styles["HeadingPL"]))
    metric_rows = [
        ["Metryka", "Srednia", "Odch.", "Min", "Max"],
    ]
    for label, column, digits in [
        ("Test balanced accuracy", "test_balanced_accuracy", 4),
        ("Pełna walidacja balanced accuracy", "full_validation_balanced_accuracy", 4),
        ("Aktualne V' balanced accuracy", "current_v_balanced_accuracy", 4),
        ("Liczba cykli", "cycles", 1),
        ("K końcowe", "final_k", 1),
        ("Rozmiar T'", "training_rows", 1),
        ("Rozmiar V'", "validation_rows", 1),
        ("Liczba SV", "num_support_vectors", 1),
        ("Czas runu [s]", "duration_sec", 1),
    ]:
        s = _summary(df, column)
        metric_rows.append([label, _fmt(s["mean"], digits), _fmt(s["sd"], digits), _fmt(s["min"], digits), _fmt(s["max"], digits)])
    story.append(_table(metric_rows, [6.1 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Najważniejsze obserwacje", styles["HeadingPL"]))
    observations = [
        "Jakość końcowa jest stabilna: test balanced accuracy ma średnią 0.9744 i odchylenie 0.0028.",
        "Pełna walidacja ma podobną stabilność: średnia balanced accuracy 0.9710, odchylenie 0.0025.",
        "Koszt jest mniej stabilny: liczba cykli waha się od 44 do 81, a czas runu od ok. 10.9 do 26.3 min.",
        "Końcowe K zwykle trafia w podobny zakres 782-869, średnio 836.7.",
        "Rozmiar końcowego V' zmienia się mocniej: od 3024 do 3826 próbek.",
        "Każdy run zakończył się decyzją stop_no_growth_signal, czyli algorytm nie zatrzymał się przez błąd lub brak kandydatów.",
    ]
    for obs in observations:
        story.append(Paragraph("- " + obs, styles["BodyPL"]))

    story.append(PageBreak())
    story.append(Paragraph("Wykresy", styles["HeadingPL"]))
    story.append(Paragraph("Jakość końcowa jest ciasno skupiona, natomiast przebieg i rozmiary zbiorów robią się bardziej zależne od seeda.", styles["BodyPL"]))
    for chart in charts[:2]:
        story.append(Image(str(chart), width=17.0 * cm, height=8.1 * cm))
        story.append(Spacer(1, 0.25 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Decyzje fazowe i relacja koszt-jakość", styles["HeadingPL"]))
    for chart in charts[2:]:
        story.append(Image(str(chart), width=17.0 * cm, height=8.5 * cm))
        story.append(Spacer(1, 0.25 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Tabela runów", styles["HeadingPL"]))
    run_rows = [["Seed", "Cykle", "K", "|T'|", "|V'|", "SV", "bAcc test", "bAcc pełne V", "Czas [min]"]]
    for _, row in df.iterrows():
        run_rows.append(
            [
                str(int(row["seed"])),
                str(int(row["cycles"])),
                str(int(row["final_k"])),
                str(int(row["training_rows"])),
                str(int(row["validation_rows"])),
                str(int(row["num_support_vectors"])),
                _fmt(float(row["test_balanced_accuracy"]), 4),
                _fmt(float(row["full_validation_balanced_accuracy"]), 4),
                _fmt(float(row["duration_sec"]) / 60.0, 1),
            ]
        )
    story.append(_table(run_rows, [1.3 * cm, 1.4 * cm, 1.2 * cm, 1.4 * cm, 1.5 * cm, 1.2 * cm, 2.0 * cm, 2.2 * cm, 1.7 * cm]))

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Notatka metodologiczna", styles["HeadingPL"]))
    story.append(
        Paragraph(
            "Raport opiera się na pliku representative_metrics.csv wygenerowanym po sweepie 20 seedów. "
            "Do wniosków użyto metryk reprezentatywnego modelu z ostatniego ukończonego cyklu, a nie rankingu po report_row.full_validation_f1, "
            "ponieważ ranking miał identyczną wartość objective dla wszystkich runów i nie różnicował faktycznych przebiegów.",
            styles["BodyPL"],
        )
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUT_PDF)
