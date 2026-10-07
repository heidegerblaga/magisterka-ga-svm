from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons

from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import EarlyStopping, GAConfig, run_ga_k_per_class
from ga.models.svm import SVMConfig, fit_svm_model
from ga.ui.data import find_summary_by_run_id, load_history_with_metadata, load_summary


SLIDES = [
    "Problem i motywacja",
    "GASVM: selekcja treningu",
    "DAGASVM: selekcja walidacji",
    "Realne wyniki",
    "Wnioski do obrony",
]


def _hero(slide_label: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="eyebrow">{slide_label}</div>
          <h2 class="hero-title">{title}</h2>
          <p class="hero-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _spotlight(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="spotlight-card">
          <div class="spotlight-title">{title}</div>
          <div class="spotlight-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="signal-card">
          <div class="signal-label">{label}</div>
          <div class="signal-value">{value}</div>
          <div class="signal-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _formula(label: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="formula-card">
          <div class="formula-title">{label}</div>
          <div class="formula-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _story_steps(cards: list[tuple[str, str]]) -> None:
    html = ['<div class="story-grid">']
    for idx, (title, copy) in enumerate(cards, start=1):
        html.append(
            f"""
            <div class="process-step">
              <div class="process-index">{idx}</div>
              <div class="process-title">{title}</div>
              <div class="process-copy">{copy}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def demo_gasvm_payload():
    X, y = make_moons(n_samples=320, noise=0.22, random_state=42)
    X_pool, y_pool = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    kcfg = KPerClassConfig(k_per_class=22, random_state=42)
    svm_cfg = SVMConfig(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    fcfg = FitnessConfig(metric="balanced_accuracy", class_imbalance_penalty=0.0)
    gcfg = GAConfig(
        population_size=24,
        generations=10,
        tournament_k=3,
        crossover_rate=0.8,
        mutation_rate=0.05,
        elitism=2,
        early_stopping=EarlyStopping(patience=4, min_delta=1e-3),
    )

    history, best_train_mask = run_ga_k_per_class(
        X_pool=X_pool,
        y_pool=y_pool,
        X_val=X_val,
        y_val=y_val,
        kcfg=kcfg,
        svm_cfg=svm_cfg,
        fcfg=fcfg,
        gcfg=gcfg,
        seed=42,
    )
    clf = fit_svm_model(X_pool[best_train_mask], y_pool[best_train_mask], svm_cfg)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    decision = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    return {
        "X_pool": X_pool,
        "y_pool": y_pool,
        "X_val": X_val,
        "y_val": y_val,
        "best_mask": best_train_mask,
        "history": [float(x) for x in history],
        "xx": xx,
        "yy": yy,
        "decision": decision,
    }


def _select_story_figure(payload) -> go.Figure:
    X_pool = payload["X_pool"]
    y_pool = payload["y_pool"]
    X_val = payload["X_val"]
    y_val = payload["y_val"]
    best_mask = payload["best_mask"]
    xx = payload["xx"]
    yy = payload["yy"]
    decision = payload["decision"]

    pool_gray = go.Scatter(
        x=X_pool[:, 0],
        y=X_pool[:, 1],
        mode="markers",
        name="Pula treningowa",
        marker=dict(size=9, color="rgba(16,36,24,0.30)"),
    )
    pool_colored = go.Scatter(
        x=X_pool[:, 0],
        y=X_pool[:, 1],
        mode="markers",
        name="Kandydaci z etykietami",
        marker=dict(size=10, color=y_pool, colorscale="Tealrose", line=dict(width=0.5, color="#102418")),
    )
    selected = go.Scatter(
        x=X_pool[best_mask, 0],
        y=X_pool[best_mask, 1],
        mode="markers",
        name="Wybrany podzbiór",
        marker=dict(size=12, color=y_pool[best_mask], colorscale="Tealrose", line=dict(width=1.2, color="#102418")),
    )
    rejected = go.Scatter(
        x=X_pool[~best_mask, 0],
        y=X_pool[~best_mask, 1],
        mode="markers",
        name="Próbki odrzucone",
        marker=dict(size=8, color="rgba(16,36,24,0.15)"),
    )
    validation = go.Scatter(
        x=X_val[:, 0],
        y=X_val[:, 1],
        mode="markers",
        name="Stała walidacja",
        marker=dict(size=10, color=y_val, colorscale="Tealrose", symbol="x", line=dict(width=1.4, color="#102418")),
    )
    contour = go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=decision,
        colorscale="BrBG",
        opacity=0.26,
        showscale=False,
        hoverinfo="skip",
        contours=dict(showlines=False),
        name="Granica decyzyjna",
    )

    frames = [
        go.Frame(name="0", data=[pool_gray, go.Scatter(), go.Scatter(), go.Scatter(), go.Contour(z=[[0, 0], [0, 0]], opacity=0.0, showscale=False, hoverinfo="skip", contours=dict(showlines=False))]),
        go.Frame(name="1", data=[go.Scatter(), pool_colored, go.Scatter(), go.Scatter(), go.Contour(z=[[0, 0], [0, 0]], opacity=0.0, showscale=False, hoverinfo="skip", contours=dict(showlines=False))]),
        go.Frame(name="2", data=[rejected, go.Scatter(), selected, go.Scatter(), go.Contour(z=[[0, 0], [0, 0]], opacity=0.0, showscale=False, hoverinfo="skip", contours=dict(showlines=False))]),
        go.Frame(name="3", data=[rejected, go.Scatter(), selected, validation, contour]),
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )
    fig.update_layout(
        title="Animacja: od pełnej puli do najlepszego małego TRAIN",
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", y=1.08),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.17,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 700, "redraw": True}, "transition": {"duration": 350}, "fromcurrent": True}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Etap: "},
                "steps": [
                    {"label": label, "method": "animate", "args": [[str(idx)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for idx, label in enumerate(["pełna pula", "ocena kandydatów", "najlepszy podzbiór", "SVM + walidacja"])
                ],
            }
        ],
    )
    return fig


def _history_line(values: list[float], title: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=9),
            fill="tozeroy",
            fillcolor="rgba(31,122,79,0.10)",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=330,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Generacja",
        yaxis_title="Fitness",
    )
    return fig


def _history_summary() -> pd.DataFrame:
    return load_history_with_metadata()


def _display_label(row: pd.Series | dict) -> str:
    if isinstance(row, pd.Series):
        experiment_name = row.get("experiment_name")
        dataset = row.get("dataset")
    else:
        experiment_name = row.get("experiment_name")
        dataset = row.get("dataset")
    return str(experiment_name or dataset or "-")


def _full_vs_ga_reference(history: pd.DataFrame) -> tuple[int, int, float, float]:
    if history.empty:
        return 20000, 200, 0.0, 0.0

    full = history[history["algorithm"] == "Full SVM"]
    ga = history[history["algorithm"] != "Full SVM"]
    full_n = int(full["training_set_size"].max()) if not full.empty else int(history["training_set_size"].max())
    ga_n = int(ga["training_set_size"].min()) if not ga.empty else max(10, full_n // 40)
    full_time = float(full["training_time_sec"].median()) if not full.empty else float(history["training_time_sec"].max())
    ga_time = float(ga["training_time_sec"].median()) if not ga.empty else float(history["training_time_sec"].min())
    return full_n, ga_n, full_time, ga_time


def _complexity_animation_figure(full_n: int, subset_n: int) -> go.Figure:
    subset_n = max(10, subset_n)
    full_n = max(subset_n + 1, full_n)
    sizes = np.linspace(subset_n, full_n, 16).astype(int)
    relative_cost = (sizes / subset_n) ** 3
    highlight_y = (full_n / subset_n) ** 3

    frames = []
    for idx in range(1, len(sizes) + 1):
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(
                        x=sizes[:idx],
                        y=relative_cost[:idx],
                        mode="lines+markers",
                        line=dict(color="#0d5b4f", width=5),
                        marker=dict(size=10, color="#f0c48e"),
                    ),
                    go.Scatter(
                        x=[subset_n, full_n],
                        y=[1, highlight_y],
                        mode="markers+text",
                        text=["mały podzbiór", "pełny zbiór"],
                        textposition="top center",
                        marker=dict(size=[14, 18], color=["#1f7a4f", "#d66c2f"]),
                    ),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Dlaczego pełny kernel SVM boli przy dużym n",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis_title="Liczba próbek n",
        yaxis_title="Względny koszt treningu ~ (n / n_subset)^3",
        yaxis_type="log",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 420, "redraw": True}, "transition": {"duration": 180}, "fromcurrent": True}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Rozwijanie n: "},
                "steps": [
                    {"label": str(size), "method": "animate", "args": [[str(idx)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for idx, size in enumerate(sizes, start=1)
                ],
            }
        ],
    )
    return fig


def _load_best_dagasvm_summary():
    history = _history_summary()
    if history.empty:
        return None
    daga = history[history["algorithm"] == "DAGASVM"]
    if daga.empty:
        return None
    best_row = daga.sort_values("full_validation_f1", ascending=False).iloc[0]
    summary_path = find_summary_by_run_id(str(best_row["run_id"]))
    if summary_path is None or not summary_path.exists():
        return None
    return load_summary(summary_path)


def _dagasvm_animation_figure(summary: dict) -> go.Figure:
    rows = []
    for cycle in summary.get("cycles") or []:
        val_ga = cycle.get("val_ga") or {}
        rows.append(
            {
                "cycle": int(cycle.get("cycle")),
                "before": int(val_ga.get("global_val_rows_before", 0)),
                "after": int(val_ga.get("global_val_rows_after", 0)),
                "added": int(val_ga.get("global_val_rows_added", 0)),
                "train_best": float((cycle.get("train_ga") or {}).get("history_summary", {}).get("best_final_fitness", 0.0)),
                "val_best": float((val_ga.get("history_summary") or {}).get("best_final_fitness", 0.0)),
            }
        )
    cycle_df = pd.DataFrame(rows)

    frames = []
    for _, row in cycle_df.iterrows():
        frames.append(
            go.Frame(
                name=str(int(row["cycle"])),
                data=[
                    go.Bar(
                        x=["przed", "po"],
                        y=[row["before"], row["after"]],
                        marker_color=["#cfe8d9", "#1f7a4f"],
                        text=[int(row["before"]), int(row["after"])],
                        textposition="outside",
                        name="walidacja",
                    ),
                    go.Scatter(
                        x=cycle_df["cycle"][cycle_df["cycle"] <= row["cycle"]],
                        y=cycle_df["train_best"][cycle_df["cycle"] <= row["cycle"]],
                        mode="lines+markers",
                        line=dict(color="#0d5b4f", width=4),
                        marker=dict(size=9),
                        name="TRAIN-GA fitness",
                        yaxis="y2",
                    ),
                ],
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Animacja: walidacja rośnie, bo dokładamy próbki najtrudniejsze dla modelu",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis_title="Stan walidacji w cyklu",
        yaxis_title="Liczba próbek w walidacji",
        yaxis2=dict(title="TRAIN-GA best fitness", overlaying="y", side="right", showgrid=False),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.14,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 750, "redraw": True}, "transition": {"duration": 250}, "fromcurrent": True}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Cykl: "},
                "steps": [
                    {"label": str(int(cycle)), "method": "animate", "args": [[str(int(cycle))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                    for cycle in cycle_df["cycle"]
                ],
            }
        ],
    )
    return fig


def _results_bubble_figure(history: pd.DataFrame) -> go.Figure:
    color_map = {
        "Full SVM": "#d66c2f",
        "GASVM": "#1f7a4f",
        "DAGASVM": "#0d5b4f",
    }
    fig = go.Figure()
    for algorithm in history["algorithm"].dropna().unique():
        subset = history[history["algorithm"] == algorithm]
        labels = subset["experiment_name"].fillna(subset["dataset"]) if "experiment_name" in subset.columns else subset["dataset"]
        created_at = subset["created_at"].fillna("-") if "created_at" in subset.columns else pd.Series("-", index=subset.index)
        fig.add_trace(
            go.Scatter(
                x=subset["training_time_sec"],
                y=subset["full_validation_f1"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                name=algorithm,
                marker=dict(
                    size=np.clip(subset["num_support_vectors"].fillna(1).astype(float) / 10.0, 12, 42),
                    color=color_map.get(algorithm, "#6b7b6d"),
                    opacity=0.82,
                    line=dict(width=1, color="#102418"),
                ),
                customdata=np.column_stack(
                    [
                        subset["training_set_size"].to_numpy(),
                        subset["num_support_vectors"].to_numpy(),
                        subset["dataset"].fillna("-").astype(str).to_numpy(),
                        labels.fillna("-").astype(str).to_numpy(),
                        created_at.astype(str).to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "eksperyment=%{customdata[3]}"
                    "<br>dataset=%{customdata[2]}"
                    "<br>data=%{customdata[4]}"
                    "<br>czas=%{x:.1f}s"
                    "<br>F1=%{y:.3f}"
                    "<br>train=%{customdata[0]}"
                    "<br>SV=%{customdata[1]}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title="Mapa kompromisu: czas vs jakość vs liczba support vectors",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis_title="Pełny czas runu [s]",
        yaxis_title="F1 na pełnym validation",
    )
    return fig


def _seminar_note(text: str) -> None:
    st.markdown(f'<div class="seminar-note">{text}</div>', unsafe_allow_html=True)


def _render_problem_slide():
    history = _history_summary()
    full_n, ga_n, full_time, ga_time = _full_vs_ga_reference(history)
    theoretical_gain = (full_n / max(ga_n, 1)) ** 3

    _hero(
        "Seminarium · slide 1",
        "Punkt wyjścia: SVM potrafi być bardzo mocny, ale trening na dużym zbiorze szybko staje się kosztowny",
        "Dla kernelowego SVM koszt treningu rośnie bardzo agresywnie wraz z liczbą próbek. W praktyce można myśleć o tym jak o skali rzędu O(n³), więc nawet sensowny model bazowy zaczyna boleć, kiedy zbiór robi się duży. To jest miejsce, w którym selekcja danych przestaje być dodatkiem, a staje się częścią samego algorytmu.",
    )

    top = st.columns([1.18, 0.82])
    with top[0]:
        st.plotly_chart(_complexity_animation_figure(full_n=full_n, subset_n=ga_n), use_container_width=True)
    with top[1]:
        _spotlight(
            "Intuicja problemu",
            "Jeżeli pełny SVM widzi wszystkie próbki naraz, to płaci za nie nie tylko liniowo, ale dużo ostrzej. Dlatego zamiast pytać 'czy użyć wszystkich danych?', lepiej pytać 'które dane naprawdę niosą informację o granicy decyzyjnej?'."
        )
        _formula("Koszt obliczeń", "trening kernel SVM  ~  O(n³)")
        _formula("Cel pracy", "wybrać mały, ale mocny podzbiór zamiast trenować ślepo na całości")

    metrics = st.columns(3)
    with metrics[0]:
        _signal("Pełny train", f"{full_n}", "próbek w pełnym wariancie referencyjnym")
    with metrics[1]:
        _signal("Mały train GA", f"{ga_n}", "próbek, które mają wystarczyć do zbudowania dobrego modelu")
    with metrics[2]:
        _signal("Skala teoretyczna", f"{theoretical_gain:,.0f}×", "orientacyjny mnożnik kosztu gdy przejdziesz z małego podzbioru do pełnego zbioru")

    st.markdown("### Oś narracji")
    _story_steps(
        [
            (
                "Full SVM jako baseline",
                "Najpierw pokazujesz klasyczny punkt odniesienia: uczymy model na pełnym train i sprawdzamy jakość, czas oraz liczbę support vectors.",
            ),
            (
                "GASVM redukuje trening",
                "Potem wchodzi algorytm genetyczny, którego zadaniem nie jest zgadywanie etykiet, tylko znalezienie takiego małego podzbioru TRAIN, który zachowa jakość modelu możliwie blisko pełnego wariantu.",
            ),
            (
                "DAGASVM wzmacnia walidację",
                "Na końcu dokładamy jeszcze dobór trudnych próbek walidacyjnych, żeby iteracyjnie pokazać modelowi te przypadki, na których wcześniej przegrywał.",
            ),
        ]
    )

    _seminar_note(
        "Co powiedzieć: 'Nie próbuję poprawiać SVM przez zmianę samego klasyfikatora. Zmieniam pytanie z pełnego trenowania na problem selekcji danych: które próbki warto zostawić, żeby zachować informację, a ograniczyć koszt treningu?'."
    )


def _render_gasvm_slide():
    payload = demo_gasvm_payload()
    reduction = int(payload["best_mask"].sum())
    full_pool = int(payload["best_mask"].shape[0])

    _hero(
        "Seminarium · slide 2",
        "GASVM: algorytm genetyczny szuka najlepszego małego TRAIN",
        "Na tym etapie walidacja jest stała. Każdy osobnik reprezentuje wybór dokładnie k próbek na klasę. Fitness to jakość SVM na walidacji, więc GA nie szuka przypadkowego skrótu, tylko podzbioru, który rzeczywiście dobrze generalizuje.",
    )

    st.markdown(
        """
        <span class="flow-pill">1. wygeneruj populację masek</span>
        <span class="flow-pill">2. wytrenuj SVM dla każdego kandydata</span>
        <span class="flow-pill">3. oceń na stałej walidacji</span>
        <span class="flow-pill">4. zachowaj najlepszy mały subset</span>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1.28, 0.72])
    with cols[0]:
        st.plotly_chart(_select_story_figure(payload), use_container_width=True)
    with cols[1]:
        _signal("Pełna pula", str(full_pool), "próbki dostępne do wyboru dla TRAIN")
        _signal("Wybrany TRAIN", str(reduction), "próbki zostawione przez najlepszy osobnik")
        _signal("Redukcja", f"{100 * (1 - reduction / full_pool):.1f}%", "część puli, którą można odrzucić bez całkowitej utraty jakości")
        st.plotly_chart(
            _history_line(payload["history"], "Postęp fitness w TRAIN-GA", "#1f7a4f"),
            use_container_width=True,
        )

    _story_steps(
        [
            (
                "Chromosom",
                "Chromosom nie koduje parametrów SVM, tylko decyzję: które próbki z puli trafiają do TRAIN.",
            ),
            (
                "Selekcja i mutacja",
                "Krzyżowanie oraz mutacja tworzą nowe podzbiory, ale cały czas utrzymujemy ograniczenie k na klasę, żeby porównanie było uczciwe.",
            ),
            (
                "Kryterium sukcesu",
                "Najlepszy jest ten podzbiór, który przy małym rozmiarze nadal daje wysoką metrykę na walidacji, a nie ten, który tylko dobrze dopasowuje trening.",
            ),
        ]
    )

    _seminar_note(
        "Co powiedzieć: 'To nie jest undersampling robiony raz i na sztywno. To jest optymalizacja kombinatoryczna, w której każdy kandydat odpowiada na pytanie: czy te konkretne próbki wystarczą, żeby SVM zbudował sensowną granicę?'."
    )


def _render_dagasvm_slide():
    summary = _load_best_dagasvm_summary()
    _hero(
        "Seminarium · slide 3",
        "DAGASVM: po doborze treningu zaczynam celowo wzmacniać walidację o próbki najtrudniejsze",
        "Tutaj pomysł robi się ciekawszy. Po każdej fazie TRAIN-GA biorę ustalony model i pytam: na których próbkach walidacyjnych wypada on najsłabiej? To właśnie te próbki są informacyjne, bo pokazują, gdzie model nadal nie umie dobrze oddzielić klas.",
    )

    if summary is None:
        st.info("Brakuje runu `DAGASVM`, żeby pokazać animację kumulacji walidacji na realnych danych.")
        return

    cols = st.columns([1.15, 0.85])
    with cols[0]:
        st.plotly_chart(_dagasvm_animation_figure(summary), use_container_width=True)
    with cols[1]:
        _spotlight(
            "Po co ruszać walidację?",
            "Jeżeli walidacja jest tylko biernym testem, to model nie dostaje informacji o tym, które przypadki są dla niego najtrudniejsze. DAGASVM zamienia walidację w aktywny mechanizm doboru trudnych przykładów."
        )
        _formula("Fitness VAL-GA", "1 - metric(model, subset walidacji)")
        _formula("Efekt", "kolejny TRAIN widzi trudniejszy, bardziej informacyjny validation set")

    _story_steps(
        [
            (
                "TRAIN-GA",
                "Najpierw wybieram mały, sensowny podzbiór treningowy i uczę na nim SVM.",
            ),
            (
                "VAL-GA",
                "Potem szukam tych próbek walidacyjnych, które dla tego modelu są najbardziej problematyczne.",
            ),
            (
                "Kumulacja",
                "Te próbki dokładam do walidacji w kolejnym cyklu, więc algorytm uczy się na coraz mniej komfortowej, ale bardziej informacyjnej ocenie.",
            ),
        ]
    )

    _seminar_note(
        "Co powiedzieć: 'GASVM odpowiada na pytanie, jaki trening wystarczy. DAGASVM idzie dalej i pyta, jaka walidacja naprawdę testuje model. To dlatego walidacja nie jest tu statycznym benchmarkiem, tylko aktywnym składnikiem procedury'."
    )


def _render_results_slide():
    history = _history_summary()
    _hero(
        "Seminarium · slide 4",
        "Realne wyniki: nie chodzi tylko o F1, ale o kompromis między jakością, kosztem i liczbą support vectors",
        "W tym miejscu można przejść z intuicji do twardych danych. Interesuje mnie nie tylko wynik końcowy, ale też to, ile kosztuje jego osiągnięcie i jak bardzo model opiera się na support vectors.",
    )
    if history.empty:
        st.info("Historia eksperymentów jest pusta. Uruchom co najmniej jeden run, żeby ten slajd pokazywał realne wyniki.")
        return

    top = st.columns([1.25, 0.75])
    with top[0]:
        st.plotly_chart(_results_bubble_figure(history), use_container_width=True)
    with top[1]:
        best = history.sort_values("full_validation_f1", ascending=False).iloc[0]
        smallest = history.sort_values("training_set_size", ascending=True).iloc[0]
        fastest = history.sort_values("training_time_sec", ascending=True).iloc[0]
        _signal("Najlepszy F1", f"{best['full_validation_f1']:.3f}", f"{best['algorithm']} · {_display_label(best)}")
        _signal("Najmniejszy TRAIN", f"{int(smallest['training_set_size'])}", f"{smallest['algorithm']} · {_display_label(smallest)}")
        _signal("Najszybszy run", f"{fastest['training_time_sec']:.1f}s", f"{fastest['algorithm']} · {_display_label(fastest)}")

    ranking = history.sort_values("full_validation_f1", ascending=False).head(8).copy()
    if "created_at" in ranking.columns:
        ranking["created_at"] = pd.to_datetime(ranking["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    ranking = ranking[
        [
            "created_at",
            "experiment_name",
            "dataset",
            "algorithm",
            "training_set_size",
            "training_time_sec",
            "num_support_vectors",
            "full_validation_f1",
        ]
    ]
    st.markdown("### Najmocniejsze runy do pokazania na seminarium")
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    _seminar_note(
        "Co powiedzieć: 'Jeżeli dwa warianty mają podobne F1, ale jeden potrzebuje dużo mniejszego train i mniej support vectors, to on jest praktycznie ciekawszy. Ten slajd ma pokazać właśnie koszt uzyskania jakości, a nie tylko samą jakość'."
    )


def _render_conclusions_slide():
    history = _history_summary()
    best_line = "Brak danych historycznych."
    if not history.empty:
        best = history.sort_values("full_validation_f1", ascending=False).iloc[0]
        best_line = (
            f"Najlepszy dotąd wynik ma {best['algorithm']} dla eksperymentu {_display_label(best)} "
            f"z F1={best['full_validation_f1']:.3f}, TRAIN={int(best['training_set_size'])} "
            f"i czasem {best['training_time_sec']:.1f}s."
        )

    _hero(
        "Seminarium · slide 5",
        "Jak to sprzedać na obronie: nie optymalizuję modelu w próżni, tylko cały proces uczenia",
        "Najważniejsza myśl jest taka: klasyfikator pozostaje SVM, ale zmienia się sposób konstruowania danych, na których ten SVM uczy się i jest oceniany. To pozwala zyskać lepszy kompromis między kosztem treningu a jakością modelu.",
    )

    cols = st.columns(3)
    with cols[0]:
        _spotlight("Teza 1", "Full SVM jest dobrym baseline'em, ale płaci pełny koszt za wszystkie próbki, niezależnie od ich realnej wartości dla granicy decyzyjnej.")
    with cols[1]:
        _spotlight("Teza 2", "GASVM pokazuje, że mały, dobrze dobrany TRAIN może zachować dużą część jakości, a jednocześnie wyraźnie ograniczyć koszt.")
    with cols[2]:
        _spotlight("Teza 3", "DAGASVM dokłada aktywne wzmacnianie walidacji i przesuwa akcent z biernego testowania na iteracyjne wystawianie modelu na trudne przypadki.")

    _formula("Główny przekaz", "nie wszystkie próbki są równie warte treningu SVM")
    _card = st.columns([1.0])[0]
    with _card:
        st.markdown(
            f"""
            <div class="section-card">
              <div class="eyebrow">Do zapamiętania</div>
              <div class="muted-copy">{best_line}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Gotowa ściąga do mówienia", expanded=False):
        st.markdown(
            """
            1. Zaczynam od problemu kosztu: kernel SVM daje dobrą jakość, ale przy dużym `n` trening rośnie bardzo szybko.
            2. Potem pokazuję GASVM: algorytm genetyczny wybiera mały podzbiór treningowy, który nadal dobrze działa na walidacji.
            3. Następnie pokazuję DAGASVM: walidacja przestaje być bierna i zaczyna dostarczać modelowi trudne przypadki.
            4. Na końcu wracam do danych: porównuję F1, czas i liczbę support vectors, żeby pokazać praktyczny sens tej procedury.
            """.strip()
        )


def render_presentation_tab():
    st.session_state.setdefault("seminarium_slide", 0)

    controls = st.columns([0.16, 0.16, 0.68])
    if controls[0].button("Poprzedni", use_container_width=True):
        st.session_state["seminarium_slide"] = max(0, st.session_state["seminarium_slide"] - 1)
    if controls[1].button("Następny", use_container_width=True):
        st.session_state["seminarium_slide"] = min(len(SLIDES) - 1, st.session_state["seminarium_slide"] + 1)
    current = controls[2].select_slider(
        "Etap prezentacji",
        options=list(range(len(SLIDES))),
        format_func=lambda idx: f"{idx + 1}. {SLIDES[idx]}",
        value=st.session_state["seminarium_slide"],
        label_visibility="collapsed",
    )
    st.session_state["seminarium_slide"] = current

    slide_idx = st.session_state["seminarium_slide"]
    if slide_idx == 0:
        _render_problem_slide()
    elif slide_idx == 1:
        _render_gasvm_slide()
    elif slide_idx == 2:
        _render_dagasvm_slide()
    elif slide_idx == 3:
        _render_results_slide()
    else:
        _render_conclusions_slide()
