from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from ga.reporting import rebuild_history_table
from ga.ui.data import ARTIFACTS_ROOT, find_summary_by_run_id, load_history_with_metadata, load_summary


VISIBLE_COLUMNS = [
    "created_at",
    "experiment_name",
    "dataset",
    "algorithm",
    "training_set_selection",
    "validation_set_selection",
    "training_time_sec",
    "training_set_size",
    "validation_set_size",
    "num_support_vectors",
    "full_train_precision",
    "full_train_recall",
    "full_train_f1",
    "full_validation_precision",
    "full_validation_recall",
    "full_validation_f1",
    "test_precision",
    "test_recall",
    "test_f1",
]


def _fmt(value, column: str):
    if pd.isna(value):
        return "-"
    if column == "created_at":
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _grouped_html_table(df: pd.DataFrame) -> str:
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(row.get(col), col)}</td>" for col in VISIBLE_COLUMNS)
        rows_html.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="table-card">
      <table class="history-table">
        <thead>
          <tr>
            <th rowspan="2">Data</th>
            <th rowspan="2">Experiment</th>
            <th rowspan="2">Dataset</th>
            <th rowspan="2">Algorithm</th>
            <th rowspan="2">Training set selection</th>
            <th rowspan="2">Validation set selection</th>
            <th rowspan="2">Training time</th>
            <th rowspan="2">Training set size</th>
            <th rowspan="2">Validation set size</th>
            <th rowspan="2">No. of SVs</th>
            <th colspan="3">Full training set</th>
            <th colspan="3">Full validation set</th>
            <th colspan="3">Test set</th>
          </tr>
          <tr>
            <th>precision</th>
            <th>recall</th>
            <th>F-score</th>
            <th>precision</th>
            <th>recall</th>
            <th>F-score</th>
            <th>precision</th>
            <th>recall</th>
            <th>F-score</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def _safe_lookup(payload: dict[str, Any], *path: str):
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _yaml_dump(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, default_flow_style=False)


def _flatten_payload(payload: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_payload(value, dotted))
        return rows
    if isinstance(payload, list):
        if not payload:
            return [{"parameter": prefix, "value": "[]"}]
        for idx, value in enumerate(payload):
            rows.extend(_flatten_payload(value, f"{prefix}[{idx}]"))
        return rows
    rows.append({"parameter": prefix or "-", "value": "-" if payload is None else str(payload)})
    return rows


def _config_highlights(summary: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    trial_overrides = summary.get("trial_overrides") or config.get("trial_overrides") or {}
    rows = [
        ("run_id", summary.get("run_id")),
        ("experiment_name", summary.get("experiment_name") or config.get("experiment_name")),
        ("dataset_name", summary.get("dataset_name") or config.get("dataset_name")),
        ("algorithm_mode", summary.get("algorithm_mode") or config.get("algorithm_mode")),
        ("fitness.metric", _safe_lookup(config, "fitness", "metric")),
        ("fitness.class_imbalance_penalty", _safe_lookup(config, "fitness", "class_imbalance_penalty")),
        ("svm.kernel", _safe_lookup(config, "svm", "kernel")),
        ("svm.C", _safe_lookup(config, "svm", "C")),
        ("svm.gamma", _safe_lookup(config, "svm", "gamma")),
        ("encoding.k_per_class", _safe_lookup(config, "encoding", "k_per_class")),
        ("ga.population_size", _safe_lookup(config, "ga", "population_size")),
        ("ga.generations", _safe_lookup(config, "ga", "generations")),
        ("ga.mutation_rate", _safe_lookup(config, "ga", "mutation_rate")),
        ("val_ga.subset_fraction", _safe_lookup(config, "val_ga", "subset_fraction")),
        ("k_schedule.num_cycles", _safe_lookup(config, "k_schedule", "num_cycles")),
        ("k_schedule.k_start", _safe_lookup(config, "k_schedule", "k_start")),
        ("k_schedule.k_multiplier", _safe_lookup(config, "k_schedule", "k_multiplier")),
        ("trial_overrides_count", len(trial_overrides)),
    ]
    return pd.DataFrame(
        [{"parameter": key, "value": "-" if value is None else str(value)} for key, value in rows]
    )


def _render_run_configuration(filtered: pd.DataFrame):
    st.markdown("### Konfiguracja runu")
    run_id = st.selectbox(
        "Wybierz run do podglądu konfiguracji",
        filtered["run_id"].tolist(),
        index=0,
        key="history_config_run_id",
    )
    summary_path = find_summary_by_run_id(str(run_id))
    if summary_path is None:
        st.warning("Nie udało się znaleźć `summary.json` dla wybranego runu.")
        return

    summary = load_summary(summary_path)
    config_path = summary_path.parent / "config_snapshot.json"
    if not config_path.exists():
        st.warning("Brak `config_snapshot.json` dla wybranego runu.")
        return
    config = load_summary(config_path)

    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Validation F1</div>
          <div class="kpi-value">{float(summary.get("report_row", {}).get("full_validation_f1") or 0):.3f}</div>
          <div class="muted-copy">pełna walidacja</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Test F1</div>
          <div class="kpi-value">{float(summary.get("report_row", {}).get("test_f1") or 0):.3f}</div>
          <div class="muted-copy">zbiór testowy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Czas</div>
          <div class="kpi-value">{float(summary.get("report_row", {}).get("training_time_sec") or 0):.1f}s</div>
          <div class="muted-copy">trening całego runu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">TRAIN size</div>
          <div class="kpi-value">{int(summary.get("report_row", {}).get("training_set_size") or 0)}</div>
          <div class="muted-copy">reprezentatywny podzbiór</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta_cols = st.columns(3)
    meta_cols[0].caption(f"`run_id`: `{summary.get('run_id', '-')}`")
    meta_cols[1].caption(f"`summary.json`: `{summary_path}`")
    meta_cols[2].caption(f"`config_snapshot.json`: `{config_path}`")

    tabs = st.tabs(["Skrót", "Trial Overrides", "Pełny Config", "Pełny Summary"])
    with tabs[0]:
        st.dataframe(
            _config_highlights(summary, config),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[1]:
        overrides = summary.get("trial_overrides") or config.get("trial_overrides") or {}
        if not overrides:
            st.info("Ten run nie ma `trial_overrides` albo był odpalony bez sweepu.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [{"parameter": key, "value": str(value)} for key, value in sorted(overrides.items())]
                ),
                hide_index=True,
                use_container_width=True,
            )

    with tabs[2]:
        flat_config = pd.DataFrame(_flatten_payload(config))
        st.dataframe(flat_config, hide_index=True, use_container_width=True, height=360)
        st.code(_yaml_dump(config), language="yaml")

    with tabs[3]:
        st.code(_yaml_dump(summary), language="yaml")


def _load_cycles_for_run(run_id: str) -> pd.DataFrame:
    summary_path = find_summary_by_run_id(run_id)
    if summary_path is None:
        return pd.DataFrame()
    summary = load_summary(summary_path)
    cycles_path = summary_path.parent / "cycles.csv"
    if not cycles_path.exists() and summary.get("summary_path"):
        fallback = Path(summary["summary_path"])
        cycles_path = fallback if fallback.name == "cycles.csv" else fallback.with_name("cycles.csv")
    if not cycles_path.exists() or cycles_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(cycles_path)


def _render_comparison(selected: pd.DataFrame):
    if selected.empty:
        st.info("Wybierz runy w sekcji porównania.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Średnie validation F1</div>
          <div class="kpi-value">{selected["full_validation_f1"].mean():.3f}</div>
          <div class="muted-copy">dla wybranych runów</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Najlepszy czas</div>
          <div class="kpi-value">{selected["training_time_sec"].min():.1f}s</div>
          <div class="muted-copy">najkrótsze uczenie</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Najmniej SV</div>
          <div class="kpi-value">{int(selected["num_support_vectors"].min())}</div>
          <div class="muted-copy">liczba support vectors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Najmniejszy TRAIN</div>
          <div class="kpi-value">{int(selected["training_set_size"].min())}</div>
          <div class="muted-copy">rozmiar podzbioru</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for column, name in [
        ("full_validation_f1", "Validation F1"),
        ("training_time_sec", "Training time"),
        ("num_support_vectors", "No. of SVs"),
    ]:
        fig.add_trace(go.Bar(x=selected["run_id"], y=selected[column], name=name))
    fig.update_layout(
        title="Porównanie wybranych runów",
        template="plotly_white",
        barmode="group",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    cycle_fig = go.Figure()
    has_cycle_data = False
    for run_id in selected["run_id"]:
        cycles = _load_cycles_for_run(str(run_id))
        if cycles.empty or cycles["cycle"].dropna().empty:
            continue
        has_cycle_data = True
        cycle_fig.add_trace(
            go.Scatter(
                x=cycles["cycle"],
                y=cycles["train_ga_best_val_metric"],
                mode="lines+markers",
                name=f"{run_id} TRAIN-GA",
            )
        )
    if has_cycle_data:
        cycle_fig.update_layout(
            title="Przebieg `train_ga_best_val_metric` po cyklach",
            template="plotly_white",
            height=380,
            xaxis_title="Cykl",
            yaxis_title="Validation metric",
        )
        st.plotly_chart(cycle_fig, use_container_width=True)

    detail_run_id = st.selectbox("Szczegóły runu", selected["run_id"].tolist())
    summary_path = find_summary_by_run_id(str(detail_run_id))
    if summary_path is not None:
        summary = load_summary(summary_path)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("`summary.json`")
        st.json(summary)
        config_path = summary_path.parent / "config_snapshot.json"
        if config_path.exists():
            st.write("`config_snapshot.json`")
            st.json(load_summary(config_path))
        st.markdown("</div>", unsafe_allow_html=True)


def render_history_tab():
    header_cols = st.columns([0.82, 0.18])
    header_cols[0].markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Historia eksperymentów</div>
          <h2 class="hero-title">Tabela porównawcza i analityka runów</h2>
          <p class="hero-copy">
            Ten widok korzysta z `artifacts/history_table.csv` jako źródła prawdy i
            pozwala porównywać zapisane runy w formacie zbliżonym do tabeli z pracy.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if header_cols[1].button("Przebuduj historię", use_container_width=True):
        rebuild_history_table(ARTIFACTS_ROOT)

    history = load_history_with_metadata()
    if history.empty:
        st.info("Historia jest pusta. Uruchom eksperyment albo przebuduj tabelę z artefaktów.")
        return

    controls = st.columns(5)
    dataset_options = ["Wszystkie"] + sorted(history["dataset"].dropna().unique().tolist())
    algorithm_options = ["Wszystkie"] + sorted(history["algorithm"].dropna().unique().tolist())
    seed_values = ["Wszystkie"]
    if "seed" in history.columns:
        seed_values += [str(value) for value in sorted(history["seed"].dropna().unique().tolist())]

    selected_dataset = controls[0].selectbox("Dataset", dataset_options)
    selected_algorithm = controls[1].selectbox("Algorithm", algorithm_options)
    selected_seed = controls[2].selectbox("Seed", seed_values)
    sort_by = controls[3].selectbox(
        "Sortuj po",
        ["full_validation_f1", "training_time_sec", "num_support_vectors", "created_at"],
    )
    descending = controls[4].toggle("Malejąco", value=True)

    filtered = history.copy()
    if selected_dataset != "Wszystkie":
        filtered = filtered[filtered["dataset"] == selected_dataset]
    if selected_algorithm != "Wszystkie":
        filtered = filtered[filtered["algorithm"] == selected_algorithm]
    if selected_seed != "Wszystkie" and "seed" in filtered.columns:
        filtered = filtered[filtered["seed"].astype(str) == selected_seed]

    filtered = filtered.sort_values(sort_by, ascending=not descending)
    if filtered.empty:
        st.info("Brak runów dla wybranych filtrów.")
        return

    st.markdown(_grouped_html_table(filtered[VISIBLE_COLUMNS].head(50)), unsafe_allow_html=True)
    _render_run_configuration(filtered)

    st.markdown("### Porównanie runów")
    selected_run_ids = st.multiselect(
        "Wybierz runy do porównania",
        filtered["run_id"].tolist(),
        default=filtered["run_id"].head(3).tolist(),
    )
    selected_rows = filtered[filtered["run_id"].isin(selected_run_ids)]
    _render_comparison(selected_rows)
