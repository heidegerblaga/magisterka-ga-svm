from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import time

import streamlit as st
import yaml

from ga.ui.data import list_config_presets, load_summary, load_yaml_config
from ga.ui.runtime import (
    clear_active_run_state,
    completed_run_from_state,
    load_active_run_state,
    parse_progress,
    read_log_tail,
    start_run,
)


ALGORITHM_OPTIONS = {
    "full_svm": "Full SVM",
    "gasvm": "GASVM",
    "dagasvm": "DAGASVM",
}


def _preset_config_editor(base_cfg: dict) -> dict:
    cfg = deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    svm_cfg = cfg.setdefault("svm", {})
    fitness_cfg = cfg.setdefault("fitness", {})
    ga_cfg = cfg.setdefault("ga", {})
    val_ga_cfg = cfg.setdefault("val_ga", {})
    encoding_cfg = cfg.setdefault("encoding", {})
    k_schedule_cfg = cfg.setdefault("k_schedule", {})
    val_cum_cfg = cfg.setdefault("val_cumulation", {})

    form_cols = st.columns(3)
    cfg["experiment_name"] = form_cols[0].text_input(
        "Nazwa eksperymentu",
        value=cfg.get("experiment_name", "ga_svm_dashboard"),
    )
    cfg["dataset_name"] = form_cols[1].text_input(
        "Nazwa datasetu",
        value=cfg.get("dataset_name", ""),
        placeholder="np. Chessboard",
    )
    cfg["algorithm_mode"] = form_cols[2].selectbox(
        "Tryb algorytmu",
        options=list(ALGORITHM_OPTIONS.keys()),
        format_func=lambda value: ALGORITHM_OPTIONS[value],
        index=list(ALGORITHM_OPTIONS.keys()).index(cfg.get("algorithm_mode", "dagasvm")),
    )

    general_cols = st.columns(4)
    cfg["seed"] = int(general_cols[0].number_input("Seed", value=int(cfg.get("seed", 42)), step=1))
    data_cfg["max_rows"] = int(
        general_cols[1].number_input(
            "Limit danych (0 = brak limitu)",
            value=int(data_cfg.get("max_rows", 1000) or 0),
            step=100,
            min_value=0,
        )
    )
    fitness_cfg["metric"] = general_cols[2].selectbox(
        "Metryka fitness",
        options=["balanced_accuracy", "accuracy", "f1", "roc_auc", "pr_auc"],
        index=["balanced_accuracy", "accuracy", "f1", "roc_auc", "pr_auc"].index(
            fitness_cfg.get("metric", "balanced_accuracy")
        ),
    )
    fitness_cfg["class_imbalance_penalty"] = float(
        general_cols[3].number_input(
            "Kara za imbalance",
            value=float(fitness_cfg.get("class_imbalance_penalty", 0.2)),
            step=0.05,
            min_value=0.0,
            max_value=1.0,
        )
    )

    svm_cols = st.columns(4)
    svm_cfg["kernel"] = svm_cols[0].selectbox(
        "Kernel SVM",
        options=["rbf", "linear", "poly", "sigmoid"],
        index=["rbf", "linear", "poly", "sigmoid"].index(svm_cfg.get("kernel", "rbf")),
    )
    svm_cfg["C"] = float(svm_cols[1].number_input("C", value=float(svm_cfg.get("C", 1.0)), min_value=0.01, step=0.1))
    svm_cfg["gamma"] = svm_cols[2].text_input("Gamma", value=str(svm_cfg.get("gamma", "scale")))
    svm_cfg["class_weight"] = svm_cols[3].selectbox(
        "Class weight",
        options=["balanced", "none"],
        index=0 if svm_cfg.get("class_weight", "balanced") == "balanced" else 1,
    )
    if svm_cfg["class_weight"] == "none":
        svm_cfg["class_weight"] = None

    if cfg["algorithm_mode"] in {"gasvm", "dagasvm"}:
        train_cols = st.columns(5)
        encoding_cfg["k_per_class"] = int(
            train_cols[0].number_input(
                "k_per_class",
                value=int(encoding_cfg.get("k_per_class", 20)),
                min_value=1,
                step=1,
            )
        )
        ga_cfg["population_size"] = int(
            train_cols[1].number_input(
                "GA population",
                value=int(ga_cfg.get("population_size", 40)),
                min_value=2,
                step=2,
            )
        )
        ga_cfg["generations"] = int(
            train_cols[2].number_input(
                "GA generations",
                value=int(ga_cfg.get("generations", 20)),
                min_value=1,
                step=1,
            )
        )
        ga_cfg["mutation_rate"] = float(
            train_cols[3].number_input(
                "Mutation rate",
                value=float(ga_cfg.get("mutation_rate", 0.05)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        )
        ga_cfg["crossover_rate"] = float(
            train_cols[4].number_input(
                "Crossover rate",
                value=float(ga_cfg.get("crossover_rate", 0.8)),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
        )

    if cfg["algorithm_mode"] == "dagasvm":
        dag_cols = st.columns(5)
        k_schedule_cfg["num_cycles"] = int(
            dag_cols[0].number_input(
                "Liczba cykli",
                value=int(k_schedule_cfg.get("num_cycles", 3)),
                min_value=1,
                step=1,
            )
        )
        k_schedule_cfg["k_start"] = int(
            dag_cols[1].number_input(
                "k_start",
                value=int(k_schedule_cfg.get("k_start", encoding_cfg.get("k_per_class", 20))),
                min_value=1,
                step=1,
            )
        )
        k_schedule_cfg["k_multiplier"] = float(
            dag_cols[2].number_input(
                "k_multiplier",
                value=float(k_schedule_cfg.get("k_multiplier", 1.5)),
                min_value=1.0,
                step=0.1,
            )
        )
        val_ga_cfg["subset_fraction"] = float(
            dag_cols[3].number_input(
                "VAL-GA subset_fraction",
                value=float(val_ga_cfg.get("subset_fraction", 0.5)),
                min_value=0.05,
                max_value=1.0,
                step=0.05,
            )
        )
        val_cum_cfg["initial_fraction"] = float(
            dag_cols[4].number_input(
                "Początkowa walidacja",
                value=float(val_cum_cfg.get("initial_fraction", 0.5)),
                min_value=0.05,
                max_value=1.0,
                step=0.05,
            )
        )

    if data_cfg.get("max_rows", 0) <= 0:
        data_cfg["max_rows"] = None
    return cfg


def _render_active_run(state):
    st.markdown("### Status runu")
    log_text = read_log_tail(state.get("live_log_path") or state.get("console_log_path"))
    progress = parse_progress(log_text)
    run_id_label = state.get("run_id") or "-"
    if run_id_label != "-" and len(run_id_label) > 18:
        run_id_label = f"...{run_id_label[-18:]}"
    elapsed = "-"
    if state.get("started_at"):
        started_at = datetime.fromisoformat(state["started_at"])
        elapsed_seconds = max(0.0, (datetime.now() - started_at).total_seconds())
        elapsed = f"{elapsed_seconds:.1f}s"

    status_cols = st.columns(6)
    status_cols[0].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Status</div>
          <div class="kpi-value">{'RUN' if state.get('running') else 'DONE'}</div>
          <div class="muted-copy">aktywny proces</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_cols[1].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Faza</div>
          <div class="kpi-value">{progress.get('phase') or '-'}</div>
          <div class="muted-copy">ostatnio wykryta</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cycle_label = "-"
    if progress.get("cycle_current") and progress.get("cycle_total"):
        cycle_label = f"{progress['cycle_current']}/{progress['cycle_total']}"
    status_cols[2].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Cykl</div>
          <div class="kpi-value">{cycle_label}</div>
          <div class="muted-copy">postęp cykli</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gen_label = "-"
    if progress.get("generation_current") and progress.get("generation_total"):
        gen_label = f"{progress['generation_current']}/{progress['generation_total']}"
    status_cols[3].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Generacja</div>
          <div class="kpi-value">{gen_label}</div>
          <div class="muted-copy">ostatni log GA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_cols[4].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Run ID</div>
          <div class="kpi-value">{run_id_label}</div>
          <div class="muted-copy">wykryty z logu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_cols[5].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Czas</div>
          <div class="kpi-value">{elapsed}</div>
          <div class="muted-copy">od startu subprocessu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(progress.get("overall_ratio", 0.0))
    if progress.get("last_message"):
        st.caption(progress["last_message"])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.code(log_text or "Brak logu procesu.", language="text")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Szczegóły wykonania", expanded=False):
        st.write("`temp_config_path`")
        st.code(state.get("temp_config_path") or "-", language="text")
        st.write("`console_log_path`")
        st.code(state.get("console_log_path") or "-", language="text")
        st.write("`live_log_path`")
        st.code(state.get("live_log_path") or "-", language="text")


def _render_completed_run(state):
    latest = completed_run_from_state(state)
    if latest is None:
        return
    summary = load_summary(latest["summary_path"])
    row = summary.get("report_row") or {}
    st.success(f"Run zakończony: `{latest['run_id']}`")
    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Algorithm</div>
          <div class="kpi-value">{row.get('algorithm', '-')}</div>
          <div class="muted-copy">tryb raportowy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Validation F1</div>
          <div class="kpi-value">{float(row.get('full_validation_f1') or 0):.3f}</div>
          <div class="muted-copy">na pełnym validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Czas</div>
          <div class="kpi-value">{float(row.get('training_time_sec') or 0):.1f}s</div>
          <div class="muted-copy">pełny wall time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f"""
        <div class="metric-card">
          <div class="kpi-label">Support vectors</div>
          <div class="kpi-value">{int(row.get('num_support_vectors') or 0)}</div>
          <div class="muted-copy">liczba SV</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Podsumowanie runu", expanded=False):
        st.json(summary)


def render_runs_tab():
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Uruchamianie eksperymentów</div>
          <h2 class="hero-title">Preset YAML + nadpisania parametrów</h2>
          <p class="hero-copy">
            Dashboard nie replikuje logiki eksperymentu. Generuje tymczasowy config,
            uruchamia istniejące CLI w subprocessie i pokazuje status, log oraz wynik po zakończeniu.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_state = load_active_run_state()
    action_cols = st.columns([0.2, 0.2, 0.6])
    action_cols[0].button("Odśwież status", use_container_width=True)
    if action_cols[1].button("Wyczyść stan UI", use_container_width=True):
        clear_active_run_state()
        active_state = None
    auto_refresh = action_cols[2].toggle("Auto-refresh co 2s podczas runu", value=True)

    presets = list_config_presets()
    if not presets:
        st.error("Brak presetów YAML w katalogu `configs/`.")
        return

    preset_path = st.selectbox("Preset konfiguracyjny", presets, format_func=lambda path: path.name)
    base_cfg = load_yaml_config(preset_path)
    base_cfg.setdefault("dataset_name", "")
    base_cfg.setdefault("algorithm_mode", "dagasvm")

    with st.form("run_form"):
        edited_cfg = _preset_config_editor(base_cfg)
        st.markdown("#### Podgląd finalnego configu")
        st.code(yaml.safe_dump(edited_cfg, sort_keys=False), language="yaml")
        submitted = st.form_submit_button(
            "Uruchom run",
            use_container_width=True,
            disabled=bool(active_state and active_state.get("running")),
        )

    if submitted:
        if not edited_cfg.get("dataset_name", "").strip():
            st.error("Pole `dataset_name` jest wymagane.")
        else:
            started = start_run(edited_cfg)
            st.success(f"Uruchomiono proces PID={started['pid']}")
            active_state = started

    if active_state is not None:
        _render_active_run(active_state)
        _render_completed_run(active_state)
        if active_state.get("running") and auto_refresh:
            time.sleep(2)
            st.rerun()
