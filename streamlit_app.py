from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from ga.ui.history import render_history_tab
from ga.ui.presentation import render_presentation_tab
from ga.ui.runs import render_runs_tab
from ga.ui.styles import inject_global_styles


st.set_page_config(
    page_title="GA + SVM Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_styles()

st.markdown(
    """
    <div class="hero-card" style="margin-bottom: 1rem;">
      <div class="eyebrow">Magisterka</div>
      <h1 class="hero-title">Dashboard eksperymentów GA + SVM</h1>
      <p class="hero-copy">
        Jedna aplikacja do prezentacji idei, uruchamiania kolejnych runów oraz porównywania
        historii eksperymentów w formacie raportowym.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

seminarium_tab, runy_tab, historia_tab = st.tabs(["Seminarium", "Runy", "Historia"])

with seminarium_tab:
    render_presentation_tab()

with runy_tab:
    render_runs_tab()

with historia_tab:
    render_history_tab()
