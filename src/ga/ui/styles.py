from __future__ import annotations

import streamlit as st


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap');

:root {
  --bg-soft: #f4f7f4;
  --card: #ffffff;
  --ink: #102418;
  --muted: #56705d;
  --accent: #1f7a4f;
  --accent-2: #cfe8d9;
  --accent-3: #f0c48e;
  --accent-4: #0d5b4f;
  --line: rgba(16, 36, 24, 0.12);
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(207, 232, 217, 0.9), transparent 28%),
    radial-gradient(circle at top right, rgba(245, 217, 184, 0.55), transparent 22%),
    linear-gradient(180deg, #f7faf7 0%, #eef4ef 100%);
  color: var(--ink);
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 1380px;
}

.hero-card,
.metric-card,
.section-card,
.table-card {
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: 0 22px 50px rgba(33, 66, 49, 0.08);
  padding: 1.2rem 1.4rem;
  backdrop-filter: blur(12px);
  animation: fadeSlide 0.55s ease both;
}

.hero-card {
  padding: 1.8rem 2rem;
  background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(232,244,236,0.92));
}

.eyebrow {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: rgba(31, 122, 79, 0.1);
  color: var(--accent);
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}

.hero-title {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 2.35rem;
  line-height: 1.05;
  margin: 0 0 0.55rem 0;
  color: var(--ink);
}

.hero-copy,
.muted-copy {
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.6;
}

.kpi-label {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}

.kpi-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--ink);
}

.pill {
  display: inline-block;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: var(--accent-2);
  color: var(--ink);
  font-size: 0.85rem;
}

.spotlight-card,
.process-step,
.formula-card,
.signal-card {
  background: linear-gradient(160deg, rgba(14, 61, 47, 0.94), rgba(30, 94, 72, 0.9));
  color: #f7fbf7;
  border-radius: 28px;
  padding: 1.35rem 1.45rem;
  box-shadow: 0 30px 70px rgba(13, 48, 39, 0.24);
  position: relative;
  overflow: hidden;
  animation: shimmerIn 0.7s ease both;
}

.spotlight-card::after,
.process-step::after,
.formula-card::after,
.signal-card::after {
  content: "";
  position: absolute;
  inset: -40% auto auto -10%;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(240, 196, 142, 0.42), transparent 70%);
}

.spotlight-title {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 2.05rem;
  line-height: 1.05;
  margin: 0 0 0.6rem 0;
}

.spotlight-copy {
  color: rgba(247, 251, 247, 0.84);
  line-height: 1.75;
  font-size: 1rem;
}

.formula-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.9), rgba(240, 246, 241, 0.88));
  color: var(--ink);
  border: 1px solid rgba(16, 36, 24, 0.1);
}

.formula-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 0.45rem;
}

.formula-body {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.7rem;
  line-height: 1.2;
}

.story-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
}

.process-step {
  min-height: 180px;
}

.process-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.18);
  margin-bottom: 0.8rem;
  font-weight: 700;
}

.process-title {
  font-size: 1.15rem;
  margin-bottom: 0.4rem;
  font-weight: 700;
}

.process-copy {
  color: rgba(247, 251, 247, 0.82);
  line-height: 1.65;
}

.signal-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.96), rgba(241, 247, 242, 0.94));
  color: var(--ink);
  border: 1px solid rgba(16, 36, 24, 0.08);
  box-shadow: 0 16px 34px rgba(33, 66, 49, 0.08);
}

.signal-label {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.signal-value {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 2.3rem;
  line-height: 1.05;
  margin: 0.35rem 0;
}

.signal-copy {
  color: var(--muted);
  line-height: 1.55;
}

.seminar-note {
  border-left: 4px solid var(--accent-3);
  padding-left: 1rem;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.7;
}

.flow-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.45rem 0.8rem;
  margin-right: 0.45rem;
  margin-bottom: 0.55rem;
  border-radius: 999px;
  background: rgba(31, 122, 79, 0.1);
  color: var(--accent-4);
  font-size: 0.88rem;
  font-weight: 600;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 18px;
}

.history-table thead tr:first-child th {
  background: #eef5ef;
  border-bottom: 2px solid rgba(16, 36, 24, 0.2);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.history-table th,
.history-table td {
  border: 1px solid rgba(16, 36, 24, 0.08);
  padding: 0.55rem 0.6rem;
  text-align: center;
  font-size: 0.88rem;
}

.history-table tbody tr:nth-child(even) {
  background: rgba(240, 246, 241, 0.72);
}

.history-table tbody tr:hover {
  background: rgba(207, 232, 217, 0.5);
}

@keyframes fadeSlide {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes shimmerIn {
  from {
    opacity: 0;
    transform: translateY(16px) scale(0.985);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@media (max-width: 900px) {
  .story-grid {
    grid-template-columns: 1fr;
  }
}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
