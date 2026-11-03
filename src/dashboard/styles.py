"""
src/dashboard/styles.py
───────────────────────
Centralized CSS design system for ClimateShield AI Streamlit Dashboard.
Deep navy theme, accessible contrast, unified font stack, consistent card layouts.
"""

CUSTOM_CSS = """
<style>
/* ── Typography & Global Theme ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

/* Deep Navy Theme Backgrounds */
.stApp {
    background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
    color: #e2e8f0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b132b !important;
    border-right: 1px solid #1f293d;
}

/* ── Top Header & Badges ──────────────────────────────────────────────────── */
.app-header {
    padding: 1.2rem 0 0.8rem 0;
    border-bottom: 1px solid #2a3756;
    margin-bottom: 1.5rem;
}

.app-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.35rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge-prototype { background: #334155; color: #cbd5e1; border: 1px solid #475569; }
.badge-modelled { background: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; }
.badge-timestamp { background: #14532d; color: #86efac; border: 1px solid #16a34a; }

/* ── KPI Cards (Uniform Height & Clean Flex Layout) ───────────────────────── */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.75rem;
    padding: 1.1rem;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.35rem;
}

/* ── Risk Badges ──────────────────────────────────────────────────────────── */
.risk-low {
    background-color: #166534;
    color: #dcfce7;
    border-left: 4px solid #22c55e;
}
.risk-moderate {
    background-color: #854d0e;
    color: #fef9c3;
    border-left: 4px solid #eab308;
}
.risk-high {
    background-color: #9a3412;
    color: #ffedd5;
    border-left: 4px solid #f97316;
}
.risk-severe {
    background-color: #991b1b;
    color: #fee2e2;
    border-left: 4px solid #ef4444;
}
.risk-unavailable {
    background-color: #334155;
    color: #e2e8f0;
    border-left: 4px solid #94a3b8;
}

/* ── Content Blocks & Callouts ────────────────────────────────────────────── */
.guidance-box {
    background: #0f172a;
    border-left: 4px solid #38bdf8;
    padding: 1rem 1.25rem;
    border-radius: 0 0.5rem 0.5rem 0;
    margin: 1rem 0;
}

.disclaimer-box {
    background: #1e1e24;
    border: 1px solid #3b3b48;
    border-radius: 0.5rem;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-top: 1rem;
}

/* Coverage table badge */
.status-pill-green {
    background-color: #14532d;
    color: #4ade80;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
.status-pill-amber {
    background-color: #713f12;
    color: #fde047;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
.status-pill-red {
    background-color: #7f1d1d;
    color: #fca5a5;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
.status-pill-grey {
    background-color: #334155;
    color: #cbd5e1;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
</style>
"""

def inject_custom_css():
    """Render CSS block in Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
