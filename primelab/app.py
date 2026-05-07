"""
PrimeLab — Advanced Prime Research Laboratory
Main Streamlit Application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import json
import time
import pandas as pd
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrimeLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #080D1A;
    --surface: #0F172A;
    --surface2: #1E293B;
    --border: #1E3A5F;
    --prime: #00D4FF;
    --gold: #FFB800;
    --magenta: #FF3CAC;
    --green: #00FF87;
    --text: #E2E8F0;
    --muted: #64748B;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }
.stApp > header { background-color: transparent !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080D1A 0%, #0A1628 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Cards */
.prime-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, #1A2744 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--prime), var(--magenta));
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--prime);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* Header */
.primelab-header {
    background: linear-gradient(135deg, #080D1A 0%, #0D1F3C 50%, #080D1A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.primelab-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, var(--prime) 0%, var(--magenta) 50%, var(--gold) 100%);
}
.primelab-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--prime), #7B9FFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.primelab-subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
    letter-spacing: 0.05em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border: none !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E3A5F, #1A2744) !important;
    color: var(--prime) !important;
    border: 1px solid var(--border) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1A3A5C, #0D2544) !important;
    color: var(--prime) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #244A7A, #1A3A5C) !important;
    border-color: var(--prime) !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
}

/* Input widgets */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Status tags */
.tag-pass {
    background: rgba(0, 255, 135, 0.15);
    color: var(--green);
    border: 1px solid rgba(0, 255, 135, 0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    display: inline-block;
}
.tag-fail {
    background: rgba(255, 60, 172, 0.15);
    color: var(--magenta);
    border: 1px solid rgba(255, 60, 172, 0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    display: inline-block;
}
.tag-warn {
    background: rgba(255, 184, 0, 0.15);
    color: var(--gold);
    border: 1px solid rgba(255, 184, 0, 0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    display: inline-block;
}

/* Alerts */
.alert-box {
    background: rgba(255, 184, 0, 0.08);
    border-left: 3px solid var(--gold);
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* Code blocks */
code {
    background: var(--surface2) !important;
    color: var(--prime) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* DataFrames */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Slider */
.stSlider [data-testid="stSlider"] { color: var(--prime) !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, var(--prime), var(--magenta)) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ───────────────────────────────────────────────────
def init_state():
    defaults = {
        "primes": None,
        "primes_range": (2, 100000),
        "results": {},
        "experiment_config": None,
        "last_compute_time": 0.0,
        "alerts": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Imports from PrimeLab modules ───────────────────────────────────────────
@st.cache_resource
def load_modules():
    from core.sieve import get_primes, cache_info, clear_cache, is_prime_miller_rabin
    from analysis.math_analysis import (
        pi_comparison, gap_statistics, maximal_gaps, find_twin_primes,
        find_constellation, test_cramer_model, test_gap_distribution,
        test_poisson_model, anomaly_detection, fft_analysis, autocorrelation,
        convergence_study, normalized_gaps, prime_density, compute_gaps,
    )
    from viz.plots import (
        plot_pi_comparison, plot_pi_error, plot_gap_histogram,
        plot_normalized_gaps, plot_gaps_scatter, plot_maximal_gaps,
        plot_twin_prime_density, plot_prime_heatmap, plot_fft,
        plot_autocorrelation, plot_convergence, plot_anomalies, plot_dashboard,
    )
    from experiments.runner import (
        ExperimentConfig, run_experiment, PRESET_EXPERIMENTS,
        AVAILABLE_METRICS, list_saved_experiments,
    )
    from exports.exporter import (
        primes_to_csv, gaps_to_csv, pi_comparison_to_csv, export_figure,
    )
    return {
        "get_primes": get_primes,
        "cache_info": cache_info,
        "clear_cache": clear_cache,
        "is_prime_miller_rabin": is_prime_miller_rabin,
        "pi_comparison": pi_comparison,
        "gap_statistics": gap_statistics,
        "maximal_gaps": maximal_gaps,
        "find_twin_primes": find_twin_primes,
        "find_constellation": find_constellation,
        "test_cramer_model": test_cramer_model,
        "test_gap_distribution": test_gap_distribution,
        "test_poisson_model": test_poisson_model,
        "anomaly_detection": anomaly_detection,
        "fft_analysis": fft_analysis,
        "autocorrelation": autocorrelation,
        "convergence_study": convergence_study,
        "normalized_gaps": normalized_gaps,
        "prime_density": prime_density,
        "compute_gaps": compute_gaps,
        "plot_pi_comparison": plot_pi_comparison,
        "plot_pi_error": plot_pi_error,
        "plot_gap_histogram": plot_gap_histogram,
        "plot_normalized_gaps": plot_normalized_gaps,
        "plot_gaps_scatter": plot_gaps_scatter,
        "plot_maximal_gaps": plot_maximal_gaps,
        "plot_twin_prime_density": plot_twin_prime_density,
        "plot_prime_heatmap": plot_prime_heatmap,
        "plot_fft": plot_fft,
        "plot_autocorrelation": plot_autocorrelation,
        "plot_convergence": plot_convergence,
        "plot_anomalies": plot_anomalies,
        "plot_dashboard": plot_dashboard,
        "ExperimentConfig": ExperimentConfig,
        "run_experiment": run_experiment,
        "PRESET_EXPERIMENTS": PRESET_EXPERIMENTS,
        "AVAILABLE_METRICS": AVAILABLE_METRICS,
        "list_saved_experiments": list_saved_experiments,
        "primes_to_csv": primes_to_csv,
        "gaps_to_csv": gaps_to_csv,
        "pi_comparison_to_csv": pi_comparison_to_csv,
        "export_figure": export_figure,
    }

try:
    M = load_modules()
except Exception as e:
    st.error(f"Failed to load PrimeLab modules: {e}")
    st.stop()


# ─── Helper: generate primes ──────────────────────────────────────────────────
def compute_primes(low: int, high: int):
    with st.spinner("🔬 Generating primes with segmented sieve..."):
        t0 = time.time()
        primes = M["get_primes"](low, high, use_cache=True)
        elapsed = time.time() - t0
    st.session_state["primes"] = primes
    st.session_state["primes_range"] = (low, high)
    st.session_state["results"] = {}
    st.session_state["last_compute_time"] = elapsed
    st.session_state["alerts"] = []
    return primes, elapsed


def get_primes_state():
    return st.session_state.get("primes")


# ─── Metric card helper ───────────────────────────────────────────────────────
def metric_card(label: str, value: str, col=None):
    html = f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700;
             background:linear-gradient(135deg,#00D4FF,#7B9FFF);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🔬 PrimeLab
        </div>
        <div style="font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono',monospace;
             letter-spacing:0.15em; margin-top:4px;">
            PRIME RESEARCH LABORATORY
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Range Configuration")
    range_preset = st.selectbox("Quick Preset", [
        "Custom", "10³ (1K)", "10⁴ (10K)", "10⁵ (100K)",
        "10⁶ (1M)", "10⁷ (10M)", "Near 10⁹",
    ])
    preset_map = {
        "10³ (1K)": (2, 1000),
        "10⁴ (10K)": (2, 10000),
        "10⁵ (100K)": (2, 100000),
        "10⁶ (1M)": (2, 1000000),
        "10⁷ (10M)": (2, 10000000),
        "Near 10⁹": (10**9, 10**9 + 200000),
    }
    if range_preset != "Custom" and range_preset in preset_map:
        default_low, default_high = preset_map[range_preset]
    else:
        prev = st.session_state.get("primes_range", (2, 100000))
        default_low, default_high = prev

    col_l, col_h = st.columns(2)
    low_val = col_l.number_input("Low", value=int(default_low), min_value=2, step=1, format="%d")
    high_val = col_h.number_input("High", value=int(default_high), min_value=3, step=1, format="%d")

    use_cache = st.checkbox("Use disk cache", value=True,
                            help="Cache primes to .npy for reuse")

    if st.button("🚀 Generate Primes", use_container_width=True):
        if high_val <= low_val:
            st.error("High must be > Low")
        elif high_val - low_val > 5 * 10**8:
            st.warning("⚠️ Range > 5×10⁸ may take several minutes. Consider narrowing.")
        else:
            primes, elapsed = compute_primes(int(low_val), int(high_val))
            st.success(f"✓ {len(primes):,} primes in {elapsed:.2f}s")

    st.markdown("---")

    # Miller-Rabin spot check
    st.markdown("### 🎯 Primality Check")
    check_n = st.number_input("Check number", value=104729, min_value=2, format="%d")
    if st.button("Test primality", use_container_width=True):
        result_mr = M["is_prime_miller_rabin"](int(check_n))
        if result_mr:
            st.markdown(f'<span class="tag-pass">✓ {check_n:,} is PRIME</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="tag-fail">✗ {check_n:,} is COMPOSITE</span>',
                        unsafe_allow_html=True)

    st.markdown("---")

    # Cache info
    st.markdown("### 💾 Cache")
    ci = M["cache_info"]()
    st.caption(f"Files: {ci['n_files']} | Size: {ci['total_size_mb']:.1f} MB")
    if st.button("Clear cache", use_container_width=True):
        M["clear_cache"]()
        st.success("Cache cleared")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono',monospace;
         text-align:center; line-height:1.8;">
        PrimeLab v1.0<br>
        Segmented Sieve · Miller-Rabin<br>
        Li(x) · Cramér · KS Test · FFT
    </div>
    """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
primes = get_primes_state()
range_str = f"[{st.session_state['primes_range'][0]:,} → {st.session_state['primes_range'][1]:,}]"

st.markdown(f"""
<div class="primelab-header">
    <div class="primelab-title">🔬 PrimeLab</div>
    <div class="primelab-subtitle">
        ► ADVANCED PRIME NUMBER RESEARCH LABORATORY &nbsp;|&nbsp;
        RANGE: {range_str} &nbsp;|&nbsp;
        N = {len(primes) if primes is not None else "—"}
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Quick metrics ────────────────────────────────────────────────────────────
if primes is not None and len(primes) > 0:
    twin_count = int(np.sum(np.diff(primes) == 2))
    gaps = np.diff(primes) if len(primes) > 1 else np.array([0])
    max_gap = int(gaps.max()) if len(gaps) > 0 else 0

    cols = st.columns(6)
    metric_card("Total Primes", f"{len(primes):,}", cols[0])
    metric_card("Largest Prime", f"{int(primes[-1]):,}", cols[1])
    metric_card("Twin Primes", f"{twin_count:,}", cols[2])
    metric_card("Max Gap", f"{max_gap}", cols[3])
    metric_card("Avg Gap", f"{gaps.mean():.2f}", cols[4])
    metric_card("Compute Time", f"{st.session_state['last_compute_time']:.2f}s", cols[5])
else:
    st.info("👈 Configure a range and click **Generate Primes** to begin your research.")

st.markdown("")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
if primes is not None and len(primes) > 1:
    tab_names = [
        "📊 Dashboard", "π(x) Analysis", "🕳️ Gaps", "👥 Twins & Constellations",
        "🧪 Conjectures", "📡 Spectral", "🧬 Experiments", "💾 Export",
    ]
    tabs = st.tabs(tab_names)

    # ── Tab 0: Dashboard ──────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Overview Dashboard")
        with st.spinner("Building dashboard..."):
            results_for_dash = {}
            if "pi_comparison" not in st.session_state["results"]:
                st.session_state["results"]["pi_comparison"] = M["pi_comparison"](primes)
            if "gap_statistics" not in st.session_state["results"]:
                st.session_state["results"]["gap_statistics"] = M["gap_statistics"](primes)
            if "maximal_gaps" not in st.session_state["results"]:
                st.session_state["results"]["maximal_gaps"] = M["maximal_gaps"](primes)
            if "normalized_gaps" not in st.session_state["results"]:
                ng = M["normalized_gaps"](primes)
                st.session_state["results"]["normalized_gaps"] = {
                    "norm_gaps": ng, "primes": primes[:-1]
                }

            fig_dash = M["plot_dashboard"](primes, st.session_state["results"])
            st.plotly_chart(fig_dash, use_container_width=True)

        # Quick stats table
        st.markdown("### 📋 Quick Statistics")
        gs = st.session_state["results"].get("gap_statistics", {})
        if gs:
            stats_data = {
                "Metric": [
                    "Range low", "Range high", "Total primes", "Largest prime",
                    "Twin primes", "Mean gap", "Std gap",
                    "Max gap", "Max gap at prime", "Most common gap",
                ],
                "Value": [
                    f"{int(primes[0]):,}", f"{int(primes[-1]):,}",
                    f"{len(primes):,}", f"{int(primes[-1]):,}",
                    f"{int(np.sum(np.diff(primes) == 2)):,}",
                    f"{gs.get('gap_mean', 0):.4f}",
                    f"{gs.get('gap_std', 0):.4f}",
                    f"{gs.get('gap_max', 0):,}",
                    f"{gs.get('gap_max_at_prime', 0):,}",
                    f"{gs.get('most_common_gap', 0)}",
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # ── Tab 1: π(x) Analysis ─────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### π(x) — Prime Counting Function")
        if "pi_comparison" not in st.session_state["results"]:
            with st.spinner("Computing π(x) comparison..."):
                st.session_state["results"]["pi_comparison"] = M["pi_comparison"](primes)
        pi_data = st.session_state["results"]["pi_comparison"]

        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig = M["plot_pi_comparison"](pi_data)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown('<div class="prime-card">', unsafe_allow_html=True)
            st.markdown("**Model Comparison**")
            if pi_data:
                li_err = float(np.abs(pi_data["error_li_rel"]).mean())
                log_err = float(np.abs(pi_data["error_log_rel"]).mean())
                st.markdown(f"**Li(x)** mean err: `{li_err:.3f}%`")
                st.markdown(f"**x/ln(x)** mean err: `{log_err:.3f}%`")
                winner = "Li(x)" if li_err < log_err else "x/ln(x)"
                st.markdown(f"Best fit: **{winner}**")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Error Analysis")
        fig_err = M["plot_pi_error"](pi_data)
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("#### Convergence Study")
        if "convergence_study" not in st.session_state["results"]:
            with st.spinner("Computing convergence..."):
                st.session_state["results"]["convergence_study"] = M["convergence_study"](primes)
        fig_conv = M["plot_convergence"](st.session_state["results"]["convergence_study"])
        st.plotly_chart(fig_conv, use_container_width=True)

    # ── Tab 2: Gaps ───────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🕳️ Prime Gap Analysis")
        if "gap_statistics" not in st.session_state["results"]:
            with st.spinner("Computing gap statistics..."):
                st.session_state["results"]["gap_statistics"] = M["gap_statistics"](primes)
        gs = st.session_state["results"]["gap_statistics"]

        col1, col2 = st.columns(2)
        with col1:
            fig_gh = M["plot_gap_histogram"](gs)
            st.plotly_chart(fig_gh, use_container_width=True)
        with col2:
            fig_scatter = M["plot_gaps_scatter"](primes, gs)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Normalized Gaps — Cramér Model")
        if "normalized_gaps" not in st.session_state["results"]:
            with st.spinner("Computing normalized gaps..."):
                ng = M["normalized_gaps"](primes)
                st.session_state["results"]["normalized_gaps"] = {
                    "norm_gaps": ng, "primes": primes[:-1]
                }
        fig_ng = M["plot_normalized_gaps"](st.session_state["results"]["normalized_gaps"])
        st.plotly_chart(fig_ng, use_container_width=True)

        st.markdown("#### Maximal Gaps (Record Gaps)")
        if "maximal_gaps" not in st.session_state["results"]:
            with st.spinner("Finding maximal gaps..."):
                st.session_state["results"]["maximal_gaps"] = M["maximal_gaps"](primes)
        fig_mg = M["plot_maximal_gaps"](st.session_state["results"]["maximal_gaps"])
        st.plotly_chart(fig_mg, use_container_width=True)

        # Anomaly detection
        st.markdown("#### 🚨 Anomaly Detection")
        anom_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.5, 0.1)
        with st.spinner("Detecting anomalies..."):
            anom = M["anomaly_detection"](primes, anom_thresh)
        st.session_state["results"]["anomaly_detection"] = anom

        fig_anom = M["plot_anomalies"](primes, anom)
        st.plotly_chart(fig_anom, use_container_width=True)

        if anom and anom["n_anomalies"] > 0:
            st.markdown(
                f'<div class="alert-box">⚠️ Detected <b>{anom["n_anomalies"]}</b> anomalous gaps '
                f'with |z| > {anom_thresh:.1f}. These primes have unusually large/small gaps compared '
                f'to the local density prediction.</div>',
                unsafe_allow_html=True
            )
            df_anom = pd.DataFrame({
                "Prime p": anom["anomaly_primes"],
                "Gap": anom["anomaly_gaps"],
                "Z-score": anom["anomaly_z_scores"].round(3),
            })
            st.dataframe(df_anom.head(20), use_container_width=True, hide_index=True)

    # ── Tab 3: Twins & Constellations ─────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 👥 Twin Primes & Prime Constellations")
        if "twin_primes" not in st.session_state["results"]:
            with st.spinner("Finding twin primes..."):
                pairs = M["find_twin_primes"](primes)
                st.session_state["results"]["twin_primes"] = {
                    "pairs": pairs, "count": len(pairs)
                }
        twin_data = st.session_state["results"]["twin_primes"]
        pairs = twin_data.get("pairs", np.array([]).reshape(0, 2))

        c1, c2, c3 = st.columns(3)
        metric_card("Twin Pairs", f"{len(pairs):,}", c1)
        twin_density = len(pairs) / len(primes) * 100 if len(primes) > 0 else 0
        metric_card("Twin Density", f"{twin_density:.2f}%", c2)
        last_twin = f"{int(pairs[-1, 0]):,}" if len(pairs) > 0 else "—"
        metric_card("Last Twin", last_twin, c3)
        st.markdown("")

        fig_twin = M["plot_twin_prime_density"](primes, twin_data)
        st.plotly_chart(fig_twin, use_container_width=True)

        # Show table of first/last twins
        if len(pairs) > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**First 15 twin pairs**")
                df_first = pd.DataFrame(pairs[:15], columns=["p", "p+2"])
                st.dataframe(df_first, use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Last 15 twin pairs**")
                df_last = pd.DataFrame(pairs[-15:], columns=["p", "p+2"])
                st.dataframe(df_last, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🌌 Custom Constellation Search")
        st.caption("Enter a gap pattern to find prime constellations, e.g. `2,4` = (p, p+2, p+6)")
        pattern_str = st.text_input("Gap pattern (comma-separated)", "2,4")
        if st.button("Search constellation"):
            try:
                pattern = [int(x.strip()) for x in pattern_str.split(",")]
                with st.spinner(f"Searching for pattern {pattern}..."):
                    found = M["find_constellation"](primes, pattern)
                st.markdown(f"**Found {len(found):,} constellations** with pattern `{pattern}`")
                if len(found) > 0:
                    df_const = pd.DataFrame({
                        "Starting prime": found[:50],
                        "Next primes": [f"{p+sum(pattern[:i+1]):,}" for i in range(len(pattern)) for p in found[:1]][:50]
                    })
                    st.dataframe(pd.DataFrame({"Starting prime": found[:50]}),
                                use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Invalid pattern: {e}")

        # Heatmap
        st.markdown("---")
        st.markdown("### 🗺️ Prime Density Heatmap")
        grid_sz = st.slider("Grid resolution", 10, 60, 25)
        fig_hm = M["plot_prime_heatmap"](primes, grid_sz)
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab 4: Conjectures / Statistical Tests ─────────────────────────────────
    with tabs[4]:
        st.markdown("### 🧪 Conjecture Validation & Statistical Tests")

        test_cols = st.columns(3)
        with test_cols[0]:
            run_cramer = st.button("Run Cramér KS Test", use_container_width=True)
        with test_cols[1]:
            run_chi = st.button("Run Gap χ² Test", use_container_width=True)
        with test_cols[2]:
            run_poisson = st.button("Run Poisson Test", use_container_width=True)

        if run_cramer:
            with st.spinner("Running Kolmogorov-Smirnov test vs Exp(1)..."):
                res = M["test_cramer_model"](primes)
                st.session_state["results"]["cramer_test"] = res

        if run_chi:
            with st.spinner("Running chi-square test on gap distribution..."):
                res = M["test_gap_distribution"](primes)
                st.session_state["results"]["gap_chi_square"] = res

        if run_poisson:
            interval_sz = st.number_input("Interval size", value=1000, min_value=10)
            with st.spinner("Running Poisson test..."):
                res = M["test_poisson_model"](primes, int(interval_sz))
                st.session_state["results"]["poisson_test"] = res

        # Display results
        for test_key, test_name in [
            ("cramer_test", "Cramér Model (Exp(1) gaps)"),
            ("gap_chi_square", "Gap Distribution χ²"),
            ("poisson_test", "Poisson Count Model"),
        ]:
            if test_key in st.session_state["results"]:
                res = st.session_state["results"][test_key]
                if not res:
                    continue
                with st.expander(f"📊 {test_name}", expanded=True):
                    reject = res.get("reject_h0") or res.get("reject_uniform") or res.get("reject_poisson", False)
                    tag = "tag-fail" if reject else "tag-pass"
                    tag_text = "REJECT H₀" if reject else "FAIL TO REJECT H₀"

                    col_r, col_d = st.columns([1, 3])
                    with col_r:
                        st.markdown(f'<span class="{tag}">{tag_text}</span>', unsafe_allow_html=True)
                        p_val = res.get("p_value", "—")
                        st.markdown(f"**p-value**: `{p_val:.6f}`" if isinstance(p_val, float) else f"p: {p_val}")
                    with col_d:
                        for k, v in res.items():
                            if k in ("reject_h0", "reject_uniform", "reject_poisson", "model",
                                     "interpretation", "gap_counts", "counts"):
                                continue
                            if isinstance(v, (int, float)):
                                st.markdown(f"- **{k}**: `{v:.6g}`")
                        if "interpretation" in res:
                            st.info(res["interpretation"])
                        if test_key == "poisson_test" and "dispersion_index" in res:
                            di = res["dispersion_index"]
                            if di > 1.1:
                                st.markdown(
                                    '<div class="alert-box">⚠️ Dispersion index > 1 suggests '
                                    'overdispersion — prime counts are more variable than Poisson predicts.</div>',
                                    unsafe_allow_html=True
                                )

        # Smart suggestions
        st.markdown("---")
        st.markdown("### 🤖 Automated Insights")
        with st.spinner("Analyzing patterns..."):
            insights = []
            gs = st.session_state["results"].get("gap_statistics")
            if gs:
                if gs.get("gap_max", 0) > 2 * gs.get("gap_mean", 1) * 10:
                    insights.append(("warn", f"Unusually large max gap ({gs['gap_max']}) detected — "
                                    f"consider anomaly analysis."))
                twin_frac = twin_count / max(len(primes), 1) * 100
                if twin_frac < 1.0:
                    insights.append(("info", f"Twin prime density is low ({twin_frac:.2f}%) "
                                    f"for this range — expected near large primes."))
            cramer = st.session_state["results"].get("cramer_test")
            if cramer:
                if cramer.get("reject_h0"):
                    insights.append(("warn", "KS test rejects Cramér's model — "
                                    "normalized gaps do not follow Exp(1). Investigate with FFT."))
                else:
                    insights.append(("pass", "Cramér model consistent with observed gap distribution."))

            if not insights:
                insights.append(("info", "Run conjecture tests to generate automated insights."))

            for kind, msg in insights:
                icon = "✅" if kind == "pass" else "⚠️" if kind == "warn" else "ℹ️"
                border = "#00FF87" if kind == "pass" else "#FFB800" if kind == "warn" else "#00D4FF"
                st.markdown(
                    f'<div class="alert-box" style="border-left-color:{border};">{icon} {msg}</div>',
                    unsafe_allow_html=True
                )

    # ── Tab 5: Spectral Analysis ──────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 📡 Spectral & Autocorrelation Analysis")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run FFT Analysis", use_container_width=True):
                max_n = st.session_state.get("fft_max_n", 50000)
                with st.spinner("Computing FFT of prime indicator function..."):
                    fft_res = M["fft_analysis"](primes, max_n)
                    st.session_state["results"]["fft_analysis"] = fft_res
        with col2:
            if st.button("Run Autocorrelation", use_container_width=True):
                with st.spinner("Computing autocorrelation of prime gaps..."):
                    acf_res = M["autocorrelation"](primes)
                    st.session_state["results"]["autocorrelation"] = acf_res

        max_fft = st.slider("FFT series length (points)", 1000, 200000, 50000, 1000,
                             key="fft_max_n", help="Longer = more resolution but slower")

        if "fft_analysis" in st.session_state["results"]:
            fft_data = st.session_state["results"]["fft_analysis"]
            st.markdown(f"**Series length**: `{fft_data.get('series_length', '?')}` | "
                        f"**Dominant frequency**: `{fft_data.get('dominant_freq', 0):.6f}`")
            fig_fft = M["plot_fft"](fft_data)
            st.plotly_chart(fig_fft, use_container_width=True)
            st.markdown(
                '<div class="alert-box">ℹ️ The FFT of the prime indicator function shows no sharp peaks '
                'consistent with the pseudorandom nature of prime distribution. '
                'Deviation from white noise would indicate unexpected periodicity.</div>',
                unsafe_allow_html=True
            )

        if "autocorrelation" in st.session_state["results"]:
            acf_data = st.session_state["results"]["autocorrelation"]
            sig_lags = acf_data.get("significant_lags", [])
            if len(sig_lags) > 0:
                st.markdown(
                    f'<div class="alert-box">⚠️ Significant autocorrelation found at lags: '
                    f'<code>{", ".join(map(str, sig_lags[:10]))}</code></div>',
                    unsafe_allow_html=True
                )
            fig_acf = M["plot_autocorrelation"](acf_data)
            st.plotly_chart(fig_acf, use_container_width=True)

    # ── Tab 6: Experiments ────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 🧬 Experiment System")
        exp_mode = st.radio("Mode", ["Run Preset", "Custom Experiment", "Saved Results"],
                            horizontal=True)

        if exp_mode == "Run Preset":
            preset_names = list(M["PRESET_EXPERIMENTS"].keys())
            selected_preset = st.selectbox("Select preset", preset_names)
            preset = M["PRESET_EXPERIMENTS"][selected_preset]

            with st.expander("Experiment Details", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Name**: {preset.name}")
                    st.markdown(f"**Range**: [{preset.low:,}, {preset.high:,}]")
                    st.markdown(f"**Description**: {preset.description}")
                with col_b:
                    st.markdown(f"**Metrics**: {', '.join(preset.metrics)}")
                    st.markdown(f"**Models**: {', '.join(preset.models)}")
                    st.markdown(f"**Tags**: {', '.join(preset.tags)}")

            if st.button("▶️ Run Experiment", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def on_progress(msg, pct):
                    status_text.text(f"[{pct}%] {msg}")
                    progress_bar.progress(pct)

                result = M["run_experiment"](preset, on_progress)
                summary = result.to_summary()
                result.save()

                st.success(f"✓ Experiment complete in {summary['computation_time_s']}s")
                st.json(summary)

        elif exp_mode == "Custom Experiment":
            st.markdown("#### Define Custom Experiment")
            exp_name = st.text_input("Experiment name", "My Experiment")
            exp_low = st.number_input("Low", value=2, min_value=2, format="%d")
            exp_high = st.number_input("High", value=100000, min_value=3, format="%d")
            exp_desc = st.text_area("Description", "Custom experiment")

            selected_metrics = st.multiselect(
                "Metrics to compute",
                M["AVAILABLE_METRICS"],
                default=["pi_comparison", "gap_statistics", "twin_primes"],
            )
            exp_interval = st.number_input("Interval size (Poisson)", value=1000, min_value=10)
            exp_anom = st.slider("Anomaly threshold", 2.0, 5.0, 3.5, 0.1)

            if st.button("▶️ Run Custom Experiment", use_container_width=True):
                try:
                    cfg = M["ExperimentConfig"](
                        name=exp_name,
                        low=int(exp_low),
                        high=int(exp_high),
                        metrics=selected_metrics,
                        models=["x_over_logx", "li_x"],
                        description=exp_desc,
                        interval_size=int(exp_interval),
                        anomaly_threshold=float(exp_anom),
                    )
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress2(msg, pct):
                        status_text.text(f"[{pct}%] {msg}")
                        progress_bar.progress(pct)

                    result = M["run_experiment"](cfg, on_progress2)
                    result.save()
                    st.success(f"✓ Done: {result.n_primes:,} primes, {result.computation_time:.2f}s")
                    if result.errors:
                        st.warning(f"Errors: {result.errors}")
                    st.json(result.to_summary())
                except Exception as e:
                    st.error(f"Experiment failed: {e}")

        elif exp_mode == "Saved Results":
            saved = M["list_saved_experiments"]()
            if not saved:
                st.info("No saved experiments yet. Run an experiment to save results.")
            else:
                for exp_sum in saved:
                    with st.expander(f"📁 {exp_sum.get('name', 'Unknown')} — {exp_sum.get('experiment_id', '')}"):
                        st.json(exp_sum)

    # ── Tab 7: Export ─────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("### 💾 Export Data & Figures")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📄 Data Export")
            if st.button("Export primes → CSV", use_container_width=True):
                path = M["primes_to_csv"](primes, "primes.csv")
                st.success(f"Saved: `{path}`")
                # Download
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download primes.csv", f.read(),
                                      "primes.csv", "text/csv")

            if st.button("Export gaps → CSV", use_container_width=True):
                path = M["gaps_to_csv"](primes, "gaps.csv")
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download gaps.csv", f.read(),
                                      "gaps.csv", "text/csv")

            if "pi_comparison" in st.session_state["results"]:
                if st.button("Export π(x) data → CSV", use_container_width=True):
                    path = M["pi_comparison_to_csv"](
                        st.session_state["results"]["pi_comparison"], "pi_comparison.csv"
                    )
                    with open(path, "rb") as f:
                        st.download_button("⬇️ Download pi_comparison.csv", f.read(),
                                          "pi_comparison.csv", "text/csv")

            st.markdown("#### 🔢 NumPy Export")
            if st.button("Export primes → .npz", use_container_width=True):
                from exports.exporter import export_numpy
                path = export_numpy({"primes": primes}, "primes.npz")
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download primes.npz", f.read(),
                                      "primes.npz", "application/octet-stream")

        with col2:
            st.markdown("#### 📊 Figure Export")
            avail_figs = []
            if "pi_comparison" in st.session_state["results"]:
                avail_figs.append("π(x) comparison")
            if "gap_statistics" in st.session_state["results"]:
                avail_figs.append("Gap histogram")
            if "fft_analysis" in st.session_state["results"]:
                avail_figs.append("FFT spectrum")

            if avail_figs:
                fig_choice = st.selectbox("Choose figure", avail_figs)
                fmt = st.radio("Format", ["HTML", "JSON"], horizontal=True)

                if st.button("Export figure", use_container_width=True):
                    fig_map = {
                        "π(x) comparison": lambda: M["plot_pi_comparison"](
                            st.session_state["results"]["pi_comparison"]),
                        "Gap histogram": lambda: M["plot_gap_histogram"](
                            st.session_state["results"]["gap_statistics"]),
                        "FFT spectrum": lambda: M["plot_fft"](
                            st.session_state["results"]["fft_analysis"]),
                    }
                    fig = fig_map[fig_choice]()
                    fname = fig_choice.replace("(", "").replace(")", "").replace(" ", "_").lower()
                    path = M["export_figure"](fig, f"{fname}.html", "html")
                    with open(path, "rb") as f:
                        st.download_button(f"⬇️ Download {fname}.html", f.read(),
                                          f"{fname}.html", "text/html")
            else:
                st.info("Compute analyses first to enable figure export.")

            st.markdown("#### 📋 Full Results JSON")
            if st.button("Export all results → JSON", use_container_width=True):
                from exports.exporter import export_json, _numpy_serializer
                # Make serializable
                serializable = {}
                for k, v in st.session_state["results"].items():
                    if isinstance(v, dict):
                        serializable[k] = {
                            kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                            for kk, vv in v.items()
                        }
                    else:
                        serializable[k] = str(v)
                path = export_json(serializable, "primelab_results.json")
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download results.json", f.read(),
                                      "primelab_results.json", "application/json")

else:
    st.markdown("""
    <div class="prime-card" style="text-align:center; padding:60px 40px;">
        <div style="font-size:3rem; margin-bottom:16px;">🔬</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem;
             color:#00D4FF; margin-bottom:12px;">
            Welcome to PrimeLab
        </div>
        <div style="color:#64748B; max-width:500px; margin:0 auto; line-height:1.7;">
            Configure a number range in the sidebar and click
            <strong style="color:#E2E8F0;">Generate Primes</strong>
            to begin your mathematical research.
            <br><br>
            The lab will compute primes using a <em>Segmented Sieve of Eratosthenes</em>
            and enable deep analysis of their distribution.
        </div>
    </div>
    """, unsafe_allow_html=True)
