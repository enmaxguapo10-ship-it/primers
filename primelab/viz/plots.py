"""
PrimeLab Viz: All Plotly visualizations.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = {
    "prime": "#00D4FF",
    "li": "#FF6B35",
    "log": "#A855F7",
    "exact": "#22C55E",
    "gap": "#F59E0B",
    "twin": "#EC4899",
    "anomaly": "#EF4444",
    "bg": "#0A0E1A",
    "grid": "#1E2D40",
    "text": "#E2E8F0",
    "secondary": "#64748B",
}

TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=COLORS["grid"]),
        margin=dict(l=60, r=30, t=50, b=50),
    )
)


def apply_template(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**TEMPLATE["layout"], title=dict(text=title, font=dict(size=16)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  π(x) vs Approximations
# ─────────────────────────────────────────────────────────────────────────────

def plot_pi_comparison(pi_data: Dict) -> go.Figure:
    if not pi_data:
        return go.Figure()
    x = pi_data["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["pi_exact"], name="π(x) exact",
        line=dict(color=COLORS["exact"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["pi_li"], name="Li(x)",
        line=dict(color=COLORS["li"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["pi_log"], name="x/ln(x)",
        line=dict(color=COLORS["log"], width=2, dash="dot"),
    ))
    apply_template(fig, "π(x) — Prime Counting Function vs Approximations")
    fig.update_layout(
        xaxis_title="x", yaxis_title="π(x)",
        hovermode="x unified",
    )
    return fig


def plot_pi_error(pi_data: Dict) -> go.Figure:
    if not pi_data:
        return go.Figure()
    x = pi_data["x"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Absolute Error", "Relative Error (%)"])
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["error_li_abs"], name="Li(x) error",
        line=dict(color=COLORS["li"]), fill="tozeroy",
        fillcolor="rgba(255,107,53,0.15)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["error_log_abs"], name="x/ln(x) error",
        line=dict(color=COLORS["log"]),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["error_li_rel"], name="Li(x) % error",
        line=dict(color=COLORS["li"]),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=pi_data["error_log_rel"], name="x/ln(x) % error",
        line=dict(color=COLORS["log"]),
    ), row=2, col=1)
    apply_template(fig, "Approximation Error Analysis")
    fig.update_xaxes(title_text="x", row=2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Gap Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_gap_histogram(gap_stats: Dict) -> go.Figure:
    if not gap_stats or "gaps" not in gap_stats:
        return go.Figure()
    gaps = gap_stats["gaps"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=gaps, nbinsx=min(60, len(np.unique(gaps))),
        marker=dict(color=COLORS["gap"], opacity=0.85,
                    line=dict(color=COLORS["bg"], width=1)),
        name="Gap frequency",
    ))
    apply_template(fig, "Prime Gap Distribution")
    fig.update_layout(
        xaxis_title="Gap size",
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


def plot_normalized_gaps(norm_data: Dict) -> go.Figure:
    if not norm_data or "norm_gaps" not in norm_data:
        return go.Figure()
    norm_gaps = norm_data["norm_gaps"]
    primes = norm_data.get("primes", np.arange(len(norm_gaps)))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Normalized Gaps vs p", "Histogram vs Exp(1)"])
    # Scatter
    sample = min(5000, len(norm_gaps))
    idx = np.random.choice(len(norm_gaps), sample, replace=False)
    fig.add_trace(go.Scatter(
        x=primes[idx], y=norm_gaps[idx], mode="markers",
        marker=dict(color=COLORS["prime"], size=3, opacity=0.5),
        name="g / ln(p)",
    ), row=1, col=1)
    # Histogram vs Exp(1)
    x_hist = np.linspace(0, max(5, norm_gaps.max()), 200)
    exp_pdf = np.exp(-x_hist)
    fig.add_trace(go.Histogram(
        x=norm_gaps, nbinsx=60, histnorm="probability density",
        marker_color=COLORS["gap"], opacity=0.75, name="Observed",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_hist, y=exp_pdf, line=dict(color=COLORS["anomaly"], width=2),
        name="Exp(1) model",
    ), row=1, col=2)
    apply_template(fig, "Normalized Prime Gaps — Cramér Model Comparison")
    return fig


def plot_gaps_scatter(primes: np.ndarray, gap_stats: Dict) -> go.Figure:
    if not gap_stats or "gaps" not in gap_stats or len(primes) < 2:
        return go.Figure()
    gaps = gap_stats["gaps"]
    p_vals = primes[:-1]
    sample = min(10000, len(gaps))
    idx = np.random.choice(len(gaps), sample, replace=False)

    log_p = np.log(p_vals[idx].astype(float))
    expected = log_p  # Cramér expectation

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_vals[idx], y=gaps[idx], mode="markers",
        marker=dict(color=COLORS["gap"], size=3, opacity=0.5),
        name="Gap(p)",
    ))
    fig.add_trace(go.Scatter(
        x=p_vals[idx], y=expected,
        line=dict(color=COLORS["li"], dash="dash", width=2),
        name="ln(p) — Cramér expected",
    ))
    apply_template(fig, "Prime Gaps vs p — Scatter with Cramér Prediction")
    fig.update_layout(xaxis_title="p", yaxis_title="gap")
    return fig


def plot_maximal_gaps(max_gaps: Dict) -> go.Figure:
    if not max_gaps:
        return go.Figure()
    p_vals = max_gaps["primes_at_max_gap"]
    g_vals = max_gaps["max_gap_values"]
    log_p = np.log(p_vals.astype(float))
    log_p2 = log_p ** 2  # Cramér bound

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_vals, y=g_vals, mode="lines+markers",
        line=dict(color=COLORS["prime"]), marker=dict(size=8),
        name="Maximal gap",
    ))
    fig.add_trace(go.Scatter(
        x=p_vals, y=log_p, line=dict(color=COLORS["log"], dash="dot"),
        name="ln(p)",
    ))
    fig.add_trace(go.Scatter(
        x=p_vals, y=log_p2, line=dict(color=COLORS["li"], dash="dash"),
        name="ln²(p) — Cramér bound",
    ))
    apply_template(fig, "Maximal Prime Gaps — Record Gaps")
    fig.update_layout(xaxis_title="p", yaxis_title="gap", xaxis_type="log")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Twin Primes
# ─────────────────────────────────────────────────────────────────────────────

def plot_twin_prime_density(primes: np.ndarray, twin_data: Dict,
                            window: int = 100) -> go.Figure:
    if not twin_data or "pairs" not in twin_data:
        return go.Figure()
    pairs = twin_data["pairs"]
    if len(pairs) == 0:
        return go.Figure()
    twin_p = pairs[:, 0]

    # Cumulative count
    cum_twins = np.arange(1, len(twin_p) + 1)
    # Expected: Brun-type estimate ~ C₂ · x / ln²(x)
    x = twin_p.astype(float)
    C2 = 1.3203236  # Twin prime constant
    brun_est = C2 * x / np.log(x)**2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twin_p, y=cum_twins, name="π₂(x) exact",
        line=dict(color=COLORS["twin"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=twin_p, y=brun_est, name="C₂·x/ln²(x) estimate",
        line=dict(color=COLORS["log"], dash="dash", width=2),
    ))
    apply_template(fig, "Twin Prime Counting Function π₂(x)")
    fig.update_layout(xaxis_title="x", yaxis_title="π₂(x)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Density Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_prime_heatmap(primes: np.ndarray, grid_size: int = 30) -> go.Figure:
    if len(primes) == 0:
        return go.Figure()
    low, high = int(primes[0]), int(primes[-1])
    span = high - low
    cell = max(1, span // grid_size**2)

    # Build 2D grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for p in primes:
        idx = min(int((p - low) / (span + 1) * grid_size**2), grid_size**2 - 1)
        r, c = divmod(idx, grid_size)
        grid[r, c] += 1

    fig = go.Figure(go.Heatmap(
        z=grid, colorscale="Plasma",
        colorbar=dict(title="# primes"),
    ))
    apply_template(fig, "Prime Density Heatmap")
    fig.update_layout(
        xaxis_title="Block (column)", yaxis_title="Block (row)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  FFT / Spectral
# ─────────────────────────────────────────────────────────────────────────────

def plot_fft(fft_data: Dict) -> go.Figure:
    if not fft_data or "freqs" not in fft_data:
        return go.Figure()
    freqs = fft_data["freqs"][1:]  # Skip DC
    power = fft_data["power"][1:]

    # Log scale for better visibility
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=np.log10(power + 1),
        line=dict(color=COLORS["prime"], width=1.2),
        name="log₁₀(Power)",
    ))
    apply_template(fig, "Spectral Analysis — FFT of Prime Indicator Function")
    fig.update_layout(xaxis_title="Frequency", yaxis_title="log₁₀(Power)")
    return fig


def plot_autocorrelation(acf_data: Dict) -> go.Figure:
    if not acf_data or "lags" not in acf_data:
        return go.Figure()
    lags = acf_data["lags"]
    acf = acf_data["acf"]
    n = len(acf) + 1
    ci = 2 / np.sqrt(n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags, y=acf, marker_color=np.where(np.abs(acf) > ci, COLORS["anomaly"], COLORS["prime"]),
        name="ACF",
    ))
    fig.add_hline(y=ci, line_dash="dash", line_color=COLORS["li"],
                  annotation_text=f"95% CI (+{ci:.3f})")
    fig.add_hline(y=-ci, line_dash="dash", line_color=COLORS["li"])
    apply_template(fig, "Autocorrelation of Prime Gaps")
    fig.update_layout(xaxis_title="Lag", yaxis_title="ACF")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(conv_data: Dict) -> go.Figure:
    if not conv_data or "x" not in conv_data:
        return go.Figure()
    x = conv_data["x"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["π(x) and Approximations", "Relative Error (%)"])
    fig.add_trace(go.Scatter(x=x, y=conv_data["pi"],
                             line=dict(color=COLORS["exact"]), name="π(x)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=conv_data["li"],
                             line=dict(color=COLORS["li"], dash="dash"), name="Li(x)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=conv_data["log_approx"],
                             line=dict(color=COLORS["log"], dash="dot"), name="x/ln(x)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=conv_data["li_rel_error_pct"],
                             line=dict(color=COLORS["li"]), name="Li(x) error %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=conv_data["log_rel_error_pct"],
                             line=dict(color=COLORS["log"]), name="x/ln(x) error %"), row=2, col=1)
    apply_template(fig, "Convergence Study — Error as x → ∞")
    fig.update_xaxes(type="log")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Anomalies
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomalies(primes: np.ndarray, anomaly_data: Dict,
                   norm_data: Optional[Dict] = None) -> go.Figure:
    if not anomaly_data or "anomaly_primes" not in anomaly_data:
        return go.Figure()
    anom_p = anomaly_data["anomaly_primes"]
    anom_g = anomaly_data["anomaly_gaps"]
    anom_z = anomaly_data["anomaly_z_scores"]

    if len(primes) < 2:
        return go.Figure()
    all_gaps = np.diff(primes)
    all_p = primes[:-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_p, y=all_gaps, mode="markers",
        marker=dict(color=COLORS["prime"], size=2, opacity=0.4),
        name="All gaps",
    ))
    fig.add_trace(go.Scatter(
        x=anom_p, y=anom_g, mode="markers",
        marker=dict(color=COLORS["anomaly"], size=8, symbol="star",
                    line=dict(color="white", width=1)),
        name=f"Anomalies (z>{anomaly_data.get('anomaly_z_scores', [3.5])[0] if len(anom_z) else 3.5:.1f})",
        text=[f"p={p:,}<br>gap={g}<br>z={z:.2f}"
              for p, g, z in zip(anom_p, anom_g, anom_z)],
        hovertemplate="%{text}<extra></extra>",
    ))
    apply_template(fig, "Anomaly Detection — Unusual Prime Gaps")
    fig.update_layout(xaxis_title="p", yaxis_title="gap")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Summary Dashboard (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(primes: np.ndarray, results: Dict) -> go.Figure:
    """4-panel overview dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "π(x) vs Approximations",
            "Gap Distribution",
            "Normalized Gaps",
            "Maximal Gaps",
        ],
    )
    # π(x)
    if "pi_comparison" in results:
        d = results["pi_comparison"]
        x = d["x"]
        fig.add_trace(go.Scatter(x=x, y=d["pi_exact"], name="π(x)",
                                 line=dict(color=COLORS["exact"])), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["pi_li"], name="Li(x)",
                                 line=dict(color=COLORS["li"], dash="dash")), row=1, col=1)
    # Gaps
    if "gap_statistics" in results:
        g = results["gap_statistics"].get("gaps", [])
        if len(g) > 0:
            fig.add_trace(go.Histogram(x=g, nbinsx=40,
                                       marker_color=COLORS["gap"], name="Gaps",
                                       showlegend=False), row=1, col=2)
    # Normalized gaps
    if "normalized_gaps" in results:
        ng = results["normalized_gaps"].get("norm_gaps", [])
        if len(ng) > 0:
            sample = ng[np.random.choice(len(ng), min(3000, len(ng)), replace=False)]
            fig.add_trace(go.Histogram(x=sample, nbinsx=50, histnorm="probability density",
                                       marker_color=COLORS["prime"], name="Norm gaps",
                                       showlegend=False), row=2, col=1)
            x_e = np.linspace(0, 5, 200)
            fig.add_trace(go.Scatter(x=x_e, y=np.exp(-x_e),
                                     line=dict(color=COLORS["anomaly"]),
                                     name="Exp(1)", showlegend=False), row=2, col=1)
    # Maximal gaps
    if "maximal_gaps" in results:
        mg = results["maximal_gaps"]
        if "primes_at_max_gap" in mg and len(mg["primes_at_max_gap"]) > 0:
            fig.add_trace(go.Scatter(
                x=mg["primes_at_max_gap"], y=mg["max_gap_values"],
                mode="lines+markers", line=dict(color=COLORS["twin"]),
                name="Max gaps", showlegend=False,
            ), row=2, col=2)

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(gridcolor=COLORS["grid"], row=row, col=col)
            fig.update_yaxes(gridcolor=COLORS["grid"], row=row, col=col)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="JetBrains Mono, monospace"),
        height=700,
        title="PrimeLab — Overview Dashboard",
        showlegend=True,
    )
    return fig
