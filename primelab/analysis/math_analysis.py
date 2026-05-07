"""
PrimeLab Analysis: Mathematical Analysis Functions
π(x), Li(x), gaps, twin primes, constellations, FFT, statistical tests.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from scipy import stats, special, fft as scipy_fft
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
#  Prime Counting & Approximations
# ─────────────────────────────────────────────────────────────────────────────

def pi_x(primes: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Exact π(x): count primes ≤ x for each x in x_values."""
    return np.searchsorted(primes, x_values, side="right")


def approx_pi_log(x: np.ndarray) -> np.ndarray:
    """Approximation π(x) ≈ x / ln(x)  (Gauss 1792)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(x > 1, x / np.log(np.maximum(x, 2)), 0.0)
    return result


def li_integrand(t: float) -> float:
    """Integrand 1/ln(t) for Li(x)."""
    if t <= 1.0 or abs(t - 1.0) < 1e-10:
        return 0.0
    return 1.0 / math.log(t)


def li_x_single(x: float) -> float:
    """Logarithmic integral Li(x) = ∫₂ˣ dt/ln(t)."""
    if x <= 2:
        return 0.0
    result, _ = quad(li_integrand, 2, x, limit=200)
    return result


def li_x(x_values: np.ndarray) -> np.ndarray:
    """Vectorized Li(x) computation."""
    return np.array([li_x_single(float(x)) for x in x_values])


def approx_pi_riemann_r(x: np.ndarray, n_terms: int = 20) -> np.ndarray:
    """
    Riemann R function: R(x) = Σ μ(n)/n · Li(x^(1/n))
    Better approximation than Li(x).
    """
    from sympy import mobius, Integer
    results = np.zeros(len(x))
    for n in range(1, n_terms + 1):
        mu_n = int(mobius(Integer(n)))
        if mu_n == 0:
            continue
        x_power = np.power(x.astype(float), 1.0 / n)
        li_vals = li_x(x_power)
        results += (mu_n / n) * li_vals
    return results


def pi_comparison(primes: np.ndarray, n_points: int = 200) -> Dict:
    """
    Compare π(x) with x/log(x) and Li(x) at n_points values.
    Returns dict with x, pi_exact, pi_log, pi_li, errors.
    """
    if len(primes) == 0:
        return {}
    x_max = primes[-1]
    x_min = max(primes[0], 10)
    x_vals = np.unique(np.concatenate([
        np.logspace(np.log10(x_min), np.log10(x_max), n_points).astype(np.int64),
        primes[np.linspace(0, len(primes)-1, min(50, len(primes))).astype(int)]
    ]))
    x_vals = np.sort(x_vals[x_vals >= 2])

    pi_exact = pi_x(primes, x_vals).astype(float)
    pi_log = approx_pi_log(x_vals.astype(float))
    pi_li = li_x(x_vals.astype(float))

    # Avoid division by zero
    safe_pi = np.where(pi_exact > 0, pi_exact, 1.0)

    return {
        "x": x_vals,
        "pi_exact": pi_exact,
        "pi_log": pi_log,
        "pi_li": pi_li,
        "error_log_abs": pi_exact - pi_log,
        "error_li_abs": pi_exact - pi_li,
        "error_log_rel": (pi_exact - pi_log) / safe_pi * 100,
        "error_li_rel": (pi_exact - pi_li) / safe_pi * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Gap Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_gaps(primes: np.ndarray) -> np.ndarray:
    """Compute gaps between consecutive primes."""
    if len(primes) < 2:
        return np.array([], dtype=np.int64)
    return np.diff(primes).astype(np.int64)


def normalized_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Normalized gaps g / ln(p) following Cramér's model.
    Gaps should be approximately Exp(1) distributed.
    """
    if len(primes) < 2:
        return np.array([])
    gaps = compute_gaps(primes)
    log_p = np.log(primes[:-1].astype(float))
    log_p = np.where(log_p > 0, log_p, 1.0)
    return gaps / log_p


def gap_statistics(primes: np.ndarray) -> Dict:
    """Full gap analysis summary."""
    if len(primes) < 2:
        return {}
    gaps = compute_gaps(primes)
    norm_gaps = normalized_gaps(primes)

    return {
        "gaps": gaps,
        "normalized_gaps": norm_gaps,
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "gap_max": int(np.max(gaps)),
        "gap_max_at_prime": int(primes[np.argmax(gaps)]),
        "gap_min": int(np.min(gaps)),
        "gap_distribution": np.bincount(gaps)[2::2],  # even gaps starting at 2
        "most_common_gap": int(np.bincount(gaps).argmax()),
        "norm_gap_mean": float(np.mean(norm_gaps)),
        "norm_gap_std": float(np.std(norm_gaps)),
    }


def maximal_gaps(primes: np.ndarray) -> Dict:
    """
    Find maximal prime gaps: gaps larger than all previous gaps.
    Returns prime positions and gap sizes.
    """
    if len(primes) < 2:
        return {}
    gaps = compute_gaps(primes)
    max_so_far = 0
    max_gap_primes = []
    max_gap_values = []
    for i, g in enumerate(gaps):
        if g > max_so_far:
            max_so_far = g
            max_gap_primes.append(int(primes[i]))
            max_gap_values.append(int(g))
    return {
        "primes_at_max_gap": np.array(max_gap_primes),
        "max_gap_values": np.array(max_gap_values),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Twin Primes & Constellations
# ─────────────────────────────────────────────────────────────────────────────

def find_twin_primes(primes: np.ndarray) -> np.ndarray:
    """Return array of (p, p+2) twin prime pairs."""
    if len(primes) < 2:
        return np.array([]).reshape(0, 2)
    gaps = compute_gaps(primes)
    mask = gaps == 2
    p1 = primes[:-1][mask]
    p2 = primes[1:][mask]
    return np.column_stack([p1, p2])


def count_twin_primes(primes: np.ndarray) -> int:
    return len(find_twin_primes(primes))


def find_constellation(primes: np.ndarray, pattern: List[int]) -> np.ndarray:
    """
    Find prime constellations matching a gap pattern.
    E.g., pattern=[2,4] finds (p, p+2, p+6) cousin triples.
    """
    if len(primes) < len(pattern) + 1:
        return np.array([])
    gaps = compute_gaps(primes)
    results = []
    n = len(pattern)
    for i in range(len(gaps) - n + 1):
        if list(gaps[i:i+n]) == pattern:
            results.append(int(primes[i]))
    return np.array(results, dtype=np.int64)


def prime_density(primes: np.ndarray, window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window density: number of primes per window around each prime.
    """
    if len(primes) == 0:
        return np.array([]), np.array([])
    densities = np.zeros(len(primes))
    for i, p in enumerate(primes):
        lo = np.searchsorted(primes, p - window // 2)
        hi = np.searchsorted(primes, p + window // 2)
        densities[i] = hi - lo
    return primes, densities


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral Analysis (FFT / Autocorrelation)
# ─────────────────────────────────────────────────────────────────────────────

def prime_indicator_series(low: int, high: int, primes: np.ndarray) -> np.ndarray:
    """
    Binary indicator series: 1 at prime positions, 0 elsewhere.
    For FFT analysis.
    """
    arr = np.zeros(high - low + 1, dtype=np.float32)
    idx = primes[(primes >= low) & (primes <= high)] - low
    arr[idx] = 1.0
    return arr


def fft_analysis(primes: np.ndarray, max_n: int = 10**5) -> Dict:
    """
    FFT of the prime indicator function.
    Returns frequencies and power spectrum.
    """
    if len(primes) == 0:
        return {}
    low, high = int(primes[0]), min(int(primes[-1]), int(primes[0]) + max_n)
    series = prime_indicator_series(low, high, primes)

    n = len(series)
    freqs = scipy_fft.rfftfreq(n)
    spectrum = np.abs(scipy_fft.rfft(series))**2

    return {
        "freqs": freqs,
        "power": spectrum,
        "dominant_freq": float(freqs[np.argmax(spectrum[1:]) + 1]),
        "series_length": n,
    }


def autocorrelation(primes: np.ndarray, max_lag: int = 1000) -> Dict:
    """
    Autocorrelation of prime gaps.
    """
    if len(primes) < 3:
        return {}
    gaps = compute_gaps(primes).astype(float)
    gaps -= gaps.mean()
    n = len(gaps)
    lags = np.arange(1, min(max_lag, n // 2))
    acf = np.array([np.corrcoef(gaps[:n-lag], gaps[lag:])[0, 1] for lag in lags])
    return {
        "lags": lags,
        "acf": acf,
        "significant_lags": lags[np.abs(acf) > 2 / np.sqrt(n)],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Statistical Tests & Conjecture Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_cramer_model(primes: np.ndarray) -> Dict:
    """
    Test Cramér's conjecture: normalized gaps ~ Exp(1).
    Uses Kolmogorov-Smirnov test.
    """
    if len(primes) < 10:
        return {}
    norm_gaps = normalized_gaps(primes)
    ks_stat, p_value = stats.kstest(norm_gaps, "expon", args=(0, 1))
    return {
        "model": "Cramér (Exp(1))",
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "reject_h0": p_value < 0.05,
        "interpretation": (
            "Consistent with Cramér model" if p_value > 0.05
            else "Deviates from Cramér model"
        ),
    }


def test_gap_distribution(primes: np.ndarray) -> Dict:
    """Chi-square test for even gap frequencies."""
    if len(primes) < 20:
        return {}
    gaps = compute_gaps(primes)
    even_gaps = gaps[gaps % 2 == 0]
    if len(even_gaps) == 0:
        return {}
    counts = np.bincount(even_gaps // 2)
    counts = counts[1:]  # Start from gap 2
    if len(counts) == 0:
        return {}
    # Expected under uniform distribution (very rough baseline)
    total = counts.sum()
    expected = np.full(len(counts), total / len(counts))
    chi2, p_value = stats.chisquare(counts, expected)
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "reject_uniform": p_value < 0.05,
        "gap_counts": counts,
        "most_frequent_gap": int((np.argmax(counts) + 1) * 2),
    }


def test_poisson_model(primes: np.ndarray, interval_size: int = 1000) -> Dict:
    """
    Test Poisson model: count primes in non-overlapping intervals of size `interval_size`.
    Under prime number theorem, rate ≈ interval_size / log(midpoint).
    """
    if len(primes) < 5:
        return {}
    low, high = int(primes[0]), int(primes[-1])
    counts = []
    for start in range(low, high - interval_size, interval_size):
        end = start + interval_size
        n = int(np.searchsorted(primes, end) - np.searchsorted(primes, start))
        counts.append(n)
    counts = np.array(counts)
    mean_count = float(np.mean(counts))
    expected_poisson = stats.poisson(mean_count)
    # KS test against fitted Poisson
    unique_counts = np.arange(int(counts.max()) + 1)
    obs_freq = np.bincount(counts, minlength=len(unique_counts))
    exp_freq = expected_poisson.pmf(unique_counts) * len(counts)
    # Chi-square (merge small bins)
    chi2, p_value = stats.chisquare(obs_freq[obs_freq > 0], exp_freq[obs_freq > 0])
    return {
        "model": "Poisson",
        "interval_size": interval_size,
        "observed_mean": mean_count,
        "observed_var": float(np.var(counts)),
        "dispersion_index": float(np.var(counts) / mean_count) if mean_count > 0 else 0,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "reject_poisson": p_value < 0.05,
        "counts": counts,
    }


def anomaly_detection(primes: np.ndarray, z_threshold: float = 3.5) -> Dict:
    """
    Detect anomalous gaps using Z-score on normalized gaps.
    """
    if len(primes) < 10:
        return {}
    norm_gaps = normalized_gaps(primes)
    z_scores = np.abs((norm_gaps - norm_gaps.mean()) / (norm_gaps.std() + 1e-10))
    anomalies_idx = np.where(z_scores > z_threshold)[0]
    return {
        "anomaly_indices": anomalies_idx,
        "anomaly_primes": primes[anomalies_idx],
        "anomaly_gaps": compute_gaps(primes)[anomalies_idx],
        "anomaly_z_scores": z_scores[anomalies_idx],
        "n_anomalies": len(anomalies_idx),
    }


def convergence_study(primes: np.ndarray, metric: str = "li_error") -> Dict:
    """
    Study convergence of error metrics as x grows.
    """
    if len(primes) < 20:
        return {}
    checkpoints = np.logspace(
        np.log10(max(primes[0], 10)),
        np.log10(primes[-1]),
        30
    ).astype(np.int64)
    checkpoints = np.unique(checkpoints)

    pi_vals = pi_x(primes, checkpoints).astype(float)
    li_vals = li_x(checkpoints.astype(float))
    log_vals = approx_pi_log(checkpoints.astype(float))

    safe_pi = np.where(pi_vals > 0, pi_vals, 1.0)
    li_rel_err = np.abs(pi_vals - li_vals) / safe_pi * 100
    log_rel_err = np.abs(pi_vals - log_vals) / safe_pi * 100

    return {
        "x": checkpoints,
        "pi": pi_vals,
        "li": li_vals,
        "log_approx": log_vals,
        "li_rel_error_pct": li_rel_err,
        "log_rel_error_pct": log_rel_err,
    }
