"""
PrimeLab Experiments: Define, run, save and compare experiments.
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments" / "results"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    low: int
    high: int
    metrics: List[str]
    models: List[str]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    interval_size: int = 1000
    n_fft_points: int = 50000
    twin_prime_search: bool = True
    constellations: List[List[int]] = field(default_factory=list)
    anomaly_threshold: float = 3.5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        return cls(**d)

    def experiment_id(self) -> str:
        key = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]


AVAILABLE_METRICS = [
    "pi_comparison",
    "gap_statistics",
    "maximal_gaps",
    "twin_primes",
    "cramer_test",
    "gap_chi_square",
    "poisson_test",
    "anomaly_detection",
    "fft_analysis",
    "autocorrelation",
    "convergence_study",
    "normalized_gaps",
    "prime_density",
]

AVAILABLE_MODELS = [
    "x_over_logx",
    "li_x",
    "cramer",
    "poisson",
]

PRESET_EXPERIMENTS = {
    "quick_survey": ExperimentConfig(
        name="Quick Survey",
        low=2,
        high=10**5,
        metrics=["pi_comparison", "gap_statistics", "twin_primes"],
        models=["x_over_logx", "li_x"],
        description="Fast overview up to 100,000",
        tags=["quick", "overview"],
    ),
    "gap_deep_dive": ExperimentConfig(
        name="Gap Deep Dive",
        low=2,
        high=10**6,
        metrics=["gap_statistics", "maximal_gaps", "normalized_gaps",
                 "cramer_test", "anomaly_detection", "autocorrelation"],
        models=["cramer"],
        description="Deep analysis of prime gaps up to 1 million",
        tags=["gaps", "research"],
        twin_prime_search=True,
    ),
    "distribution_study": ExperimentConfig(
        name="Distribution Study",
        low=2,
        high=5 * 10**5,
        metrics=["pi_comparison", "convergence_study", "poisson_test", "fft_analysis"],
        models=["x_over_logx", "li_x", "poisson"],
        description="π(x) vs approximations convergence study",
        tags=["distribution", "pi_x"],
        interval_size=500,
    ),
    "twin_prime_hunt": ExperimentConfig(
        name="Twin Prime Hunt",
        low=2,
        high=2 * 10**6,
        metrics=["twin_primes", "gap_statistics", "prime_density"],
        models=["x_over_logx"],
        description="Focus on twin prime distribution and density",
        tags=["twins", "constellations"],
        twin_prime_search=True,
        constellations=[[2], [4], [2, 4], [4, 2]],
    ),
    "large_range": ExperimentConfig(
        name="Large Range (10^9 area)",
        low=10**9,
        high=10**9 + 5 * 10**5,
        metrics=["gap_statistics", "pi_comparison", "cramer_test", "anomaly_detection"],
        models=["li_x", "cramer"],
        description="Primes near 10^9 — tests segmented sieve at scale",
        tags=["large", "scale"],
    ),
}


class ExperimentResult:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = config.experiment_id()
        self.timestamp = time.time()
        self.results: Dict[str, Any] = {}
        self.metadata: Dict = {}
        self.n_primes: int = 0
        self.computation_time: float = 0.0
        self.errors: List[str] = []

    def add_result(self, metric: str, data: Any):
        self.results[metric] = data

    def add_error(self, metric: str, error: str):
        self.errors.append(f"{metric}: {error}")

    def to_summary(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.config.name,
            "range": f"[{self.config.low:,}, {self.config.high:,}]",
            "n_primes": self.n_primes,
            "computation_time_s": round(self.computation_time, 3),
            "metrics_computed": list(self.results.keys()),
            "errors": self.errors,
            "timestamp": self.timestamp,
        }

    def save(self):
        """Save results to disk (metadata + numpy arrays)."""
        result_dir = EXPERIMENTS_DIR / self.experiment_id
        result_dir.mkdir(exist_ok=True)

        # Save config and summary
        with open(result_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        with open(result_dir / "summary.json", "w") as f:
            json.dump(self.to_summary(), f, indent=2, default=str)

        # Save numpy arrays
        for key, val in self.results.items():
            if isinstance(val, dict):
                arr_data = {}
                scalar_data = {}
                for k, v in val.items():
                    if isinstance(v, np.ndarray):
                        arr_data[k] = v
                    else:
                        scalar_data[k] = v
                if arr_data:
                    np.savez(str(result_dir / f"{key}_arrays.npz"), **arr_data)
                if scalar_data:
                    with open(result_dir / f"{key}_scalars.json", "w") as f:
                        json.dump(scalar_data, f, indent=2, default=str)

        return result_dir

    @classmethod
    def load(cls, experiment_id: str) -> Optional["ExperimentResult"]:
        result_dir = EXPERIMENTS_DIR / experiment_id
        if not result_dir.exists():
            return None
        config_path = result_dir / "config.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            config = ExperimentConfig.from_dict(json.load(f))
        result = cls(config)
        return result


def list_saved_experiments() -> List[Dict]:
    """List all saved experiment summaries."""
    summaries = []
    for d in EXPERIMENTS_DIR.iterdir():
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    try:
                        summaries.append(json.load(f))
                    except Exception:
                        pass
    return sorted(summaries, key=lambda x: x.get("timestamp", 0), reverse=True)


def run_experiment(config: ExperimentConfig,
                   progress_callback=None) -> ExperimentResult:
    """
    Execute an experiment: generate primes, compute all metrics.
    """
    from core.sieve import get_primes
    from analysis.math_analysis import (
        pi_comparison, gap_statistics, maximal_gaps,
        find_twin_primes, find_constellation,
        test_cramer_model, test_gap_distribution, test_poisson_model,
        anomaly_detection, fft_analysis, autocorrelation,
        convergence_study, normalized_gaps, prime_density
    )

    result = ExperimentResult(config)
    t_start = time.time()

    def progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    progress("Generating primes...", 5)
    try:
        primes = get_primes(config.low, config.high, use_cache=True)
    except Exception as e:
        result.add_error("sieve", str(e))
        return result

    result.n_primes = len(primes)
    progress(f"Found {len(primes):,} primes", 20)

    metric_fns = {
        "pi_comparison": lambda: pi_comparison(primes),
        "gap_statistics": lambda: gap_statistics(primes),
        "maximal_gaps": lambda: maximal_gaps(primes),
        "twin_primes": lambda: {
            "pairs": find_twin_primes(primes),
            "count": int(np.sum(np.diff(primes) == 2)),
        },
        "cramer_test": lambda: test_cramer_model(primes),
        "gap_chi_square": lambda: test_gap_distribution(primes),
        "poisson_test": lambda: test_poisson_model(primes, config.interval_size),
        "anomaly_detection": lambda: anomaly_detection(primes, config.anomaly_threshold),
        "fft_analysis": lambda: fft_analysis(primes, config.n_fft_points),
        "autocorrelation": lambda: autocorrelation(primes),
        "convergence_study": lambda: convergence_study(primes),
        "normalized_gaps": lambda: {"norm_gaps": normalized_gaps(primes),
                                    "primes": primes[:-1]},
        "prime_density": lambda: dict(zip(
            ["primes", "density"],
            prime_density(primes, window=min(1000, max(10, len(primes) // 100)))
        )),
    }

    n_metrics = len(config.metrics)
    for i, metric in enumerate(config.metrics):
        pct = 20 + int(70 * i / max(n_metrics, 1))
        progress(f"Computing {metric}...", pct)
        if metric in metric_fns:
            try:
                result.add_result(metric, metric_fns[metric]())
            except Exception as e:
                result.add_error(metric, str(e))

    # Constellations
    if config.constellations:
        progress("Finding constellations...", 90)
        const_results = {}
        from analysis.math_analysis import find_constellation
        for pat in config.constellations:
            key = "_".join(map(str, pat))
            try:
                found = find_constellation(primes, pat)
                const_results[f"pattern_{key}"] = {
                    "pattern": pat, "count": len(found),
                    "first_10": found[:10].tolist()
                }
            except Exception as e:
                result.add_error(f"constellation_{key}", str(e))
        if const_results:
            result.add_result("constellations", const_results)

    result.computation_time = time.time() - t_start
    result.metadata = {
        "range": [config.low, config.high],
        "n_primes": result.n_primes,
        "prime_density": result.n_primes / max(config.high - config.low, 1),
    }
    progress("Done!", 100)
    return result
