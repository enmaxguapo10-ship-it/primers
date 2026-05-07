"""
PrimeLab Exports: Save results as CSV, JSON, NumPy, and plots.
"""

import json
import csv
import io
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import base64

EXPORTS_DIR = Path(__file__).parent.parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)


def _numpy_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def export_json(data: Dict, filename: str) -> Path:
    path = EXPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_numpy_serializer)
    return path


def export_csv(rows: list, headers: list, filename: str) -> Path:
    path = EXPORTS_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def export_numpy(arrays: Dict[str, np.ndarray], filename: str) -> Path:
    path = EXPORTS_DIR / filename
    np.savez_compressed(str(path), **arrays)
    return path


def primes_to_csv(primes: np.ndarray, filename: str = "primes.csv") -> Path:
    rows = [(i + 1, int(p)) for i, p in enumerate(primes)]
    return export_csv(rows, ["index", "prime"], filename)


def gaps_to_csv(primes: np.ndarray, filename: str = "gaps.csv") -> Path:
    if len(primes) < 2:
        return export_csv([], ["p", "next_p", "gap"], filename)
    gaps = np.diff(primes)
    rows = [(int(primes[i]), int(primes[i+1]), int(gaps[i])) for i in range(len(gaps))]
    return export_csv(rows, ["p", "next_p", "gap"], filename)


def pi_comparison_to_csv(pi_data: Dict, filename: str = "pi_comparison.csv") -> Path:
    if not pi_data:
        return export_csv([], [], filename)
    x = pi_data.get("x", [])
    pi_e = pi_data.get("pi_exact", [])
    pi_l = pi_data.get("pi_log", [])
    pi_li = pi_data.get("pi_li", [])
    rows = [
        (int(x[i]), int(pi_e[i]), float(pi_l[i]), float(pi_li[i]))
        for i in range(len(x))
    ]
    return export_csv(rows, ["x", "pi_exact", "pi_log", "li_x"], filename)


def fig_to_bytes(fig) -> bytes:
    """Convert a Plotly figure to PNG bytes."""
    return fig.to_image(format="png", scale=2)


def fig_to_html(fig) -> str:
    """Convert a Plotly figure to standalone HTML."""
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def export_figure(fig, filename: str, format: str = "html") -> Path:
    path = EXPORTS_DIR / filename
    if format == "html":
        with open(path, "w") as f:
            f.write(fig_to_html(fig))
    elif format == "json":
        path = path.with_suffix(".json")
        fig.write_json(str(path))
    return path


def get_download_link_data(data: Any, filename: str, mime: str) -> dict:
    """Return base64 encoded data for download buttons in Streamlit."""
    if isinstance(data, str):
        b = data.encode()
    elif isinstance(data, bytes):
        b = data
    else:
        b = json.dumps(data, default=_numpy_serializer).encode()
    encoded = base64.b64encode(b).decode()
    return {"filename": filename, "mime": mime, "b64": encoded}
