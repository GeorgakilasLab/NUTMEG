from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def ensure_path(x) -> Path:
    return x if isinstance(x, Path) else Path(x)


def load_json(json_path):
    with open(ensure_path(json_path), "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_peak_type(x) -> str:
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    if "promoter" in x:
        return "promoter"
    if "distal" in x:
        return "distal"
    return x if x else "unknown"


def to_numeric_or_nan(series):
    return pd.to_numeric(series, errors="coerce")


def minmax_scale(values, out_min, out_max):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, (out_min + out_max) / 2.0, dtype=float)

    vmin = np.nanmin(arr[finite])
    vmax = np.nanmax(arr[finite])

    if vmin == vmax:
        return np.full(arr.shape, (out_min + out_max) / 2.0, dtype=float)

    scaled = (arr - vmin) / (vmax - vmin)
    return out_min + scaled * (out_max - out_min)