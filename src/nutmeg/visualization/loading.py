from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ensure_path, normalize_peak_type, to_numeric_or_nan


RENAME_MAP = {
    # multiome column aliases
    "tfGlobal": "tf_global",
    "creGlobal": "cre_global",
    "geneGlobal": "gene_global",
    "tfName": "tf_name",
    "creName": "cre_name",
    "geneName": "gene_name",
    "nCells": "n_cells",
    "meanPathScore": "mean_path_score",
    "cellFreq": "cell_freq",
    "qValue": "q_value",
    "tfname": "tf_name",
    "crename": "cre_name",
    "genename": "gene_name",
    "cellfreq": "cell_freq",
    "meanpathscore": "mean_path_score",
    "qvalue": "q_value",
    "ncells": "n_cells",
    
    # rna_only significant_edges.csv column names
    "src_name": "tf_name",
    "dst_name": "gene_name",
    "src_global": "tf_global",
    "dst_global": "gene_global",
    "mean_alpha": "mean_path_score",
}


def _detect_mode_from_columns(df: pd.DataFrame) -> str:
    """Infer graph mode from CSV columns."""
    if "cre_name" in df.columns:
        return "multiome"
    return "rna_only"


def load_significant_paths(csv_path):
    """Load significant_paths.csv (multiome) or significant_edges.csv (rna_only)."""
    csv_path = ensure_path(csv_path)
    df = pd.read_csv(csv_path).rename(columns=RENAME_MAP).copy()

    mode = _detect_mode_from_columns(df)

    if mode == "multiome":
        required = ["tf_name", "cre_name", "gene_name"]
    else:
        required = ["tf_name", "gene_name"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    if mode == "rna_only" and "cre_name" not in df.columns:
        df["cre_name"] = None

    if "mean_edge_score" in df.columns and "mean_path_score" not in df.columns:
        df["mean_path_score"] = df["mean_edge_score"]

    for col in ["cell_freq", "mean_path_score", "median_path_score", "max_path_score",
                "q_value", "n_cells", "null_mean", "null_std", "p_value"]:
        if col in df.columns:
            df[col] = to_numeric_or_nan(df[col])

    for col in ["tf_name", "gene_name"]:
        df[col] = df[col].astype(str)
    if "cre_name" in df.columns:
        df["cre_name"] = df["cre_name"].where(df["cre_name"].notna(), other=None)

    df["_mode"] = mode
    return df


def load_peak_annotations(peak_gene_csv):
    peak_gene_csv = ensure_path(peak_gene_csv)
    peaks = pd.read_csv(peak_gene_csv).copy()

    missing = [c for c in ["peak", "peak_type"] if c not in peaks.columns]
    if missing:
        raise ValueError(f"Missing required columns in peak annotation file: {missing}")

    peaks["peak"] = peaks["peak"].astype(str)
    peaks["peak_type"] = peaks["peak_type"].map(normalize_peak_type)

    return peaks[["peak", "peak_type"]].drop_duplicates(subset=["peak"])


def filter_paths(df, max_q_value=0.05, min_cell_freq=0.05, min_n_cells=None, top_n_paths=150):
    out = df.copy()

    if max_q_value is not None and "q_value" in out.columns:
        out = out[out["q_value"].fillna(float("inf")) <= max_q_value].copy()

    if min_cell_freq is not None and "cell_freq" in out.columns:
        out = out[out["cell_freq"].fillna(0) >= min_cell_freq].copy()

    if min_n_cells is not None and "n_cells" in out.columns:
        out = out[out["n_cells"].fillna(0) >= min_n_cells].copy()

    sort_cols = [c for c in ["cell_freq", "mean_path_score", "n_cells"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).copy()

    if top_n_paths is not None:
        out = out.head(int(top_n_paths)).copy()

    return out.reset_index(drop=True)


def infer_class_name_from_path(csv_path, class_depth=2):
    csv_path = ensure_path(csv_path)
    parent = csv_path.parent
    for _ in range(max(int(class_depth) - 1, 0)):
        parent = parent.parent
    return parent.name


def find_class_csvs(interpretation_dir, filename=None, class_depth=2):
    """
    Find per-class significant CSV files under interpretation_dir.
    Supports both multiome (significant_paths.csv) and rna_only (significant_edges.csv).
    If filename is None, auto-detects which is present.
    """
    interpretation_dir = ensure_path(interpretation_dir)

    if filename is not None:
        files = sorted(interpretation_dir.rglob(filename))
        if not files:
            raise FileNotFoundError(f"No '{filename}' files found under {interpretation_dir}")
    else:
        files = sorted(interpretation_dir.rglob("significant_paths.csv"))
        if not files:
            files = sorted(interpretation_dir.rglob("significant_edges.csv"))
        if not files:
            raise FileNotFoundError(
                f"No 'significant_paths.csv' or 'significant_edges.csv' found under {interpretation_dir}"
            )

    out = {}
    for file in files:
        out[infer_class_name_from_path(file, class_depth=class_depth)] = file

    return out


def load_class_table(class_to_csv, class_name):
    if class_name not in class_to_csv:
        raise KeyError(f"Class '{class_name}' not found. Available: {sorted(class_to_csv)}")
    return load_significant_paths(class_to_csv[class_name])