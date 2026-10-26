from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import minmax_scale, normalize_peak_type


def _is_rna_only(df: pd.DataFrame) -> bool:
    return "_mode" in df.columns and (df["_mode"] == "rna_only").any() or \
           df.get("cre_name", pd.Series(dtype=object)).isna().all()


# ── Multiome: TF → CRE → Gene ─────────────────────────────────────────────────

def aggregate_edges_multiome(df: pd.DataFrame, style_config: dict) -> pd.DataFrame:
    tf_cre = (
        df.groupby(["tf_name", "cre_name"], as_index=False)
        .agg(
            support=("cell_freq", "max"),
            mean_attention_score=("mean_alpha_g2c", "mean") if "mean_alpha_g2c" in df.columns else ("cell_freq", "mean"),
            best_q_value=("q_value", "min"),
            n_paths=("gene_name", "size"),
        )
    )
    tf_cre["source"] = tf_cre["tf_name"].astype(str)
    tf_cre["target"] = tf_cre["cre_name"].astype(str)
    tf_cre["edge_type"] = "TF_CRE"

    cre_gene = (
        df.groupby(["cre_name", "gene_name"], as_index=False)
        .agg(
            support=("cell_freq", "max"),
            mean_attention_score=("mean_alpha_c2g", "mean") if "mean_alpha_c2g" in df.columns else ("cell_freq", "mean"),
            best_q_value=("q_value", "min"),
            n_paths=("tf_name", "size"),
        )
    )
    cre_gene["source"] = cre_gene["cre_name"].astype(str)
    cre_gene["target"] = cre_gene["gene_name"].astype(str)
    cre_gene["edge_type"] = "CRE_GENE"

    edges = pd.concat([tf_cre, cre_gene], ignore_index=True, sort=False)
    return _scale_edges(edges, style_config)


# ── RNA-only: TF → Gene ────────────────────────────────────────────────────────

def aggregate_edges_rna_only(df: pd.DataFrame, style_config: dict) -> pd.DataFrame:
    score_col = "mean_path_score" if "mean_path_score" in df.columns else "cell_freq"

    tf_gene = (
        df.groupby(["tf_name", "gene_name"], as_index=False)
        .agg(
            support=("cell_freq", "max"),
            mean_attention_score=(score_col, "mean"),
            best_q_value=("q_value", "min"),
            n_paths=("tf_name", "size"),
        )
    )
    tf_gene["source"] = tf_gene["tf_name"].astype(str)
    tf_gene["target"] = tf_gene["gene_name"].astype(str)
    tf_gene["edge_type"] = "TF_GENE"

    return _scale_edges(tf_gene, style_config)


def _scale_edges(edges: pd.DataFrame, style_config: dict) -> pd.DataFrame:
    edge_scale = style_config["size_scales"]["edges"]
    edges["edge_width"] = minmax_scale(
        edges["support"].fillna(0.0),
        edge_scale["min"],
        edge_scale["max"],
    )

    edge_colors = {
        "TF_CRE":  style_config["edge_styles"]["TF_CRE"]["color"],
        "CRE_GENE": style_config["edge_styles"]["CRE_GENE"]["color"],
        "TF_GENE": style_config["edge_styles"].get("TF_GENE", {}).get(
            "color", style_config["defaults"]["fallback_edge_color"]
        ),
    }
    edges["edge_color"] = edges["edge_type"].map(edge_colors).fillna(
        style_config["defaults"]["fallback_edge_color"]
    )
    return edges.reset_index(drop=True)


def aggregate_edges(df: pd.DataFrame, style_config: dict) -> pd.DataFrame:
    if _is_rna_only(df):
        return aggregate_edges_rna_only(df, style_config)
    return aggregate_edges_multiome(df, style_config)


# ── Node builder ───────────────────────────────────────────────────────────────

def build_nodes(
    df: pd.DataFrame,
    peak_annot: pd.DataFrame | None,
    style_config: dict,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    rna_only = _is_rna_only(df)

    tf_ids   = set(df["tf_name"].astype(str).unique())
    gene_ids = set(df["gene_name"].astype(str).unique())

    tf_nodes   = pd.DataFrame({"id": sorted(tf_ids)})
    tf_nodes["node_role"] = "TF"
    tf_nodes["node_type"] = "TF"

    gene_nodes = pd.DataFrame({"id": sorted(gene_ids)})
    gene_nodes["node_role"] = "Gene"
    gene_nodes["node_type"] = "Gene"

    if rna_only:
        nodes = pd.concat([tf_nodes, gene_nodes], ignore_index=True)
    else:
        cre_ids  = set(df["cre_name"].dropna().astype(str).unique())
        cre_nodes = pd.DataFrame({"id": sorted(cre_ids)})
        cre_nodes["node_role"] = "CRE"
        cre_nodes["node_type"] = "CRE"
        nodes = pd.concat([tf_nodes, gene_nodes, cre_nodes], ignore_index=True)

    nodes = nodes.drop_duplicates(subset=["id"], keep="first")
    nodes["label"] = nodes["id"]

    dual_role_ids = tf_ids.intersection(gene_ids)
    nodes.loc[nodes["id"].isin(dual_role_ids), "node_role"] = "TF|Gene"

    # Path-level support rolled up per node.
    support_parts = [
        df.groupby("tf_name",   as_index=False)["cell_freq"].max().rename(columns={"tf_name":   "id", "cell_freq": "max_path_cell_freq"}),
        df.groupby("gene_name", as_index=False)["cell_freq"].max().rename(columns={"gene_name": "id", "cell_freq": "max_path_cell_freq"}),
    ]
    if not rna_only:
        support_parts.append(
            df.dropna(subset=["cre_name"]).groupby("cre_name", as_index=False)["cell_freq"].max()
            .rename(columns={"cre_name": "id", "cell_freq": "max_path_cell_freq"})
        )
    support = pd.concat(support_parts, ignore_index=True)
    support = support.groupby("id", as_index=False)["max_path_cell_freq"].max()

    nodes["id"] = nodes["id"].astype(str)
    support["id"] = support["id"].astype(str)

    nodes = nodes.merge(support, on="id", how="left")
    nodes["max_path_cell_freq"] = nodes["max_path_cell_freq"].fillna(0.0)

    # Peak annotation (multiome only)
    if not rna_only and peak_annot is not None:
        peak_map = peak_annot.rename(columns={"peak": "id"})[["id", "peak_type"]]
        nodes = nodes.merge(peak_map, on="id", how="left")
    nodes["peak_type"] = nodes.get("peak_type", pd.Series("unknown", index=nodes.index)).fillna("unknown").map(normalize_peak_type)

    # n_sources / n_targets
    tf_targets = df.groupby("tf_name",   as_index=False)["gene_name"].nunique().rename(columns={"tf_name": "id",   "gene_name": "n_targets"})
    gene_sources = df.groupby("gene_name", as_index=False)["tf_name"].nunique().rename(columns={"gene_name": "id", "tf_name": "n_sources"})

    nodes = nodes.merge(tf_targets,    on="id", how="left")
    nodes = nodes.merge(gene_sources,  on="id", how="left")

    if not rna_only:
        cre_src  = df.dropna(subset=["cre_name"]).groupby("cre_name", as_index=False)["tf_name"].nunique().rename(columns={"cre_name": "id", "tf_name": "n_sources_cre"})
        cre_tgt  = df.dropna(subset=["cre_name"]).groupby("cre_name", as_index=False)["gene_name"].nunique().rename(columns={"cre_name": "id", "gene_name": "n_targets_cre"})
        nodes = nodes.merge(cre_src, on="id", how="left")
        nodes = nodes.merge(cre_tgt, on="id", how="left")
        cre_mask = nodes["node_role"] == "CRE"
        nodes.loc[cre_mask, "n_sources"] = nodes.loc[cre_mask, "n_sources_cre"].fillna(0)
        nodes.loc[cre_mask, "n_targets"] = nodes.loc[cre_mask, "n_targets_cre"].fillna(0)
        nodes = nodes.drop(columns=[c for c in ["n_sources_cre", "n_targets_cre"] if c in nodes.columns])

    nodes["n_sources"] = nodes["n_sources"].fillna(0).astype(int)
    nodes["n_targets"] = nodes["n_targets"].fillna(0).astype(int)

    # Styles
    node_styles = style_config["node_styles"]
    defaults    = style_config["defaults"]
    size_scales = style_config["size_scales"]

    nodes["node_shape"]        = node_styles["Gene"]["shape"]
    nodes["node_fill_color"]   = node_styles["Gene"]["color"]
    nodes["node_border_color"] = defaults["border_color"]
    nodes["node_border_width"] = defaults["border_width"]

    tf_mask   = nodes["node_role"].isin(["TF", "TF|Gene"])
    dual_mask = nodes["node_role"] == "TF|Gene"
    gene_mask = nodes["node_role"] == "Gene"

    nodes.loc[tf_mask,   "node_shape"]        = node_styles["TF"]["shape"]
    nodes.loc[tf_mask,   "node_fill_color"]   = node_styles["TF"]["color"]
    nodes.loc[tf_mask,   "node_border_color"] = node_styles["TF"].get("border_color", defaults["border_color"])
    nodes.loc[tf_mask,   "node_border_width"] = node_styles["TF"].get("border_width", defaults["border_width"])

    nodes.loc[gene_mask, "node_border_color"] = node_styles["Gene"].get("border_color", defaults["border_color"])
    nodes.loc[gene_mask, "node_border_width"] = node_styles["Gene"].get("border_width", defaults["border_width"])

    nodes.loc[dual_mask, "node_border_color"] = node_styles["Gene"]["color"]
    nodes.loc[dual_mask, "node_border_width"] = defaults.get("dual_role_border_width", 4)

    if not rna_only:
        cre_mask_p = (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "promoter")
        cre_mask_d = (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "distal")
        cre_mask_u = (nodes["node_role"] == "CRE") & (~nodes["peak_type"].isin(["promoter", "distal"]))

        for mask, key in [(cre_mask_p, "CRE_promoter"), (cre_mask_d, "CRE_distal"), (cre_mask_u, "CRE_unknown")]:
            nodes.loc[mask, "node_shape"]        = node_styles[key]["shape"]
            nodes.loc[mask, "node_fill_color"]   = node_styles[key]["color"]
            nodes.loc[mask, "node_border_color"] = node_styles[key].get("border_color", defaults["border_color"])
            nodes.loc[mask, "node_border_width"] = node_styles[key].get("border_width", defaults["border_width"])

    nodes["display_label"] = nodes["label"]
    if not rna_only and not style_config["layout"].get("show_cre_labels", False):
        nodes.loc[nodes["node_type"] == "CRE", "display_label"] = ""

    # ── Node size ─────────────────────────────────────────────────────────────
    text_cfg = style_config["text"]
    font_px  = float(text_cfg["node_font_size"])
    pad      = float(text_cfg.get("label_padding", 5))
    char_w   = float(text_cfg.get("char_width_ratio", 0.62))

    def fit_diameter(labels: pd.Series) -> float:
        n = int(labels.fillna("").str.len().max() or 0)
        if n == 0:
            return 2.0 * pad
        return float(np.hypot(n * font_px * char_w, font_px)) + 2.0 * pad

    labeled = nodes["node_type"].isin(["TF", "Gene"])

    lab_size = size_scales.get("labeled_nodes", {}).get("size")
    if lab_size is None:
        lab_size = fit_diameter(nodes.loc[labeled, "display_label"])
    lab_size = float(lab_size)

    cre_size = float(size_scales.get("cre_nodes", {}).get("size", 28))

    nodes["node_size"] = np.where(nodes["node_type"] == "CRE", cre_size, lab_size)

    return nodes.reset_index(drop=True)