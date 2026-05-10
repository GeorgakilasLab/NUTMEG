from __future__ import annotations

import pandas as pd

from .utils import minmax_scale, normalize_peak_type


def aggregate_edges(df: pd.DataFrame, style_config: dict) -> pd.DataFrame:
    tf_cre = (
        df.groupby(["tf_name", "cre_name"], as_index=False)
        .agg(
            support=("cell_freq", "sum"),
            mean_attention_score=("mean_alpha_g2c", "mean"),
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
            support=("cell_freq", "sum"),
            mean_attention_score=("mean_alpha_c2g", "mean"),
            best_q_value=("q_value", "min"),
            n_paths=("tf_name", "size"),
        )
    )
    cre_gene["source"] = cre_gene["cre_name"].astype(str)
    cre_gene["target"] = cre_gene["gene_name"].astype(str)
    cre_gene["edge_type"] = "CRE_GENE"

    edges = pd.concat([tf_cre, cre_gene], ignore_index=True, sort=False)

    edge_scale = style_config["size_scales"]["edges"]
    edges["edge_width"] = minmax_scale(
        edges["support"].fillna(0.0),
        edge_scale["min"],
        edge_scale["max"],
    )

    edge_colors = {
        "TF_CRE": style_config["edge_styles"]["TF_CRE"]["color"],
        "CRE_GENE": style_config["edge_styles"]["CRE_GENE"]["color"],
    }
    edges["edge_color"] = edges["edge_type"].map(edge_colors).fillna(
        style_config["defaults"]["fallback_edge_color"]
    )

    return edges.reset_index(drop=True)


def build_nodes(
    df: pd.DataFrame,
    peak_annot: pd.DataFrame,
    style_config: dict,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    tf_ids = set(df["tf_name"].astype(str).unique())
    gene_ids = set(df["gene_name"].astype(str).unique())
    cre_ids = set(df["cre_name"].astype(str).unique())

    tf_nodes = pd.DataFrame({"id": sorted(tf_ids)})
    tf_nodes["node_role"] = "TF"
    tf_nodes["node_type"] = "TF"

    gene_nodes = pd.DataFrame({"id": sorted(gene_ids)})
    gene_nodes["node_role"] = "Gene"
    gene_nodes["node_type"] = "Gene"

    cre_nodes = pd.DataFrame({"id": sorted(cre_ids)})
    cre_nodes["node_role"] = "CRE"
    cre_nodes["node_type"] = "CRE"

    nodes = pd.concat([tf_nodes, gene_nodes, cre_nodes], ignore_index=True)
    nodes = nodes.drop_duplicates(subset=["id"], keep="first")
    nodes["label"] = nodes["id"]

    dual_role_ids = tf_ids.intersection(gene_ids)
    nodes.loc[nodes["id"].isin(dual_role_ids), "node_role"] = "TF|Gene"

    tf_support = (
        df.groupby("tf_name", as_index=False)["cell_freq"]
        .sum()
        .rename(columns={"tf_name": "id", "cell_freq": "total_support"})
    )
    gene_support = (
        df.groupby("gene_name", as_index=False)["cell_freq"]
        .sum()
        .rename(columns={"gene_name": "id", "cell_freq": "total_support"})
    )
    cre_support = (
        df.groupby("cre_name", as_index=False)["cell_freq"]
        .sum()
        .rename(columns={"cre_name": "id", "cell_freq": "total_support"})
    )

    support = pd.concat([tf_support, gene_support, cre_support], ignore_index=True)
    support = support.groupby("id", as_index=False)["total_support"].sum()

    nodes = nodes.merge(support, on="id", how="left")
    nodes["total_support"] = nodes["total_support"].fillna(0.0)

    peak_map = peak_annot.rename(columns={"peak": "id"})[["id", "peak_type"]]
    nodes = nodes.merge(peak_map, on="id", how="left")
    nodes["peak_type"] = nodes["peak_type"].fillna("unknown").map(normalize_peak_type)

    tf_target_gene_counts = (
        df.groupby("tf_name", as_index=False)["gene_name"]
        .nunique()
        .rename(columns={"tf_name": "id", "gene_name": "n_targets"})
    )

    gene_source_tf_counts = (
        df.groupby("gene_name", as_index=False)["tf_name"]
        .nunique()
        .rename(columns={"gene_name": "id", "tf_name": "n_sources"})
    )

    cre_source_tf_counts = (
        df.groupby("cre_name", as_index=False)["tf_name"]
        .nunique()
        .rename(columns={"cre_name": "id", "tf_name": "n_sources"})
    )

    cre_target_gene_counts = (
        df.groupby("cre_name", as_index=False)["gene_name"]
        .nunique()
        .rename(columns={"cre_name": "id", "gene_name": "n_targets"})
    )

    dual_source_tf_counts = (
        df[df["gene_name"].astype(str).isin(dual_role_ids)]
        .groupby("gene_name", as_index=False)["tf_name"]
        .nunique()
        .rename(columns={"gene_name": "id", "tf_name": "dual_n_sources"})
    )

    dual_target_gene_counts = (
        df[df["tf_name"].astype(str).isin(dual_role_ids)]
        .groupby("tf_name", as_index=False)["gene_name"]
        .nunique()
        .rename(columns={"tf_name": "id", "gene_name": "dual_n_targets"})
    )

    nodes = nodes.merge(tf_target_gene_counts, on="id", how="left")
    nodes = nodes.merge(gene_source_tf_counts, on="id", how="left")
    nodes = nodes.merge(cre_source_tf_counts, on="id", how="left", suffixes=("", "_cre"))
    nodes = nodes.merge(cre_target_gene_counts, on="id", how="left", suffixes=("", "_cre_target"))
    nodes = nodes.merge(dual_source_tf_counts, on="id", how="left")
    nodes = nodes.merge(dual_target_gene_counts, on="id", how="left")

    nodes["n_sources"] = nodes["n_sources"].fillna(0)
    nodes["n_targets"] = nodes["n_targets"].fillna(0)

    cre_source_mask = nodes["node_role"] == "CRE"
    cre_target_mask = nodes["node_role"] == "CRE"
    dual_mask = nodes["node_role"] == "TF|Gene"

    if "n_sources_cre" in nodes.columns:
        nodes.loc[cre_source_mask, "n_sources"] = nodes.loc[cre_source_mask, "n_sources_cre"].fillna(0)

    if "n_targets_cre_target" in nodes.columns:
        nodes.loc[cre_target_mask, "n_targets"] = nodes.loc[cre_target_mask, "n_targets_cre_target"].fillna(0)

    nodes.loc[dual_mask, "n_sources"] = nodes.loc[dual_mask, "dual_n_sources"].fillna(
        nodes.loc[dual_mask, "n_sources"]
    )
    nodes.loc[dual_mask, "n_targets"] = nodes.loc[dual_mask, "dual_n_targets"].fillna(
        nodes.loc[dual_mask, "n_targets"]
    )

    nodes["n_sources"] = nodes["n_sources"].fillna(0).astype(int)
    nodes["n_targets"] = nodes["n_targets"].fillna(0).astype(int)

    drop_cols = [
        "n_sources_cre",
        "n_targets_cre_target",
        "dual_n_sources",
        "dual_n_targets",
    ]
    drop_cols = [c for c in drop_cols if c in nodes.columns]
    if drop_cols:
        nodes = nodes.drop(columns=drop_cols)

    node_styles = style_config["node_styles"]
    default_border_color = style_config["defaults"]["border_color"]
    default_border_width = style_config["defaults"]["border_width"]
    dual_role_border_width = style_config["defaults"].get("dual_role_border_width", 4)
    node_scale = style_config["size_scales"]["nodes"]

    nodes["node_shape"] = node_styles["Gene"]["shape"]
    nodes.loc[nodes["node_type"] == "TF", "node_shape"] = node_styles["TF"]["shape"]
    nodes.loc[nodes["node_type"] == "CRE", "node_shape"] = node_styles["CRE_unknown"]["shape"]

    nodes["display_label"] = nodes["label"]
    if not style_config["layout"].get("show_cre_labels", False):
        nodes.loc[nodes["node_type"] == "CRE", "display_label"] = ""

    nodes["node_fill_color"] = node_styles["Gene"]["color"]
    nodes.loc[nodes["node_role"].isin(["TF", "TF|Gene"]), "node_fill_color"] = node_styles["TF"]["color"]

    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "promoter"),
        "node_fill_color"
    ] = node_styles["CRE_promoter"]["color"]
    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "distal"),
        "node_fill_color"
    ] = node_styles["CRE_distal"]["color"]
    nodes.loc[
        (nodes["node_role"] == "CRE") & (~nodes["peak_type"].isin(["promoter", "distal"])),
        "node_fill_color"
    ] = node_styles["CRE_unknown"]["color"]

    nodes["node_border_color"] = default_border_color
    nodes["node_border_width"] = default_border_width

    nodes.loc[nodes["node_role"] == "TF", "node_border_color"] = node_styles["TF"].get(
        "border_color", default_border_color
    )
    nodes.loc[nodes["node_role"] == "TF", "node_border_width"] = node_styles["TF"].get(
        "border_width", default_border_width
    )

    nodes.loc[nodes["node_role"] == "Gene", "node_border_color"] = node_styles["Gene"].get(
        "border_color", default_border_color
    )
    nodes.loc[nodes["node_role"] == "Gene", "node_border_width"] = node_styles["Gene"].get(
        "border_width", default_border_width
    )

    nodes.loc[nodes["node_role"] == "TF|Gene", "node_border_color"] = node_styles["Gene"]["color"]
    nodes.loc[nodes["node_role"] == "TF|Gene", "node_border_width"] = dual_role_border_width

    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "promoter"),
        "node_border_color"
    ] = node_styles["CRE_promoter"].get("border_color", default_border_color)
    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "promoter"),
        "node_border_width"
    ] = node_styles["CRE_promoter"].get("border_width", default_border_width)

    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "distal"),
        "node_border_color"
    ] = node_styles["CRE_distal"].get("border_color", default_border_color)
    nodes.loc[
        (nodes["node_role"] == "CRE") & (nodes["peak_type"] == "distal"),
        "node_border_width"
    ] = node_styles["CRE_distal"].get("border_width", default_border_width)

    nodes.loc[
        (nodes["node_role"] == "CRE") & (~nodes["peak_type"].isin(["promoter", "distal"])),
        "node_border_color"
    ] = node_styles["CRE_unknown"].get("border_color", default_border_color)
    nodes.loc[
        (nodes["node_role"] == "CRE") & (~nodes["peak_type"].isin(["promoter", "distal"])),
        "node_border_width"
    ] = node_styles["CRE_unknown"].get("border_width", default_border_width)

    nodes["node_size"] = minmax_scale(
        nodes["total_support"].fillna(0.0),
        node_scale["min"],
        node_scale["max"],
    )

    return nodes.reset_index(drop=True)