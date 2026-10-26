import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist

from .collect import collect_interpretability
from .contributors import (
    top_genes_for_class,
    top_genes_for_class_linear_shap,
)
from .grn import (
    attention_distribution_df,
    summarize_attention,
    significant_grn_all_classes_batched,
)
from .utils import (
    _safe_name,
    _write_json,
)


def _runner_interpretability_dir(runner):
    if runner is None:
        raise ValueError("runner is required to infer interpretability directory")
    return Path(runner.train_dir).parent.parent / "interpretability" / Path(runner.train_dir).name


def _default_interpretability_dir(bundle, runner=None):
    saved = bundle.get("default_interpretability_dir", None)
    if saved is not None:
        return Path(saved)
    if runner is None:
        raise ValueError("runner is required to infer interpretability directory")
    return _runner_interpretability_dir(runner)


def _ordered_contributor_plot_df(contrib, top_k=10):
    df = contrib["full"].copy()
    value_col = "shap_value" if "shap_value" in df.columns else "contribution"

    neg = df[df[value_col] < 0].nsmallest(top_k, value_col).copy()
    pos = df[df[value_col] > 0].nlargest(top_k, value_col).copy()

    neg = neg.sort_values(value_col, ascending=True).copy()
    pos = pos.sort_values(value_col, ascending=False).copy()

    plot_df = pd.concat([neg, pos], ignore_index=True)
    plot_df["direction"] = np.where(plot_df[value_col] < 0, "negative", "positive")
    return plot_df


def plot_contributor_full(contrib, out_path, class_name, top_k=10):
    plot_df = _ordered_contributor_plot_df(contrib, top_k=top_k)
    if plot_df.empty:
        return None

    value_col = "shap_value" if "shap_value" in plot_df.columns else "contribution"
    xlab = "mean linear SHAP" if value_col == "shap_value" else "mean(score × weight)"

    y = np.arange(len(plot_df))
    colors = np.where(plot_df[value_col] >= 0, "#ff4f79", "#1f77b4")

    plt.figure(figsize=(9, 8))
    plt.axvline(0, color="gray", lw=1)
    plt.barh(y, plot_df[value_col], color=colors, edgecolor="black", linewidth=0.3)
    plt.yticks(y, plot_df["gene"])
    plt.gca().invert_yaxis()
    plt.title(f"{class_name} — dominant contributors")
    plt.xlabel(xlab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_attention_by_edge_type(attn_df, out_path, class_name):
    df = attn_df.copy()
    if df.empty or "relation" not in df.columns:
        return None

    order = [x for x in ["t2g", "g2c", "c2g"] if x in df["relation"].unique()]
    if len(order) == 0:
        return None

    plt.figure(figsize=(7, 5))
    sns.violinplot(
        data=df[df["relation"].isin(order)].copy(),
        x="relation",
        y="alpha",
        order=order,
        palette={
            "t2g": "#4C78A8",
            "g2c": "#54A24B",
            "c2g": "#F58518",
        },
        cut=0,
        inner=None,
    )
    plt.title(f"{class_name} — attention by edge type")
    plt.xlabel("")
    plt.ylabel("attention")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_top_regulators(df, out_path, class_name):
    if df is None or df.empty:
        return None

    name_col = "tf_name" if "tf_name" in df.columns else "src_name"
    top = df.head(20).iloc[::-1]

    plt.figure(figsize=(8, 7))
    plt.barh(top[name_col], top["total_support"], color="#1f77b4")
    plt.title(f"{class_name} — top regulators")
    plt.xlabel("total_support")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_top_paths(df, out_path, class_name, mode):
    if df is None or df.empty:
        return None

    if mode == "multiome":
        score_col = "mean_path_score"
        top = df.nlargest(20, score_col).copy()
        top["label"] = top["tf_name"] + " → " + top["cre_name"] + " → " + top["gene_name"]
        title = f"{class_name} — top TF→CRE→gene paths"
    else:
        score_col = "mean_alpha"
        top = df.nlargest(20, score_col).copy()
        top["label"] = top["src_name"] + " → " + top["dst_name"]
        title = f"{class_name} — top significant edges"

    top = top.iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(top["label"], top[score_col], color="#9467bd")
    plt.title(title)
    plt.xlabel(score_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def save_contributor_result(contrib, out_dir, prefix, class_name, top_k=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / f"{prefix}_full.csv"
    plot_path = out_dir / f"{prefix}_plot.png"

    contrib["full"].to_csv(full_path, index=False)
    plot_contributor_full(contrib, plot_path, class_name, top_k=top_k)

    return {
        "full_csv": str(full_path),
        "plot": str(plot_path),
    }


def save_attention_artifacts(attn_df, out_dir, class_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = out_dir / "attention_by_edge_type.png"
    saved = plot_attention_by_edge_type(attn_df, plot_path, class_name)

    return {
        "plot": str(saved) if saved is not None else None,
    }


def save_significant_grn_artifacts(grn, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regs_path = out_dir / "top_regulators.csv"
    grn["regulators"].to_csv(regs_path, index=False)
    regs_plot = plot_top_regulators(grn["regulators"], out_dir / "top_regulators.png", grn["class_name"])

    info = {
        "top_regulators_csv": str(regs_path),
        "top_regulators_plot": str(regs_plot) if regs_plot is not None else None,
    }

    if grn["mode"] == "rna_only":
        sig_path = out_dir / "significant_edges.csv"
        grn["significant_edges"].to_csv(sig_path, index=False)
        top_plot = plot_top_paths(grn["significant_edges"], out_dir / "top_paths.png", grn["class_name"], "rna_only")
        info["significant_edges_csv"] = str(sig_path)
        info["top_paths_plot"] = str(top_plot) if top_plot is not None else None
    else:
        sig_path = out_dir / "significant_paths.csv"
        grn["significant_paths"].to_csv(sig_path, index=False)
        top_plot = plot_top_paths(grn["significant_paths"], out_dir / "top_paths.png", grn["class_name"], "multiome")
        info["significant_paths_csv"] = str(sig_path)
        info["top_paths_plot"] = str(top_plot) if top_plot is not None else None

    return info


def select_top_features_from_matrix(M, top_k=20, mode="var"):
    M = np.asarray(M, dtype=float)

    if mode == "var":
        score = M.var(axis=0)
    elif mode == "pairwise":
        score = M.max(axis=0) - M.min(axis=0)
    elif mode == "ovr":
        C = M.shape[0]
        if C < 2:
            score = np.zeros(M.shape[1], dtype=float)
        else:
            rest = (M.sum(axis=0, keepdims=True) - M) / (C - 1)
            score = np.abs(M - rest).max(axis=0)
    else:
        raise ValueError("mode must be one of: 'var', 'pairwise', 'ovr'")

    return np.argsort(-score)[:top_k]


def build_matrix_from_tables(tables, feature_col, value_col, top_k_per_class=20, fill_value=0.0):
    class_labels = [str(c) for c in tables.keys()]

    features = []
    cleaned = {}

    for c in class_labels:
        df = tables[c].copy()
        if feature_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Missing columns in class {c}: need {feature_col} and {value_col}")
        df = df[[feature_col, value_col]].dropna().copy()
        df[feature_col] = df[feature_col].astype(str)
        df = df.sort_values(value_col, ascending=False)
        cleaned[c] = df
        features.extend(df.head(top_k_per_class)[feature_col].tolist())

    features = list(dict.fromkeys(features))
    M = np.full((len(class_labels), len(features)), fill_value, dtype=float)

    for i, c in enumerate(class_labels):
        s = cleaned[c].drop_duplicates(subset=[feature_col]).set_index(feature_col)[value_col]
        for j, f in enumerate(features):
            if f in s.index:
                M[i, j] = float(s.loc[f])

    return M, np.asarray(features), np.asarray(class_labels)


def load_csv_tables(base_dir, class_labels, subdir, filename):
    base_dir = Path(base_dir)
    out = {}
    for c in class_labels:
        class_name_safe = _safe_name(c)
        p = base_dir / "classes" / class_name_safe / subdir / filename
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        out[str(c)] = pd.read_csv(p)
    return out


def load_contributor_tables(base_dir, class_labels, filename="linear_shap_full.csv"):
    return load_csv_tables(base_dir, class_labels, subdir="contributors", filename=filename)


def contributor_matrix_from_csvs(
    base_dir,
    class_labels,
    filename="linear_shap_full.csv",
    value_col="shap_importance",
    top_k_per_class=20,
):
    tables = load_contributor_tables(base_dir, class_labels, filename=filename)
    M, features, classes = build_matrix_from_tables(
        tables=tables,
        feature_col="gene",
        value_col=value_col,
        top_k_per_class=top_k_per_class,
    )
    return M, features, classes


def _cluster_threshold(Z, n_clusters):
    if n_clusters is None or n_clusters <= 1 or Z is None or len(Z) == 0:
        return None
    idx = len(Z) - (n_clusters - 1)
    idx = max(0, min(idx, len(Z) - 1))
    return Z[idx, 2]


def plot_biclustered(
    M,
    feature_labels,
    class_labels,
    title="Signature Map",
    cbar_label="value",
    cmap="viridis",
    row_metric="correlation",
    col_metric="correlation",
    method="average",
    n_row_clusters=4,
    n_col_clusters=5,
    figsize=None,
    save_path=None,
    dpi=300,
):
    M = np.asarray(M, dtype=float)
    feature_labels = np.asarray(feature_labels).astype(str)
    class_labels = np.asarray(class_labels).astype(str)

    C, K = M.shape
    if C < 2 or K < 2:
        raise ValueError("Need at least 2 classes and 2 features.")

    Z_rows = linkage(pdist(M, metric=row_metric), method=method, optimal_ordering=True)
    Z_cols = linkage(pdist(M.T, metric=col_metric), method=method, optimal_ordering=True)

    row_order = dendrogram(Z_rows, no_plot=True)["leaves"]
    col_order = dendrogram(Z_cols, no_plot=True)["leaves"]

    M_ord = M[row_order][:, col_order]
    cls_ord = class_labels[row_order]
    feat_ord = feature_labels[col_order]

    row_ct = _cluster_threshold(Z_rows, n_row_clusters)
    col_ct = _cluster_threshold(Z_cols, n_col_clusters)

    if figsize is None:
        figsize = (max(12, min(24, 7.5 + 0.42 * K)), max(7, min(18, 4.6 + 0.62 * C)))

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=4,
        ncols=2,
        height_ratios=[2.8, 12.0, 2.2, 1.25],
        width_ratios=[13.0, 3.3],
        hspace=0.06,
        wspace=0.02,
    )

    ax_dend_col = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_dend_row = fig.add_subplot(gs[1, 1])
    ax_spacer = fig.add_subplot(gs[2, :])
    cax = fig.add_subplot(gs[3, :])

    fig.suptitle(title, y=0.985, fontsize=16)
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.11, right=0.98)

    set_link_color_palette([
        "#f28e2b", "#59a14f", "#e15759", "#9467bd",
        "#4e79a7", "#edc948", "#76b7b2", "#b07aa1",
    ])

    dendrogram(
        Z_cols,
        ax=ax_dend_col,
        color_threshold=col_ct,
        above_threshold_color="#4e79a7",
        no_labels=True,
    )
    ax_dend_col.set_xticks([])
    ax_dend_col.set_yticks([])
    for spine in ax_dend_col.spines.values():
        spine.set_visible(False)

    dendrogram(
        Z_rows,
        ax=ax_dend_row,
        orientation="right",
        color_threshold=row_ct,
        above_threshold_color="#4e79a7",
        no_labels=True,
    )
    ax_dend_row.invert_yaxis()
    ax_dend_row.set_xticks([])
    ax_dend_row.set_yticks([])
    for spine in ax_dend_row.spines.values():
        spine.set_visible(False)

    im = ax_heat.imshow(M_ord, aspect="auto", interpolation="nearest", cmap=cmap)

    ax_heat.set_yticks(np.arange(C))
    ax_heat.set_yticklabels(cls_ord)
    ax_heat.yaxis.tick_left()
    ax_heat.tick_params(axis="y", labelleft=True, labelright=False, labelsize=11, pad=6)

    ax_heat.set_xticks(np.arange(K))
    ax_heat.set_xticklabels(
        feat_ord,
        rotation=58,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax_heat.tick_params(axis="x", labelsize=9, pad=2)

    ax_heat.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax_heat.grid(which="minor", linestyle=":", linewidth=0.35, color=(1, 1, 1, 0.28))
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    ax_spacer.axis("off")

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label)
    cax.xaxis.labelpad = 6

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def make_contributor_signature_map(
    base_dir,
    class_labels,
    filename="linear_shap_full.csv",
    value_col="shap_importance",
    top_k_per_class=20,
    top_k_by_variance=20,
    title=None,
    cbar_label=None,
    cmap="viridis",
    row_metric="correlation",
    col_metric="correlation",
    method="average",
    n_row_clusters=4,
    n_col_clusters=5,
    figsize=(14, 9),
    save_path="contributors_signature_map.png",
):
    M_full, features, classes = contributor_matrix_from_csvs(
        base_dir=base_dir,
        class_labels=class_labels,
        filename=filename,
        value_col=value_col,
        top_k_per_class=top_k_per_class,
    )

    sel = select_top_features_from_matrix(M_full, top_k=top_k_by_variance, mode="var")
    M = M_full[:, sel]
    features = features[sel]

    if title is None:
        title = f"Contributor Signature Map (Top {top_k_by_variance} genes by variance)"
    if cbar_label is None:
        cbar_label = value_col

    plot_biclustered(
        M=M,
        feature_labels=features,
        class_labels=classes,
        title=title,
        cbar_label=cbar_label,
        cmap=cmap,
        row_metric=row_metric,
        col_metric=col_metric,
        method=method,
        n_row_clusters=n_row_clusters,
        n_col_clusters=n_col_clusters,
        figsize=figsize,
        save_path=save_path,
    )

    return M, features, classes


def save_contributor_signature_maps(
    base_dir,
    class_labels,
    contributor_methods=("score_weight", "linear_shap"),
    top_k_per_class=20,
    top_k_by_variance=20,
):
    base_dir = Path(base_dir)
    outputs = {}

    if len(class_labels) < 2:
        return outputs

    first_class_safe = _safe_name(class_labels[0])

    if "linear_shap" in contributor_methods:
        shap_csv = base_dir / "classes" / first_class_safe / "contributors" / "linear_shap_full.csv"
        if shap_csv.exists():
            make_contributor_signature_map(
                base_dir=base_dir,
                class_labels=class_labels,
                filename="linear_shap_full.csv",
                value_col="shap_importance",
                top_k_per_class=top_k_per_class,
                top_k_by_variance=top_k_by_variance,
                title="Contributor Signature Map (Linear SHAP)",
                cbar_label="mean|SHAP| within predicted class",
                cmap="viridis",
                save_path=base_dir / "contributors_shap.png",
            )
            outputs["linear_shap_signature_map"] = str(base_dir / "contributors_shap.png")

    if "score_weight" in contributor_methods:
        score_csv = base_dir / "classes" / first_class_safe / "contributors" / "score_weight_full.csv"
        if score_csv.exists():
            make_contributor_signature_map(
                base_dir=base_dir,
                class_labels=class_labels,
                filename="score_weight_full.csv",
                value_col="abs_contribution",
                top_k_per_class=top_k_per_class,
                top_k_by_variance=top_k_by_variance,
                title="Contributor Signature Map (Score Weight)",
                cbar_label="mean |score × weight|",
                cmap="viridis",
                save_path=base_dir / "contributors_score_weights.png",
            )
            outputs["score_weight_signature_map"] = str(base_dir / "contributors_score_weights.png")

    return outputs


class NUTMEGResults:
    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def from_runner(cls, runner, ds, name="dataset", save=False, out_dir=None, batch_size=1):
        bundle = collect_interpretability(
            runner,
            ds,
            name=name,
            out_dir=out_dir,
            save_bundle=save,
            batch_size=batch_size,
        )
        bundle["default_interpretability_dir"] = str(_runner_interpretability_dir(runner))
        bundle["runner_train_dir"] = str(runner.train_dir)
        return cls(bundle)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bundle, f)

    @property
    def classes(self):
        if "label_names_present" in self.bundle and self.bundle["label_names_present"] is not None:
            return list(self.bundle["label_names_present"])
        return list(self.bundle.get("label_names_all", []))

    @property
    def mode(self):
        return self.bundle["mode"]

    def resolve_classes(self, cell_type=None):
        if cell_type is None:
            return list(self.classes)
        return [cell_type]

    def attention_distribution(self, cell_type=None, mode="true_positive"):
        return attention_distribution_df(self.bundle, cell_type=cell_type, mode=mode)

    def attention_summary(self, cell_type=None, mode="true_positive"):
        df = self.attention_distribution(cell_type=cell_type, mode=mode)
        return summarize_attention(df)

    def top_contributors(
        self,
        cell_type=None,
        k=20,
        mode="true_positive",
        method="linear_shap",
        contrastive=True,
        background="all",
        background_mode="pred",
        balance_background=True,
    ):
        classes = self.resolve_classes(cell_type)
        out = {}
        for c in classes:
            if method == "score_weight":
                out[c] = top_genes_for_class(
                    self.bundle,
                    c,
                    k=k,
                    mode=mode,
                    contrastive=contrastive,
                )
            elif method == "linear_shap":
                out[c] = top_genes_for_class_linear_shap(
                    self.bundle,
                    c,
                    k=k,
                    mode=mode,
                    background=background,
                    background_mode=background_mode,
                    balance_background=balance_background,
                )
            else:
                raise ValueError("method must be one of: 'score_weight', 'linear_shap'")
        return out if cell_type is None else out[cell_type]

    def contribution_analysis(
        self,
        runner=None,
        cell_type=None,
        k=20,
        mode="true_positive",
        contributor_methods=("linear_shap",),
        contrastive=True,
        background="all",
        background_mode="pred",
        balance_background=True,
        top_k_plot=10,
        top_k_per_class_heatmap=20,
        top_k_by_variance=20,
        save_dir=None,
    ):
        classes = self.resolve_classes(cell_type)

        valid_methods = {"score_weight", "linear_shap"}
        unknown = set(contributor_methods) - valid_methods
        if unknown:
            raise ValueError(f"Unknown contributor methods: {sorted(unknown)}")

        if save_dir is None:
            save_dir = _default_interpretability_dir(self.bundle, runner=runner)

        outputs = {
            "root": str(save_dir),
            "classes": {},
            "summary_plots": {},
        }

        for c in classes:
            class_name_safe = _safe_name(c)
            class_dir = save_dir / "classes" / class_name_safe
            contrib_dir = class_dir / "contributors"

            class_outputs = {
                "class_name": str(c),
                "root": str(class_dir),
                "contributors": {},
            }

            for method in contributor_methods:
                contrib = self.top_contributors(
                    cell_type=c,
                    k=k,
                    mode=mode,
                    method=method,
                    contrastive=contrastive,
                    background=background,
                    background_mode=background_mode,
                    balance_background=balance_background,
                )

                class_outputs["contributors"][method] = save_contributor_result(
                    contrib,
                    out_dir=contrib_dir,
                    prefix=method,
                    class_name=str(c),
                    top_k=top_k_plot,
                )

            outputs["classes"][str(c)] = class_outputs

        outputs["summary_plots"] = save_contributor_signature_maps(
            base_dir=save_dir,
            class_labels=[str(c) for c in classes],
            contributor_methods=contributor_methods,
            top_k_per_class=top_k_per_class_heatmap,
            top_k_by_variance=top_k_by_variance,
        )

        return outputs

    def attention_analysis(
        self,
        runner=None,
        cell_type=None,
        mode="true_positive",
        save_dir=None,
    ):
        classes = self.resolve_classes(cell_type)

        if save_dir is None:
            save_dir = _default_interpretability_dir(self.bundle, runner=runner)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "root": str(save_dir),
            "classes": {},
        }

        for c in classes:
            class_name_safe = _safe_name(c)
            class_dir = save_dir / "classes" / class_name_safe
            attention_dir = class_dir / "attention"

            attn_df = self.attention_distribution(cell_type=c, mode=mode)

            class_outputs = {
                "class_name": str(c),
                "root": str(class_dir),
                "attention": save_attention_artifacts(
                    attn_df,
                    attention_dir,
                    str(c),
                ),
            }

            outputs["classes"][str(c)] = class_outputs

        return outputs

    def significant_grn(
        self,
        runner,
        ds,
        cell_type=None,
        mode="true_positive",
        top_k_per_cell=50,
        min_cells=3,
        path_score="geometric_mean",
        n_permutations=100,
        q_value_threshold=0.05,
        random_seed=0,
        forward_batch_size=64,
        permutation_batch_size=8,
    ):
        classes = self.resolve_classes(cell_type)
        all_graphs = runner._materialize_graphs_from_ds(ds)

        out = significant_grn_all_classes_batched(
            runner,
            ds,
            self.bundle,
            classes,
            mode=mode,
            top_k_per_cell=top_k_per_cell,
            min_cells=min_cells,
            path_score=path_score,
            n_permutations=n_permutations,
            q_value_threshold=q_value_threshold,
            random_seed=random_seed,
            forward_batch_size=forward_batch_size,
            permutation_batch_size=permutation_batch_size,
            all_graphs=all_graphs,
        )
        return out if cell_type is None else out[str(classes[0])]

    def reconstruct_grn(
        self,
        runner,
        ds,
        cell_type=None,
        mode="true_positive",
        top_k_per_cell=50,
        min_cells=3,
        path_score="geometric_mean",
        n_permutations=100,
        q_value_threshold=0.05,
        random_seed=0,
        forward_batch_size=64,
        permutation_batch_size=8,
        save_dir=None,
    ):
        classes = self.resolve_classes(cell_type)

        if save_dir is None:
            save_dir = _default_interpretability_dir(self.bundle, runner=runner)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        grn_all = self.significant_grn(
            runner,
            ds,
            cell_type=cell_type,
            mode=mode,
            top_k_per_cell=top_k_per_cell,
            min_cells=min_cells,
            path_score=path_score,
            n_permutations=n_permutations,
            q_value_threshold=q_value_threshold,
            random_seed=random_seed,
            forward_batch_size=forward_batch_size,
            permutation_batch_size=permutation_batch_size,
        )

        outputs = {
            "root": str(save_dir),
            "classes": {},
        }

        if cell_type is not None:
            grn_all = {str(classes[0]): grn_all}

        for c in classes:
            class_name_safe = _safe_name(c)
            class_dir = save_dir / "classes" / class_name_safe
            grn_dir = class_dir / "grn"

            grn = grn_all[str(c)]

            class_outputs = {
                "class_name": str(c),
                "root": str(class_dir),
                "grn": save_significant_grn_artifacts(grn, grn_dir),
                "grn_meta": {
                    "mode": grn["mode"],
                    "class_name": grn["class_name"],
                    "n_permutations": grn["n_permutations"],
                    "q_value_threshold": grn["q_value_threshold"],
                    "stat_col": grn["stat_col"],
                    "forward_batch_size": grn["forward_batch_size"],
                    "permutation_batch_size": grn["permutation_batch_size"],
                },
            }

            outputs["classes"][str(c)] = class_outputs

        return outputs

    def interpret(
        self,
        runner,
        ds,
        cell_type=None,
        mode="true_positive",
        top_k=10,
        top_k_per_cell=50,
        min_cells=3,
        path_score="geometric_mean",
        contributor_methods=("score_weight", "linear_shap"),
        contrastive=True,
        background="all",
        background_mode="pred",
        balance_background=True,
        n_permutations=100,
        q_value_threshold=0.05,
        random_seed=0,
        forward_batch_size=64,
        permutation_batch_size=8,
        save_dir=None,
    ):
        classes = self.resolve_classes(cell_type)

        if save_dir is None:
            save_dir = _default_interpretability_dir(self.bundle, runner=runner)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        run_settings = {
            "run_id": self.bundle["run_id"],
            "name": self.bundle["name"],
            "bundle_path": self.bundle.get("saved_bundle_path", None),
            "bundle_mode": self.bundle["mode"],
            "cell_type": cell_type,
            "classes": [str(c) for c in classes],
            "mode": mode,
            "top_k": int(top_k),
            "top_k_per_cell": int(top_k_per_cell) if top_k_per_cell is not None else None,
            "min_cells": int(min_cells),
            "path_score": path_score,
            "contributor_methods": list(contributor_methods),
            "contrastive": bool(contrastive),
            "background": background,
            "background_mode": background_mode,
            "balance_background": bool(balance_background),
            "n_permutations": int(n_permutations),
            "q_value_threshold": float(q_value_threshold),
            "random_seed": int(random_seed),
            "forward_batch_size": int(forward_batch_size),
            "permutation_batch_size": int(permutation_batch_size),
        }
        _write_json(run_settings, save_dir / "interpret_params.json")

        contrib_out = self.contribution_analysis(
            runner=runner,
            cell_type=cell_type,
            k=max(20, 2 * top_k),
            mode=mode,
            contributor_methods=contributor_methods,
            contrastive=contrastive,
            background=background,
            background_mode=background_mode,
            balance_background=balance_background,
            top_k_plot=top_k,
            top_k_per_class_heatmap=max(20, 2 * top_k),
            top_k_by_variance=max(20, 2 * top_k),
            save_dir=save_dir,
        )

        attn_out = self.attention_analysis(
            runner=runner,
            cell_type=cell_type,
            mode=mode,
            save_dir=save_dir,
        )

        grn_out = self.reconstruct_grn(
            runner,
            ds,
            cell_type=cell_type,
            mode=mode,
            top_k_per_cell=top_k_per_cell,
            min_cells=min_cells,
            path_score=path_score,
            n_permutations=n_permutations,
            q_value_threshold=q_value_threshold,
            random_seed=random_seed,
            forward_batch_size=forward_batch_size,
            permutation_batch_size=permutation_batch_size,
            save_dir=save_dir,
        )

        outputs = {
            "root": str(save_dir),
            "interpret_params_json": str(save_dir / "interpret_params.json"),
            "contributors": contrib_out,
            "attention": attn_out,
            "grn": grn_out,
        }

        return outputs