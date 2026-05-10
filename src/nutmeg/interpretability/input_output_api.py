import pickle
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    _default_analysis_dir,
    _safe_name,
    _write_json,
)


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
            save_dir = _default_analysis_dir(self.bundle, runner=runner)
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

        outputs = {
            "root": str(save_dir),
            "interpret_params_json": str(save_dir / "interpret_params.json"),
            "classes": {},
        }

        all_graphs = runner._materialize_graphs_from_ds(ds)

        grn_all = significant_grn_all_classes_batched(
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

        for c in classes:
            class_name_safe = _safe_name(c)
            class_dir = save_dir / "classes" / class_name_safe
            contrib_dir = class_dir / "contributors"
            attention_dir = class_dir / "attention"
            grn_dir = class_dir / "grn"

            class_outputs = {
                "class_name": c,
                "root": str(class_dir),
                "contributors": {},
            }

            for method in contributor_methods:
                if method == "score_weight":
                    contrib = top_genes_for_class(
                        self.bundle,
                        c,
                        k=max(20, 2 * top_k),
                        mode=mode,
                        contrastive=contrastive,
                    )
                elif method == "linear_shap":
                    contrib = top_genes_for_class_linear_shap(
                        self.bundle,
                        c,
                        k=max(20, 2 * top_k),
                        mode=mode,
                        background=background,
                        background_mode=background_mode,
                        balance_background=balance_background,
                    )
                else:
                    raise ValueError(f"Unknown contributor method: {method}")

                class_outputs["contributors"][method] = save_contributor_result(
                    contrib,
                    contrib_dir,
                    prefix=method,
                    class_name=str(c),
                    top_k=top_k,
                )

            attn_df = self.attention_distribution(cell_type=c, mode=mode)
            class_outputs["attention"] = save_attention_artifacts(
                attn_df,
                attention_dir,
                str(c),
            )

            grn = grn_all[str(c)]
            class_outputs["grn"] = save_significant_grn_artifacts(grn, grn_dir)
            class_outputs["grn_meta"] = {
                "mode": grn["mode"],
                "class_name": grn["class_name"],
                "n_permutations": grn["n_permutations"],
                "q_value_threshold": grn["q_value_threshold"],
                "stat_col": grn["stat_col"],
                "forward_batch_size": grn["forward_batch_size"],
                "permutation_batch_size": grn["permutation_batch_size"],
            }

            outputs["classes"][str(c)] = class_outputs

        return outputs