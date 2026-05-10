import numpy as np

from .utils import (
    _background_mask,
    _balanced_background_mean,
    _class_index,
    _contribution_tables,
    _labels_for_mode,
    class_mask,
)


def class_mean_contrib(results, c, mode="true_positive", contrastive=False):
    scores = np.asarray(results["scores"])
    W = np.asarray(results["params"]["classifier_weight"])

    c_idx, names = _class_index(results, c)
    mask = class_mask(results, c, mode=mode)

    if mask.sum() == 0:
        raise ValueError(f"No cells found for class '{names[c_idx]}' with mode='{mode}'")

    if contrastive:
        other = np.delete(np.arange(W.shape[0]), c_idx)
        w = W[c_idx] - W[other].mean(axis=0)
    else:
        w = W[c_idx]

    contrib = scores[mask] * w[None, :]
    mean_signed = contrib.mean(axis=0)
    mean_abs = np.abs(contrib).mean(axis=0)
    return mean_signed, mean_abs, contrib


def top_genes_for_class(results, c, k=20, mode="true_positive", contrastive=False):
    mean_signed, mean_abs, per_cell = class_mean_contrib(
        results,
        c,
        mode=mode,
        contrastive=contrastive,
    )

    out = _contribution_tables(
        mean_signed,
        results["gene_names"],
        k=k,
        value_col="contribution",
        abs_values=mean_abs,
        abs_col="abs_contribution",
    )
    out["per_cell_values"] = per_cell
    out["method"] = "score_weight"
    out["contrastive"] = bool(contrastive)
    return out


def class_mean_shap_linear(
    results,
    c,
    mode="true_positive",
    background="all",
    background_mode="pred",
    balance_background=True,
):
    scores = np.asarray(results["scores"])
    W = np.asarray(results["params"]["classifier_weight"])

    c_idx, names = _class_index(results, c)
    target_mask = class_mask(results, c, mode=mode)

    if target_mask.sum() == 0:
        raise ValueError(f"No cells found for class '{names[c_idx]}' with mode='{mode}'")

    bg_mask = _background_mask(
        results,
        background=background,
        class_name=c,
        mode=background_mode,
    )
    if bg_mask.sum() == 0:
        raise ValueError("Background mask selected zero cells.")

    if balance_background:
        mu, bg_class_counts = _balanced_background_mean(
            results,
            bg_mask,
            label_mode=background_mode,
        )
    else:
        mu = scores[bg_mask].mean(axis=0)
        bg_labels = _labels_for_mode(results, mode=background_mode)[bg_mask]
        uniq, cnt = np.unique(bg_labels, return_counts=True)
        bg_class_counts = {int(k): int(v) for k, v in zip(uniq, cnt)}

    shap_vals = (scores[target_mask] - mu[None, :]) * W[c_idx][None, :]
    mean_signed = shap_vals.mean(axis=0)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return mean_signed, mean_abs, shap_vals, mu, bg_class_counts


def top_genes_for_class_linear_shap(
    results,
    c,
    k=20,
    mode="true_positive",
    background="all",
    background_mode="pred",
    balance_background=True,
):
    mean_signed, mean_abs, per_cell, bg_mean, bg_class_counts = class_mean_shap_linear(
        results,
        c,
        mode=mode,
        background=background,
        background_mode=background_mode,
        balance_background=balance_background,
    )

    out = _contribution_tables(
        mean_signed,
        results["gene_names"],
        k=k,
        value_col="shap_value",
        abs_values=mean_abs,
        abs_col="shap_importance",
    )
    out["per_cell_values"] = per_cell
    out["background_mean_scores"] = bg_mean
    out["background_class_counts"] = bg_class_counts
    out["method"] = "linear_shap"
    out["background"] = background
    out["background_mode"] = background_mode
    out["balance_background"] = bool(balance_background)
    return out