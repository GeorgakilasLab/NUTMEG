import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from nutmeg.trainer import ListDataset
from nutmeg.utils import (
    _batch_ptr,
    _default_bundle_dir,
    _get_cre_local_to_global,
    _get_gene_local_to_global,
    _label_names_all_from_runner,
    _mean_alpha,
    _resolve_label_name,
    _safe_cre_names,
    _safe_gene_names,
    _sample_id_from_ds,
    _slice_pack_value,
    _stack_or_none,
    _to_numpy,
)


@torch.no_grad()
def collect_interpretability(
    runner,
    ds,
    *,
    name="dataset",
    out_dir=None,
    save_bundle=False,
    batch_size=1,
):
    if batch_size != 1:
        raise ValueError("Use batch_size=1 for clean per-cell attention extraction.")

    model = runner.model
    device = runner.device
    was_training = model.training
    old_return_intermediates = getattr(model, "return_intermediates", False)

    model.eval()
    if hasattr(model, "return_intermediates"):
        model.return_intermediates = True

    graphs = runner._materialize_graphs_from_ds(ds)
    loader = DataLoader(
        ListDataset(graphs),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    gene_names = _safe_gene_names(runner.dataset)
    cre_names = _safe_cre_names(runner.dataset)
    label_names_all = _label_names_all_from_runner(runner, runner.dataset)
    class_ids_all = list(range(int(model.C)))

    results = {
        "run_id": runner.run_id,
        "name": name,
        "mode": getattr(model, "mode", None),
        "gene_names": gene_names,
        "cre_names": cre_names,
        "label_names_all": label_names_all,
        "class_ids_all": class_ids_all,
        "sample_ids": [],
        "y_true": [],
        "y_pred": [],
        "logits": [],
        "probs": [],
        "scores": [],
        "aligned": [],
        "presence": [],
        "aligned_in": [],
        "cells": [],
        "params": {},
    }

    if hasattr(model, "classifier"):
        results["params"]["classifier_weight"] = _to_numpy(model.classifier.weight)
        results["params"]["classifier_bias"] = (
            _to_numpy(model.classifier.bias) if model.classifier.bias is not None else None
        )

    try:
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch)

            if isinstance(out, (tuple, list)) and len(out) >= 4:
                logits, scores, attn, pack = out[0], out[1], out[2], out[3]
            elif isinstance(out, (tuple, list)) and len(out) == 3:
                logits, scores, attn = out
                pack = {}
            else:
                raise ValueError(
                    "Expected model output (logits, scores, attn) or (logits, scores, attn, pack)."
                )

            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            y_true = batch.y.view(-1).long() if hasattr(batch, "y") and batch.y is not None else None

            results["sample_ids"].append(_sample_id_from_ds(ds, i))
            results["y_pred"].append(int(pred.item()))
            results["logits"].append(_to_numpy(logits[0]))
            results["probs"].append(_to_numpy(probs[0]))
            results["scores"].append(_to_numpy(scores[0]))
            results["y_true"].append(int(y_true.item()) if y_true is not None else None)

            if isinstance(pack, dict):
                a = pack.get("aligned", None)
                p = pack.get("presence", None)
                ai = pack.get("aligned_in", None)
                results["aligned"].append(_to_numpy(a[0]) if a is not None else None)
                results["presence"].append(_to_numpy(p[0]) if p is not None else None)
                results["aligned_in"].append(_to_numpy(ai[0]) if ai is not None else None)
            else:
                results["aligned"].append(None)
                results["presence"].append(None)
                results["aligned_in"].append(None)

            gene_l2g = _get_gene_local_to_global(batch)
            cre_l2g = _get_cre_local_to_global(batch)

            yt = results["y_true"][-1]
            yp = results["y_pred"][-1]

            cell_rec = {
                "cell_index": i,
                "sample_id": results["sample_ids"][-1],
                "y_true": yt,
                "y_pred": yp,
                "y_true_name": _resolve_label_name(results, yt),
                "y_pred_name": _resolve_label_name(results, yp),
                "logits": results["logits"][-1],
                "probs": results["probs"][-1],
                "scores": results["scores"][-1],
                "gene_l2g": gene_l2g,
                "cre_l2g": cre_l2g,
            }

            if results["presence"][-1] is not None:
                cell_rec["presence"] = results["presence"][-1]
            if results["aligned"][-1] is not None:
                cell_rec["aligned"] = results["aligned"][-1]
            if results["aligned_in"][-1] is not None:
                cell_rec["aligned_in"] = results["aligned_in"][-1]

            if attn is not None:
                if getattr(model, "mode", None) == "rna_only":
                    ei = attn["t2g"]["edge_index"]
                    alpha = _mean_alpha(attn["t2g"]["alpha"])

                    src_local = _to_numpy(ei[0]).astype(int)
                    dst_local = _to_numpy(ei[1]).astype(int)
                    src_global = gene_l2g[src_local]
                    dst_global = gene_l2g[dst_local]

                    cell_rec["attn"] = {
                        "t2g": {
                            "src_local": src_local,
                            "dst_local": dst_local,
                            "src_global": src_global,
                            "dst_global": dst_global,
                            "src_name": [gene_names[j] for j in src_global] if len(gene_names) else src_global.tolist(),
                            "dst_name": [gene_names[j] for j in dst_global] if len(gene_names) else dst_global.tolist(),
                            "alpha": alpha,
                        }
                    }
                else:
                    ei_g2c = attn["g2c"]["edge_index"]
                    ei_c2g = attn["c2g"]["edge_index"]
                    alpha_g2c = _mean_alpha(attn["g2c"]["alpha"])
                    alpha_c2g = _mean_alpha(attn["c2g"]["alpha"])

                    g_src_local = _to_numpy(ei_g2c[0]).astype(int)
                    c_dst_local = _to_numpy(ei_g2c[1]).astype(int)
                    c_src_local = _to_numpy(ei_c2g[0]).astype(int)
                    g_dst_local = _to_numpy(ei_c2g[1]).astype(int)

                    g_src_global = gene_l2g[g_src_local]
                    g_dst_global = gene_l2g[g_dst_local]
                    c_dst_global = cre_l2g[c_dst_local] if cre_l2g is not None else c_dst_local
                    c_src_global = cre_l2g[c_src_local] if cre_l2g is not None else c_src_local

                    cell_rec["attn"] = {
                        "g2c": {
                            "src_gene_local": g_src_local,
                            "dst_cre_local": c_dst_local,
                            "src_gene_global": g_src_global,
                            "dst_cre_global": c_dst_global,
                            "src_gene_name": [gene_names[j] for j in g_src_global] if len(gene_names) else g_src_global.tolist(),
                            "dst_cre_name": [cre_names[j] for j in c_dst_global] if cre_names is not None else c_dst_global.tolist(),
                            "alpha": alpha_g2c,
                        },
                        "c2g": {
                            "src_cre_local": c_src_local,
                            "dst_gene_local": g_dst_local,
                            "src_cre_global": c_src_global,
                            "dst_gene_global": g_dst_global,
                            "src_cre_name": [cre_names[j] for j in c_src_global] if cre_names is not None else c_src_global.tolist(),
                            "dst_gene_name": [gene_names[j] for j in g_dst_global] if len(gene_names) else g_dst_global.tolist(),
                            "alpha": alpha_c2g,
                        },
                    }

            results["cells"].append(cell_rec)

        results["y_pred"] = np.asarray(results["y_pred"], dtype=int)
        results["y_true"] = None if any(v is None for v in results["y_true"]) else np.asarray(results["y_true"], dtype=int)
        results["logits"] = np.stack(results["logits"], axis=0)
        results["probs"] = np.stack(results["probs"], axis=0)
        results["scores"] = np.stack(results["scores"], axis=0)
        results["aligned"] = _stack_or_none(results["aligned"])
        results["presence"] = _stack_or_none(results["presence"])
        results["aligned_in"] = _stack_or_none(results["aligned_in"])

        if results["y_true"] is not None:
            labels_present = sorted(set(results["y_true"].tolist()) | set(results["y_pred"].tolist()))
        else:
            labels_present = sorted(set(results["y_pred"].tolist()))

        results["labels_present"] = labels_present
        results["label_names_present"] = [results["label_names_all"][i] for i in labels_present]
        results["metrics"] = {
            "n_cells": int(len(ds)),
            "accuracy": None if results["y_true"] is None else float((results["y_true"] == results["y_pred"]).mean()),
        }

        if save_bundle:
            if out_dir is None:
                out_dir = _default_bundle_dir(runner, name)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            bundle_path = out_dir / f"{name}_bundle.pkl"
            with open(bundle_path, "wb") as f:
                pickle.dump(results, f)

            results["saved_bundle_path"] = str(bundle_path)
            results["saved_bundle_dir"] = str(out_dir)

        return results

    finally:
        if was_training:
            model.train()
        if hasattr(model, "return_intermediates"):
            model.return_intermediates = old_return_intermediates


@torch.no_grad()
def _forward_graphs_to_cells(
    runner,
    ds,
    graphs,
    cell_indices,
    sample_ids,
    forward_batch_size=64,
):
    model = runner.model
    device = runner.device
    was_training = model.training
    old_return_intermediates = getattr(model, "return_intermediates", False)

    model.eval()
    if hasattr(model, "return_intermediates"):
        model.return_intermediates = True

    gene_names = _safe_gene_names(runner.dataset)
    cre_names = _safe_cre_names(runner.dataset)
    label_names_all = _label_names_all_from_runner(runner, runner.dataset)
    tmp_results = {"label_names_all": label_names_all}

    loader = DataLoader(
        ListDataset(graphs),
        batch_size=int(forward_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    cells = []
    offset = 0
    model_mode = getattr(model, "mode", None)

    try:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            if isinstance(out, (tuple, list)) and len(out) >= 4:
                logits, scores, attn, pack = out[0], out[1], out[2], out[3]
            elif isinstance(out, (tuple, list)) and len(out) == 3:
                logits, scores, attn = out
                pack = {}
            else:
                raise ValueError(
                    "Expected model output (logits, scores, attn) or (logits, scores, attn, pack)."
                )

            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            y_true = batch.y.view(-1).long() if hasattr(batch, "y") and batch.y is not None else None
            B = int(logits.shape[0])

            gene_l2g_all = _get_gene_local_to_global(batch)
            gene_ptr = _batch_ptr(batch["gene"])
            gene_batch = (
                _to_numpy(batch["gene"].batch).astype(int)
                if hasattr(batch["gene"], "batch") and batch["gene"].batch is not None
                else np.repeat(np.arange(len(gene_ptr) - 1), np.diff(gene_ptr))
            )

            cre_l2g_all = _get_cre_local_to_global(batch)
            if cre_l2g_all is not None:
                cre_ptr = _batch_ptr(batch["cre"])
                cre_batch = (
                    _to_numpy(batch["cre"].batch).astype(int)
                    if hasattr(batch["cre"], "batch") and batch["cre"].batch is not None
                    else np.repeat(np.arange(len(cre_ptr) - 1), np.diff(cre_ptr))
                )
            else:
                cre_ptr = None
                cre_batch = None

            if attn is not None and model_mode == "rna_only":
                ei_t2g = _to_numpy(attn["t2g"]["edge_index"]).astype(int)
                alpha_t2g = _mean_alpha(attn["t2g"]["alpha"])
                edge_graph_t2g = gene_batch[ei_t2g[0]] if ei_t2g.size else np.array([], dtype=int)

            elif attn is not None and model_mode != "rna_only":
                ei_g2c = _to_numpy(attn["g2c"]["edge_index"]).astype(int)
                ei_c2g = _to_numpy(attn["c2g"]["edge_index"]).astype(int)
                alpha_g2c = _mean_alpha(attn["g2c"]["alpha"])
                alpha_c2g = _mean_alpha(attn["c2g"]["alpha"])
                edge_graph_g2c = gene_batch[ei_g2c[0]] if ei_g2c.size else np.array([], dtype=int)
                edge_graph_c2g = cre_batch[ei_c2g[0]] if ei_c2g.size else np.array([], dtype=int)

            for bi in range(B):
                j = offset + bi

                gene_l2g = gene_l2g_all[gene_ptr[bi]:gene_ptr[bi + 1]]
                cre_l2g = None if cre_l2g_all is None else cre_l2g_all[cre_ptr[bi]:cre_ptr[bi + 1]]

                yt = int(y_true[bi].item()) if y_true is not None else None
                yp = int(pred[bi].item())

                cell_rec = {
                    "cell_index": int(cell_indices[j]),
                    "sample_id": sample_ids[j],
                    "y_true": yt,
                    "y_pred": yp,
                    "y_true_name": _resolve_label_name(tmp_results, yt),
                    "y_pred_name": _resolve_label_name(tmp_results, yp),
                    "logits": _to_numpy(logits[bi]),
                    "probs": _to_numpy(probs[bi]),
                    "scores": _to_numpy(scores[bi]),
                    "gene_l2g": gene_l2g,
                    "cre_l2g": cre_l2g,
                }

                if isinstance(pack, dict):
                    a = _slice_pack_value(pack.get("aligned", None), bi, B)
                    p = _slice_pack_value(pack.get("presence", None), bi, B)
                    ai = _slice_pack_value(pack.get("aligned_in", None), bi, B)
                    if a is not None:
                        cell_rec["aligned"] = a
                    if p is not None:
                        cell_rec["presence"] = p
                    if ai is not None:
                        cell_rec["aligned_in"] = ai

                if attn is not None:
                    if model_mode == "rna_only":
                        m = edge_graph_t2g == bi
                        src_local = ei_t2g[0, m] - gene_ptr[bi]
                        dst_local = ei_t2g[1, m] - gene_ptr[bi]
                        alpha_local = np.asarray(alpha_t2g)[m]

                        src_global = gene_l2g[src_local]
                        dst_global = gene_l2g[dst_local]

                        cell_rec["attn"] = {
                            "t2g": {
                                "src_local": src_local,
                                "dst_local": dst_local,
                                "src_global": src_global,
                                "dst_global": dst_global,
                                "src_name": [gene_names[k] for k in src_global] if len(gene_names) else src_global.tolist(),
                                "dst_name": [gene_names[k] for k in dst_global] if len(gene_names) else dst_global.tolist(),
                                "alpha": alpha_local,
                            }
                        }
                    else:
                        mg2c = edge_graph_g2c == bi
                        mc2g = edge_graph_c2g == bi

                        g_src_local = ei_g2c[0, mg2c] - gene_ptr[bi]
                        c_dst_local = ei_g2c[1, mg2c] - cre_ptr[bi]
                        c_src_local = ei_c2g[0, mc2g] - cre_ptr[bi]
                        g_dst_local = ei_c2g[1, mc2g] - gene_ptr[bi]

                        g_src_global = gene_l2g[g_src_local]
                        g_dst_global = gene_l2g[g_dst_local]
                        c_dst_global = cre_l2g[c_dst_local] if cre_l2g is not None else c_dst_local
                        c_src_global = cre_l2g[c_src_local] if cre_l2g is not None else c_src_local

                        cell_rec["attn"] = {
                            "g2c": {
                                "src_gene_local": g_src_local,
                                "dst_cre_local": c_dst_local,
                                "src_gene_global": g_src_global,
                                "dst_cre_global": c_dst_global,
                                "src_gene_name": [gene_names[k] for k in g_src_global] if len(gene_names) else g_src_global.tolist(),
                                "dst_cre_name": [cre_names[k] for k in c_dst_global] if cre_names is not None else c_dst_global.tolist(),
                                "alpha": np.asarray(alpha_g2c)[mg2c],
                            },
                            "c2g": {
                                "src_cre_local": c_src_local,
                                "dst_gene_local": g_dst_local,
                                "src_cre_global": c_src_global,
                                "dst_gene_global": g_dst_global,
                                "src_cre_name": [cre_names[k] for k in c_src_global] if cre_names is not None else c_src_global.tolist(),
                                "dst_gene_name": [gene_names[k] for k in g_dst_global] if len(gene_names) else g_dst_global.tolist(),
                                "alpha": np.asarray(alpha_c2g)[mc2g],
                            },
                        }

                cells.append(cell_rec)

            offset += B

        return cells

    finally:
        if was_training:
            model.train()
        if hasattr(model, "return_intermediates"):
            model.return_intermediates = old_return_intermediates