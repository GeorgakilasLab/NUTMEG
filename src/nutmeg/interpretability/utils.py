from pathlib import Path
import copy
import json
import re

import numpy as np
import pandas as pd
import torch


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mean_alpha(alpha):
    a = _to_numpy(alpha)
    if a is None:
        return None
    if a.ndim == 2:
        return a.mean(axis=-1)
    return a


def _safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = x.strip("_")
    return x or "item"


def _write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _stack_or_none(xs):
    if len(xs) == 0:
        return None
    if any(x is None for x in xs):
        return None
    return np.stack(xs, axis=0)


def _experiment_root_from_runner(runner):
    return Path(runner.plots_dir).resolve().parent.parent


def _default_bundle_dir(runner, name):
    return _experiment_root_from_runner(runner) / "interpretability" / runner.run_id / name


def _default_analysis_dir(bundle, runner=None):
    saved = bundle.get("saved_bundle_path", None)
    if saved is not None:
        return Path(saved).parent
    if runner is not None:
        return _default_bundle_dir(runner, bundle["name"])
    return Path("interpretability") / bundle["run_id"] / bundle["name"]


def _safe_gene_names(ds):
    return list(getattr(ds, "gene_node_names", []))


def _safe_cre_names(ds):
    x = getattr(ds, "cre_node_names", None)
    return list(x) if x is not None else None


def _sample_id_from_ds(ds, i):
    if hasattr(ds, "valid_cells") and ds.valid_cells is not None and i < len(ds.valid_cells):
        return ds.valid_cells[i]
    return i


def _label_names_all_from_runner(runner, ds):
    C = int(runner.model.C)
    if hasattr(ds, "id_to_label"):
        try:
            return [str(ds.id_to_label[i]) for i in range(C)]
        except Exception:
            pass
    return [str(i) for i in range(C)]


def _get_gene_local_to_global(batch):
    gene_obj = batch["gene"]
    if hasattr(gene_obj, "kept_idx") and gene_obj.kept_idx is not None:
        return _to_numpy(gene_obj.kept_idx).astype(int)
    if hasattr(gene_obj, "keptidx") and gene_obj.keptidx is not None:
        return _to_numpy(gene_obj.keptidx).astype(int)
    if hasattr(gene_obj, "gene_feats") and gene_obj.gene_feats is not None:
        return np.arange(int(gene_obj.gene_feats.shape[0]), dtype=int)
    if hasattr(gene_obj, "x") and gene_obj.x is not None:
        return np.arange(int(gene_obj.x.shape[0]), dtype=int)
    return np.array([], dtype=int)


def _get_cre_local_to_global(batch):
    try:
        cre_obj = batch["cre"]
    except Exception:
        return None

    if hasattr(cre_obj, "kept_idx") and cre_obj.kept_idx is not None:
        return _to_numpy(cre_obj.kept_idx).astype(int)
    if hasattr(cre_obj, "keptidx") and cre_obj.keptidx is not None:
        return _to_numpy(cre_obj.keptidx).astype(int)
    if hasattr(cre_obj, "x") and cre_obj.x is not None:
        return np.arange(int(cre_obj.x.shape[0]), dtype=int)
    return None


def _get_mode(results):
    return str(results.get("mode", "")).lower()


def _class_index(results, c):
    names = results.get("label_names_all", None)
    if names is None:
        C = int(np.asarray(results["params"]["classifier_weight"]).shape[0])
        names = [str(i) for i in range(C)]

    if isinstance(c, str):
        if c not in names:
            raise ValueError(f"Unknown class name: {c}")
        return names.index(c), names

    idx = int(c)
    if idx < 0 or idx >= len(names):
        raise ValueError(f"class index out of range: {idx}")
    return idx, names


def _resolve_label_name(results, idx):
    if idx is None:
        return None
    names = results.get("label_names_all", None)
    if names is None:
        return str(idx)
    idx = int(idx)
    if 0 <= idx < len(names):
        return names[idx]
    return str(idx)


def class_mask(results, c, mode="true_positive"):
    c_idx, _ = _class_index(results, c)
    y_pred = np.asarray(results["y_pred"])

    if mode == "pred":
        return y_pred == c_idx

    y_true = results.get("y_true", None)
    if y_true is None:
        raise ValueError("results['y_true'] is None; use mode='pred' for unlabeled datasets.")
    y_true = np.asarray(y_true)

    if mode == "true":
        return y_true == c_idx
    if mode == "true_positive":
        return (y_true == c_idx) & (y_pred == c_idx)

    raise ValueError("mode must be one of: 'pred', 'true', 'true_positive'")


def _background_mask(results, background="all", class_name=None, mode="pred"):
    y_pred = np.asarray(results["y_pred"])

    if background == "all":
        return np.ones(len(y_pred), dtype=bool)
    if background == "class":
        return class_mask(results, class_name, mode=mode)
    if background == "not_class":
        return ~class_mask(results, class_name, mode=mode)

    raise ValueError("background must be one of: 'all', 'class', 'not_class'")


def _labels_for_mode(results, mode="pred"):
    if mode == "pred":
        return np.asarray(results["y_pred"])

    y_true = results.get("y_true", None)
    if y_true is None:
        raise ValueError(f"results['y_true'] is required when mode='{mode}'")
    return np.asarray(y_true)


def _balanced_background_mean(results, bg_mask, label_mode="pred"):
    scores = np.asarray(results["scores"])
    labels = _labels_for_mode(results, mode=label_mode)

    X = scores[bg_mask]
    y = labels[bg_mask]

    if len(X) == 0:
        raise ValueError("Balanced background selected zero cells.")

    classes = np.unique(y)
    if len(classes) == 0:
        raise ValueError("Balanced background has zero classes.")

    class_means = []
    class_counts = {}
    for cls in classes:
        cls_mask = y == cls
        cls_X = X[cls_mask]
        if len(cls_X) == 0:
            continue
        class_means.append(cls_X.mean(axis=0))
        class_counts[int(cls)] = int(cls_mask.sum())

    if len(class_means) == 0:
        raise ValueError("Balanced background has no non-empty classes.")

    mu = np.stack(class_means, axis=0).mean(axis=0)
    return mu, class_counts


def _contribution_tables(values, gene_names, k=20, value_col="value", abs_values=None, abs_col="abs_value"):
    values = np.asarray(values)
    genes = np.asarray(gene_names)

    if abs_values is None:
        abs_values = np.abs(values)
    abs_values = np.asarray(abs_values)

    pos = np.argsort(-values)[:k]
    neg = np.argsort(values)[:k]
    imp = np.argsort(-abs_values)[:k]

    full = pd.DataFrame({
        "gene": genes,
        value_col: values,
        abs_col: abs_values,
    }).sort_values(abs_col, ascending=False).reset_index(drop=True)

    return {
        "positive": pd.DataFrame({"gene": genes[pos], value_col: values[pos]}),
        "negative": pd.DataFrame({"gene": genes[neg], value_col: values[neg]}),
        "importance": pd.DataFrame({"gene": genes[imp], abs_col: abs_values[imp]}),
        "full": full,
        "values": values,
        "abs_values": abs_values,
        "gene_names": genes,
    }


def _clone_graph(graph):
    return copy.deepcopy(graph)


def _graph_local_to_global(graph, node_type):
    obj = graph[node_type]
    if hasattr(obj, "kept_idx") and obj.kept_idx is not None:
        return _to_numpy(obj.kept_idx).astype(int)
    if hasattr(obj, "keptidx") and obj.keptidx is not None:
        return _to_numpy(obj.keptidx).astype(int)
    if hasattr(obj, "gene_feats") and obj.gene_feats is not None:
        return np.arange(int(obj.gene_feats.shape[0]), dtype=int)
    if hasattr(obj, "x") and obj.x is not None:
        return np.arange(int(obj.x.shape[0]), dtype=int)
    return np.array([], dtype=int)


def _edge_store(graph, relation_name):
    relation_map = {
        "t2g": ("gene", "regulates", "gene"),
        "g2c": ("gene", "binds", "cre"),
        "c2g": ("cre", "regulates", "gene"),
    }

    if relation_name not in relation_map:
        raise ValueError(f"Unknown relation_name: {relation_name}")

    edge_type = relation_map[relation_name]
    store = graph[edge_type]

    if not hasattr(store, "edge_index"):
        raise TypeError(
            f"Resolved {relation_name} -> {edge_type}, "
            f"but graph[{edge_type}] is not an edge store."
        )

    return store


def _edge_global_pairs_from_graph(graph, relation_name):
    store = _edge_store(graph, relation_name)
    ei = _to_numpy(store.edge_index).astype(int)
    if ei.size == 0:
        return np.empty((0, 2), dtype=int)

    if relation_name == "t2g":
        gene_l2g = _graph_local_to_global(graph, "gene")
        src_global = gene_l2g[ei[0]]
        dst_global = gene_l2g[ei[1]]
    elif relation_name == "g2c":
        gene_l2g = _graph_local_to_global(graph, "gene")
        cre_l2g = _graph_local_to_global(graph, "cre")
        src_global = gene_l2g[ei[0]]
        dst_global = cre_l2g[ei[1]]
    elif relation_name == "c2g":
        cre_l2g = _graph_local_to_global(graph, "cre")
        gene_l2g = _graph_local_to_global(graph, "gene")
        src_global = cre_l2g[ei[0]]
        dst_global = gene_l2g[ei[1]]
    else:
        raise ValueError(f"Unknown relation_name: {relation_name}")

    return np.stack([src_global, dst_global], axis=1).astype(int)


def _collect_class_edge_bank(graphs, relation_name):
    rows = []
    for g in graphs:
        pairs = _edge_global_pairs_from_graph(g, relation_name)
        if len(pairs):
            rows.append(pairs)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=int)
    return np.concatenate(rows, axis=0)


def _build_class_edge_banks(class_graphs, mode):
    if mode == "rna_only":
        relations = ["t2g"]
    elif mode == "multiome":
        relations = ["g2c", "c2g"]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {rel: _collect_class_edge_bank(class_graphs, rel) for rel in relations}


def _valid_bank_for_graph(bank_pairs, graph, relation_name):
    if len(bank_pairs) == 0:
        return bank_pairs

    if relation_name == "t2g":
        src_nodes = set(_graph_local_to_global(graph, "gene").tolist())
        dst_nodes = src_nodes
    elif relation_name == "g2c":
        src_nodes = set(_graph_local_to_global(graph, "gene").tolist())
        dst_nodes = set(_graph_local_to_global(graph, "cre").tolist())
    elif relation_name == "c2g":
        src_nodes = set(_graph_local_to_global(graph, "cre").tolist())
        dst_nodes = set(_graph_local_to_global(graph, "gene").tolist())
    else:
        raise ValueError(f"Unknown relation_name: {relation_name}")

    keep = np.array([(s in src_nodes and d in dst_nodes) for s, d in bank_pairs], dtype=bool)
    return bank_pairs[keep]


def _global_to_local_map(global_ids):
    return {int(g): int(i) for i, g in enumerate(np.asarray(global_ids).astype(int))}


def _sample_local_edge_index_from_bank(graph, relation_name, bank_pairs, n_edges, rng):
    valid = _valid_bank_for_graph(bank_pairs, graph, relation_name)
    if len(valid) == 0:
        return None

    replace = len(valid) < n_edges
    take = rng.choice(len(valid), size=n_edges, replace=replace)
    sampled = valid[take]

    if relation_name == "t2g":
        src_map = _global_to_local_map(_graph_local_to_global(graph, "gene"))
        dst_map = src_map
    elif relation_name == "g2c":
        src_map = _global_to_local_map(_graph_local_to_global(graph, "gene"))
        dst_map = _global_to_local_map(_graph_local_to_global(graph, "cre"))
    elif relation_name == "c2g":
        src_map = _global_to_local_map(_graph_local_to_global(graph, "cre"))
        dst_map = _global_to_local_map(_graph_local_to_global(graph, "gene"))
    else:
        raise ValueError(f"Unknown relation_name: {relation_name}")

    src_local = np.array([src_map[int(s)] for s in sampled[:, 0]], dtype=np.int64)
    dst_local = np.array([dst_map[int(d)] for d in sampled[:, 1]], dtype=np.int64)
    return np.stack([src_local, dst_local], axis=0)


def _permute_edge_aligned_attrs(store, n_edges, rng):
    perm = rng.permutation(n_edges)
    for key in list(store.keys()):
        if key == "edge_index":
            continue
        val = store[key]
        if torch.is_tensor(val) and val.ndim >= 1 and val.shape[0] == n_edges:
            store[key] = val[perm]
        elif isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n_edges:
            store[key] = val[perm]
    return store


def _replace_relation_edges(graph, relation_name, bank_pairs, rng):
    g = _clone_graph(graph)
    store = _edge_store(g, relation_name)
    old_ei = _to_numpy(store.edge_index).astype(np.int64)
    n_edges = int(old_ei.shape[1]) if old_ei.ndim == 2 else 0
    if n_edges == 0:
        return g

    new_ei = _sample_local_edge_index_from_bank(g, relation_name, bank_pairs, n_edges, rng)
    if new_ei is None:
        return g

    _permute_edge_aligned_attrs(store, n_edges, rng)
    store.edge_index = torch.as_tensor(new_ei, dtype=torch.long)
    return g


def _perturb_class_graphs(class_graphs, mode, rng, banks=None):
    if mode == "rna_only":
        relations = ["t2g"]
    elif mode == "multiome":
        relations = ["g2c", "c2g"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if banks is None:
        banks = _build_class_edge_banks(class_graphs, mode)

    perturbed = []
    for graph in class_graphs:
        g = _clone_graph(graph)
        for rel in relations:
            g = _replace_relation_edges(g, rel, banks[rel], rng)
        perturbed.append(g)
    return perturbed


def _batch_ptr(node_store):
    if hasattr(node_store, "ptr") and node_store.ptr is not None:
        return _to_numpy(node_store.ptr).astype(int)

    if hasattr(node_store, "batch") and node_store.batch is not None:
        b = _to_numpy(node_store.batch).astype(int)
        if b.size == 0:
            return np.array([0], dtype=int)
        counts = np.bincount(b, minlength=int(b.max()) + 1)
        return np.concatenate([[0], np.cumsum(counts)]).astype(int)

    if hasattr(node_store, "gene_feats") and node_store.gene_feats is not None:
        n = int(node_store.gene_feats.shape[0])
    elif hasattr(node_store, "x") and node_store.x is not None:
        n = int(node_store.x.shape[0])
    else:
        n = 0
    return np.array([0, n], dtype=int)


def _slice_pack_value(x, i, batch_size):
    arr = _to_numpy(x)
    if arr is None:
        return None
    if arr.ndim >= 1 and arr.shape[0] == batch_size:
        return arr[i]
    return arr if batch_size == 1 else None


def _structure_key(df, cols):
    return df[cols].astype(str).agg("||".join, axis=1)


def _bh_qvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)

    q = np.empty_like(ranked)
    q[order] = ranked
    return q


def _empirical_pvalue(obs, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return (1.0 + float((null_values >= obs).sum())) / (len(null_values) + 1.0)


def _attach_perm_stats(observed_df, perm_dfs, key_cols, stat_col):
    if observed_df is None or len(observed_df) == 0:
        return observed_df

    obs = observed_df.copy()
    obs["_key"] = _structure_key(obs, key_cols)
    all_perm = []

    for perm_id, df in enumerate(perm_dfs):
        if df is None or len(df) == 0:
            continue
        tmp = df.copy()
        tmp["_key"] = _structure_key(tmp, key_cols)
        tmp["perm_id"] = perm_id
        all_perm.append(tmp[["_key", stat_col, "perm_id"]])

    if len(all_perm) == 0:
        obs["null_mean"] = 0.0
        obs["null_std"] = 0.0
        obs["p_value"] = 1.0
        obs["q_value"] = 1.0
        return obs.drop(columns="_key")

    perm_long = pd.concat(all_perm, ignore_index=True)
    grouped = perm_long.groupby("_key")[stat_col].apply(list).to_dict()
    n_perm = len(perm_dfs)

    pvals, means, stds = [], [], []
    for _, row in obs.iterrows():
        null_vals = grouped.get(row["_key"], [])
        if len(null_vals) < n_perm:
            null_vals = list(null_vals) + [0.0] * (n_perm - len(null_vals))
        means.append(float(np.mean(null_vals)))
        stds.append(float(np.std(null_vals)))
        pvals.append(_empirical_pvalue(row[stat_col], null_vals))

    obs["null_mean"] = means
    obs["null_std"] = stds
    obs["p_value"] = pvals
    obs["q_value"] = _bh_qvalues(obs["p_value"].to_numpy())
    return obs.drop(columns="_key")


def _empty_like_with_columns(df):
    if df is None:
        return pd.DataFrame()
    return df.iloc[0:0].copy()


def _sort_structure_df(df, stat_col):
    if df is None or len(df) == 0:
        return df
    sort_cols = [c for c in ["cell_freq", stat_col, "n_cells", "edge_hits"] if c in df.columns]
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


def _candidate_key_set(df, key_cols):
    if df is None or len(df) == 0:
        return set()
    tmp = df.copy()
    tmp["_key"] = _structure_key(tmp, key_cols)
    return set(tmp["_key"].tolist())


def _filter_df_to_candidate_keys(df, key_cols, candidate_keys):
    if df is None or len(df) == 0:
        return _empty_like_with_columns(df)
    if candidate_keys is None:
        return df.copy()
    candidate_keys = set(candidate_keys)
    if len(candidate_keys) == 0:
        return df.iloc[0:0].copy()
    tmp = df.copy()
    tmp["_key"] = _structure_key(tmp, key_cols)
    tmp = tmp[tmp["_key"].isin(candidate_keys)].copy()
    return tmp.drop(columns="_key").reset_index(drop=True)


def _init_perm_accumulator(candidate_keys):
    return {str(k): [] for k in candidate_keys}


def _update_perm_accumulator(accumulator, df, key_cols, stat_col):
    if accumulator is None or len(accumulator) == 0:
        return

    if df is None or len(df) == 0:
        present = {}
    else:
        tmp = df.copy()
        tmp["_key"] = _structure_key(tmp, key_cols)
        tmp = tmp.sort_values(stat_col, ascending=False).drop_duplicates("_key")
        present = dict(zip(tmp["_key"], tmp[stat_col]))

    for key in accumulator.keys():
        accumulator[key].append(float(present.get(key, 0.0)))


def _attach_perm_stats_from_accumulator(observed_df, key_cols, stat_col, accumulator):
    if observed_df is None or len(observed_df) == 0:
        return observed_df

    obs = observed_df.copy()
    obs["_key"] = _structure_key(obs, key_cols)

    means, stds, pvals = [], [], []
    for _, row in obs.iterrows():
        vals = np.asarray(accumulator.get(row["_key"], []), dtype=float)
        if vals.size == 0:
            means.append(0.0)
            stds.append(0.0)
            pvals.append(1.0)
        else:
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
            pvals.append(_empirical_pvalue(row[stat_col], vals))

    obs["null_mean"] = means
    obs["null_std"] = stds
    obs["p_value"] = pvals
    obs["q_value"] = _bh_qvalues(obs["p_value"].to_numpy())
    return obs.drop(columns="_key")


def _subset_perm_accumulator(accumulator, candidate_keys):
    if accumulator is None:
        return {}
    return {k: list(accumulator.get(k, [])) for k in candidate_keys}


def _merge_perm_accumulators(*accumulators):
    merged = {}
    for acc in accumulators:
        if acc is None:
            continue
        for k, vals in acc.items():
            merged.setdefault(k, []).extend(list(vals))
    return merged


def _top_n_per_group(df, group_col, n, stat_col):
    if df is None or len(df) == 0 or n is None:
        return df
    n = int(n)
    if n <= 0:
        return df.iloc[0:0].copy()

    sort_cols = [stat_col]
    if "cell_freq" in df.columns:
        sort_cols.append("cell_freq")

    return (
        df.sort_values(sort_cols, ascending=False)
        .groupby(group_col, group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )