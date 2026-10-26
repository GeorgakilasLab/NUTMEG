import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from .collect import _forward_graphs_to_cells
from .utils import (
    _attach_perm_stats_from_accumulator,
    _build_class_edge_banks,
    _candidate_key_set,
    _empty_like_with_columns,
    _filter_df_to_candidate_keys,
    _get_mode,
    _init_perm_accumulator,
    _merge_perm_accumulators,
    _perturb_class_graphs,
    _sort_structure_df,
    _subset_perm_accumulator,
    _update_perm_accumulator,
    class_mask,
)


def attention_distribution_df(results, cell_type=None, mode="true_positive"):
    rows = []

    if cell_type is None:
        keep = np.ones(len(results["cells"]), dtype=bool)
    else:
        keep = class_mask(results, cell_type, mode=mode)

    for i, cell in enumerate(results["cells"]):
        if not keep[i]:
            continue
        if "attn" not in cell:
            continue

        y_true_name = cell.get("y_true_name", None)
        y_pred_name = cell.get("y_pred_name", None)

        if "t2g" in cell["attn"]:
            e = cell["attn"]["t2g"]
            alpha = np.asarray(e["alpha"]).reshape(-1)
            for j, a in enumerate(alpha):
                rows.append({
                    "cell_index": cell["cell_index"],
                    "sample_id": cell["sample_id"],
                    "relation": "t2g",
                    "src_global": int(e["src_global"][j]),
                    "dst_global": int(e["dst_global"][j]),
                    "src_name": e["src_name"][j],
                    "dst_name": e["dst_name"][j],
                    "alpha": float(a),
                    "y_true": cell.get("y_true", None),
                    "y_pred": cell.get("y_pred", None),
                    "y_true_name": y_true_name,
                    "y_pred_name": y_pred_name,
                })

        if "g2c" in cell["attn"]:
            e = cell["attn"]["g2c"]
            alpha = np.asarray(e["alpha"]).reshape(-1)
            for j, a in enumerate(alpha):
                rows.append({
                    "cell_index": cell["cell_index"],
                    "sample_id": cell["sample_id"],
                    "relation": "g2c",
                    "src_gene_global": int(e["src_gene_global"][j]),
                    "dst_cre_global": int(e["dst_cre_global"][j]),
                    "src_gene_name": e["src_gene_name"][j],
                    "dst_cre_name": e["dst_cre_name"][j],
                    "alpha": float(a),
                    "y_true": cell.get("y_true", None),
                    "y_pred": cell.get("y_pred", None),
                    "y_true_name": y_true_name,
                    "y_pred_name": y_pred_name,
                })

        if "c2g" in cell["attn"]:
            e = cell["attn"]["c2g"]
            alpha = np.asarray(e["alpha"]).reshape(-1)
            for j, a in enumerate(alpha):
                rows.append({
                    "cell_index": cell["cell_index"],
                    "sample_id": cell["sample_id"],
                    "relation": "c2g",
                    "src_cre_global": int(e["src_cre_global"][j]),
                    "dst_gene_global": int(e["dst_gene_global"][j]),
                    "src_cre_name": e["src_cre_name"][j],
                    "dst_gene_name": e["dst_gene_name"][j],
                    "alpha": float(a),
                    "y_true": cell.get("y_true", None),
                    "y_pred": cell.get("y_pred", None),
                    "y_true_name": y_true_name,
                    "y_pred_name": y_pred_name,
                })

    return pd.DataFrame(rows)


def summarize_attention(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["relation", "count", "mean", "std", "min", "median", "max", "q90", "q95", "q99"]
        )

    grp = df.groupby("relation")["alpha"]
    out = grp.agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    out["q90"] = out["relation"].map(grp.quantile(0.90))
    out["q95"] = out["relation"].map(grp.quantile(0.95))
    out["q99"] = out["relation"].map(grp.quantile(0.99))
    return out


def t2g_edges_for_cell(cell):
    if "attn" not in cell or "t2g" not in cell["attn"]:
        return pd.DataFrame(columns=[
            "cell_index", "sample_id", "src_global", "dst_global",
            "src_name", "dst_name", "alpha"
        ])

    e = cell["attn"]["t2g"]
    return pd.DataFrame({
        "cell_index": cell["cell_index"],
        "sample_id": cell["sample_id"],
        "src_global": e["src_global"],
        "dst_global": e["dst_global"],
        "src_name": e["src_name"],
        "dst_name": e["dst_name"],
        "alpha": e["alpha"],
    })


def multiome_paths_for_cell(cell, path_score="geometric_mean"):
    if "attn" not in cell or "g2c" not in cell["attn"] or "c2g" not in cell["attn"]:
        return pd.DataFrame(columns=[
            "cell_index", "sample_id",
            "tf_global", "cre_global", "gene_global",
            "tf_name", "cre_name", "gene_name",
            "alpha_g2c", "alpha_c2g", "path_score"
        ])

    g2c = pd.DataFrame({
        "tf_global": cell["attn"]["g2c"]["src_gene_global"],
        "cre_global": cell["attn"]["g2c"]["dst_cre_global"],
        "tf_name": cell["attn"]["g2c"]["src_gene_name"],
        "cre_name": cell["attn"]["g2c"]["dst_cre_name"],
        "alpha_g2c": cell["attn"]["g2c"]["alpha"],
    })

    c2g = pd.DataFrame({
        "cre_global": cell["attn"]["c2g"]["src_cre_global"],
        "gene_global": cell["attn"]["c2g"]["dst_gene_global"],
        "cre_name": cell["attn"]["c2g"]["src_cre_name"],
        "gene_name": cell["attn"]["c2g"]["dst_gene_name"],
        "alpha_c2g": cell["attn"]["c2g"]["alpha"],
    })

    paths = g2c.merge(c2g, on=["cre_global", "cre_name"], how="inner")
    if paths.empty:
        return pd.DataFrame(columns=[
            "cell_index", "sample_id",
            "tf_global", "cre_global", "gene_global",
            "tf_name", "cre_name", "gene_name",
            "alpha_g2c", "alpha_c2g", "path_score"
        ])

    if path_score == "product":
        paths["path_score"] = paths["alpha_g2c"] * paths["alpha_c2g"]
    elif path_score == "min":
        paths["path_score"] = np.minimum(paths["alpha_g2c"], paths["alpha_c2g"])
    else:
        paths["path_score"] = np.sqrt(paths["alpha_g2c"] * paths["alpha_c2g"])

    paths["cell_index"] = cell["cell_index"]
    paths["sample_id"] = cell["sample_id"]

    cols = [
        "cell_index", "sample_id",
        "tf_global", "cre_global", "gene_global",
        "tf_name", "cre_name", "gene_name",
        "alpha_g2c", "alpha_c2g", "path_score"
    ]
    return paths[cols]


def aggregate_t2g_by_cells(cells, top_k_per_cell=50, min_cells=3):
    if len(cells) == 0:
        return pd.DataFrame()

    rows = []
    for cell in cells:
        df = t2g_edges_for_cell(cell)
        if df.empty:
            continue
        if top_k_per_cell is not None and len(df) > top_k_per_cell:
            df = df.nlargest(top_k_per_cell, "alpha")
        rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    out = (
        df.groupby(["src_global", "dst_global", "src_name", "dst_name"], as_index=False)
        .agg(
            n_cells=("cell_index", "nunique"),
            edge_hits=("cell_index", "size"),
            mean_alpha=("alpha", "mean"),
            median_alpha=("alpha", "median"),
            max_alpha=("alpha", "max"),
        )
    )
    out["cell_freq"] = out["n_cells"] / len(cells)
    out = out[out["n_cells"] >= min_cells].copy()
    out = out.sort_values(["cell_freq", "mean_alpha", "edge_hits"], ascending=False).reset_index(drop=True)
    return out


def aggregate_paths_by_cells(cells, top_k_per_cell=50, min_cells=3, path_score="geometric_mean"):
    if len(cells) == 0:
        return pd.DataFrame()

    rows = []
    for cell in cells:
        df = multiome_paths_for_cell(cell, path_score=path_score)
        if df.empty:
            continue
        if top_k_per_cell is not None and len(df) > top_k_per_cell:
            df = df.nlargest(top_k_per_cell, "path_score")
        rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    out = (
        df.groupby(
            ["tf_global", "cre_global", "gene_global", "tf_name", "cre_name", "gene_name"],
            as_index=False
        ).agg(
            n_cells=("cell_index", "nunique"),
            mean_path_score=("path_score", "mean"),
            median_path_score=("path_score", "median"),
            max_path_score=("path_score", "max"),
            mean_alpha_g2c=("alpha_g2c", "mean"),
            mean_alpha_c2g=("alpha_c2g", "mean"),
        )
    )
    out["cell_freq"] = out["n_cells"] / len(cells)
    out = out[out["n_cells"] >= min_cells].copy()
    out = out.sort_values(["cell_freq", "mean_path_score"], ascending=False).reset_index(drop=True)
    return out


def top_regulators_from_t2g(edge_df):
    if edge_df is None or len(edge_df) == 0:
        return pd.DataFrame(columns=["src_global", "src_name", "n_targets", "mean_alpha", "total_support"])

    out = (
        edge_df.groupby(["src_global", "src_name"], as_index=False)
        .agg(
            n_targets=("dst_global", "nunique"),
            mean_alpha=("mean_alpha", "mean"),
            total_support=("cell_freq", "sum"),
        )
        .sort_values(["total_support", "n_targets", "mean_alpha"], ascending=False)
        .reset_index(drop=True)
    )
    return out


def top_regulators_from_paths(path_df):
    if path_df is None or len(path_df) == 0:
        return pd.DataFrame(columns=["tf_global", "tf_name", "n_cres", "n_targets", "mean_path_score", "total_support"])

    out = (
        path_df.groupby(["tf_global", "tf_name"], as_index=False)
        .agg(
            n_cres=("cre_global", "nunique"),
            n_targets=("gene_global", "nunique"),
            mean_path_score=("mean_path_score", "mean"),
            total_support=("cell_freq", "sum"),
        )
        .sort_values(["total_support", "n_targets", "mean_path_score"], ascending=False)
        .reset_index(drop=True)
    )
    return out


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


def _select_hybrid_candidates(
    observed_df,
    model_mode,
    *,
    stat_col,
    max_candidate_regulators,
    max_candidates_per_regulator,
    max_candidates,
    candidate_min_cell_freq,
):
    if observed_df is None or len(observed_df) == 0:
        return _empty_like_with_columns(observed_df)

    cand = observed_df.copy()

    if candidate_min_cell_freq is not None and "cell_freq" in cand.columns:
        cand = cand[cand["cell_freq"] >= float(candidate_min_cell_freq)].copy()

    cand = _sort_structure_df(cand, stat_col)
    if len(cand) == 0:
        return cand

    if model_mode == "multiome":
        regulators = top_regulators_from_paths(cand)
        regulator_col = "tf_global"
    else:
        regulators = top_regulators_from_t2g(cand)
        regulator_col = "src_global"

    if max_candidate_regulators is not None and len(regulators) > int(max_candidate_regulators):
        keep_regs = set(regulators.iloc[:int(max_candidate_regulators)][regulator_col].astype(int).tolist())
        cand = cand[cand[regulator_col].astype(int).isin(keep_regs)].copy()

    if max_candidates_per_regulator is not None and regulator_col in cand.columns:
        cand = _top_n_per_group(cand, regulator_col, max_candidates_per_regulator, stat_col)

    cand = _sort_structure_df(cand, stat_col)
    if max_candidates is not None and len(cand) > int(max_candidates):
        cand = cand.head(int(max_candidates)).copy()

    return cand.reset_index(drop=True)


def _screening_survivors(screening_stats, stat_col, *, screening_pvalue_threshold, max_refinement_candidates):
    if screening_stats is None or len(screening_stats) == 0:
        return _empty_like_with_columns(screening_stats)

    surv = screening_stats[screening_stats["p_value"] <= float(screening_pvalue_threshold)].copy()
    surv = surv.sort_values(["p_value", stat_col], ascending=[True, False]).reset_index(drop=True)

    if max_refinement_candidates is not None and len(surv) > int(max_refinement_candidates):
        surv = surv.head(int(max_refinement_candidates)).copy()

    if len(surv) == 0:
        fallback_n = 1 if max_refinement_candidates is None else max(
            1, min(int(max_refinement_candidates), min(20, len(screening_stats)))
        )
        surv = (
            screening_stats.sort_values(["p_value", stat_col], ascending=[True, False])
            .head(fallback_n)
            .copy()
            .reset_index(drop=True)
        )

    return surv


def _prepare_grn_jobs(
    results,
    classes,
    *,
    mode="true_positive",
    top_k_per_cell=50,
    min_cells=3,
    path_score="geometric_mean",
    all_graphs=None,
):
    model_mode = _get_mode(results)
    jobs = []

    for c in classes:
        keep = class_mask(results, c, mode=mode)
        class_idx = np.where(keep)[0]
        if len(class_idx) == 0:
            raise ValueError(f"No cells selected for class '{c}'")

        class_graphs = [all_graphs[i] for i in class_idx]
        class_cells_obs = [results["cells"][i] for i in class_idx]
        class_sample_ids = [results["sample_ids"][i] for i in class_idx]

        if model_mode == "rna_only":
            observed = aggregate_t2g_by_cells(
                class_cells_obs,
                top_k_per_cell=top_k_per_cell,
                min_cells=min_cells,
            )
            stat_col = "mean_alpha"
            key_cols = ["src_global", "dst_global", "src_name", "dst_name"]
        elif model_mode == "multiome":
            observed = aggregate_paths_by_cells(
                class_cells_obs,
                top_k_per_cell=top_k_per_cell,
                min_cells=min_cells,
                path_score=path_score,
            )
            stat_col = "mean_path_score"
            key_cols = ["tf_global", "cre_global", "gene_global", "tf_name", "cre_name", "gene_name"]
        else:
            raise ValueError(f"Unknown mode: {model_mode}")

        jobs.append({
            "class_name": c,
            "class_key": str(c),
            "class_idx": class_idx,
            "class_graphs": class_graphs,
            "class_cells_obs": class_cells_obs,
            "class_sample_ids": class_sample_ids,
            "observed": observed,
            "stat_col": stat_col,
            "key_cols": key_cols,
            "banks": _build_class_edge_banks(class_graphs, model_mode),
            "needs_perm": observed is not None and len(observed) > 0,
        })

    return jobs, model_mode


def _aggregate_perm_result(model_mode, perm_cells, top_k_per_cell, min_cells, path_score):
    if model_mode == "rna_only":
        return aggregate_t2g_by_cells(
            perm_cells,
            top_k_per_cell=top_k_per_cell,
            min_cells=min_cells,
        )
    return aggregate_paths_by_cells(
        perm_cells,
        top_k_per_cell=top_k_per_cell,
        min_cells=min_cells,
        path_score=path_score,
    )


def _reduce_single_slice(args):
    model_mode, perm_cells, top_k_per_cell, min_cells, path_score, key_cols, candidate_df, stat_col = args
    perm_df = _aggregate_perm_result(model_mode, perm_cells, top_k_per_cell, min_cells, path_score)
    perm_df = _filter_df_to_candidate_keys(
        perm_df,
        key_cols,
        _candidate_key_set(candidate_df, key_cols),
    )
    acc = _init_perm_accumulator(_candidate_key_set(candidate_df, key_cols))
    _update_perm_accumulator(acc, perm_df, key_cols, stat_col)
    return acc


def _run_hybrid_permutation_stage(
    runner,
    ds,
    jobs,
    model_mode,
    *,
    n_permutations,
    top_k_per_cell,
    min_cells,
    path_score,
    random_seed,
    forward_batch_size,
    permutation_batch_size,
    candidate_field,
    accumulator_field,
    n_jobs=1,
):
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        return

    active_jobs = [j for j in jobs if j.get(candidate_field, None) is not None and len(j[candidate_field]) > 0]
    if len(active_jobs) == 0:
        return

    for job in active_jobs:
        job[accumulator_field] = _init_perm_accumulator(_candidate_key_set(job[candidate_field], job["key_cols"]))

    job_map = {job["class_key"]: job for job in jobs}
    rng = np.random.default_rng(int(random_seed))
    perm_ids = list(range(n_permutations))
    permutation_batch_size = max(1, int(permutation_batch_size))
    n_jobs = max(1, int(n_jobs))

    for start in range(0, len(perm_ids), permutation_batch_size):
        perm_chunk = perm_ids[start:start + permutation_batch_size]

        big_graphs = []
        big_cell_indices = []
        big_sample_ids = []
        slices = []

        for perm_id in perm_chunk:
            for job in active_jobs:
                perturbed_graphs = _perturb_class_graphs(
                    job["class_graphs"],
                    mode=model_mode,
                    rng=rng,
                    banks=job["banks"],
                )

                s = len(big_graphs)
                n = len(perturbed_graphs)

                big_graphs.extend(perturbed_graphs)
                big_cell_indices.extend(job["class_idx"].tolist())
                big_sample_ids.extend(job["class_sample_ids"])

                slices.append({
                    "class_key": job["class_key"],
                    "perm_id": perm_id,
                    "start": s,
                    "stop": s + n,
                })

        if len(big_graphs) == 0:
            continue

        perm_cells_all = _forward_graphs_to_cells(
            runner,
            ds,
            big_graphs,
            cell_indices=big_cell_indices,
            sample_ids=big_sample_ids,
            forward_batch_size=forward_batch_size,
        )

        work_items = []
        for sl in slices:
            job = job_map[sl["class_key"]]
            perm_cells = perm_cells_all[sl["start"]:sl["stop"]]
            work_items.append((
                sl["class_key"],
                _reduce_single_slice,
                (
                    model_mode,
                    perm_cells,
                    top_k_per_cell,
                    min_cells,
                    path_score,
                    job["key_cols"],
                    job[candidate_field],
                    job["stat_col"],
                ),
            ))

        if n_jobs > 1 and len(work_items) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                future_map = {
                    ex.submit(fn, args): class_key
                    for class_key, fn, args in work_items
                }
                for fut in as_completed(future_map):
                    class_key = future_map[fut]
                    acc_part = fut.result()
                    job = job_map[class_key]
                    merged = _merge_perm_accumulators(job[accumulator_field], acc_part)
                    job[accumulator_field] = merged
        else:
            for class_key, fn, args in work_items:
                acc_part = fn(args)
                job = job_map[class_key]
                merged = _merge_perm_accumulators(job[accumulator_field], acc_part)
                job[accumulator_field] = merged


def _observed_only_result(model_mode, c, observed, stat_col, q_value_threshold, n_permutations, forward_batch_size, permutation_batch_size):
    observed = observed.copy()
    observed["null_mean"] = 0.0
    observed["null_std"] = 0.0
    observed["p_value"] = 1.0
    observed["q_value"] = 1.0
    if model_mode == "rna_only":
        return {
            "mode": "rna_only",
            "class_name": c,
            "significant_edges": observed,
            "regulators": top_regulators_from_t2g(observed),
            "n_permutations": int(n_permutations),
            "q_value_threshold": float(q_value_threshold),
            "stat_col": stat_col,
            "forward_batch_size": int(forward_batch_size),
            "permutation_batch_size": int(permutation_batch_size),
        }
    return {
        "mode": "multiome",
        "class_name": c,
        "significant_paths": observed,
        "regulators": top_regulators_from_paths(observed),
        "n_permutations": int(n_permutations),
        "q_value_threshold": float(q_value_threshold),
        "stat_col": stat_col,
        "forward_batch_size": int(forward_batch_size),
        "permutation_batch_size": int(permutation_batch_size),
    }


def significant_grn_all_classes_batched(
    runner,
    ds,
    results,
    classes,
    *,
    mode="true_positive",
    top_k_per_cell=50,
    min_cells=3,
    path_score="geometric_mean",
    n_permutations=100,
    q_value_threshold=0.05,
    random_seed=0,
    forward_batch_size=64,
    permutation_batch_size=8,
    all_graphs=None,
    n_jobs=1,
):
    if all_graphs is None:
        all_graphs = runner._materialize_graphs_from_ds(ds)

    jobs, model_mode = _prepare_grn_jobs(
        results,
        classes,
        mode=mode,
        top_k_per_cell=top_k_per_cell,
        min_cells=min_cells,
        path_score=path_score,
        all_graphs=all_graphs,
    )

    total_perm = int(n_permutations)
    if total_perm <= 0:
        out = {}
        for job in jobs:
            c = job["class_name"]
            ckey = job["class_key"]
            observed = job["observed"]
            stat_col = job["stat_col"]
            if observed is None or len(observed) == 0:
                if model_mode == "rna_only":
                    out[ckey] = {
                        "mode": "rna_only",
                        "class_name": c,
                        "significant_edges": pd.DataFrame(),
                        "regulators": top_regulators_from_t2g(pd.DataFrame()),
                        "n_permutations": int(n_permutations),
                        "q_value_threshold": float(q_value_threshold),
                        "stat_col": stat_col,
                        "forward_batch_size": int(forward_batch_size),
                        "permutation_batch_size": int(permutation_batch_size),
                    }
                else:
                    out[ckey] = {
                        "mode": "multiome",
                        "class_name": c,
                        "significant_paths": pd.DataFrame(),
                        "regulators": top_regulators_from_paths(pd.DataFrame()),
                        "n_permutations": int(n_permutations),
                        "q_value_threshold": float(q_value_threshold),
                        "stat_col": stat_col,
                        "forward_batch_size": int(forward_batch_size),
                        "permutation_batch_size": int(permutation_batch_size),
                    }
            else:
                out[ckey] = _observed_only_result(
                    model_mode, c, observed, stat_col, q_value_threshold,
                    n_permutations, forward_batch_size, permutation_batch_size,
                )
        return out

    screening_permutations = min(32, max(16, total_perm // 4)) if total_perm >= 16 else max(1, total_perm // 2)
    refinement_permutations = max(1, total_perm - screening_permutations)

    if model_mode == "multiome":
        max_candidate_regulators = 25
        max_candidates_per_regulator = 20
        max_candidates = 300
        max_refinement_candidates = 120
        candidate_min_cell_freq = None
    else:
        max_candidate_regulators = 30
        max_candidates_per_regulator = 25
        max_candidates = 400
        max_refinement_candidates = 160
        candidate_min_cell_freq = None

    screening_pvalue_threshold = 0.25

    for job in jobs:
        observed = job["observed"]
        stat_col = job["stat_col"]

        if observed is None or len(observed) == 0:
            job["candidate_observed"] = _empty_like_with_columns(observed)
            job["screening_stats"] = _empty_like_with_columns(observed)
            job["refinement_observed"] = _empty_like_with_columns(observed)
            continue

        job["candidate_observed"] = _select_hybrid_candidates(
            observed,
            model_mode,
            stat_col=stat_col,
            max_candidate_regulators=max_candidate_regulators,
            max_candidates_per_regulator=max_candidates_per_regulator,
            max_candidates=max_candidates,
            candidate_min_cell_freq=candidate_min_cell_freq,
        )
        job["screening_stats"] = None
        job["refinement_observed"] = None

    _run_hybrid_permutation_stage(
        runner,
        ds,
        jobs,
        model_mode,
        n_permutations=screening_permutations,
        top_k_per_cell=top_k_per_cell,
        min_cells=min_cells,
        path_score=path_score,
        random_seed=random_seed,
        forward_batch_size=forward_batch_size,
        permutation_batch_size=permutation_batch_size,
        candidate_field="candidate_observed",
        accumulator_field="screening_accumulator",
        n_jobs=n_jobs,
    )

    for job in jobs:
        candidate_observed = job.get("candidate_observed", None)
        if candidate_observed is None or len(candidate_observed) == 0:
            job["screening_stats"] = _empty_like_with_columns(candidate_observed)
            job["refinement_observed"] = _empty_like_with_columns(candidate_observed)
            continue

        screening_stats = _attach_perm_stats_from_accumulator(
            candidate_observed,
            key_cols=job["key_cols"],
            stat_col=job["stat_col"],
            accumulator=job.get("screening_accumulator", {}),
        )
        job["screening_stats"] = screening_stats
        job["refinement_observed"] = _screening_survivors(
            screening_stats,
            job["stat_col"],
            screening_pvalue_threshold=screening_pvalue_threshold,
            max_refinement_candidates=max_refinement_candidates,
        )

    _run_hybrid_permutation_stage(
        runner,
        ds,
        jobs,
        model_mode,
        n_permutations=refinement_permutations,
        top_k_per_cell=top_k_per_cell,
        min_cells=min_cells,
        path_score=path_score,
        random_seed=int(random_seed) + 100003,
        forward_batch_size=forward_batch_size,
        permutation_batch_size=permutation_batch_size,
        candidate_field="refinement_observed",
        accumulator_field="refinement_accumulator",
        n_jobs=n_jobs,
    )

    out = {}

    for job in jobs:
        c = job["class_name"]
        ckey = job["class_key"]
        observed = job["observed"]
        stat_col = job["stat_col"]

        if observed is None or len(observed) == 0:
            if model_mode == "rna_only":
                out[ckey] = {
                    "mode": "rna_only",
                    "class_name": c,
                    "significant_edges": pd.DataFrame(),
                    "regulators": top_regulators_from_t2g(pd.DataFrame()),
                    "n_permutations": int(n_permutations),
                    "q_value_threshold": float(q_value_threshold),
                    "stat_col": stat_col,
                    "forward_batch_size": int(forward_batch_size),
                    "permutation_batch_size": int(permutation_batch_size),
                }
            else:
                out[ckey] = {
                    "mode": "multiome",
                    "class_name": c,
                    "significant_paths": pd.DataFrame(),
                    "regulators": top_regulators_from_paths(pd.DataFrame()),
                    "n_permutations": int(n_permutations),
                    "q_value_threshold": float(q_value_threshold),
                    "stat_col": stat_col,
                    "forward_batch_size": int(forward_batch_size),
                    "permutation_batch_size": int(permutation_batch_size),
                }
            continue

        refinement_observed = job.get("refinement_observed", _empty_like_with_columns(observed))

        if refinement_observed is None or len(refinement_observed) == 0:
            observed_stats = _empty_like_with_columns(refinement_observed)
        else:
            final_keys = _candidate_key_set(refinement_observed, job["key_cols"])
            final_acc = _merge_perm_accumulators(
                _subset_perm_accumulator(job.get("screening_accumulator", {}), final_keys),
                _subset_perm_accumulator(job.get("refinement_accumulator", {}), final_keys),
            )
            observed_stats = _attach_perm_stats_from_accumulator(
                refinement_observed,
                key_cols=job["key_cols"],
                stat_col=stat_col,
                accumulator=final_acc,
            )

        significant = (
            observed_stats[observed_stats["q_value"] <= float(q_value_threshold)].copy()
            if len(observed_stats)
            else observed_stats
        )

        if model_mode == "rna_only":
            out[ckey] = {
                "mode": "rna_only",
                "class_name": c,
                "significant_edges": significant,
                "regulators": top_regulators_from_t2g(significant),
                "n_permutations": int(n_permutations),
                "q_value_threshold": float(q_value_threshold),
                "stat_col": stat_col,
                "forward_batch_size": int(forward_batch_size),
                "permutation_batch_size": int(permutation_batch_size),
            }
        else:
            out[ckey] = {
                "mode": "multiome",
                "class_name": c,
                "significant_paths": significant,
                "regulators": top_regulators_from_paths(significant),
                "n_permutations": int(n_permutations),
                "q_value_threshold": float(q_value_threshold),
                "stat_col": stat_col,
                "forward_batch_size": int(forward_batch_size),
                "permutation_batch_size": int(permutation_batch_size),
            }

    return out


def significant_class_grn(
    runner,
    ds,
    results,
    c,
    *,
    mode="true_positive",
    top_k_per_cell=50,
    min_cells=3,
    path_score="geometric_mean",
    n_permutations=100,
    q_value_threshold=0.05,
    random_seed=0,
    forward_batch_size=64,
    permutation_batch_size=8,
    all_graphs=None,
    n_jobs=1,
):
    out = significant_grn_all_classes_batched(
        runner,
        ds,
        results,
        [c],
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
        n_jobs=n_jobs,
    )
    return out[str(c)]