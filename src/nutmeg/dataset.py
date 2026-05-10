from __future__ import annotations

import os
import glob
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Optional, Tuple, List, Any, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from scipy.io import mmread


# ============================================================
# 1) Base GRN 
# ============================================================

@dataclass
class BaseGRNConfig:
    tf_peak_scores_csv: Optional[str] = None
    peaks_genes_csv: Optional[str] = None
    out_plot_dir: str = "plots/base_grn"
    plot_filename: str = "base_grn_degree_distributions.png"
    tf_col: str = "tf"
    peak_col: str = "peak"
    gene_col: str = "gene"
    tf_score_col: str = "score_z"
    mode: str = "multiome"
    tf_score_agg: str = "max"  
    dist_agg: str = "min"  

class BaseGRN:
    def __init__(self, cfg: BaseGRNConfig):
        self.cfg = cfg
        assert self.cfg.mode in ("multiome", "rna_only")
        self.tf_peak_df: Optional[pd.DataFrame] = None
        self.peaks_genes_df: Optional[pd.DataFrame] = None
        self.all_genes: Optional[pd.Index] = None
        self.all_cres: Optional[pd.Index] = None
        self.gene_to_id: Dict[str, int] = {}
        self.cre_to_id: Dict[str, int] = {}
        self.graph: Optional[HeteroData] = None
        self.dataset: Optional[PerCellGraphDataset] = None

    def create_base_grn(self, save_graph_pt: Optional[str] = None) -> HeteroData:
        self._load()
        if self.cfg.mode == "multiome":
            self._build_indices_multiome()
            e_g2c, a_g2c = self._build_binds()
            e_c2g, a_c2g = self._build_regulates()
            self.graph = self._pack_multiome(e_g2c, a_g2c, e_c2g, a_c2g)
        else:
            self._build_indices_rna_only()
            e_g2g, a_g2g = self._build_direct_tf_gene()
            self.graph = self._pack_rna_only(e_g2g, a_g2g)
        
        self.save_degree_plot()
        if save_graph_pt:
            torch.save(self.graph, save_graph_pt)
            print(f"[BaseGRN] Saved base graph ({self.cfg.mode}) -> {os.path.abspath(save_graph_pt)}")
        return self.graph

    def _load(self) -> None:
        self.tf_peak_df = pd.read_csv(self.cfg.tf_peak_scores_csv)
        self.peaks_genes_df = pd.read_csv(self.cfg.peaks_genes_csv)

        # Validate required columns:
        need_tf = {self.cfg.tf_col, self.cfg.peak_col, self.cfg.tf_score_col}
        need_pg = {self.cfg.peak_col, self.cfg.gene_col}


        missing_tf = need_tf - set(self.tf_peak_df.columns)
        missing_pg = need_pg - set(self.peaks_genes_df.columns)
        if missing_tf:
            raise ValueError(f"tf_peak_scores_csv missing columns: {sorted(missing_tf)}")
        if missing_pg:
            raise ValueError(f"peaks_genes_csv missing columns: {sorted(missing_pg)}")

        # Normalize dtypes:
        self.tf_peak_df[self.cfg.tf_col] = self.tf_peak_df[self.cfg.tf_col].astype(str)
        self.tf_peak_df[self.cfg.peak_col] = self.tf_peak_df[self.cfg.peak_col].astype(str)

        self.peaks_genes_df[self.cfg.peak_col] = self.peaks_genes_df[self.cfg.peak_col].astype(str)
        self.peaks_genes_df[self.cfg.gene_col] = self.peaks_genes_df[self.cfg.gene_col].astype(str)

        print("[BaseGRN] Loaded:")
        print(f"  - mode:          {self.cfg.mode}")
        print(f"  - tf_peak_scores {self.tf_peak_df.shape}")
        print(f"  - peaks_genes    {self.peaks_genes_df.shape}")

    def _build_indices_multiome(self) -> None:
        tfs = set(self.tf_peak_df[self.cfg.tf_col].astype(str).unique())
        targets = set(self.peaks_genes_df[self.cfg.gene_col].astype(str).unique())

        overlap = tfs & targets
        tf_only = tfs - targets
        target_only = targets - tfs

        all_genes = pd.Index(sorted(tfs | targets))
        all_cres = pd.Index(self.tf_peak_df[self.cfg.peak_col].astype(str)).append(
            pd.Index(self.peaks_genes_df[self.cfg.peak_col].astype(str))
        ).unique()

        self.all_genes = all_genes
        self.all_cres = all_cres
        self.gene_to_id = {g: i for i, g in enumerate(all_genes)}
        self.cre_to_id = {c: i for i, c in enumerate(all_cres)}

        print("\n[BaseGRN] Unified nodes (multiome):")
        print(f"  - TFs:                 {len(tfs)}")
        print(f"  - Targets (genes):     {len(targets)}")
        print(f"  - TF ∩ Target:         {len(overlap)}")
        print(f"  - TF only:             {len(tf_only)}")
        print(f"  - Target only:         {len(target_only)}")
        print(f"  - Genes (TF ∪ Target): {len(all_genes)}")
        print(f"  - CREs:                {len(all_cres)}")

    def _build_binds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        df = self.tf_peak_df[[self.cfg.tf_col, self.cfg.peak_col, self.cfg.tf_score_col]].dropna().copy()
        df[self.cfg.tf_col] = df[self.cfg.tf_col].astype(str)
        df[self.cfg.peak_col] = df[self.cfg.peak_col].astype(str)

        mask = np.array(
            [(t in self.gene_to_id) and (p in self.cre_to_id)
            for t, p in zip(df[self.cfg.tf_col].values, df[self.cfg.peak_col].values)],
            dtype=bool
        )
        df = df.loc[mask]

        src = [self.gene_to_id[t] for t in df[self.cfg.tf_col].values]   
        dst = [self.cre_to_id[p]  for p in df[self.cfg.peak_col].values] 

        a = df[[self.cfg.tf_score_col]].astype(float).values  
        return torch.tensor([src, dst], dtype=torch.long), torch.tensor(a, dtype=torch.float32)

    def _build_regulates(self) -> Tuple[torch.Tensor, torch.Tensor]:

        df = self.peaks_genes_df[[self.cfg.peak_col, self.cfg.gene_col]].dropna(
            subset=[self.cfg.peak_col, self.cfg.gene_col]
        ).copy()

        df[self.cfg.peak_col] = df[self.cfg.peak_col].astype(str)
        df[self.cfg.gene_col] = df[self.cfg.gene_col].astype(str)

        mask = np.array(
            [(p in self.cre_to_id) and (g in self.gene_to_id)
            for p, g in zip(df[self.cfg.peak_col].values, df[self.cfg.gene_col].values)],
            dtype=bool
        )
        df = df.loc[mask]

        src = [self.cre_to_id[p] for p in df[self.cfg.peak_col].values]
        dst = [self.gene_to_id[g] for g in df[self.cfg.gene_col].values]

        feats = np.zeros((len(src), 0), dtype=float)

        return (
            torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(feats, dtype=torch.float32),
        )

    
    def _pack_multiome(self, e_g2c, a_g2c, e_c2g, a_c2g) -> HeteroData:
        g = HeteroData()
        g["gene"].num_nodes = len(self.all_genes)
        g["cre"].num_nodes = len(self.all_cres)
        g["gene"].names = list(self.all_genes)
        g["cre"].names = list(self.all_cres)

        g["gene", "binds", "cre"].edge_index = e_g2c
        g["gene", "binds", "cre"].edge_attr  = a_g2c

        g["cre", "regulates", "gene"].edge_index = e_c2g
        g["cre", "regulates", "gene"].edge_attr  = a_c2g
        return g
    
    def _build_indices_rna_only(self) -> None:
        tfs = set(self.tf_peak_df[self.cfg.tf_col].astype(str).unique())
        targets = set(self.peaks_genes_df[self.cfg.gene_col].astype(str).unique())

        overlap = tfs & targets
        tf_only = tfs - targets
        target_only = targets - tfs

        all_genes = pd.Index(sorted(tfs | targets))

        self.all_genes = all_genes
        self.all_cres = pd.Index([])          
        self.gene_to_id = {g: i for i, g in enumerate(all_genes)}
        self.cre_to_id = {}                   

        print("\n[BaseGRN] Unified nodes (rna_only):")
        print(f"  - TFs:                 {len(tfs)}")
        print(f"  - Targets (genes):     {len(targets)}")
        print(f"  - TF ∩ Target:         {len(overlap)}")
        print(f"  - TF only:             {len(tf_only)}")
        print(f"  - Target only:         {len(target_only)}")
        print(f"  - Genes (TF ∪ Target): {len(all_genes)}")


    def _build_direct_tf_gene(self):
        tf_df = self.tf_peak_df[[self.cfg.tf_col, self.cfg.peak_col, self.cfg.tf_score_col]].dropna().copy()
        pg_df = self.peaks_genes_df[[self.cfg.peak_col, self.cfg.gene_col]].dropna(
            subset=[self.cfg.peak_col, self.cfg.gene_col]
        ).copy()

        
        tf_df[self.cfg.peak_col] = tf_df[self.cfg.peak_col].astype(str).str.strip()
        pg_df[self.cfg.peak_col] = pg_df[self.cfg.peak_col].astype(str).str.strip()

        merged = tf_df.merge(pg_df, on=self.cfg.peak_col, how="inner")

        merged = (merged
            .groupby([self.cfg.tf_col, self.cfg.gene_col], as_index=False)
            .agg({
                self.cfg.tf_score_col: self.cfg.tf_score_agg,  
                self.cfg.peak_col: "size",
            })
            .rename(columns={self.cfg.peak_col: "n_paths"})
        )
        merged["n_paths_log1p"] = np.log1p(merged["n_paths"].astype(float))

        mask = np.array(
            [(t in self.gene_to_id) and (g in self.gene_to_id)
            for t, g in zip(merged[self.cfg.tf_col].astype(str).values,
                             merged[self.cfg.gene_col].astype(str).values)],
            dtype=bool
        )
        merged = merged.loc[mask]

        src = [self.gene_to_id[t] for t in merged[self.cfg.tf_col].astype(str).values]
        dst = [self.gene_to_id[g] for g in merged[self.cfg.gene_col].astype(str).values]

        edge_attr = merged[[self.cfg.tf_score_col, "n_paths_log1p"]].astype(float).values  

        e = torch.tensor([src, dst], dtype=torch.long)
        a = torch.tensor(edge_attr, dtype=torch.float32)
        return e, a

    
    def _pack_rna_only(self, e_g2g: torch.Tensor, a_g2g: torch.Tensor) -> HeteroData:
        g = HeteroData()
        g["gene"].num_nodes = len(self.all_genes)
        g["gene"].names = list(self.all_genes)

        g["gene", "regulates", "gene"].edge_index = e_g2g
        g["gene", "regulates", "gene"].edge_attr  = a_g2g
        return g


    def save_degree_plot(self) -> str:
        os.makedirs(self.cfg.out_plot_dir, exist_ok=True)
        out_path = os.path.join(self.cfg.out_plot_dir, self.cfg.plot_filename)

        if self.graph is None:
            raise RuntimeError("Graph not built yet. Call create_base_grn() first.")

        g = self.graph

        def degree_counts(indices: torch.Tensor) -> List[int]:
            c = defaultdict(int)
            for i in indices.detach().cpu().tolist():
                c[int(i)] += 1
            return list(c.values())

        if self.cfg.mode == "multiome":
            src_tf, dst_cre = g["gene", "binds", "cre"].edge_index
            src_cre, dst_gene = g["cre", "regulates", "gene"].edge_index

            tf_out = degree_counts(src_tf)
            cre_in = degree_counts(dst_cre)
            cre_out = degree_counts(src_cre)
            gene_in = degree_counts(dst_gene)

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()

            axs[0].hist(tf_out, bins=30);  axs[0].set_title("TF → CRE (TF out-degree)")
            axs[1].hist(cre_in, bins=30);  axs[1].set_title("TF → CRE (CRE in-degree)")
            axs[2].hist(cre_out, bins=30); axs[2].set_title("CRE → Gene (CRE out-degree)")
            axs[3].hist(gene_in, bins=30); axs[3].set_title("CRE → Gene (Gene in-degree)")

            plt.tight_layout()
            fig.suptitle("Edge degree distributions (multiome)", fontsize=16, y=1.03)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        else: 
            src, dst = g["gene", "regulates", "gene"].edge_index

            out_deg = degree_counts(src)  
            in_deg = degree_counts(dst)   

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            axs[0].hist(out_deg, bins=30)
            axs[0].set_title("Gene→Gene (source out-degree)")

            axs[1].hist(in_deg, bins=30)
            axs[1].set_title("Gene→Gene (target in-degree)")

            plt.tight_layout()
            fig.suptitle("Edge degree distributions (rna_only)", fontsize=16, y=1.05)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"[BaseGRN] Degree plot saved -> {os.path.abspath(out_path)}")
        return out_path



# ============================================================
# 2) Auto-discovery for feature_mtx/
# ============================================================

@dataclass
class AutoDiscoveredFiles:
    barcodes_file: str
    tf_features: str
    gene_features: str

    # raw matrices (in raw/)
    tf_raw_mtx: str
    gene_raw_mtx: str

    # processed matrices (in processed/)
    tf_sct_mtx: str
    gene_sct_mtx: str

    # CRE bits (multiome only)
    cre_features: Optional[str] = None
    cre_counts_raw_mtx: Optional[str] = None
    cre_counts_mtx: Optional[str] = None


class AutoFeatureLoader:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.feature_mtx_dir = os.path.join(experiment_dir, "interm", "feature_mtx")
        self.raw_dir = os.path.join(self.feature_mtx_dir, "raw")
        self.processed_dir = os.path.join(self.feature_mtx_dir, "processed")

    def discover(self, mode: str = "multiome") -> AutoDiscoveredFiles:
        assert mode in ("multiome", "rna_only")

        for dir_path, name in [
            (self.feature_mtx_dir, "feature_mtx"),
            (self.raw_dir, "raw"),
            (self.processed_dir, "processed"),
        ]:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"{name} dir not found: {dir_path}")

        def _find_one(base_dir: str, candidates: List[str], label: str) -> str:
            for name in candidates:
                path = os.path.join(base_dir, name)
                if os.path.exists(path):
                    return path
            for pat in candidates:
                hits = glob.glob(os.path.join(base_dir, pat))
                if hits:
                    return sorted(hits)[0]
            raise FileNotFoundError(f"Missing {label} in {base_dir}. Tried: {candidates}")

        # feature_mtx dir (txt files)
        barcodes = _find_one(self.feature_mtx_dir, ["barcodes.txt", "*barcodes*.txt"], "barcodes.txt")
        tf_features = _find_one(self.feature_mtx_dir, ["tf_features.txt", "tf*features*.txt"], "tf_features.txt")
        gene_features = _find_one(self.feature_mtx_dir, ["gene_features.txt", "gene*features*.txt"], "gene_features.txt")

        # raw dir (mtx files)
        tf_raw_mtx = _find_one(self.raw_dir, ["tf_raw.mtx", "tf*raw*.mtx"], "tf_raw.mtx")
        gene_raw_mtx = _find_one(self.raw_dir, ["gene_raw.mtx", "gene*raw*.mtx"], "gene_raw.mtx")

        # processed dir (mtx files)
        tf_sct_mtx = _find_one(self.processed_dir, ["tf_sct.mtx", "tf*sct*.mtx"], "tf_sct.mtx")
        gene_sct_mtx = _find_one(self.processed_dir, ["gene_sct.mtx", "gene*sct*.mtx"], "gene_sct.mtx")

        # default CRE to None for rna_only
        cre_features = None
        cre_counts_raw_mtx = None
        cre_counts_mtx = None

        if mode == "multiome":
            cre_features = _find_one(self.feature_mtx_dir, ["cre_features.txt", "cre*features*.txt"], "cre_features.txt")
            cre_counts_raw_mtx = _find_one(self.raw_dir, ["cre_counts_raw.mtx", "cre*raw*.mtx"], "cre_counts_raw.mtx")
            cre_counts_mtx = _find_one(self.processed_dir, ["cre_counts.mtx", "cre*counts*.mtx"], "cre_counts.mtx")

        files = AutoDiscoveredFiles(
            barcodes_file=barcodes,
            tf_features=tf_features,
            gene_features=gene_features,
            tf_raw_mtx=tf_raw_mtx,
            gene_raw_mtx=gene_raw_mtx,
            tf_sct_mtx=tf_sct_mtx,
            gene_sct_mtx=gene_sct_mtx,
            cre_features=cre_features,
            cre_counts_raw_mtx=cre_counts_raw_mtx,
            cre_counts_mtx=cre_counts_mtx,
        )

        print(f"[discover] Found (mode={mode}):")
        for k, v in files.__dict__.items():
            if v is None:
                continue

            name = os.path.basename(v) if isinstance(v, str) else "None"
            print(f"  - {k}: {name}")
        return files

    @staticmethod
    def _read_names(path: str) -> List[str]:
        return pd.read_csv(path, header=None)[0].astype(str).tolist()

    @staticmethod
    def _read_mtx(path: str):
        return mmread(path).tocsc()

    def load_all(self, files: AutoDiscoveredFiles, mode: str = "multiome") -> Dict[str, Tuple[Any, List[str], List[str]]]:
        assert mode in ("multiome", "rna_only")
        barcodes = self._read_names(files.barcodes_file)

        def load_triplet(mtx_path: str, feat_path: str, key: str):
            if mtx_path is None or feat_path is None:
                raise ValueError(f"[{key}] missing path(s): mtx_path={mtx_path}, feat_path={feat_path}")
            mat = self._read_mtx(mtx_path)
            feats = self._read_names(feat_path)
            if mat.shape != (len(feats), len(barcodes)):
                raise ValueError(f"[{key}] shape mismatch {mat.shape} vs {(len(feats), len(barcodes))}")
            print(f"[load] {key:12s} shape={mat.shape}")
            return mat, feats, barcodes

        mats: Dict[str, Tuple[Any, List[str], List[str]]] = {}

        # Processed (SCT)
        mats["tf_feat"] = load_triplet(files.tf_sct_mtx, files.tf_features, "tf_feat")
        mats["gene_feat"] = load_triplet(files.gene_sct_mtx, files.gene_features, "gene_feat")

        # Raw
        mats["tf_raw"] = load_triplet(files.tf_raw_mtx, files.tf_features, "tf_raw")
        mats["gene_raw"] = load_triplet(files.gene_raw_mtx, files.gene_features, "gene_raw")

        # CRE only in multiome
        if mode == "multiome":
            if files.cre_counts_mtx is None or files.cre_counts_raw_mtx is None or files.cre_features is None:
                raise FileNotFoundError("mode=multiome but CRE files are missing (cre_features/cre_counts*.mtx).")
            mats["cre_feat"] = load_triplet(files.cre_counts_mtx, files.cre_features, "cre_feat")
            mats["cre_raw"] = load_triplet(files.cre_counts_raw_mtx, files.cre_features, "cre_raw")

        return mats


# ============================================================
# 3) Per-cell dataset 
# ============================================================

class PerCellGraphDataset(Dataset):
    def __init__(self,
                 base_graph,
                 cells,
                 tf_mtx_dict, tf_names,
                 gene_mtx_dict, gene_names,
                 cre_mtx_dict, cre_names,
                 gene_node_names, cre_node_names,
                 labels_dict=None,
                 label_to_id=None,          
                 prune=True,
                 raw_expr_matrices=None,
                 min_cells_per_class=None,
                 max_cells_per_class=None,
                 mode: str = "multiome"):
        assert mode in ("multiome", "rna_only")
        self.mode = mode
        self.base_graph = base_graph
        self.tf_mtx_dict = tf_mtx_dict
        self.gene_mtx_dict = gene_mtx_dict

        self.tf_names = tf_names
        self.gene_names = gene_names

        self.gene_node_names = gene_node_names

        self.prune = prune
        self.raw_expr_matrices = raw_expr_matrices if prune else {}

        if self.mode == "rna_only":
            self.cre_mtx_dict = {}
            self.cre_names = []
            self.cre_node_names = []
        else:
            self.cre_mtx_dict = cre_mtx_dict or {}
            self.cre_names = cre_names or []
            self.cre_node_names = cre_node_names or []

        # keep original barcode order
        self.master_cols = list(cells)
        self.barcode2col = {cid: i for i, cid in enumerate(self.master_cols)}

        # ---------- Labels: prune -> build contiguous IDs  ----------
        self.global_label_to_id_input = dict(label_to_id) if label_to_id is not None else None

        if labels_dict is not None:
            from collections import Counter
            counts = Counter([labels_dict[c] for c in self.master_cols if c in labels_dict])
            valid_labels = {
                lab for lab, cnt in counts.items()
                if (min_cells_per_class is None or cnt >= min_cells_per_class)
                and (max_cells_per_class is None or cnt <= max_cells_per_class)
            }
            kept = [c for c in self.master_cols if labels_dict.get(c) in valid_labels]
            self.labels_dict = {c: labels_dict[c] for c in kept}

            # Build contiguous IDs *after* pruning:
            surviving_labels = sorted(set(self.labels_dict.values()))
            self.orig_label_to_compact = {lab: i for i, lab in enumerate(surviving_labels)}
            self.compact_to_orig = {i: lab for lab, i in self.orig_label_to_compact.items()}

            self.label_to_id = dict(self.orig_label_to_compact)   
            self.id_to_label = dict(self.compact_to_orig)         

            print(f"[Labels] Pruned to {len(surviving_labels)} classes; remapped to contiguous IDs 0..{len(surviving_labels)-1}")
            if self.global_label_to_id_input is not None:
                print(f"[Labels] (Info) Received a global label_to_id with {len(self.global_label_to_id_input)} entries; "
                      f"kept only surviving labels and rebuilt a compact mapping for training.")
        else:
            kept = self.master_cols
            self.labels_dict = None
            self.label_to_id = None
            self.id_to_label = None
            self.orig_label_to_compact = None
            self.compact_to_orig = None

        self.all_cells = list(kept)
        self.cell_id_to_col = {cid: self.barcode2col[cid] for cid in self.all_cells}

        self.gene_pos = {g: i for i, g in enumerate(gene_node_names)}
        if self.mode == "multiome":
            self.cre_pos = {p: i for i, p in enumerate(self.cre_node_names)}
        else:
            self.cre_pos = {}

        # ---------- build row→scaffold maps ----------
        def _build_row_map(src_names, dst_pos):
            src_rows, dst_nodes = [], []
            for i, name in enumerate(src_names):
                j = dst_pos.get(name)
                if j is None:
                    continue
                src_rows.append(i)
                dst_nodes.append(j)
            return (np.asarray(src_rows, dtype=np.int64),
                    torch.as_tensor(dst_nodes, dtype=torch.long))

        # row maps
        self.tf_src_rows, self.tf_dst_nodes = _build_row_map(self.tf_names, self.gene_pos)
        self.gene_src_rows, self.gene_dst_nodes = _build_row_map(self.gene_names, self.gene_pos)

        if self.mode == "multiome":
            self.cre_src_rows, self.cre_dst_nodes = _build_row_map(self.cre_names, self.cre_pos)
        else:
            self.cre_src_rows = np.asarray([], dtype=np.int64)
            self.cre_dst_nodes = torch.empty((0,), dtype=torch.long)
        
        # probe valid cells
        self.valid_cells = []
        for cid in self.all_cells:
            g = self.build_graph(cid)
            if g is not None:
                self.valid_cells.append(cid)

        print(f"Kept {len(self.valid_cells)} cells ")

    def __len__(self):
        return len(self.valid_cells)

    def __getitem__(self, idx):
        cid = self.valid_cells[idx]
        g = self.build_graph(cid)
        if g is None:
            raise IndexError(f"[__getitem__] Failed to build graph for {cid}")
        return g

    # -------- internals --------
    def _col_from(self, mtx, rows, col):
        return torch.tensor(mtx[rows, col].toarray().ravel(), dtype=torch.float32)

    def build_graph(self, cell_id):
        col = self.cell_id_to_col.get(cell_id)
        if col is None:
            return None

        # 1) per-cell features from aligned rows
        try:
            tf_cols   = [self._col_from(mtx,   self.tf_src_rows,   col) for mtx in self.tf_mtx_dict.values()]
            gene_cols = [self._col_from(mtx, self.gene_src_rows,   col) for mtx in self.gene_mtx_dict.values()]
            if self.mode == "multiome":
                cre_cols = [self._col_from(mtx, self.cre_src_rows, col) for mtx in self.cre_mtx_dict.values()]
            else:
                cre_cols = []

            tf_feats   = torch.stack(tf_cols,   dim=1) if tf_cols   else torch.zeros((len(self.tf_src_rows),   0), dtype=torch.float32)
            gene_feats = torch.stack(gene_cols, dim=1) if gene_cols else torch.zeros((len(self.gene_src_rows), 0), dtype=torch.float32)
            cre_feats  = torch.stack(cre_cols,  dim=1) if cre_cols  else torch.zeros((len(self.cre_src_rows),  0), dtype=torch.float32)
        except Exception as e:
            print(f"[build_graph] column slicing failed for cell {cell_id}: {e}")
            return None

        # 2) place into scaffold-sized tensors
        graph = self.base_graph.clone()

        G = len(self.gene_node_names)

        gene_tf_full   = torch.zeros((G, tf_feats.shape[1]),   dtype=torch.float32)
        gene_feat_full = torch.zeros((G, gene_feats.shape[1]), dtype=torch.float32)

        if tf_feats.numel() > 0:
            gene_tf_full[self.tf_dst_nodes] = tf_feats
        if gene_feats.numel() > 0:
            gene_feat_full[self.gene_dst_nodes] = gene_feats


        graph['gene'].tf_feats   = gene_tf_full
        graph['gene'].gene_feats = gene_feat_full

        if self.mode == "multiome":
            C = len(self.cre_node_names)
            cre_feat_full = torch.zeros((C, cre_feats.shape[1]), dtype=torch.float32)
            if cre_feats.numel() > 0:
                cre_feat_full[self.cre_dst_nodes] = cre_feats
            graph["cre"].x = cre_feat_full


        # 3) prune to active subgraph 
        if self.prune:
            graph = self.prune_graph(graph, col)
            if graph is None:
                return None
        else:
            graph['gene'].kept_idx = torch.arange(len(self.gene_node_names), dtype=torch.long)
            if self.mode == "multiome":
                graph["cre"].kept_idx = torch.arange(len(self.cre_node_names), dtype=torch.long)
        
        # 4) label → compact id
        if self.labels_dict is not None and self.orig_label_to_compact is not None:
            orig_lab = self.labels_dict.get(cell_id, None)
            y_id = self.orig_label_to_compact.get(orig_lab, -1)
            graph.y = torch.tensor([y_id], dtype=torch.long)
        else:
            graph.y = torch.tensor([-1], dtype=torch.long)

        # 5) numeric identifier
        graph.cell_idx = torch.tensor([col], dtype=torch.long)

        # 6) counts
        graph['gene'].num_nodes = graph['gene'].tf_feats.size(0)
        if self.mode == "multiome":
            graph['cre'].num_nodes  = graph['cre'].x.size(0)

        
        for et in getattr(graph, "edge_types", []):
            store = graph[et]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                store.edge_attr = store.edge_attr.float()

        return graph

    def prune_graph(self, graph, cell_col):
        if self.mode == "multiome":
            return self._prune_graph_multiome(graph, cell_col)
        else:
            return self._prune_graph_rna_only(graph, cell_col)

    # ---------------- multiome prune ----------------
    def _prune_graph_multiome(self, graph, cell_col):
        ei_g2c = graph["gene", "binds", "cre"].edge_index
        ei_c2g = graph["cre", "regulates", "gene"].edge_index
        g_src, c_dst1 = ei_g2c
        c_src2, g_dst = ei_c2g

        tf_raw_mtx, _, _ = self.raw_expr_matrices["tf_raw"]
        gene_raw_mtx, _, _ = self.raw_expr_matrices["gene_raw"]
        cre_raw_mtx, _, _ = self.raw_expr_matrices["cre_raw"]

        tf_expr = tf_raw_mtx[self.tf_src_rows, cell_col].toarray().ravel()
        gene_expr = gene_raw_mtx[self.gene_src_rows, cell_col].toarray().ravel()
        cre_counts = cre_raw_mtx[self.cre_src_rows, cell_col].toarray().ravel()

        tf_nonzero = torch.zeros(len(self.gene_node_names), dtype=torch.bool)
        gene_nonzero = torch.zeros(len(self.gene_node_names), dtype=torch.bool)
        cre_nonzero = torch.zeros(len(self.cre_node_names), dtype=torch.bool)

        tf_nonzero[self.tf_dst_nodes] = torch.from_numpy(tf_expr > 0)
        gene_nonzero[self.gene_dst_nodes] = torch.from_numpy(gene_expr > 0)
        cre_nonzero[self.cre_dst_nodes] = torch.from_numpy(cre_counts > 0)

        keep_tf_src = torch.zeros_like(tf_nonzero); keep_tf_src[g_src] = True
        keep_gene_tgt = torch.zeros_like(gene_nonzero); keep_gene_tgt[g_dst] = True
        keep_gene = (keep_tf_src & tf_nonzero) | (keep_gene_tgt & gene_nonzero)

        cre_in_binds = torch.zeros_like(cre_nonzero); cre_in_binds[c_dst1] = True
        cre_in_regs = torch.zeros_like(cre_nonzero); cre_in_regs[c_src2] = True
        keep_cre = cre_nonzero & cre_in_binds & cre_in_regs

        kept_gene_idx = torch.where(keep_gene)[0]
        kept_cre_idx = torch.where(keep_cre)[0]
        if kept_gene_idx.numel() == 0 or kept_cre_idx.numel() == 0:
            return None

        graph["gene"].kept_idx = kept_gene_idx.clone().long()
        graph["cre"].kept_idx = kept_cre_idx.clone().long()

 
        gene_map = -torch.ones(len(self.gene_node_names), dtype=torch.long)
        cre_map = -torch.ones(len(self.cre_node_names), dtype=torch.long)
        gene_map[kept_gene_idx] = torch.arange(kept_gene_idx.numel(), dtype=torch.long)
        cre_map[kept_cre_idx] = torch.arange(kept_cre_idx.numel(), dtype=torch.long)

        # slice features
        graph["gene"].tf_feats = graph["gene"].tf_feats[kept_gene_idx]
        graph["gene"].gene_feats = graph["gene"].gene_feats[kept_gene_idx]
        graph["cre"].x = graph["cre"].x[kept_cre_idx]

        # remap edges (gene->cre)
        new_g_src = gene_map[ei_g2c[0]]
        new_c_dst = cre_map[ei_g2c[1]]
        mask_g2c = (new_g_src >= 0) & (new_c_dst >= 0)
        if mask_g2c.sum() == 0:
            return None
        graph["gene", "binds", "cre"].edge_index = torch.stack([new_g_src[mask_g2c], new_c_dst[mask_g2c]], dim=0)
        if hasattr(graph["gene", "binds", "cre"], "edge_attr") and graph["gene", "binds", "cre"].edge_attr is not None:
            graph["gene", "binds", "cre"].edge_attr = graph["gene", "binds", "cre"].edge_attr[mask_g2c].float()

        # remap edges (cre->gene)
        new_c_src = cre_map[ei_c2g[0]]
        new_g_dst = gene_map[ei_c2g[1]]
        mask_c2g = (new_c_src >= 0) & (new_g_dst >= 0)
        if mask_c2g.sum() == 0:
            return None
        graph["cre", "regulates", "gene"].edge_index = torch.stack([new_c_src[mask_c2g], new_g_dst[mask_c2g]], dim=0)
        if hasattr(graph["cre", "regulates", "gene"], "edge_attr") and graph["cre", "regulates", "gene"].edge_attr is not None:
            graph["cre", "regulates", "gene"].edge_attr = graph["cre", "regulates", "gene"].edge_attr[mask_c2g].float()

        # role masks after pruning
        N_gene = graph["gene"].tf_feats.size(0)
        outdeg = torch.bincount(graph[("gene", "binds", "cre")].edge_index[0], minlength=N_gene)
        indeg = torch.bincount(graph[("cre", "regulates", "gene")].edge_index[1], minlength=N_gene)

        tf_nz_kept = tf_nonzero[kept_gene_idx]
        gene_nz_kept = gene_nonzero[kept_gene_idx]

        is_tf = (outdeg > 0) & tf_nz_kept
        is_target = (indeg > 0) & gene_nz_kept

        graph["gene"].is_tf = is_tf
        graph["gene"].is_target = is_target

        role_id = torch.zeros(N_gene, dtype=torch.long)
        role_id[is_tf & ~is_target] = 1
        role_id[is_tf & is_target] = 2
        graph["gene"].role_id = role_id

        graph["gene"].num_nodes = N_gene
        graph["cre"].num_nodes = graph["cre"].x.size(0)
        return graph

    # ---------------- rna_only prune ----------------
    def _prune_graph_rna_only(self, graph, cell_col):
        ei = graph["gene", "regulates", "gene"].edge_index
        src, dst = ei[0], ei[1]

        tf_raw_mtx, _, _ = self.raw_expr_matrices["tf_raw"]
        gene_raw_mtx, _, _ = self.raw_expr_matrices["gene_raw"]

        tf_expr = tf_raw_mtx[self.tf_src_rows, cell_col].toarray().ravel()
        gene_expr = gene_raw_mtx[self.gene_src_rows, cell_col].toarray().ravel()

        G = len(self.gene_node_names)

        tf_nonzero = torch.zeros(G, dtype=torch.bool)
        gene_nonzero = torch.zeros(G, dtype=torch.bool)
        tf_nonzero[self.tf_dst_nodes] = torch.from_numpy(tf_expr > 0)
        gene_nonzero[self.gene_dst_nodes] = torch.from_numpy(gene_expr > 0)

        # keep genes that (a) are expressed and participate in at least one edge
        has_out = torch.zeros(G, dtype=torch.bool); has_out[src] = True
        has_in = torch.zeros(G, dtype=torch.bool); has_in[dst] = True
        keep_gene = (has_out & tf_nonzero) | (has_in & gene_nonzero)


        kept_gene_idx = torch.where(keep_gene)[0]
        if kept_gene_idx.numel() == 0:
            return None

        graph["gene"].kept_idx = kept_gene_idx.clone().long()

        gene_map = -torch.ones(G, dtype=torch.long)
        gene_map[kept_gene_idx] = torch.arange(kept_gene_idx.numel(), dtype=torch.long)

        # slice features
        graph["gene"].tf_feats = graph["gene"].tf_feats[kept_gene_idx]
        graph["gene"].gene_feats = graph["gene"].gene_feats[kept_gene_idx]

        # remap edges (gene->gene)
        new_src = gene_map[src]
        new_dst = gene_map[dst]
        mask_e = (new_src >= 0) & (new_dst >= 0)
        if mask_e.sum() == 0:
            return None

        graph["gene", "regulates", "gene"].edge_index = torch.stack([new_src[mask_e], new_dst[mask_e]], dim=0)
        if hasattr(graph["gene", "regulates", "gene"], "edge_attr") and graph["gene", "regulates", "gene"].edge_attr is not None:
            graph["gene", "regulates", "gene"].edge_attr = graph["gene", "regulates", "gene"].edge_attr[mask_e].float()

        # role masks after pruning 
        N_gene = graph["gene"].tf_feats.size(0)
        outdeg = torch.bincount(graph[("gene", "regulates", "gene")].edge_index[0], minlength=N_gene)
        indeg = torch.bincount(graph[("gene", "regulates", "gene")].edge_index[1], minlength=N_gene)

        tf_nz_kept = tf_nonzero[kept_gene_idx]
        gene_nz_kept = gene_nonzero[kept_gene_idx]

        is_tf = (outdeg > 0) & tf_nz_kept
        is_target = (indeg > 0) & gene_nz_kept

        graph["gene"].is_tf = is_tf
        graph["gene"].is_target = is_target

        role_id = torch.zeros(N_gene, dtype=torch.long)
        role_id[is_tf & ~is_target] = 1
        role_id[is_tf & is_target] = 2
        graph["gene"].role_id = role_id

        graph["gene"].num_nodes = N_gene
        return graph
    
    def split(
            self,
            train_frac: float = 0.80,
            test_frac: float = 0.20,
            seed: int = 42,
        ):


            if abs((train_frac + test_frac) - 1.0) > 1e-8:
                raise ValueError("train_frac  + test_frac must sum to 1.0")

            if self.labels_dict is None or self.label_to_id is None:
                raise ValueError("Cannot stratify/split: dataset has no labels_dict/label_to_id.")

            # y aligned with self.valid_cells order
            y_all = np.array(
                [self.label_to_id[self.labels_dict[c]] for c in self.valid_cells],
                dtype=np.int64
            )
            idx = np.arange(len(y_all))

            train_idx, test_idx = train_test_split(
                idx,
                test_size=(1.0 - train_frac),
                stratify=y_all,
                random_state=seed,
            )

            train_ds = Subset(self, train_idx.tolist())
            test_ds  = Subset(self, test_idx.tolist())  
            print("Training subset: " , len(train_ds),"\nTesting subset: " ,len(test_ds))
            return train_ds, test_ds


# ============================================================
# 4) Manager: user gives only feature_dir + labels_path
# ============================================================

@dataclass
class GRNDatasetPaths:
    experiment_dir: str
    tf_peak_scores_csv: Optional[str] = None
    peaks_genes_csv: Optional[str] = None

    plots_dir: str = "plots"
    cache_dir: str = "per_cell_graphs"

    label_barcode_col: str = "barcode"
    label_value_col: str = "label"


class GRNDatasetManager:
    def __init__(self, paths: GRNDatasetPaths):
        self.paths = paths
        os.makedirs(self.paths.plots_dir, exist_ok=True)
        self.mode: Optional[str] = None

        self.base_graph: Optional[HeteroData] = None
        self.all_genes: Optional[List[str]] = None
        self.all_cres: Optional[List[str]] = None

        self.feature_files: Optional[AutoDiscoveredFiles] = None
        self.mats: Optional[Dict[str, Tuple[Any, List[str], List[str]]]] = None
        self.common_barcodes: Optional[List[str]] = None

        self.labels_df: Optional[pd.DataFrame] = None
        self.labels_dict: Optional[Dict[str, str]] = None
        self.label_to_id: Optional[Dict[str, int]] = None


    # ---------- step 1: base GRN ----------
    def _discover_csvs(self) -> None:
        csv_dir = os.path.join(self.paths.experiment_dir, "interm", "csv")
    
        if not os.path.isdir(csv_dir):
            raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    
        def _find_file(patterns: List[str], name: str) -> str:
            for pat in patterns:
                hits = glob.glob(os.path.join(csv_dir, pat))
                if hits:
                    return sorted(hits)[0]
            raise FileNotFoundError(f"No match in {csv_dir}: {patterns}")
    
        if self.paths.tf_peak_scores_csv is None:
            self.paths.tf_peak_scores_csv = _find_file(["tf_peak_scores*.csv", "tf*peaks*.csv"],"TF peak scores CSV")
    
        if self.paths.peaks_genes_csv is None:
            self.paths.peaks_genes_csv = _find_file(["peaks_genes*.csv", "peaks*genes*.csv"], "Peaks-Genes CSV")

        if not hasattr(self.paths, 'labels_csv') or self.paths.labels_csv is None:
            self.paths.labels_csv = _find_file(
                ["*labels*.csv", "*cell_type*.csv", "*annotation*.csv", "predicted*.csv"],
                "Labels CSV"
            )

    def build_base_grn(self, mode: str = "multiome", save_base_graph_pt: Optional[str] = None) -> HeteroData:
        assert mode in ("multiome", "rna_only")
        self._discover_csvs()
        self.mode = mode
        cfg = BaseGRNConfig(
            tf_peak_scores_csv=self.paths.tf_peak_scores_csv,
            peaks_genes_csv=self.paths.peaks_genes_csv,
            out_plot_dir=os.path.join(self.paths.plots_dir, "base_grn"),
            mode=mode,
        )
        grn = BaseGRN(cfg)
        self.base_graph = grn.create_base_grn(save_graph_pt=save_base_graph_pt)

        self.all_genes = list(self.base_graph["gene"].names)
        if "cre" in self.base_graph.node_types:
            self.all_cres = list(self.base_graph["cre"].names)
        else:
            self.all_cres = []
        
        return self.base_graph

    # ---------- step 2: discover + load feature matrices ----------
    def load_feature_dir(self) -> None:
        if self.mode is None:
            raise RuntimeError("Call mgr.build_base_grn(mode=...) first so mode is known.")
        loader = AutoFeatureLoader(self.paths.experiment_dir)
        self.feature_files = loader.discover(mode=self.mode)
        self.mats = loader.load_all(self.feature_files,mode=self.mode)

        self.common_barcodes = self.mats["tf_feat"][2]
        print(f"[Feature] Loaded matrices for {len(self.common_barcodes)} barcodes (from barcodes.txt)")

    # ---------- step 3: inspect label distribution BEFORE building dataset ----------
    def inspect_cell_distribution(
        self,
        min_cells_per_class: Optional[int] = None,
        max_cells_per_class: Optional[int] = None,
        save_plot: bool = True,
        plot_dir: Optional[str] = None
    ) -> pd.Series:
        """
        Reads labels_csv and intersects with feature barcodes.
        Returns counts per class (after optional min/max filtering preview).
        """
        if self.common_barcodes is None:
            raise RuntimeError("Run load_feature_dir() first (need barcodes).")

        self._discover_csvs()
        df = pd.read_csv(self.paths.labels_csv)
        bc_col = self.paths.label_barcode_col
        y_col  = self.paths.label_value_col
        if bc_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"labels_csv must contain columns: '{bc_col}', '{y_col}'")

        df[bc_col] = df[bc_col].astype(str)
        df[y_col]  = df[y_col].astype(str)

        # keep only barcodes present in matrices
        in_feat = set(self.common_barcodes)
        df = df[df[bc_col].isin(in_feat)].copy()

        self.labels_df = df
        self.labels_dict = dict(zip(df[bc_col], df[y_col]))
        labs = sorted(set(self.labels_dict.values()))
        self.label_to_id = {lab: i for i, lab in enumerate(labs)}

        counts = df[y_col].value_counts()

        if min_cells_per_class is not None:
            counts = counts[counts >= min_cells_per_class]
        if max_cells_per_class is not None:
            counts = counts[counts <= max_cells_per_class]

        print("\n[Cell distribution] (after intersect with feature barcodes)")

        if save_plot:
            out_dir = plot_dir  or os.path.join(self.paths.plots_dir, "dataset")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(counts.index.tolist(), counts.values)
            ax.set_title("Cell Type Distribution (intersected)")
            ax.set_ylabel("Count")
            ax.set_xticklabels(counts.index.tolist(), rotation=45, ha="right")
            fig.tight_layout()
            p = os.path.join(out_dir, "cell_type_distribution.png")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[inspect_plot] Saved -> {os.path.abspath(p)}")

        return counts

    # ---------- step 4: build dataset ----------
    def build_per_cell_dataset(self, prune: bool = True,
                               min_cells_per_class: Optional[int] = 50,
                               max_cells_per_class: Optional[int] = None) -> PerCellGraphDataset:
        if self.mode is None:
            raise RuntimeError("Call build_base_grn(mode=...) first.")
        if self.base_graph is None:
            raise RuntimeError("Run build_base_grn() first.")
        if self.mats is None or self.common_barcodes is None:
            raise RuntimeError("Run load_feature_dir() first.")
        if self.paths.labels_csv is not None and self.labels_dict is None:
            self.inspect_cell_distribution(save_plot=False)

        # unpack matrices
        tf_mat, tf_names, _ = self.mats["tf_feat"]
        gene_mat, gene_names, _ = self.mats["gene_feat"] 
        cells = self.common_barcodes
        raw_expr_matrices = self.mats if prune else None

        if self.mode == "multiome":
            cre_mat, cre_names, _ = self.mats["cre_feat"]
            cre_mtx_dict = {"cre_feat": cre_mat}
            cre_names_arg = cre_names
            cre_node_names = self.all_cres
        
        else:
            cre_mtx_dict = {}
            cre_names_arg = []
            cre_node_names = []


        ds = PerCellGraphDataset(
            base_graph=self.base_graph,
            cells=cells,
            tf_mtx_dict={"tf_feat": tf_mat}, tf_names=tf_names,
            gene_mtx_dict={"gene_feat": gene_mat}, gene_names=gene_names,
            cre_mtx_dict=cre_mtx_dict, cre_names=cre_names_arg,
            gene_node_names=self.all_genes,
            cre_node_names=cre_node_names,
            mode=self.mode,
            labels_dict=self.labels_dict,
            label_to_id=self.label_to_id,
            prune=prune,
            raw_expr_matrices=raw_expr_matrices,
            min_cells_per_class=min_cells_per_class,
            max_cells_per_class=max_cells_per_class
        )

        return ds
    
    # ---------- step 5: summarize dataset ----------
    def summarize_dataset(self, dataset, max_graphs=None, plot_dir=None, dpi=200):
        if plot_dir is None:
            plot_dir = os.path.join(self.paths.plots_dir, "dataset_summary")
        os.makedirs(plot_dir, exist_ok=True)

        ntotal = len(dataset)
        n = ntotal if max_graphs is None else min(ntotal, int(max_graphs))

        print("\nDataset Summary")
        print("=" * 60)
        print(f"Total cells/graphs: {ntotal}")
        if max_graphs is not None:
            print(f"Summarizing only:   {n}")
        print(f"Plots will be saved in: {os.path.abspath(plot_dir)}")

        base_ds = dataset
        subset_indices = None
        if isinstance(dataset, Subset):
            base_ds = dataset.dataset
            subset_indices = list(dataset.indices)

        if hasattr(base_ds, "labels_dict") and base_ds.labels_dict is not None and hasattr(base_ds, "valid_cells"):
            if subset_indices is None:
                cids = base_ds.valid_cells[:n]
            else:
                cids = [base_ds.valid_cells[i] for i in subset_indices[:n]]

            labels = [base_ds.labels_dict.get(cid, "Unknown") for cid in cids]
            counts = pd.Series(labels).value_counts()

            print("\n[Labels] Distribution (over summarized graphs):")
            print(counts)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(counts.index.tolist(), counts.values)
            ax.set_title("Cell Type Distribution")
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index.tolist(), rotation=45, ha="right")
            fig.tight_layout()
            out = os.path.join(plot_dir, "label_distribution.png")
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out}")

        gene_node_counts = []
        cre_node_counts = []

        tf_edge_counts = []       
        target_edge_counts = []   
        g2g_edge_counts = []      # only exists in rna_only graphs

        tf_node_counts = []
        target_node_counts = []
        autoreg_tf_counts = []

        tf_feat_dim = None
        gene_feat_dim = None
        cre_feat_dim = None

        for i in range(n):
            g = dataset[i]

            gene_store = g["gene"]
            tf_feats = getattr(gene_store, "tf_feats", None)
            gene_feats = getattr(gene_store, "gene_feats", None)

            if tf_feats is None or gene_feats is None:
                raise RuntimeError(
                    "Expected g['gene'].tf_feats and g['gene'].gene_feats to exist. "
                    "Check PerCellGraphDataset.build_graph()."
                )

            if tf_feat_dim is None:
                tf_feat_dim = int(tf_feats.shape[1])
            if gene_feat_dim is None:
                gene_feat_dim = int(gene_feats.shape[1])

            gene_node_counts.append(int(tf_feats.shape[0]))

            has_cre = ("cre" in getattr(g, "node_types", [])) and hasattr(g["cre"], "x")
            if has_cre:
                cre_x = g["cre"].x
                cre_node_counts.append(int(cre_x.shape[0]))
                if cre_feat_dim is None:
                    cre_feat_dim = int(cre_x.shape[1])
            else:
                cre_node_counts.append(0)

            edge_types = set(getattr(g, "edge_types", []))

            if ("gene", "binds", "cre") in edge_types:
                e_g2c = g[("gene", "binds", "cre")].edge_index
                tf_edge_counts.append(int(e_g2c.shape[1]))
                tf_nodes = torch.unique(e_g2c[0]) if e_g2c.numel() else torch.empty((0,), dtype=torch.long)
            else:
                if ("gene", "regulates", "gene") in edge_types:
                    e_g2g = g[("gene", "regulates", "gene")].edge_index
                    tf_edge_counts.append(int(e_g2g.shape[1]))
                    tf_nodes = torch.unique(e_g2g[0]) if e_g2g.numel() else torch.empty((0,), dtype=torch.long)
                else:
                    tf_edge_counts.append(0)
                    tf_nodes = torch.empty((0,), dtype=torch.long)

            if ("cre", "regulates", "gene") in edge_types:
                e_c2g = g[("cre", "regulates", "gene")].edge_index
                target_edge_counts.append(int(e_c2g.shape[1]))
                tgt_nodes = torch.unique(e_c2g[1]) if e_c2g.numel() else torch.empty((0,), dtype=torch.long)
            else:
                if ("gene", "regulates", "gene") in edge_types:
                    e_g2g = g[("gene", "regulates", "gene")].edge_index
                    target_edge_counts.append(int(e_g2g.shape[1]))
                    tgt_nodes = torch.unique(e_g2g[1]) if e_g2g.numel() else torch.empty((0,), dtype=torch.long)
                else:
                    target_edge_counts.append(0)
                    tgt_nodes = torch.empty((0,), dtype=torch.long)

            if ("gene", "regulates", "gene") in edge_types:
                g2g_edge_counts.append(int(g[("gene", "regulates", "gene")].edge_index.shape[1]))
            else:
                g2g_edge_counts.append(0)

            tf_node_counts.append(int(tf_nodes.numel()))
            target_node_counts.append(int(tgt_nodes.numel()))

            if hasattr(gene_store, "role_id") and gene_store.role_id is not None:
                autoreg_tf_counts.append(int((gene_store.role_id == 2).sum().item()))
            else:
                autoreg_tf_counts.append(0)

        def _save_hist(data, title, xlabel, fname):
            if len(data) == 0:
                return
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(data, bins=20, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Graph Count")
            fig.tight_layout()
            out = os.path.join(plot_dir, fname)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out}")

        _save_hist(gene_node_counts, "Gene node count per graph", "Genes", "gene_nodes_hist.png")
        _save_hist(tf_node_counts, "TF source node count per graph", "TF/Regulators", "tf_nodes_hist.png")
        _save_hist(target_node_counts, "Target node count per graph", "Targets", "target_nodes_hist.png")
        _save_hist(autoreg_tf_counts, "Autoregulated TFs per graph", "Autoreg TFs", "autoreg_tfs_hist.png")

        if any(c > 0 for c in cre_node_counts):
            _save_hist(cre_node_counts, "CRE node count per graph", "CREs", "cre_nodes_hist.png")
            _save_hist(tf_edge_counts, "TF→CRE edge count per graph", "TF→CRE edges", "tf_cre_edges_hist.png")
            _save_hist(target_edge_counts, "CRE→Gene edge count per graph", "CRE→Gene edges", "cre_gene_edges_hist.png")
        else:
            if any(e > 0 for e in g2g_edge_counts):
                _save_hist(g2g_edge_counts, "Gene→Gene edge count per graph", "Gene→Gene edges", "gene_gene_edges_hist.png")

        print("\n[Stats] Averages over summarized graphs:")
        print(f"  Avg gene nodes:          {np.mean(gene_node_counts):.1f}")
        if any(c > 0 for c in cre_node_counts):
            print(f"  Avg CRE nodes:           {np.mean(cre_node_counts):.1f}")
            print(f"  Avg TF→CRE edges:        {np.mean(tf_edge_counts):.1f}")
            print(f"  Avg CRE→Gene edges:      {np.mean(target_edge_counts):.1f}")
        else:
            if any(e > 0 for e in g2g_edge_counts):
                print(f"  Avg Gene→Gene edges:     {np.mean(g2g_edge_counts):.1f}")
            print(f"  Avg regulator nodes:     {np.mean(tf_node_counts):.1f}")
            print(f"  Avg target nodes:        {np.mean(target_node_counts):.1f}")

        print("\n[Dims] Feature dimensionality:")
        print(f"  TF feature dim:          {tf_feat_dim if tf_feat_dim is not None else 'NA'}")
        print(f"  Gene feature dim:        {gene_feat_dim if gene_feat_dim is not None else 'NA'}")
        if any(c > 0 for c in cre_node_counts):
            print(f"  CRE feature dim:         {cre_feat_dim if cre_feat_dim is not None else 'NA'}")
