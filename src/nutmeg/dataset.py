from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split





def _read(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require(df, columns, source):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def _ids(df, columns, source):
    for col in columns:
        df[col] = df[col].str.strip()
        if df[col].eq("").any():
            raise ValueError(f"{source}: empty {col}")


def _direction(value):
    token = str(value).strip().lower()
    if token in {"true", "1", "t"}:
        return True
    if token in {"false", "0", "f"}:
        return False
    raise ValueError(f"Invalid is_directional: {value!r}")


def _features(names, values, context):

    names = str(names).strip()
    values = str(values).strip()
    try:
        ns = json.loads(names) if names.startswith("[") else [names]
        vs = json.loads(values) if values.startswith("[") else [float(values)]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid JSON feature list: {context}") from exc
    if not isinstance(ns, list) or not isinstance(vs, list) or not ns or len(ns) != len(vs):
        raise ValueError(f"Feature lists must be nonempty and equal length: {context}")
    if any(not isinstance(n, str) or not n.strip() for n in ns):
        raise ValueError(f"Feature names must be nonempty strings: {context}")
    if len(set(ns)) != len(ns):
        raise ValueError(f"Duplicate feature name: {context}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vs):
        raise ValueError(f"Feature values must be numeric: {context}")
    arr = np.asarray(vs, dtype=np.float64)
    if not np.isfinite(arr).all() or (np.abs(arr) > np.finfo(np.float32).max).any():
        raise ValueError(f"Nonfinite or out-of-range feature value: {context}")
    return ns, arr.astype(np.float32).tolist()


def _nodes(path):
    df = _read(path)
    _require(df, ["node_id", "node_type", "node_subtype", "feature_name", "feature_value", "metadata_id"], "node.csv")
    if df.empty:
        raise ValueError("node.csv is empty")
    _ids(df, ["node_id", "node_type", "metadata_id"], "node.csv")
    df["node_subtype"] = df["node_subtype"].str.strip()
    if df.duplicated(["metadata_id", "node_id"]).any():
        raise ValueError("Duplicate (metadata_id, node_id): use one row with feature lists")
    for col in ["node_type", "node_subtype"]:
        if (df.groupby("node_id")[col].nunique() > 1).any():
            raise ValueError(f"A node_id has conflicting {col} annotations")
    schemas = {}
    vectors = []
    for row in df.itertuples(index=False):
        ns, vs = _features(row.feature_name, row.feature_value, f"node={row.node_id}, metadata_id={row.metadata_id}")
        if row.node_type in schemas and schemas[row.node_type] != ns:
            raise ValueError(f"Inconsistent feature names/order for node type {row.node_type!r}; expected {schemas[row.node_type]}, got {ns}")
        schemas[row.node_type] = ns
        vectors.append(vs)
    df["_feature_vector"] = vectors
    annotation = df.drop_duplicates("node_id").set_index("node_id")
    return df, schemas, annotation["node_type"].to_dict(), annotation["node_subtype"].to_dict()


@dataclass
class BaseGraphConfig:
    edge_csv: str
    node_csv: str
    out_plot_dir: str = "plots/base_graph"
    plot_filename: str = "base_graph_degree_distributions.png"


class BaseGraph:
    def __init__(self, cfg):
        self.cfg = cfg
        self.graph = None
        self.edge_df = None
        self.node_df = None

    def _load(self):
        self.node_df, self.feature_names_by_type, self.node_type, self.node_subtype = _nodes(self.cfg.node_csv)
        self.edge_df = _read(self.cfg.edge_csv)
        _require(self.edge_df, ["left_node", "right_node", "is_directional"], "edge.csv")
        _ids(self.edge_df, ["left_node", "right_node"], "edge.csv")
        self.edge_df["is_directional"] = self.edge_df["is_directional"].map(_direction)
        endpoints = set(self.edge_df["left_node"]) | set(self.edge_df["right_node"])
        missing = endpoints - set(self.node_type)
        if missing:
            raise ValueError(f"Edge endpoints absent from node.csv: {sorted(missing)[:10]}")

    def _build_node_indices(self):
        self.node_types = sorted(set(self.node_type.values()))
        self.node_names = {t: sorted(n for n in self.node_type if self.node_type[n] == t) for t in self.node_types}
        self.node_to_local_id = {t: {n:i for i,n in enumerate(ns)} for t,ns in self.node_names.items()}
        names = [n for t in self.node_types for n in self.node_names[t]]
        self.node_to_global_id = {n:i for i,n in enumerate(names)}
        self.global_id_to_node = {i:(self.node_type[n],n) for n,i in self.node_to_global_id.items()}

    def _build_graph(self):
        g = HeteroData()
        for t, names in self.node_names.items():
            g[t].num_nodes = len(names)
            g[t].names = names
            g[t].global_id = torch.tensor([self.node_to_global_id[n] for n in names], dtype=torch.long)
        pairs = set()
        for source,target,directional in self.edge_df[["left_node","right_node","is_directional"]].itertuples(index=False, name=None):
            pairs.add((source,target))
            if not directional:
                pairs.add((target,source))
        self.edge_to_global_id = {pair:i for i,pair in enumerate(sorted(pairs))}
        self.global_id_to_edge = {i:p for p,i in self.edge_to_global_id.items()}
        groups = defaultdict(list)
        for source,target in sorted(pairs):
            groups[(self.node_type[source],"edge",self.node_type[target])].append((source,target))
        for e, edges in groups.items():
            g[e].edge_index = torch.tensor([[self.node_to_local_id[e[0]][s] for s,t in edges],
                                           [self.node_to_local_id[e[2]][t] for s,t in edges]], dtype=torch.long)
            g[e].edge_attr = torch.empty((len(edges),0), dtype=torch.float32)
            g[e].global_edge_id = torch.tensor([self.edge_to_global_id[p] for p in edges], dtype=torch.long)
        return g

    def create_base_graph(self, save_graph_pt=None):
        self._load()
        self._build_node_indices()
        self.graph = self._build_graph()
        self.save_degree_plot()
        if save_graph_pt:
            Path(save_graph_pt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.graph, save_graph_pt)
        return self.graph, self.edge_df, self.node_type, self.node_to_global_id, self.edge_to_global_id

    def save_degree_plot(self):
        import matplotlib.pyplot as plt
        if self.graph is None:
            raise RuntimeError("Call create_base_graph first")
        path = Path(self.cfg.out_plot_dir)/self.cfg.plot_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        edges = list(self.graph.edge_types)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(range(len(edges)), [self.graph[e].edge_index.shape[1] for e in edges])
        ax.set_xticks(range(len(edges)))
        ax.set_xticklabels([" -> ".join(e) for e in edges], rotation=45, ha="right")
        ax.set_ylabel("Number of edges")
        ax.set_title("Edge counts by edge type")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)


@dataclass
class UnifiedFiles:
    edge_csv: str
    node_csv: str
    metadata_csv: str


class UnifiedLoader:
    @staticmethod
    def validate_files(files):
        for path in [files.edge_csv,files.node_csv,files.metadata_csv]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    def load(self, files, *, node_df=None, feature_names_by_type=None, edge_df=None):
        self.validate_files(files)
        if node_df is None:
            node_df, feature_names_by_type, _, _ = _nodes(files.node_csv)
        metadata = _read(files.metadata_csv)
        _require(metadata,["sample_id","cell_id","label"],"metadata.csv")
        if metadata.empty:
            raise ValueError("metadata.csv is empty")
        for c in ["sample_id","cell_id","label"]:
            metadata[c] = metadata[c].str.strip()
        if metadata["label"].eq("").any():
            raise ValueError("Missing graph labels; supervised dataset requires labels")
        valid_sample = metadata["sample_id"].ne("")
        per_sample = metadata.loc[valid_sample].groupby("sample_id")["cell_id"].nunique()
        level = "cell" if (per_sample > 1).any() or not valid_sample.all() else "sample"
        if level == "cell":
            if metadata["cell_id"].eq("").any() or metadata["cell_id"].duplicated().any():
                raise ValueError("Cell graphs require nonempty, globally unique cell_id values")
            observations = metadata["cell_id"].tolist()
            lookup = dict(zip(observations,observations))
            label_map = dict(zip(observations,metadata["label"]))
        else:
            if metadata["sample_id"].duplicated().any():
                raise ValueError("Sample graphs require exactly one metadata row per sample_id")
            observations = metadata["sample_id"].tolist()

            keys = metadata["cell_id"].where(metadata["cell_id"].ne(""),metadata["sample_id"])
            if keys.duplicated().any():
                raise ValueError("Ambiguous metadata IDs across samples")
            lookup = dict(zip(keys,observations))
            label_map = dict(zip(observations,metadata["label"]))
        unknown = set(node_df["metadata_id"]) - set(lookup)
        if unknown:
            raise ValueError(f"node.csv metadata_id not found in metadata.csv: {sorted(unknown)[:10]}")
        node_observations = {o:{} for o in observations}
        feature_values = {o:{} for o in observations}
        for row in node_df.itertuples(index=False, name=None):
            values = dict(zip(node_df.columns,row))
            o = lookup[values["metadata_id"]]
            t,n = values["node_type"],values["node_id"]
            node_observations[o].setdefault(t,[]).append(n)
            feature_values[o].setdefault(t,{})[n] = values["_feature_vector"]
        empty = [o for o in observations if not node_observations[o]]
        if empty:
            raise ValueError(f"Observations with no nodes: {empty[:10]}")
        for by_type in node_observations.values():
            for t in by_type:
                by_type[t].sort()
        return dict(edge_df=edge_df,metadata=metadata,graph_level=level,
                    observation_ids=observations,labels_dict=label_map,
                    node_observations=node_observations,feature_values=feature_values,
                    feature_names_by_type=feature_names_by_type)


class GeneralGraphDataset(Dataset):
    def __init__(self, base_graph, edge_df, node_type, node_to_global_id,
                 edge_to_global_id, observation_ids, node_observations, feature_values,
                 labels_dict=None, label_to_id=None, min_obs_per_class=None,
                 max_obs_per_class=None, *, feature_names_by_type,
                 node_subtype=None, metadata=None, graph_level=None):
        self.base_graph,self.edge_df,self.node_type = base_graph,edge_df,node_type
        self.node_to_global_id,self.edge_to_global_id = node_to_global_id,edge_to_global_id
        self.global_id_to_node = {i:(node_type[n],n) for n,i in node_to_global_id.items()}
        self.global_id_to_edge = {i:p for p,i in edge_to_global_id.items()}
        self.feature_names_by_type = feature_names_by_type
        self.input_dims = {t:len(ns) for t,ns in feature_names_by_type.items()}
        self.node_subtype = node_subtype or {}
        self.metadata,self.graph_level = metadata,graph_level
        self.observation_ids = list(observation_ids)
        self.observation_to_index = {o:i for i,o in enumerate(observation_ids)}
        self.node_observations,self.feature_values = node_observations,feature_values
        self.labels_dict,self.label_to_id,self.id_to_label = None,None,None
        kept = self.observation_ids
        if labels_dict is not None:
            counts = pd.Series([labels_dict[o] for o in observation_ids]).value_counts()
            valid = {lab for lab,count in counts.items()
                     if (min_obs_per_class is None or count >= min_obs_per_class)
                     and (max_obs_per_class is None or count <= max_obs_per_class)}
            kept = [o for o in observation_ids if labels_dict[o] in valid]
            self.labels_dict = {o:labels_dict[o] for o in kept}
            # Contiguous labels are rebuilt after whole-class filtering.
            self.label_to_id = {lab:i for i,lab in enumerate(sorted(valid))}
            self.id_to_label = {i:lab for lab,i in self.label_to_id.items()}
        if not kept:
            raise ValueError("No observations remain after label filtering")
        self.valid_observation_ids = list(kept)
        self._edge_groups = defaultdict(list)
        for (s,t),gid in edge_to_global_id.items():
            self._edge_groups[(node_type[s],"edge",node_type[t])].append((s,t,gid))

    def __len__(self):
        return len(self.valid_observation_ids)

    def __getitem__(self,idx):
        return self.build_graph(self.valid_observation_ids[idx])

    def build_graph(self,observation_id):
        if observation_id not in self.node_observations:
            raise KeyError(observation_id)
        g = HeteroData()
        local = {}
        observed = self.node_observations[observation_id]
        for t in self.base_graph.node_types:
            names = sorted(observed.get(t,[]))
            local[t] = {n:i for i,n in enumerate(names)}
            g[t].num_nodes = len(names)
            g[t].names = names
            vectors = [self.feature_values[observation_id][t][n] for n in names]
            g[t].x = torch.tensor(vectors,dtype=torch.float32).reshape(len(names),self.input_dims[t])
            g[t].global_id = torch.tensor([self.node_to_global_id[n] for n in names],dtype=torch.long)
        for e in self.base_graph.edge_types:
            edges = [(local[e[0]][s],local[e[2]][t],gid)
                     for s,t,gid in self._edge_groups[e]
                     if s in local[e[0]] and t in local[e[2]]]
            g[e].edge_index = torch.tensor([[s for s,t,i in edges],[t for s,t,i in edges]],dtype=torch.long)
            g[e].edge_attr = torch.empty((len(edges),0),dtype=torch.float32)
            g[e].global_edge_id = torch.tensor([i for s,t,i in edges],dtype=torch.long)

        label = -1 if self.labels_dict is None else self.label_to_id[self.labels_dict[observation_id]]
        g.y = torch.tensor([label],dtype=torch.long)
        g.observation_idx = torch.tensor([self.observation_to_index[observation_id]],dtype=torch.long)
        return g

    def split(self,train_frac=0.80,test_frac=0.20,seed=42):
        return dataset_split(self,train_frac,test_frac,seed)


def build_graph_dataset(
    edge_csv,
    node_csv,
    metadata_csv,
    min_obs_per_class=None,
    max_obs_per_class=None,
    save_base_graph_pt=None,
    plots_dir="plots"
):
    files = UnifiedFiles(
        edge_csv,
        node_csv,
        metadata_csv
    )

    UnifiedLoader.validate_files(files)

    builder = BaseGraph(
        BaseGraphConfig(
            edge_csv,
            node_csv,
            str(Path(plots_dir) / "base_graph")
        )
    )

    (
        base,
        edges,
        types,
        node_ids,
        edge_ids
    ) = builder.create_base_graph(
        save_base_graph_pt
    )

    data = UnifiedLoader().load(
        files,
        node_df=builder.node_df,
        feature_names_by_type=(
            builder.feature_names_by_type
        ),
        edge_df=edges
    )

    ds = GeneralGraphDataset(
        base,
        edges,
        types,
        node_ids,
        edge_ids,
        data["observation_ids"],
        data["node_observations"],
        data["feature_values"],
        labels_dict=data["labels_dict"],
        min_obs_per_class=min_obs_per_class,
        max_obs_per_class=max_obs_per_class,
        feature_names_by_type=(
            builder.feature_names_by_type
        ),
        node_subtype=builder.node_subtype,
        metadata=data["metadata"],
        graph_level=data["graph_level"]
    )

    print(
        f"[build_graph_dataset] Built {len(ds)} "
        f"{ds.graph_level} graphs."
    )

    print(
        f"[build_graph_dataset] Features per node type: "
        f"{ds.input_dims}"
    )

    return ds

def summarize_dataset(
    dataset,
    max_graphs: Optional[int] = None,
    plot_dir: str = "plots/dataset_summary",
    dpi: int = 200
) -> dict:
   

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(
            dataset.indices
        )
    else:
        base_dataset = dataset
        indices = list(
            range(len(dataset))
        )

    if max_graphs is not None:
        indices = indices[
            :int(max_graphs)
        ]

    os.makedirs(
        plot_dir,
        exist_ok=True
    )

    node_types = list(
        base_dataset.feature_names_by_type.keys()
    )

    edge_types = list(
        base_dataset.base_graph.edge_types
    )

    node_counts = {
        node_type: []
        for node_type in node_types
    }

    edge_counts = {
        edge_type: []
        for edge_type in edge_types
    }

    labels = []

    for idx in indices:
        graph = base_dataset[
            idx
        ]

        label_id = int(
            graph.y.item()
        )

        if label_id >= 0:
            labels.append(
                base_dataset.id_to_label[
                    label_id
                ]
            )
        else:
            labels.append(
                "unknown"
            )

        for node_type in node_types:
            if node_type in graph.node_types:
                node_counts[
                    node_type
                ].append(
                    graph[
                        node_type
                    ].num_nodes
                )
            else:
                node_counts[
                    node_type
                ].append(0)

        for edge_type in edge_types:
            if edge_type in graph.edge_types:
                edge_counts[
                    edge_type
                ].append(
                    graph[
                        edge_type
                    ].edge_index.size(1)
                )
            else:
                edge_counts[
                    edge_type
                ].append(0)

    label_counts = pd.Series(
        labels
    ).value_counts()

    print("Dataset summary")
    print("-" * 60)
    print(
        f"Graph level: "
        f"{base_dataset.graph_level}"
    )
    print(
        f"Total graphs: "
        f"{len(dataset)}"
    )
    print(
        f"Summarized graphs: "
        f"{len(indices)}"
    )

    print("\nLabel distribution:")
    print(label_counts)

    print("\nNode counts:")

    for node_type in node_types:
        values = node_counts[
            node_type
        ]

        if values:
            print(
                f" - {node_type}: "
                f"mean={np.mean(values):.1f}, "
                f"median={np.median(values):.1f}, "
                f"min={min(values)}, "
                f"max={max(values)}"
            )

    print("\nEdge counts:")

    for edge_type in edge_types:
        values = edge_counts[
            edge_type
        ]

        if values:
            print(
                f" - {' -> '.join(edge_type)}: "
                f"mean={np.mean(values):.1f}, "
                f"median={np.median(values):.1f}, "
                f"min={min(values)}, "
                f"max={max(values)}"
            )

    print("\nFeature dimensions:")

    for node_type in node_types:
        print(
            f" - {node_type}: "
            f"{base_dataset.input_dims[node_type]} "
            f"features"
        )

    def save_hist(
        values,
        title,
        xlabel,
        filename
    ):
        if not values:
            return

        fig, ax = plt.subplots(
            figsize=(6, 3.5)
        )

        ax.hist(
            values,
            bins=20,
            edgecolor="black"
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Graph count")

        fig.tight_layout()

        out_path = os.path.join(
            plot_dir,
            filename
        )

        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close(fig)

    for node_type in node_types:
        save_hist(
            node_counts[node_type],
            f"{node_type} node count per graph",
            f"Number of {node_type} nodes",
            f"{node_type}_node_count_histogram.png"
        )

    for edge_type in edge_types:
        edge_name = "_".join(
            edge_type
        )

        save_hist(
            edge_counts[edge_type],
            f"{' -> '.join(edge_type)} edge count",
            "Number of edges",
            f"{edge_name}_edge_count_histogram.png"
        )

    label_plot_path = os.path.join(
        plot_dir,
        "label_distribution.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        label_counts.index.tolist(),
        label_counts.values.tolist()
    )

    ax.set_title("Label distribution")
    ax.set_ylabel("Number of graphs")
    ax.set_xticks(
        range(
            len(label_counts)
        )
    )
    ax.set_xticklabels(
        label_counts.index.tolist(),
        rotation=45,
        ha="right"
    )

    fig.tight_layout()
    fig.savefig(
        label_plot_path,
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close(fig)

    return {
        "graph_level": (
            base_dataset.graph_level
        ),
        "total_graphs": len(
            dataset
        ),
        "summarized_graphs": len(
            indices
        ),
        "label_counts": (
            label_counts.to_dict()
        ),
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "feature_dimensions": (
            base_dataset.input_dims
        ),
        "plot_dir": plot_dir
    }

def dataset_split(
    dataset,
    train_frac=0.80,
    test_frac=0.20,
    seed=42
):
    if (
        not 0 < train_frac < 1 or
        not 0 < test_frac < 1 or
        not np.isclose(
            train_frac + test_frac,
            1
        )
    ):
        raise ValueError(
            "Positive train_frac and test_frac "
            "must sum to 1"
        )

    if dataset.labels_dict is None:
        raise ValueError(
            "Stratified splitting requires labels"
        )

    labels = [
        dataset.label_to_id[
            dataset.labels_dict[
                observation_id
            ]
        ]
        for observation_id in (
            dataset.valid_observation_ids
        )
    ]

    a, b = train_test_split(
        np.arange(len(dataset)),
        test_size=test_frac,
        stratify=labels,
        random_state=seed
    )

    train_dataset = Subset(
        dataset,
        a.tolist()
    )

    test_dataset = Subset(
        dataset,
        b.tolist()
    )

    print(
        f"[dataset_split] Train graphs: "
        f"{len(train_dataset)}"
    )

    print(
        f"[dataset_split] Test graphs: "
        f"{len(test_dataset)}"
    )

    return train_dataset, test_dataset