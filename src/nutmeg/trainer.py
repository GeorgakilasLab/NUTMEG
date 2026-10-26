import json
import gc
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import os, random

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)




# ---------------------------
# Utilities
# ---------------------------
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



@dataclass
class FitResult:
    best_epoch: int
    best_val: dict
    history: list
    paths: dict


# ---------------------------
# Main trainer 
# ---------------------------
class NUTMEGTrainer:

    def _apply_seed(self):
        seed = int(self.cfg.get("train", {}).get("seed", 0))
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self._log(f"[seed] seed={seed}")
        return seed

    def __init__(self, cfg: dict, dataset, model_ctor, device=None, run_name=None, results_root: str = None):
        self.cfg = cfg
        self.dataset = dataset            
        self.model_ctor = model_ctor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_name = run_name or "default_run"
        self.run_id = f"{base_name}"
        
        results_root = Path(results_root) if results_root else Path("results")

        self.plots_dir = results_root / "plots" / self.run_id
        self.train_dir = results_root / "training" / self.run_id
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.train_dir / "train.log.txt"

        # scalers
        self.scaler_tf = StandardScaler()
        self.scaler_gene = StandardScaler()
        self.scaler_cre = StandardScaler()

        # objects
        self.model = None
        self.criterion = None
        self.optimizer = None

        # tracking
        self.history = []
        self.best_state_dict = None
        self.best_epoch = -1

    @classmethod
    def initialize(cls, *, json_path: str, dataset, model_ctor, device=None, run_name=None, results_root: str = None):
        with open(json_path, "r") as f:
            cfg = json.load(f)
        obj = cls(cfg=cfg, dataset=dataset, model_ctor=model_ctor, device=device, run_name=run_name, results_root=results_root)
        (obj.train_dir / "params.used.json").write_text(Path(json_path).read_text(encoding="utf-8"), encoding="utf-8")
        obj._apply_seed()
        obj._build_model()
        return obj

    def _log(self, msg: str):
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def _infer_num_genes(self):
        if not hasattr(self.dataset, "gene_node_names"):
            raise ValueError("Base dataset must have gene_node_names to infer num_genes")
        return int(len(self.dataset.gene_node_names))

    def _infer_num_classes_from_ds(self, ds):
        ys = [int(ds[i].y.item()) for i in range(len(ds))]
        return int(max(ys)) + 1 if ys else 0

    def _infer_target_names(self, num_classes: int):
        if hasattr(self.dataset, "id_to_label"):
            id_to_label = getattr(self.dataset, "id_to_label")
            try:
                return [str(id_to_label[i]) for i in range(num_classes)]
            except Exception:
                pass
        return [str(i) for i in range(num_classes)]
    
    def _model_kwargs(self, *, num_classes: int):
        mcfg = dict(self.cfg["model"])
        return dict(
            hidden_dim=int(mcfg["hidden_dim"]),
            num_classes=int(num_classes),
            num_genes=self._infer_num_genes(),
            mode=str(mcfg.get("mode", "multiome")),          
            heads=int(mcfg.get("heads", 1)),
            dropout=float(mcfg.get("dropout", 0.0)),
            per_gene_hidden=int(mcfg["per_gene_hidden"]),
            include_presence_bit=bool(mcfg.get("include_presence_bit", True)),
            return_intermediates=bool(mcfg.get("return_intermediates", False)),
        )

    def _build_model(self):
        ys = [int(self.dataset[i].y.item())
              for i in range(len(self.dataset))
              if hasattr(self.dataset[i], "y")]
        num_classes = int(max(ys)) + 1 if ys else 2

        self.model = self.model_ctor(**self._model_kwargs(num_classes=num_classes)).to(self.device)

        self._log(f"[init] run_id={self.run_id} device={self.device}")
        self._log(f"[init] num_genes={self._infer_num_genes()} (num_classes initial={num_classes})")
    
    # --------------------------
    # Scalers (fit on a dataset)
    # --------------------------
    def fit_scalers(self, train_ds):
        X_tf_list, X_gene_list, X_cre_list = [], [], []
        for i in range(len(train_ds)):
            g = train_ds[i]

            if hasattr(g["gene"], "tf_feats") and hasattr(g["gene"], "is_tf"):
                tf = g["gene"].tf_feats.cpu().numpy()
                m = g["gene"].is_tf.cpu().numpy().astype(bool)
                if tf.shape[1] > 0 and m.any():
                    X_tf_list.append(tf[m])

            if hasattr(g["gene"], "gene_feats") and hasattr(g["gene"], "is_target"):
                ge = g["gene"].gene_feats.cpu().numpy()
                m = g["gene"].is_target.cpu().numpy().astype(bool)
                if ge.shape[1] > 0 and m.any():
                    X_gene_list.append(ge[m])

            if hasattr(g["cre"], "x"):
                cx = g["cre"].x.cpu().numpy()
                if cx.shape[1] > 0:
                    X_cre_list.append(cx)

        if X_tf_list:
            self.scaler_tf.fit(np.vstack(X_tf_list))
        if X_gene_list:
            self.scaler_gene.fit(np.vstack(X_gene_list))
        if X_cre_list:
            self.scaler_cre.fit(np.vstack(X_cre_list))

        self._log(f"[scalers] fitted on train_ds len={len(train_ds)}")
        return self
    
    def _scaler_is_fitted(self, scaler) -> bool:
            return hasattr(scaler, "mean_")

    def transform_graph_with_scalers(self, g):
        if hasattr(g["gene"], "tf_feats") and hasattr(g["gene"], "is_tf") and g["gene"].tf_feats.numel() > 0:
            tf = g["gene"].tf_feats.cpu().numpy()
            m = g["gene"].is_tf.cpu().numpy().astype(bool)
            tf_scaled = tf.copy()
            if m.any():
                tf_scaled[m] = self.scaler_tf.transform(tf[m])
            tf_scaled[~m] = 0.0
            g["gene"].tf_feats = torch.tensor(tf_scaled, dtype=torch.float, device=g["gene"].tf_feats.device)

        if hasattr(g["gene"], "gene_feats") and hasattr(g["gene"], "is_target") and g["gene"].gene_feats.numel() > 0:
            ge = g["gene"].gene_feats.cpu().numpy()
            m = g["gene"].is_target.cpu().numpy().astype(bool)
            ge_scaled = ge.copy()
            if m.any():
                ge_scaled[m] = self.scaler_gene.transform(ge[m])
            ge_scaled[~m] = 0.0
            g["gene"].gene_feats = torch.tensor(ge_scaled, dtype=torch.float, device=g["gene"].gene_feats.device)

        if hasattr(g["cre"], "x") and g["cre"].x.numel() > 0 and self._scaler_is_fitted(self.scaler_cre):
            cx = g["cre"].x.cpu().numpy()
            g["cre"].x = torch.tensor(self.scaler_cre.transform(cx), dtype=torch.float, device=g["cre"].x.device)

        return g

    def _materialize_graphs_from_ds(self, ds):
        out = []
        for i in range(len(ds)):
            g = ds[i]
            g2 = g.clone() if hasattr(g, "clone") else deepcopy(g)
            out.append(self.transform_graph_with_scalers(g2))
        return out

    # --------------------------
    # Train/val split inside train_ds
    # --------------------------
    def _split_train_val_ds(self, train_ds):
        tcfg = self.cfg["train"]
        frac = float(tcfg.get("es_val_frac", 0.10))
        seed = int(tcfg.get("seed", 0))

        idx = np.arange(len(train_ds), dtype=int)
        y = np.array([int(train_ds[i].y.item()) for i in range(len(train_ds))], dtype=int)

        try:
            tr_i, va_i = train_test_split(idx, test_size=frac, random_state=seed, shuffle=True, stratify=y)
        except Exception:
            tr_i, va_i = train_test_split(idx, test_size=frac, random_state=seed, shuffle=True, stratify=None)

        tr_ds = Subset(train_ds, tr_i.tolist())
        va_ds = Subset(train_ds, va_i.tolist())

        y_tr = np.array([int(tr_ds[i].y.item()) for i in range(len(tr_ds))], dtype=int)
        self.class_ids_ = sorted(np.unique(y_tr).tolist())
        self.class_names_ = [self._infer_target_names(max(self.class_ids_) + 1)[cid] for cid in self.class_ids_]

        return tr_ds, va_ds

    # --------------------------
    # Loss/optimizer from train subset
    # --------------------------
    def _make_class_weights_from_ds(self, num_classes, ds):
        y_train = np.array([int(ds[i].y.item()) for i in range(len(ds))], dtype=int)
        present = np.unique(y_train)
        cw_present = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
        w = torch.ones(num_classes, dtype=torch.float32)
        w[present] = torch.tensor(cw_present, dtype=torch.float32)
        return w.to(self.device)

    def _rebuild_classifier_if_needed(self, num_classes_new: int):
        if int(self.model.C) == int(num_classes_new):
            return

        self.model = self.model_ctor(**self._model_kwargs(num_classes=num_classes_new)).to(self.device)
        self._log(f"[model] rebuilt for num_classes={num_classes_new}")

    def _build_criterion_optimizer(self, train_ds_internal):
        tcfg = self.cfg["train"]

        num_classes = self._infer_num_classes_from_ds(train_ds_internal)
        self._rebuild_classifier_if_needed(num_classes)

        cw = self._make_class_weights_from_ds(num_classes, train_ds_internal)

        loss_name = str(tcfg.get("loss_name", "ce")).lower()
        if loss_name != "ce":
            raise ValueError(f"Only loss_name='ce' implemented, got {loss_name}")

        label_smooth = float(tcfg.get("label_smooth_train", 0.0))
        self.criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smooth)

        lr = float(tcfg.get("lr", 5e-4))
        wd = float(tcfg.get("weight_decay", 1e-3))
        opt_name = str(tcfg.get("optimizer", "adamw")).lower()

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer={opt_name}")

        self._log(f"[optim] {opt_name} lr={lr} wd={wd} label_smooth={label_smooth} num_classes={num_classes}")

    # --------------------------
    # Core loop
    # --------------------------
    def _run_loader(self, loader, *, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, n_seen = 0.0, 0
        y_true, y_pred = [], []

        for data in loader:
            data = data.to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits, *_ = self.model(data)  
                y = data.y.view(-1).long()
                loss = self.criterion(logits, y)

                if train:
                    loss.backward()
                    clip = float(self.cfg["train"].get("grad_clip_norm", 2.0))
                    if clip and clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()

            bs = int(y.size(0))
            total_loss += float(loss.detach().cpu()) * bs
            n_seen += bs
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(logits.argmax(-1).detach().cpu().numpy())

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.size == 0:
            return {"loss": float("nan"), "accuracy": float("nan"), "bal_acc": float("nan"), "macro_f1": float("nan"),
                    "y_true": y_true, "y_pred": y_pred}

        return {
            "loss": total_loss / max(1, n_seen),
            "accuracy": accuracy_score(y_true, y_pred),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def _save_history_csv(self, history, path: Path):
        fields = [
            "epoch",
            "train_loss", "train_accuracy", "train_bal_acc", "train_macro_f1",
            "val_loss", "val_accuracy", "val_bal_acc", "val_macro_f1",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in history:
                ep = row["epoch"]
                tr = row["train"]
                va = row["val"]
                w.writerow({
                    "epoch": ep,
                    "train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "train_bal_acc": tr["bal_acc"], "train_macro_f1": tr["macro_f1"],
                    "val_loss": va["loss"], "val_accuracy": va["accuracy"], "val_bal_acc": va["bal_acc"], "val_macro_f1": va["macro_f1"],
                })

    def _plot_training_curves(self, history, out_path: Path):
        epochs = [h["epoch"] for h in history]
        tr_loss = [h["train"]["loss"] for h in history]
        va_loss = [h["val"]["loss"] for h in history]
        tr_f1 = [h["train"]["macro_f1"] for h in history]
        va_f1 = [h["val"]["macro_f1"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, tr_loss, label="Train Loss")
        axes[0].plot(epochs, va_loss, label="Val Loss")
        axes[0].set_title("Training vs Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, tr_f1, label="Train Macro-F1")
        axes[1].plot(epochs, va_f1, label="Val Macro-F1")
        axes[1].set_title("Training vs Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _save_confusion_matrix_plot(self, y_true, y_pred, out_path: Path, title: str, labels, display_labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    # --------------------------
    def train(self, train_ds, *, epochs=None):
        """
        Input: train_ds (Subset or Dataset)
        Output folders:
          - training/<run_id>/ : logs, csv, curves, weights, summary
        """
        tcfg = self.cfg["train"]
        max_epochs = int(epochs or tcfg.get("max_epochs", 30))
        patience = int(tcfg.get("patience", 5))
        min_delta = float(tcfg.get("min_delta", 1e-3))
        bs = int(tcfg.get("train_batch_size", 64))

        tr_ds, va_ds = self._split_train_val_ds(train_ds)
        self._log(f"[split] train={len(tr_ds)} val={len(va_ds)} (from train_ds={len(train_ds)})")

        # scalers fit on internal train split only 
        self.fit_scalers(tr_ds)

        # build loss/opt using internal train subset
        self._build_criterion_optimizer(tr_ds)

        # materialize
        tr_graphs = self._materialize_graphs_from_ds(tr_ds)
        va_graphs = self._materialize_graphs_from_ds(va_ds)

        tr_loader = DataLoader(ListDataset(tr_graphs), batch_size=bs, shuffle=True,
                               num_workers=0, pin_memory=(self.device.type == "cuda"), persistent_workers=False)
        va_loader = DataLoader(ListDataset(va_graphs), batch_size=bs, shuffle=False,
                               num_workers=0, pin_memory=(self.device.type == "cuda"), persistent_workers=False)

        best_f1, best_loss = -1e18, 1e18
        no_improve = 0
        self.best_state_dict = None
        self.best_epoch = -1
        self.history = []

        self._log(f"[train] max_epochs={max_epochs} patience={patience} min_delta={min_delta} batch_size={bs}")

        for ep in range(1, max_epochs + 1):
            tr = self._run_loader(tr_loader, train=True)
            va = self._run_loader(va_loader, train=False)

            improved = (va["macro_f1"] > best_f1 + min_delta) or \
                       (abs(va["macro_f1"] - best_f1) <= min_delta and va["loss"] < best_loss - 1e-6)

            self.history.append({"epoch": ep, "train": tr, "val": va})

            self._log(
                f"[ep {ep:03d}] "
                f"tr loss={tr['loss']:.4f} acc={tr['accuracy']:.3f} bal={tr['bal_acc']:.3f} f1={tr['macro_f1']:.3f} | "
                f"va loss={va['loss']:.4f} acc={va['accuracy']:.3f} bal={va['bal_acc']:.3f} f1={va['macro_f1']:.3f}"
                + (" *" if improved else "")
            )

            if improved:
                best_f1, best_loss = va["macro_f1"], va["loss"]
                self.best_epoch = ep
                self.best_state_dict = deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                self._log(f"[early_stop] at epoch={ep} best_epoch={self.best_epoch} best_val_f1={best_f1:.4f}")
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        weights_path = self.train_dir / "best_state_dict.pt"
        torch.save(self.model.state_dict(), weights_path)

        hist_csv_path = self.train_dir / "history.csv"
        self._save_history_csv(self.history, hist_csv_path)

        curves_path = self.train_dir / "training_curves.png"
        if self.history:
            self._plot_training_curves(self.history, curves_path)

        best_row = max(self.history, key=lambda r: r["val"]["macro_f1"]) if self.history else None
        best_val = best_row["val"] if best_row else {}

        summary = {
            "run_id": self.run_id,
            "best_epoch": int(self.best_epoch),
            "best_val": {k: float(v) for k, v in best_val.items() if k not in ("y_true", "y_pred")},
            "paths": {
                "train_log": str(self.log_path),
                "history_csv": str(hist_csv_path),
                "training_curves_png": str(curves_path),
                "best_state_dict": str(weights_path),
            }
        }
        (self.train_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return FitResult(best_epoch=self.best_epoch, best_val=best_val, history=self.history, paths=summary["paths"])

    @torch.no_grad()
    def _save_per_class_metrics_plot(self, report_dict, out_path: Path, title="Per-class metrics"):
        class_keys = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

        labels, precision, recall, f1 = [], [], [], []
        for k in class_keys:
            v = report_dict.get(k, {})
            if isinstance(v, dict) and ("precision" in v) and ("recall" in v) and ("f1-score" in v):
                labels.append(str(k))
                precision.append(float(v["precision"]))
                recall.append(float(v["recall"]))
                f1.append(float(v["f1-score"]))

        if not labels:
            return

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(labels)), 6))
        ax.bar(x - width, precision, width, label="precision")
        ax.bar(x,         recall,    width, label="recall")
        ax.bar(x + width, f1,        width, label="f1-score")

        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def evaluate(self, test_ds):
        """
        Input: test_ds (Subset or Dataset)
        Output folder:
        - plots/<run_id>/ : classification report, confusion matrix, test metrics, per-class metrics plot
        """
        bs = int(self.cfg["train"].get("train_batch_size", 64))

        test_graphs = self._materialize_graphs_from_ds(test_ds)
        test_loader = DataLoader(
            ListDataset(test_graphs),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=False,
        )

        m = self._run_loader(test_loader, train=False)

        labels = self.class_ids_
        target_names = self.class_names_

        rep_txt = classification_report(
            m["y_true"], m["y_pred"],
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0
     )
        rep_dict = classification_report(
            m["y_true"], m["y_pred"],
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

        (self.plots_dir / "classification_report.txt").write_text(rep_txt, encoding="utf-8")
        (self.plots_dir / "classification_report.json").write_text(json.dumps(rep_dict, indent=2), encoding="utf-8")

        cm_path = self.plots_dir / "confusion_matrix.png"
        self._save_confusion_matrix_plot(
            m["y_true"], m["y_pred"],
            cm_path,
            title="Confusion Matrix",
            labels=labels,
            display_labels=target_names
        )

        # per-class metrics 
        per_class_path = self.plots_dir / "per_class_metrics.png"
        self._save_per_class_metrics_plot(rep_dict, per_class_path, title="Per-class metrics")

        test_metrics = {
            "run_id": self.run_id,
            "n_test": int(len(test_ds)),
            "y_true": m["y_true"].tolist(),
            "y_pred": m["y_pred"].tolist(),
            "test_loss": float(m["loss"]),
            "test_accuracy": float(m["accuracy"]),
            "test_bal_acc": float(m["bal_acc"]),
            "test_macro_f1": float(m["macro_f1"]),
            "paths": {
                "classification_report_txt": str(self.plots_dir / "classification_report.txt"),
                "classification_report_json": str(self.plots_dir / "classification_report.json"),
                "confusion_matrix_png": str(cm_path),
                "per_class_metrics_png": str(per_class_path),
            }
        }

        (self.plots_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

        self._log(f"[eval] saved plots to {self.plots_dir}")


        return test_metrics 
