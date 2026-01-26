# NUTMEG

NUTMEG is a Python framework for cell-type classification and regulatory program discovery from paired single-cell multiome data (scRNA-seq + scATAC-seq), built around graph-based attention.

It brings together data preparation, model training/evaluation, and interpretation into a unified workflow for extracting cell-type-specific regulatory signals.

---

## Features

### End-to-end multiome workflow
- Data preparation and standardized exports for downstream graph construction and modeling
- Dataset-driven global GRN construction (TF→CRE and CRE→Gene priors)
- Per-cell graph dataset construction
- Model training and evaluation for cell-type classification

### Graph-attention modeling
- Two-hop message passing over regulatory graphs (TF → CRE → Gene)
- Graph-based attention weights for edge/path importance

### Interpretability & explainability
- Gene contributors per cell type
- Attention-derived cell-type-specific TF–CRE–Gene regulatory programs (eGRNs)

## Tutorials
NUTMEG is demonstrated via two notebook tutorials:

1) **Data preparation**
   - QC, filtering, and dataset exports
   - Notebook: `tutorials/01_data_preparation.ipynb`

2) **End-to-end NUTMEG workflow**
   - Build the global GRN
   - Construct the per-cell graph dataset
   - Train + evaluate the model
   - Interpret results
   - Notebook: `tutorials/02_nutmeg.ipynb`
  
---

## Installation
Clone and install in editable mode:

```bash
git clone https://github.com/<USER>/nutmeg.git
cd nutmeg
pip install -e .
