"""Compare gene expression across infection states by timepoint.

For each timepoint (5, 10, 15, 20 min):
- Select the top N genes (by overall mean expression at that timepoint)
- Plot a grouped bar chart showing mean expression in:
    - Uninfected (no phage)
    - Single infection (either luz19 OR lkd16)
    - Co-infected (both luz19 AND lkd16)
"""

import os
import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------
# USER SETTINGS (edit in PyCharm)
# ----------------------------------
DATA_DIR = "processed_data"
DATA_FILE = "JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Minimum counts for filtering
MIN_COUNTS_CELLS = 5
MIN_COUNTS_GENES = 5

# How many genes to show per timepoint
N_TOP_GENES = 10

# Whether to normalize + log1p before computing means
DO_NORMALIZE = True

# ----------------------------------
# BAR PLOT STYLE BLOCK (edit here)
# ----------------------------------
BAR_PLOT_STYLE = {
    "figsize": (10, 6),
    "colors": {
        "Uninfected": "tab:gray",
        "Single infection": "tab:blue",
        "Coinfected": "tab:red",
    },
    "bar_width": 0.25,
    "xlabel": "Gene",
    "ylabel": "Mean expression",
    "title_prefix": "Top genes by infection state at ",
    "font_size": 12,
    "xtick_rotation": 45,
    "legend_loc": "best",
    "grid": True,
    "ylim": None,   # e.g. (0, 5) or None for auto
}


# ----------------------------------
# Helper: classify cells by timepoint
# ----------------------------------
def classify_cell(cell_name: str) -> str:
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "5min"
    elif bc1_value < 49:
        return "10min"
    elif bc1_value < 73:
        return "15min"
    else:
        return "20min"


# -----------------------------
# Load data (text matrix or h5ad)
# -----------------------------
if FILE_PATH.endswith(".h5ad"):
    adata = sc.read_h5ad(FILE_PATH)
else:
    # Gene matrix text file:
    #   - tab-separated
    #   - first column = cell IDs
    raw_data = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
    adata = sc.AnnData(raw_data)

print(adata)

# Remove genes with commas in their names (optional; matches previous script)
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")

# Filtering
sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)
sc.pp.filter_genes(adata, min_counts=MIN_COUNTS_GENES)

# Optional normalization/log transform (per-cell library size → 1e4; then log1p)
if DO_NORMALIZE:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# Classify cells into timepoints
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

# Phage analysis: mark genes and compute per-cell expression
phage_patterns = ["luz19:", "lkd16:"]
for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    if sparse.issparse(adata.X):
        phage_expression = (
            adata[:, adata.var[f'{phage.strip(":")}_genes']]
            .X.sum(axis=1)
            .A.flatten()
        )
    else:
        phage_expression = (
            adata[:, adata.var[f'{phage.strip(":")}_genes']]
            .X.sum(axis=1)
        )

    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression

# Encode infection state:
#   0 = no phage
#   1 = only luz19
#   2 = only lkd16
#   3 = both
adata.obs["phage_presence"] = (
    (adata.obs["luz19_expression"] > 0).astype(int) * 1 +
    (adata.obs["lkd16_expression"] > 0).astype(int) * 2
)

# For plotting, define labels for 3 conditions:
#   Uninfected      -> phage_presence == 0
#   Single infection -> phage_presence in {1, 2}
#   Coinfected      -> phage_presence == 3
condition_labels = ["Uninfected", "Single infection", "Coinfected"]

timepoints = ["5min", "10min", "15min", "20min"]

# -----------------------------
# Per-timepoint bar plots
# -----------------------------
for tp in timepoints:
    adata_tp = adata[adata.obs["timepoint"] == tp]

    if adata_tp.n_obs == 0:
        print(f"No cells found for {tp}, skipping.")
        continue

    X_tp = adata_tp.X
    if sparse.issparse(X_tp):
        # mean along axis=0 (cells) → 1 x n_genes
        overall_mean = np.asarray(X_tp.mean(axis=0)).ravel()
    else:
        overall_mean = X_tp.mean(axis=0)

    # Select top N genes by overall mean expression at this timepoint
    n_genes = adata_tp.n_vars
    n_top = min(N_TOP_GENES, n_genes)
    top_idx = np.argsort(overall_mean)[::-1][:n_top]
    top_genes = adata_tp.var_names[top_idx]

    # Compute mean expression per condition for these genes
    means_by_condition = {}

    # Define masks for the three conditions
    mask_uninfected = adata_tp.obs["phage_presence"] == 0
    mask_single = adata_tp.obs["phage_presence"].isin([1, 2])
    mask_coinfected = adata_tp.obs["phage_presence"] == 3

    condition_masks = {
        "Uninfected": mask_uninfected,
        "Single infection": mask_single,
        "Coinfected": mask_coinfected,
    }

    for cond_label, mask in condition_masks.items():
        if mask.sum() == 0:
            # No cells in this condition at this timepoint → zeros
            means_by_condition[cond_label] = np.zeros(n_top)
            continue

        X_cond = adata_tp[mask, :].X
        if sparse.issparse(X_cond):
            mean_cond_all = np.asarray(X_cond.mean(axis=0)).ravel()
        else:
            mean_cond_all = X_cond.mean(axis=0)

        means_by_condition[cond_label] = mean_cond_all[top_idx]

    # -----------------------------
    # Make grouped bar plot
    # -----------------------------
    x = np.arange(n_top)  # positions for genes
    width = BAR_PLOT_STYLE["bar_width"]

    plt.figure(figsize=BAR_PLOT_STYLE["figsize"])

    offsets = {
        "Uninfected": -width,
        "Single infection": 0.0,
        "Coinfected": width,
    }

    for cond_label in condition_labels:
        y = means_by_condition[cond_label]
        color = BAR_PLOT_STYLE["colors"].get(cond_label, None)
        plt.bar(
            x + offsets[cond_label],
            y,
            width=width,
            label=cond_label,
            color=color,
            edgecolor="black",
        )

    plt.xticks(
        x,
        top_genes,
        rotation=BAR_PLOT_STYLE["xtick_rotation"],
        ha="right",
        fontsize=BAR_PLOT_STYLE["font_size"] - 1,
    )
    plt.xlabel(BAR_PLOT_STYLE["xlabel"], fontsize=BAR_PLOT_STYLE["font_size"])
    plt.ylabel(BAR_PLOT_STYLE["ylabel"], fontsize=BAR_PLOT_STYLE["font_size"])
    plt.title(
        BAR_PLOT_STYLE["title_prefix"] + tp,
        fontsize=BAR_PLOT_STYLE["font_size"] + 2,
    )
    plt.legend(loc=BAR_PLOT_STYLE["legend_loc"], fontsize=BAR_PLOT_STYLE["font_size"] - 1)

    if BAR_PLOT_STYLE["grid"]:
        plt.grid(axis="y", alpha=0.3)

    if BAR_PLOT_STYLE["ylim"] is not None:
        plt.ylim(BAR_PLOT_STYLE["ylim"])

    plt.tight_layout()
    plt.show()
