"""Compute chi-square co-infection stats by timepoint and save the results table."""

import os
import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import pandas as pd
import plotly.graph_objects as go

# ----------------------------------
# USER SETTINGS (edit in PyCharm)
# ----------------------------------
DATA_DIR = "processed_data"
DATA_FILE = "JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Plot formatting (used if you add plots later)
MPL_DPI = 120
MPL_FONT_SIZE = 11
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_FONT_FAMILY = "Arial"
PLOTLY_FONT_SIZE = 12

plt.rcParams.update({
    "figure.dpi": MPL_DPI,
    "font.size": MPL_FONT_SIZE,
})

# ----------------------------------
# LINE PLOT STYLE BLOCK (edit here)
# ----------------------------------
LINE_PLOT_STYLE = {
    "figsize": (7, 5),
    "colors": {  # per-category colors
        "No phage": "black",
        "Only luz19": "tab:blue",
        "Only lkd16": "tab:orange",
        "luz19 and lkd16": "tab:red",
    },
    "linewidth": 2.0,
    "linestyle": "-",          # e.g. "-", "--", "-.", ":"
    "marker": "o",             # e.g. "o", "s", "D", None
    "markersize": 6,
    "xlabel": "Timepoint",
    "ylabel": "Difference (Observed - Expected)",
    "title": "Observed - Expected Co-infection Counts by Timepoint",
    "legend_loc": "best",      # e.g. "upper right", "lower left"
    "font_size": 12,
    "grid": True,
    "ylim": None,              # e.g. (-200, 200) or None for auto
}


# Function to classify cells into timepoints
def classify_cell(cell_name):
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
    # Allow h5ad if you ever want to use it
    adata = sc.read_h5ad(FILE_PATH)
else:
    # Gene matrix text file:
    #   - tab-separated
    #   - first column = cell IDs
    raw_data = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
    adata = sc.AnnData(raw_data)

print(adata)

# Remove genes with commas in their names
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")

# Filtering
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_counts=5)

# Classify cells into timepoints
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

# Phage analysis
phage_patterns = ["luz19:", "lkd16:"]
for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)

    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression

# Encode phage presence
adata.obs["phage_presence"] = (
    (adata.obs["luz19_expression"] > 0).astype(int) * 1 +
    (adata.obs["lkd16_expression"] > 0).astype(int) * 2
)

# Define timepoints and categories
timepoints = ["5min", "10min", "15min", "20min"]
combination_labels = {
    0: "No phage",
    1: "Only luz19",
    2: "Only lkd16",
    3: "luz19 and lkd16"
}
MOI_values = {"luz19": 0.57, "lkd16": 0.35}
categories_order = ["No phage", "Only luz19", "Only lkd16", "luz19 and lkd16"]

# Initialize list for saving results
chi_square_results = []

for tp in timepoints:
    adata_tp = adata[adata.obs["timepoint"] == tp]

    if adata_tp.n_obs == 0:
        continue

    # Observed infection counts
    phage_combinations = adata_tp.obs["phage_presence"].value_counts()
    phage_combinations.index = phage_combinations.index.map(combination_labels)
    observed_list = [phage_combinations.get(cat, 0) for cat in categories_order]

    # Expected counts based on MOI
    N_total = adata_tp.n_obs
    p_l = 1 - np.exp(-MOI_values["luz19"])
    p_k = 1 - np.exp(-MOI_values["lkd16"])

    expected_counts = {
        "No phage": N_total * ((1 - p_l) * (1 - p_k)),
        "Only luz19": N_total * (p_l * (1 - p_k)),
        "Only lkd16": N_total * ((1 - p_l) * p_k),
        "luz19 and lkd16": N_total * (p_l * p_k),
    }
    expected_list = [expected_counts[cat] for cat in categories_order]

    # Chi-square test
    chi2, p_val = chisquare(f_obs=observed_list, f_exp=expected_list)

    # Save results rows (long format)
    for cat, obs, exp in zip(categories_order, observed_list, expected_list):
        chi_square_results.append([
            tp,
            cat,
            obs,
            int(round(exp)),
            obs - int(round(exp)),   # Difference
            round(chi2, 2),
            f"{p_val:.3e}"
        ])

# Convert results to DataFrame and save as CSV (long format)
chi_square_df = pd.DataFrame(
    chi_square_results,
    columns=["Timepoint", "Category", "Observed", "Expected", "Difference", "Chi2", "P-value"]
)
chi_square_df.to_csv("chi_square_results.csv", index=False)

# ---- Wide-format difference table (ordered + transposed) ----
# Enforce desired timepoint order
timepoint_order = ["5min", "10min", "15min", "20min"]

chi_square_df["Timepoint"] = pd.Categorical(
    chi_square_df["Timepoint"],
    categories=timepoint_order,
    ordered=True
)

# Pivot: rows = Category, columns = Timepoint, values = Difference
diff_wide = chi_square_df.pivot(
    index="Category",
    columns="Timepoint",
    values="Difference"
)

# Ensure consistent row order
diff_wide = diff_wide.reindex(categories_order)

# Save to CSV
diff_wide.to_csv("chi_square_differences_wide.csv")

print("\nChi-square results saved to 'chi_square_results.csv'.")
print("Difference matrix saved to 'chi_square_differences_wide.csv'.")

# -------------------------------------------------------------
# LINE PLOT: timepoints on x-axis, one line per category/sample
# -------------------------------------------------------------
# You can either re-read from CSV or just use diff_wide directly.
# We'll use diff_wide (already in memory and ordered).
plt.figure(figsize=LINE_PLOT_STYLE["figsize"])

# Ensure we use the timepoint order we defined
x = timepoint_order

for category in diff_wide.index:
    y = diff_wide.loc[category, x].values

    color = LINE_PLOT_STYLE["colors"].get(category, None)
    plt.plot(
        x,
        y,
        label=category,
        color=color,
        linewidth=LINE_PLOT_STYLE["linewidth"],
        linestyle=LINE_PLOT_STYLE["linestyle"],
        marker=LINE_PLOT_STYLE["marker"],
        markersize=LINE_PLOT_STYLE["markersize"],
    )

plt.xlabel(LINE_PLOT_STYLE["xlabel"], fontsize=LINE_PLOT_STYLE["font_size"])
plt.ylabel(LINE_PLOT_STYLE["ylabel"], fontsize=LINE_PLOT_STYLE["font_size"])
plt.title(LINE_PLOT_STYLE["title"], fontsize=LINE_PLOT_STYLE["font_size"] + 2)
plt.legend(loc=LINE_PLOT_STYLE["legend_loc"], fontsize=LINE_PLOT_STYLE["font_size"] - 1)

if LINE_PLOT_STYLE["grid"]:
    plt.grid(True, alpha=0.3)

if LINE_PLOT_STYLE["ylim"] is not None:
    plt.ylim(LINE_PLOT_STYLE["ylim"])

plt.tight_layout()
plt.show()
