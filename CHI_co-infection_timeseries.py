import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import pandas as pd
import plotly.graph_objects as go

########################################
# 1) Define classify_cell function
########################################
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

########################################
# 2) Load data and filter
########################################

# With h5ad file
file_path = "working_data/preprocessed_PETRI_outputs/13Nov2024RTmultiinfection_15000/RTmulti_mixed_species_gene_matrix_preprocessed.h5ad"
adata = sc.read_h5ad(file_path)

#file_path = "working_data/Unprocessed_data/13Nov2024RTmultiinfection/13Nov24_RT_multi_infection_gene_matrix.txt"
#adata = sc.read(file_path, delimiter="\t")  # Use delimiter="\t" for tab-separated

print(adata)

# Remove genes with commas in their names
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")
print(removed_genes)

# Basic filtering
min_counts_cells = 5
min_counts_genes = 5
sc.pp.filter_cells(adata, min_counts=min_counts_cells)
sc.pp.filter_genes(adata, min_counts=min_counts_genes)

########################################
# 3) Classify each cell into a timepoint
########################################
adata.obs["timepoint"] = [classify_cell(cell_name) for cell_name in adata.obs_names]

########################################
# 4) Phage Analysis on the *entire* adata
#    (So that columns for each phage expression
#    are available in adata.obs)
########################################
phage_patterns = ["luz19:", "lkd16:"]

for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    # Mark these genes in var
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    # Sum expression across these genes
    if sparse.issparse(adata.X):
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1).A.flatten()
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1).A.flatten()
    else:
        phage_expression = adata[:, adata.var[f'{phage.strip(":")}_genes']].X.sum(axis=1)
        phage_n_genes = (adata[:, adata.var[f'{phage.strip(":")}_genes']].X > 0).sum(axis=1)

    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression
    adata.obs[f"{phage.strip(':')}_n_genes"] = phage_n_genes

# Create a combined presence/absence encoding
adata.obs["phage_presence"] = (
    (adata.obs["luz19_expression"] > 0).astype(int) * 1 +
    (adata.obs["lkd16_expression"] > 0).astype(int) * 2
)

########################################
# Helper function to color differences
########################################
def diff_color(val, vmin, vmax):
    """
    Maps a difference value to an RGB color.
    For negative values, interpolates from blue (at vmin) to white (0).
    For positive values, interpolates from white (0) to red (at vmax).
    """
    if val < 0:
        frac = (val - vmin) / (0 - vmin) if vmin != 0 else 0.5
        r = int(255 * frac)
        g = int(255 * frac)
        b = 255
    else:
        frac = val / vmax if vmax != 0 else 0.5
        r = 255
        g = int(255 * (1 - frac))
        b = int(255 * (1 - frac))
    return f"rgb({r},{g},{b})"

########################################
# 5) Loop over each timepoint, do the
#    infection analysis & chi-square
########################################
timepoints = ["5min", "10min", "15min", "20min"]

# For labeling combos
combination_labels = {
    0: "No phage",
    1: "Only luz19",
    2: "Only lkd16",
    3: "luz19 and lkd16",
}

# Hard-coded MOIs for your phages
MOI_values = {"luz19": 0.57, "lkd16": 0.35}

categories_order = [
    "No phage",
    "Only luz19",
    "Only lkd16",
    "luz19 and lkd16"
]

for tp in timepoints:
    print(f"\n============================")
    print(f"Analyzing timepoint: {tp}")
    print(f"============================")

    # Subset the data for this timepoint
    adata_tp = adata[adata.obs["timepoint"] == tp]

    if adata_tp.n_obs == 0:
        print(f"No cells found for {tp}, skipping analysis.")
        continue

    # Count combos in this subset
    phage_combinations = adata_tp.obs["phage_presence"].value_counts()
    phage_combinations.index = phage_combinations.index.map(combination_labels)

    # Observed list in the same order
    observed_list = [phage_combinations.get(cat, 0) for cat in categories_order]

    # Calculate expected counts using the timepoint’s total cell count
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
    print(f"  Chi-square test results: chi2 = {chi2:.2f}, p-value = {p_val:.3e}")

    # Print summary
    print(f"  Total cells in {tp}: {N_total}")
    for cat, obs, exp in zip(categories_order, observed_list, expected_list):
        print(f"    {cat}: Obs={obs}, Exp={int(round(exp))}")

    # Make a quick table (matplotlib) for this timepoint
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_axis_off()
    table_values = pd.DataFrame({
        "Observed": [int(x) for x in observed_list],
        "Expected": [int(round(x)) for x in expected_list],
        "Difference": [int(round(o - e)) for o, e in zip(observed_list, expected_list)]
    }, index=categories_order).astype(str).values

    table = ax.table(
        cellText=table_values,
        colLabels=["Observed", "Expected", "Difference"],
        rowLabels=categories_order,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0,1,2])
    plt.title(f"Chi-Square Obs vs Exp ({tp})", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Colorize differences in a Plotly table
    diff_vals = [int(round(o - e)) for o, e in zip(observed_list, expected_list)]
    vmin, vmax = min(diff_vals), max(diff_vals)
    diff_colors = [diff_color(val, vmin, vmax) for val in diff_vals]

    header_values = ["Observed", "Expected", "Difference"]
    cell_values = [
        [cat for cat in categories_order],        # first column: categories
        [int(x) for x in observed_list],          # Observed
        [int(round(x)) for x in expected_list],   # Expected
        diff_vals                                 # Difference
    ]
    fill_colors = [
        ["white"] * len(categories_order),  # Category column
        ["white"] * len(categories_order),  # Observed
        ["white"] * len(categories_order),  # Expected
        diff_colors                         # Difference
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Category"] + header_values,
                    align='left',
                    fill_color='lightgray'),
        cells=dict(
            values=cell_values,
            align='left',
            fill_color=fill_colors
        )
    )])
    fig.update_layout(
        title=f"Observed vs Expected - Chi-Square Difference ({tp})",
        height=500
    )
    fig.show()

    # Stacked bar (well, single bar set) for this timepoint
    plt.figure(figsize=(6, 4))
    bars = plt.bar(categories_order, observed_list, edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 1,
            int(yval),
            ha="center",
            fontsize=10,
            fontweight="bold"
        )
    plt.title(f"Phage Co-Occurrence in Cells ({tp})")
    plt.ylabel("Number of Cells")
    plt.xlabel("Phage Combination")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
