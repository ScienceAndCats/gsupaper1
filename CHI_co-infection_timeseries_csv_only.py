import scanpy as sc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import pandas as pd
import plotly.graph_objects as go

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

# Load data
file_path = "working_data/Unprocessed_data/13Nov2024RTmultiinfection/13Nov24_RT_multi_infection_gene_matrix.txt"
adata = sc.read(file_path, delimiter="\t")

# Remove genes with commas in their names
adata = adata[:, ~adata.var_names.str.contains(",")]

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
combination_labels = {0: "No phage", 1: "Only luz19", 2: "Only lkd16", 3: "luz19 and lkd16"}
MOI_values = {"luz19": 0.57, "lkd16": 0.35}
categories_order = ["No phage", "Only luz19", "Only lkd16", "luz19 and lkd16"]

# Initialize dataframe for saving results
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
    p_l, p_k = 1 - np.exp(-MOI_values["luz19"]), 1 - np.exp(-MOI_values["lkd16"])

    expected_counts = {
        "No phage": N_total * ((1 - p_l) * (1 - p_k)),
        "Only luz19": N_total * (p_l * (1 - p_k)),
        "Only lkd16": N_total * ((1 - p_l) * p_k),
        "luz19 and lkd16": N_total * (p_l * p_k),
    }
    expected_list = [expected_counts[cat] for cat in categories_order]

    # Chi-square test
    chi2, p_val = chisquare(f_obs=observed_list, f_exp=expected_list)

    # Save results to dataframe
    for cat, obs, exp in zip(categories_order, observed_list, expected_list):
        chi_square_results.append([tp, cat, obs, int(round(exp)), obs - int(round(exp)), round(chi2, 2), f"{p_val:.3e}"])

# Convert results to DataFrame and save as CSV
chi_square_df = pd.DataFrame(chi_square_results, columns=["Timepoint", "Category", "Observed", "Expected", "Difference", "Chi2", "P-value"])
chi_square_df.to_csv("chi_square_results.csv", index=False)

print("\nChi-square results saved to 'chi_square_results.csv'.")
