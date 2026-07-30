"""
Compute chi-square co-infection stats (no timepoints) and save results.
"""

import os
import scanpy as sc
import numpy as np
from scipy import sparse
from scipy.stats import chisquare
import pandas as pd

# ----------------------------------
# USER SETTINGS
# ----------------------------------
DATA_DIR = "processed_data"
DATA_FILE = "JRG09-3PMP20/JRG09-3PMP20_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

# ----------------------------------
# Load data
# ----------------------------------
if FILE_PATH.endswith(".h5ad"):
    adata = sc.read_h5ad(FILE_PATH)
else:
    raw_data = pd.read_csv(FILE_PATH, sep="\t", index_col=0)
    adata = sc.AnnData(raw_data)

print(adata)

# ----------------------------------
# Remove genes with commas
# ----------------------------------
removed_genes = adata.var_names[adata.var_names.str.contains(",")]
adata = adata[:, ~adata.var_names.str.contains(",")]
print(f"Removed {len(removed_genes)} genes with commas.")

# ----------------------------------
# Filtering
# ----------------------------------
sc.pp.filter_cells(adata, min_counts=5)
sc.pp.filter_genes(adata, min_counts=5)

# ----------------------------------
# Phage analysis
# ----------------------------------
phage_patterns = ["pev2:", "lkd16:"]

for phage in phage_patterns:
    phage_genes = adata.var_names[adata.var_names.str.contains(phage)]
    adata.var[f'{phage.strip(":")}_genes'] = adata.var_names.isin(phage_genes)

    if sparse.issparse(adata.X):
        phage_expression = (
            adata[:, adata.var[f'{phage.strip(":")}_genes']]
            .X.sum(axis=1).A.flatten()
        )
    else:
        phage_expression = (
            adata[:, adata.var[f'{phage.strip(":")}_genes']]
            .X.sum(axis=1)
        )

    adata.obs[f"{phage.strip(':')}_expression"] = phage_expression

# ----------------------------------
# Encode phage presence
# ----------------------------------
adata.obs["phage_presence"] = (
    (adata.obs["pev2_expression"] > 0).astype(int) * 1 +
    (adata.obs["lkd16_expression"] > 0).astype(int) * 2
)

# ----------------------------------
# Export co-infected cells (pev2 + lkd16)
# ----------------------------------
pev2_mask = adata.var["pev2_genes"]
lkd_mask = adata.var["lkd16_genes"]

coinfected = adata.obs["phage_presence"] == 3
adata_coinf = adata[coinfected]

if sparse.issparse(adata_coinf.X):
    pev2_hits = adata_coinf[:, pev2_mask].X.sum(axis=1).A.flatten()
    lkd_hits = adata_coinf[:, lkd_mask].X.sum(axis=1).A.flatten()
else:
    pev2_hits = adata_coinf[:, pev2_mask].X.sum(axis=1)
    lkd_hits = adata_coinf[:, lkd_mask].X.sum(axis=1)

with open("pev2_lkd16_coinfected_cells.txt", "w") as f:
    f.write("cell_id\tpev2_hits\tlkd16_hits\n")
    for cell, p_hits, l_hits in zip(
        adata_coinf.obs_names, pev2_hits, lkd_hits
    ):
        f.write(f"{cell}\t{int(p_hits)}\t{int(l_hits)}\n")

print(
    f"Saved {adata_coinf.n_obs} co-infected cells "
    f"to pev2_lkd16_coinfected_cells.txt"
)

# ----------------------------------
# Chi-square analysis (GLOBAL)
# ----------------------------------
combination_labels = {
    0: "No phage",
    1: "Only pev2",
    2: "Only lkd16",
    3: "pev2 and lkd16",
}

categories_order = [
    "No phage",
    "Only pev2",
    "Only lkd16",
    "pev2 and lkd16",
]

MOI_values = {
    "pev2": 1,
    "lkd16": 1,
}

# Observed counts
phage_combinations = adata.obs["phage_presence"].value_counts()
phage_combinations.index = phage_combinations.index.map(combination_labels)
observed = [phage_combinations.get(cat, 0) for cat in categories_order]

# Expected counts
N_total = adata.n_obs
p_p = 1 - np.exp(-MOI_values["pev2"])
p_l = 1 - np.exp(-MOI_values["lkd16"])

expected = [
    N_total * ((1 - p_p) * (1 - p_l)),  # No phage
    N_total * (p_p * (1 - p_l)),        # Only pev2
    N_total * ((1 - p_p) * p_l),        # Only lkd16
    N_total * (p_p * p_l),              # Both
]

# Chi-square test
chi2, p_val = chisquare(f_obs=observed, f_exp=expected)

# Save results
chi_square_df = pd.DataFrame({
    "Category": categories_order,
    "Observed": observed,
    "Expected": [int(round(x)) for x in expected],
    "Difference": [obs - int(round(exp)) for obs, exp in zip(observed, expected)],
})

chi_square_df["Chi2"] = round(chi2, 2)
chi_square_df["P-value"] = f"{p_val:.3e}"

chi_square_df.to_csv("chi_square_results.csv", index=False)

print("\nChi-square results saved to 'chi_square_results.csv'.")
print(chi_square_df)
