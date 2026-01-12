import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Load and Prepare Data
# ---------------------------
adata = sc.read_csv("working_data/preprocessed_PETRI_outputs/17Sep24_Luz19_20min_10000/17Sep24_mixed_species_gene_matrix_preprocessed.txt", delimiter="\t")

# Convert sparse matrix to dense if necessary
if hasattr(adata.X, "toarray"):
    expr = adata.X.toarray()
else:
    expr = adata.X

# ---------------------------
# Step 2: Compute Pairwise Gene Correlations in Real Data
# ---------------------------
gene_corr_real = np.corrcoef(expr.T)  # Compute gene-gene correlations
corr_values = gene_corr_real[np.triu_indices_from(gene_corr_real, k=1)]  # Extract upper triangle (excluding diagonal)

# Apply filtering: Remove weak correlations (|r| ≤ 0.1)
filtered_corr_values = corr_values[np.abs(corr_values) > 0.05]

# Compute the mean of the filtered correlations
mean_corr_real = np.mean(np.abs(filtered_corr_values))

print("Filtered Mean pairwise gene correlation (real data):", mean_corr_real)

# ---------------------------
# Step 3: Monte Carlo Simulation (Shuffle Gene Expression in Each Cell)
# ---------------------------
n_sim = 1000  # Number of simulation iterations
simulated_means = []

for _ in range(n_sim):
    permuted_expr = expr.copy()

    # Shuffle gene expression within each cell
    for i in range(permuted_expr.shape[0]):
        np.random.shuffle(permuted_expr[i, :])

    # Compute gene-gene correlation on shuffled data
    gene_corr_shuffled = np.corrcoef(permuted_expr.T)
    shuffled_corr_values = gene_corr_shuffled[np.triu_indices_from(gene_corr_shuffled, k=1)]

    # Apply the same filtering to shuffled data
    filtered_shuffled_values = shuffled_corr_values[np.abs(shuffled_corr_values) > 0.05]
    mean_corr_shuffled = np.mean(np.abs(filtered_shuffled_values))

    simulated_means.append(mean_corr_shuffled)

# ---------------------------
# Step 4: Visualize and Assess the Result
# ---------------------------
plt.hist(simulated_means, bins=30, color='black', edgecolor='black')
plt.axvline(mean_corr_real, color='red', linestyle='dashed', linewidth=2, label='Observed Mean')
plt.xlabel("Mean pairwise gene correlation")
plt.xticks(np.arange(0, 0.4, 0.1))  # Tick marks from 0 to 1 in steps of 0.1
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Gene Co-Occurrence")
plt.legend()
plt.show()

# Compute p-value
# p_val = np.sum(np.array(simulated_means) >= mean_corr_real) / n_sim
p_val = (np.sum(np.array(simulated_means) >= mean_corr_real) + 1) / (n_sim + 1) # To ensure p_val never equals exactly 0, apply a "continuity correction" (or pseudo-count). This method is standard in Monte Carlo p-value estimation.

print(f"p-value: {p_val:.3e}")  # Show in scientific notation with 3 decimal places
