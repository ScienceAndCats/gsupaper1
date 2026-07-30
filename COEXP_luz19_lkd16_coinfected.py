"""
Co-occurrence of luz19 & lkd16 genes within coinfected cells, per timepoint.

What this script does
---------------------
1) Loads your mixed-species gene matrix (text or .h5ad).
2) Filters cells/genes by total counts.
3) Assigns each cell to a timepoint from its name.
4) Defines infection_state (Uninfected / Single infection / Coinfected)
   based on expression of genes with prefixes "luz19:" and "lkd16:".
5) For each TIMEPOINT:
   - Restrict to **coinfected** cells only.
   - For each luz19 gene (rows) and each lkd16 gene (columns):
       * Build a 2x2 table of binary detection (count>0).
       * Compute:
           - log2 odds ratio (with Haldane-Anscombe correction)
           - phi coefficient
           - Fisher's exact p-value
   - Apply Benjamini–Hochberg FDR correction over all gene pairs.
   - Save CSVs of:
       * log2_OR
       * phi
       * p_value
       * q_value
   - Save a heatmap PNG of log2_OR.

Coinfected cells are defined as cells where BOTH phages have total counts > 0
(based on raw counts layer).
"""

import os
import re
from copy import copy

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_DIR = "processed_data"
DATA_FILE = "JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

GRAPH_OUTPUT_DIR = "graph_outputs"

# Filtering
MIN_COUNTS_CELLS = 5
MIN_COUNTS_GENES = 5

# Optional normalization (does NOT affect co-occurrence, which uses raw counts)
DO_NORMALIZE = True
TARGET_SUM = 1e4

# Timepoints (order matters)
TIMEPOINT_ORDER = ["5min", "10min", "15min", "20min"]

# Phage gene prefixes
PHAGE_PREFIXES = ["luz19:", "lkd16:"]

# Threshold for calling "phage present" (based on raw counts)
PHAGE_PRESENT_THRESHOLD = 0.0

# Minimum coinfected cells needed at a timepoint to analyze
MIN_COINF_CELLS = 5

# =============================================================================
# STYLE / FORMATTING
# =============================================================================

STYLE = {
    "dpi": 180,
    "font_size": 11,

    "heatmap_figsize": (10, 8),
    "heatmap_cmap": "bwr",  # red = positive co-occurrence, blue = negative
    "heatmap_center_zero": True,  # center color scale at 0 for log2_OR
    "png_dpi": 220,
}

plt.rcParams.update({
    "figure.dpi": STYLE["dpi"],
    "font.size": STYLE["font_size"],
})

# =============================================================================
# HELPERS
# =============================================================================

def ensure_outdir():
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:180]


def save_png(fig, name: str):
    ensure_outdir()
    out_path = os.path.join(GRAPH_OUTPUT_DIR, sanitize_filename(name) + ".png")
    fig.savefig(out_path, bbox_inches="tight", dpi=STYLE["png_dpi"])
    plt.close(fig)
    print(f"Saved: {out_path}")


def to_dense_if_needed(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def classify_cell(cell_name: str) -> str:
    """Timepoint classifier based on your barcode convention."""
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "5min"
    elif bc1_value < 49:
        return "10min"
    elif bc1_value < 73:
        return "15min"
    else:
        return "20min"


def load_gene_matrix_to_adata(path: str) -> sc.AnnData:
    """Loads either .h5ad or tab-delimited gene matrix text file."""
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)
    raw = pd.read_csv(path, sep="\t", index_col=0)
    return sc.AnnData(raw)


def setup_adata() -> sc.AnnData:
    """Load, filter, define timepoints & infection_state, keep raw counts in layers['counts']."""
    adata = load_gene_matrix_to_adata(FILE_PATH)
    print(adata)

    # Filter on raw counts
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)
    sc.pp.filter_genes(adata, min_counts=MIN_COUNTS_GENES)

    # Preserve raw counts AFTER filtering so they align with genes/cells
    adata.layers["counts"] = adata.X.copy()

    # Optional normalize/log1p on adata.X (doesn't affect co-occurrence logic)
    if DO_NORMALIZE:
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

    # Timepoint assignment
    adata.obs["timepoint"] = [classify_cell(n) for n in adata.obs_names]
    adata.obs["timepoint"] = pd.Categorical(
        adata.obs["timepoint"], categories=TIMEPOINT_ORDER, ordered=True
    )

    # Phage expression sums from raw counts
    for phage in PHAGE_PREFIXES:
        mask = adata.var_names.str.contains(re.escape(phage))
        if mask.sum() == 0:
            print(f"WARNING: No genes matched prefix '{phage}'")

        X_counts = adata.layers["counts"]
        if sparse.issparse(X_counts):
            expr = adata[:, mask].layers["counts"].sum(axis=1).A.flatten()
        else:
            expr = np.asarray(adata[:, mask].layers["counts"].sum(axis=1)).ravel()

        adata.obs[f"{phage.strip(':')}_expression"] = expr

    # Infection encoding: 0=no phage, 1=only luz19, 2=only lkd16, 3=both
    adata.obs["phage_presence"] = (
        (adata.obs["luz19_expression"] > PHAGE_PRESENT_THRESHOLD).astype(int) * 1 +
        (adata.obs["lkd16_expression"] > PHAGE_PRESENT_THRESHOLD).astype(int) * 2
    )

    def infection_state(code: int) -> str:
        if code == 0:
            return "Uninfected"
        if code in (1, 2):
            return "Single infection"
        return "Coinfected"

    adata.obs["infection_state"] = [infection_state(int(x)) for x in adata.obs["phage_presence"]]
    adata.obs["infection_state"] = pd.Categorical(
        adata.obs["infection_state"],
        categories=["Uninfected", "Single infection", "Coinfected"],
        ordered=True
    )

    return adata


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    pvals: 1D array of p-values (may contain NaNs).
    Returns an array of q-values of the same shape (NaNs preserved).
    """
    p = pvals.copy().astype(float)
    q = np.full_like(p, np.nan, dtype=float)

    # Work on finite entries only
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q

    p_valid = p[mask]
    n = p_valid.size

    order = np.argsort(p_valid)
    ranks = np.arange(1, n + 1)

    q_temp = p_valid[order] * n / ranks
    q_temp = np.minimum.accumulate(q_temp[::-1])[::-1]  # enforce monotonicity

    q_valid = np.empty_like(p_valid)
    q_valid[order] = np.minimum(q_temp, 1.0)

    q[mask] = q_valid
    return q


# =============================================================================
# CO-OCCURRENCE ANALYSIS
# =============================================================================

def compute_cooccurrence_for_timepoint(adata: sc.AnnData, tp: str):
    """
    For one timepoint, restricted to coinfected cells:
      - build binary detection matrices for luz19 and lkd16 genes
      - compute 2x2 tables per gene pair
      - compute log2_OR, phi, p, q
      - save CSVs and return them
    """
    # Restrict to this timepoint + coinfected cells
    sub = adata[(adata.obs["timepoint"] == tp) & (adata.obs["infection_state"] == "Coinfected")]
    n_cells = sub.n_obs
    print(f"\nTimepoint {tp}: coinfected cells = {n_cells}")
    if n_cells < MIN_COINF_CELLS:
        print(f"  Skipping {tp}: only {n_cells} coinfected cells (min required = {MIN_COINF_CELLS})")
        return None

    # Identify luz19 and lkd16 genes
    var_names = sub.var_names
    luz_mask = var_names.str.contains(r"^luz19:")
    lkd_mask = var_names.str.contains(r"^lkd16:")

    luz_genes = list(var_names[luz_mask])
    lkd_genes = list(var_names[lkd_mask])

    print(f"  Found {len(luz_genes)} luz19 genes, {len(lkd_genes)} lkd16 genes at this timepoint.")
    if len(luz_genes) == 0 or len(lkd_genes) == 0:
        print("  Skipping: need at least one gene from each phage.")
        return None

    # Raw counts matrix
    X_counts = sub.layers["counts"]
    if sparse.issparse(X_counts):
        X_counts = X_counts.tocsr()

    # Boolean detection matrices: n_cells x n_genes
    if sparse.issparse(X_counts):
        X_luz = (X_counts[:, luz_mask] > 0).astype(int).toarray()
        X_lkd = (X_counts[:, lkd_mask] > 0).astype(int).toarray()
    else:
        X_luz = (X_counts[:, luz_mask] > 0).astype(int)
        X_lkd = (X_counts[:, lkd_mask] > 0).astype(int)

    n_luz = X_luz.shape[1]
    n_lkd = X_lkd.shape[1]

    # Prepare result matrices
    log2_or = np.full((n_luz, n_lkd), np.nan, dtype=float)
    phi = np.full((n_luz, n_lkd), np.nan, dtype=float)
    pval = np.full((n_luz, n_lkd), np.nan, dtype=float)

    # Compute 2x2 tables per gene pair
    for i in range(n_luz):
        L = X_luz[:, i].astype(bool)
        for j in range(n_lkd):
            K = X_lkd[:, j].astype(bool)

            a = np.sum(L & K)         # both detected
            b = np.sum(L & ~K)        # luz only
            c = np.sum(~L & K)        # lkd only
            d = np.sum(~L & ~K)       # neither

            # Skip completely degenerate tables
            if (a + b + c + d) == 0:
                continue

            # Fisher's exact test on raw counts (SciPy handles zeros)
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            pval[i, j] = p

            # Haldane-Anscombe correction for odds ratio
            a_h = a + 0.5
            b_h = b + 0.5
            c_h = c + 0.5
            d_h = d + 0.5
            or_h = (a_h * d_h) / (b_h * c_h)
            log2_or[i, j] = np.log2(or_h)

            # Phi coefficient
            denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            if denom > 0:
                phi[i, j] = (a * d - b * c) / denom
            else:
                phi[i, j] = np.nan

    # Benjamini–Hochberg FDR within this timepoint
    qval = benjamini_hochberg(pval.ravel()).reshape(pval.shape)

    # Build labeled DataFrames
    log2_or_df = pd.DataFrame(log2_or, index=luz_genes, columns=lkd_genes)
    phi_df = pd.DataFrame(phi, index=luz_genes, columns=lkd_genes)
    pval_df = pd.DataFrame(pval, index=luz_genes, columns=lkd_genes)
    qval_df = pd.DataFrame(qval, index=luz_genes, columns=lkd_genes)

    # Save CSVs
    ensure_outdir()
    for name, df in [
        (f"cooccurrence_log2OR_{tp}", log2_or_df),
        (f"cooccurrence_phi_{tp}", phi_df),
        (f"cooccurrence_pval_{tp}", pval_df),
        (f"cooccurrence_qval_{tp}", qval_df),
    ]:
        out_csv = os.path.join(GRAPH_OUTPUT_DIR, sanitize_filename(name) + ".csv")
        df.to_csv(out_csv)
        print(f"Saved: {out_csv}")

    return log2_or_df, phi_df, pval_df, qval_df


def plot_log2_or_heatmap(tp: str, log2_or_df: pd.DataFrame):
    """Plot a heatmap of log2 odds ratios for one timepoint."""
    if log2_or_df is None or log2_or_df.empty:
        return

    fig = plt.figure(figsize=STYLE["heatmap_figsize"])
    ax = fig.add_subplot(111)

    data = log2_or_df.values
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    cmap = plt.cm.get_cmap(STYLE["heatmap_cmap"] or "bwr")
    cmap = copy(cmap)
    cmap.set_bad(color="white")

    if STYLE.get("heatmap_center_zero", True):
        max_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -max_abs, max_abs

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(log2_or_df.shape[1]))
    ax.set_xticklabels(log2_or_df.columns, rotation=90)
    ax.set_yticks(np.arange(log2_or_df.shape[0]))
    ax.set_yticklabels(log2_or_df.index)

    ax.set_xlabel("lkd16 genes")
    ax.set_ylabel("luz19 genes")
    ax.set_title(f"log2 OR (detection co-occurrence) - {tp} coinfected cells")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(odds ratio)")

    fig.tight_layout()
    save_png(fig, f"log2OR_heatmap_{tp}_coinfected")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_outdir()
    adata = setup_adata()

    for tp in TIMEPOINT_ORDER:
        result = compute_cooccurrence_for_timepoint(adata, tp)
        if result is None:
            continue
        log2_or_df, phi_df, pval_df, qval_df = result
        plot_log2_or_heatmap(tp, log2_or_df)

    print(f"\nDone. Outputs saved in: {GRAPH_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
