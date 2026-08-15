"""
Co-occurrence of Luz19 & LKD16 genes within coinfected cells, per timepoint.

What this script does
---------------------
1) Loads your mixed-species gene matrix (text or .h5ad).
2) Filters cells/genes by total counts while retaining all Luz19/LKD16 genes.
3) Assigns each cell to a timepoint from its name.
4) Defines infection_state (Uninfected / Single infection / Coinfected)
   based on expression of genes with prefixes "luz19:" and "lkd16:".
5) Selects genes from an optional ordered text file, or defaults to all
   Luz19/LKD16 genes in matrix order.
6) For each TIMEPOINT:
   - Restrict to **coinfected** cells only.
   - For each Luz19 gene (rows) and each LKD16 gene (columns):
       * Build a 2x2 table of binary detection (count>0).
       * Compute:
           - log2 odds ratio (with Haldane-Anscombe correction)
           - phi coefficient
           - Fisher's exact p-value
   - Apply Benjamini-Hochberg FDR correction over all gene pairs.
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

# Optional text file containing genes to display, one gene per line.
# The relative order of Luz19 genes determines the row order, and the relative
# order of LKD16 genes determines the column order. Blank lines and lines
# beginning with # are ignored.
#
# The file may use either raw matrix names or cleaned names, for example:
#   luz19:gp23
#   Luz19:gp23
#   lkd16:PPLKD16_gp01
#   LKD16:gp01
#
# Set to None or "" to use every Luz19/LKD16 gene in matrix order.
# GENE_DISPLAY_FILE = None
GENE_DISPLAY_FILE = "heatmap_genes_displayed_sorted.txt"
# GENE_DISPLAY_FILE = os.path.join(DATA_DIR, "heatmap_gene_order.txt")

# Threshold for calling "phage present" (based on raw counts)
PHAGE_PRESENT_THRESHOLD = 0.0

# Minimum coinfected cells needed at a timepoint to analyze
MIN_COINF_CELLS = 5

# =============================================================================
# STYLE / FORMATTING
# =============================================================================

STYLE = {
    # Global matplotlib defaults
    "dpi": 180,
    "font_size": 11,

    # Heatmap appearance
    "heatmap_figsize": (10, 8),
    "heatmap_cmap": "bwr",  # red = positive co-occurrence, blue = negative
    "heatmap_center_zero": True,  # center color scale at 0 for log2_OR

    # Heatmap wording; {timepoint} is replaced automatically
    "heatmap_title": "log2 OR (detection co-occurrence) - {timepoint} coinfected cells",
    "heatmap_xlabel": "LKD16 genes",
    "heatmap_ylabel": "Luz19 genes",
    "heatmap_colorbar_label": "log2(odds ratio)",

    # Heatmap font sizes, controlled separately
    "heatmap_title_fontsize": 14,
    "heatmap_xlabel_fontsize": 12,
    "heatmap_ylabel_fontsize": 12,
    "heatmap_xtick_fontsize": 9,
    "heatmap_ytick_fontsize": 9,
    "heatmap_colorbar_label_fontsize": 11,
    "heatmap_colorbar_tick_fontsize": 9,

    # Tick-label rotations
    "heatmap_xtick_rotation": 90,
    "heatmap_ytick_rotation": 0,

    # Save options
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


def is_luz19_gene(gene: str) -> bool:
    return str(gene).lower().startswith("luz19:")


def is_lkd16_gene(gene: str) -> bool:
    return str(gene).lower().startswith("lkd16:")


def is_phage_gene(gene: str) -> bool:
    """Return True for a configured Luz19 or LKD16 gene."""
    gene_lower = str(gene).lower()
    return any(gene_lower.startswith(prefix.lower()) for prefix in PHAGE_PREFIXES)


def format_gene_display_name(gene: str) -> str:
    """Return a cleaner label without changing the underlying matrix gene ID.

    Examples
    --------
    lkd16:PPLKD16_gp01 -> LKD16:gp01
    luz19:gp23         -> Luz19:gp23
    luz19:PPLUZ19_PPLUZ19_gp20 -> Luz19:gp20
    """
    gene = str(gene)
    if ":" not in gene:
        return gene

    prefix, suffix = gene.split(":", 1)
    prefix_lower = prefix.lower()

    if prefix_lower == "lkd16":
        suffix = re.sub(r"^(?:PPLKD16_)+", "", suffix, flags=re.IGNORECASE)
        return f"LKD16:{suffix}"

    if prefix_lower == "luz19":
        suffix = re.sub(r"^(?:PPLUZ19_)+", "", suffix, flags=re.IGNORECASE)
        return f"Luz19:{suffix}"

    return gene


def make_unique_display_labels(genes: list[str]) -> list[str]:
    """Create cleaned labels and disambiguate only if two labels collapse."""
    base_labels = [format_gene_display_name(gene) for gene in genes]
    totals = {}
    for label in base_labels:
        totals[label] = totals.get(label, 0) + 1

    seen = {}
    labels = []
    for label in base_labels:
        if totals[label] == 1:
            labels.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        labels.append(f"{label} [{seen[label]}]")
    return labels


def read_requested_gene_list(path: str) -> list[str]:
    """Read one gene per line while preserving order and removing duplicates."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gene display file not found: {path}")

    genes = []
    seen = set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            gene = line.strip()
            if not gene or gene.startswith("#"):
                continue
            if gene in seen:
                print(
                    f"WARNING: Duplicate gene '{gene}' on line {line_number} "
                    "was ignored; the first occurrence determines its position."
                )
                continue
            genes.append(gene)
            seen.add(gene)

    if not genes:
        raise ValueError(f"No gene names were found in gene display file: {path}")
    return genes


def build_gene_name_lookups(var_names: list[str]):
    exact_names = set(var_names)
    lower_to_actual = {}
    display_to_actual = {}

    for gene in var_names:
        lower_to_actual.setdefault(gene.lower(), []).append(gene)
        display_name = format_gene_display_name(gene)
        display_to_actual.setdefault(display_name.lower(), []).append(gene)

    return exact_names, lower_to_actual, display_to_actual


def resolve_gene_name(
    requested: str,
    exact_names: set[str],
    lower_to_actual: dict[str, list[str]],
    display_to_actual: dict[str, list[str]],
):
    """Resolve either a raw matrix ID or its cleaned display-name alias."""
    if requested in exact_names:
        return requested

    matches = lower_to_actual.get(requested.lower(), [])
    match_type = "case-insensitive raw-name comparison"

    if not matches:
        matches = display_to_actual.get(requested.lower(), [])
        match_type = "cleaned display-name alias"

    if len(matches) == 1:
        actual = matches[0]
        if requested != actual:
            print(
                f"Matched requested gene '{requested}' to matrix gene '{actual}' "
                f"using {match_type}."
            )
        return actual

    if len(matches) > 1:
        print(
            f"WARNING: Requested gene '{requested}' resolves to multiple matrix genes "
            f"and was skipped: {matches}"
        )
    return None


def select_cooccurrence_genes(
    adata: sc.AnnData,
    gene_file: str | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Select ordered genes for the co-occurrence rows and columns.

    Returns
    -------
    displayed_genes
        Combined selected list in the supplied order.
    luz_genes
        Selected Luz19 genes in their relative supplied order.
    lkd_genes
        Selected LKD16 genes in their relative supplied order.
    missing_genes
        Requested names that could not be matched.
    """
    var_names = [str(gene) for gene in adata.var_names]
    exact_names, lower_to_actual, display_to_actual = build_gene_name_lookups(var_names)

    if gene_file:
        requested = read_requested_gene_list(gene_file)
        displayed = []
        missing = []
        already_added = set()

        for requested_gene in requested:
            actual = resolve_gene_name(
                requested_gene,
                exact_names,
                lower_to_actual,
                display_to_actual,
            )
            if actual is None:
                missing.append(requested_gene)
                continue

            if not is_phage_gene(actual):
                print(
                    f"WARNING: Requested gene '{requested_gene}' is not a Luz19 or "
                    "LKD16 gene and was ignored."
                )
                continue

            if actual in already_added:
                print(
                    f"WARNING: '{requested_gene}' resolves to the already selected "
                    f"gene '{actual}' and was ignored."
                )
                continue

            displayed.append(actual)
            already_added.add(actual)

        source_description = f"custom list: {gene_file}"
    else:
        displayed = [gene for gene in var_names if is_phage_gene(gene)]
        missing = []
        source_description = "all Luz19/LKD16 genes in matrix order"

    luz_genes = [gene for gene in displayed if is_luz19_gene(gene)]
    lkd_genes = [gene for gene in displayed if is_lkd16_gene(gene)]

    if not luz_genes or not lkd_genes:
        raise ValueError(
            "The co-occurrence heatmap requires at least one Luz19 gene and one "
            "LKD16 gene. Check GENE_DISPLAY_FILE and the matrix gene names."
        )

    if missing:
        print(
            f"WARNING: {len(missing)} requested genes were not found and will not "
            "be displayed."
        )

    print(
        f"Selected {len(luz_genes)} Luz19 genes and {len(lkd_genes)} LKD16 genes "
        f"from {source_description}."
    )
    return displayed, luz_genes, lkd_genes, missing


def write_displayed_gene_lists(
    displayed_genes: list[str],
    luz_genes: list[str],
    lkd_genes: list[str],
    missing_genes: list[str],
):
    """Write the cleaned displayed names, separate phage lists, and name mapping."""
    ensure_outdir()

    combined_labels = [format_gene_display_name(gene) for gene in displayed_genes]
    luz_labels = [format_gene_display_name(gene) for gene in luz_genes]
    lkd_labels = [format_gene_display_name(gene) for gene in lkd_genes]

    output_lists = [
        ("heatmap_genes_displayed.txt", combined_labels),
        ("heatmap_luz19_genes_displayed.txt", luz_labels),
        ("heatmap_lkd16_genes_displayed.txt", lkd_labels),
    ]
    for filename, labels in output_lists:
        path = os.path.join(GRAPH_OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(labels) + "\n")
        print(f"Saved: {path}")

    mapping_path = os.path.join(GRAPH_OUTPUT_DIR, "heatmap_gene_name_mapping.tsv")
    pd.DataFrame({
        "phage": [
            "Luz19" if is_luz19_gene(gene) else "LKD16"
            for gene in displayed_genes
        ],
        "matrix_gene_name": displayed_genes,
        "display_gene_name": combined_labels,
    }).to_csv(mapping_path, sep="\t", index=False)
    print(f"Saved: {mapping_path}")

    missing_path = os.path.join(GRAPH_OUTPUT_DIR, "heatmap_genes_missing_from_matrix.txt")
    if missing_genes:
        with open(missing_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(missing_genes) + "\n")
        print(f"Saved: {missing_path}")
    elif os.path.exists(missing_path):
        os.remove(missing_path)
        print(f"Removed stale file: {missing_path}")


def setup_adata() -> sc.AnnData:
    """Load, filter, define timepoints & infection_state, keep raw counts in layers['counts']."""
    adata = load_gene_matrix_to_adata(FILE_PATH)
    print(adata)

    # Filter cells on raw counts.
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)

    # Apply the general gene-count filter, but always retain every Luz19/LKD16
    # gene so the default really does include all phage genes in the matrix.
    X_for_filter = adata.X
    if sparse.issparse(X_for_filter):
        total_gene_counts = np.asarray(X_for_filter.sum(axis=0)).ravel()
    else:
        total_gene_counts = np.asarray(X_for_filter.sum(axis=0)).ravel()

    keep_gene_mask = total_gene_counts >= MIN_COUNTS_GENES
    phage_gene_mask = np.asarray([is_phage_gene(gene) for gene in adata.var_names])
    keep_gene_mask = keep_gene_mask | phage_gene_mask

    n_removed = int((~keep_gene_mask).sum())
    n_phage_retained_below_threshold = int(
        (phage_gene_mask & (total_gene_counts < MIN_COUNTS_GENES)).sum()
    )

    adata = adata[:, keep_gene_mask].copy()
    print(
        f"Removed {n_removed} non-phage genes with fewer than "
        f"{MIN_COUNTS_GENES} total counts."
    )
    if n_phage_retained_below_threshold:
        print(
            f"Retained {n_phage_retained_below_threshold} low-count Luz19/LKD16 "
            "genes for complete phage analysis."
        )

    # Preserve raw counts AFTER filtering so they align with genes/cells.
    adata.layers["counts"] = adata.X.copy()

    # Optional normalize/log1p on adata.X (doesn't affect co-occurrence logic).
    if DO_NORMALIZE:
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

    # Timepoint assignment.
    adata.obs["timepoint"] = [classify_cell(n) for n in adata.obs_names]
    adata.obs["timepoint"] = pd.Categorical(
        adata.obs["timepoint"], categories=TIMEPOINT_ORDER, ordered=True
    )

    # Phage expression sums from raw counts.
    for phage in PHAGE_PREFIXES:
        mask = np.asarray([
            str(gene).lower().startswith(phage.lower())
            for gene in adata.var_names
        ])
        if mask.sum() == 0:
            print(f"WARNING: No genes matched prefix '{phage}'")

        X_counts = adata.layers["counts"]
        if sparse.issparse(X_counts):
            expr = adata[:, mask].layers["counts"].sum(axis=1).A.flatten()
        else:
            expr = np.asarray(
                adata[:, mask].layers["counts"].sum(axis=1)
            ).ravel()

        adata.obs[f"{phage.strip(':')}_expression"] = expr

    # Infection encoding: 0=no phage, 1=only luz19, 2=only lkd16, 3=both.
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

    adata.obs["infection_state"] = [
        infection_state(int(x)) for x in adata.obs["phage_presence"]
    ]
    adata.obs["infection_state"] = pd.Categorical(
        adata.obs["infection_state"],
        categories=["Uninfected", "Single infection", "Coinfected"],
        ordered=True,
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

    # Work on finite entries only.
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

def compute_cooccurrence_for_timepoint(
    adata: sc.AnnData,
    tp: str,
    luz_genes: list[str],
    lkd_genes: list[str],
):
    """
    For one timepoint, restricted to coinfected cells:
      - build binary detection matrices for selected Luz19 and LKD16 genes
      - compute 2x2 tables per gene pair
      - compute log2_OR, phi, p, q
      - save CSVs and return them
    """
    # Restrict to this timepoint + coinfected cells.
    sub = adata[
        (adata.obs["timepoint"] == tp) &
        (adata.obs["infection_state"] == "Coinfected")
    ]
    n_cells = sub.n_obs
    print(f"\nTimepoint {tp}: coinfected cells = {n_cells}")
    if n_cells < MIN_COINF_CELLS:
        print(
            f"  Skipping {tp}: only {n_cells} coinfected cells "
            f"(min required = {MIN_COINF_CELLS})"
        )
        return None

    # Use the selected genes in the exact requested order.
    gene_to_idx = {str(gene): i for i, gene in enumerate(sub.var_names)}
    luz_genes_present = [gene for gene in luz_genes if gene in gene_to_idx]
    lkd_genes_present = [gene for gene in lkd_genes if gene in gene_to_idx]

    print(
        f"  Using {len(luz_genes_present)} Luz19 genes and "
        f"{len(lkd_genes_present)} LKD16 genes at this timepoint."
    )
    if len(luz_genes_present) == 0 or len(lkd_genes_present) == 0:
        print("  Skipping: need at least one gene from each phage.")
        return None

    luz_idxs = [gene_to_idx[gene] for gene in luz_genes_present]
    lkd_idxs = [gene_to_idx[gene] for gene in lkd_genes_present]

    # Raw counts matrix.
    X_counts = sub.layers["counts"]
    if sparse.issparse(X_counts):
        X_counts = X_counts.tocsr()

    # Boolean detection matrices: n_cells x n_genes.
    if sparse.issparse(X_counts):
        X_luz = (X_counts[:, luz_idxs] > 0).astype(int).toarray()
        X_lkd = (X_counts[:, lkd_idxs] > 0).astype(int).toarray()
    else:
        X_luz = (X_counts[:, luz_idxs] > 0).astype(int)
        X_lkd = (X_counts[:, lkd_idxs] > 0).astype(int)

    n_luz = X_luz.shape[1]
    n_lkd = X_lkd.shape[1]

    # Prepare result matrices.
    log2_or = np.full((n_luz, n_lkd), np.nan, dtype=float)
    phi = np.full((n_luz, n_lkd), np.nan, dtype=float)
    pval = np.full((n_luz, n_lkd), np.nan, dtype=float)

    # Compute 2x2 tables per gene pair.
    for i in range(n_luz):
        L = X_luz[:, i].astype(bool)
        for j in range(n_lkd):
            K = X_lkd[:, j].astype(bool)

            a = np.sum(L & K)          # both detected
            b = np.sum(L & ~K)         # Luz19 only
            c = np.sum(~L & K)         # LKD16 only
            d = np.sum(~L & ~K)        # neither

            # Skip completely degenerate tables.
            if (a + b + c + d) == 0:
                continue

            # Fisher's exact test on raw counts (SciPy handles zeros).
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            pval[i, j] = p

            # Haldane-Anscombe correction for odds ratio.
            a_h = a + 0.5
            b_h = b + 0.5
            c_h = c + 0.5
            d_h = d + 0.5
            or_h = (a_h * d_h) / (b_h * c_h)
            log2_or[i, j] = np.log2(or_h)

            # Phi coefficient.
            denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            if denom > 0:
                phi[i, j] = (a * d - b * c) / denom
            else:
                phi[i, j] = np.nan

    # Benjamini-Hochberg FDR within this timepoint.
    qval = benjamini_hochberg(pval.ravel()).reshape(pval.shape)

    # Use cleaned, user-facing names in CSVs and heatmaps.
    luz_labels = make_unique_display_labels(luz_genes_present)
    lkd_labels = make_unique_display_labels(lkd_genes_present)

    # Build labeled DataFrames.
    log2_or_df = pd.DataFrame(log2_or, index=luz_labels, columns=lkd_labels)
    phi_df = pd.DataFrame(phi, index=luz_labels, columns=lkd_labels)
    pval_df = pd.DataFrame(pval, index=luz_labels, columns=lkd_labels)
    qval_df = pd.DataFrame(qval, index=luz_labels, columns=lkd_labels)

    # Save CSVs.
    ensure_outdir()
    for name, df in [
        (f"cooccurrence_log2OR_{tp}", log2_or_df),
        (f"cooccurrence_phi_{tp}", phi_df),
        (f"cooccurrence_pval_{tp}", pval_df),
        (f"cooccurrence_qval_{tp}", qval_df),
    ]:
        out_csv = os.path.join(
            GRAPH_OUTPUT_DIR,
            sanitize_filename(name) + ".csv",
        )
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
    ax.set_xticklabels(
        log2_or_df.columns,
        rotation=STYLE["heatmap_xtick_rotation"],
        fontsize=STYLE["heatmap_xtick_fontsize"],
    )
    ax.set_yticks(np.arange(log2_or_df.shape[0]))
    ax.set_yticklabels(
        log2_or_df.index,
        rotation=STYLE["heatmap_ytick_rotation"],
        fontsize=STYLE["heatmap_ytick_fontsize"],
    )

    ax.set_xlabel(
        STYLE["heatmap_xlabel"],
        fontsize=STYLE["heatmap_xlabel_fontsize"],
    )
    ax.set_ylabel(
        STYLE["heatmap_ylabel"],
        fontsize=STYLE["heatmap_ylabel_fontsize"],
    )
    ax.set_title(
        STYLE["heatmap_title"].format(timepoint=tp),
        fontsize=STYLE["heatmap_title_fontsize"],
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        STYLE["heatmap_colorbar_label"],
        fontsize=STYLE["heatmap_colorbar_label_fontsize"],
    )
    cbar.ax.tick_params(labelsize=STYLE["heatmap_colorbar_tick_fontsize"])

    fig.tight_layout()
    save_png(fig, f"log2OR_heatmap_{tp}_coinfected")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_outdir()
    adata = setup_adata()

    displayed_genes, luz_genes, lkd_genes, missing_genes = select_cooccurrence_genes(
        adata,
        GENE_DISPLAY_FILE,
    )
    write_displayed_gene_lists(
        displayed_genes,
        luz_genes,
        lkd_genes,
        missing_genes,
    )

    for tp in TIMEPOINT_ORDER:
        result = compute_cooccurrence_for_timepoint(
            adata,
            tp,
            luz_genes,
            lkd_genes,
        )
        if result is None:
            continue
        log2_or_df, phi_df, pval_df, qval_df = result
        plot_log2_or_heatmap(tp, log2_or_df)

    print(f"\nDone. Outputs saved in: {GRAPH_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
