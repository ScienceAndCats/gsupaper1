"""
Heatmaps of mean gene expression by infection state, one heatmap per timepoint,
plus raw-support tables (n_cells, raw hits, mean hits/cell),
PLUS statistical tests: coinfected vs single-infected (per phage, per timepoint).

What gets tested:
- For luz19 genes: Coinfected vs Only Luz19
- For lkd16 genes: Coinfected vs Only LKD16

Tests:
- Mann–Whitney U on per-cell expression values (default uses normalized log1p in adata.X)
- Fisher exact test on detection rate (>0 raw counts)
- Benjamini–Hochberg FDR correction across genes

Outputs (in graph_outputs/):
- heatmap_genes_displayed.txt
- heatmap_gene_name_mapping.tsv
- heatmap_genes_missing_from_matrix.txt (only when requested genes are absent)
- gene_support_table_<tp>.csv
- n_cells_<tp>.csv
- coinfection_gene_tests_<tp>_<phage>.csv
- coinfection_gene_tests_ALL.csv
- coinfection_phage_totals_tests.csv
"""

import os
import re
from copy import copy

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt

# NEW: stats tests
from scipy.stats import mannwhitneyu, fisher_exact


# =============================================================================
# USER SETTINGS
# =============================================================================
DATA_DIR = "processed_data"
DATA_FILE = "JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

GRAPH_OUTPUT_DIR = "graph_outputs"

# Filtering
MIN_COUNTS_CELLS = 4
MIN_COUNTS_GENES = 4

# Normalization
DO_NORMALIZE = True     # normalize_total + log1p
TARGET_SUM = 1e4

# Timepoints (order matters)
TIMEPOINT_ORDER = ["5min", "10min", "15min", "20min"]

# Timepoints used *for heatmaps* (subset or re-ordering of TIMEPOINT_ORDER)
HEATMAP_TIMEPOINTS = ["15min", "20min"]  #["5min", "10min", "15min", "20min"]

# Phage gene prefixes used when no custom gene list is supplied
PHAGE_PREFIXES = ["luz19:", "lkd16:"]

# Optional text file containing genes to display, one gene per line.
# The heatmaps will follow this exact order. Blank lines and lines beginning
# with # are ignored. Set to None or "" to display every Luz19/LKD16 gene.
# Examples:
GENE_DISPLAY_FILE = "heatmap_genes_displayed_sorted.txt"
# GENE_DISPLAY_FILE = os.path.join(DATA_DIR, "heatmap_gene_order.txt")
#GENE_DISPLAY_FILE = None

# Heatmap dimensions are calculated from the number of displayed genes.
HEATMAP_WIDTH = 10
HEATMAP_HEIGHT_PER_GENE = 0.22
HEATMAP_MIN_HEIGHT = 6

# Threshold for calling "phage present" (phage_expression > threshold)
PHAGE_PRESENT_THRESHOLD = 0.0


# =============================================================================
# STYLE / FORMATTING + TOGGLES
# =============================================================================
STYLE = {
    # Global matplotlib
    "dpi": 180,
    "font_size": 11,

    # Mean-expression heatmap (normalized/log1p mean if DO_NORMALIZE)
    "mean_heatmap_cmap": "viridis",
    "mean_heatmap_share_color_scale": True,
    "mean_title": "Mean expression heatmap at {timepoint}",
    "mean_xlabel": "", #"Infection state"
    "mean_ylabel": "Gene",
    "mean_colorbar_label": "Mean normalized log1p expression",
    "mean_title_fontsize": 32,
    "mean_xlabel_fontsize": 24,
    "mean_ylabel_fontsize": 24,
    "mean_xtick_fontsize": 16,
    "mean_ytick_fontsize": 12,
    "mean_colorbar_label_fontsize": 18,
    "mean_colorbar_tick_fontsize": 14,

    # Mean-hits-per-cell heatmap (raw counts / n_cells)
    "make_hits_per_cell_heatmaps": True,
    "hits_per_cell_cmap": "magma",
    "hits_per_cell_share_color_scale": True,
    "hits_title": "Mean hits/cell (raw) heatmap at {timepoint}",
    "hits_xlabel": "Infection state",
    "hits_ylabel": "Gene",
    "hits_colorbar_label": "Mean raw hits per cell",
    "hits_title_fontsize": 14,
    "hits_xlabel_fontsize": 12,
    "hits_ylabel_fontsize": 12,
    "hits_xtick_fontsize": 10,
    "hits_ytick_fontsize": 9,
    "hits_colorbar_label_fontsize": 11,
    "hits_colorbar_tick_fontsize": 9,

    # Annotate numbers on hits-per-cell heatmap blocks
    "annotate_hits_per_cell": True,
    "annotate_fmt": "{:.2f}",
    "annotate_fontsize": 8,
    "annotate_max_genes": 60,

    # Total-count QC boxplots
    "total_counts_title": "Total raw counts per cell by infection state ({timepoint})",
    "total_counts_xlabel": "Infection state",
    "total_counts_ylabel": "Total raw counts per cell",
    "total_counts_title_fontsize": 14,
    "total_counts_xlabel_fontsize": 12,
    "total_counts_ylabel_fontsize": 12,
    "total_counts_xtick_fontsize": 10,
    "total_counts_ytick_fontsize": 10,

    # Save options
    "png_dpi": 220,

    # Tables
    "write_tables": True,

    # NEW: stats testing toggles
    "run_coinfection_stats": True,
    # Test expression on normalized/log1p values (adata.X). Recommended.
    # If False, will test on raw counts per cell (can be confounded by library size).
    "stats_use_normalized_X": True,
    # Minimum cells required in each group to run a gene-level test
    "stats_min_cells_per_group": 10,
    # Whether to run Fisher exact on detection (>0 raw counts)
    "stats_run_detection_test": True,
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


def mean_by_group(adata_sub: sc.AnnData, layer: str | None = None) -> np.ndarray:
    X = adata_sub.layers[layer] if layer else adata_sub.X
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X.mean(axis=0)).ravel()


def sum_by_group(adata_sub: sc.AnnData, layer: str) -> np.ndarray:
    X = adata_sub.layers[layer]
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=0)).ravel()
    return np.asarray(X.sum(axis=0)).ravel()


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


def load_gene_matrix_to_adata(path: str) -> sc.AnnData:
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)
    raw = pd.read_csv(path, sep="\t", index_col=0)
    return sc.AnnData(raw)


def is_phage_gene(gene: str) -> bool:
    """Return True when a gene name begins with a configured phage prefix."""
    gene_lower = str(gene).lower()
    return any(gene_lower.startswith(prefix.lower()) for prefix in PHAGE_PREFIXES)


def format_gene_display_name(gene: str) -> str:
    """Return a cleaner label without changing the underlying matrix gene ID.

    Examples
    --------
    lkd16:PPLKD16_gp01 -> LKD16:gp01
    luz19:gp23         -> Luz19:gp23

    Repeated annotation prefixes are also removed, for example:
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
            print(f"Matched requested gene '{requested}' to matrix gene '{actual}' using {match_type}.")
        return actual

    if len(matches) > 1:
        print(
            f"WARNING: Requested gene '{requested}' resolves to multiple matrix genes "
            f"and was skipped: {matches}"
        )
    return None


def select_heatmap_genes(adata: sc.AnnData, gene_file: str | None = None) -> tuple[list[str], list[str]]:
    """
    Choose genes for both heatmap types.

    If gene_file is supplied, use its order exactly after removing missing genes.
    Otherwise, use every Luz19/LKD16 gene in the order found in adata.var_names.

    Returns
    -------
    displayed_genes, missing_requested_genes
    """
    var_names = [str(g) for g in adata.var_names]
    exact_names = set(var_names)
    lower_to_actual = {}
    display_to_actual = {}
    for gene in var_names:
        lower_to_actual.setdefault(gene.lower(), []).append(gene)
        display_name = format_gene_display_name(gene)
        display_to_actual.setdefault(display_name.lower(), []).append(gene)

    if gene_file:
        requested = read_requested_gene_list(gene_file)
        displayed = []
        missing = []
        already_added = set()

        for gene in requested:
            actual = resolve_gene_name(
                gene, exact_names, lower_to_actual, display_to_actual
            )
            if actual is None:
                missing.append(gene)
                continue
            if actual in already_added:
                print(
                    f"WARNING: '{gene}' resolves to the already selected gene '{actual}' "
                    "and was ignored."
                )
                continue
            displayed.append(actual)
            already_added.add(actual)

        if missing:
            print(
                f"WARNING: {len(missing)} requested genes were not found after filtering "
                "and will not be displayed."
            )
        source_description = f"custom list: {gene_file}"
    else:
        displayed = [gene for gene in var_names if is_phage_gene(gene)]
        missing = []
        source_description = "all Luz19/LKD16 genes in matrix order"

    if not displayed:
        raise ValueError(
            "No genes are available for the heatmaps. Either provide valid names in "
            "GENE_DISPLAY_FILE or confirm that matrix gene names begin with one of "
            f"these prefixes: {PHAGE_PREFIXES}"
        )

    print(f"Selected {len(displayed)} heatmap genes from {source_description}.")
    return displayed, missing


def write_displayed_gene_lists(displayed_genes: list[str], missing_genes: list[str]):
    """Write the final heatmap order and, when applicable, missing requested genes."""
    ensure_outdir()

    displayed_path = os.path.join(GRAPH_OUTPUT_DIR, "heatmap_genes_displayed.txt")
    displayed_labels = [format_gene_display_name(gene) for gene in displayed_genes]
    with open(displayed_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(displayed_labels) + "\n")
    print(f"Saved: {displayed_path}")

    mapping_path = os.path.join(GRAPH_OUTPUT_DIR, "heatmap_gene_name_mapping.tsv")
    pd.DataFrame({
        "matrix_gene_name": displayed_genes,
        "display_gene_name": displayed_labels,
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


def heatmap_figsize(n_genes: int) -> tuple[float, float]:
    """Scale heatmap height to the number of displayed genes."""
    height = max(HEATMAP_MIN_HEIGHT, HEATMAP_HEIGHT_PER_GENE * n_genes + 4)
    return HEATMAP_WIDTH, height


# NEW: Benjamini–Hochberg FDR
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out

    p_ok = p[ok]
    n = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]

    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    q_back = np.empty_like(q)
    q_back[order] = q
    out[ok] = q_back
    return out


# NEW: pull per-cell gene vector without densifying everything
def get_gene_vector(adata_sub: sc.AnnData, gene_idx: int, use_normalized_X: bool) -> np.ndarray:
    if use_normalized_X:
        X = adata_sub.X
    else:
        X = adata_sub.layers["counts"]

    if sparse.issparse(X):
        return np.asarray(X[:, gene_idx].todense()).ravel()
    return np.asarray(X[:, gene_idx]).ravel()


# =============================================================================
# SETUP ADATA
# =============================================================================
def setup_adata() -> sc.AnnData:
    adata = load_gene_matrix_to_adata(FILE_PATH)
    print(adata)

    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)

    # Apply the general gene-count filter, but ALWAYS retain:
    #   1. every Luz19/LKD16 gene, and
    #   2. every gene requested in GENE_DISPLAY_FILE.
    # This ensures "all phage genes" really means all phage genes present in
    # the input matrix, including genes with fewer than MIN_COUNTS_GENES hits.
    X_for_filter = adata.X
    if sparse.issparse(X_for_filter):
        total_gene_counts = np.asarray(X_for_filter.sum(axis=0)).ravel()
    else:
        total_gene_counts = np.asarray(X_for_filter.sum(axis=0)).ravel()

    keep_gene_mask = total_gene_counts >= MIN_COUNTS_GENES
    phage_gene_mask = np.asarray([is_phage_gene(gene) for gene in adata.var_names])
    keep_gene_mask = keep_gene_mask | phage_gene_mask

    if GENE_DISPLAY_FILE:
        requested_for_retention = {
            gene.lower() for gene in read_requested_gene_list(GENE_DISPLAY_FILE)
        }
        requested_gene_mask = np.asarray([
            str(gene).lower() in requested_for_retention for gene in adata.var_names
        ])
        keep_gene_mask = keep_gene_mask | requested_gene_mask
    else:
        requested_gene_mask = np.zeros(adata.n_vars, dtype=bool)

    n_removed = int((~keep_gene_mask).sum())
    n_phage_retained_below_threshold = int((phage_gene_mask & (total_gene_counts < MIN_COUNTS_GENES)).sum())
    n_requested_retained_below_threshold = int(
        (requested_gene_mask & ~phage_gene_mask & (total_gene_counts < MIN_COUNTS_GENES)).sum()
    )

    adata = adata[:, keep_gene_mask].copy()
    print(f"Removed {n_removed} non-selected genes with fewer than {MIN_COUNTS_GENES} total counts.")
    if n_phage_retained_below_threshold:
        print(
            f"Retained {n_phage_retained_below_threshold} low-count Luz19/LKD16 genes "
            "for complete phage heatmaps."
        )
    if n_requested_retained_below_threshold:
        print(
            f"Retained {n_requested_retained_below_threshold} additional low-count genes "
            "because they were requested in GENE_DISPLAY_FILE."
        )

    # Preserve raw counts AFTER filtering
    adata.layers["counts"] = adata.X.copy()

    if DO_NORMALIZE:
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

    adata.obs["timepoint"] = [classify_cell(n) for n in adata.obs_names]
    adata.obs["timepoint"] = pd.Categorical(
        adata.obs["timepoint"], categories=TIMEPOINT_ORDER, ordered=True
    )

    # Phage expression sums (raw counts)
    for phage in PHAGE_PREFIXES:
        mask = np.asarray([
            str(gene).lower().startswith(phage.lower()) for gene in adata.var_names
        ])
        if mask.sum() == 0:
            print(f"WARNING: No genes matched prefix '{phage}'")

        X_counts = adata.layers["counts"]
        if sparse.issparse(X_counts):
            expr = adata[:, mask].layers["counts"].sum(axis=1).A.flatten()
        else:
            expr = np.asarray(adata[:, mask].layers["counts"].sum(axis=1)).ravel()

        adata.obs[f"{phage.strip(':')}_expression"] = expr

    # Infection encoding: 0=no phage, 1=Only Luz19, 2=Only LKD16, 3=both
    adata.obs["phage_presence"] = (
        (adata.obs["luz19_expression"] > PHAGE_PRESENT_THRESHOLD).astype(int) * 1 +
        (adata.obs["lkd16_expression"] > PHAGE_PRESENT_THRESHOLD).astype(int) * 2
    )

    # 3-state used in heatmaps
    def infection_state3(code: int) -> str:
        if code == 0:
            return "Uninfected"
        if code in (1, 2):
            return "Single infection"
        return "Coinfected"

    adata.obs["infection_state"] = [infection_state3(int(x)) for x in adata.obs["phage_presence"]]
    adata.obs["infection_state"] = pd.Categorical(
        adata.obs["infection_state"],
        categories=["Uninfected", "Single infection", "Coinfected"],
        ordered=True
    )

    # NEW: 4-state used for stats (phage-specific single)
    def phage_state(code: int) -> str:
        if code == 0:
            return "Uninfected"
        if code == 1:
            return "Only Luz19"
        if code == 2:
            return "Only LKD16"
        return "Coinfected"

    adata.obs["phage_state"] = [phage_state(int(x)) for x in adata.obs["phage_presence"]]
    adata.obs["phage_state"] = pd.Categorical(
        adata.obs["phage_state"],
        categories=["Uninfected", "Only Luz19", "Only LKD16", "Coinfected"],
        ordered=True
    )

    # NEW: total counts per cell (raw) — useful sanity check / optional stratification later
    Xc = adata.layers["counts"]
    if sparse.issparse(Xc):
        adata.obs["total_counts_raw"] = np.asarray(Xc.sum(axis=1)).ravel()
    else:
        adata.obs["total_counts_raw"] = np.asarray(Xc.sum(axis=1)).ravel()

    return adata


# =============================================================================
# COMPUTE MATRICES + TABLES PER TIMEPOINT
# =============================================================================
def compute_tp_matrices_and_tables(adata: sc.AnnData, genes: list[str]):
    conds = ["Uninfected", "Single infection", "Coinfected"]

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    genes_present = [g for g in genes if g in gene_to_idx]
    if len(genes_present) == 0:
        raise ValueError("No selected genes found in adata.var_names.")
    gene_idxs = [gene_to_idx[g] for g in genes_present]

    mats_mean = {}
    mats_hits = {}
    mats_hpc = {}
    ncells_by_tp = {}
    tables_by_tp = {}

    for tp in TIMEPOINT_ORDER:
        ad_tp = adata[adata.obs["timepoint"] == tp]
        if ad_tp.n_obs == 0:
            continue

        mean_mat = np.full((len(genes_present), len(conds)), np.nan, dtype=float)
        hits_mat = np.full((len(genes_present), len(conds)), np.nan, dtype=float)
        hpc_mat = np.full((len(genes_present), len(conds)), np.nan, dtype=float)

        ncells = {}

        for j, c in enumerate(conds):
            sub = ad_tp[ad_tp.obs["infection_state"] == c]
            n = int(sub.n_obs)
            ncells[c] = n
            if n == 0:
                continue

            mean_mat[:, j] = mean_by_group(sub)[gene_idxs]
            raw_hits = sum_by_group(sub, layer="counts")[gene_idxs]
            hits_mat[:, j] = raw_hits
            hpc_mat[:, j] = raw_hits / n

        mats_mean[tp] = mean_mat
        mats_hits[tp] = hits_mat
        mats_hpc[tp] = hpc_mat
        ncells_by_tp[tp] = ncells

        data = {}
        for c in conds:
            data[f"{c}__n_cells"] = [ncells[c]] * len(genes_present)
            data[f"{c}__raw_hits"] = hits_mat[:, conds.index(c)]
            data[f"{c}__mean_hits_per_cell"] = hpc_mat[:, conds.index(c)]
            data[f"{c}__mean_expr"] = mean_mat[:, conds.index(c)]

        df = pd.DataFrame(data, index=genes_present)
        df.index.name = "Gene"
        tables_by_tp[tp] = df

    return genes_present, mats_mean, mats_hits, mats_hpc, ncells_by_tp, tables_by_tp


# =============================================================================
# PLOTTING HEATMAPS
# =============================================================================
def plot_heatmaps(
    genes_present,
    mats_by_tp,
    title_template,
    xlabel,
    ylabel,
    colorbar_label,
    title_fontsize,
    xlabel_fontsize,
    ylabel_fontsize,
    xtick_fontsize,
    ytick_fontsize,
    colorbar_label_fontsize,
    colorbar_tick_fontsize,
    cmap_name,
    figsize,
    share_scale,
    out_prefix,
    annotate=False,
    annotate_fmt="{:.2f}",
    annotate_fontsize=8,
    annotate_max_genes=60,
    timepoints=None,   # NEW
):
    conds = ["Uninfected", "Single infection", "Coinfected"]

    # Decide which timepoints to actually plot
    if timepoints is None:
        timepoints = TIMEPOINT_ORDER

    vmin_global = vmax_global = None
    if share_scale:
        # Only use selected timepoints for global vmin/vmax
        all_vals = []
        for tp in timepoints:
            if tp in mats_by_tp:
                all_vals.append(mats_by_tp[tp].ravel())
        if all_vals:
            flat = np.concatenate(all_vals)
            valid = flat[~np.isnan(flat)]
            if valid.size > 0:
                vmin_global, vmax_global = float(valid.min()), float(valid.max())

    base_cmap = plt.cm.get_cmap(cmap_name or "viridis")
    cmap = copy(base_cmap)
    cmap.set_bad(color="white")

    do_annotate = annotate and (len(genes_present) <= annotate_max_genes)

    for tp in timepoints:
        if tp not in mats_by_tp:
            print(f"No cells at {tp}; skipping {out_prefix} heatmap.")
            continue

        mat = mats_by_tp[tp]

        if share_scale and (vmin_global is not None):
            vmin, vmax = vmin_global, vmax_global
        else:
            flat = mat.ravel()
            valid = flat[~np.isnan(flat)]
            if valid.size == 0:
                print(f"All values NaN at {tp}; skipping {out_prefix} heatmap.")
                continue
            vmin, vmax = float(valid.min()), float(valid.max())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)
        cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(conds, rotation=0, fontsize=xtick_fontsize)

        display_labels = [format_gene_display_name(gene) for gene in genes_present]
        ax.set_yticks(np.arange(len(genes_present)))
        ax.set_yticklabels(display_labels, fontsize=ytick_fontsize)

        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.set_title(title_template.format(timepoint=tp), fontsize=title_fontsize)

        if do_annotate:
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    val = mat[r, c]
                    if np.isnan(val):
                        continue
                    txt_color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(
                        c, r,
                        annotate_fmt.format(val),
                        ha="center", va="center",
                        fontsize=annotate_fontsize,
                        color=txt_color
                    )
        elif annotate and not do_annotate:
            print(f"Annotation disabled for {out_prefix} (genes={len(genes_present)} > {annotate_max_genes}).")

        fig.tight_layout()
        save_png(fig, f"{out_prefix}_heatmap_{tp}_displayed_genes")


# =============================================================================
# TABLE OUTPUT
# =============================================================================
def write_tables_to_csv(tables_by_tp, ncells_by_tp):
    ensure_outdir()
    for tp, df in tables_by_tp.items():
        out1 = os.path.join(GRAPH_OUTPUT_DIR, f"gene_support_table_{tp}.csv")
        df.to_csv(out1)
        print(f"Saved: {out1}")

        ncells = ncells_by_tp[tp]
        df_n = pd.DataFrame([ncells])
        df_n.index = ["n_cells"]
        out2 = os.path.join(GRAPH_OUTPUT_DIR, f"n_cells_{tp}.csv")
        df_n.to_csv(out2)
        print(f"Saved: {out2}")


# =============================================================================
# NEW: COINFECTION SIGNIFICANCE TESTS
# =============================================================================
def run_coinfection_stats(adata: sc.AnnData):
    """
    Per timepoint, per phage gene:
      - Compare Coinfected vs phage-specific Single (Only Luz19 OR Only LKD16)
      - Mann–Whitney U on per-cell expression vector (adata.X by default)
      - Fisher exact on detection (>0 raw counts) (optional)

    Writes CSVs to graph_outputs/.
    """
    ensure_outdir()

    use_norm = bool(STYLE.get("stats_use_normalized_X", True))
    min_n = int(STYLE.get("stats_min_cells_per_group", 10))
    do_det = bool(STYLE.get("stats_run_detection_test", True))

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    all_rows = []

    # For a quick “total phage expression per cell” test too
    phage_total_rows = []

    for tp in TIMEPOINT_ORDER:
        ad_tp = adata[adata.obs["timepoint"] == tp]
        if ad_tp.n_obs == 0:
            continue

        for phage_prefix in PHAGE_PREFIXES:
            phage_name = phage_prefix.strip(":")
            if phage_name == "luz19":
                single_label = "Only Luz19"
            elif phage_name == "lkd16":
                single_label = "Only LKD16"
            else:
                # fallback
                single_label = f"Only {phage_name}"

            # groups
            ad_single = ad_tp[ad_tp.obs["phage_state"] == single_label]
            ad_co = ad_tp[ad_tp.obs["phage_state"] == "Coinfected"]

            n_single = int(ad_single.n_obs)
            n_co = int(ad_co.n_obs)

            # record totals test even if few genes
            # use precomputed per-cell total phage expression in raw counts
            col_expr = f"{phage_name}_expression"
            if col_expr in ad_tp.obs.columns and n_single >= 1 and n_co >= 1:
                x_single = np.asarray(ad_single.obs[col_expr]).ravel()
                x_co = np.asarray(ad_co.obs[col_expr]).ravel()
                # MW requires at least 2 values per group for a stable p-value, but we’ll guard.
                if (n_single >= min_n) and (n_co >= min_n):
                    try:
                        u_tot, p_tot = mannwhitneyu(x_co, x_single, alternative="two-sided")
                    except Exception:
                        u_tot, p_tot = np.nan, np.nan
                else:
                    u_tot, p_tot = np.nan, np.nan

                phage_total_rows.append({
                    "Timepoint": tp,
                    "Phage": phage_name,
                    "Group_single": single_label,
                    "Group_coinfected": "Coinfected",
                    "n_single": n_single,
                    "n_coinfected": n_co,
                    "mean_total_phage_expr_single_raw": float(np.mean(x_single)) if n_single else np.nan,
                    "mean_total_phage_expr_coinfected_raw": float(np.mean(x_co)) if n_co else np.nan,
                    "mw_u": u_tot,
                    "mw_p": p_tot,
                })

            # get phage genes
            phage_genes = [
                g for g in adata.var_names
                if str(g).lower().startswith(phage_prefix.lower())
            ]
            if len(phage_genes) == 0:
                continue

            # run per-gene tests
            p_mw = []
            p_fisher = []
            rows_this = []

            for g in phage_genes:
                gi = gene_to_idx.get(g, None)
                if gi is None:
                    continue

                # Must have enough cells
                if (n_single < min_n) or (n_co < min_n):
                    mw_u, mw_p, rbc = np.nan, np.nan, np.nan
                else:
                    v_single = get_gene_vector(ad_single, gi, use_normalized_X=use_norm)
                    v_co = get_gene_vector(ad_co, gi, use_normalized_X=use_norm)

                    # Mann–Whitney U
                    try:
                        mw_u, mw_p = mannwhitneyu(v_co, v_single, alternative="two-sided")
                        # rank-biserial correlation (effect size): 1 - 2U/(n1*n2), with U for co vs single
                        rbc = 1.0 - (2.0 * mw_u) / (n_co * n_single)
                    except Exception:
                        mw_u, mw_p, rbc = np.nan, np.nan, np.nan

                # detection Fisher test on RAW counts (>0)
                if do_det and (n_single >= min_n) and (n_co >= min_n):
                    v_single_raw = get_gene_vector(ad_single, gi, use_normalized_X=False)
                    v_co_raw = get_gene_vector(ad_co, gi, use_normalized_X=False)

                    det_single = int(np.sum(v_single_raw > 0))
                    det_co = int(np.sum(v_co_raw > 0))
                    nodet_single = n_single - det_single
                    nodet_co = n_co - det_co

                    try:
                        _, fish_p = fisher_exact([[det_co, nodet_co], [det_single, nodet_single]], alternative="two-sided")
                    except Exception:
                        fish_p = np.nan
                else:
                    det_single = det_co = nodet_single = nodet_co = None
                    fish_p = np.nan

                # Means (for interpretability)
                # Use whichever layer we tested for mean comparison, but also report raw mean hits/cell.
                if n_single > 0 and n_co > 0:
                    v_single_test = get_gene_vector(ad_single, gi, use_normalized_X=use_norm)
                    v_co_test = get_gene_vector(ad_co, gi, use_normalized_X=use_norm)

                    v_single_raw = get_gene_vector(ad_single, gi, use_normalized_X=False)
                    v_co_raw = get_gene_vector(ad_co, gi, use_normalized_X=False)

                    mean_single = float(np.mean(v_single_test))
                    mean_co = float(np.mean(v_co_test))
                    mean_single_raw = float(np.mean(v_single_raw))
                    mean_co_raw = float(np.mean(v_co_raw))
                else:
                    mean_single = mean_co = np.nan
                    mean_single_raw = mean_co_raw = np.nan

                row = {
                    "Timepoint": tp,
                    "Phage": phage_name,
                    "Gene": g,
                    "Group_single": single_label,
                    "Group_coinfected": "Coinfected",
                    "n_single": n_single,
                    "n_coinfected": n_co,
                    "mean_expr_single_testlayer": mean_single,
                    "mean_expr_coinfected_testlayer": mean_co,
                    "delta_mean_testlayer": (mean_co - mean_single) if np.isfinite(mean_single) and np.isfinite(mean_co) else np.nan,
                    "mean_raw_counts_per_cell_single": mean_single_raw,
                    "mean_raw_counts_per_cell_coinfected": mean_co_raw,
                    "mw_u": mw_u,
                    "mw_p": mw_p,
                    "mw_rank_biserial": rbc,
                    "detected_cells_single_raw": det_single,
                    "detected_cells_coinfected_raw": det_co,
                    "fisher_p_detection": fish_p,
                }

                rows_this.append(row)
                p_mw.append(mw_p)
                p_fisher.append(fish_p)

            # FDR corrections (within this tp+phage block)
            mw_adj = bh_fdr(np.array(p_mw))
            if do_det:
                fish_adj = bh_fdr(np.array(p_fisher))
            else:
                fish_adj = np.array([np.nan] * len(rows_this), dtype=float)

            for r, q1, q2 in zip(rows_this, mw_adj, fish_adj):
                r["mw_fdr_bh"] = q1
                r["fisher_fdr_bh_detection"] = q2
                all_rows.append(r)

            # write per-tp per-phage CSV
            df_block = pd.DataFrame(rows_this)
            if len(df_block) > 0:
                df_block["mw_fdr_bh"] = mw_adj
                if do_det:
                    df_block["fisher_fdr_bh_detection"] = fish_adj
                out = os.path.join(GRAPH_OUTPUT_DIR, f"coinfection_gene_tests_{tp}_{phage_name}.csv")
                df_block.to_csv(out, index=False)
                print(f"Saved: {out}")

    # write combined
    df_all = pd.DataFrame(all_rows)
    out_all = os.path.join(GRAPH_OUTPUT_DIR, "coinfection_gene_tests_ALL.csv")
    df_all.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    # write totals
    df_tot = pd.DataFrame(phage_total_rows)
    # FDR across all totals tests (optional)
    if len(df_tot) > 0 and "mw_p" in df_tot.columns:
        df_tot["mw_fdr_bh"] = bh_fdr(df_tot["mw_p"].to_numpy())
    out_tot = os.path.join(GRAPH_OUTPUT_DIR, "coinfection_phage_totals_tests.csv")
    df_tot.to_csv(out_tot, index=False)
    print(f"Saved: {out_tot}")

    # quick console summary
    if len(df_all) > 0:
        sig = df_all[np.isfinite(df_all["mw_fdr_bh"]) & (df_all["mw_fdr_bh"] < 0.05)]
        print(f"[Stats] Significant genes at FDR<0.05 (MW): {len(sig)} / {len(df_all)}")
    else:
        print("[Stats] No gene-level rows written (possibly no phage genes or too few cells per group).")


# =============================================================================
# TOTAL RAW COUNTS QC PLOTS
# =============================================================================
def plot_total_counts_by_group(adata: sc.AnnData):
    """
    Make per-timepoint boxplots of total raw counts per cell for:
      - Only Luz19
      - Only LKD16
      - Coinfected

    Uses:
      - adata.layers["counts"] if present (raw counts)
      - adata.obs["phage_presence"] to distinguish groups
      - adata.obs["timepoint"] to split by time

    This is meant to show capture / library-size differences
    between infection states at each timepoint.
    """
    # -------------------------------------------------------------------------
    # 1) Ensure we have total_counts_raw
    # -------------------------------------------------------------------------
    if "total_counts_raw" not in adata.obs.columns:
        if "counts" in adata.layers:
            X_counts = adata.layers["counts"]
        else:
            X_counts = adata.X  # fall back if counts layer missing

        if sparse.issparse(X_counts):
            total_counts = np.asarray(X_counts.sum(axis=1)).ravel()
        else:
            total_counts = np.asarray(X_counts.sum(axis=1)).ravel()

        adata.obs["total_counts_raw"] = total_counts

    # -------------------------------------------------------------------------
    # 2) Build detailed infection labels:
    #    0 = No phage
    #    1 = Only Luz19
    #    2 = Only LKD16
    #    3 = Coinfected (both)
    # -------------------------------------------------------------------------
    if "phage_presence" not in adata.obs.columns:
        raise ValueError(
            "phage_presence not found in adata.obs. "
            "Make sure setup_adata() has been run before calling plot_total_counts_by_group()."
        )

    def detailed_state(code: int) -> str:
        if code == 0:
            return "No phage"
        elif code == 1:
            return "Only Luz19"
        elif code == 2:
            return "Only LKD16"
        else:
            return "Coinfected"

    if "infection_state_detail" not in adata.obs.columns:
        adata.obs["infection_state_detail"] = [
            detailed_state(int(x)) for x in adata.obs["phage_presence"]
        ]
        adata.obs["infection_state_detail"] = pd.Categorical(
            adata.obs["infection_state_detail"],
            categories=["No phage", "Only Luz19", "Only LKD16", "Coinfected"],
            ordered=True,
        )

    # We’ll only plot these three conditions
    conds_to_plot = ["Only Luz19", "Only LKD16", "Coinfected"]

    # -------------------------------------------------------------------------
    # 3) Make one figure per timepoint
    # -------------------------------------------------------------------------
    for tp in TIMEPOINT_ORDER:
        ad_tp = adata[adata.obs["timepoint"] == tp]
        if ad_tp.n_obs == 0:
            print(f"[total_counts QC] No cells at {tp}, skipping.")
            continue

        # Collect total_counts_raw per condition
        data_per_cond = []
        labels = []
        for cond in conds_to_plot:
            mask = (ad_tp.obs["infection_state_detail"] == cond)
            vals = ad_tp.obs.loc[mask, "total_counts_raw"].values
            # Only include conditions that actually have cells
            if vals.size > 0:
                data_per_cond.append(vals)
                labels.append(cond)

        if not data_per_cond:
            print(f"[total_counts QC] No cells in any of the requested conditions at {tp}, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(
            data_per_cond,
            labels=labels,
            showfliers=True,
            patch_artist=True,
        )

        # Simple coloring
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for patch, color in zip(bp["boxes"], colors[: len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(
            STYLE["total_counts_title"].format(timepoint=tp),
            fontsize=STYLE["total_counts_title_fontsize"],
        )
        ax.set_xlabel(
            STYLE["total_counts_xlabel"],
            fontsize=STYLE["total_counts_xlabel_fontsize"],
        )
        ax.set_ylabel(
            STYLE["total_counts_ylabel"],
            fontsize=STYLE["total_counts_ylabel_fontsize"],
        )
        ax.tick_params(
            axis="x",
            labelsize=STYLE["total_counts_xtick_fontsize"],
        )
        ax.tick_params(
            axis="y",
            labelsize=STYLE["total_counts_ytick_fontsize"],
        )
        #ax.set_yscale("log")  # often helpful for count data; remove if you prefer linear

        fig.tight_layout()
        save_png(fig, f"total_counts_raw_boxplot_{tp}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_outdir()
    adata = setup_adata()

    # Select genes from an optional ordered text file, or default to all phage genes.
    genes, missing_genes = select_heatmap_genes(adata, GENE_DISPLAY_FILE)
    write_displayed_gene_lists(genes, missing_genes)
    print(f"Heatmaps will use {len(genes)} genes in the listed order for every timepoint.")

    # Compute per-timepoint matrices + tables
    genes_present, mats_mean, mats_hits, mats_hpc, ncells_by_tp, tables_by_tp = compute_tp_matrices_and_tables(
        adata, genes
    )

    # Mean-expression heatmaps
    plot_heatmaps(
        genes_present=genes_present,
        mats_by_tp=mats_mean,
        title_template=STYLE["mean_title"],
        xlabel=STYLE["mean_xlabel"],
        ylabel=STYLE["mean_ylabel"],
        colorbar_label=STYLE["mean_colorbar_label"],
        title_fontsize=STYLE["mean_title_fontsize"],
        xlabel_fontsize=STYLE["mean_xlabel_fontsize"],
        ylabel_fontsize=STYLE["mean_ylabel_fontsize"],
        xtick_fontsize=STYLE["mean_xtick_fontsize"],
        ytick_fontsize=STYLE["mean_ytick_fontsize"],
        colorbar_label_fontsize=STYLE["mean_colorbar_label_fontsize"],
        colorbar_tick_fontsize=STYLE["mean_colorbar_tick_fontsize"],
        cmap_name=STYLE["mean_heatmap_cmap"],
        figsize=heatmap_figsize(len(genes_present)),
        share_scale=STYLE["mean_heatmap_share_color_scale"],
        out_prefix="mean_expr",
        annotate=False,
        timepoints=HEATMAP_TIMEPOINTS,
    )

    # Hits-per-cell heatmaps (raw)
    if STYLE.get("make_hits_per_cell_heatmaps", True):
        plot_heatmaps(
            genes_present=genes_present,
            mats_by_tp=mats_hpc,
            title_template=STYLE["hits_title"],
            xlabel=STYLE["hits_xlabel"],
            ylabel=STYLE["hits_ylabel"],
            colorbar_label=STYLE["hits_colorbar_label"],
            title_fontsize=STYLE["hits_title_fontsize"],
            xlabel_fontsize=STYLE["hits_xlabel_fontsize"],
            ylabel_fontsize=STYLE["hits_ylabel_fontsize"],
            xtick_fontsize=STYLE["hits_xtick_fontsize"],
            ytick_fontsize=STYLE["hits_ytick_fontsize"],
            colorbar_label_fontsize=STYLE["hits_colorbar_label_fontsize"],
            colorbar_tick_fontsize=STYLE["hits_colorbar_tick_fontsize"],
            cmap_name=STYLE["hits_per_cell_cmap"],
            figsize=heatmap_figsize(len(genes_present)),
            share_scale=STYLE["hits_per_cell_share_color_scale"],
            out_prefix="mean_hits_per_cell_raw",
            annotate=STYLE.get("annotate_hits_per_cell", True),
            annotate_fmt=STYLE.get("annotate_fmt", "{:.2f}"),
            annotate_fontsize=STYLE.get("annotate_fontsize", 8),
            annotate_max_genes=STYLE.get("annotate_max_genes", 60),
            timepoints=HEATMAP_TIMEPOINTS,
        )

    # Tables
    if STYLE.get("write_tables", True):
        write_tables_to_csv(tables_by_tp, ncells_by_tp)

    # NEW: coinfection vs single significance tests
    if STYLE.get("run_coinfection_stats", True):
        run_coinfection_stats(adata)

    print(f"Done. Outputs saved to: {GRAPH_OUTPUT_DIR}/")

    # QC: total raw counts per cell by infection state & timepoint
    plot_total_counts_by_group(adata)


if __name__ == "__main__":
    main()
