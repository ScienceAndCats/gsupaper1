"""
Compare gene expression across infection states by timepoint using multiple plot types.

Outputs (into graph_outputs/):
  - Bar plots (top N genes) per timepoint
  - Dot plot (mean expression color + % expressing size) across timepoint×condition
  - Heatmap (mean expression) across timepoint×condition
  - Violin plots (per gene; panels by timepoint; split by infection state)
  - Time-course line plots (per gene; time on x; one line per infection state)

Data expected:
  - Tab-delimited gene matrix text file
  - Rows = cells, columns = genes, first column = cell IDs
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
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

# Normalization
DO_NORMALIZE = True          # normalize_total + log1p
TARGET_SUM = 1e4             # for normalize_total

# Timepoints
TIMEPOINT_ORDER = ["5min", "10min", "15min", "20min"]

# Phage gene prefixes
PHAGE_PREFIXES = ["luz19:", "lkd16:"]

# Top genes selection
TOP_N_GENES = 15             # used for bar/dot/heatmap by default

# Violin / time-course gene list behavior:
# - If None: will use the global TOP_N_GENES
# - Or supply a list like ["rpoD", "groEL", "luz19:geneX", ...]
GENES_OF_INTEREST = None

# Expression "on" threshold for percent-expressing in dot plot
EXPRESSING_THRESHOLD = 0.0   # >0 is typical

# =============================================================================
# STYLE / FORMATTING BLOCK
# =============================================================================
STYLE = {
    # Global matplotlib
    "dpi": 160,
    "font_size": 11,

    # Condition colors
    "condition_colors": {
        "Uninfected": "tab:gray",
        "Single infection": "tab:blue",
        "Coinfected": "tab:red",
    },

    # Bar plots
    "bar_figsize": (11, 6),
    "bar_width": 0.25,
    "bar_edgecolor": "black",
    "bar_xtick_rotation": 45,
    "bar_grid": True,

    # Dot plot
    "dot_figsize": (12, 7),
    "dot_size_min": 20,       # marker area
    "dot_size_max": 350,
    "dot_cmap": None,         # leave None to use matplotlib default

    # Heatmap
    "heatmap_figsize": (12, 7),
    "heatmap_cmap": None,     # leave None to use matplotlib default

    # Violin plots
    "violin_figsize_per_timepoint": (11, 3.2),  # per timepoint row
    "violin_alpha": 0.85,

    # Time-course line plots (one per gene)
    "line_figsize": (7.5, 4.5),
    "line_marker": "o",
    "line_linewidth": 2.0,
    "line_grid": True,
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
    path = os.path.join(GRAPH_OUTPUT_DIR, sanitize_filename(name) + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

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

def to_dense_if_needed(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

def mean_by_group(adata_sub: sc.AnnData) -> np.ndarray:
    """Mean expression per gene for given AnnData subset; returns 1D array length n_vars."""
    X = adata_sub.X
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    return np.asarray(X.mean(axis=0)).ravel()

def frac_expressing_by_group(adata_sub: sc.AnnData, threshold: float) -> np.ndarray:
    """Fraction of cells expressing each gene (>threshold); returns 1D array length n_vars."""
    X = adata_sub.X
    if sparse.issparse(X):
        # sparse comparison creates sparse matrix; sum over rows
        expressed = (X > threshold)
        return np.asarray(expressed.mean(axis=0)).ravel()
    return np.mean(np.asarray(X) > threshold, axis=0)

# =============================================================================
# LOAD + PREP
# =============================================================================
def load_gene_matrix_to_adata(path: str) -> sc.AnnData:
    if path.endswith(".h5ad"):
        return sc.read_h5ad(path)

    raw = pd.read_csv(path, sep="\t", index_col=0)
    return sc.AnnData(raw)

def setup_adata() -> sc.AnnData:
    adata = load_gene_matrix_to_adata(FILE_PATH)
    print(adata)

    # Filter
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)
    sc.pp.filter_genes(adata, min_counts=MIN_COUNTS_GENES)

    # Optional normalize/log1p
    if DO_NORMALIZE:
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

    # Timepoint
    adata.obs["timepoint"] = [classify_cell(n) for n in adata.obs_names]
    adata.obs["timepoint"] = pd.Categorical(
        adata.obs["timepoint"], categories=TIMEPOINT_ORDER, ordered=True
    )

    # Phage expression + presence encoding
    for phage in PHAGE_PREFIXES:
        mask = adata.var_names.str.contains(re.escape(phage))
        if mask.sum() == 0:
            print(f"WARNING: No genes matched prefix '{phage}'")
        if sparse.issparse(adata.X):
            expr = adata[:, mask].X.sum(axis=1).A.flatten()
        else:
            expr = np.asarray(adata[:, mask].X.sum(axis=1)).ravel()
        adata.obs[f"{phage.strip(':')}_expression"] = expr

    # 0=no phage, 1=only luz19, 2=only lkd16, 3=both
    adata.obs["phage_presence"] = (
        (adata.obs["luz19_expression"] > 0).astype(int) * 1 +
        (adata.obs["lkd16_expression"] > 0).astype(int) * 2
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

    # Combined highlighting group for dot/heatmap columns
    adata.obs["tp_state"] = adata.obs["timepoint"].astype(str) + "_" + adata.obs["infection_state"].astype(str)

    return adata

# =============================================================================
# GENE SELECTION
# =============================================================================
def pick_top_genes_global(adata: sc.AnnData, n: int) -> list[str]:
    m = mean_by_group(adata)
    n = min(n, adata.n_vars)
    idx = np.argsort(m)[::-1][:n]
    return list(adata.var_names[idx])

# =============================================================================
# PLOT 1: BAR PLOTS (per timepoint)
# =============================================================================
def plot_bars_per_timepoint(adata: sc.AnnData, genes: list[str]):
    for tp in TIMEPOINT_ORDER:
        ad_tp = adata[adata.obs["timepoint"] == tp]
        if ad_tp.n_obs == 0:
            continue

        # Choose top genes at this timepoint by overall mean
        m_tp = mean_by_group(ad_tp)
        n = min(TOP_N_GENES, ad_tp.n_vars)
        top_idx = np.argsort(m_tp)[::-1][:n]
        top_genes = list(ad_tp.var_names[top_idx])

        # Means per condition
        conds = ["Uninfected", "Single infection", "Coinfected"]
        means = {}
        for c in conds:
            sub = ad_tp[ad_tp.obs["infection_state"] == c]
            if sub.n_obs == 0:
                means[c] = np.zeros(len(top_genes), dtype=float)
            else:
                m = mean_by_group(sub)
                means[c] = m[top_idx]

        x = np.arange(len(top_genes))
        w = STYLE["bar_width"]

        fig = plt.figure(figsize=STYLE["bar_figsize"])
        ax = fig.add_subplot(111)

        offsets = {"Uninfected": -w, "Single infection": 0.0, "Coinfected": w}
        for c in conds:
            ax.bar(
                x + offsets[c],
                means[c],
                width=w,
                label=c,
                color=STYLE["condition_colors"][c],
                edgecolor=STYLE["bar_edgecolor"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(top_genes, rotation=STYLE["bar_xtick_rotation"], ha="right")
        ax.set_title(f"Top {len(top_genes)} genes at {tp} by infection state")
        ax.set_xlabel("Gene")
        ax.set_ylabel("Mean expression" + (" (log1p normalized)" if DO_NORMALIZE else ""))
        ax.legend(loc="best")

        if STYLE["bar_grid"]:
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        save_png(fig, f"bar_top{len(top_genes)}_{tp}")

# =============================================================================
# PLOT 2: DOT PLOT (mean color + % expressing size)
# =============================================================================
def plot_dotplot(adata: sc.AnnData, genes: list[str]):
    # Build table: rows = genes, cols = tp_state
    groups = [f"{tp}_{st}" for tp in TIMEPOINT_ORDER for st in ["Uninfected", "Single infection", "Coinfected"]]

    mean_mat = np.zeros((len(genes), len(groups)), dtype=float)
    frac_mat = np.zeros((len(genes), len(groups)), dtype=float)

    # Map gene indices once
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    genes = [g for g in genes if g in gene_to_idx]

    for j, grp in enumerate(groups):
        sub = adata[adata.obs["tp_state"] == grp]
        if sub.n_obs == 0:
            continue
        m = mean_by_group(sub)[gene_idxs]
        f = frac_expressing_by_group(sub, EXPRESSING_THRESHOLD)[gene_idxs]
        mean_mat[:, j] = m
        frac_mat[:, j] = f

    # Scale marker sizes by frac_mat (0..1) -> [min..max]
    smin, smax = STYLE["dot_size_min"], STYLE["dot_size_max"]
    sizes = smin + (smax - smin) * frac_mat

    fig = plt.figure(figsize=STYLE["dot_figsize"])
    ax = fig.add_subplot(111)

    # Coordinates for scatter
    xs = np.tile(np.arange(len(groups)), len(genes))
    ys = np.repeat(np.arange(len(genes)), len(groups))
    cs = mean_mat.reshape(-1)
    ss = sizes.reshape(-1)

    sca = ax.scatter(xs, ys, s=ss, c=cs, cmap=STYLE["dot_cmap"])
    cbar = fig.colorbar(sca, ax=ax)
    cbar.set_label("Mean expression" + (" (log1p normalized)" if DO_NORMALIZE else ""))

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)

    ax.set_title("Dot plot: color=mean expression; size=% cells expressing")
    ax.set_xlabel("Timepoint_InfectionState")
    ax.set_ylabel("Gene")

    fig.tight_layout()
    save_png(fig, f"dotplot_top{len(genes)}")

# =============================================================================
# PLOT 3: HEATMAP (mean expression)
# =============================================================================
def plot_heatmap(adata: sc.AnnData, genes: list[str]):
    groups = [f"{tp}_{st}" for tp in TIMEPOINT_ORDER for st in ["Uninfected", "Single infection", "Coinfected"]]

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    genes = [g for g in genes if g in gene_to_idx]

    mean_mat = np.zeros((len(genes), len(groups)), dtype=float)
    for j, grp in enumerate(groups):
        sub = adata[adata.obs["tp_state"] == grp]
        if sub.n_obs == 0:
            continue
        mean_mat[:, j] = mean_by_group(sub)[gene_idxs]

    fig = plt.figure(figsize=STYLE["heatmap_figsize"])
    ax = fig.add_subplot(111)

    im = ax.imshow(mean_mat, aspect="auto", cmap=STYLE["heatmap_cmap"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean expression" + (" (log1p normalized)" if DO_NORMALIZE else ""))

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)

    ax.set_title("Heatmap: mean expression across timepoint×infection state")
    ax.set_xlabel("Timepoint_InfectionState")
    ax.set_ylabel("Gene")

    fig.tight_layout()
    save_png(fig, f"heatmap_top{len(genes)}")

# =============================================================================
# PLOT 4: VIOLIN PLOTS (per gene; timepoints as panels)
# =============================================================================
def plot_violins(adata: sc.AnnData, genes: list[str]):
    # For each gene, create a multi-panel figure (one row per timepoint)
    conds = ["Uninfected", "Single infection", "Coinfected"]
    colors = [STYLE["condition_colors"][c] for c in conds]

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    for g in genes:
        if g not in gene_to_idx:
            continue
        gi = gene_to_idx[g]

        fig = plt.figure(figsize=(STYLE["violin_figsize_per_timepoint"][0],
                                  STYLE["violin_figsize_per_timepoint"][1] * len(TIMEPOINT_ORDER)))

        for i, tp in enumerate(TIMEPOINT_ORDER, start=1):
            ax = fig.add_subplot(len(TIMEPOINT_ORDER), 1, i)
            ad_tp = adata[adata.obs["timepoint"] == tp]
            if ad_tp.n_obs == 0:
                ax.set_axis_off()
                continue

            data = []
            ns = []
            for c in conds:
                sub = ad_tp[ad_tp.obs["infection_state"] == c]
                if sub.n_obs == 0:
                    arr = np.array([], dtype=float)
                else:
                    x = sub.X[:, gi]
                    arr = np.asarray(x.todense()).ravel() if sparse.issparse(x) else np.asarray(x).ravel()
                data.append(arr)
                ns.append(len(arr))

            # Matplotlib violinplot needs >=2 points per group. We handle small-n groups gracefully.
            positions = np.arange(1, len(conds) + 1)

            # Draw violins only for groups with >=2 points
            valid_positions = [positions[k] for k in range(len(conds)) if ns[k] >= 2]
            valid_data = [data[k] for k in range(len(conds)) if ns[k] >= 2]

            if len(valid_data) > 0:
                parts = ax.violinplot(
                    valid_data,
                    positions=valid_positions,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False
                )
                # Color each violin body according to its condition
                valid_colors = [colors[k] for k in range(len(conds)) if ns[k] >= 2]
                for pc, col in zip(parts["bodies"], valid_colors):
                    pc.set_facecolor(col)
                    pc.set_edgecolor("black")
                    pc.set_alpha(STYLE["violin_alpha"])

            # For groups with 1 point, draw a marker at that value
            for k in range(len(conds)):
                if ns[k] == 1:
                    ax.scatter(
                        positions[k],
                        data[k][0],
                        s=35,
                        color=colors[k],
                        edgecolor="black",
                        zorder=3
                    )

            # For groups with 0 points, do nothing (optionally annotate)
            # Example: annotate missing groups lightly
            for k in range(len(conds)):
                if ns[k] == 0:
                    ax.text(
                        positions[k],
                        0.95,
                        "n=0",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=9,
                        alpha=0.6
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels([f"{c}\n(n={ns[idx]})" for idx, c in enumerate(conds)])
            ax.set_ylabel("Expr")
            ax.set_title(f"{g} — {tp}")
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(f"Violin distributions by infection state across time: {g}", y=1.01)
        fig.tight_layout()
        save_png(fig, f"violin_{g}")


# =============================================================================
# PLOT 5: TIME-COURSE LINE PLOTS (one plot per gene)
# =============================================================================
def plot_timecourse_lines(adata: sc.AnnData, genes: list[str]):
    conds = ["Uninfected", "Single infection", "Coinfected"]

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    for g in genes:
        if g not in gene_to_idx:
            continue
        gi = gene_to_idx[g]

        # Build mean expression per timepoint per condition
        y_by_cond = {c: [] for c in conds}
        for tp in TIMEPOINT_ORDER:
            ad_tp = adata[adata.obs["timepoint"] == tp]
            for c in conds:
                sub = ad_tp[ad_tp.obs["infection_state"] == c]
                if sub.n_obs == 0:
                    y_by_cond[c].append(np.nan)
                else:
                    x = sub.X[:, gi]
                    x = np.asarray(x.mean()).item() if sparse.issparse(x) else float(np.mean(x))
                    y_by_cond[c].append(x)

        fig = plt.figure(figsize=STYLE["line_figsize"])
        ax = fig.add_subplot(111)

        for c in conds:
            ax.plot(
                TIMEPOINT_ORDER,
                y_by_cond[c],
                marker=STYLE["line_marker"],
                linewidth=STYLE["line_linewidth"],
                color=STYLE["condition_colors"][c],
                label=c,
            )

        ax.set_title(f"Time-course mean expression: {g}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Mean expression" + (" (log1p normalized)" if DO_NORMALIZE else ""))
        ax.legend(loc="best")

        if STYLE["line_grid"]:
            ax.grid(alpha=0.3)

        fig.tight_layout()
        save_png(fig, f"timecourse_{g}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_outdir()
    adata = setup_adata()

    # Decide which genes to plot
    if GENES_OF_INTEREST is None:
        genes = pick_top_genes_global(adata, TOP_N_GENES)
    else:
        # Keep only those present
        genes = [g for g in GENES_OF_INTEREST if g in adata.var_names]
        if len(genes) == 0:
            raise ValueError("None of GENES_OF_INTEREST were found in adata.var_names.")
    print(f"Using {len(genes)} genes for dot/heatmap/violin/timecourse plots.")

    # Bar plots use top genes per timepoint internally
    plot_bars_per_timepoint(adata, genes)

    # Dot plot + heatmap across all timepoint×condition combos
    plot_dotplot(adata, genes)
    plot_heatmap(adata, genes)

    # Distribution + dynamics plots
    plot_violins(adata, genes)
    plot_timecourse_lines(adata, genes)

    print(f"Done. PNGs saved in: {GRAPH_OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
