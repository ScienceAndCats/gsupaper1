#!/usr/bin/env python3
"""
QC summary + plots from:
  (1) gene matrix file (used mainly to define the set of expected cell barcodes)
  (2) filtered mapped UMIs file (Cell Barcode, UMI, contig:gene, total_reads)

Generates:
  1) Reads per cell (from sum(total_reads) in filtered mapped UMIs)
  2) Mapping ambiguity stats: fraction of mapped reads that are unique vs multi-hit
  3) UMIs per cell (violin plots; choice of counting mode)
  4) Library complexity:
       - duplicate rate per cell (1 - UMIs/reads)
       - reads-per-UMI distribution
       - saturation/complexity curve (expected UMIs discovered vs reads sampled)

Multi-dataset (up to 6):
  - All plots include all datasets in the same figure for easy comparison.

Optional stratification (OFF by default):
  - timepoints (bc1 bins)
  - infection states (Uninfected / Only luz19 / Only lkd16 / Coinfected),
    inferred from presence of mapped reads containing 'luz19:' / 'lkd16:' in contig:gene.

Dependencies (conda-forge):
  conda install -c conda-forge numpy pandas matplotlib scipy
"""

import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTPUT_DIR = "qc_outputs"

# Up to 6 datasets
# Provide a gene matrix file AND a filtered mapped UMIs file per dataset.
# Example UMI file format is the one you attached:
#   Cell Barcode, UMI, contig:gene, total_reads (tab-separated)
#   e.g. "... bc1_10 ...   ATACAAT   PA01:PA0525   102"
DATASETS = [
    {
        "name": "JRG07-Sample-P",
        "gene_matrix_path": "processed_data/JRG07-Sample-P/JRG07-Sample-P_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt",
        "filtered_umis_path": "processed_data/JRG07-Sample-P/JRG07-Sample-P_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt",
    },
    {
        "name": "JRG09-UI",
        "gene_matrix_path": "processed_data/JRG09-UI/JRG09-UI_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt",
        "filtered_umis_path": "processed_data/JRG09-UI/JRG09-UI_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt",
    },
    {
        "name": "luz19timeseries",
        "gene_matrix_path": "processed_data/luz19timeseries/luz19timeseries_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt",
        "filtered_umis_path": "processed_data/luz19timeseries/luz19timeseries_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt",
    },
    {
        "name": "JRG07-Sample-P3",
        "gene_matrix_path": "processed_data/JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt",
        "filtered_umis_path": "processed_data/JRG07-Sample-P3/JRG07-Sample-P3_v11_threshold_0_filtered_mapped_UMIs_multihitcombo.txt",
    },
    # Add up to 5 more:
    # {"name": "dataset2", "gene_matrix_path": "...", "filtered_umis_path": "..."},
]

# Stratification toggles (DEFAULT OFF)
USE_TIMEPOINTS = False
USE_INFECTION_STATES = False

# Timepoint assignment (same convention as before)
TIMEPOINT_ORDER = ["5min", "10min", "15min", "20min"]

# Infection calling
PHAGE_PREFIXES = ["luz19:", "lkd16:"]
PHAGE_PRESENT_THRESHOLD_READS = 0  # >0 => present

# UMI counting mode for "UMIs per cell"
#   "records": count rows in filtered_umis per cell (after grouping identical (cell, UMI) pairs)
#   "unique":  count unique UMI sequences per cell
UMIS_PER_CELL_MODE = "records"  # or "unique"

# Complexity curve settings
MAKE_SATURATION_CURVE = True
MAX_CELLS_FOR_SATCURVE = 250   # sample cells per dataset for speed
SATCURVE_POINTS = 25           # number of read-depth points on curve
SATCURVE_USE_GLOBAL_MAX = True # same x-axis max across datasets for comparison

# =============================================================================
# FIGURE STYLE OPTIONS (edit freely)
# =============================================================================
STYLE = {
    "dpi": 180,
    "font_size": 11,

    # General
    "tight_layout": True,
    "save_png_dpi": 220,

    # Read/UMI/dup plots
    "figsize_violin": (11, 5),
    "figsize_box": (11, 5),
    "figsize_hist": (12, 5),
    "figsize_scatter": (6.5, 5),
    "figsize_satcurve": (7, 5),

    # Axes scaling
    "reads_per_cell_logy": False,
    "umis_per_cell_logy": False,
    "dup_rate_logy": False,          # usually False
    "reads_per_umi_hist_logx": False,
    "reads_per_umi_hist_logy": False,
    "scatter_logx": False,
    "scatter_logy": False,

    # Histogram bins
    "hist_bins_reads_per_umi": 60,

    # Labels
    "ylabel_reads": "Mapped reads per cell (sum total_reads)",
    "ylabel_umis": "UMIs per cell",
    "ylabel_dup": "Duplicate rate per cell (1 - UMIs/reads)",
    "ylabel_multi": "Fraction multi-hit reads (mapped)",
}

plt.rcParams.update({
    "figure.dpi": STYLE["dpi"],
    "font.size": STYLE["font_size"],
})


# =============================================================================
# HELPERS
# =============================================================================
def ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:180]


def save_png(fig, name: str):
    ensure_outdir()
    out_path = os.path.join(OUTPUT_DIR, sanitize_filename(name) + ".png")
    fig.savefig(out_path, bbox_inches="tight", dpi=STYLE["save_png_dpi"])
    plt.close(fig)
    print(f"Saved: {out_path}")


def classify_timepoint(cell_name: str) -> str:
    """
    Uses your bc1 convention:
      bc1 < 25  => 5min
      bc1 < 49  => 10min
      bc1 < 73  => 15min
      else      => 20min

    Works with names like:
      JRG07-Sample-P3_bc1_10_bc2_10_bc3_82
      (cell_name.split('_')[2] == '10')
    """
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "5min"
    elif bc1_value < 49:
        return "10min"
    elif bc1_value < 73:
        return "15min"
    else:
        return "20min"


def is_multihit(contig_gene: str) -> bool:
    """
    Heuristic:
      - if starts with 'ambiguous:' => multi-hit
      - if contains ','            => multi-hit
    This matches patterns visible in your file.
    """
    s = str(contig_gene).strip()
    if s.lower().startswith("ambiguous:"):
        return True
    if "," in s:
        return True
    return False


def contains_phage(contig_gene: str, phage_prefix: str) -> bool:
    return phage_prefix in str(contig_gene)


def load_cell_barcodes_from_gene_matrix(path: str) -> List[str]:
    """
    We only need the cell barcodes (first column).
    For .h5ad, uses obs_names (if scanpy is available).
    """
    if path.endswith(".h5ad"):
        try:
            import scanpy as sc  # optional
        except ImportError as e:
            raise RuntimeError("scanpy not installed, but gene_matrix_path ends with .h5ad") from e
        adata = sc.read_h5ad(path)
        return list(adata.obs_names)

    # Text matrix: first column is cell barcode
    # Using usecols=[0] keeps it fast even for huge matrices.
    df = pd.read_csv(path, sep="\t", usecols=[0])
    return df.iloc[:, 0].astype(str).tolist()


def load_filtered_umis(path: str) -> pd.DataFrame:
    """
    Expected columns (tab-separated):
      Cell Barcode, UMI, contig:gene, total_reads
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"Cell Barcode", "UMI", "contig:gene", "total_reads"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["total_reads"] = pd.to_numeric(df["total_reads"], errors="coerce").fillna(0).astype(int)
    df["Cell Barcode"] = df["Cell Barcode"].astype(str)
    df["UMI"] = df["UMI"].astype(str)
    df["contig:gene"] = df["contig:gene"].astype(str)
    return df


def collapse_to_unique_umi_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated rows with same (Cell Barcode, UMI) by summing total_reads.
    This is important for:
      - reads-per-UMI distribution
      - duplicate rate estimates
      - saturation curves
    """
    g = df.groupby(["Cell Barcode", "UMI"], as_index=False)["total_reads"].sum()
    return g


def compute_per_cell_metrics(
    df_umis: pd.DataFrame,
    expected_cells: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per_cell_df: one row per cell with QC metrics
      per_umi_df : one row per (cell, UMI) with summed total_reads (for complexity metrics)
    """
    # Collapse to per-(cell, UMI)
    per_umi = collapse_to_unique_umi_per_cell(df_umis)

    # Reads per cell
    reads_per_cell = per_umi.groupby("Cell Barcode")["total_reads"].sum().rename("reads_per_cell")

    # UMIs per cell
    if UMIS_PER_CELL_MODE == "unique":
        umis_per_cell = per_umi.groupby("Cell Barcode")["UMI"].nunique().rename("umis_per_cell")
    elif UMIS_PER_CELL_MODE == "records":
        # one record per (cell, UMI) after collapsing => effectively "UMIs"
        umis_per_cell = per_umi.groupby("Cell Barcode").size().rename("umis_per_cell")
    else:
        raise ValueError("UMIS_PER_CELL_MODE must be 'records' or 'unique'")

    # Duplicate rate (per cell)
    # Duplicate rate = 1 - (UMIs / reads)
    # If reads are very low, guard division.
    df_cell = pd.concat([reads_per_cell, umis_per_cell], axis=1)
    df_cell["dup_rate"] = 1.0 - (df_cell["umis_per_cell"] / df_cell["reads_per_cell"].replace(0, np.nan))
    df_cell["dup_rate"] = df_cell["dup_rate"].fillna(0.0).clip(lower=0.0, upper=1.0)

    # Reads per UMI (per cell)
    df_cell["reads_per_umi"] = df_cell["reads_per_cell"] / df_cell["umis_per_cell"].replace(0, np.nan)
    df_cell["reads_per_umi"] = df_cell["reads_per_umi"].fillna(0.0)

    # Multi-hit fraction (use original df_umis, not collapsed by UMI)
    tmp = df_umis.copy()
    tmp["is_multihit"] = tmp["contig:gene"].apply(is_multihit)
    multi_reads = tmp[tmp["is_multihit"]].groupby("Cell Barcode")["total_reads"].sum().rename("multi_hit_reads")
    all_reads = tmp.groupby("Cell Barcode")["total_reads"].sum().rename("mapped_reads_all_hits")
    df_m = pd.concat([multi_reads, all_reads], axis=1).fillna(0)
    df_m["multi_hit_frac"] = df_m["multi_hit_reads"] / df_m["mapped_reads_all_hits"].replace(0, np.nan)
    df_m["multi_hit_frac"] = df_m["multi_hit_frac"].fillna(0.0).clip(0.0, 1.0)

    df_cell = df_cell.join(df_m[["multi_hit_reads", "mapped_reads_all_hits", "multi_hit_frac"]], how="left").fillna(0)

    # Ensure expected cells are represented (cells with zero UMIs/reads become zeros)
    if expected_cells is not None:
        df_cell = df_cell.reindex(expected_cells).fillna(0)
        df_cell.index.name = "Cell Barcode"
    else:
        df_cell.index.name = "Cell Barcode"

    df_cell = df_cell.reset_index()
    return df_cell, per_umi


def add_optional_groups(
    per_cell_df: pd.DataFrame,
    df_umis: pd.DataFrame,
) -> pd.DataFrame:
    out = per_cell_df.copy()

    if USE_TIMEPOINTS:
        out["timepoint"] = out["Cell Barcode"].apply(classify_timepoint)
        out["timepoint"] = pd.Categorical(out["timepoint"], categories=TIMEPOINT_ORDER, ordered=True)

    if USE_INFECTION_STATES:
        # infer phage presence from mapped reads containing each prefix in contig:gene
        phage_reads = {}
        for phage in PHAGE_PREFIXES:
            name = phage.strip(":")
            mask = df_umis["contig:gene"].str.contains(re.escape(phage), regex=True)
            s = df_umis.loc[mask].groupby("Cell Barcode")["total_reads"].sum()
            phage_reads[f"{name}_reads"] = s

        phage_df = pd.DataFrame(phage_reads).fillna(0).reset_index().rename(columns={"index": "Cell Barcode"})
        out = out.merge(phage_df, on="Cell Barcode", how="left").fillna(0)

        def state(row) -> str:
            luz = row.get("luz19_reads", 0) > PHAGE_PRESENT_THRESHOLD_READS
            lkd = row.get("lkd16_reads", 0) > PHAGE_PRESENT_THRESHOLD_READS
            if luz and lkd:
                return "Coinfected"
            if luz:
                return "Only luz19"
            if lkd:
                return "Only lkd16"
            return "Uninfected"

        out["infection_state"] = out.apply(state, axis=1)
        out["infection_state"] = pd.Categorical(
            out["infection_state"],
            categories=["Uninfected", "Only luz19", "Only lkd16", "Coinfected"],
            ordered=True
        )

    return out


def group_slices(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Returns list of (label, df_subset) based on enabled grouping toggles.
    If no grouping enabled -> one group ("All cells").
    """
    group_cols = []
    if USE_TIMEPOINTS:
        group_cols.append("timepoint")
    if USE_INFECTION_STATES:
        group_cols.append("infection_state")

    if not group_cols:
        return [("All cells", df)]

    out = []
    for keys, sub in df.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join([str(k) for k in keys])
        out.append((label, sub))
    return out


# =============================================================================
# SATURATION / COMPLEXITY CURVE
# =============================================================================
def log_comb(n: int, k: int) -> float:
    """log(C(n,k)) using gammaln; valid for 0<=k<=n."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def expected_umis_at_depth(k_reads_per_umi: np.ndarray, depth: int) -> float:
    """
    Exact expected number of unique UMIs observed after sampling `depth` reads (without replacement)
    from a cell with per-UMI read counts k_i.

    E[UMIs] = sum_i (1 - C(N-k_i, depth) / C(N, depth))
    """
    if depth <= 0:
        return 0.0

    N = int(k_reads_per_umi.sum())
    if N <= 0:
        return 0.0
    depth = min(depth, N)

    denom = log_comb(N, depth)
    total = 0.0
    for k_i in k_reads_per_umi:
        k_i = int(k_i)
        if k_i <= 0:
            continue
        if (N - k_i) < depth:
            p_not_seen = 0.0
        else:
            p_not_seen = math.exp(log_comb(N - k_i, depth) - denom)
        total += (1.0 - p_not_seen)
    return float(total)


def compute_saturation_curve(per_umi_df: pd.DataFrame, dataset_name: str, max_depth_global: Optional[int] = None) -> pd.DataFrame:
    """
    Samples up to MAX_CELLS_FOR_SATCURVE cells.
    Returns DataFrame with columns: dataset, depth, expected_umis_mean
    """
    # Build per-cell list of k_i
    grouped = per_umi_df.groupby("Cell Barcode")["total_reads"].apply(lambda x: x.values)
    cells = grouped.index.values

    if len(cells) == 0:
        return pd.DataFrame(columns=["dataset", "depth", "expected_umis_mean"])

    # sample cells for speed
    if len(cells) > MAX_CELLS_FOR_SATCURVE:
        rng = np.random.default_rng(42)
        cells = rng.choice(cells, size=MAX_CELLS_FOR_SATCURVE, replace=False)

    k_lists = [grouped.loc[c] for c in cells]
    Ns = np.array([int(k.sum()) for k in k_lists], dtype=int)
    Nmax = int(Ns.max()) if Ns.size else 0
    if max_depth_global is not None:
        Nmax = min(Nmax, int(max_depth_global))

    if Nmax <= 0:
        return pd.DataFrame(columns=["dataset", "depth", "expected_umis_mean"])

    depths = np.unique(np.linspace(0, Nmax, SATCURVE_POINTS).astype(int))
    depths[0] = 0

    means = []
    for d in depths:
        vals = [expected_umis_at_depth(k, int(d)) for k in k_lists]
        means.append(float(np.mean(vals)) if vals else 0.0)

    return pd.DataFrame({"dataset": dataset_name, "depth": depths, "expected_umis_mean": means})


# =============================================================================
# PLOTTING
# =============================================================================
def violin_by_dataset(ax, data_by_dataset: List[np.ndarray], labels: List[str], ylabel: str, title: str, logy: bool):
    parts = ax.violinplot(data_by_dataset, showmeans=False, showmedians=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")


def box_by_dataset(ax, data_by_dataset: List[np.ndarray], labels: List[str], ylabel: str, title: str, logy: bool):
    ax.boxplot(data_by_dataset, labels=labels, showfliers=True)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")


def plot_metric_violin(all_cells: pd.DataFrame, metric: str, ylabel: str, outname: str, logy: bool):
    datasets = list(all_cells["dataset"].unique())
    groups = group_slices(all_cells)

    fig = plt.figure(figsize=STYLE["figsize_violin"])
    # If multiple groups, stack subplots vertically
    n = len(groups)
    for i, (glabel, gdf) in enumerate(groups, start=1):
        ax = fig.add_subplot(n, 1, i)
        data = []
        labels = []
        for ds in datasets:
            vals = gdf.loc[gdf["dataset"] == ds, metric].astype(float).values
            if vals.size == 0:
                vals = np.array([0.0])
            data.append(vals)
            labels.append(ds)
        violin_by_dataset(ax, data, labels, ylabel=ylabel, title=f"{metric} | {glabel}", logy=logy)

    if STYLE["tight_layout"]:
        fig.tight_layout()
    save_png(fig, outname)


def plot_metric_box(all_cells: pd.DataFrame, metric: str, ylabel: str, outname: str, logy: bool):
    datasets = list(all_cells["dataset"].unique())
    groups = group_slices(all_cells)

    fig = plt.figure(figsize=STYLE["figsize_box"])
    n = len(groups)
    for i, (glabel, gdf) in enumerate(groups, start=1):
        ax = fig.add_subplot(n, 1, i)
        data = []
        labels = []
        for ds in datasets:
            vals = gdf.loc[gdf["dataset"] == ds, metric].astype(float).values
            if vals.size == 0:
                vals = np.array([0.0])
            data.append(vals)
            labels.append(ds)
        box_by_dataset(ax, data, labels, ylabel=ylabel, title=f"{metric} | {glabel}", logy=logy)

    if STYLE["tight_layout"]:
        fig.tight_layout()
    save_png(fig, outname)


def plot_reads_vs_umis_scatter(all_cells: pd.DataFrame):
    datasets = list(all_cells["dataset"].unique())
    groups = group_slices(all_cells)

    for glabel, gdf in groups:
        fig = plt.figure(figsize=STYLE["figsize_scatter"])
        ax = fig.add_subplot(111)

        # plot each dataset as its own cloud (default colors)
        for ds in datasets:
            sub = gdf[gdf["dataset"] == ds]
            ax.scatter(
                sub["reads_per_cell"].astype(float).values,
                sub["umis_per_cell"].astype(float).values,
                s=8,
                alpha=0.6,
                label=ds
            )

        ax.set_xlabel("Mapped reads per cell")
        ax.set_ylabel("UMIs per cell")
        ax.set_title(f"Reads vs UMIs | {glabel}")
        if STYLE["scatter_logx"]:
            ax.set_xscale("log")
        if STYLE["scatter_logy"]:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9)

        if STYLE["tight_layout"]:
            fig.tight_layout()
        save_png(fig, f"scatter_reads_vs_umis__{glabel}")


def plot_reads_per_umi_hist(all_per_umi: pd.DataFrame):
    """
    Histogram of reads-per-UMI (per (cell, UMI)).
    One subplot per dataset (same figure).
    """
    datasets = list(all_per_umi["dataset"].unique())
    fig = plt.figure(figsize=STYLE["figsize_hist"])

    n = len(datasets)
    for i, ds in enumerate(datasets, start=1):
        ax = fig.add_subplot(1, n, i)
        vals = all_per_umi.loc[all_per_umi["dataset"] == ds, "total_reads"].astype(int).values
        ax.hist(vals, bins=STYLE["hist_bins_reads_per_umi"])
        ax.set_title(ds)
        ax.set_xlabel("Reads per UMI")
        ax.set_ylabel("Count")
        if STYLE["reads_per_umi_hist_logx"]:
            ax.set_xscale("log")
        if STYLE["reads_per_umi_hist_logy"]:
            ax.set_yscale("log")

    if STYLE["tight_layout"]:
        fig.tight_layout()
    save_png(fig, "hist_reads_per_umi__by_dataset")


def plot_saturation_curves(curves: pd.DataFrame):
    """
    curves columns: dataset, depth, expected_umis_mean
    One plot with one line per dataset (same figure).
    """
    if curves.empty:
        print("No saturation curve data; skipping.")
        return

    fig = plt.figure(figsize=STYLE["figsize_satcurve"])
    ax = fig.add_subplot(111)

    for ds, sub in curves.groupby("dataset"):
        ax.plot(sub["depth"].values, sub["expected_umis_mean"].values, marker="o", linewidth=1.5, label=ds)

    ax.set_xlabel("Reads sampled (per cell)")
    ax.set_ylabel("Expected UMIs observed (mean across sampled cells)")
    ax.set_title("Library complexity / saturation curve")
    ax.legend(frameon=False, fontsize=9)

    if STYLE["tight_layout"]:
        fig.tight_layout()
    save_png(fig, "saturation_curve__by_dataset")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_outdir()

    if len(DATASETS) == 0:
        raise ValueError("DATASETS is empty. Add at least 1 dataset.")
    if len(DATASETS) > 6:
        raise ValueError("Please provide at most 6 datasets (per your requirement).")

    all_cells_list = []
    all_per_umi_list = []
    satcurve_list = []

    # For shared satcurve x-axis max
    global_max_depth = None
    if MAKE_SATURATION_CURVE and SATCURVE_USE_GLOBAL_MAX:
        # compute max reads per cell across all datasets quickly
        maxes = []
        for ds in DATASETS:
            df_u = load_filtered_umis(ds["filtered_umis_path"])
            per_umi = collapse_to_unique_umi_per_cell(df_u)
            reads_per_cell = per_umi.groupby("Cell Barcode")["total_reads"].sum()
            if len(reads_per_cell) > 0:
                maxes.append(int(reads_per_cell.max()))
        global_max_depth = int(max(maxes)) if maxes else None

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n--- Loading dataset: {name} ---")

        expected_cells = load_cell_barcodes_from_gene_matrix(ds["gene_matrix_path"])
        df_umis = load_filtered_umis(ds["filtered_umis_path"])

        per_cell, per_umi = compute_per_cell_metrics(df_umis, expected_cells=expected_cells)
        per_cell = add_optional_groups(per_cell, df_umis)

        per_cell["dataset"] = name
        per_umi["dataset"] = name

        # Save per-dataset per-cell table
        out_csv = os.path.join(OUTPUT_DIR, f"qc_per_cell__{sanitize_filename(name)}.csv")
        per_cell.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        # Save per-dataset summary
        summary = {
            "dataset": name,
            "n_cells_expected_from_matrix": len(expected_cells),
            "n_cells_with_any_mapped_umi": int((per_cell["reads_per_cell"] > 0).sum()),
            "median_reads_per_cell": float(np.median(per_cell["reads_per_cell"])),
            "median_umis_per_cell": float(np.median(per_cell["umis_per_cell"])),
            "median_dup_rate": float(np.median(per_cell["dup_rate"])),
            "median_multi_hit_frac": float(np.median(per_cell["multi_hit_frac"])),
        }
        out_sum = os.path.join(OUTPUT_DIR, f"qc_summary__{sanitize_filename(name)}.csv")
        pd.DataFrame([summary]).to_csv(out_sum, index=False)
        print(f"Saved: {out_sum}")

        all_cells_list.append(per_cell)
        all_per_umi_list.append(per_umi)

        # Saturation curves
        if MAKE_SATURATION_CURVE:
            curve = compute_saturation_curve(per_umi, dataset_name=name, max_depth_global=global_max_depth)
            satcurve_list.append(curve)

    all_cells = pd.concat(all_cells_list, ignore_index=True)
    all_per_umi = pd.concat(all_per_umi_list, ignore_index=True)
    satcurves = pd.concat(satcurve_list, ignore_index=True) if satcurve_list else pd.DataFrame()

    # --- Plots (multi-dataset in same figures) ---

    # 1) Reads per cell
    plot_metric_box(
        all_cells,
        metric="reads_per_cell",
        ylabel=STYLE["ylabel_reads"],
        outname="box_reads_per_cell__by_dataset",
        logy=STYLE["reads_per_cell_logy"]
    )

    # 2) Mapping ambiguity (multi-hit fraction)
    plot_metric_box(
        all_cells,
        metric="multi_hit_frac",
        ylabel=STYLE["ylabel_multi"],
        outname="box_multi_hit_fraction__by_dataset",
        logy=False
    )

    # 3) UMIs per cell (violin)
    plot_metric_violin(
        all_cells,
        metric="umis_per_cell",
        ylabel=STYLE["ylabel_umis"],
        outname=f"violin_umis_per_cell__by_dataset__mode_{UMIS_PER_CELL_MODE}",
        logy=STYLE["umis_per_cell_logy"]
    )

    # 4) Library complexity: duplicate rate + reads/UMI hist + saturation curve
    plot_metric_box(
        all_cells,
        metric="dup_rate",
        ylabel=STYLE["ylabel_dup"],
        outname="box_duplicate_rate__by_dataset",
        logy=STYLE["dup_rate_logy"]
    )

    plot_reads_per_umi_hist(all_per_umi)

    plot_reads_vs_umis_scatter(all_cells)

    if MAKE_SATURATION_CURVE:
        plot_saturation_curves(satcurves)

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
