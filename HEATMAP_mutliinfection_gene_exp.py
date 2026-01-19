"""
Heatmaps of mean gene expression by infection state, one heatmap per timepoint,
plus raw-support tables (n_cells, raw hits, mean hits/cell),
PLUS statistical tests: coinfected vs single-infected (per phage, per timepoint).

What gets tested:
- For luz19 genes: Coinfected vs Only luz19
- For lkd16 genes: Coinfected vs Only lkd16

Tests:
- Mann–Whitney U on per-cell expression values (default uses normalized log1p in adata.X)
- Fisher exact test on detection rate (>0 raw counts)
- Benjamini–Hochberg FDR correction across genes

Outputs (in graph_outputs/):
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
MIN_COUNTS_CELLS = 5
MIN_COUNTS_GENES = 5

# Normalization
DO_NORMALIZE = True     # normalize_total + log1p
TARGET_SUM = 1e4

# Timepoints (order matters)
TIMEPOINT_ORDER = ["5min", "10min", "15min", "20min"]

# Phage gene prefixes
PHAGE_PREFIXES = ["luz19:", "lkd16:"]

# Gene selection: top variable genes per timepoint, then union across timepoints
TOP_N_VAR_GENES = 50

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
    "mean_heatmap_figsize": (10, 0.22 * TOP_N_VAR_GENES + 4),
    "mean_heatmap_cmap": "viridis",
    "mean_heatmap_share_color_scale": True,

    # Mean-hits-per-cell heatmap (raw counts / n_cells)
    "make_hits_per_cell_heatmaps": True,
    "hits_per_cell_figsize": (10, 0.22 * TOP_N_VAR_GENES + 4),
    "hits_per_cell_cmap": "magma",
    "hits_per_cell_share_color_scale": True,

    # Annotate numbers on hits-per-cell heatmap blocks
    "annotate_hits_per_cell": True,
    "annotate_fmt": "{:.2f}",
    "annotate_fontsize": 8,
    "annotate_max_genes": 60,

    # Labels
    "xlabel": "Infection state",
    "ylabel": "Gene",
    "title_mean_prefix": "Mean expression heatmap at ",
    "title_hits_prefix": "Mean hits/cell (raw) heatmap at ",

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


def pick_genes_union_high_variance_per_timepoint(adata: sc.AnnData, n_per_tp: int) -> list[str]:
    genes_union = set()

    for tp in TIMEPOINT_ORDER:
        ad_tp = adata[adata.obs["timepoint"] == tp]
        if ad_tp.n_obs == 0:
            print(f"WARNING: No cells at {tp}; skipping variance selection for that timepoint.")
            continue

        X_tp = to_dense_if_needed(ad_tp.X)
        var_tp = X_tp.var(axis=0)

        n = min(n_per_tp, ad_tp.n_vars)
        top_idx = np.argsort(var_tp)[::-1][:n]
        genes_union.update(list(ad_tp.var_names[top_idx]))

    genes_list = sorted(list(genes_union))
    print(f"Selected {len(genes_list)} genes (union of top-{n_per_tp} variance per timepoint).")
    return genes_list


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
    sc.pp.filter_genes(adata, min_counts=MIN_COUNTS_GENES)

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
            return "Only luz19"
        if code == 2:
            return "Only lkd16"
        return "Coinfected"

    adata.obs["phage_state"] = [phage_state(int(x)) for x in adata.obs["phage_presence"]]
    adata.obs["phage_state"] = pd.Categorical(
        adata.obs["phage_state"],
        categories=["Uninfected", "Only luz19", "Only lkd16", "Coinfected"],
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
def plot_heatmaps(genes_present, mats_by_tp, title_prefix, cmap_name, figsize, share_scale, out_prefix,
                  annotate=False, annotate_fmt="{:.2f}", annotate_fontsize=8, annotate_max_genes=60):
    conds = ["Uninfected", "Single infection", "Coinfected"]

    vmin_global = vmax_global = None
    if share_scale:
        flat = np.concatenate([m.ravel() for m in mats_by_tp.values()])
        valid = flat[~np.isnan(flat)]
        if valid.size > 0:
            vmin_global, vmax_global = float(valid.min()), float(valid.max())

    base_cmap = plt.cm.get_cmap(cmap_name or "viridis")
    cmap = copy(base_cmap)
    cmap.set_bad(color="white")

    do_annotate = annotate and (len(genes_present) <= annotate_max_genes)

    for tp in TIMEPOINT_ORDER:
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
        cbar.set_label(out_prefix)

        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(conds, rotation=0)

        ax.set_yticks(np.arange(len(genes_present)))
        ax.set_yticklabels(genes_present)

        ax.set_xlabel(STYLE["xlabel"])
        ax.set_ylabel(STYLE["ylabel"])
        ax.set_title(f"{title_prefix}{tp}")

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
        save_png(fig, f"{out_prefix}_heatmap_{tp}_genes_union_topvar{TOP_N_VAR_GENES}")


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
      - Compare Coinfected vs phage-specific Single (Only luz19 OR Only lkd16)
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
                single_label = "Only luz19"
            elif phage_name == "lkd16":
                single_label = "Only lkd16"
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
            phage_genes = [g for g in adata.var_names if g.startswith(phage_prefix)]
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
      - Only luz19
      - Only lkd16
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
    #    1 = Only luz19
    #    2 = Only lkd16
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
            return "Only luz19"
        elif code == 2:
            return "Only lkd16"
        else:
            return "Coinfected"

    if "infection_state_detail" not in adata.obs.columns:
        adata.obs["infection_state_detail"] = [
            detailed_state(int(x)) for x in adata.obs["phage_presence"]
        ]
        adata.obs["infection_state_detail"] = pd.Categorical(
            adata.obs["infection_state_detail"],
            categories=["No phage", "Only luz19", "Only lkd16", "Coinfected"],
            ordered=True,
        )

    # We’ll only plot these three conditions
    conds_to_plot = ["Only luz19", "Only lkd16", "Coinfected"]

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

        ax.set_title(f"Total raw counts per cell by infection state ({tp})")
        ax.set_xlabel("Infection state")
        ax.set_ylabel("total_counts_raw")
        #ax.set_yscale("log")  # often helpful for count data; remove if you prefer linear

        fig.tight_layout()
        save_png(fig, f"total_counts_raw_boxplot_{tp}")




# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_outdir()
    adata = setup_adata()

    # Select genes: union of top-variance genes per timepoint
    genes = pick_genes_union_high_variance_per_timepoint(adata, TOP_N_VAR_GENES)
    print(f"Heatmaps will use {len(genes)} genes (same gene list for all timepoints).")

    # Compute per-timepoint matrices + tables
    genes_present, mats_mean, mats_hits, mats_hpc, ncells_by_tp, tables_by_tp = compute_tp_matrices_and_tables(
        adata, genes
    )

    # Mean-expression heatmaps
    plot_heatmaps(
        genes_present=genes_present,
        mats_by_tp=mats_mean,
        title_prefix=STYLE["title_mean_prefix"],
        cmap_name=STYLE["mean_heatmap_cmap"],
        figsize=STYLE["mean_heatmap_figsize"],
        share_scale=STYLE["mean_heatmap_share_color_scale"],
        out_prefix="mean_expr",
        annotate=False,
    )

    # Hits-per-cell heatmaps (raw)
    if STYLE.get("make_hits_per_cell_heatmaps", True):
        plot_heatmaps(
            genes_present=genes_present,
            mats_by_tp=mats_hpc,
            title_prefix=STYLE["title_hits_prefix"],
            cmap_name=STYLE["hits_per_cell_cmap"],
            figsize=STYLE["hits_per_cell_figsize"],
            share_scale=STYLE["hits_per_cell_share_color_scale"],
            out_prefix="mean_hits_per_cell_raw",
            annotate=STYLE.get("annotate_hits_per_cell", True),
            annotate_fmt=STYLE.get("annotate_fmt", "{:.2f}"),
            annotate_fontsize=STYLE.get("annotate_fontsize", 8),
            annotate_max_genes=STYLE.get("annotate_max_genes", 60),
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
