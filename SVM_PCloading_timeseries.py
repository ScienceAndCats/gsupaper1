"""
PCA-SVM for prediction + gene-level interpretation (correctly).

- Train/evaluate linear SVM using PCA features (standard scRNA workflow)
- Interpret back to genes:
    (A) Project PCA-SVM weights back to genes using PCA loadings  ✅ correct
    (B) Train a gene-space linear SVM for interpretation only     ✅ optional

Keeps your same FILE_PATH convention and preprocessing steps.
"""

import os
import multiprocessing
import colorsys

import numpy as np
import pandas as pd
import scanpy as sc

import plotly.express as px
import plotly.graph_objects as go
import importlib.util

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


# ==============================================================================
# USER SETTINGS (edit in PyCharm) - kept the same
# ==============================================================================
DATA_DIR = "processed_data"
DATA_FILE = "luz19timeseries/luz19timeseries_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Folder to save graph outputs
GRAPH_OUTPUT_DIR = "graph_outputs"

SHOW_FIGURES = False  # set True if you really want pop-up/browser display

sc.settings.verbosity = 3  # 0 silent, 3 = info, 4 = debug

# ==============================================================================
# STYLE / FORMATTING BLOCK (edit in PyCharm)
# Change colors, templates, sizes, etc. in ONE place.
# ==============================================================================
STYLE = {
    # Global layout
    "template": "plotly_white",
    "font_family": "Arial",
    "font_size": 12,

    # UMAP
    "umap_marker_size": 3,
    "umap_width": 1000,
    "umap_height": 1000,

    # Confusion matrix colorscale
    "cm_colorscale": "Blues",

    # Decision boundary colorscale
    "decision_colorscale": "Rainbow",

    # Class colors (used for overlays)
    "class_colors": {
        "10min": "#1f77b4",   # blue-ish
        ">30min": "#d62728",  # red-ish
        "N/A": "#bdbdbd"
    }
}


def apply_plotly_style(fig):
    fig.update_layout(
        template=STYLE["template"],
        font=dict(family=STYLE["font_family"], size=STYLE["font_size"]),
    )
    return fig


def ensure_kaleido_available():
    """
    Plotly needs Kaleido to export static images (PNG, PDF, etc.).
    This enforces PNG export instead of silently skipping it.
    """
    if importlib.util.find_spec("kaleido") is None:
        raise RuntimeError(
            "PNG export requires 'kaleido'. Install it in your PyCharm env:\n"
            "  pip install -U kaleido\n"
            "or (conda/miniforge):\n"
            "  conda install -c conda-forge python-kaleido"
        )


def save_fig(fig, base_name: str):
    """
    Save a Plotly figure to GRAPH_OUTPUT_DIR as BOTH:
      - PNG (always; requires kaleido)
      - HTML (always; interactive)
    """
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

    # Always save interactive HTML
    html_path = os.path.join(GRAPH_OUTPUT_DIR, base_name + ".html")
    fig.write_html(html_path)

    # Always save PNG
    ensure_kaleido_available()
    png_path = os.path.join(GRAPH_OUTPUT_DIR, base_name + ".png")
    fig.write_image(png_path, scale=2)  # scale=2 gives nicer resolution

# ==============================================================================
# SVM/UMAP SETTINGS (edit in PyCharm)
# ==============================================================================
MIN_COUNTS_CELLS = 5
MIN_COUNTS_GENES = 5
N_NEIGHBORS = 60
MIN_DIST = 0.1
N_PCS = 12

TEST_SIZE = 0.25          # 25% holdout for test
RANDOM_STATE = 42

PC_INDEX_FOR_LOADINGS_PLOT = 0   # PC1 if 0, PC2 if 1, etc.
TOP_N_PCA_LOADINGS = 20
TOP_N_GENE_IMPORTANCE = 30       # for gene-weight plots


# ------------------------------------------------------------------------------
# MULTIPROCESSING SETTINGS
# ------------------------------------------------------------------------------
sc.settings.n_jobs = multiprocessing.cpu_count()


# ------------------------------------------------------------------------------
# TIME-BASED CLASSIFICATION FUNCTION (MERGING 30min + 40min -> ">30min")
# ------------------------------------------------------------------------------
def classify_cell(cell_name: str) -> str:
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "Preinfection"
    elif bc1_value < 49:
        return "10min"
    else:
        return ">30min"


# ------------------------------------------------------------------------------
# HELPER: Convert hue (0-360) to hex
# ------------------------------------------------------------------------------
def hue_to_hex(hue, saturation=1.0, lightness=0.5):
    h = hue / 360.0
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------------------------------------
# HELPER: PCA gene loadings plot
# ------------------------------------------------------------------------------
def get_top_genes_for_pc(adata, pc_index=0, top_n=20) -> pd.DataFrame:
    loadings = adata.varm["PCs"]  # genes x PCs
    pc_loadings = loadings[:, pc_index]
    abs_loadings = np.abs(pc_loadings)
    top_idx = np.argsort(abs_loadings)[::-1][:top_n]

    return pd.DataFrame({
        "Gene": adata.var_names[top_idx],
        f"PC{pc_index + 1}_loading": pc_loadings[top_idx]
    }).reset_index(drop=True)


def get_pca_loadings_figure(adata, pc_index=0, top_n=20):
    df = get_top_genes_for_pc(adata, pc_index, top_n)
    fig = px.bar(df, x="Gene", y=f"PC{pc_index + 1}_loading",
                 title=f"Top {top_n} Gene Loadings for PC{pc_index + 1}")
    fig.update_layout(xaxis_tickangle=45)
    return apply_plotly_style(fig)


# ------------------------------------------------------------------------------
# LOAD + PREPROCESS
# ------------------------------------------------------------------------------
def load_and_preprocess_data(file_name: str, min_counts_cells: int, min_counts_genes: int):
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

    # Remove genes containing commas
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)

    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for g in removed_genes:
            f.write(g + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. See {removed_genes_file}")

    raw_data_copy = raw_data.copy()

    # Convert to AnnData
    adata = sc.AnnData(raw_data)

    # Shuffle rows
    np.random.seed(RANDOM_STATE)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    # Filter cells/genes
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Label each cell
    adata.obs["time_group"] = [classify_cell(cell) for cell in adata.obs_names]

    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)

    # PCA
    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy


# ------------------------------------------------------------------------------
# CREATE UMAP
# ------------------------------------------------------------------------------
def create_umap(adata, n_neighbors: int, min_dist: float, n_pcs: int) -> pd.DataFrame:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist, n_components=2, random_state=RANDOM_STATE)
    sc.tl.leiden(adata)

    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['UMAP1', 'UMAP2'],
        index=adata.obs_names
    )
    umap_df['leiden'] = adata.obs['leiden']
    umap_df['time_group'] = adata.obs['time_group']
    umap_df['cell_name'] = umap_df.index
    return umap_df


# ------------------------------------------------------------------------------
# CORE: Train/Eval PCA-SVM + Gene Interpretations
# ------------------------------------------------------------------------------
def run_umap_and_svm():
    adata, raw_data = load_and_preprocess_data(FILE_PATH, MIN_COUNTS_CELLS, MIN_COUNTS_GENES)
    umap_df = create_umap(adata, N_NEIGHBORS, MIN_DIST, N_PCS)

    # Only include these two groups for SVM
    valid_groups = ["10min", ">30min"]
    mask = adata.obs["time_group"].isin(valid_groups)
    filtered_adata = adata[mask, :]

    # Class mapping
    group_map = {"10min": 0, ">30min": 1}
    inv_map = {v: k for k, v in group_map.items()}
    y_full = np.array([group_map[g] for g in filtered_adata.obs["time_group"]])

    # Features for prediction model: PCA
    X_pca_full = filtered_adata.obsm["X_pca"][:, :N_PCS]

    # Create ONE split index so interpretation models match the same train/test split
    idx_all = np.arange(filtered_adata.n_obs)
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y_full, random_state=RANDOM_STATE
    )

    X_pca_train, X_pca_test = X_pca_full[idx_train], X_pca_full[idx_test]
    y_train, y_test = y_full[idx_train], y_full[idx_test]

    # -------------------------
    # (1) PCA SVM: TRAIN + EVAL
    # -------------------------
    svm_pca = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    svm_pca.fit(X_pca_train, y_train)

    y_pred_test = svm_pca.predict(X_pca_test)
    cm = confusion_matrix(y_test, y_pred_test)

    x_labels = [inv_map[i] + " (Pred)" for i in [0, 1]]
    y_labels = [inv_map[i] + " (True)" for i in [0, 1]]

    fig_cm = px.imshow(
        cm,
        x=x_labels,
        y=y_labels,
        color_continuous_scale=STYLE["cm_colorscale"],
        text_auto=True,
        title=f"Confusion Matrix (PCA-SVM, test={int(TEST_SIZE*100)}% holdout)"
    )

    decision_vals = svm_pca.decision_function(X_pca_test)
    precision, recall, _ = precision_recall_curve(y_test, decision_vals)
    avg_prec = average_precision_score(y_test, decision_vals)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                name='Precision-Recall (test set)'))
    fig_pr.update_layout(
        title=f"Precision-Recall Curve (PCA-SVM, AP={avg_prec:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    # Predict across ALL filtered cells for coloring UMAP
    y_pred_all = svm_pca.predict(X_pca_full)
    umap_df["svm_pred_class_2class"] = "N/A"
    umap_df.loc[mask, "sv_map"] = [inv_map[v] for v in y_pred_all]
    umap_df.loc[mask, "svm_pred_class_2class"] = [inv_map[v] for v in y_pred_all]

    fig_umap = px.scatter(
        umap_df,
        x="UMAP1", y="UMAP2",
        color="svm_pred_class_2class",
        hover_data=["cell_name", "time_group"],
        title="UMAP - PCA-SVM Predicted Group (2-class: 10min vs >30min)",
        color_discrete_map=STYLE["class_colors"]
    )
    fig_umap.update_traces(marker=dict(size=STYLE["umap_marker_size"], opacity=0.8))
    fig_umap.update_layout(width=STYLE["umap_width"], height=STYLE["umap_height"])

    # -----------------------------------------
    # (2) Interpret PCA-SVM back to GENE weights
    # -----------------------------------------
    w_pc = svm_pca.coef_.ravel()                       # (N_PCS,)
    loadings = filtered_adata.varm["PCs"][:, :N_PCS]   # (n_genes, N_PCS)
    w_gene_from_pca = loadings @ w_pc                  # (n_genes,)

    abs_w = np.abs(w_gene_from_pca)
    top_idx = np.argsort(abs_w)[::-1][:TOP_N_GENE_IMPORTANCE]
    top_genes = filtered_adata.var_names[top_idx]
    top_vals = abs_w[top_idx]

    fig_gene_from_pca = px.bar(
        x=top_genes,
        y=top_vals,
        labels={"x": "Gene", "y": "|Projected Weight|"},
        title=f"Top {TOP_N_GENE_IMPORTANCE} Genes (Projected from PCA-SVM weights)"
    )
    fig_gene_from_pca.update_layout(xaxis_tickangle=45)

    # ----------------------------------------------------
    # (3) OPTIONAL: Gene-space SVM for interpretation only
    # ----------------------------------------------------
    X_gene_full = filtered_adata.X.toarray()
    X_gene_train, X_gene_test = X_gene_full[idx_train], X_gene_full[idx_test]

    svm_gene = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    svm_gene.fit(X_gene_train, y_train)

    w_gene_direct = svm_gene.coef_.ravel()
    abs_w2 = np.abs(w_gene_direct)
    top_idx2 = np.argsort(abs_w2)[::-1][:TOP_N_GENE_IMPORTANCE]
    top_genes2 = filtered_adata.var_names[top_idx2]
    top_vals2 = abs_w2[top_idx2]

    fig_gene_direct = px.bar(
        x=top_genes2,
        y=top_vals2,
        labels={"x": "Gene", "y": "|Coefficient|"},
        title=f"Top {TOP_N_GENE_IMPORTANCE} Genes (Direct Gene-SVM coefficients; interpretability model)"
    )
    fig_gene_direct.update_layout(xaxis_tickangle=45)

    # -------------------------
    # (4) Decision boundary plot
    # -------------------------
    X_2d_full = filtered_adata.obsm["X_pca"][:, :2]
    X2_train, X2_test = X_2d_full[idx_train], X_2d_full[idx_test]

    svm_2d = SVC(kernel="linear", random_state=RANDOM_STATE)
    svm_2d.fit(X2_train, y_train)

    x_min, x_max = X_2d_full[:, 0].min() - 1, X_2d_full[:, 0].max() + 1
    y_min, y_max = X_2d_full[:, 1].min() - 1, X_2d_full[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig_decision = go.Figure(data=go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        colorscale=STYLE["decision_colorscale"],
        showscale=True,
        contours=dict(start=0, end=1, size=1, coloring="lines"),
        hoverinfo='skip',
        name="Decision Regions"
    ))

    # Overlay TEST points only (so you're looking at generalization)
    for label_val in [0, 1]:
        label_str = inv_map[label_val]
        subset = (y_test == label_val)
        fig_decision.add_trace(go.Scatter(
            x=X2_test[subset, 0],
            y=X2_test[subset, 1],
            mode='markers',
            marker=dict(color=STYLE["class_colors"][label_str], size=6),
            name=f"{label_str} (test)"
        ))

    fig_decision.update_layout(
        title="Decision Boundary (2-class, First 2 PCs; test points overlaid)",
        xaxis_title="PC1",
        yaxis_title="PC2"
    )

    # -------------------------
    # (5) PCA gene loadings plot
    # -------------------------
    fig_pca_loadings = get_pca_loadings_figure(
        adata, pc_index=PC_INDEX_FOR_LOADINGS_PLOT, top_n=TOP_N_PCA_LOADINGS
    )

    # Apply global styling
    for fig in (
        fig_umap, fig_cm, fig_pr, fig_gene_from_pca, fig_gene_direct, fig_decision, fig_pca_loadings
    ):
        apply_plotly_style(fig)

    # Save figures to disk
    save_fig(fig_umap, "umap_pca_svm_predictions")
    save_fig(fig_cm, "confusion_matrix_testset")
    save_fig(fig_pr, "precision_recall_curve_testset")
    save_fig(fig_gene_from_pca, "gene_importance_from_pca_svm")
    save_fig(fig_gene_direct, "gene_importance_direct_gene_svm")
    save_fig(fig_decision, "decision_boundary_pc1_pc2")
    save_fig(fig_pca_loadings, "pca_loadings_PC{}".format(PC_INDEX_FOR_LOADINGS_PLOT + 1))

    # Optionally: still show figures interactively
    fig_umap.show()
    fig_cm.show()
    fig_pr.show()
    fig_gene_from_pca.show()
    fig_gene_direct.show()
    fig_decision.show()
    fig_pca_loadings.show()

    return {
        "fig_umap": fig_umap,
        "fig_cm": fig_cm,
        "fig_pr": fig_pr,
        "fig_gene_from_pca": fig_gene_from_pca,
        "fig_gene_direct": fig_gene_direct,
        "fig_decision": fig_decision,
        "fig_pca_loadings": fig_pca_loadings,
        "umap_df": umap_df,
        "raw_data": raw_data,
        "svm_pca": svm_pca,
        "svm_gene": svm_gene,
    }


if __name__ == "__main__":
    run_umap_and_svm()
