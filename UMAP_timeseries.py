"""Plot a UMAP of time-based single-cell groups and report group counts."""

import os

# Detailed notes:
# - Loads and preprocesses scRNA-seq data (TSV).
# - Filters cells/genes, groups by barcode timepoints, computes PCA + UMAP.
# - Provides interactive Dash controls and CSV export of selected points.

import scanpy as sc
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------------
# USER SETTINGS (edit in PyCharm)
# ----------------------------------
DATA_DIR = "processed_data"
DATA_FILE = "luz19timeseries/luz19timeseries_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt"
FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Plotly formatting
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_FONT_FAMILY = "Arial"
PLOTLY_FONT_SIZE = 12

def apply_plotly_style(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_FONT_SIZE),
    )
    return fig

# ----------------------------------
# UMAP SETTINGS (edit in PyCharm)
# ----------------------------------
MIN_COUNTS_CELLS = 5
MIN_COUNTS_GENES = 5
N_NEIGHBORS = 60
MIN_DIST = 0.10
N_PCS = 12
PLOT_BG_COLOR = "lightgrey"
GRAPH_WIDTH = 1000
GRAPH_HEIGHT = 1000
UMAP_MARKER_SIZE = 3

COLOR_PREINFECTION = "#00008B"
COLOR_10MIN = "#2ca02c"
COLOR_30PLUS = "#ff7f0e"
LEGEND_ORDER = ["Preinfection", "10min", ">30min"]

# Updated classification function merging "30min" and "40min" into ">30min"
def classify_cell(cell_name):
    bc1_value = int(cell_name.split('_')[2])
    if bc1_value < 25:
        return "Preinfection"
    elif bc1_value < 49:
        return "10min"
    else:
        return ">30min"

# Load and preprocess the dataset
def load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes):
    # Read the raw data (tab-delimited). This DataFrame has all rows/columns unfiltered.
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)

    # Remove genes (columns) that contain a comma in their name
    removed_genes = [gene for gene in raw_data.columns if "," in gene]
    raw_data = raw_data.drop(columns=removed_genes)

    # Store removed genes separately
    removed_genes_file = "removed_genes.txt"
    with open(removed_genes_file, "w") as f:
        for gene in removed_genes:
            f.write(gene + "\n")
    print(f"Removed {len(removed_genes)} genes with commas. Saved list to {removed_genes_file}")

    # Make a copy that we'll keep for downloading later (used for before preprocessing counts)
    raw_data_copy = raw_data.copy()

    # Convert the raw data to an AnnData object for processing
    adata = sc.AnnData(raw_data)

    # Shuffle the rows of adata with a fixed seed for reproducibility.
    np.random.seed(42)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Preprocess the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)

    # Classify cells into groups based on barcode information.
    adata.obs['cell_group'] = adata.obs_names.map(classify_cell)

    # Perform PCA
    sc.tl.pca(adata, svd_solver='arpack')

    return adata, raw_data_copy

# Create UMAP DataFrame
def create_umap_df(adata, n_neighbors, min_dist, n_pcs):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    # Generate a reproducible UMAP embedding using random_state=42.
    sc.tl.umap(adata, min_dist=min_dist, n_components=2, random_state=42)
    sc.tl.leiden(adata)

    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        columns=['UMAP1', 'UMAP2'],
        index=adata.obs_names
    )
    umap_df['leiden'] = adata.obs['leiden']
    umap_df['cell_group'] = adata.obs['cell_group']
    umap_df['cell_name'] = umap_df.index

    return umap_df

def run_umap():
    adata, raw_data_copy = load_and_preprocess_data(FILE_PATH, MIN_COUNTS_CELLS, MIN_COUNTS_GENES)
    umap_df = create_umap_df(adata, N_NEIGHBORS, MIN_DIST, N_PCS)

    color_map = {
        "Preinfection": COLOR_PREINFECTION,
        "10min": COLOR_10MIN,
        ">30min": COLOR_30PLUS,
    }

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="cell_group",
        hover_data=["cell_name", "leiden", "cell_group"],
        color_discrete_map=color_map,
        category_orders={"cell_group": LEGEND_ORDER},
        width=GRAPH_WIDTH,
        height=GRAPH_HEIGHT,
        title="UMAP by Time Group",
    )
    fig.update_traces(marker=dict(size=UMAP_MARKER_SIZE, opacity=0.8))
    fig.update_layout(dragmode="lasso", plot_bgcolor=PLOT_BG_COLOR)
    apply_plotly_style(fig)

    print("Cell counts by group (before preprocessing):")
    for grp in LEGEND_ORDER:
        count = sum(1 for cell in raw_data_copy.index if classify_cell(cell) == grp)
        print(f"  {grp}: {count}")

    print("Cell counts by group (after preprocessing):")
    for grp in LEGEND_ORDER:
        count = umap_df[umap_df["cell_group"] == grp].shape[0]
        print(f"  {grp}: {count}")

    fig.show()
    return fig


if __name__ == "__main__":
    run_umap()
