"""
Interactive Dash App for Single-Cell RNA-seq UMAP Visualization and Analysis

## Functionality:
- Loads and preprocesses single-cell gene expression data from a tab-delimited file.
- Filters cells and genes based on user-defined minimum count thresholds.
- Classifies cells into time-based groups based on barcode identifiers.
- Computes PCA and generates UMAP embeddings for visualization.
- Allows interactive selection of cell groups using a Dash-based web interface.
- Supports dynamic parameter tuning for UMAP visualization.
- Enables color customization for different cell groups.
- Displays before/after preprocessing cell counts per group.
- Provides an option to download selected data points from UMAP.

## Inputs:
- Tab-separated gene expression matrix (CSV/TSV).
  - Rows: Cells (named with barcode identifiers).
  - Columns: Genes.
  - Values: Expression counts.
- User-defined parameters for:
  - Minimum counts per cell and gene.
  - UMAP parameters (`n_neighbors`, `min_dist`, `n_pcs`).
  - Custom colors for different cell groups.
  - Legend order for cell group display.
  - Plot customization: background color, graph dimensions, and color scale range.

## Outputs:
- Interactive UMAP plot with cell group classifications.
- Printed preprocessing details (genes removed, filtering steps).
- Cell counts per group before and after filtering.
- CSV file containing selected data points (`selected_points.csv`).

## Dependencies:
- scanpy, pandas, numpy, dash, plotly, matplotlib

## Usage:
1. Run the script (`python script.py`).
2. Open the Dash web interface in the browser.
3. Adjust parameters, update UMAP, and explore the data interactively.
4. Select data points and download them as a CSV file.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

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

# App layout
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value='working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt'
    ),
    html.Br(), html.Br(),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5, step=1, min=1),
    html.Br(),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5, step=1, min=1),
    html.Br(),
    html.Label("Set n_neighbors for UMAP:"),
    dcc.Input(id="n-neighbors-input", type="number", value=60, step=1, min=2, max=200),
    html.Br(),
    html.Label("Set min_dist for UMAP:"),
    dcc.Input(id="min-dist-input", type="number", value=0.10, step=0.01, min=0.01, max=1.00),
    html.Br(),
    html.Label("Set n_pcs for UMAP:"),
    dcc.Input(id="n-pcs-input", type="number", value=12, step=1, min=2),
    html.Br(), html.Br(),
    # Updated group color controls (only three groups now):
    html.Label("Preinfection Color:"),
    dcc.Input(id="color-preinfection", type="text", value="#00008B"),
    html.Br(),
    html.Label("10min Color:"),
    dcc.Input(id="color-10min", type="text", value="#2ca02c"),
    html.Br(),
    html.Label(">30min Color:"),
    dcc.Input(id="color-30plus", type="text", value="#ff7f0e"),
    html.Br(),
    html.Label("Legend Order (comma separated):"),
    dcc.Input(id="legend-order", type="text", value="Preinfection,10min,>30min"),
    html.Br(), html.Br(),
    # New plot customization inputs:
    html.Label("Set plot background color (plot_bgcolor):"),
    dcc.Input(id="plot-bgcolor-input", type="text", value="lightgrey"),
    html.Br(),
    html.Label("Set graph width:"),
    dcc.Input(id="graph-width-input", type="number", value=1000, step=10),
    html.Br(),
    html.Label("Set graph height:"),
    dcc.Input(id="graph-height-input", type="number", value=1000, step=10),
    html.Br(),
    html.Label("Set color scale minimum (no effect for discrete colors):"),
    dcc.Input(id="color-min-input", type="number", value=0, step=0.1),
    html.Br(),
    html.Label("Set color scale maximum (no effect for discrete colors):"),
    dcc.Input(id="color-max-input", type="number", value=10, step=0.1),
    html.Br(), html.Br(),
    html.Button("Update UMAP", id="update-button", n_clicks=0),
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[dcc.Graph(id='umap-plot')]
    ),
    # New div to display cell counts by group (before and after preprocessing)
    html.Div(id="group-counts", style={"marginTop": "20px", "fontSize": "16px"}),
    dcc.Store(id="umap-data"),
    dcc.Store(id="raw-data"),
    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])

# Callback to update UMAP, store data, and show cell counts by group.
@app.callback(
    [Output("umap-plot", "figure"),
     Output("umap-data", "data"),
     Output("raw-data", "data"),
     Output("group-counts", "children")],
    Input("update-button", "n_clicks"),
    State("file-name-input", "value"),
    State("min-counts-cells", "value"),
    State("min-counts-genes", "value"),
    State("n-neighbors-input", "value"),
    State("min-dist-input", "value"),
    State("n-pcs-input", "value"),
    State("color-preinfection", "value"),
    State("color-10min", "value"),
    State("color-30plus", "value"),
    State("legend-order", "value"),
    # New customization inputs:
    State("plot-bgcolor-input", "value"),
    State("graph-width-input", "value"),
    State("graph-height-input", "value"),
    State("color-min-input", "value"),
    State("color-max-input", "value"),
    prevent_initial_call=True
)
def update_umap(n_clicks, file_name, min_counts_cells, min_counts_genes,
                n_neighbors, min_dist, n_pcs,
                color_preinfection, color_10min, color_30plus, legend_order,
                plot_bgcolor, graph_width, graph_height, color_min, color_max):
    # Load and preprocess data
    adata, raw_data_copy = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap_df(adata, n_neighbors, min_dist, n_pcs)

    # Build the discrete color map for cell groups.
    color_map = {
        "Preinfection": color_preinfection,
        "10min": color_10min,
        ">30min": color_30plus
    }

    # Parse the legend order from the comma-separated string.
    legend_order_list = [grp.strip() for grp in legend_order.split(',') if grp.strip()]

    # Create the UMAP scatter plot.
    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color='cell_group',
        hover_data=['cell_name', 'leiden', 'cell_group'],
        custom_data=['cell_name'],
        color_discrete_map=color_map,
        category_orders={'cell_group': legend_order_list},
        width=graph_width,
        height=graph_height
        # range_color=[color_min, color_max]  # Not used for discrete color mapping.
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(
        dragmode='lasso',
        plot_bgcolor=plot_bgcolor
    )

    # Compute cell counts per group BEFORE preprocessing (using raw_data_copy)
    before_counts_text = "<u>Before Preprocessing:</u><br>"
    for grp in legend_order_list:
        count = sum(1 for cell in raw_data_copy.index if classify_cell(cell) == grp)
        before_counts_text += f"{grp}: {count}<br>"

    # Compute cell counts per group AFTER preprocessing (using umap_df)
    after_counts_text = "<u>After Preprocessing:</u><br>"
    for grp in legend_order_list:
        count = umap_df[umap_df["cell_group"] == grp].shape[0]
        after_counts_text += f"{grp}: {count}<br>"

    counts_text = "<b>Cell Counts:</b><br>" + before_counts_text + "<br>" + after_counts_text

    return fig, umap_df.to_json(date_format='iso', orient='split'), raw_data_copy.to_json(date_format='iso', orient='split'), counts_text

# Callback to download selected points using the raw_data copy.
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-plot", "selectedData"),
    State("raw-data", "data"),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, selectedData, raw_data_json):
    if not selectedData or "points" not in selectedData:
        return dash.no_update

    raw_df = pd.read_json(raw_data_json, orient='split')
    selected_names = [point['customdata'][0] for point in selectedData["points"] if "customdata" in point]
    selected_df = raw_df[raw_df.index.isin(selected_names)]

    if selected_df.empty:
        return dash.no_update

    return dcc.send_data_frame(selected_df.to_csv, "selected_points.csv")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
