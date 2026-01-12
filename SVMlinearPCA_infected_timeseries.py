import scanpy as sc
import multiprocessing
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import colorsys

# sklearn imports
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize



"""
NOTE: The SVM Feature Importance plot on this DOES NOT WORK!!!! IT MESSES UP AND WILL TELL YOU THE WRONG GENES.





"""




# ------------------------------------------------------------------------------
# MULTIPROCESSING SETTINGS
# ------------------------------------------------------------------------------
sc.settings.n_jobs = multiprocessing.cpu_count()


# ------------------------------------------------------------------------------
# TIME-BASED CLASSIFICATION FUNCTION (MERGING 30min + 40min -> ">30min")
# ------------------------------------------------------------------------------
def classify_cell(cell_name):
    """
    Example: If cell_name = 'AAACCTGAGCGTAGC_1_23'
      - We split by '_' and look at the 3rd part (index=2).
      - Convert to int and assign a group based on bc1_value.
      - Now, bc1_value < 25 => "Preinfection"
               bc1_value < 49 => "10min"
               else => ">30min"
    """
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
# HELPER FUNCTIONS FOR PCA GENE LOADINGS
# ------------------------------------------------------------------------------
def get_top_genes_for_pc(adata, pc_index=0, top_n=20):
    """
    Extracts the top genes contributing to a specified principal component.

    Parameters:
    - adata: AnnData object with PCA already computed.
    - pc_index: index of the PC (0 = PC1, 1 = PC2, etc.)
    - top_n: number of top genes to return.

    Returns:
    - A pandas DataFrame with gene names and their loading values.
    """
    loadings = adata.varm["PCs"]  # shape: genes x PCs
    pc_loadings = loadings[:, pc_index]
    abs_loadings = np.abs(pc_loadings)
    top_indices = np.argsort(abs_loadings)[::-1][:top_n]
    top_genes = adata.var_names[top_indices]
    top_values = pc_loadings[top_indices]

    return pd.DataFrame({
        "Gene": top_genes,
        f"PC{pc_index + 1}_loading": top_values
    }).reset_index(drop=True)


def get_pca_loadings_figure(adata, pc_index=0, top_n=20):
    """
    Creates a Plotly bar chart of top gene loadings for a given PC.
    """
    df = get_top_genes_for_pc(adata, pc_index, top_n)
    fig = px.bar(df, x="Gene", y=f"PC{pc_index + 1}_loading",
                 title=f"Top {top_n} Gene Loadings for PC{pc_index + 1}")
    fig.update_layout(xaxis_tickangle=45)
    return fig


# ------------------------------------------------------------------------------
# DASH APP
# ------------------------------------------------------------------------------
app = dash.Dash(__name__)


# ------------------------------------------------------------------------------
# LOAD + PREPROCESS
# ------------------------------------------------------------------------------
def load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes):
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
    np.random.seed(42)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]
    print("unfiltered", adata)

    # Filter cells/genes
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)
    print("filtered", adata)

    # Label each cell based on the classify_cell function
    adata.obs["time_group"] = [classify_cell(cell) for cell in adata.obs_names]

    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.scale(adata, max_value=10)
    gene_list = adata.var_names.tolist()
    print(gene_list)

    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    return adata, raw_data_copy


# ------------------------------------------------------------------------------
# CREATE UMAP
# ------------------------------------------------------------------------------
def create_umap(adata, n_neighbors, min_dist, n_pcs):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist, n_components=2, random_state=42)
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
# DASH LAYOUT
# ------------------------------------------------------------------------------
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input",
        type="text",
        value="working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt"
    ),
    html.Br(),
    html.Label("Set min_counts for cells:"),
    dcc.Input(id="min-counts-cells", type="number", value=5),
    html.Br(),
    html.Label("Set min_counts for genes:"),
    dcc.Input(id="min-counts-genes", type="number", value=5),
    html.Br(),
    html.Label("Set n_neighbors for UMAP:"),
    dcc.Input(id="n-neighbors", type="number", value=60),
    html.Br(),
    html.Label("Set min_dist for UMAP:"),
    dcc.Input(id="min-dist", type="number", value=0.1),
    html.Br(),
    html.Label("Set n_pcs for UMAP:"),
    dcc.Input(id="n-pcs", type="number", value=12),
    html.Br(),
    html.Label("Set UMAP marker size:"),
    dcc.Input(id="umap-marker-size", type="number", value=3),
    html.Br(),
    html.Label("Set UMAP graph width:"),
    dcc.Input(id="umap-graph-width", type="number", value=1000),
    html.Br(),
    html.Label("Set UMAP graph height:"),
    dcc.Input(id="umap-graph-height", type="number", value=1000),
    html.Br(),
    # New dropdown for selecting which PC to view gene loadings for
    html.Label("Select Principal Component for Gene Loadings:"),
    dcc.Dropdown(
        id="pc-dropdown",
        options=[{'label': f'PC{i}', 'value': i - 1} for i in range(1, 13)],  # Assuming n_pcs=12 by default
        value=0
    ),
    html.Br(),
    html.Button("Update UMAP + SVM", id="update-button", n_clicks=0),
    dcc.Loading(
        type="default",
        children=[
            dcc.Graph(id='umap-svm-plot'),
            dcc.Graph(id='confusion-matrix-plot'),
            dcc.Graph(id='precision-recall-plot'),
            dcc.Graph(id='feature-importance-plot'),
            dcc.Graph(id='decision-boundary-plot'),
            dcc.Graph(id='pca-loadings-plot')
        ]
    ),
    dcc.Store(id="umap-data"),
    dcc.Store(id="raw-data"),
    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])


# ------------------------------------------------------------------------------
# CALLBACK: Main SVM + UMAP + PCA Loadings Plot (Modified for 2-class: 10min vs >30min)
# ------------------------------------------------------------------------------
@app.callback(
    [
        Output("umap-svm-plot", "figure"),
        Output("confusion-matrix-plot", "figure"),
        Output("precision-recall-plot", "figure"),
        Output("feature-importance-plot", "figure"),
        Output("decision-boundary-plot", "figure"),
        Output("umap-data", "data"),
        Output("raw-data", "data"),
        Output("pca-loadings-plot", "figure")
    ],
    Input("update-button", "n_clicks"),
    State("file-name-input", "value"),
    State("min-counts-cells", "value"),
    State("min-counts-genes", "value"),
    State("n-neighbors", "value"),
    State("min-dist", "value"),
    State("n-pcs", "value"),
    State("umap-marker-size", "value"),
    State("umap-graph-width", "value"),
    State("umap-graph-height", "value"),
    State("pc-dropdown", "value"),
    prevent_initial_call=True
)
def update_umap_and_svm(n_clicks, file_name,
                        min_counts_cells, min_counts_genes,
                        n_neighbors, min_dist, n_pcs,
                        umap_marker_size, umap_graph_width, umap_graph_height,
                        pc_value):
    # 1) Load data & create UMAP (all cells)
    adata, raw_data = load_and_preprocess_data(file_name, min_counts_cells, min_counts_genes)
    umap_df = create_umap(adata, n_neighbors, min_dist, n_pcs)

    # 2) Filter cells for SVM: only include "10min" and ">30min" groups
    valid_groups = ["10min", ">30min"]
    mask = adata.obs["time_group"].isin(valid_groups)
    filtered_adata = adata[mask, :]

    # 3) Create 2-class mapping
    group_map_2class = {"10min": 0, ">30min": 1}
    y_full = np.array([group_map_2class[g] for g in filtered_adata.obs["time_group"]])
    X_full = filtered_adata.obsm["X_pca"][:, :n_pcs]  # Using PCA data for SVM

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, stratify=y_full, random_state=42
    )

    # 5) Train the 2-class SVM
    svm_model_2class = SVC(kernel="linear", probability=True, random_state=42)
    svm_model_2class.fit(X_train, y_train)

    # 6) Predict on the filtered dataset for coloring UMAP (only cells in valid groups)
    y_pred_filtered = svm_model_2class.predict(X_full)
    inv_map_2class = {v: k for k, v in group_map_2class.items()}
    umap_df["svm_pred_class_2class"] = "N/A"
    umap_df.loc[mask, "svm_pred_class_2class"] = [inv_map_2class[val] for val in y_pred_filtered]

    # 7) Confusion Matrix for test set
    y_pred_test = svm_model_2class.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    class_order = [0, 1]
    x_labels = [inv_map_2class[i] + " (Pred)" for i in class_order]
    y_labels = [inv_map_2class[i] + " (True)" for i in class_order]

    fig_cm = px.imshow(
        cm,
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix (2-class: 10min vs >30min)"
    )

    # 8) Precision-Recall Curve (binary)
    decision_vals = svm_model_2class.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, decision_vals)
    avg_prec = average_precision_score(y_test, decision_vals)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='Precision-Recall (binary)'
    ))
    fig_pr.update_layout(
        title=f"Precision-Recall Curve (2-class, AP={avg_prec:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    # 9) Feature Importance for the 2-class linear SVM
    coefs = svm_model_2class.coef_[0]  # binary SVM returns shape=(1, n_features)
    abs_values = np.abs(coefs)
    top_indices = np.argsort(abs_values)[::-1][:100]
    print("top_indices", top_indices)
    top_features = filtered_adata.var_names[top_indices]
    top_importances = abs_values[top_indices]
    print("coefs", svm_model_2class.coef_)

    fig_feat = px.bar(
        x=top_features,
        y=top_importances,
        labels={"x": "Gene/Feature", "y": "Absolute Coefficient"},
        title="Top 10 Feature Importances (2-class Linear SVM: 10min vs >30min)"
    )
    fig_feat.update_layout(xaxis_tickangle=45)

    # 10) Decision Boundary Plot (using first 2 PCs)
    X_2d_full = filtered_adata.obsm["X_pca"][:, :2]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X_2d_full, y_full, test_size=0.3, stratify=y_full, random_state=42
    )
    svm_2d = SVC(kernel="linear", random_state=42)
    svm_2d.fit(X2_train, y2_train)

    x_min, x_max = X_2d_full[:, 0].min() - 1, X_2d_full[:, 0].max() + 1
    y_min, y_max = X_2d_full[:, 1].min() - 1, X_2d_full[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig_decision = go.Figure(data=go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z,
        colorscale="Rainbow",
        showscale=True,
        contours=dict(start=0, end=1, size=1, coloring="lines"),
        hoverinfo='skip',
        name="Decision Regions"
    ))
    color_map_2 = {0: "blue", 1: "red"}
    for label_val, label_str in inv_map_2class.items():
        subset = (y2_test == label_val)
        fig_decision.add_trace(go.Scatter(
            x=X2_test[subset, 0],
            y=X2_test[subset, 1],
            mode='markers',
            marker=dict(color=color_map_2[label_val], size=5),
            name=label_str
        ))
    fig_decision.update_layout(
        title="Decision Boundary (2-class, First 2 PCs: 10min vs >30min)",
        xaxis_title="PC1",
        yaxis_title="PC2"
    )

    # 11) UMAP Plot with 2-class SVM predictions overlay
    fig_umap = px.scatter(
        umap_df,
        x="UMAP1", y="UMAP2",
        color="svm_pred_class_2class",
        hover_data=["cell_name", "time_group"],
        title="UMAP - SVM Predicted Group (2-class: 10min vs >30min)"
    )
    fig_umap.update_traces(marker=dict(size=umap_marker_size, opacity=0.8))
    fig_umap.update_layout(width=umap_graph_width, height=umap_graph_height)

    # 12) PCA Gene Loadings Plot for the selected PC (using the full adata)
    fig_pca = get_pca_loadings_figure(adata, pc_index=pc_value, top_n=20)

    return (
        fig_umap,
        fig_cm,
        fig_pr,
        fig_feat,
        fig_decision,
        umap_df.to_json(date_format='iso', orient='split'),
        raw_data.to_json(date_format='iso', orient='split'),
        fig_pca
    )


# ------------------------------------------------------------------------------
# CALLBACK: Download Points
# ------------------------------------------------------------------------------
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-svm-plot", "selectedData"),
    State("raw-data", "data"),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, selectedData, raw_data_json):
    if not selectedData or "points" not in selectedData:
        return dash.no_update
    raw_df = pd.read_json(raw_data_json, orient='split')
    selected_ids = [pt['customdata'][0] for pt in selectedData["points"] if 'customdata' in pt]
    subset_df = raw_df.loc[selected_ids]
    if subset_df.empty:
        return dash.no_update
    return dcc.send_data_frame(subset_df.to_csv, "selected_points.csv")


# ------------------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
