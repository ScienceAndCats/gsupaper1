import scanpy as sc
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px



"""
This just displays a heatmap of the phage genes by time point. You can customize abunch of stuff in it.
It displays MEAN gene expression.


Plug in this data:
working_data/preprocessed_PETRI_outputs/09Oct2024_Luz19_0-40min_18000/09Oct2024_mixed_species_gene_matrix_preprocessed.txt

gp0.1,gp1,gp2,gp3,gp4,gp5,gp6,gp7,gp8,gp9,gp10,gp11,gp12,gp12.1,gp13,gp13.1,gp14,gp15,gp16,gp18,gp19,gp20,gp21,gp22,gp23,gp24,gp25,gp25.1,gp26,gp27,gp28,gp29,gp30,gp31,gp32,gp33,gp34,gp35,gp36,gp37,gp38,gp39,gp40,gp41,gp42,gp43,gp44,gp45,gp46,gp46.1,gp47,gp48,gp49

"""




# Initialize Dash app
app = dash.Dash(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data(file_name, min_counts_cells=4, min_counts_genes=4):
    # Read the raw data (tab-delimited). This DataFrame has all rows/columns unfiltered.
    raw_data = pd.read_csv(file_name, sep='\t', index_col=0)
    # Make a copy for downloading later
    raw_data_copy = raw_data.copy()

    # Convert raw data to an AnnData object
    adata = sc.AnnData(raw_data)

    # Shuffle rows
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices, :]

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=min_counts_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts_genes)

    # Identify uninfected cells based on "luz19:" genes
    luz19_genes = adata.var_names[adata.var_names.str.contains("luz19:")]
    luz19_gene_indices = np.where(adata.var_names.isin(luz19_genes))[0]

    def label_infection(cell_expression):
        luz19_expression = cell_expression[luz19_gene_indices]
        return "uninfected" if np.all(luz19_expression == 0) else "infected"

    adata.obs['infection_status'] = [label_infection(cell) for cell in adata.X]

    # Classify cells into groups based on barcodes (these groups represent timepoints)
    def classify_cell(cell_name):
        bc1_value = int(cell_name.split('_')[2])
        if bc1_value < 25:
            return "Preinfection"
        elif bc1_value < 49:
            return "10min"
        elif bc1_value < 73:
            return ">30min" # modified so that 30 and 40min lumped together
        else:
            return ">30min"   # modified so that 30 and 40min lumped together

    adata.obs['cell_group'] = adata.obs_names.map(classify_cell)

    return adata, raw_data_copy

# Create Bulk DataFrame by aggregating single-cell data for genes with "luz19"
def create_bulk_df(adata):
    # Convert the processed data matrix into a DataFrame
    expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    # Keep only genes with "luz19" in the name
    luz19_cols = expr_df.columns[expr_df.columns.str.contains("luz19")]
    expr_df = expr_df[luz19_cols]
    # Add cell group information
    expr_df['cell_group'] = adata.obs['cell_group']
    # Group cells by their cell_group and sum the expression (i.e. create a "bulk" sample)
    #bulk_df = expr_df.groupby('cell_group').sum().reset_index()
    bulk_df = expr_df.groupby('cell_group').mean().reset_index() #taking the mean instead of sum

    # Melt the DataFrame to long format for Plotly
    bulk_melt = bulk_df.melt(id_vars='cell_group', var_name='gene', value_name='expression')
    # Remove any genes that have a comma in their name, these are multi-hits
    bulk_melt = bulk_melt[~bulk_melt['gene'].str.contains(",")]
    return bulk_melt








# App layout with additional inputs for custom text, font sizes, and graph size.
app.layout = html.Div([
    html.Label("Set CSV file name:"),
    dcc.Input(
        id="file-name-input", type="text",
        value='09Oct2024_luz19_initial_v11_threshold_0_mixed_species_gene_matrix_multihitcombo.txt'
    ),
    html.Br(),
    html.Label("Set color scale minimum (zmin):"),
    dcc.Input(
        id="yaxis-min-input", type="number", value=0, step=1
    ),
    html.Br(),
    html.Label("Set color scale maximum (zmax):"),
    dcc.Input(
        id="yaxis-max-input", type="number", value=1, step=0.2
    ),
    html.Br(),
    html.Label("Set Title Font Size:"),
    dcc.Input(
        id="title-font-size-input", type="number", value=20, step=1
    ),
    html.Br(),
    html.Label("Set Axis Title Font Size:"),
    dcc.Input(
        id="axis-title-font-size-input", type="number", value=14, step=1
    ),
    html.Br(),
    html.Label("Set X Tick Font Size:"),
    dcc.Input(
        id="x-tick-font-size-input", type="number", value=12, step=1
    ),
    html.Br(),
    html.Label("Set Y Tick Font Size:"),
    dcc.Input(
        id="y-tick-font-size-input", type="number", value=12, step=1
    ),
    html.Br(),
    html.Label("Set Legend Font Size:"),
    dcc.Input(
        id="legend-font-size-input", type="number", value=12, step=1
    ),
    html.Br(),
    html.Label("Set Plot Title:"),
    dcc.Input(
        id="plot-title-input", type="text",
        value="Heatmap of 'luz19' Gene Expression Across Timepoints"
    ),
    html.Br(),
    html.Label("Set X-Axis Label:"),
    dcc.Input(
        id="x-axis-label-input", type="text", value="Timepoint"
    ),
    html.Br(),
    html.Label("Set Y-Axis Label:"),
    dcc.Input(
        id="y-axis-label-input", type="text", value="Gene"
    ),
    html.Br(),
    html.Label("Set Gene Order (comma-separated):"),
    dcc.Input(
        id="gene-order-input", type="text", value="",
        placeholder="Enter gene names separated by commas"
    ),
    html.Br(),
    html.Label("Set Graph Width (in px):"),
    dcc.Input(
        id="graph-width-input", type="number", value=800, step=10
    ),
    html.Br(),
    html.Label("Set Graph Height (in px):"),
    dcc.Input(
        id="graph-height-input", type="number", value=600, step=10
    ),
    html.Br(),
    html.Button("Update Heatmap", id="update-button", n_clicks=0),
    dcc.Loading(
        id="loading-spinner",
        type="default",
        children=[html.Div(id='plots-container')]
    ),
    # Store the aggregated bulk DataFrame as JSON
    dcc.Store(id="umap-data"),
    # Store the raw (unprocessed) DataFrame as JSON
    dcc.Store(id="raw-data"),
    html.Button("Download Selected Points", id="download-btn", n_clicks=0),
    dcc.Download(id="download-dataframe")
])

# Callback to update the heatmap based on inputs.
@app.callback(
    [Output("plots-container", "children"),
     Output("umap-data", "data"),
     Output("raw-data", "data")],
    Input("update-button", "n_clicks"),
    State("file-name-input", "value"),
    State("yaxis-min-input", "value"),
    State("yaxis-max-input", "value"),
    State("title-font-size-input", "value"),
    State("axis-title-font-size-input", "value"),
    State("x-tick-font-size-input", "value"),
    State("y-tick-font-size-input", "value"),
    State("legend-font-size-input", "value"),
    State("plot-title-input", "value"),
    State("x-axis-label-input", "value"),
    State("y-axis-label-input", "value"),
    State("gene-order-input", "value"),
    State("graph-width-input", "value"),
    State("graph-height-input", "value"),
    prevent_initial_call=True
)
def update_heatmap(n_clicks, file_name, yaxis_min, yaxis_max,
                   title_font_size, axis_title_font_size,
                   x_tick_font_size, y_tick_font_size, legend_font_size,
                   plot_title, x_axis_label, y_axis_label,
                   gene_order_str, graph_width, graph_height):
    # Load and preprocess the data using default filtering values
    adata, raw_data = load_and_preprocess_data(file_name, 4, 4)

    # Aggregate the single-cell data into bulk data (only for genes containing "luz19")
    bulk_melt = create_bulk_df(adata)

    # Pivot the data so that rows = genes and columns = timepoints (cell_group)
    heat_df = bulk_melt.pivot(index='gene', columns='cell_group', values='expression')
    # Ensure the columns appear in the desired order:
    expected_order = ['Preinfection', '10min', '>30min']
    columns_order = [col for col in expected_order if col in heat_df.columns]
    heat_df = heat_df[columns_order]

    # Remove "luz19:" prefix from gene names for display.
    heat_df.index = heat_df.index.str.replace("luz19:", "", regex=False)

    # If a custom gene order is provided, reorder the rows accordingly.
    if gene_order_str and gene_order_str.strip():
        # Parse the comma-separated gene names
        gene_order_list = [gene.strip() for gene in gene_order_str.split(',') if gene.strip()]
        # Only use genes that exist in the current DataFrame.
        new_order = [gene for gene in gene_order_list if gene in heat_df.index]
        # Append any genes not mentioned.
        # remaining_genes = [gene for gene in heat_df.index if gene not in new_order]  # took this out, I don't want to display genes I don't specify
        # new_order.extend(remaining_genes) # took this out, I don't want to display genes I don't specify
        heat_df = heat_df.loc[new_order]

    # Create heatmap using px.imshow.
    fig = px.imshow(
        heat_df,
        labels=dict(x=x_axis_label, y=y_axis_label, color="Expression"),
        x=columns_order,
        y=heat_df.index,
        aspect="auto",
        zmin=yaxis_min,
        zmax=yaxis_max
    )
    # Update layout with custom text, font sizes, and graph dimensions.
    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=title_font_size)),
        xaxis=dict(
            title=dict(text=x_axis_label, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=x_tick_font_size)
        ),
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(size=axis_title_font_size)),
            tickfont=dict(size=y_tick_font_size)
        ),
        legend=dict(font=dict(size=legend_font_size)),
        width=graph_width,
        height=graph_height
    )

    return (
        [dcc.Graph(figure=fig)],
        bulk_melt.to_json(date_format='iso', orient='split'),
        raw_data.to_json(date_format='iso', orient='split')
    )

# Callback to download selected points (unprocessed data)
@app.callback(
    Output("download-dataframe", "data"),
    Input("download-btn", "n_clicks"),
    State("umap-data", "data"),
    State("raw-data", "data"),
    prevent_initial_call=True
)
def download_selected_points(n_clicks, bulk_data, raw_data):
    """
    When the user clicks 'Download Selected Points', this returns the entire raw data as a CSV.
    """
    raw_df = pd.read_json(raw_data, orient='split')
    return dcc.send_data_frame(
        raw_df.to_csv,
        filename="selected_points.csv",
        sep='\t'
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
