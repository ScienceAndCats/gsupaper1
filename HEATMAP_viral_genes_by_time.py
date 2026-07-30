"""Build a heatmap of phage gene expression by timepoint."""

import os
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

# Gene order file (must sit in the same directory as this script)
GENE_ORDER_FILE = "heatmap_genes.txt"

# Output settings
OUTPUT_DIR = "graph_outputs"
OUTPUT_PNG = "luz19_gene_expression_heatmap.png"

# Plotly formatting
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_FONT_FAMILY = "Arial"
PLOTLY_FONT_SIZE = 8


def apply_plotly_style(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=PLOTLY_FONT_FAMILY, size=PLOTLY_FONT_SIZE),
    )
    return fig


# Notes:
# - Displays mean expression for luz19 genes by timepoint.
# - Expected input is a tab-delimited gene expression matrix in processed_data.


# ----------------------------------
# HEATMAP SETTINGS (edit in PyCharm)
# ----------------------------------
MIN_COUNTS_CELLS = 4
MIN_COUNTS_GENES = 4
ZMIN = 0
ZMAX = 1
TITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 14
X_TICK_FONT_SIZE = 12
Y_TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
PLOT_TITLE = "Heatmap of 'luz19' Gene Expression Across Timepoints"
X_AXIS_LABEL = "Timepoint"
Y_AXIS_LABEL = "Gene"
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 600


def load_gene_order(file_path: str):
    """Load gene order from a comma-separated text file."""
    with open(file_path, "r") as f:
        line = f.read().strip()
    return [g.strip() for g in line.split(",") if g.strip()]


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
            return ">30min"  # modified so that 30 and 40min lumped together
        else:
            return ">30min"  # modified so that 30 and 40min lumped together

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
    # Group cells by their cell_group and take the mean expression (bulk sample)
    bulk_df = expr_df.groupby('cell_group').mean().reset_index()

    # Melt the DataFrame to long format for Plotly
    bulk_melt = bulk_df.melt(id_vars='cell_group', var_name='gene', value_name='expression')
    # Remove any genes that have a comma in their name, these are multi-hits
    bulk_melt = bulk_melt[~bulk_melt['gene'].str.contains(",")]
    return bulk_melt


def build_heatmap():
    adata, raw_data = load_and_preprocess_data(FILE_PATH, MIN_COUNTS_CELLS, MIN_COUNTS_GENES)

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

    # Load gene order from file located in the same directory as this script
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    gene_order_path = os.path.join(base_dir, GENE_ORDER_FILE)
    gene_order_list = load_gene_order(gene_order_path)

    # Apply ordering and drop genes not listed
    ordered_genes = [g for g in gene_order_list if g in heat_df.index]
    heat_df = heat_df.loc[ordered_genes]

    # Print the genes that are actually plotted
    print("\nGenes plotted in heatmap:")
    print(", ".join(heat_df.index.tolist()))

    # Rotate heatmap 90 degrees clockwise:
    # transpose + reverse rows
    rotated_df = heat_df.T.iloc[::-1]

    fig = px.imshow(
        rotated_df,
        labels=dict(x=Y_AXIS_LABEL, y=X_AXIS_LABEL, color="Expression"),
        x=rotated_df.columns,
        y=rotated_df.index,
        aspect="auto",
        zmin=ZMIN,
        zmax=ZMAX
    )
    fig.update_xaxes(
        tickangle=90,
        tickmode="array",
        tickvals=list(rotated_df.columns),
        ticktext=list(rotated_df.columns),
    )

    # Update layout with custom text, font sizes, and graph dimensions.
    fig.update_layout(
        title=dict(text=PLOT_TITLE, font=dict(size=TITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text=X_AXIS_LABEL, font=dict(size=AXIS_TITLE_FONT_SIZE)),
            tickfont=dict(size=X_TICK_FONT_SIZE)
        ),
        yaxis=dict(
            title=dict(text=Y_AXIS_LABEL, font=dict(size=AXIS_TITLE_FONT_SIZE)),
            tickfont=dict(size=Y_TICK_FONT_SIZE)
        ),
        legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
        width=GRAPH_WIDTH,
        height=GRAPH_HEIGHT
    )
    apply_plotly_style(fig)

    # Ensure output directory exists and save PNG
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_PNG)
    fig.write_image(output_path, scale=2)
    print(f"\nHeatmap saved to: {output_path}")

    fig.show()
    return fig


if __name__ == "__main__":
    build_heatmap()
