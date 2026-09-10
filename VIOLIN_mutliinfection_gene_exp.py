"""
Plot total raw counts per cell as violin plots by infection state and timepoint.

Outputs:
1. One black-and-white violin plot PNG per timepoint.
2. A CSV containing every individual value used to create the violin plots.
3. A separate CSV containing summary statistics used for the median/error bars.

The plotted error bars represent the interquartile range:
    lower = Q1 (25th percentile)
    center = median
    upper = Q3 (75th percentile)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_DIR = "processed_data"

DATA_FILE = (
    "JRG07-Sample-P3/"
    "JRG07-Sample-P3_v11_threshold_0_mixed_species_gene_matrix.txt"
)

FILE_PATH = os.path.join(DATA_DIR, DATA_FILE)

OUTPUT_DIR = "graph_outputs"


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------

MIN_COUNTS_CELLS = 4
MIN_COUNTS_GENES = 4


# -----------------------------------------------------------------------------
# Timepoints
# -----------------------------------------------------------------------------

TIMEPOINT_ORDER = [
    "5min",
    "10min",
    "15min",
    "20min",
]


# -----------------------------------------------------------------------------
# Phage identification
# -----------------------------------------------------------------------------

LUZ19_PREFIX = "luz19:"
LKD16_PREFIX = "lkd16:"

PHAGE_PRESENT_THRESHOLD = 0.0


# -----------------------------------------------------------------------------
# Groups displayed in violin plots
# -----------------------------------------------------------------------------

GROUP_ORDER = [
    "Only luz19",
    "Only lkd16",
    "Coinfected",
]


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

RAW_VALUES_CSV = "total_counts_violin_values.csv"

SUMMARY_STATS_CSV = "total_counts_violin_summary.csv"


# -----------------------------------------------------------------------------
# Plot appearance
# -----------------------------------------------------------------------------

FIGURE_SIZE = (6, 4)

PNG_DPI = 220

TITLE_FONT_SIZE = 14
AXIS_TITLE_FONT_SIZE = 12
XTICK_FONT_SIZE = 10
YTICK_FONT_SIZE = 10

VIOLIN_WIDTH = 0.8

# Number of points used by matplotlib to calculate the violin KDE curve.
VIOLIN_POINTS = 100

# Explicitly specify the KDE bandwidth so the plot can be reproduced.
VIOLIN_BW_METHOD = "scott"


# =============================================================================
# HELPERS
# =============================================================================

def ensure_output_dir():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


def classify_cell(cell_name):
    """
    Assign timepoint from the BC1 value encoded in the cell name.
    """

    bc1_value = int(
        str(cell_name).split("_")[2]
    )

    if bc1_value < 25:
        return "5min"

    elif bc1_value < 49:
        return "10min"

    elif bc1_value < 73:
        return "15min"

    else:
        return "20min"


def is_phage_gene(gene_name):
    """
    Return True for Luz19 or LKD16 genes.
    """

    gene_name = str(gene_name).lower()

    return (
        gene_name.startswith(LUZ19_PREFIX.lower())
        or
        gene_name.startswith(LKD16_PREFIX.lower())
    )


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def prepare_plot_data():
    """
    Load the original count matrix and produce one row per cell containing:

        cell_name
        timepoint
        infection_state
        total_counts_raw

    These are the exact values that will be supplied to the violin plots.
    """

    print(f"Loading: {FILE_PATH}")

    raw = pd.read_csv(
        FILE_PATH,
        sep="\t",
        index_col=0
    )

    print(
        f"Loaded {raw.shape[0]} cells "
        f"and {raw.shape[1]} genes."
    )


    # -------------------------------------------------------------------------
    # FILTER CELLS
    # Equivalent to:
    # sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_CELLS)
    # -------------------------------------------------------------------------

    cell_total_counts = raw.sum(axis=1)

    keep_cells = (
        cell_total_counts >= MIN_COUNTS_CELLS
    )

    raw = raw.loc[
        keep_cells
    ].copy()

    print(
        f"Cells remaining after filtering: "
        f"{raw.shape[0]}"
    )


    # -------------------------------------------------------------------------
    # FILTER GENES
    #
    # Retain:
    #   - genes with >= MIN_COUNTS_GENES total counts
    #   - ALL Luz19 genes
    #   - ALL LKD16 genes
    #
    # This matches the important filtering behavior of the original script.
    # -------------------------------------------------------------------------

    gene_total_counts = raw.sum(axis=0)

    phage_gene_mask = np.array(
        [
            is_phage_gene(gene)
            for gene in raw.columns
        ]
    )

    count_filter_mask = (
        gene_total_counts.to_numpy()
        >= MIN_COUNTS_GENES
    )

    keep_genes = (
        count_filter_mask
        |
        phage_gene_mask
    )

    raw = raw.loc[
        :,
        keep_genes
    ].copy()

    print(
        f"Genes remaining after filtering: "
        f"{raw.shape[1]}"
    )


    # -------------------------------------------------------------------------
    # IDENTIFY PHAGE GENES
    # -------------------------------------------------------------------------

    luz19_genes = [
        gene
        for gene in raw.columns
        if str(gene).lower().startswith(
            LUZ19_PREFIX.lower()
        )
    ]

    lkd16_genes = [
        gene
        for gene in raw.columns
        if str(gene).lower().startswith(
            LKD16_PREFIX.lower()
        )
    ]

    print(
        f"Luz19 genes found: {len(luz19_genes)}"
    )

    print(
        f"LKD16 genes found: {len(lkd16_genes)}"
    )


    # -------------------------------------------------------------------------
    # TOTAL RAW COUNTS PER CELL
    # -------------------------------------------------------------------------

    total_counts_raw = raw.sum(
        axis=1
    )


    # -------------------------------------------------------------------------
    # TOTAL PHAGE COUNTS PER CELL
    # -------------------------------------------------------------------------

    if luz19_genes:

        luz19_expression = raw[
            luz19_genes
        ].sum(axis=1)

    else:

        luz19_expression = pd.Series(
            0,
            index=raw.index
        )


    if lkd16_genes:

        lkd16_expression = raw[
            lkd16_genes
        ].sum(axis=1)

    else:

        lkd16_expression = pd.Series(
            0,
            index=raw.index
        )


    # -------------------------------------------------------------------------
    # PHAGE PRESENCE
    #
    # 0 = no phage
    # 1 = Luz19 only
    # 2 = LKD16 only
    # 3 = both
    # -------------------------------------------------------------------------

    luz19_present = (
        luz19_expression
        > PHAGE_PRESENT_THRESHOLD
    ).astype(int)

    lkd16_present = (
        lkd16_expression
        > PHAGE_PRESENT_THRESHOLD
    ).astype(int)

    phage_presence = (
        luz19_present
        +
        2 * lkd16_present
    )


    def infection_state(code):

        if code == 0:
            return "No phage"

        elif code == 1:
            return "Only luz19"

        elif code == 2:
            return "Only lkd16"

        else:
            return "Coinfected"


    # -------------------------------------------------------------------------
    # BUILD DATAFRAME THAT WILL DIRECTLY FEED THE PLOT
    # -------------------------------------------------------------------------

    plot_df = pd.DataFrame({
        "cell_name": raw.index.astype(str),

        "timepoint": [
            classify_cell(cell)
            for cell in raw.index
        ],

        "infection_state": [
            infection_state(code)
            for code in phage_presence
        ],

        "total_counts_raw":
            total_counts_raw.to_numpy(),

        "luz19_counts_raw":
            luz19_expression.to_numpy(),

        "lkd16_counts_raw":
            lkd16_expression.to_numpy(),
    })


    # Only retain groups that are actually displayed
    plot_df = plot_df[
        plot_df["infection_state"].isin(
            GROUP_ORDER
        )
    ].copy()


    # Preserve intended plotting order
    plot_df["timepoint"] = pd.Categorical(
        plot_df["timepoint"],
        categories=TIMEPOINT_ORDER,
        ordered=True
    )

    plot_df["infection_state"] = pd.Categorical(
        plot_df["infection_state"],
        categories=GROUP_ORDER,
        ordered=True
    )


    return plot_df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def calculate_summary_statistics(plot_df):
    """
    Calculate statistics for every timepoint/group.

    The graph uses:

        median
        Q1
        Q3

    with the plotted error bar extending from Q1 to Q3.
    """

    rows = []


    for timepoint in TIMEPOINT_ORDER:

        for group in GROUP_ORDER:

            values = plot_df.loc[
                (
                    plot_df["timepoint"] == timepoint
                )
                &
                (
                    plot_df["infection_state"] == group
                ),
                "total_counts_raw"
            ].to_numpy(dtype=float)


            if len(values) == 0:
                continue


            n = len(values)

            median = np.median(values)

            q1 = np.percentile(
                values,
                25
            )

            q3 = np.percentile(
                values,
                75
            )

            iqr = q3 - q1

            mean = np.mean(values)

            if n > 1:

                standard_deviation = np.std(
                    values,
                    ddof=1
                )

                sem = (
                    standard_deviation
                    /
                    np.sqrt(n)
                )

            else:

                standard_deviation = np.nan
                sem = np.nan


            rows.append({

                "timepoint":
                    timepoint,

                "infection_state":
                    group,

                "n_cells":
                    n,

                "median":
                    median,

                "q1_25th_percentile":
                    q1,

                "q3_75th_percentile":
                    q3,

                "iqr":
                    iqr,

                # Values actually used as error-bar endpoints
                "errorbar_low":
                    q1,

                "errorbar_high":
                    q3,

                # Distances passed to matplotlib yerr
                "errorbar_lower_distance":
                    median - q1,

                "errorbar_upper_distance":
                    q3 - median,

                "mean":
                    mean,

                "standard_deviation":
                    standard_deviation,

                "sem":
                    sem,

                "minimum":
                    np.min(values),

                "maximum":
                    np.max(values),
            })


    return pd.DataFrame(rows)


# =============================================================================
# SAVE CSV FILES
# =============================================================================

def save_csv_outputs(
    plot_df,
    summary_df
):
    """
    Save both reproducibility tables.
    """

    ensure_output_dir()


    # -------------------------------------------------------------------------
    # RAW VALUES
    #
    # This is the underlying data actually supplied to the violin plots.
    # One row = one plotted cell.
    # -------------------------------------------------------------------------

    raw_path = os.path.join(
        OUTPUT_DIR,
        RAW_VALUES_CSV
    )

    plot_df.to_csv(
        raw_path,
        index=False
    )

    print(
        f"Saved raw violin values: "
        f"{raw_path}"
    )


    # -------------------------------------------------------------------------
    # SUMMARY VALUES
    # -------------------------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        SUMMARY_STATS_CSV
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Saved violin summary statistics: "
        f"{summary_path}"
    )


# =============================================================================
# BLACK-AND-WHITE VIOLIN PLOTS
# =============================================================================

def plot_violins(
    plot_df,
    summary_df
):
    """
    Make one black-and-white violin plot per timepoint.

    White violin fill
    Black violin outline
    Black median point
    Black error bar from Q1 to Q3
    """

    ensure_output_dir()


    for timepoint in TIMEPOINT_ORDER:

        tp_df = plot_df[
            plot_df["timepoint"] == timepoint
        ]


        if tp_df.empty:

            print(
                f"No cells at {timepoint}; "
                f"skipping."
            )

            continue


        # ---------------------------------------------------------------------
        # Get groups that actually contain data
        # ---------------------------------------------------------------------

        groups_present = []

        data_per_group = []


        for group in GROUP_ORDER:

            values = tp_df.loc[
                tp_df["infection_state"] == group,
                "total_counts_raw"
            ].to_numpy(dtype=float)


            if len(values) > 0:

                groups_present.append(
                    group
                )

                data_per_group.append(
                    values
                )


        if not data_per_group:
            continue


        positions = np.arange(
            1,
            len(groups_present) + 1
        )


        # ---------------------------------------------------------------------
        # CREATE FIGURE
        # ---------------------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE
        )


        # ---------------------------------------------------------------------
        # VIOLIN DISTRIBUTIONS
        # ---------------------------------------------------------------------

        violin = ax.violinplot(
            data_per_group,

            positions=positions,

            widths=VIOLIN_WIDTH,

            showmeans=False,

            showmedians=False,

            showextrema=False,

            points=VIOLIN_POINTS,

            bw_method=VIOLIN_BW_METHOD,
        )


        # ---------------------------------------------------------------------
        # BLACK-AND-WHITE STYLE
        # ---------------------------------------------------------------------

        for body in violin["bodies"]:

            body.set_facecolor(
                "white"
            )

            body.set_edgecolor(
                "black"
            )

            body.set_linewidth(
                1.5
            )

            body.set_alpha(
                1.0
            )


        # ---------------------------------------------------------------------
        # MEDIAN + IQR ERROR BARS
        # ---------------------------------------------------------------------

        for position, group in zip(
            positions,
            groups_present
        ):

            row = summary_df[
                (
                    summary_df["timepoint"]
                    == timepoint
                )
                &
                (
                    summary_df["infection_state"]
                    == group
                )
            ]


            if row.empty:
                continue


            median = float(
                row["median"].iloc[0]
            )

            lower_error = float(
                row[
                    "errorbar_lower_distance"
                ].iloc[0]
            )

            upper_error = float(
                row[
                    "errorbar_upper_distance"
                ].iloc[0]
            )


            ax.errorbar(

                position,

                median,

                yerr=np.array([
                    [lower_error],
                    [upper_error]
                ]),

                fmt="o",

                color="black",

                ecolor="black",

                markerfacecolor="black",

                markeredgecolor="black",

                markersize=5,

                elinewidth=1.5,

                capsize=5,

                capthick=1.5,

                zorder=10,
            )


        # ---------------------------------------------------------------------
        # LABELS
        # ---------------------------------------------------------------------

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            groups_present,
            fontsize=XTICK_FONT_SIZE
        )

        ax.tick_params(
            axis="y",
            labelsize=YTICK_FONT_SIZE
        )


        ax.set_title(
            (
                "Total raw counts per cell by "
                f"infection state ({timepoint})"
            ),
            fontsize=TITLE_FONT_SIZE
        )


        ax.set_xlabel(
            "Infection state",
            fontsize=AXIS_TITLE_FONT_SIZE
        )


        ax.set_ylabel(
            "Total raw counts per cell",
            fontsize=AXIS_TITLE_FONT_SIZE
        )


        # ---------------------------------------------------------------------
        # KEEP FIGURE PURE BLACK + WHITE
        # ---------------------------------------------------------------------

        ax.set_facecolor(
            "white"
        )

        fig.patch.set_facecolor(
            "white"
        )


        # Black axes
        for spine in ax.spines.values():

            spine.set_color(
                "black"
            )


        ax.tick_params(
            colors="black"
        )


        # No colored/grid background
        ax.grid(
            False
        )


        fig.tight_layout()


        # ---------------------------------------------------------------------
        # SAVE PNG
        # ---------------------------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"total_counts_raw_violin_{timepoint}.png"
        )


        fig.savefig(
            output_path,
            dpi=PNG_DPI,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


        print(
            f"Saved violin plot: "
            f"{output_path}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    ensure_output_dir()


    # Get exact per-cell values used for plotting
    plot_df = prepare_plot_data()


    # Calculate median / error bars / other descriptive statistics
    summary_df = calculate_summary_statistics(
        plot_df
    )


    # Save data BEFORE making the figures
    save_csv_outputs(
        plot_df,
        summary_df
    )


    # Generate violin plots directly from the same dataframe
    plot_violins(
        plot_df,
        summary_df
    )


    print(
        f"\nDone. Outputs saved to: "
        f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()