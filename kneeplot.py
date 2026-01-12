import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- SETTINGS YOU CAN TWEAK -----------------
# Axis limits (set to None to auto-scale)
X_MIN = None      # e.g. 1
X_MAX = None      # e.g. 5000
Y_MIN = 0.0       # e.g. 0.0
Y_MAX = 1.0       # e.g. 0.5 to zoom in

# ----------------------------------------------------------

# 1. Read your table
df = pd.read_csv(
    "~/PhETRIseq/together/JRG08-2PMP_output/JRG08-2PMP_bc1_cumulative_frequency_table.txt", #"data/JRG09-UI_bc1_cumulative_frequency_table.txt",
    sep="\t"

)

print(df.columns)  # sanity check

# 2. Make sure barcodes are ordered by read count (largest → smallest)
df = df.sort_values("count", ascending=False).reset_index(drop=True)

# 3. Recompute cumulative fraction of reads
total_reads = df["count"].sum()
df["cum_frac"] = df["count"].cumsum() / total_reads

# 4. Define x-axis as barcode rank (1 = highest read count)
df["rank"] = df.index + 1

# ---------- (Optional) find approximate knee point ----------
# Normalize rank and cum_frac to [0,1]
x = df["rank"].values.astype(float)
y = df["cum_frac"].values.astype(float)

x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Line between first and last point
p1 = np.array([x_norm[0], y_norm[0]])
p2 = np.array([x_norm[-1], y_norm[-1]])
line_vec = p2 - p1
line_vec_norm = line_vec / np.linalg.norm(line_vec)

# Distance of each point to the straight line
points = np.vstack([x_norm, y_norm]).T
vec_from_p1 = points - p1
proj_lengths = np.dot(vec_from_p1, line_vec_norm)
proj_points = np.outer(proj_lengths, line_vec_norm) + p1
distances = np.linalg.norm(points - proj_points, axis=1)

knee_idx = np.argmax(distances)
knee_rank = df.loc[knee_idx, "rank"]
knee_cum = df.loc[knee_idx, "cum_frac"]
knee_count = df.loc[knee_idx, "count"]
knee_bc = df.loc[knee_idx, "bc1"] if "bc1" in df.columns else None

print(f"Approx knee at rank {knee_rank}, cum_frac={knee_cum:.3f}, "
      f"count={knee_count}, bc={knee_bc}")

# ----------------- PLOT KNEEPLOT -----------------
plt.figure(figsize=(7, 5))

# main curve
plt.plot(df["rank"], df["cum_frac"], marker=".", linestyle="-", linewidth=1)

# mark knee
plt.axvline(knee_rank, color="red", linestyle="--", alpha=0.7)
plt.scatter([knee_rank], [knee_cum], color="red", s=30, zorder=3)
plt.text(
    knee_rank,
    knee_cum,
    f"  knee ~ {knee_rank}",
    va="bottom",
    ha="left",
    fontsize=8
)

plt.xlabel("Barcode rank (reads high → low)")
plt.ylabel("Cumulative fraction of reads")

# Axis limits (only applied if not None)
if X_MIN is not None or X_MAX is not None:
    plt.xlim(X_MIN, X_MAX)
if Y_MIN is not None or Y_MAX is not None:
    plt.ylim(Y_MIN, Y_MAX)

# IMPORTANT: no log scale on x
# plt.xscale("log")  # <- removed per your request

plt.tight_layout()
plt.show()
