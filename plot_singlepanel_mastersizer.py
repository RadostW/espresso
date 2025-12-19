import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# =========================
# Paths
# =========================

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "measurements_mastersizer" / "mastersizer.csv"
OUTPUT_FILE = HERE / "figures" / "abundance_vs_size.pdf"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"

# =========================
# Figure and plot settings
# =========================

FIGURE_WIDTH = 3.3
FIGURE_HEIGHT = 2
FONT_SIZE = 9

plt.style.use(STYLE_FILE)

# =========================
# Load and clean data
# =========================

df = pd.read_csv(DATA_FILE, sep=";", header=None)
df = df.apply(pd.to_numeric, errors="coerce")

# Row 0: bin edges
bin_edges = df.iloc[0].to_numpy()

# Rows 1–3: replicates
replicates = df.iloc[1:4].to_numpy()
n_replicates = replicates.shape[0]

# =========================
# Statistics & normalization
# =========================

mean_value = np.nanmean(replicates, axis=0)
std_value = np.nanstd(replicates, axis=0)

max_val = np.nanmax(mean_value)
if max_val > 0:
    mean_value /= max_val
    std_value /= max_val

stderr_value = std_value / np.sqrt(n_replicates)

# =========================
# Plot
# =========================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    constrained_layout=True,
)

ax.bar(
    bin_edges[1:],    
    mean_value[1:],
    width=bin_edges[1:]-bin_edges[:-1],
    yerr=stderr_value[1:],    
    ls='',
    color='#ccc'
)

ax.set_xscale("log")
ax.set_xlim(5, 500)
ax.set_ylim(-0.01, 1.08)

ax.set_xlabel(r"Grind size [$\mu$m]")
ax.set_ylabel("Volume abundance [a.u.]")

# =========================
# Save
# =========================

plt.savefig(OUTPUT_FILE)
if __name__ == "__main__":
    plt.show()

print(f"Saved {OUTPUT_FILE}")
