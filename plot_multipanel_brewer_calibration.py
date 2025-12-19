import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
HERE = pathlib.Path(__file__).resolve().parent
DATA_CSV = HERE / "formatted_measurements" / "brewer_calibration.csv"
FIT_CSV = HERE / "fit_parameters" / "brewer_calibration.csv"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"
OUTPUT_DIR = HERE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "multipanel_brewer_calibration.pdf"


# =========================
# Figure settings
# =========================
FIG_WIDTH = 2
FIG_HEIGHT = 2
FONT_SIZE = 9

plt.style.use(STYLE_FILE)

# =========================
# Helper function for consistent error bars
# =========================
def ax_errorbar_black(ax, *args, **kwargs):
    """
    Wrapper around ax.errorbar that defaults ecolor to black,
    sets thin vertical lines, caps, and no connecting lines.
    """
    kwargs.setdefault("ecolor", "black")  # error bars color
    kwargs.setdefault("elinewidth", 0.8)  # vertical line width
    kwargs.setdefault("ls", "")  # no connecting line
    return ax.errorbar(*args, **kwargs)


# =========================
# Load data
# =========================
df = pd.read_csv(DATA_CSV)
fit_params = pd.read_csv(FIT_CSV).set_index("parameter")["value"]
a, b, c = fit_params["a"], fit_params["b"], fit_params["c"]

# =========================
# Create figure with 2 panels
# =========================
fig, axs = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
ax_quadratic = axs

# -------------------------
# Panel 2: delta_p vs flow rate with quadratic fit
# -------------------------
colors = plt.cm.tab10.colors
measurement_series_list = df["measurement_series"].unique()

for i, series in enumerate(measurement_series_list):
    df_series = df[df["measurement_series"] == series]
    ax_errorbar_black(
        ax_quadratic,
        df_series["flow_rate_mean__g_per_s"],
        df_series["delta_p_mean__bar"],
        yerr=df_series["delta_p_std__bar"],
        color=colors[i % len(colors)],
    )

# Quadratic fit line
flow_fit = np.linspace(
    df["flow_rate_mean__g_per_s"].min(), df["flow_rate_mean__g_per_s"].max(), 300
)
delta_p_fit = a * flow_fit**2 + b * flow_fit + c
ax_quadratic.plot(
    flow_fit,
    delta_p_fit,
    marker="",
    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)

ax_quadratic.set_xlabel("Flow rate [g/s]")
ax_quadratic.set_xticks([0, 2, 4, 6, 8, 10])
ax_quadratic.set_xlim([0, 7.2])

ax_quadratic.set_ylabel("Brewer pressure drop [bar]")
ax_quadratic.set_ylim([0, 1.6])

# -------------------------
# Add panel labels
# -------------------------
panel_labels = [r"\textbf{(B)}"]
panel_label_x, panel_label_y = 0.05, 0.90

ax_quadratic.text(
    panel_label_x,
    panel_label_y,
    panel_labels[0],
    transform=ax_quadratic.transAxes,
    fontsize=FONT_SIZE,
    fontweight="bold",
)

# =========================
# Save figure
# =========================
plt.savefig(OUTPUT_FILE)

if __name__ == "__main__":
    plt.show()

plt.close(fig)
print(f"Saved figure to {OUTPUT_FILE}")
