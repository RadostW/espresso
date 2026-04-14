"""
Pressure, mass, and mass flow rate multipanel figure
from long-format measurement CSV.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import colormaps as cmaps

# =========================
# Paths
# =========================
HERE = pathlib.Path(__file__).resolve().parent
DATA_CSV = HERE / "formatted_measurements" / "time_dependent.csv"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"
OUTPUT_DIR = HERE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# Plot parameters
# =========================
TARGET_MASS = 5.0  # g
FIG_WIDTH, FIG_HEIGHT = 7, 2
FONT_SIZE = 9
SHADOW_ALPHA = 0.25

plt.style.use(STYLE_FILE)
plt.rcParams.update({"lines.marker": ""})  # no default markers

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_CSV)

reference_pressures = sorted(df["reference_pressure_round__bar"].unique())

# Color map
cmap = cmaps.haline
norm = mpl.colors.Normalize(
    vmin=min(reference_pressures),
    vmax=max(reference_pressures),
)
colors = {p: cmap(norm(p)) for p in reference_pressures}

# =========================
# Create figure
# =========================
fig, axs = plt.subplots(
    1,
    3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    width_ratios=(1, 2, 2),
    constrained_layout=True,
)

ax_pressure, ax_mass, ax_flow = axs

# =========================
# Plot data
# =========================
for p in reference_pressures:
    df_p = df[df["reference_pressure_round__bar"] == p].reset_index(drop=True)

    # Time shift at target mass
    idx_target = (df_p["mass__g"] > TARGET_MASS).idxmax()
    t_shift = df_p.loc[idx_target, "time__s"]

    time = df_p["time__s"]
    color = colors[p]

    # --- Panel A: Pressure ---
    mean_p = df_p["basket_pressure__bar"]
    std_p = df_p["basket_pressure_std__bar"]
    ax_pressure.plot(time, mean_p, color=color)
    ax_pressure.fill_between(time, mean_p - std_p, mean_p + std_p,
                             color=color, alpha=SHADOW_ALPHA)

    # --- Panel B: Mass ---
    mean_m = df_p["mass__g"]
    std_m = df_p["mass_std__g"]
    ax_mass.plot(time - t_shift, mean_m, color=color)
    ax_mass.fill_between(time - t_shift, mean_m - std_m, mean_m + std_m,
                         color=color, alpha=SHADOW_ALPHA)

    # --- Panel C: Mass flow rate ---
    mean_q = df_p["mass_flow_rate__g_per_s"]
    std_q = df_p["mass_flow_rate_std__g_per_s"]
    ax_flow.plot(time, mean_q, color=color)
    ax_flow.fill_between(time, mean_q - std_q, mean_q + std_q,
                         color=color, alpha=SHADOW_ALPHA)

# =========================
# Axis formatting
# =========================
ax_pressure.set_ylabel("Basket pressure [bar]")
ax_pressure.set_xlabel("Time [s]")
ax_pressure.axhline(0, color="k", linestyle=":", linewidth=1)

ax_mass.set_ylabel("Mass [g]")
ax_mass.set_xlabel(rf"Time relative to $m={TARGET_MASS:.0f}$ g [s]")
ax_mass.axvline(0, color="k", linestyle=":", linewidth=1)
ax_mass.axhline(0, color="k", linestyle=":", linewidth=1)

ax_flow.set_ylabel("Mass flow rate [g/s]")
ax_flow.set_xlabel("Time [s]")
ax_flow.axhline(0, color="k", linestyle=":", linewidth=1)

# =========================
# Panel labels
# =========================
panel_labels = [r"\textbf{(A)}", r"\textbf{(B)}", r"\textbf{(C)}"]
for ax, label in zip(axs, panel_labels):
    ax.text(
        0.05, 0.90, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
    )

# =========================
# Save figure
# =========================
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.02, hspace=0.02)
out_file = OUTPUT_DIR / "multipanel_flow_vs_time.pdf"

plt.savefig(out_file)
if __name__ == "__main__":
    plt.show()
plt.close(fig)

print(f"Saved {out_file}")
