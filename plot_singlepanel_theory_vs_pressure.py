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
FIT_CSV = HERE / "fit_parameters" / "static_model_calibration.csv"
STYLE_FILE = HERE / "styles" / "espresso.mplstyle"
OUTPUT_DIR = HERE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# Figure and plot settings
# =========================

FIGURE_WIDTH = 3.3
FIGURE_HEIGHT = 2

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
df_time_dependent = pd.read_csv(DATA_CSV)

# Use last value per pressure series (steady-state)
df = df_time_dependent.groupby("reference_pressure_round__bar").last()

fit_params = pd.read_csv(FIT_CSV).set_index("parameter")["value"]
pref, qref = fit_params["p_ref__bar"], fit_params["q_ref__g_per_s"]


# =========================
# Theory
# =========================
def qhat_teo(phat):
    return phat * (4 - 6 * phat + 4 * phat**2 - phat**3)


def q_teo(p, pref, qref):
    return qref * qhat_teo(p / pref)


# =========================
# Plot
# =========================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), constrained_layout=True)

# Experimental
ax_errorbar_black(
    ax,
    df["basket_pressure__bar"],
    df["mass_flow_rate__g_per_s"],
    yerr=df["mass_flow_rate_std__g_per_s"],
)
# Theoretical

p_fit = np.linspace(0, 15, 300)
q_fit = q_teo(p_fit, pref, qref)

ax.plot(
    p_fit,
    q_fit,
    marker="",

    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)
ax.set_xlim([0,13])
ax.set_ylim([0,2.2])

ax.set_xlabel("Basket pressure [bar]")
ax.set_ylabel("Long run flow [g/s]")

# Save figure
output_file = OUTPUT_DIR / "final_q_vs_final_p.pdf"
plt.savefig(output_file)

if __name__ == "__main__":
    plt.show()

print(f"Saved {output_file}")
