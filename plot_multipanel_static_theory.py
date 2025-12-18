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

# =========================
# Figure settings
# =========================
FIG_WIDTH = 7
FIG_HEIGHT = 2

FONT_SIZE = 9

plt.style.use(STYLE_FILE)
plt.rcParams.update({"lines.marker": ""})  # no default markers

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
    kwargs.setdefault("marker","o")
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

fig, axs = plt.subplots(
    1,
    2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    width_ratios=(1, 1),
    constrained_layout=True,
)

ax_limit, ax_comparison = axs

# Experimental
ax_errorbar_black(
    ax_comparison,
    df["basket_pressure__bar"],
    df["mass_flow_rate__g_per_s"],
    yerr=df["mass_flow_rate_std__g_per_s"],
)
# Theoretical

p_fit = np.linspace(0, 15, 300)
q_fit = q_teo(p_fit, pref, qref)

ax_comparison.plot(
    p_fit,
    q_fit,
    marker="",

    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)
ax_comparison.set_xlim([0,13])
ax_comparison.set_ylim([0,2.2])

ax_comparison.set_xlabel("Basket pressure [bar]")
ax_comparison.set_ylabel("Long run flow [g/s]")


# =========================
# theoretical functions
# =========================


def limit_teo(p):
    return p * (4 - 6 * p + 4 * p**2 - p**3)


def full_teo(p, f):
    """
    Compute Q_hat given p and f.

    Note:
        Variable names are changed from the original LaTeX:
        - p corresponds to \hat{P}
        - f corresponds to \Phi
    """
    numerator = 6 * (f - 1) ** 3 * np.log(1 - p * f) - p * f * (
        (p * (2 * p - 9) + 18) * f**2 + 3 * (p - 6) * f + 6
    )

    denominator = f * ((15 - 11 * f) * f - 6) + 6 * (f - 1) ** 3 * np.log(1 - f)

    return numerator / denominator


# =========================
# Plot
# =========================

p_vals = np.linspace(0, 1, 100)
ax_limit.plot(p_vals, full_teo(p_vals, f=0.5), label=r"$\Phi = 0.5$")
ax_limit.plot(p_vals, full_teo(p_vals, f=0.2), label=r"$\Phi = 0.2$")
ax_limit.plot(p_vals, full_teo(p_vals, f=0.1), label=r"$\Phi = 0.1$")

ax_limit.plot(
    p_vals,
    limit_teo(p_vals),
    label=r"$\Phi \to 0$",
    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)


ax_limit.set_xlim(0, 1)
ax_limit.set_ylim(0, 1.16)

ax_limit.set_xlabel(r"$\hat{P}$")
ax_limit.set_ylabel(r"$\hat{Q}$")

ax_limit.legend()

# =========================
# Panel labels
# =========================
panel_labels = ["A", "B"]
for ax, label in zip(axs, panel_labels):
    ax.text(
        0.05, 0.90, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
    )

fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.02, hspace=0.02)

# Save figure
output_file = OUTPUT_DIR / "multipanel_static_theory.pdf"
plt.savefig(output_file)

if __name__ == "__main__":
    plt.show()

print(f"Saved {output_file}")
