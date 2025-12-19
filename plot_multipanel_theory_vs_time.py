"""
Pressure, mass, and mass flow rate multipanel figure
from long-format measurement CSV together with theoretical predictions
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

SOLIDS_FIT_CSV = HERE / "fit_parameters" / "solids_calibration.csv"
STATIC_FIT_CSV = HERE / "fit_parameters" / "static_model_calibration.csv"

CONSTANTS_CSV = HERE / "constant_parameters" / "constants.csv"

STYLE_FILE = HERE / "styles" / "espresso.mplstyle"
OUTPUT_DIR = HERE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# TAKE_PRESSURES = [1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0]
TAKE_PRESSURES = [
    # 1.0,
    2.0,
    # 3.5,
    4.0,
    # 5.0, 6.0, 7.0,
    8.0,
    9.0,
    # 11.0,
    13.0,
]

# =========================
# Plot parameters
# =========================
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

fit_params = pd.read_csv(STATIC_FIT_CSV).set_index("parameter")["value"]
final_p_ref, final_q_ref = fit_params["p_ref__bar"], fit_params["q_ref__g_per_s"]

fit_params = pd.read_csv(SOLIDS_FIT_CSV).set_index("parameter")["value"]
k_solids, l_solids, m_solids = (
    fit_params["k_solids__g"],
    fit_params["l_solids__s"],
    fit_params["m_solids__s"],
)

constant_params = pd.read_csv(CONSTANTS_CSV).set_index("parameter")["value"]
dose = constant_params["dose__g"]

final_dissolved = k_solids
final_phi = k_solids / dose

# =========================
# Theoretical
# =========================


def phi_factor(phi):
    return (
        phi * (phi * (11 * phi - 15) + 6) - 6 * (phi - 1) ** 3 * np.log(1 - phi)
    ) / (6 * (phi - 1) ** 2)


def solids_teo(t, k_solids, l_solids, m_solids):
    return 0.5 * k_solids * (1 + np.tanh((t - l_solids) / m_solids))


def q_hat(p_hat):
    return p_hat * (4 - 6 * p_hat + 4 * p_hat**2 - p_hat**3)


final_phi_factor = phi_factor(final_phi)
q_master = final_q_ref / final_phi_factor
p_master = final_p_ref / final_phi

teo_time = np.array(sorted(df["time__s"].unique()))
solids_td = solids_teo(teo_time, k_solids, l_solids, m_solids)
phi_td = solids_td / dose

q_ref_td = q_master * phi_factor(phi_td)
p_ref_td = p_master * phi_td

# =========================
# Create figure
# =========================

# Color map
cmap = cmaps.haline
norm = mpl.colors.Normalize(
    vmin=min(reference_pressures),
    vmax=max(reference_pressures),
)
colors = {p: cmap(norm(p)) for p in reference_pressures}


fig, axs = plt.subplots(
    1,
    2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    width_ratios=(1, 1),
    constrained_layout=True,
)

ax_single, ax_flow = axs

# =========================
# Plot data
# =========================
for p in reference_pressures:
    if p not in TAKE_PRESSURES:
        continue
    df_p = df[df["reference_pressure_round__bar"] == p].reset_index(drop=True)

    exp_time = df_p["time__s"]
    color = colors[p]

    p_hat_td = p / p_ref_td
    q_hat_td = q_hat(p_hat_td)
    q_td = np.clip(q_hat_td * q_ref_td, a_min=0, a_max=None)

    # --- Panel C: many lines ---
    mean_q = df_p["mass_flow_rate__g_per_s"]
    std_q = df_p["mass_flow_rate_std__g_per_s"]
    ax_flow.plot(exp_time, mean_q, color=color)
    # ax_flow.fill_between(time, mean_q - std_q, mean_q + std_q,
    #                      color=color, alpha=SHADOW_ALPHA)
    ax_flow.plot(
        teo_time,
        q_td,
        color=color,
        marker="",
        linestyle="-",
        dashes=[3, 1],
    )

    # --- Panel B: single line ---
    if p == 9.0:
        ax_single.plot(exp_time, mean_q, color=color)
        ax_single.fill_between(
            exp_time, mean_q - std_q, mean_q + std_q, color=color, alpha=SHADOW_ALPHA
        )

        ax_single.plot(
            teo_time,
            q_td,
            # color=color,
            linestyle="-",
            dashes=[3, 1],
            color="k",
        )

# =========================
# Axis formatting
# =========================

ax_single.set_ylabel("Mass flow rate [g/s]")
ax_single.set_xlabel("Time [s]")
ax_single.axhline(0, color="k", linestyle=":", linewidth=1)

ax_flow.set_ylabel("Mass flow rate [g/s]")
ax_flow.set_xlabel("Time [s]")
ax_flow.axhline(0, color="k", linestyle=":", linewidth=1)

# =========================
# Panel labels
# =========================
panel_labels = [r"\textbf{(A)}", r"\textbf{(B)}"]
for ax, label in zip(axs, panel_labels):
    ax.text(
        0.05,
        0.90,
        label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
    )

# =========================
# Save figure
# =========================
fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.02, hspace=0.02)
out_file = OUTPUT_DIR / "multipanel_theory_vs_time.pdf"

plt.savefig(out_file)
if __name__ == "__main__":
    plt.show()
plt.close(fig)

print(f"Saved {out_file}")
