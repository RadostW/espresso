import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
HERE = pathlib.Path(__file__).resolve().parent
TDS_DATA = HERE / "formatted_measurements" / "tds.csv"
TIME_DEPENDENT_DATA = HERE / "formatted_measurements" / "time_dependent.csv"

TDS_FIT_CSV = HERE / "fit_parameters" / "tds_calibration.csv"
SOLIDS_FIT_CSV = HERE / "fit_parameters" / "solids_calibration.csv"

STYLE_FILE = HERE / "styles" / "espresso.mplstyle"

OUTPUT_DIR = HERE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "multipanel_brewer_calibration.pdf"

# =========================
# Figure settings
# =========================
FIG_WIDTH = 7
FIG_HEIGHT = 2
FONT_SIZE = 9

plt.style.use(STYLE_FILE)
plt.rcParams.update({"lines.marker": ""})  # no default markers
SHADOW_ALPHA = 0.25


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

fit_tds_params = pd.read_csv(TDS_FIT_CSV).set_index("parameter")["value"]
k_tds = fit_tds_params["k__percent"]
l_tds = fit_tds_params["l__s"]
m_tds = fit_tds_params["m__s"]


fit_solids_params = pd.read_csv(SOLIDS_FIT_CSV).set_index("parameter")["value"]
k_solids = fit_solids_params["k_solids__g"]
l_solids = fit_solids_params["l_solids__s"]
m_solids = fit_solids_params["m_solids__s"]
first_drop_offset = fit_solids_params["first_drop_offset__s"]

df_tds = pd.read_csv(TDS_DATA)
df_time_dependent_all = pd.read_csv(TIME_DEPENDENT_DATA)
df_time_dependent = df_time_dependent_all[
    df_time_dependent_all["reference_pressure_round__bar"] == 9.0
].reset_index(drop=True)


# =========================
# Models
# =========================
def tds_teo(t, k_tds, l_tds, m_tds):
    return 0.5 * k_tds * (1 - np.tanh((t - l_tds) / m_tds))


def solids_teo(t, k_solids, l_solids, m_solids):
    return 0.5 * k_solids * (1 + np.tanh((t - l_solids) / m_solids))


# =========================
# Create figure
# =========================
fig, axs = plt.subplots(
    1,
    3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    width_ratios=(1, 1, 1),
    constrained_layout=True,
)

ax_tds, ax_flows, ax_removed = axs

# =
# Plot
# =

# tds_teo_time = np.linspace(df_tds["time__s"].min(), df_tds["time__s"].max(), 400)
tds_teo_time = np.linspace(-first_drop_offset, df_tds["time__s"].max(), 400)
tds_teo_value = tds_teo(tds_teo_time, k_tds, l_tds, m_tds)

solids_teo_time = np.linspace(
    df_time_dependent["time__s"].min(), df_time_dependent["time__s"].max(), 400
)
solids_teo_value = solids_teo(
    solids_teo_time - first_drop_offset, k_solids, l_solids, m_solids
)

df_time_dependent["tds__percent"] = tds_teo(
    df_time_dependent["time__s"] - first_drop_offset,
    k_tds,
    l_tds,
    m_tds,
)
df_time_dependent["solids_flow__g_per_s"] = (
    df_time_dependent["tds__percent"]
    / 100.0
    * df_time_dependent["mass_flow_rate__g_per_s"]
)
df_time_dependent["solids_flow_std__g_per_s"] = (
    df_time_dependent["tds__percent"]
    / 100.0
    * df_time_dependent["mass_flow_rate_std__g_per_s"]
)
dt = df_time_dependent["time__s"].iloc[1] - df_time_dependent["time__s"].iloc[0]
df_time_dependent["solids_removed__g"] = (
    df_time_dependent["solids_flow__g_per_s"].cumsum() * dt
)

ax_errorbar_black(
    ax_tds,
    df_tds["time__s"] + first_drop_offset,
    df_tds["tds__percent"],
    yerr=df_tds["tds_std__percent"],
    color="C1",
    marker="o",
)

ax_tds.plot(
    tds_teo_time + first_drop_offset,
    tds_teo_value,
    marker="",
    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)

time = df_time_dependent["time__s"]
mean_flow = df_time_dependent["mass_flow_rate__g_per_s"]
sigma_flow = df_time_dependent["mass_flow_rate_std__g_per_s"]

mean_flow_solids = df_time_dependent["solids_flow__g_per_s"]
sigma_flow_solids = df_time_dependent["solids_flow_std__g_per_s"]

ax_flows.plot(time, mean_flow, color="C0", label='total')
ax_flows.fill_between(
    time, mean_flow - sigma_flow, mean_flow + sigma_flow, color="C0", alpha=SHADOW_ALPHA
)

ax_flows.plot(time, mean_flow_solids, color="C1", label='solutes')
ax_flows.fill_between(
    time,
    mean_flow_solids - sigma_flow_solids,
    mean_flow_solids + sigma_flow_solids,
    color="C1",
    alpha=SHADOW_ALPHA,
)


ax_removed.plot(
    solids_teo_time,
    solids_teo_value,
    marker="",
    linestyle="-",
    dashes=[3, 0.3],
    color="k",
)
ax_removed.plot(time, df_time_dependent["solids_removed__g"], color="C1")

# =========================
# Axis formatting
# =========================
ax_tds.set_xlim([0,df_tds["time__s"].max()])
ax_tds.set_ylabel(r"TDS [\%]")
ax_tds.set_xlabel("Time [s]")

ax_flows.set_ylabel("Flow rate [g/s]")
ax_flows.set_xlabel("Time [s]")
ax_flows.legend()

ax_removed.set_ylabel("Removed mass [g]")
ax_removed.set_xlabel("Time [s]")

# =========================
# Panel labels
# =========================
panel_labels = ["B", "C", "D"]
for ax, label in zip(axs, panel_labels):
    ax.text(
        0.05, 0.90, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
    )

fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.02, hspace=0.02)
out_file = OUTPUT_DIR / "multipanel_tds.pdf"

plt.savefig(out_file)
if __name__ == "__main__":
    plt.show()
plt.close(fig)

print(f"Saved {out_file}")

