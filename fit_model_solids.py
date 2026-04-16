import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# Paths
# =========================
HERE = pathlib.Path(__file__).resolve().parent
DATA_CSV = HERE / "formatted_measurements" / "time_dependent.csv"
FIT_CSV = HERE / "fit_parameters" / "tds_calibration.csv"

OUTPUT_DIR = HERE / "fit_parameters"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "solids_calibration.csv"

FIRST_DROP_OFFSET = 8.0  # seconds

# =========================
# Load data
# =========================
df_all = pd.read_csv(DATA_CSV)
df = df_all[df_all["reference_pressure_round__bar"] == 9.0].reset_index(drop=True)

fit_params = pd.read_csv(FIT_CSV).set_index("parameter")["value"]
k_tds = fit_params["k__percent"]
l_tds = fit_params["l__s"]
m_tds = fit_params["m__s"]

# =========================
# Models (unchanged)
# =========================
def tds_teo(t, k, l, m):
    return 0.5 * k * (1 - np.tanh((t - l) / m))

def solids_teo(t, k_solids, l_solids, m_solids):
    return 0.5 * k_solids * (1 + np.tanh((t - l_solids) / m_solids))

# =========================
# Construct solids removed
# =========================
df["tds__percent"] = tds_teo(
    df["time__s"] - FIRST_DROP_OFFSET,
    k_tds,
    l_tds,
    m_tds,
)

df["solids_flow__g_per_s"] = (
    df["tds__percent"] / 100.0 * df["mass_flow_rate__g_per_s"]
)

df["solids_flow_std__g_per_s"] = (
    df["tds__percent"] / 100.0 * df["mass_flow_rate_std__g_per_s"]
)

dt = df["time__s"].iloc[1] - df["time__s"].iloc[0]
df["solids_removed__g"] = df["solids_flow__g_per_s"].cumsum() * dt
df["solids_removed_std__g"] = (df["solids_flow_std__g_per_s"].cumsum() * dt)*(np.array(range(len(df)))**(-0.5))

# =========================
# Fit solids_teo directly
# =========================
t_data = df["time__s"].values - FIRST_DROP_OFFSET
s_data = df["solids_removed__g"].values

initial_guess = (
    3.0,     # k_solids [g]
    l_tds,   # l_solids [s]
    m_tds,   # m_solids [s]
)

bounds = (
    (0.0, 0.0, 0.1),
    (20.0, 50.0, 20.0),
)

popt, pcov = curve_fit(
    solids_teo,
    t_data,
    s_data,
    p0=initial_guess,
    bounds=bounds,
    maxfev=20_000,
)

k_solids, l_solids, m_solids = popt
perr = np.sqrt(np.diag(pcov))

for name, val, err in zip(
    ["k_solids__g", "l_solids__s", "m_solids__s"],
    popt,
    perr,
):
    print(f"{name:>14} = {val:10.4f} ± {err:10.4f}")

# =========================
# Save fit parameters
# =========================
fit_df = pd.DataFrame(
    {
        "parameter": [
            "k_solids__g",
            "l_solids__s",
            "m_solids__s",
            "first_drop_offset__s",
            "phi_m",
        ],
        "value": [
            k_solids,
            l_solids,
            m_solids,
            FIRST_DROP_OFFSET,
            k_solids / 18.5,
        ],
        "std": [
            perr[0],
            perr[1],
            perr[2],
            0,
            perr[0] / 18.5,
        ],
        "model": ["0.5 * k * (1 + tanh((t - l) / m))"] * 5,
    }
)

fit_df.to_csv(OUTPUT_FILE, index=False)
print("Saved solids calibration:")
print(fit_df)
print(f"→ {OUTPUT_FILE}")

# =========================
# Visualization
# =========================
if __name__ == "__main__":

    t_fit = np.linspace(t_data.min(), t_data.max(), 400)
    s_fit = solids_teo(t_fit, k_solids, l_solids, m_solids)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        t_data,
        s_data,
        yerr=df['solids_removed_std__g'],
        fmt="o",
        color="black",
        capsize=3,
        label="Measurement",
    )

    plt.plot(
        t_fit,
        s_fit,
        lw=2,
        color="C0",
        label="Best-fit model",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Solids removed [g]")
    plt.legend()
    plt.tight_layout()
    plt.show()
