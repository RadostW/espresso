"""
Static porous-flow model calibration.

Fits Y (bar), dp (micron), and phi to steady-state
pressure–flow measurements using reparameterization for Y.
"""

import pathlib
import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
HERE = pathlib.Path(__file__).resolve().parent
DATA_CSV = HERE / "formatted_measurements" / "time_dependent.csv"

OUTPUT_DIR = HERE / "fit_parameters"
OUTPUT_CSV = OUTPUT_DIR / "static_model_calibration.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load data
# =========================
df_time_dependent = pd.read_csv(DATA_CSV)

# Use last value per pressure series (steady-state)
df = df_time_dependent.groupby("reference_pressure_round__bar").last()

# =========================
# Theory
# =========================
def qhat_teo(phat):
    return phat * (4 - 6 * phat + 4 * phat**2 - phat**3)

def q_teo(p, pref, qref):
    return qref * qhat_teo(p / pref)

# =========================
# Fitting
# =========================
from scipy.optimize import curve_fit

p_data = df["basket_pressure__bar"].values
q_data = df["mass_flow_rate__g_per_s"].values
q_std_data = df["mass_flow_rate_std__g_per_s"].values

# Initial guess
initial_guess = (20.0, 1.0)

# Bounds
bounds = (
    (np.max(p_data), 0.01),
    (100 * np.max(p_data), 100),
)

popt, pcov, infodict, errmsg, ier = curve_fit(
    q_teo,
    p_data,
    q_data,
    p0=initial_guess,
    bounds=bounds,
    maxfev=20_000,
    full_output=True,
)

pref, qref = popt
perr = np.sqrt(np.diag(pcov))

# =========================
# Save fit parameters
# =========================
fit_df = pd.DataFrame(
    {
        "parameter": ["p_ref__bar", "q_ref__g_per_s"],
        "value": [pref, qref],
        "std": [perr[0], perr[1]],
    }
)

fit_df.to_csv(OUTPUT_CSV, index=False)
print("Saved static model calibration:")
print(fit_df)
print(f"→ {OUTPUT_CSV}")

# =========================
# Visualization (optional)
# =========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pbar_fit = np.linspace(0, 15, 300)
    q_fit = q_teo(pbar_fit, pref, qref)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        p_data,
        q_data,
        yerr=q_std_data,
        fmt="o",
        color="black",
        capsize=3,
        label="Measurement",
    )

    plt.plot(
        pbar_fit,
        q_fit,
        lw=2,
        color="C0",
        label="Best-fit theory",
    )

    plt.xlabel("Basket pressure [bar]")
    plt.ylabel("Mass flow rate [g/s]")
    plt.xlim(0, 15)
    plt.ylim(0, 3)
    plt.legend()
    plt.tight_layout()
    plt.show()
