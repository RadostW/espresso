import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Resolve paths relative to this script
# -------------------------------------------------
here = pathlib.Path(__file__).resolve().parent

DATA_DIR = here / "formatted_measurements"
INPUT_CSV = DATA_DIR / "tds.csv"

OUTPUT_DIR = here / "fit_parameters"
OUTPUT_CSV = OUTPUT_DIR / "tds_calibration.csv"

# -------------------------------------------------
# Load data
# -------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

time_data = df["time__s"].to_numpy()
tds_data = df["tds__percent"].to_numpy()
tds_std_data = df["tds_std__percent"].to_numpy()

# =========================
# Fitting
# =========================
from scipy.optimize import curve_fit

def tds_teo(t, k, l, m):
    return 0.5 * k * (1 - np.tanh((t - l) / m))

# Initial guess
initial_guess = (30.0, 20.0, 10.0)

# Bounds
bounds = (
    (0, 0, 0),
    (100, 100, 500),
)

popt, pcov, infodict, errmsg, ier = curve_fit(
    tds_teo,
    time_data,
    tds_data,    
    p0=initial_guess,
    bounds=bounds,
    maxfev=20_000,
    full_output=True,
)

k_ref, l_ref, m_ref = popt
perr = np.sqrt(np.diag(pcov))

# =========================
# Save fit parameters
# =========================
fit_df = pd.DataFrame(
    {
        "parameter": ["k__percent", "l__s", "m__s"],
        "value": [k_ref, l_ref, m_ref],
        "std": [perr[0], perr[1], perr[2]],
        "model": ["0.5 * k * (1 - tanh((t - l) / m))"]*3
    }
)

fit_df.to_csv(OUTPUT_CSV, index=False)
print("Saved TDS calibration:")
print(fit_df)
print(f"→ {OUTPUT_CSV}")

# =========================
# Visualization (optional)
# =========================
if __name__ == "__main__":

    t_fit = np.linspace(time_data.min(), time_data.max(), 400)
    tds_fit = tds_teo(t_fit, k_ref, l_ref, m_ref)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        time_data,
        tds_data,        
        fmt="o",
        color="black",
        capsize=3,
        label="Measurement",
    )

    plt.plot(
        t_fit,
        tds_fit,
        lw=2,
        color="C0",
        label="Best-fit model",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("TDS (%)")
    plt.ylim([0, 30])
    plt.legend()
    plt.tight_layout()
    plt.show()
