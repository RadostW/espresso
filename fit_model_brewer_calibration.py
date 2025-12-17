import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Resolve paths relative to this script
# -------------------------------------------------
here = pathlib.Path(__file__).resolve().parent

DATA_DIR = here / "formatted_measurements"
INPUT_CSV = DATA_DIR / "brewer_calibration.csv"

OUTPUT_DIR = here / "fit_parameters"
OUTPUT_CSV = OUTPUT_DIR / "brewer_calibration.csv"

# -------------------------------------------------
# Load data
# -------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

flow_rate = df["flow_rate_mean__g_per_s"].to_numpy()
delta_p = df["delta_p_mean__bar"].to_numpy()

# -------------------------------------------------
# Quadratic fit: delta_p = a*Q^2 + b*Q + c
# -------------------------------------------------
coeffs = np.polyfit(flow_rate, delta_p, deg=2)
a, b, c = coeffs

# -------------------------------------------------
# Save fit parameters
# -------------------------------------------------
fit_df = pd.DataFrame(
    {
        "parameter": ["a", "b", "c"],
        "value": [a, b, c],
        "model": ["delta_p = a*Q^2 + b*Q + c"] * 3,
    }
)

fit_df.to_csv(OUTPUT_CSV, index=False)

print("Quadratic fit completed.")
print(f"a = {a:.6e}")
print(f"b = {b:.6e}")
print(f"c = {c:.6e}")
print(f"Saved fit parameters to: {OUTPUT_CSV}")

# -------------------------------------------------
# Visualization only when run as script
# -------------------------------------------------
if __name__ == "__main__":
    Q_fit = np.linspace(flow_rate.min(), flow_rate.max(), 300)
    dp_fit = np.polyval(coeffs, Q_fit)

    plt.figure(figsize=(7, 5))
    plt.scatter(flow_rate, delta_p, label="Measured data")
    plt.plot(Q_fit, dp_fit, label="Quadratic fit")

    plt.xlabel("Flow rate (mean)")
    plt.ylabel("Δp (mean)")
    plt.title("Quadratic fit of Δp vs Flow rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

