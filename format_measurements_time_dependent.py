"""
Import coffee extraction data and export to a single
long-format CSV for analysis and debugging.
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# =========================
# Processing parameters
# =========================

MAX_TIME = 100  # seconds
SAVGOL_WINDOW = 31
SAVGOL_POLYORDER = 1

# =========================
# Paths (relative to this file)
# =========================

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "measurements_time_dependent"
DEBUG_OUTPUT_CSV = HERE / "formatted_measurements" / "debug_time_dependent.csv"
OUTPUT_CSV = HERE / "formatted_measurements" / "time_dependent.csv"
FIT_CSV = HERE / "fit_parameters" / "brewer_calibration.csv"

# =========================
# Load and process data
# =========================
fit_params = pd.read_csv(FIT_CSV).set_index("parameter")["value"]
brewer_a, brewer_b, brewer_c = fit_params["a"], fit_params["b"], fit_params["c"]


rows = []

files = sorted(DATA_DIR.glob("*.txt"))
print(f"Found {len(files)} files in {DATA_DIR}")

for i, file in enumerate(files, start=1):
    filename = file.name

    # Parse pressure from filename (e.g. "9.0-run3.txt")
    try:
        pressure = float(filename.split("-")[0])
    except ValueError:
        print(f"Skipping file with invalid pressure: {filename}")
        continue

    print(f"Processing {i}/{len(files)}: {filename} (pressure={pressure})")

    # Load JSON-lines data
    df = pd.read_json(file, lines=True)

    reference_pressure = df["p2"].median()
    reference_pressure_round = np.round(reference_pressure / 100 / 0.5) * 0.5

    # small manual fixes
    if filename == "1-2.txt":
        reference_pressure_round = 1.0

    if reference_pressure_round == 1.5:
        reference_pressure_round = 2.0        

    print(f"reference_pressure = {reference_pressure_round}")

    # Boolean mask
    absolute_tolerance = 50  # kPa
    max_allowed_pressure = min(
        reference_pressure * 1.1, reference_pressure + absolute_tolerance
    )
    min_allowed_pressure = max(
        reference_pressure * 0.9, reference_pressure - absolute_tolerance
    )

    mask = (df["p2"] > max_allowed_pressure) | (df["p2"] < min_allowed_pressure)

    # Get last index where mask is True
    if mask.any():
        last_idx = mask[
            500::-1
        ].idxmax()  # reverses mask to find last True (search up to 500)
        print("Last index:", last_idx)
    else:
        last_idx = None
        print("No entries exceed tolerance.")
        raise ValueError("unable to align time")

    START_IDX = last_idx + 1

    # Time alignment (ms → s)
    time_adjusted = df["t"] - df["t"].iloc[START_IDX]
    mask = time_adjusted <= MAX_TIME * 1000

    t = 0.001 * time_adjusted[START_IDX:][mask[START_IDX:]]
    p = df["p2"][START_IDX:][mask[START_IDX:]] / 100  # kPa → bar
    w = df["m"][START_IDX:][mask[START_IDX:]]

    # Compute mass flow rate
    dwdt = np.gradient(w, t)
    dwdt = savgol_filter(
        dwdt,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
    )

    # Store in long format
    for ti, pi, wi, qi in zip(t, p, w, dwdt):

        pb = pi - (brewer_a * (qi**2) + brewer_b * qi + brewer_c)

        rows.append(
            {
                "pressure_series__bar": pressure,
                "reference_pressure_round__bar": reference_pressure_round,
                "filename": filename,
                "time__s": ti,
                "pressure__bar": pi,
                "mass__g": wi,
                "mass_flow_rate__g_per_s": qi,
                "basket_pressure__bar": pb,
            }
        )

# ===========================
# Export - debug (no groups)
# ===========================

out_df = pd.DataFrame(rows)
out_df = out_df.sort_values(by=["pressure_series__bar", "filename", "time__s"])
out_df.to_csv(DEBUG_OUTPUT_CSV, index=False, float_format="%.4f")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    i = 0
    for series_name, series_df in out_df.groupby("reference_pressure_round__bar"):
        for measurement_name, measurement_df in series_df.groupby("filename"):
            if i < 5:
                continue
            plt.plot(
                measurement_df["time__s"], measurement_df["pressure__bar"], f"C{i}"
            )
        i = i + 1
    plt.show()

    i = 0
    for series_name, series_df in out_df.groupby("reference_pressure_round__bar"):
        for measurement_name, measurement_df in series_df.groupby("filename"):
            if i < 5:
                continue
            plt.plot(
                measurement_df["time__s"],
                measurement_df["mass_flow_rate__g_per_s"],
                f"C{i}",
            )
        i = i + 1
    plt.show()


print(f"\nExport complete: {DEBUG_OUTPUT_CSV}")
print(f"Total rows written: {len(out_df)}")

# ===========================
# Interpolation to common time
# ===========================

COMMON_TIME = np.linspace(0.0, MAX_TIME, 1000)


def interpolate_measurement(df, common_time):
    """Interpolate a single measurement to a common time grid."""
    df = df.sort_values("time__s")

    return pd.DataFrame(
        {
            "time__s": common_time,
            "pressure__bar": np.interp(common_time, df["time__s"], df["pressure__bar"]),
            "basket_pressure__bar": np.interp(
                common_time, df["time__s"], df["basket_pressure__bar"]
            ),
            "mass__g": np.interp(common_time, df["time__s"], df["mass__g"]),
            "mass_flow_rate__g_per_s": np.interp(
                common_time, df["time__s"], df["mass_flow_rate__g_per_s"]
            ),
        }
    )


# Interpolate each measurement independently
interp_rows = []

for (reference_pressure_round, filename), g in out_df.groupby(
    ["reference_pressure_round__bar", "filename"]
):
    interp_df = interpolate_measurement(g, COMMON_TIME)
    interp_df["reference_pressure_round__bar"] = reference_pressure_round
    interp_df["filename"] = filename

    interp_rows.append(interp_df)

interp_df = pd.concat(interp_rows, ignore_index=True)

# ===========================
# Group statistics by pressure
# ===========================

agg_df = (
    interp_df.groupby(["reference_pressure_round__bar", "time__s"])
    .agg(
        pressure__bar=("pressure__bar", "mean"),
        pressure_std__bar=("pressure__bar", "sem"),
        basket_pressure__bar=("basket_pressure__bar", "mean"),
        basket_pressure_std__bar=("basket_pressure__bar", "sem"),
        mass__g=("mass__g", "mean"),
        mass_std__g=("mass__g", "sem"),
        mass_flow_rate__g_per_s=("mass_flow_rate__g_per_s", "mean"),
        mass_flow_rate_std__g_per_s=("mass_flow_rate__g_per_s", "sem"),
    )
    .reset_index()
)

# Final export
agg_df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

print(f"Aggregated data written to: {OUTPUT_CSV}")
print(f"Total aggregated rows: {len(agg_df)}")


# ===========================
# Optional diagnostic plots
# ===========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for pressure, g in agg_df.groupby("reference_pressure_round__bar"):
        plt.plot(g["time__s"], g["pressure__bar"])

    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [bar]")
    plt.show()

    for pressure, g in agg_df.groupby("reference_pressure_round__bar"):
        plt.plot(g["time__s"], g["mass_flow_rate__g_per_s"])

    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [bar]")
    plt.show()
