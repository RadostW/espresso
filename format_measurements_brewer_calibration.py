import pandas as pd
import pathlib

# Resolve paths relative to this script
here = pathlib.Path(__file__).resolve().parent
DATA_DIR = here / "measurements_brewer_calibration"
OUTPUT_CSV = here / "formatted_measurements" / "brewer_calibration.csv"

# Input files (located inside DATA_DIR)
input_files = [
    "low-flow-rate-2-800_interval_stats.csv",
    "low-flow-rate-500_interval_stats.csv",
    "low-flow-rate_interval_stats.csv",
    "przekroj-test-zawor_interval_stats.csv",
]

# Columns to keep in the final output
output_columns = [
    "p1_mean",
    "p1_std",
    "p2_mean",
    "p2_std",
    "delta_p_mean",
    "delta_p_std",
    "flow_rate_mean",
    "flow_rate_std",
    "measurement_series",
]

# Map for renaming columns with units
rename_map = {
    "p1_mean": "p1_mean__bar",
    "p1_std": "p1_std__bar",
    "p2_mean": "p2_mean__bar",
    "p2_std": "p2_std__bar",
    "delta_p_mean": "delta_p_mean__bar",
    "delta_p_std": "delta_p_std__bar",
    "flow_rate_mean": "flow_rate_mean__g_per_s",
    "flow_rate_std": "flow_rate_std__g_per_s",
}

dataframes = []

for filename in input_files:
    file_path = DATA_DIR / filename
    df = pd.read_csv(file_path)

    # Convert pressure columns from kPa to bar
    pressure_cols = [
        "p1_mean",
        "p1_std",
        "p2_mean",
        "p2_std",
        "delta_p_mean",
        "delta_p_std",
    ]
    df[pressure_cols] = df[pressure_cols] / 100.0

    # Add measurement series column
    df["measurement_series"] = filename

    # Keep only required columns
    df = df[output_columns]

    # Rename columns using the map
    df = df.rename(columns=rename_map)

    dataframes.append(df)

# Combine all data
combined_df = pd.concat(dataframes, ignore_index=True)

# Write output CSV with 4 decimal places
combined_df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")

print(f"Combined calibration file written to:\n{OUTPUT_CSV}")

