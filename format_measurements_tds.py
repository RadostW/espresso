import pandas as pd
import numpy as np
import pathlib

# Resolve paths relative to this script
here = pathlib.Path(__file__).resolve().parent
TDS_CSV = here / "measurements_tds_calibration" / "tds.csv"
OUTPUT_CSV = here / "formatted_measurements" / "tds.csv"

tds_df = pd.read_csv(TDS_CSV)

# compute time-dependent mean and SEM across replicate columns
rep_cols = tds_df.filter(like="tds_percent")

tds_df["tds__percent"] = rep_cols.mean(axis=1, skipna=True)
tds_df["tds_std__percent"] = rep_cols.sem(axis=1, skipna=True)
tds_df["time__s"] = tds_df["time_s"]

# optional: keep only summary
summary_df = tds_df[["time__s","tds__percent","tds_std__percent"]]

# Write output CSV with 4 decimal places
summary_df.to_csv(OUTPUT_CSV, index=False, float_format="%.4f")
