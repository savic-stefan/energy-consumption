"""Load and validate the appliance energy time series."""
from pathlib import Path
import pandas as pd

TARGET = "Appliances"
TIME = "date"
NOISE_COLUMNS = {"rv1", "rv2"}
REQUIRED = {TIME, TARGET, "lights", "T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint"}


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, parse_dates=[TIME])
    validate_data(frame)
    frame = frame.sort_values(TIME).reset_index(drop=True)
    return frame


def validate_data(frame):
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame[TIME].isna().any() or frame[TARGET].isna().any():
        raise ValueError("Timestamp and target must not contain missing values")
    if frame[TIME].duplicated().any():
        raise ValueError("Timestamps must be unique")
    timestamps = pd.to_datetime(frame[TIME]).sort_values()
    intervals = timestamps.diff().dropna()
    if not intervals.eq(pd.Timedelta(minutes=10)).all():
        raise ValueError("Dataset must have a regular 10-minute interval")
    if (pd.to_numeric(frame[TARGET], errors="coerce") < 0).any():
        raise ValueError("Energy target cannot be negative")


def data_summary(frame):
    return {"rows": int(len(frame)), "columns": int(len(frame.columns)),
            "start": str(frame[TIME].min()), "end": str(frame[TIME].max()),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_timestamps": int(frame[TIME].duplicated().sum()),
            "target_mean_wh": float(frame[TARGET].mean()),
            "target_median_wh": float(frame[TARGET].median()),
            "target_max_wh": float(frame[TARGET].max()),
            "noise_columns_excluded": sorted(NOISE_COLUMNS)}
