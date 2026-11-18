"""Causal calendar, sensor and demand-history features."""
import numpy as np
import pandas as pd
from energy_forecast.data import NOISE_COLUMNS, TARGET, TIME

HORIZON_STEPS = 6
LAGS = (0, 1, 3, 6, 12, 144)


def make_supervised(frame, horizon_steps=HORIZON_STEPS, require_target=True):
    """Use measurements available at time t to forecast demand at t+horizon."""
    data = frame.copy().sort_values(TIME).reset_index(drop=True)
    timestamp = pd.to_datetime(data[TIME])
    out = pd.DataFrame(index=data.index)
    out["target_time"] = timestamp.shift(-horizon_steps).fillna(
        timestamp + pd.Timedelta(minutes=10 * horizon_steps)
    )
    out["target"] = pd.to_numeric(data[TARGET], errors="coerce").shift(-horizon_steps)

    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    weekday = timestamp.dt.day_of_week
    year_day = timestamp.dt.day_of_year
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["year_day_sin"] = np.sin(2 * np.pi * year_day / 366)
    out["year_day_cos"] = np.cos(2 * np.pi * year_day / 366)
    out["is_weekend"] = (weekday >= 5).astype(int)

    excluded = {TIME, TARGET, *NOISE_COLUMNS}
    sensor_columns = [column for column in data.select_dtypes(include="number").columns if column not in excluded]
    out[sensor_columns] = data[sensor_columns].astype(float)
    for lag in LAGS:
        out[f"appliances_lag_{lag}"] = pd.to_numeric(data[TARGET], errors="coerce").shift(lag)
    past = pd.to_numeric(data[TARGET], errors="coerce").shift(1)
    out["appliances_mean_6"] = past.rolling(6, min_periods=6).mean()
    out["appliances_mean_36"] = past.rolling(36, min_periods=36).mean()
    out["appliances_std_36"] = past.rolling(36, min_periods=36).std()
    feature_columns = [column for column in out.columns if column not in {"target", "target_time"}]
    out = out.dropna(subset=feature_columns + (["target"] if require_target else [])).reset_index(drop=True)
    return out


def chronological_split(supervised, train_fraction=.70, validation_fraction=.15):
    times = pd.Index(supervised.target_time.unique()).sort_values()
    first = times[int(len(times) * train_fraction)]
    second = times[int(len(times) * (train_fraction + validation_fraction))]
    train = supervised.loc[supervised.target_time < first].copy()
    validation = supervised.loc[(supervised.target_time >= first) & (supervised.target_time < second)].copy()
    test = supervised.loc[supervised.target_time >= second].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    return train, validation, test


def get_feature_columns(supervised):
    return [column for column in supervised.columns if column not in {"target", "target_time"}]
