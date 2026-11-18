from pathlib import Path
import pandas as pd
import pytest
from energy_forecast.data import load_data, validate_data, data_summary

DATA = Path(__file__).resolve().parents[1] / "data/raw/KAG_energydata_complete.csv"


def test_real_dataset_shape_quality_and_summary():
    frame = load_data(DATA)
    summary = data_summary(frame)
    assert frame.shape == (19735, 29)
    assert summary["missing_cells"] == 0
    assert summary["duplicate_timestamps"] == 0
    assert summary["start"] == "2016-01-11 17:00:00"


def test_irregular_cadence_is_rejected():
    frame = load_data(DATA).iloc[:4].copy()
    frame.loc[2, "date"] = frame.loc[1, "date"] + pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="regular 10-minute"):
        validate_data(frame)
