from pathlib import Path
from energy_forecast.data import load_data
from energy_forecast.features import make_supervised, chronological_split, get_feature_columns

DATA = Path(__file__).resolve().parents[1] / "data/raw/KAG_energydata_complete.csv"


def test_features_are_causal_and_future_is_only_the_target():
    frame = load_data(DATA)
    supervised = make_supervised(frame.iloc[:300].copy())
    row = 100
    assert supervised.loc[row, "target_time"] - supervised.loc[row - 6, "target_time"] == __import__("pandas").Timedelta(hours=1)
    assert "rv1" not in get_feature_columns(supervised)
    assert "rv2" not in get_feature_columns(supervised)
    changed = frame.iloc[:300].copy()
    changed.loc[250, "Appliances"] += 10000
    before = make_supervised(frame.iloc[:300].copy())
    after = make_supervised(changed)
    feature_cols = get_feature_columns(supervised)
    import numpy as np
    assert np.allclose(before.loc[row, feature_cols], after.loc[row, feature_cols], atol=1e-9)
    assert before.loc[row, "target"] != after.loc[row, "target"]


def test_chronological_partitions_do_not_overlap_target_time():
    supervised = make_supervised(load_data(DATA))
    train, validation, test = chronological_split(supervised)
    assert train.target_time.max() < validation.target_time.min()
    assert validation.target_time.max() < test.target_time.min()


def test_custom_forecast_horizon_changes_future_target():
    frame = load_data(DATA).iloc[:400].copy()
    one_step = make_supervised(frame, horizon_steps=1)
    three_hour = make_supervised(frame, horizon_steps=18)
    assert one_step.loc[0, "target_time"] - frame.date.iloc[144] == __import__("pandas").Timedelta(minutes=10)
    assert three_hour.loc[0, "target_time"] - frame.date.iloc[144] == __import__("pandas").Timedelta(hours=3)
