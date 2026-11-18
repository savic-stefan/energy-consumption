from pathlib import Path
from energy_forecast.data import load_data
from energy_forecast.features import make_supervised, chronological_split
from energy_forecast.training import train, regression_metrics, horizon_study

DATA = Path(__file__).resolve().parents[1] / "data/raw/KAG_energydata_complete.csv"


def test_time_series_cv_training_and_future_holdout():
    supervised = make_supervised(load_data(DATA).iloc[:5000])
    bundle, report, test_result, validation, test = train(supervised)
    assert report["selected_model"] in {"persistence", "daily_seasonal", "ridge", "random_forest", "hist_gradient_boosting"}
    assert report["split_target_dates"]["train_end"] < report["split_target_dates"]["validation_start"]
    assert report["split_target_dates"]["validation_end"] < report["split_target_dates"]["test_start"]
    assert report["test"]["mae"] >= 0
    assert len(test_result) == len(test)
    assert bundle["horizon_steps"] == 6


def test_regression_metrics_known_values():
    values = regression_metrics([1, 2, 3], [1, 2, 4])
    assert values["mae"] == 1 / 3
    assert values["rmse"] > values["mae"]


def test_horizon_study_compares_baselines_with_ml():
    frame = load_data(DATA).iloc[:2500]
    table, winners = horizon_study(frame, horizons=(1,))
    assert set(table.strategy) == {"persistence", "daily_seasonal", "hist_gradient_boosting"}
    assert len(winners) == 1
    assert winners.iloc[0].strategy in set(table.strategy)
