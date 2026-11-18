import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from energy_forecast.reporting import write_forecast_report


def test_forecast_reports_include_time_and_target_size_slices(tmp_path):
    validation = pd.DataFrame({"feature": np.arange(20, dtype=float),
                               "target": np.arange(20, dtype=float) * 2})
    model = DummyRegressor(strategy="mean").fit(validation[["feature"]], validation.target)
    bundle = {"model": model, "name": "persistence", "ml_comparator": "dummy",
              "feature_columns": ["feature"]}
    times = pd.date_range("2020-01-01", periods=20, freq="10min")
    actual = np.tile([20., 60., 80., 120., 300.], 4)
    predictions = pd.DataFrame({"target_time": times, "actual": actual, "target": actual,
                                "prediction": actual + 2, "persistence": actual + 3,
                                "daily_seasonal": actual + 5})
    write_forecast_report(bundle, {"selected_model": "persistence"}, predictions,
                          validation, tmp_path)
    assert (tmp_path / "forecast_vs_actual.png").is_file()
    errors = pd.read_csv(tmp_path / "error_by_segment.csv")
    assert set(errors.dimension) == {"weekday", "target_band"}
    assert {"0-60 Wh", "61-100 Wh", "101-200 Wh", ">200 Wh"}.issubset(set(errors.segment))
