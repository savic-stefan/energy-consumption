from pathlib import Path
import joblib
import pandas as pd
from energy_forecast.data import load_data
from energy_forecast.inference import forecast_csv

DATA = Path(__file__).resolve().parents[1] / "data/raw/KAG_energydata_complete.csv"


def test_saved_persistence_strategy_forecasts_through_latest_time(tmp_path):
    frame = load_data(DATA)
    bundle = {"format_version": 1, "strategy": "persistence", "name": "persistence",
              "horizon_steps": 6, "feature_columns": []}
    model = tmp_path / "model.joblib"
    joblib.dump(bundle, model)
    predictions = forecast_csv(DATA, model, tmp_path / "forecasts.csv")
    assert len(predictions) == len(frame) - 144
    assert predictions.forecast_time.max() == frame.date.max() + pd.Timedelta(minutes=60)
    assert predictions.predicted_appliances_wh.iloc[-1] == frame.Appliances.iloc[-1]
