"""Save the fitted model and forecast from a local historical series."""
from pathlib import Path
import joblib
import pandas as pd
from energy_forecast.features import make_supervised, get_feature_columns
from energy_forecast.data import load_data


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def forecast_csv(input_path, model_path, output_path):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle")
    frame = load_data(input_path)
    features = make_supervised(frame, horizon_steps=bundle["horizon_steps"], require_target=False)
    columns = bundle["feature_columns"]
    missing = set(columns).difference(features.columns)
    if missing:
        raise ValueError(f"Forecast input lacks required features: {', '.join(sorted(missing))}")
    strategy = bundle.get("strategy", "model")
    if strategy == "persistence":
        prediction = features["appliances_lag_0"].to_numpy()
    elif strategy == "daily_seasonal":
        prediction = features["appliances_lag_144"].to_numpy()
    else:
        prediction = bundle["model"].predict(features[columns])
    result = pd.DataFrame({"forecast_time": features.target_time,
                           "predicted_appliances_wh": prediction})
    destination = Path(output_path)
    if destination.resolve() == Path(input_path).resolve() or destination.resolve() == Path(model_path).resolve():
        raise ValueError("Output path must differ from input and model paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
