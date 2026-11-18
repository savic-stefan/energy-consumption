"""Chronological baselines, time-series CV, and model selection."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from energy_forecast.features import chronological_split, get_feature_columns


def regression_metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def build_models(random_state=42):
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=3,
                                                max_features=.85, n_jobs=-1, random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=.08,
                                                                 l2_regularization=1.0, random_state=random_state),
    }


def train(supervised, random_state=42, horizon_steps=6):
    train, validation, test = chronological_split(supervised)
    features = get_feature_columns(supervised)
    X_train, y_train = train[features], train.target
    X_validation, y_validation = validation[features], validation.target
    X_test, y_test = test[features], test.target
    models = build_models(random_state)
    cv = TimeSeriesSplit(n_splits=3)
    results = {"baselines": {}, "models": {}}
    baseline_columns = {
        "persistence": "appliances_lag_0",
        "daily_seasonal": "appliances_lag_144",
    }
    baseline_cv = {name: [] for name in baseline_columns}
    for _, fold_validation in cv.split(X_train):
        for name, column in baseline_columns.items():
            baseline_cv[name].append(float(mean_absolute_error(
                y_train.iloc[fold_validation], train[column].iloc[fold_validation]
            )))
    for name, column in baseline_columns.items():
        results["baselines"][name] = {
            "cv_mae_mean": float(np.mean(baseline_cv[name])),
            "cv_mae_std": float(np.std(baseline_cv[name])),
            "validation": regression_metrics(y_validation, validation[column].to_numpy()),
        }
    for name, model in models.items():
        cv_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
        model.fit(X_train, y_train)
        validation_prediction = model.predict(X_validation)
        results["models"][name] = {
            "cv_mae_mean": float(cv_scores.mean()),
            "cv_mae_std": float(cv_scores.std()),
            "validation": regression_metrics(y_validation, validation_prediction),
        }
    all_cv_scores = {
        **{name: values["cv_mae_mean"] for name, values in results["baselines"].items()},
        **{name: values["cv_mae_mean"] for name, values in results["models"].items()},
    }
    selected_name = min(all_cv_scores, key=all_cv_scores.get)
    best_ml_name = min(models, key=lambda name: results["models"][name]["cv_mae_mean"])
    selected_model = models[best_ml_name]
    if selected_name in baseline_columns:
        test_prediction = test[baseline_columns[selected_name]].to_numpy()
    else:
        selected_model = models[selected_name]
        test_prediction = selected_model.predict(X_test)
    report = {
        "horizon_minutes": horizon_steps * 10,
        "selected_model": selected_name,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_target_dates": {"train_end": str(train.target_time.max()),
                               "validation_start": str(validation.target_time.min()),
                               "validation_end": str(validation.target_time.max()),
                               "test_start": str(test.target_time.min()),
                               "test_end": str(test.target_time.max())},
        "validation": results,
        "test": regression_metrics(y_test, test_prediction),
        "test_baselines": {name: regression_metrics(y_test, test[name].to_numpy())
                           for name in ("appliances_lag_0", "appliances_lag_144")},
    }
    bundle = {"model": models[best_ml_name], "feature_columns": features, "name": selected_name,
              "strategy": selected_name, "ml_comparator": best_ml_name,
              "horizon_steps": horizon_steps, "format_version": 1}
    test_result = test[["target_time", "target"]].copy().rename(columns={"target": "actual"})
    test_result["prediction"] = test_prediction
    test_result["persistence"] = test.appliances_lag_0.to_numpy()
    test_result["daily_seasonal"] = test.appliances_lag_144.to_numpy()
    test_result["target"] = test_result["actual"]
    return bundle, report, test_result, validation, test


def horizon_study(frame, horizons=(1, 3, 6, 18), random_state=42):
    """Compare simple baselines and gradient boosting across forecast horizons."""
    from energy_forecast.features import make_supervised

    rows = []
    for horizon in horizons:
        supervised = make_supervised(frame, horizon_steps=horizon)
        train, validation, test = chronological_split(supervised)
        features = get_feature_columns(supervised)
        X_train, y_train = train[features], train.target
        X_validation, y_validation = validation[features], validation.target
        X_test, y_test = test[features], test.target
        cv = TimeSeriesSplit(n_splits=3)
        candidates = {
            "persistence": (None, "appliances_lag_0"),
            "daily_seasonal": (None, "appliances_lag_144"),
            "hist_gradient_boosting": (HistGradientBoostingRegressor(
                max_iter=180, learning_rate=.08, l2_regularization=1.0,
                random_state=random_state), None),
        }
        for name, (model, baseline_column) in candidates.items():
            if baseline_column:
                cv_mae = []
                for _, fold in cv.split(X_train):
                    cv_mae.append(mean_absolute_error(y_train.iloc[fold], train[baseline_column].iloc[fold]))
                validation_prediction = validation[baseline_column].to_numpy()
                test_prediction = test[baseline_column].to_numpy()
            else:
                cv_mae = -cross_val_score(model, X_train, y_train, cv=cv,
                                          scoring="neg_mean_absolute_error", n_jobs=1)
                model.fit(X_train, y_train)
                validation_prediction = model.predict(X_validation)
                test_prediction = model.predict(X_test)
            rows.append({"horizon_minutes": horizon * 10, "strategy": name,
                         "cv_mae": float(np.mean(cv_mae)),
                         "validation_mae": float(mean_absolute_error(y_validation, validation_prediction)),
                         "test_mae": float(mean_absolute_error(y_test, test_prediction)),
                         "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_prediction)))})
    table = pd.DataFrame(rows)
    winners = table.loc[table.groupby("horizon_minutes").cv_mae.idxmin()].sort_values("horizon_minutes")
    return table, winners
