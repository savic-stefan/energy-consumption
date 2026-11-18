"""Data exploration and forecasting report artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from energy_forecast.data import TARGET, TIME, data_summary
from energy_forecast.features import get_feature_columns


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analyze_data(frame, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = data_summary(frame)
    summary["target_quantiles_wh"] = {str(q): float(frame[TARGET].quantile(q)) for q in (.5, .9, .95, .99)}
    summary["hourly_mean_wh"] = {str(int(hour)): float(value) for hour, value in frame.assign(hour=frame[TIME].dt.hour).groupby("hour")[TARGET].mean().items()}
    summary["weekday_mean_wh"] = {str(int(day)): float(value) for day, value in frame.assign(weekday=frame[TIME].dt.day_of_week).groupby("weekday")[TARGET].mean().items()}
    _write_json(output / "data_summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame[TARGET], bins=60, color="#4c78a8")
    axes[0].set(xlabel="Appliance energy (Wh / 10 min)", ylabel="Observations", title="Target distribution")
    hourly = frame.assign(hour=frame[TIME].dt.hour).groupby("hour")[TARGET].mean()
    axes[1].plot(hourly.index, hourly.values, marker="o")
    axes[1].set(xlabel="Hour of day", ylabel="Mean energy (Wh)", title="Mean demand by hour", xticks=range(0, 24, 3))
    fig.tight_layout()
    fig.savefig(output / "data_overview.png", dpi=160)
    plt.close(fig)
    return summary


def write_forecast_report(bundle, report, test_result, validation, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "forecast_metrics.json", report)
    test_result.to_csv(output / "test_predictions.csv", index=False)
    tail = test_result.tail(500)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(tail.target_time, tail.target, label="Observed", linewidth=1)
    ax.plot(tail.target_time, tail.prediction, label=f"Selected: {bundle['name']}", linewidth=1)
    ax.plot(tail.target_time, tail.persistence, label="Persistence", linewidth=1, alpha=.75)
    ax.set(xlabel="Target timestamp", ylabel="Appliance energy (Wh)", title="One-hour-ahead forecast: final 500 test points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "forecast_vs_actual.png", dpi=160)
    plt.close(fig)
    residual = test_result.prediction - test_result.target
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(residual, bins=50, color="#f58518")
    axes[0].set(xlabel="Prediction error (Wh)", ylabel="Count", title="Test residual distribution")
    axes[1].scatter(test_result.target, test_result.prediction, s=8, alpha=.35)
    upper = max(test_result.target.max(), test_result.prediction.max())
    axes[1].plot([0, upper], [0, upper], "--", color="black")
    axes[1].set(xlabel="Observed (Wh)", ylabel="Predicted (Wh)", title="Predicted vs observed")
    fig.tight_layout()
    fig.savefig(output / "forecast_diagnostics.png", dpi=160)
    plt.close(fig)
    by_hour = test_result.assign(hour=test_result.target_time.dt.hour).groupby("hour").apply(
        lambda group: pd.Series({"rows": len(group), "mae": np.abs(group.prediction-group.target).mean(),
                                 "rmse": np.sqrt(np.square(group.prediction-group.target).mean())}),
        include_groups=False,
    )
    by_hour.to_csv(output / "error_by_hour.csv")
    sliced = test_result.assign(
        weekday=test_result.target_time.dt.day_name(),
        target_band=pd.cut(test_result.actual, bins=[-np.inf, 60, 100, 200, np.inf],
                           labels=["0-60 Wh", "61-100 Wh", "101-200 Wh", ">200 Wh"]),
    )
    error_rows = []
    for dimension in ("weekday", "target_band"):
        for value, group in sliced.groupby(dimension, observed=True):
            difference = group.prediction - group.actual
            error_rows.append({"dimension": dimension, "segment": str(value), "rows": len(group),
                               "mae": float(difference.abs().mean()),
                               "rmse": float(np.sqrt(np.square(difference).mean()))})
    pd.DataFrame(error_rows).to_csv(output / "error_by_segment.csv", index=False)
    feature_cols = bundle["feature_columns"]
    importance = permutation_importance(bundle["model"], validation[feature_cols], validation.target,
                                        scoring="neg_mean_absolute_error", n_repeats=3,
                                        random_state=42, n_jobs=1)
    ranked = pd.DataFrame({"feature": feature_cols, "mae_increase": importance.importances_mean,
                           "std": importance.importances_std}).sort_values("mae_increase", ascending=False)
    ranked.to_csv(output / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ranked.head(15).sort_values("mae_increase").plot.barh(x="feature", y="mae_increase", ax=ax, legend=False, color="#54a24b")
    ax.set(xlabel="Increase in validation MAE when shuffled (Wh)", ylabel="", title=f"Permutation importance: CV-best ML comparator ({bundle['ml_comparator']})")
    fig.tight_layout()
    fig.savefig(output / "permutation_importance.png", dpi=160)
    plt.close(fig)


def write_horizon_report(table, winners, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "horizon_metrics.csv", index=False)
    winners.to_csv(output / "horizon_selected_strategies.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in table.groupby("strategy"):
        ax.plot(group.horizon_minutes, group.test_mae, marker="o", label=name)
    ax.set(xlabel="Forecast horizon (minutes)", ylabel="Test MAE (Wh)", title="Forecast error by horizon", xticks=sorted(table.horizon_minutes.unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "horizon_comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Forecast horizon study", "", "Strategies are selected separately for each horizon by mean MAE across three expanding time-series folds within the training period. The validation and test periods remain later in time. Test scores are descriptive final estimates.", ""]
    for row in winners.itertuples(index=False):
        lines.append(f"- {row.horizon_minutes} min: {row.strategy}; CV MAE {row.cv_mae:.2f} Wh, validation MAE {row.validation_mae:.2f} Wh, test MAE {row.test_mae:.2f} Wh and RMSE {row.test_rmse:.2f} Wh.")
    lines += ["", "These results come from one home over roughly four and a half months. They do not establish performance for other buildings or seasons."]
    (output / "horizon_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
