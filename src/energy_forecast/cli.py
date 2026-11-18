"""Command-line entry points for analysis, training and forecasting."""
import argparse
import json
from pathlib import Path
from energy_forecast.data import load_data
from energy_forecast.features import make_supervised
from energy_forecast.training import train, horizon_study
from energy_forecast.reporting import analyze_data, write_forecast_report, write_horizon_report
from energy_forecast.inference import save_bundle, forecast_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-hour-ahead appliance energy forecast")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "train"):
        command = commands.add_parser(name)
        command.add_argument("--data", default="data/raw/KAG_energydata_complete.csv")
        command.add_argument("--reports", default="reports/generated")
        if name == "train":
            command.add_argument("--model", default="artifacts/energy_forecast.joblib")
    predict = commands.add_parser("predict")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model", default="artifacts/energy_forecast.joblib")
    predict.add_argument("--output", default="reports/generated/forecasts.csv")
    study = commands.add_parser("horizon-study", help="compare 10-minute to 3-hour forecast horizons")
    study.add_argument("--data", default="data/raw/KAG_energydata_complete.csv")
    study.add_argument("--reports", default="reports/generated")
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            result = forecast_csv(args.input, args.model, args.output)
            print(f"Wrote {len(result)} forecasts to {args.output}")
            return 0
        frame = load_data(args.data)
        summary = analyze_data(frame, args.reports)
        if args.command == "analyze":
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "horizon-study":
            table, winners = horizon_study(frame)
            write_horizon_report(table, winners, args.reports)
            print(winners.to_string(index=False))
            return 0
        if Path(args.model).resolve() == Path(args.data).resolve():
            raise ValueError("Model path must differ from source data")
        supervised = make_supervised(frame)
        bundle, report, test_result, validation, _ = train(supervised)
        write_forecast_report(bundle, report, test_result, validation, args.reports)
        save_bundle(bundle, args.model)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
