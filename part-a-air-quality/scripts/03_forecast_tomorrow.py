"""Stage 8: the daily forecast. At the end of day D, print tomorrow's PM2.5 alert for
each station and save it to reports/alerts/alert_<D+1>.csv.

    python scripts/03_forecast_tomorrow.py --date 2025-10-19
    python scripts/03_forecast_tomorrow.py --date 2026-11-14 --live

Default mode uses the committed data files (reproducible). --live fills day D's
weather from the Open-Meteo forecast service when the archive does not have it yet
(the archive lags about a week); PM2.5 still comes from the CPCB files in data/.
Run scripts/02_save_model.py once first.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_weather  # noqa: E402
from src.deploy import bulletin_text, forecast, load_model, save_bulletin  # noqa: E402
from src.weather import fetch_live  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Next-day PM2.5 alert for the four Chennai stations")
    ap.add_argument("--date", required=True, help="day D (YYYY-MM-DD); the forecast is for D+1")
    ap.add_argument("--live", action="store_true",
                    help="fetch day D's weather from the Open-Meteo forecast service if the archive lacks it")
    ap.add_argument("--no-save", action="store_true", help="print only, do not write the CSV")
    args = ap.parse_args()

    day = pd.Timestamp(args.date)
    model, meta = load_model()
    weather = load_weather()
    if args.live and not (weather["date"] == day).any():
        start = (day - pd.Timedelta(days=3)).date().isoformat()    # 3-day rain total needs D-2..D
        live = fetch_live(start, day.date().isoformat())
        live = live[~live.set_index(["station_id", "date"]).index.isin(
            weather.set_index(["station_id", "date"]).index)]
        weather = pd.concat([weather, live.drop(columns="source")], ignore_index=True)
        print(f"(weather for {start} .. {day.date()} taken live from the Open-Meteo forecast service)\n")
    try:
        fc = forecast(day, model, meta, weather=weather)
    except ValueError as e:
        sys.exit(f"Cannot forecast: {e}")
    print(bulletin_text(fc, meta))
    if not args.no_save:
        print("\nsaved", save_bulletin(fc).relative_to(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    main()
