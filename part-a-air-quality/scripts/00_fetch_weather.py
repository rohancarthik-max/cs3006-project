"""Download daily weather for each station from the Open-Meteo historical archive
(ERA5 reanalysis, free, no API key) -> data/external/weather_daily.csv

Why not the station's own weather sensors? The audit (notebook 01) found every
station's rain gauge reads ~0 all year and several humidity / wind sensors are stuck.

Run from the project root:   python scripts/00_fetch_weather.py
Needs internet access. The output CSV is committed to the repo, so the rest of the
pipeline runs offline and gives identical results.
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.weather import HOURLY, to_daily  # noqa: E402  (shared with the live forecast)

URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2020-12-01"
END = (date.today() - timedelta(days=7)).isoformat()   # archive lags real time by a few days


def year_chunks(start, end):
    """Split [start, end] into calendar-year pieces. One 6-year request can time out
    on the free server; yearly requests are small and fast."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s <= e:
        stop = min(date(s.year, 12, 31), e)
        yield s.isoformat(), stop.isoformat()
        s = stop + timedelta(days=1)


def fetch_chunk(lat, lon, variables, start, end, tries=4):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(variables), "timezone": "Asia/Kolkata",
              "wind_speed_unit": "ms"}
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(URL, params=params, timeout=120)
            r.raise_for_status()
            return pd.DataFrame(r.json()["hourly"])
        except (requests.ConnectionError, requests.Timeout):
            if attempt == tries:
                raise
            print(f"  timeout on {start}..{end}, retry {attempt}/{tries - 1}")
            time.sleep(5 * attempt)


def fetch(lat, lon, variables):
    parts = [fetch_chunk(lat, lon, variables, s, e) for s, e in year_chunks(START, END)]
    return pd.concat(parts, ignore_index=True)


def main():
    C.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid, meta in C.STATIONS.items():
        try:
            h = fetch(meta["lat"], meta["lon"], HOURLY)
        except requests.HTTPError:
            # boundary_layer_height is not served for every archive model; retry without it
            print(f"{meta['name']}: retrying without boundary_layer_height")
            h = fetch(meta["lat"], meta["lon"], [v for v in HOURLY if v != "boundary_layer_height"])
        d = to_daily(h)
        d.insert(0, "station_id", sid)
        frames.append(d)
        print(f"{meta['name']:12s} {len(d)} days  {d['date'].min().date()} .. {d['date'].max().date()}")
        time.sleep(1)   # be polite to the free API
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(C.EXTERNAL_DIR / "weather_daily.csv", index=False, lineterminator="\n")
    print("Saved data/external/weather_daily.csv", out.shape)


if __name__ == "__main__":
    main()
