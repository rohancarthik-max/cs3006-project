"""Open-Meteo weather: hourly -> daily aggregation (shared by the archive download in
scripts/00_fetch_weather.py and the live mode of the daily forecast), plus a
live fetch from the Open-Meteo *forecast* service for days the archive does not
have yet (the archive lags real time by about a week)."""
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "surface_pressure", "wind_speed_10m", "wind_direction_10m",
          "boundary_layer_height"]
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def to_daily(h):
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.normalize()
    # wind direction must be averaged as a vector, not as plain degrees
    rad = np.deg2rad(h["wind_direction_10m"])
    h["u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["v"] = -h["wind_speed_10m"] * np.cos(rad)
    agg = {"temperature_2m": ["mean", "min", "max"], "relative_humidity_2m": "mean",
           "dew_point_2m": "mean", "precipitation": "sum", "surface_pressure": "mean",
           "wind_speed_10m": ["mean", "max"], "u": "mean", "v": "mean"}
    if "boundary_layer_height" in h:
        agg["boundary_layer_height"] = ["mean", "min"]
    d = h.groupby("date").agg(agg)
    d.columns = ["_".join(c) for c in d.columns]
    d = d.rename(columns={
        "temperature_2m_mean": "w_temp_mean", "temperature_2m_min": "w_temp_min",
        "temperature_2m_max": "w_temp_max", "relative_humidity_2m_mean": "w_rh_mean",
        "dew_point_2m_mean": "w_dewpoint_mean", "precipitation_sum": "w_rain_mm",
        "surface_pressure_mean": "w_pressure_hpa", "wind_speed_10m_mean": "w_wind_mean",
        "wind_speed_10m_max": "w_wind_max", "u_mean": "w_wind_u", "v_mean": "w_wind_v",
        "boundary_layer_height_mean": "w_blh_mean", "boundary_layer_height_min": "w_blh_min"})
    # Round to 4 decimals: the source has 1-2 decimals, and rounding stops the
    # last floating-point digit from changing between numpy/pandas versions.
    return d.round(4).reset_index()


def fetch_live(start, end, tries=3):
    """Daily weather for [start, end] for every station from the Open-Meteo FORECAST
    service (model analysis for past days). Used only by the --live forecast mode.
    Caution: this is not the ERA5 archive the model was trained on, so values can
    differ slightly from what the archive will later report."""
    frames = []
    for sid, meta in C.STATIONS.items():
        params = {"latitude": meta["lat"], "longitude": meta["lon"], "start_date": start,
                  "end_date": end, "hourly": ",".join(HOURLY), "timezone": "Asia/Kolkata",
                  "wind_speed_unit": "ms"}
        for attempt in range(1, tries + 1):
            try:
                r = requests.get(FORECAST_URL, params=params, timeout=60)
                r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == tries:
                    raise
                time.sleep(3 * attempt)
        d = to_daily(pd.DataFrame(r.json()["hourly"]))
        d.insert(0, "station_id", sid)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "open-meteo forecast (live)"
    return out
