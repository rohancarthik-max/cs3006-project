"""Build the modelling table: one row per (station, day D), with

* day-D pollution (cleaned CPCB) and day-D weather (Open-Meteo),
* the target: was day D+1's mean PM2.5 above the NAAQS limit?
* the split the row belongs to.

Everything a forecaster could know at the end of day D sits on the row; the only
value from D+1 is the target itself (``pm25_next`` / ``y``)."""
import numpy as np
import pandas as pd

from . import config as C


def load_clean():
    return pd.read_csv(C.PROCESSED_DIR / "daily_clean.csv", parse_dates=["date"])


def load_weather():
    return pd.read_csv(C.EXTERNAL_DIR / "weather_daily.csv", parse_dates=["date"])


def split_of(target_date):
    """Split by the year of the TARGET day (D+1). Splitting by D would put the row
    for 31 Dec 2023 (target = 1 Jan 2024) in training while its answer lies in the
    validation year - a one-day leak across the boundary."""
    year_to_split = {y: s for s, years in C.SPLIT_YEARS.items() for y in years}
    return target_date.dt.year.map(year_to_split)


def build_frame(clean=None, weather=None):
    clean = load_clean() if clean is None else clean
    weather = load_weather() if weather is None else weather

    parts = []
    for sid, g in clean.sort_values("date").groupby("station_id"):
        g = g.copy()
        # The cleaned table has one row for every calendar day, so the next row is
        # D+1. Check this rather than assume it.
        assert g["date"].diff().dropna().eq(pd.Timedelta(days=1)).all(), f"gap in calendar for {sid}"
        g["pm25_next"] = g["pm25"].shift(-1)
        g["target_date"] = g["date"] + pd.Timedelta(days=1)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    df["y"] = np.where(df["pm25_next"].isna(), np.nan,
                       (df["pm25_next"] > C.NAAQS_PM25_24H).astype(float))
    df["exceed_today"] = np.where(df["pm25"].isna(), np.nan,
                                  (df["pm25"] > C.NAAQS_PM25_24H).astype(float))
    df["split"] = split_of(df["target_date"])

    wcols = [c for c in weather.columns if c.startswith("w_")]
    df = df.merge(weather[["station_id", "date"] + wcols], on=["station_id", "date"], how="left")
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def festival_flags(dates):
    """Name of the festival on each date ('' if none)."""
    lookup = {pd.Timestamp(d): name for name, ds in C.FESTIVALS.items() for d in ds}
    return dates.map(lambda d: lookup.get(d, ""))
