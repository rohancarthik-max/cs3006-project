"""Load the raw CPCB 'Raw Data' daily CSVs into one tidy table:
one row per (station, date), short column names, a complete daily calendar."""
import re

import pandas as pd

from .config import COLUMN_MAP, RAW_DIR, STATIONS

FILE_PATTERN = re.compile(r"Raw_Data_(\d{4})_site_(\d+)_(.+?)_1D\.csv$")


def list_raw_files(raw_dir=RAW_DIR):
    """Return [(path, year, station_id)] for every raw CSV of a known station."""
    files = []
    for path in sorted(raw_dir.glob("Raw_Data_*_1D.csv")):
        m = FILE_PATTERN.search(path.name)
        if m is None:
            raise ValueError(f"Unexpected file name: {path.name}")
        year, site = int(m.group(1)), int(m.group(2))
        if site in STATIONS:
            files.append((path, year, site))
    return files


def read_one(path, station_id):
    df = pd.read_csv(path, na_values=["NA", ""], float_precision="round_trip")  # exact, version-independent parsing
    unknown = set(df.columns) - set(COLUMN_MAP) - {"Timestamp"}
    if unknown:
        raise ValueError(f"{path.name}: unexpected columns {unknown}")
    df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df.pop("Timestamp")).dt.normalize()
    df.insert(0, "station_id", station_id)
    return df


def load_raw(raw_dir=RAW_DIR):
    """Concatenate all files, then give every station a complete daily calendar
    so that a missing day is an explicit NaN row, not a silently absent row."""
    frames = [read_one(p, sid) for p, _, sid in list_raw_files(raw_dir)]
    df = pd.concat(frames, ignore_index=True)

    dupes = df.duplicated(["station_id", "date"]).sum()
    df = df.drop_duplicates(["station_id", "date"], keep="first")

    full = []
    for sid, g in df.groupby("station_id"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["station_id"] = sid
        full.append(g.rename_axis("date").reset_index())
    df = pd.concat(full, ignore_index=True)

    df.insert(1, "station", df["station_id"].map(lambda s: STATIONS[s]["name"]))
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    df.attrs["duplicates_dropped"] = int(dupes)
    return df
