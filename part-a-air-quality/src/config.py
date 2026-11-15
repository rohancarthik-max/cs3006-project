"""Project-wide settings. Every threshold used in cleaning lives here so it can be
quoted in the report and changed in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_SEED = 42

# CPCB site id -> readable name, area of the city, approximate coordinates.
# Coordinates are approximate (used only to fetch gridded weather, ~10 km grid).
STATIONS = {
    306:  {"name": "Manali",      "area": "North (industrial)", "lat": 13.1644, "lon": 80.2588},
    5365: {"name": "Kodungaiyur", "area": "North",              "lat": 13.1372, "lon": 80.2478},
    5361: {"name": "Arumbakkam",  "area": "Central",            "lat": 13.0727, "lon": 80.2109},
    5363: {"name": "Perungudi",   "area": "South",              "lat": 12.9532, "lon": 80.2357},
}

# Raw CSV header -> short column name
COLUMN_MAP = {
    "PM2.5 (µg/m³)": "pm25", "PM10 (µg/m³)": "pm10", "NO (µg/m³)": "no",
    "NO2 (µg/m³)": "no2", "NOx (ppb)": "nox", "NH3 (µg/m³)": "nh3",
    "SO2 (µg/m³)": "so2", "CO (mg/m³)": "co", "Ozone (µg/m³)": "ozone",
    "Benzene (µg/m³)": "benzene", "Toluene (µg/m³)": "toluene",
    "Xylene (µg/m³)": "xylene", "O Xylene (µg/m³)": "o_xylene",
    "Eth-Benzene (µg/m³)": "eth_benzene", "MP-Xylene (µg/m³)": "mp_xylene",
    "AT (°C)": "temp", "RH (%)": "rh", "WS (m/s)": "ws", "WD (deg)": "wd",
    "RF (mm)": "rf", "TOT-RF (mm)": "tot_rf", "SR (W/mt2)": "sr",
    "BP (mmHg)": "bp", "VWS (m/s)": "vws",
}
POLLUTANTS = ["pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "ozone"]
STATION_WEATHER = ["temp", "rh", "ws", "wd", "rf", "sr", "bp"]

# ---- Problem definition -------------------------------------------------
TARGET_POLLUTANT = "pm25"
NAAQS_PM25_24H = 60.0          # India NAAQS 24-hour limit for PM2.5 (µg/m³)
STUDY_START = "2021-01-01"     # all four stations live from Nov-Dec 2020

# ---- Cleaning thresholds --------------------------------------------------
# Physically possible daily-mean ranges; outside -> sensor/logging error
VALID_RANGE = {
    "pm25": (0.5, 1000), "pm10": (1, 2000), "no": (0, 1000), "no2": (0, 1000),
    "nox": (0, 1000), "nh3": (0, 1000), "so2": (0, 1000), "co": (0, 50),
    "ozone": (0, 1000), "temp": (5, 48), "rh": (5, 100), "ws": (0, 30),
    "wd": (0, 360), "rf": (0, 500), "sr": (0, 1400),
}
PM_RATIO_TOL = 1.05            # PM2.5 is a subset of PM10: flag if PM2.5 > 1.05 x PM10
STUCK_MIN_RUN = 3              # >= 3 identical consecutive daily means = frozen value
STUCK_TOL = 0.005              # "identical" = within 0.005 units
FLAT_WINDOW = 14               # days
FLAT_MAX_CV = 0.02             # 14-day coefficient of variation < 2% = sensor not responding
PEER_WINDOW = 90               # days, rolling window for the neighbour comparison
PEER_RATIO_LIMITS = (0.6, 1.67)  # station / median(other stations) outside this -> drift

STUCK_SKIP = ["rf"]            # rain is legitimately 0 for many days in a row

# Decisions taken after reviewing every peer-drift flag (see notebook 01).
# Each is backed by two independent signals, not by the ratio alone.
MANUAL_EXCLUSIONS = [
    {"station_id": 5365, "columns": ["pm25", "pm10"], "start": "2021-05-12", "end": "2021-05-24",
     "reason": "Failing instrument just before a months-long outage: PM2.5 jumps 7 -> 214 -> 5 -> 397 "
               "between consecutive days while all three neighbours stay at 5-24 µg/m³."},
    {"station_id": 5365, "columns": ["pm25"], "start": "2022-05-10", "end": "2025-03-09",
     "reason": "Calibration drift: PM2.5 falls to 0.3-0.8x the other stations while its "
               "own PM10 stays ~1x and its PM2.5/PM10 ratio sits at 0.22 vs 0.4-0.5 "
               "elsewhere. Returns to normal after a maintenance gap 2025-02-20..03-09."},
    {"station_id": 306, "columns": ["pm25"], "start": "2024-03-28", "end": "2024-09-30",
     "reason": "Over-reading after an outage: PM2.5 jumps to 2-4x the other stations while "
               "PM10 does not; PM2.5/PM10 = 0.65 vs 0.25 elsewhere. Normal again in Oct 2024."},
]
REVIEWED_AND_KEPT = [
    {"station": "Perungudi", "period": "2021-06 .. 2021-10",
     "why_kept": "Ratio 0.5-0.6, but Perungudi reads low by a similar factor all of 2021 "
                 "with no step change; plausible for the southern, coastal site. "
                 "No second signal of a fault, so the data are kept."},
    {"station": "Manali", "period": "2023-08-29 .. 2023-09-25",
     "why_kept": "Short (12 days total), ratio just above the limit (1.67-1.71); "
                 "consistent with local industrial episodes."},
    {"station": "Perungudi", "period": "2026-08-30 .. 2026-09-02",
     "why_kept": "4 days at ratio 0.56-0.60, at the edge of the band; too short to call drift."},
]

# ---- Splits (stage 1 decision; by the year of the TARGET day D+1, never shuffled) ----
SPLIT_YEARS = {"train": [2021, 2022, 2023], "val": [2024], "test": [2025], "monitor": [2026]}

# ---- Festival days with city-wide burning (used in EDA; candidate calendar feature) ----
# Diwali (firecrackers) and Bhogi (burning old household items, the day before Pongal).
FESTIVALS = {
    "Diwali": ["2021-11-04", "2022-10-24", "2023-11-12", "2024-10-31", "2025-10-20", "2026-11-08"],
    "Bhogi":  ["2021-01-13", "2022-01-13", "2023-01-14", "2024-01-14", "2025-01-13", "2026-01-14"],
}
