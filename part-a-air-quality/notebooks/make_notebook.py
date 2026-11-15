"""Builds 01_problem_and_data_audit.ipynb from the cells below (keeps the notebook
diff-friendly in git). Run: python notebooks/make_notebook.py"""
import json
from pathlib import Path

cells = []
# Fixed cell ids keep nbformat happy and the notebook stable between runs.
md = lambda s: cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {},
                             "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {},
                               "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# 01 · Problem definition and data audit
**Project:** Next-day PM2.5 exceedance alerts for Chennai · CS3006

This notebook covers pipeline stages **1 Define → 2 Collect → 3 Clean**.
The cleaning logic lives in `src/` so the notebook and `scripts/01_build_clean_dataset.py`
use exactly the same code.

**Problem (full version in `docs/problem_statement.md`):** for each of four CPCB stations,
predict at the end of day *D* whether the daily mean PM2.5 on day *D+1* will exceed
India's NAAQS 24-hour limit of **60 µg/m³**.
""")

code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))      # run from the notebooks/ folder

import pandas as pd
from IPython.display import Image, display

from src import config as C
from src.load import load_raw, list_raw_files
from src.health import clean, peer_ratio
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 160)
""")

md(r"""
## 2 · Data collection
**Source:** Central Pollution Control Board (CPCB), *Data Repository → Raw Data*, frequency
24 h, one CSV per station per year (downloaded Nov 2026). Each file has 12 pollutants and 9
weather channels.

**Why these four stations?** All eight Chennai stations were screened on 2026 data
(coverage, frozen values, agreement with the other stations). Alandur (empty file) and
Ennore (6% coverage) had no data; Velachery repeated filler values (PM2.5 = 17.0 on 29 days);
Royapuram had a frozen PM10 sensor and PM10 computed as a fixed 2.17 × PM2.5. The four
kept stations span the city from the industrial north to the south.
""")

code(r"""
pd.DataFrame(C.STATIONS).T
""")

code(r"""
files = list_raw_files()
print(len(files), "raw files")
raw_all = load_raw()
raw_all.groupby("station")["pm25"].apply(lambda s: s.dropna().index.size).rename("days with PM2.5")
""")

md(r"""
### First audit (before changing anything)
The slides' audit checklist: shape, dtypes, share missing, duplicates, impossible values.
""")

code(r"""
first_day = raw_all.dropna(subset=["pm25"]).groupby("station")["date"].min()
print("First day with PM2.5 per station:\n", first_day, "\n")
raw = raw_all[raw_all["date"] >= C.STUDY_START].reset_index(drop=True)
print("Study window shape:", raw.shape)
print("Duplicate (station, date) rows dropped while loading:", raw_all.attrs["duplicates_dropped"])
raw[C.POLLUTANTS + C.STATION_WEATHER].isna().mean().sort_values(ascending=False).round(3)
""")

md(r"""
### Data types: what the raw files contain and how they were converted
""")

code(r"""
one_file = pd.read_csv(files[0][0])
print("Raw file as read by pandas:", one_file.dtypes.value_counts().to_dict())
print("Example column names (units are inside the names):", list(one_file.columns[:4]))
print("\nAfter src/load.py:")
raw[["date", "station_id", "station", "pm25", "co", "nox"]].dtypes
""")

md(r"""
**Type handling and why**
* `Timestamp` arrives as **text**. It is parsed to a real date and normalised to midnight, so
  that days can be compared, sorted and shifted (D → D+1).
* Pollutant columns are already numbers (`float64`); missing readings are written as `NA` or
  blank in the CPCB files and are read as `NaN`, never as 0.
* **Units are kept in mind, not converted:** most pollutants are µg/m³, but CO is mg/m³ and NOx
  is ppb. The short names (`co`, `nox`) are set in `config.COLUMN_MAP`, and their valid ranges in
  `config.VALID_RANGE` use the matching unit.
* If CPCB ever adds an unknown column, `load.py` **stops with an error** instead of silently
  ignoring it.
* Every station gets a **complete daily calendar**: a missing day becomes an explicit row of
  `NaN`, not a silently absent row. That is what makes "tomorrow = next row" safe later.
""")

code(r"""
raw[C.POLLUTANTS + ["rh", "ws", "rf"]].describe().T.round(1)
""")

md(r"""
**What the audit already shows**
* Three stations only start in Nov-Dec 2020, which is why the study starts on 2021-01-01.
* `rf` (rainfall) is almost always 0. Chennai gets heavy north-east monsoon rain in
  Oct-Dec, so the gauges are not working → weather comes from Open-Meteo instead
  (`scripts/00_fetch_weather.py`).
* `temp` is missing for 90% of days at these stations.
* PM2.5 has a long right tail (max >> 75th percentile), which is the exceedance days we care about.

## 3 · Cleaning: sensor-health rules
Each rule targets one known way a CPCB sensor fails. Thresholds are in `src/config.py`.

| Rule | What it catches | Action |
|---|---|---|
| out_of_range | physically impossible values (e.g. RH > 100%) | set NaN |
| pm25_gt_pm10 | PM2.5 > 1.05 × PM10 (PM2.5 is *part of* PM10) | set both NaN |
| stuck_value | ≥ 3 identical daily means in a row (frozen output) | keep first, NaN the repeats |
| flatline | 14-day coefficient of variation < 2% (sensor stops responding) | set NaN |
| peer_drift | 90-day station ÷ median-of-others outside 0.6-1.67 | **flag only**, then review |
| manual exclusion | flagged periods confirmed by a second, independent signal | set NaN, reason logged |
""")

code(r"""
cleaned, audit, drift, spikes, summary = clean(raw)
summary.assign(pct_removed=(summary.removed / summary.valid_before * 100).round(1))
""")

code(r"""
audit.pivot_table(index=["rule", "column"], columns="station", values="values_removed",
                  aggfunc="sum").fillna(0).astype(int)
""")

code(r"""
display(Image(str(C.FIG_DIR / "fig04_stuck_sensor_examples.png")))
""")

md(r"""
### Peer check: catching slow calibration drift
All four stations sit in the same city under the same weather, so a healthy sensor's
reading divided by its neighbours' median stays roughly steady. Drift shows as a slow slide.
""")

code(r"""
drift
""")

code(r"""
display(Image(str(C.FIG_DIR / "fig02_peer_ratio_drift.png")))
display(Image(str(C.FIG_DIR / "fig03_pm25_pm10_ratio.png")))
""")

md(r"""
**Decision rule:** a period is removed only when a second, independent signal agrees.
Here that second signal is the station's PM2.5/PM10 ratio compared with the others.
* **Kodungaiyur 2022-05 → 2025-03:** peer ratio falls to 0.3-0.8 *and* PM2.5/PM10 is flat at
  ~0.22 while its PM10 stays normal → the PM2.5 instrument under-read. It recovers right
  after a maintenance gap (20 Feb - 9 Mar 2025). Removed.
* **Manali 2024-03-28 → 2024-09-30:** peer ratio 2-4 *and* PM2.5/PM10 = 0.65 vs 0.25 elsewhere
  → over-reading. Removed.
* **Perungudi mid-2021:** ratio 0.5-0.6 but no second signal and no step change. Kept.
""")

code(r"""
display(pd.DataFrame(C.MANUAL_EXCLUSIONS))
display(pd.DataFrame(C.REVIEWED_AND_KEPT))
""")

md(r"""
### Single-station spikes: error or the most interesting event?
Days where one station reads > 3× the others and > 100 µg/m³. These are **kept**:
5 Nov 2021 is the day after Diwali, and short local events (waste fires, crackers)
are real exposure. The model will most likely miss them, and the error analysis will say so.
""")

code(r"""
spikes
""")

md(r"""
### Why the textbook outlier rule (IQR) was *not* applied
The slides' IQR rule flags values above Q3 + 1.5 × IQR. What would it do to PM2.5?
""")

code(r"""
pm = cleaned.loc[cleaned.date.dt.year <= 2023, "pm25"].dropna()     # training years only
q1, q3 = pm.quantile([0.25, 0.75])
fence = q3 + 1.5 * (q3 - q1)
flagged = pm > fence
bad = pm > C.NAAQS_PM25_24H
print(f"Q1 = {q1:.1f}, Q3 = {q3:.1f}, upper IQR fence = {fence:.1f} µg/m³")
print(f"Days flagged as 'outliers': {flagged.sum()}  |  of these, days above the 60 limit: {(flagged & bad).sum()}")
print(f"Share of all bad days the IQR rule would delete: {(flagged & bad).sum() / bad.sum():.0%}")
""")

md(r"""
**Decision:** the IQR rule is **not** used on PM2.5. Its fence (66 µg/m³) sits just above the
limit, so every day it flags is a bad-air day: it would delete about two-thirds of the days we
are trying to predict. High PM2.5 is the signal, not noise. As the slides put it, an outlier is
either an error or the most interesting event in the data, and only domain knowledge can tell
which. Errors are removed by the **sensor-health rules above** (impossible values, frozen and
flat-lined sensors, PM2.5 > PM10, drift), each with a physical reason, and real extreme days are
kept.
""")

md("## Result")

code(r"""
display(Image(str(C.FIG_DIR / "fig01_pm25_availability.png")))
display(Image(str(C.FIG_DIR / "fig05_clean_pm25_timeseries.png")))
""")

code(r"""
d = cleaned.dropna(subset=["pm25"])
balance = (d.assign(year=d.date.dt.year, exceed=d.pm25 > C.NAAQS_PM25_24H)
             .groupby("year").agg(days=("exceed", "size"), exceed_days=("exceed", "sum"),
                                  exceed_rate=("exceed", "mean")))
balance.round(3)
""")

md(r"""
**Takeaways for the next stages**
* ~6,800 usable station-days of PM2.5 after removing 14.7% as sensor faults.
* Classes are imbalanced: 5.5-8.4% of days exceed the limit in 2021-2024, rising to ~13% in
  2025-2026. The test year has twice the training rate, which is a data-drift risk to examine.
* Station weather is unreliable → use Open-Meteo weather (next notebook).

**Next:** `02_eda.ipynb`: seasonality, persistence, weather relationships and hypothesis
tests on the training years. Feature building follows in `03_features.ipynb`.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("01_problem_and_data_audit.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
