"""Builds 06_deployment.ipynb (pipeline stage 8: Deploy) from the cells below.
Run: python notebooks/make_notebook_06.py   then execute the notebook."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {},
                             "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {},
                               "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# 06 · Deployment: the frozen model as a daily forecast
**Project:** Next-day PM2.5 exceedance alerts for Chennai · CS3006 · pipeline stage **8 Deploy**

Stage 7 measured the model. This stage packages it so it could run every evening:

| Piece | File | What it does |
|---|---|---|
| Saved model | `models/primary_logreg.joblib` + `.json` | the frozen model, fitted once, with a record of exactly what it is |
| Save script | `scripts/02_save_model.py` | refits the frozen choice and saves it (deterministic) |
| Daily script | `scripts/03_forecast_tomorrow.py --date D [--live]` | prints tomorrow's alert per station, saves `reports/alerts/alert_<D+1>.csv` |
| Model card | `docs/model_card.md` | intended use, results, weaknesses, maintenance |

This notebook **checks** the deployed pieces; it does not tune anything. Dates are taken
from 2025 (the test year, already evaluated), so 2026 stays untouched for stage 9 (Monitor).
""")

code(r"""
import sys, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))      # run from the notebooks/ folder

import numpy as np
import pandas as pd

from src import config as C, deploy as D, evaluate as E, features as F, models as M
from src.dataset import load_clean, load_weather
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)

model, meta = D.load_model()
{k: meta[k] for k in ["model", "params", "trained_on_target_years", "training_rows",
                      "training_bad_days", "threshold_probability", "reference_rule_ugm3", "scikit_learn"]}
""")

md(r"""
## 1 · Is the saved model the evaluated model?
The saved file must give exactly the predictions that were scored in notebook 05. We refit the
frozen choice from scratch and compare every 2025 probability.
""")

code(r"""
feats = F.make_features()
tr, te = F.model_rows(feats, "train"), F.model_rows(feats, "test")
refit, _ = E.frozen_models(tr, M.load_choice())
p_saved = model.predict_proba(te[F.FEATURES])[:, 1]
p_refit = refit.predict_proba(te[F.FEATURES])[:, 1]
print("largest difference over", len(te), "rows:", float(np.abs(p_saved - p_refit).max()))
assert np.array_equal(p_saved, p_refit), "saved model differs from the evaluated model"
alerts = p_saved >= meta["threshold_probability"]
print(f"2025 with the saved model: {int((alerts & (te.y == 1)).sum())} of {int(te.y.sum())} bad days caught, "
      f"{int((alerts & (te.y == 0)).sum())} false alarms (notebook 05: 129 and 58)")
""")

code(r"""
# Saving is deterministic: fitting and saving again gives a byte-identical model file.
h1 = hashlib.sha256(D.MODEL_FILE.read_bytes()).hexdigest()
D.save_model()
h2 = hashlib.sha256(D.MODEL_FILE.read_bytes()).hexdigest()
print("same file after re-saving:", h1 == h2, "| sha256", h1[:16], "...")
assert h1 == h2
model, meta = D.load_model()
""")

md(r"""
## 2 · Does the daily forecast match the evaluation, day by day?
`D.forecast(day)` cuts all data off at day D, rebuilds the features from scratch and applies the
saved model, as the script would on the evening of day D. For every day of Oct-Dec 2025
(the peak season) we compare its alerts with the alerts scored in notebook 05.
""")

code(r"""
clean, weather = load_clean(), load_weather()
ev = E.alerts_frame(te, E.scores_for(te, model), E.thresholds(M.load_choice()))
days = pd.date_range("2025-10-01", "2025-12-30")
t0, mismatches, rows = time.time(), 0, []
for day in days:
    fc = D.forecast(day, model, meta, clean, weather)
    ref = ev[ev.date == day].set_index("station")
    for r in fc.itertuples():
        if r.station in ref.index:
            same_alert = bool(r.model_alert) == bool(ref.at[r.station, "alert"])
            same_p = abs(r.probability_bad_tomorrow - round(ref.at[r.station, "score"], 3)) < 1e-9
            mismatches += (not same_alert) + (not same_p)
            rows.append(r)
secs = time.time() - t0
print(f"{len(days)} daily forecasts, {len(rows)} station-forecasts compared with notebook 05: "
      f"{mismatches} mismatches ({secs / len(days):.2f} s per forecast)")
assert mismatches == 0
""")

md(r"""
**Result:** the deployed forecast reproduces the evaluated alerts exactly, on every day of the
season, even though it rebuilds everything from data cut off at day D. Any difference would mean
the evaluation used information the live system cannot have.

## 3 · Example bulletins
The eve of Diwali 2025, and the day before the largest miss of the test year (Arumbakkam,
35 → 206 µg/m³ on 3 Dec).
""")

code(r"""
for day in ["2025-10-19", "2025-12-02"]:
    fc = D.forecast(day, model, meta, clean, weather)
    print(D.bulletin_text(fc, meta))
    actual = ev[ev.date == day].set_index("station")["pm25_next"].round(0)
    print("What actually happened tomorrow (µg/m³):", actual.to_dict(), "\n" + "-" * 70)
""")

md(r"""
* **Diwali eve:** the festival flag pushes every station's probability above 0.9. The rule alerts
  only where today was already above 55. Both stations with a reading on Diwali day exceeded the
  limit (Arumbakkam 98, Perungudi 172), so both alerts were right. Manali and Kodungaiyur have no
  reading for 20 Oct, so their alerts cannot be checked.
* **2 Dec:** nothing in the day's data pointed to the jump at Arumbakkam, so neither method alerted.
  This is the onset weakness documented in the model card.

## 4 · Does it refuse to guess?
A forecast system should say when it cannot forecast, not quietly fill in. Two cases where it
must refuse, and one where it can still forecast but must say what it filled in:
""")

code(r"""
cases = {
    "no CPCB data yet (beyond the files)": "2026-12-01",
    "weather archive not caught up (lags ~1 week)": "2026-11-14",
}
for what, day in cases.items():
    try:
        D.forecast(day, model, meta, clean, weather)
        print(f"{what}: produced a forecast (unexpected)")
    except ValueError as e:
        print(f"{what}: refused -> {e}")

# a station with no reading today is still forecast, from the other stations, and flagged
c2 = clean.copy()
c2.loc[(c2.station == "Manali") & (c2.date == "2025-11-20"), "pm25"] = np.nan
fc = D.forecast("2025-11-20", model, meta, c2, weather)
fc[["station", "pm25_today", "pm25_today_missing", "probability_bad_tomorrow", "model_alert"]]
""")

md(r"""
The 2026 dates above are only used to check that the script **refuses**; no 2026 forecast is
made or scored here.

## 5 · Live mode (not run in this notebook)
`python scripts/03_forecast_tomorrow.py --date D --live` fills day D's weather from the
Open-Meteo **forecast** service when the archive does not have it yet. It is kept out of this
notebook because it needs the internet and returns slightly different numbers each time, which
would break reproducibility. The weather it returns comes from a different source than the ERA5
archive used in training, which is listed as a limitation in the model card.

## Summary
| Check | Result |
|---|---|
| Saved model = evaluated model | identical probabilities on all 1,210 test rows; byte-identical file on re-save |
| Daily forecast = evaluation | 0 mismatches over 91 days (348 station-days with a known outcome), with data cut off at each day |
| Refuses when inputs are missing | yes: no CPCB data, or weather not yet available |
| Missing station reading | forecast from the other stations and flagged |

**Next:** stage 9 (Monitor) runs the daily forecast through 2026 as if live, and tracks recall,
precision and the alert rate month by month.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("06_deployment.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
