# Model card: next-day PM2.5 alert, Chennai

A one-page summary of the deployed model: what it is for, how it was built and tested,
and where it should not be trusted. Format follows the "model cards" idea (Mitchell et al., 2019).

## What it does
At the end of day D, for each of four CPCB stations in Chennai (Manali, Kodungaiyur,
Arumbakkam, Perungudi), it estimates the chance that **day D+1's 24-hour mean PM2.5 exceeds
60 µg/m³** (India NAAQS) and raises a yes/no alert.

| | |
|---|---|
| Model file | `models/primary_logreg.joblib` (+ `primary_logreg.json` metadata) |
| Method | Logistic regression, L2 penalty (C = 0.1), no class weights; median imputation and standard scaling inside the pipeline |
| Inputs | 22 features known at the end of day D: PM2.5 today and its recent trend, city-wide PM2.5, month of D+1 (sine/cosine), day-D weather (dew point, humidity, rain today and 3-day, pressure, wind u/v/speed), festival flags (Diwali, Bhogi), station |
| Alert rule | probability ≥ **0.265** (chosen on 2024 for ≥ 70% recall) |
| Reference rule shipped alongside | alert if today's PM2.5 > **55 µg/m³** |
| Run it | `python scripts/03_forecast_tomorrow.py --date YYYY-MM-DD [--live]` |

## Intended use
* **Yes:** a next-day public-health advisory (schools moving outdoor activity indoors, people
  with asthma or heart disease limiting time outside) at these four stations.
* **No:** regulatory compliance decisions, forecasts for other parts of the city or other
  cities, other pollutants, or horizons beyond one day. Do not publish the probability
  itself without recalibration (see below).

## Data
* PM2.5 and other pollutants: CPCB daily data, cleaned with documented sensor-health rules
  (14.7% of PM2.5 days removed; two calibration-drift periods excluded).
* Weather: Open-Meteo historical archive (ERA5 reanalysis), because station rain gauges were broken.
* Training: target years 2021-2023, 3,506 labelled station-days, 215 bad days (6.1%).
* Threshold chosen on 2024 (902 rows, 76 bad days). Tested once on 2025 (1,210 rows, 157 bad days).

## Performance (2025 test, run once after the model was frozen)

| | Caught / bad days | False alarms | Recall | Precision | PR-AUC (95% CI) |
|---|---|---|---|---|---|
| This model | 129 / 157 | 58 | 0.82 | 0.69 | 0.79 (0.69-0.88) |
| Reference rule (today > 55) | 133 / 157 | 49 | 0.85 | 0.73 | 0.75 (0.59-0.89) |
| Persistence (today > 60) | 117 / 157 | 38 | 0.75 | 0.76 | 0.60 |

The model's lead over the reference rule in PR-AUC (+0.04, 95% interval −0.07 to +0.18, resampling city-wide weeks) is **not
statistically clear**, and at the alert threshold the rule did slightly better in 2025.

## Known weaknesses
* **Onset days:** when today is clean and tomorrow turns bad, only 19 of 40 were caught
  (2025). Sudden jumps (e.g. 35 → 206 µg/m³ at Arumbakkam on 3 Dec 2025) cannot be foreseen
  from today's data.
* **Probabilities run low** in the 0.3-0.6 range (L2 shrinkage plus a rising base rate): a
  shown "40%" was about 60% in reality. Use the yes/no alert, or recalibrate first.
* **Base-rate drift:** bad days rose from about 6% (2021-23) to about 13% (2025-26). The
  alert rate of a fixed threshold changes with it.
* **Few stations:** four stations do not represent all of Chennai.
* **Live weather mismatch:** in `--live` mode, day-D weather comes from the Open-Meteo
  forecast service, not the ERA5 archive used in training. Values can differ slightly; this
  mode is not validated.
* **Cleaned with hindsight:** two cleaning rules look a few days ahead and drift exclusions were
  decided viewing all years. Checked: with backward-only rules 3 of 6,833 values change and 2025 results
  are identical, but a live system must clean causally.
* **Missing readings:** a station without a PM2.5 reading today is forecast from the other
  stations' readings and the training median (flagged in the output).

## Fairness note
Recall by station in 2025: Kodungaiyur 94%, Manali 93%, Arumbakkam 77%, Perungudi 76%. The
most exposed northern industrial stations were served best, but counts are small (15-55 bad
days per station), so this should be monitored, not assumed.

## Maintenance
* **Monitor monthly:** recall, precision and the alert rate on the latest data (stage 9).
* **Recalibrate yearly** (e.g. Platt scaling on the most recent year) and review the threshold
  before each October-February season.
* **Retrain** if recall over a season drops below 0.7, if a station's sensor is replaced, or if
  features change. Re-run the leakage test (`notebooks/03_features.ipynb`) after any feature change.
* **Version pinning:** saved with scikit-learn 1.9.1 (`requirements-lock.txt`). Loading with a
  different version prints a warning.

## Provenance
Model choice frozen in `reports/model_choice.json` before the test year was evaluated; the evaluation
code was built and checked on 2024 only (notebook 05's build mode) and committed before the single 2025
run. Both commits are recorded in the development repository's history.
