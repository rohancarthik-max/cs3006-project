# Problem statement (pipeline stage 1)

## In one sentence

> To let Chennai residents and health officials act a day early, predict for each of
> four CPCB monitoring stations whether **tomorrow's 24-hour average PM2.5 will exceed
> India's NAAQS limit of 60 µg/m³**, issued at the end of today, as a **probability**.
> Judge it by recall, precision and PR-AUC on unseen years against a
> "tomorrow = today" persistence baseline, using only public data available by then.

## The seven parts

| Part | Our answer |
|---|---|
| **Decision** | A public-health advisory for the next day: schools move outdoor activity indoors, people with asthma/heart disease limit time outside, the pollution board can step up checks on local sources. |
| **Target** | `exceed_tomorrow = 1` if the station's daily mean PM2.5 on day D+1 is **> 60 µg/m³** (India NAAQS, 24-hour standard), else 0. |
| **Unit** | One **station-day**. Four stations: Manali and Kodungaiyur (north, industrial), Arumbakkam (central), Perungudi (south). |
| **Output** | A probability in [0, 1], turned into a yes/no alert with a threshold chosen on the validation year. |
| **Horizon** | One day ahead. Made at the end of day D, using only data up to and including day D. |
| **Metric** | Recall and precision on the "exceed" class, F1, and PR-AUC. Accuracy is *not* used: about 94% of training days are below the limit, so "always no" is already ~94% accurate. **Baselines:** (1) always "no", (2) persistence: tomorrow exceeds if today did, (3) today's PM2.5 with a cut-off tuned like a model's threshold (added in stage 6 as the fair comparison). |
| **Constraints** | Daily data only; CPCB sensors fail in ways that need cleaning (stuck values, drift); station rain gauges are unusable, so weather comes from a separate public source; there are few positive days, so the classes are imbalanced. |

## Why this threshold, not AQI?

The AQI that CPCB publishes is calculated from the same pollutant readings, so it carries
every sensor fault with it. At Royapuram (excluded) AQI sat at 63–66 for all of October
2026 because it was being driven by a frozen PM10 sensor. A fixed legal limit on a single
pollutant is simpler to explain and harder to fool.

## Which error is worse?

A **missed** bad-air day (false negative) means a child with asthma plays outside on a
day they shouldn't. A **false alarm** costs one indoor PE period. So recall matters more
than precision, but an alert that fires every day gets ignored. We will pick the
threshold on the validation year to reach a target recall (for example ≥ 0.7) and then
report the precision that comes with it.

## Data split (by time, never shuffled)

| Split | Period | Why |
|---|---|---|
| Train | 2021-01-01 .. 2023-12-31 | All four stations live from Nov-Dec 2020 |
| Validation | 2024 | Choose features, model settings and alert threshold |
| Test | 2025 | Touched once, at the end |
| Monitoring | 2026-01-01 .. 2026-11-14 | Act as if the model were deployed: does it still work? |

**Known risk (found during the audit):** the share of days over the limit is 5.4-7.0% in
the training years (8.4% in 2024) but about 13% in 2025-2026. The test year therefore has
roughly twice as many "yes" days as training. Probability calibration and the threshold may
shift; this is examined in the evaluation stage rather than hidden.

## What counts as leakage here

* Any value measured on day D+1 (including that day's weather) is unavailable when
  the forecast is issued. Weather enters only as day-D observations. A separate
  experiment that adds D+1 weather is labelled as an *upper bound* ("what a perfect
  weather forecast would add") and never reported as the main result.
* Scalers and imputers are fitted on training years only.
* Cleaning thresholds were set from physical reasoning, and the drift decisions were
  confirmed with a second, independent signal (PM2.5/PM10 ratio), not tuned on test results.
