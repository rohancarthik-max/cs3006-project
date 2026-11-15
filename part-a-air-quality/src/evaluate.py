"""Stage 7: evaluate the FROZEN choice (reports/model_choice.json) on a given year.

Nothing here chooses anything: the model settings, training years and alert
thresholds are read from the choice file. The same functions run on 2024
(to build and check the analysis) and then, once, on 2025 (the test)."""
import numpy as np
import pandas as pd

from . import config as C
from . import models as M
from .dataset import festival_flags
from .features import FEATURES, NEXT_DAY_FEATURES

# India's AQI sub-index bands for 24-hour PM2.5 (CPCB), used to grade how bad a day was
AQI_BANDS = [(60, 90, "Moderate (61-90)"), (90, 120, "Poor (91-120)"),
             (120, 250, "Very poor (121-250)"), (250, np.inf, "Severe (>250)")]
METHODS = ["logreg", "pm25_score", "persistence", "never"]
LABELS = {"logreg": "Logistic regression (primary)", "pm25_score": "Today's PM2.5 > 55 (reference)",
          "persistence": "Persistence (today > 60)", "never": "Never alert"}


def frozen_models(train_rows, choice):
    """Refit exactly the frozen primary model (deterministic), plus the
    upper-bound variant: same settings, plus D+1 weather."""
    p = choice["primary"]
    assert p["trained_on_target_years"] == C.SPLIT_YEARS["train"], "choice expects 2021-23 training"
    primary = M.fit(p["family"], p["params"], train_rows)
    upper = M.fit(p["family"], p["params"], train_rows, FEATURES + NEXT_DAY_FEATURES)
    return primary, upper


def thresholds(choice):
    return {"logreg": choice["primary"]["threshold_probability"],
            "pm25_score": choice["reference"]["threshold_log_pm25"],
            "persistence": 1.0, "never": np.inf}


def scores_for(rows, primary):
    return M.baseline_scores(rows) | {"logreg": primary.predict_proba(rows[FEATURES])[:, 1]}


def headline(rows, scores, thr):
    """Frozen-threshold results plus each year's no-skill level."""
    y = rows["y"].to_numpy()
    t = M.score_table(y, scores, thr, probabilities=("logreg",)).loc[METHODS]
    t.insert(0, "base rate (no-skill PR-AUC)", y.mean())
    return t


def alerts_frame(rows, scores, thr, method="logreg"):
    """One row per station-day with the method's alert and the outcome."""
    d = rows[["date", "target_date", "station", "pm25_next", "y", "log_pm25"]].copy()
    d["pm25"] = np.exp(d["log_pm25"])                 # today's PM2.5 (NaN if missing)
    d["score"] = scores[method]
    d["alert"] = d["score"] >= thr[method]
    d["bad"] = d["y"] == 1
    d["outcome"] = np.select([d.alert & d.bad, ~d.alert & d.bad, d.alert & ~d.bad],
                             ["caught", "missed", "false alarm"], "correct no")
    d["day type"] = np.where(d["pm25"] > C.NAAQS_PM25_24H, "continuation (today > 60)",
                             "onset (today ≤ 60 or missing)")
    band = pd.Series("not bad", index=d.index)
    for lo, hi, name in AQI_BANDS:
        band[(d["pm25_next"] > lo) & (d["pm25_next"] <= hi)] = name
    d["tomorrow band"] = band
    d["festival tomorrow"] = festival_flags(d["target_date"]) != ""
    d["month"] = d["target_date"].dt.month
    return d


def recall_by(a, col):
    """Recall within groups of bad days: how many of each group's bad days were caught."""
    bad = a[a["bad"]]
    g = bad.groupby(col, observed=True)["alert"].agg(bad_days="size", caught="sum")
    g["missed"] = g["bad_days"] - g["caught"]
    g["recall"] = g["caught"] / g["bad_days"]
    return g


def false_alarm_profile(a):
    """How close to the limit were tomorrow's values on false-alarm days?"""
    fa = a[a["outcome"] == "false alarm"]
    bins = [0, 30, 45, 60]
    cut = pd.cut(fa["pm25_next"], bins, labels=["< 30 (clean)", "30-45", "45-60 (near miss)"])
    out = cut.value_counts().reindex(["< 30 (clean)", "30-45", "45-60 (near miss)"]).rename("false alarms")
    return out.to_frame().assign(share=lambda t: t["false alarms"] / max(len(fa), 1))


def reliability(y, p, edges=(0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)):
    """Mean predicted probability vs observed share of bad days, in fixed bins."""
    b = pd.cut(p, edges, include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y, "bin": b}).groupby("bin", observed=True).agg(
        rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return t


def prior_shift(p, old_rate, new_rate):
    """Textbook prior-shift correction: move the log-odds by the change in the
    base rate's log-odds. Used ONLY as an after-the-fact what-if."""
    logit = np.log(p / (1 - p))
    shift = np.log(new_rate / (1 - new_rate)) - np.log(old_rate / (1 - old_rate))
    return 1 / (1 + np.exp(-(logit + shift)))


def upper_bound(rows, primary, upper):
    """Main model vs the same model given tomorrow's ACTUAL weather. Compared by
    PR-AUC only (no threshold), so nothing is tuned on the year being scored."""
    return {"logreg": primary.predict_proba(rows[FEATURES])[:, 1],
            "with D+1 weather": upper.predict_proba(rows[FEATURES + NEXT_DAY_FEATURES])[:, 1]}
