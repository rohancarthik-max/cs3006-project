"""Stage 8: run the frozen model as a daily forecast.

* save_model(): fit the frozen primary model (reports/model_choice.json) once and
  store it in models/, with a metadata file describing exactly what it is.
* load_model(): load it back and check it matches the code's feature list.
* forecast(day): the alert bulletin for day+1, computed from data CUT OFF at
  `day`, so the forecast cannot see the future even by accident.

Nothing here chooses or tunes anything."""
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config as C
from . import models as M
from .dataset import build_frame, load_clean, load_weather
from .features import FEATURES, make_features, model_rows
from .plots import STATION_ORDER

MODEL_DIR = C.ROOT / "models"
MODEL_FILE = MODEL_DIR / "primary_logreg.joblib"
META_FILE = MODEL_DIR / "primary_logreg.json"
ALERT_DIR = C.ROOT / "reports" / "alerts"


def save_model():
    """Fit the frozen primary model on the training years and save it with metadata."""
    choice = M.load_choice()
    p = choice["primary"]
    train = model_rows(make_features(), "train")
    model = M.fit(p["family"], p["params"], train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    meta = {
        "model": "logistic regression (L2) with median imputation + standard scaling",
        "params": p["params"], "trained_on_target_years": p["trained_on_target_years"],
        "training_rows": int(len(train)), "training_bad_days": int(train["y"].sum()),
        "features": FEATURES, "threshold_probability": p["threshold_probability"],
        "reference_rule_ugm3": choice["reference"]["threshold_ugm3"],
        "limit_ugm3": C.NAAQS_PM25_24H, "scikit_learn": sklearn.__version__,
        "model_file_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "source_choice_file": "reports/model_choice.json",
    }
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return model, meta


def load_model():
    model = joblib.load(MODEL_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if meta["features"] != FEATURES:
        raise RuntimeError("Saved model was trained on a different feature list; run scripts/02_save_model.py")
    if meta["scikit_learn"] != sklearn.__version__:
        print(f"warning: model saved with scikit-learn {meta['scikit_learn']}, running {sklearn.__version__}")
    return model, meta


def forecast(day, model=None, meta=None, clean=None, weather=None):
    """Bulletin for day+1, issued at the end of `day`, from data up to `day` only.
    Returns one row per station. Raises ValueError if the inputs for `day` are not
    available, instead of guessing."""
    day = pd.Timestamp(day).normalize()
    if model is None:
        model, meta = load_model()
    clean = load_clean() if clean is None else clean
    weather = load_weather() if weather is None else weather

    # 1. cut everything off at day D: the forecast physically cannot see D+1
    clean = clean[clean["date"] <= day]
    weather = weather[weather["date"] <= day]
    if clean.empty or clean["date"].max() < day:
        raise ValueError(f"No CPCB data for {day.date()} (latest available: "
                         f"{clean['date'].max().date() if not clean.empty else 'none'}).")
    w_today = weather[weather["date"] == day]
    if len(w_today) < len(C.STATIONS):
        raise ValueError(f"No weather for {day.date()} (archive ends {weather['date'].max().date()}; "
                         "the Open-Meteo archive lags about a week; use live mode).")

    # 2. the same features as in training
    feats = make_features(build_frame(clean, weather), weather)
    today = feats[feats["date"] == day].copy()
    if today["pm25_missing"].all():
        raise ValueError(f"No station reported PM2.5 on {day.date()}; no forecast can be made.")

    # 3. the frozen model and the reference rule
    p = model.predict_proba(today[FEATURES])[:, 1]
    today_pm = np.exp(today["log_pm25"])
    # The rule scores a station by its own reading, or by the other stations' mean
    # when it has none: exactly the reference evaluated in stage 7 (M.baseline_scores).
    rule_score = M.baseline_scores(today)["pm25_score"]
    out = pd.DataFrame({
        "issued_end_of": day.date(), "forecast_for": (day + pd.Timedelta(days=1)).date(),
        "station": today["station"].to_numpy(),
        "pm25_today": today_pm.round(1).to_numpy(),
        "probability_bad_tomorrow": p.round(3),
        "model_alert": p >= meta["threshold_probability"],
        "rule_alert": rule_score >= M.load_choice()["reference"]["threshold_log_pm25"],
        "pm25_today_missing": today["pm25_missing"].astype(bool).to_numpy(),
    })
    order = {s: i for i, s in enumerate(STATION_ORDER)}          # north -> south, as in the figures
    return out.sort_values("station", key=lambda s: s.map(order)).reset_index(drop=True)


def bulletin_text(fc, meta):
    """Plain-text bulletin for printing."""
    lines = [f"Next-day PM2.5 alert for {fc['forecast_for'].iloc[0]} "
             f"(issued end of {fc['issued_end_of'].iloc[0]})",
             f"Alert = tomorrow's 24-hour PM2.5 likely above {meta['limit_ugm3']:.0f} µg/m³",
             "",
             f"{'Station':<13}{'Today':>8}{'P(bad)':>9}  {'Model alert':<12}{'Rule alert':<10}"]
    for r in fc.itertuples():
        today = "n/a" if r.pm25_today_missing else f"{r.pm25_today:.0f}"
        lines.append(f"{r.station:<13}{today:>8}{r.probability_bad_tomorrow:>9.2f}  "
                     f"{'YES' if r.model_alert else 'no':<12}{'YES' if r.rule_alert else 'no':<10}")
    lines += ["", f"Model alert: probability >= {meta['threshold_probability']:.3f} (logistic regression, "
                  f"trained {meta['trained_on_target_years'][0]}-{meta['trained_on_target_years'][-1]}).",
              f"Rule alert: today's PM2.5 > {meta['reference_rule_ugm3']:.0f} µg/m³.",
              "Probabilities run low (see model card); use the yes/no alert."]
    if fc["pm25_today_missing"].any():
        lines.append("Note: a station without a reading today is forecast (model and rule) from the other stations.")
    return "\n".join(lines)


def save_bulletin(fc):
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERT_DIR / f"alert_{fc['forecast_for'].iloc[0]}.csv"
    fc.to_csv(path, index=False, lineterminator="\n")
    return path
