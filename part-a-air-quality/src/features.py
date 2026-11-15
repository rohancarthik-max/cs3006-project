"""Stage 5: turn the modelling table (src/dataset.py) into model inputs.

Rule for every feature: it must be computable at the end of day D. Pollution and
weather come from day D or earlier. The only D+1 items are calendar facts known
in advance (is tomorrow a festival? which month is tomorrow?).

Also here:
* the forward-chaining cross-validation folds used for tuning (inside 2021-23),
* the preprocessing pipeline (median imputation + scaling), always fitted on
  training rows only because it sits inside the model pipeline,
* an automated leakage test: change everything after a cut-off date and check
  that no feature on or before that date moves."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .dataset import build_frame, festival_flags, load_clean, load_weather
from .plots import STATION_ORDER

STATION_COLS = [f"st_{s}" for s in STATION_ORDER]

# name -> (group, plain-English meaning, EDA evidence from notebook 02)
FEATURE_INFO = {
    **{c: ("station", f"1 if the row is {c[3:]}", "station medians differ (§1); one pooled model (§2)")
       for c in STATION_COLS},
    "log_pm25":        ("persistence", "log of today's PM2.5", "H2: 56% vs 3% (§4)"),
    "log_pm25_change": ("persistence", "log PM2.5 today minus yesterday (rising or falling?)", "onset days missed by persistence (§4)"),
    "log_pm25_roll3":  ("persistence", "mean log PM2.5 over the last 3 days", "lag correlations stay > 0.5 for 3 days (§4)"),
    "log_pm25_roll7":  ("persistence", "mean log PM2.5 over the last 7 days", "correlation fades over a week (§4)"),
    "pm25_missing":    ("persistence", "1 if today's PM2.5 is missing (then filled in)", "missing values are not random (§2)"),
    "city_log_pm25":   ("city-wide", "mean log PM2.5 today at the other stations", "stations correlate 0.63-0.93 (§5)"),
    "month_sin":       ("season", "month of tomorrow on a circle (sine part)", "H1: 14.4% vs 0.3% (§3)"),
    "month_cos":       ("season", "month of tomorrow on a circle (cosine part)", "H1 (§3)"),
    "dewpoint":        ("weather D", "dew point today (°C): moisture / air mass", "holds within season, r ≈ -0.46 (§7)"),
    "rh":              ("weather D", "relative humidity today (%)", "hidden by pooling: -0.47 in winter (§7)"),
    "log_rain":        ("weather D", "log(1 + rain today in mm)", "H3: rain day -> 26% lower (§6)"),
    "log_rain_3d":     ("weather D", "log(1 + rain over the last 3 days)", "wash-out lasts more than a day (§6)"),
    "pressure":        ("weather D", "surface pressure today (hPa)", "+0.41, +0.30 within winter (§7)"),
    "wind_u":          ("weather D", "west->east wind component today (m/s)", "(§6-7)"),
    "wind_v":          ("weather D", "south->north wind component today (m/s); negative = from the north", "-0.45 (§6-7)"),
    "wind_speed":      ("weather D", "mean wind speed today (m/s)", "sign differs by season (§7)"),
    "festival_tomorrow": ("calendar", "1 if tomorrow is Diwali or Bhogi (known in advance)", "79% vs 6% (§8)"),
    "festival_today":  ("calendar", "1 if today was Diwali or Bhogi (spill-over)", "Diwali spike lasts into next day (§8)"),
}
FEATURES = list(FEATURE_INFO)

WEATHER_MAP = {"w_dewpoint_mean": "dewpoint", "w_rh_mean": "rh", "w_pressure_hpa": "pressure",
               "w_wind_u": "wind_u", "w_wind_v": "wind_v", "w_wind_mean": "wind_speed"}
# Upper-bound experiment only: the same weather, but measured on D+1
# ("what if the weather forecast were perfect?"). Never part of the main model.
NEXT_DAY_FEATURES = [f"next_{v}" for v in list(WEATHER_MAP.values()) + ["log_rain"]]

META = ["date", "target_date", "station_id", "station", "split", "pm25_next", "y"]


def make_features(frame=None, weather=None):
    """Return META columns + FEATURES (+ NEXT_DAY_FEATURES, kept separate).
    `weather` is the full Open-Meteo table (starts 2020-12-01), used so that
    3-day rain totals at the start of 2021 can look back into December 2020."""
    weather = load_weather() if weather is None else weather
    df = (build_frame(weather=weather) if frame is None else frame).sort_values(["station_id", "date"]).copy()
    out = df[META].copy()

    for col, st in zip(STATION_COLS, STATION_ORDER):
        out[col] = (df["station"] == st).astype(float)

    lp = np.log(df["pm25"])
    g = lp.groupby(df["station_id"])
    out["log_pm25"] = lp
    out["log_pm25_change"] = lp - g.shift(1)
    # rolling windows end at day D (inclusive); min_periods tolerates short gaps
    out["log_pm25_roll3"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    out["log_pm25_roll7"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    out["pm25_missing"] = lp.isna().astype(float)

    # city-wide signal: mean of the OTHER stations on the same day D
    wide = lp.groupby([df["date"], df["station_id"]]).first().unstack()
    total, count = wide.sum(axis=1, min_count=1), wide.notna().sum(axis=1)
    own = wide.stack().rename("own")
    key = pd.MultiIndex.from_arrays([df["date"], df["station_id"]])
    own_v = own.reindex(key).to_numpy()
    tot_v, cnt_v = total.reindex(df["date"]).to_numpy(), count.reindex(df["date"]).to_numpy()
    others_sum = np.where(np.isnan(own_v), tot_v, tot_v - own_v)
    others_n = np.where(np.isnan(own_v), cnt_v, cnt_v - 1)
    out["city_log_pm25"] = np.where(others_n > 0, others_sum / np.maximum(others_n, 1), np.nan)

    m = df["target_date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)

    for src, name in WEATHER_MAP.items():
        out[name] = df[src]
    out["log_rain"] = np.log1p(df["w_rain_mm"])
    w = weather.sort_values(["station_id", "date"])
    rain3 = w.groupby("station_id")["w_rain_mm"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    rain3 = pd.Series(rain3.to_numpy(), index=pd.MultiIndex.from_frame(w[["station_id", "date"]]))
    out["log_rain_3d"] = np.log1p(rain3.reindex(pd.MultiIndex.from_frame(df[["station_id", "date"]])).to_numpy())

    out["festival_tomorrow"] = (festival_flags(df["target_date"]) != "").astype(float)
    out["festival_today"] = (festival_flags(df["date"]) != "").astype(float)

    # upper-bound only: D+1 weather (shift within station; calendar is complete)
    gs = df.groupby("station_id")
    for src, name in WEATHER_MAP.items():
        out[f"next_{name}"] = gs[src].shift(-1)
    out["next_log_rain"] = np.log1p(gs["w_rain_mm"].shift(-1))

    return out[META + FEATURES + NEXT_DAY_FEATURES].reset_index(drop=True)


def model_rows(feats, split):
    """Labelled rows of one split (rows without a target cannot be trained or scored)."""
    names = [split] if isinstance(split, str) else list(split)
    return feats[feats["split"].isin(names) & feats["y"].notna()].reset_index(drop=True)


def cv_folds(train_rows):
    """Forward-chaining folds inside the training years, by year of the target day:
    fold 1 = train 2021 -> validate 2022; fold 2 = train 2021-22 -> validate 2023.
    Returns a list of (train_positions, validation_positions), usable as `cv=`
    in scikit-learn. Never trains on a later year than it validates on."""
    year = train_rows["target_date"].dt.year.to_numpy()
    years = sorted(set(year))
    folds = []
    for i in range(1, len(years)):
        folds.append((np.where(year < years[i])[0], np.where(year == years[i])[0]))
    return folds


def preprocessor(scale=True):
    """Median imputation (+ standard scaling for linear models). Used as the first
    steps of each model pipeline, so in cross-validation it is re-fitted on each
    fold's training rows and never sees validation or test rows."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def leakage_check(cutoffs=("2022-06-30", "2023-11-15", "2024-12-15", "2025-10-19"), seed=C.RANDOM_SEED):
    """Scramble every pollution and weather value AFTER each cut-off date, rebuild
    the features, and compare rows dated on or before the cut-off. If any feature
    on those rows changes, it was using future data. Returns one row per cut-off."""
    rng = np.random.default_rng(seed)
    clean, weather = load_clean(), load_weather()
    base = make_features(build_frame(clean, weather), weather)
    rows = []
    for cut in pd.to_datetime(list(cutoffs)):
        c2, w2 = clean.copy(), weather.copy()
        num_c = c2.select_dtypes("number").columns.drop("station_id")
        num_w = w2.select_dtypes("number").columns.drop("station_id")
        late_c, late_w = c2["date"] > cut, w2["date"] > cut
        c2.loc[late_c, num_c] = c2.loc[late_c, num_c] * rng.uniform(0.2, 5, size=(late_c.sum(), len(num_c)))
        w2.loc[late_w, num_w] = w2.loc[late_w, num_w] * rng.uniform(0.2, 5, size=(late_w.sum(), len(num_w)))
        new = make_features(build_frame(c2, w2), w2)
        past = base["date"] <= cut
        a, b = base.loc[past, FEATURES], new.loc[past, FEATURES]
        changed = ~((a == b) | (a.isna() & b.isna()))
        # the target itself (D+1) and the upper-bound D+1 weather SHOULD change on the cut-off day
        tgt = ~((base.loc[past, "pm25_next"] == new.loc[past, "pm25_next"])
                | (base.loc[past, "pm25_next"].isna() & new.loc[past, "pm25_next"].isna()))
        nxt = ~((base.loc[past, "next_dewpoint"] == new.loc[past, "next_dewpoint"])
                | (base.loc[past, "next_dewpoint"].isna() & new.loc[past, "next_dewpoint"].isna()))
        rows.append({"cut-off": cut.date(), "rows checked (date <= cut-off)": int(past.sum()),
                     "main features changed": int(changed.to_numpy().sum()),
                     "features that changed": ", ".join(changed.columns[changed.any()]) or "none",
                     "targets changed (expected > 0)": int(tgt.sum()),
                     "D+1 weather changed (expected > 0)": int(nxt.sum())})
    return pd.DataFrame(rows)
