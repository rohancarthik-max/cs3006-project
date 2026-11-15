"""Stage 6: baselines, the two model families, tuning on the forward-chaining
folds, metrics, threshold choice and a block bootstrap for uncertainty.

Every model is a scikit-learn Pipeline whose first steps are the stage-5
preprocessor, so imputation/scaling are always fitted on training rows only."""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from . import config as C
from .features import FEATURES, cv_folds, preprocessor

TARGET_RECALL = 0.7
LOG_LIMIT = np.log(C.NAAQS_PM25_24H)

# ---------------------------------------------------------------- baselines
def baseline_scores(rows):
    """Scores for the three baselines. Higher = more likely to exceed tomorrow.
    * never:       always 0 -> never alerts
    * persistence: 1 if today > 60 else 0 (no reading today -> no alert)
    * pm25_score:  today's log PM2.5 (city-wide value if own is missing). The fair
                   baseline for probability models: same ranking idea as persistence,
                   but its threshold can be moved to trade recall for precision."""
    today = rows["log_pm25"]
    return {
        "never": np.zeros(len(rows)),
        "persistence": (today > LOG_LIMIT).astype(float).to_numpy(),
        "pm25_score": today.fillna(rows["city_log_pm25"]).fillna(-1.0).to_numpy(),
    }


# ---------------------------------------------------------------- models
def make_logreg(C_=1.0, class_weight=None):
    return Pipeline([("prep", preprocessor(scale=True)),
                     ("model", LogisticRegression(C=C_, class_weight=class_weight, max_iter=5000,
                                                  random_state=C.RANDOM_SEED))])


def make_boosting(max_depth=3, learning_rate=0.1, max_iter=100):
    # Trees do not need scaling. min_samples_leaf=40 stops a leaf being built
    # around a handful of the ~200 positive days. No early stopping: it would
    # carve a random internal validation set out of time-ordered data.
    return Pipeline([("prep", preprocessor(scale=False)),
                     ("model", HistGradientBoostingClassifier(
                         max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
                         min_samples_leaf=40, early_stopping=False, random_state=C.RANDOM_SEED))])


GRIDS = {
    "logreg": {"maker": make_logreg,
               "params": {"C_": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}},
    "boosting": {"maker": make_boosting,
                 "params": {"max_depth": [2, 3], "learning_rate": [0.05, 0.1], "max_iter": [100, 300]}},
}


def _combos(params):
    keys = list(params)
    return [dict(zip(keys, vals)) for vals in product(*params.values())]


def cv_search(train_rows, family, features=FEATURES):
    """PR-AUC of every setting in the family's grid on each forward-chaining fold."""
    spec, folds = GRIDS[family], cv_folds(train_rows)
    X, y = train_rows[features], train_rows["y"].to_numpy()
    out = []
    for params in _combos(spec["params"]):
        scores = []
        for fit_idx, val_idx in folds:
            m = spec["maker"](**params).fit(X.iloc[fit_idx], y[fit_idx])
            scores.append(average_precision_score(y[val_idx], m.predict_proba(X.iloc[val_idx])[:, 1]))
        out.append({**params, "params": params, "PR-AUC fold 1 (2022)": scores[0], "PR-AUC fold 2 (2023)": scores[1],
                    "mean PR-AUC": float(np.mean(scores))})
    return pd.DataFrame(out).sort_values("mean PR-AUC", ascending=False).reset_index(drop=True)


def best_params(cv_table):
    """Settings with the highest mean PR-AUC across the two folds (original types kept)."""
    return cv_table.iloc[0]["params"]


def fit(family, params, rows, features=FEATURES):
    return GRIDS[family]["maker"](**params).fit(rows[features], rows["y"].to_numpy())


# ---------------------------------------------------------------- metrics
def choose_threshold(y, score, target_recall=TARGET_RECALL):
    """Highest-precision cut-off that still catches >= target_recall of bad days.
    Among equally precise cut-offs, the highest one (fewest alerts) is kept."""
    p, r, t = precision_recall_curve(y, score)
    p, r = p[:-1], r[:-1]                       # last point has no threshold
    ok = r >= target_recall
    if not ok.any():
        return np.nan
    best = np.flatnonzero(ok & (p == p[ok].max()))
    return float(t[best].max())


def alert_metrics(y, alert):
    return {"alerts": int(alert.sum()), "caught": int((alert & (y == 1)).sum()),
            "missed": int((~alert & (y == 1)).sum()), "false alarms": int((alert & (y == 0)).sum()),
            "recall": recall_score(y, alert, zero_division=0),
            "precision": precision_score(y, alert, zero_division=0),
            "F1": f1_score(y, alert, zero_division=0)}


def score_table(y, scores, thresholds, probabilities=()):
    """One row per model: ranking quality (PR-AUC, ROC-AUC), probability quality
    (Brier, only for models that output probabilities) and alert counts at its threshold."""
    rows = []
    for name, s in scores.items():
        alert = s >= thresholds[name] if np.isfinite(thresholds[name]) else np.zeros(len(s), bool)
        rows.append({"model": name,
                     "PR-AUC": average_precision_score(y, s) if s.std() > 0 else y.mean(),
                     "ROC-AUC": roc_auc_score(y, s) if s.std() > 0 else 0.5,
                     "Brier": brier_score_loss(y, s) if name in probabilities else np.nan,
                     "threshold": thresholds[name], **alert_metrics(y, alert)})
    return pd.DataFrame(rows).set_index("model")


def block_bootstrap(rows, scores, n_boot=2000, seed=C.RANDOM_SEED, pairs=()):
    """95% intervals for PR-AUC, resampling whole city-wide ISO weeks (all stations'
    days in that week together). Neighbouring days are correlated, and so are the four
    stations on the same day (Spearman 0.63-0.93), so a week of the whole city is the
    unit that is roughly independent. (Correction after review: an earlier version used
    station-weeks, which treated same-day stations as independent and gave intervals
    about 1.5x too narrow.) `pairs` = [(a, b), ...] also gets the interval of
    PR-AUC(a) - PR-AUC(b): if it excludes 0, a is really better."""
    rng = np.random.default_rng(seed)
    iso = rows["target_date"].dt.isocalendar()
    block = (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()
    ids, inverse = np.unique(block, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    y = rows["y"].to_numpy()
    draws = {name: [] for name in scores} | {f"{a} - {b}": [] for a, b in pairs}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        if y[idx].sum() == 0:
            continue
        ap = {name: average_precision_score(y[idx], s[idx]) for name, s in scores.items()}
        for name in scores:
            draws[name].append(ap[name])
        for a, b in pairs:
            draws[f"{a} - {b}"].append(ap[a] - ap[b])
    out = pd.DataFrame({k: {"PR-AUC (point)": (average_precision_score(y, scores[k]) if k in scores
                                               else average_precision_score(y, scores[k.split(" - ")[0]])
                                               - average_precision_score(y, scores[k.split(" - ")[1]])),
                            "2.5%": np.percentile(v, 2.5), "97.5%": np.percentile(v, 97.5)}
                        for k, v in draws.items()}).T
    out.attrs["blocks"] = len(ids)
    return out


def logreg_coefficients(model, features=FEATURES):
    """Coefficients on the standardised scale: change in log-odds for +1 standard
    deviation of the feature (with the other features held fixed)."""
    coef = model.named_steps["model"].coef_[0]
    return pd.Series(coef, index=features).sort_values(key=np.abs, ascending=False)


def perm_importance(model, rows, features=FEATURES, n_repeats=20):
    """Drop in PR-AUC when one feature's values are shuffled (on the given rows)."""
    r = permutation_importance(model, rows[features], rows["y"].to_numpy(),
                               scoring="average_precision", n_repeats=n_repeats,
                               random_state=C.RANDOM_SEED)
    return pd.DataFrame({"mean drop in PR-AUC": r.importances_mean, "std": r.importances_std},
                        index=features).sort_values("mean drop in PR-AUC", ascending=False)


# ---------------------------------------------------------------- honest pooled comparison
def out_of_sample(train_rows, val_rows, params, families=("logreg", "boosting"), features=FEATURES):
    """Predictions that were never trained on their own year: the two CV folds
    (2022 from a 2021 model, 2023 from a 2021-22 model) plus 2024 from a 2021-23
    model. Pooling them triples the number of bad days available for comparison.
    (The settings were chosen on the same CV folds, so the 2022-23 part is very
    slightly optimistic for the models; 2024 is fully clean.)"""
    parts = []
    for fit_idx, val_idx in cv_folds(train_rows):
        d = train_rows.iloc[val_idx].copy()
        for fam in families:
            d[fam] = fit(fam, params[fam], train_rows.iloc[fit_idx], features).predict_proba(d[features])[:, 1]
        parts.append(d)
    d = val_rows.copy()
    for fam in families:
        d[fam] = fit(fam, params[fam], train_rows, features).predict_proba(d[features])[:, 1]
    parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    for name, s in baseline_scores(out).items():
        out[name] = s
    return out


# ---------------------------------------------------------------- frozen choice for stage 7
CHOICE_FILE = C.ROOT / "reports" / "model_choice.json"


def save_choice(choice):
    import json
    CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHOICE_FILE.write_text(json.dumps(choice, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    return CHOICE_FILE


def load_choice():
    import json
    return json.loads(CHOICE_FILE.read_text(encoding="utf-8"))
