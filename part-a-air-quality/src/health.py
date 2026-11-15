"""Sensor-health checks. Each rule looks for one specific way a CPCB sensor fails,
sets the bad readings to NaN, and writes one line per (rule, station, column) to an
audit log so the report can state exactly what was changed and why."""
import numpy as np
import pandas as pd

from . import config as C

CHECK_COLS = C.POLLUTANTS + ["temp", "rh", "ws", "wd", "rf", "sr"]


def _log(log, rule, station, column, n, reason):
    if n:
        log.append({"rule": rule, "station": station, "column": column,
                    "values_removed": int(n), "reason": reason})


def run_lengths_equal(s, tol):
    """For each position, length of the run of consecutive (near-)identical values
    it belongs to. NaNs break runs."""
    same = s.diff().abs().le(tol) & s.notna() & s.shift().notna()
    run_id = (~same).cumsum()
    lengths = s.groupby(run_id).transform("size")
    return lengths.where(s.notna(), 0)


def rule_out_of_range(df, log):
    for col, (lo, hi) in C.VALID_RANGE.items():
        if col not in df:
            continue
        for st, g in df.groupby("station"):
            bad = g[col].notna() & ~g[col].between(lo, hi)
            df.loc[bad[bad].index, col] = np.nan
            _log(log, "out_of_range", st, col, bad.sum(),
                 f"outside physically plausible range [{lo}, {hi}]")
    return df


def rule_pm25_above_pm10(df, log):
    bad = df["pm25"] > C.PM_RATIO_TOL * df["pm10"]
    for st, g in bad.groupby(df["station"]):
        _log(log, "pm25_gt_pm10", st, "pm25+pm10", g.sum(),
             "PM2.5 is part of PM10, so PM2.5 > PM10 is impossible; both readings dropped")
    df.loc[bad, ["pm25", "pm10"]] = np.nan
    return df


def rule_stuck_value(df, log):
    """Identical daily means on >= STUCK_MIN_RUN consecutive days.
    The first value of the run is kept; the repeats are removed."""
    for col in CHECK_COLS:
        if col in C.STUCK_SKIP:
            continue
        for st, g in df.groupby("station"):
            s = g[col]
            runs = run_lengths_equal(s, C.STUCK_TOL)
            first_of_run = ~(s.diff().abs().le(C.STUCK_TOL))
            bad = (runs >= C.STUCK_MIN_RUN) & ~first_of_run
            df.loc[bad[bad].index, col] = np.nan
            _log(log, "stuck_value", st, col, bad.sum(),
                 f">= {C.STUCK_MIN_RUN} consecutive identical daily values")
    return df


def rule_flatline(df, log):
    """Sensor still reports but barely moves: 14-day coefficient of variation < 2%.
    Real ambient concentrations vary far more than that from day to day."""
    for col in CHECK_COLS:
        if col in ("wd",):           # direction is circular; CV is meaningless
            continue
        for st, g in df.groupby("station"):
            s = g[col]
            roll = s.rolling(C.FLAT_WINDOW, min_periods=C.FLAT_WINDOW // 2, center=True)
            cv = roll.std() / roll.mean().abs()
            bad = s.notna() & cv.lt(C.FLAT_MAX_CV)
            df.loc[bad[bad].index, col] = np.nan
            _log(log, "flatline", st, col, bad.sum(),
                 f"{C.FLAT_WINDOW}-day coefficient of variation < {C.FLAT_MAX_CV:.0%}")
    return df


def peer_ratio(df, col="pm25", window=C.PEER_WINDOW):
    """Rolling median of (station / median of the other stations) for each station.
    All stations share the same city-wide weather, so a healthy sensor keeps this
    ratio fairly steady. A slow slide away from 1 means calibration drift."""
    wide = df.pivot(index="date", columns="station", values=col)
    out = {}
    for st in wide.columns:
        others = wide.drop(columns=st).median(axis=1)
        ratio = np.log(wide[st] / others)
        out[st] = np.exp(ratio.rolling(window, min_periods=window // 3, center=True).median())
    return pd.DataFrame(out)


def rule_peer_drift(df, log):
    """Flag (do not auto-remove) periods where the rolling peer ratio leaves the
    allowed band. Removal is a documented manual decision (config.MANUAL_EXCLUSIONS),
    because a ratio can also move for real reasons (e.g. local construction)."""
    lo, hi = C.PEER_RATIO_LIMITS
    ratio = peer_ratio(df)
    flags = []
    for st in ratio.columns:
        out = ~ratio[st].between(lo, hi) & ratio[st].notna()
        if out.any():
            run = (out != out.shift()).cumsum()[out]
            for _, seg in out[out].groupby(run):
                flags.append({"station": st, "start": seg.index.min().date(),
                              "end": seg.index.max().date(), "days": len(seg),
                              "min_ratio": round(ratio.loc[seg.index, st].min(), 2),
                              "max_ratio": round(ratio.loc[seg.index, st].max(), 2)})
    return pd.DataFrame(flags)


def isolated_spikes(df, col="pm25", factor=3.0, floor=100.0):
    """Days where ONE station reads > factor x the median of the others and > floor.
    Reported for review, not removed: such a day can be a real local event
    (a fire, Diwali crackers) or a glitch, and only context can tell."""
    wide = df.pivot(index="date", columns="station", values=col)
    others = pd.DataFrame({s: wide.drop(columns=s).median(axis=1) for s in wide})
    hit = (wide > factor * others) & (wide > floor)
    rows = []
    for st in wide.columns:
        for day in wide.index[hit[st]]:
            rows.append({"date": day.date(), "station": st, col: round(wide.at[day, st], 1),
                         "others_median": round(others.at[day, st], 1)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def rule_manual_exclusions(df, log):
    for ex in C.MANUAL_EXCLUSIONS:
        st = C.STATIONS[ex["station_id"]]["name"]
        sel = ((df["station_id"] == ex["station_id"])
               & df["date"].between(ex["start"], ex["end"]))
        for col in ex["columns"]:
            n = df.loc[sel, col].notna().sum()
            df.loc[sel, col] = np.nan
            _log(log, "manual_drift_exclusion", st, col, n, ex["reason"])
    return df


def clean(df):
    """Apply every rule in a fixed order.
    Returns (clean_df, audit_log, drift_flags, spike_flags, column_summary)."""
    df = df.copy()
    log = []
    before = df[CHECK_COLS].notna().sum()
    for rule in (rule_out_of_range, rule_pm25_above_pm10,
                 rule_stuck_value, rule_flatline):
        df = rule(df, log)
    drift = rule_peer_drift(df, log)
    df = rule_manual_exclusions(df, log)
    spikes = isolated_spikes(df)
    after = df[CHECK_COLS].notna().sum()
    audit = pd.DataFrame(log)
    summary = pd.DataFrame({"valid_before": before, "valid_after": after,
                            "removed": before - after})
    return df, audit, drift, spikes, summary
