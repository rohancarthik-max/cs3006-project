"""Stage 4 (EDA & statistics): hypothesis-test helpers and figures 06-13.

Tests are kept to the course toolkit: chi-square test of independence, Welch's
t-test, z-based confidence intervals, Spearman correlation. Each helper returns a
small dict/DataFrame so the notebook can print it as a table.

A caveat that applies to every p-value here: consecutive days at one station are
not independent (PM2.5 today strongly predicts tomorrow). Textbook tests assume
independence, so they make p-values look smaller than they should be.
`n_effective` gives a rough correction used as a robustness check."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .plots import INK, INK2, MUTED, GRID, STATION_COLOR, STATION_ORDER, _save

LIMIT = C.NAAQS_PM25_24H
EXCEED_COLOR = "#c4472d"     # "exceeds tomorrow" class; the other class is neutral grey
CALM_COLOR = "#b9b8b0"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WINTER = [10, 11, 12, 1, 2]          # Oct-Feb: north-east monsoon + winter

WEATHER_LABELS = {
    "w_temp_mean": "temperature, mean (°C)", "w_temp_min": "temperature, min (°C)",
    "w_temp_max": "temperature, max (°C)", "w_rh_mean": "relative humidity (%)",
    "w_dewpoint_mean": "dew point (°C)", "w_rain_mm": "rain (mm/day)",
    "w_pressure_hpa": "surface pressure (hPa)", "w_wind_mean": "wind speed, mean (m/s)",
    "w_wind_max": "wind speed, max (m/s)", "w_wind_u": "wind, west→east part u (m/s)",
    "w_wind_v": "wind, south→north part v (m/s)", "w_blh_mean": "boundary layer, mean (m)",
    "w_blh_min": "boundary layer, min (m)",
}


# ---------------------------------------------------------------- statistics
def describe(df, col="pm25", by="station"):
    """Centre, spread and shape per group (the slide-24 checklist)."""
    g = df.groupby(by)[col]
    out = pd.DataFrame({
        "n": g.count(), "mean": g.mean(), "median": g.median(), "std": g.std(),
        "IQR": g.quantile(0.75) - g.quantile(0.25), "skew": g.skew(),
        "p95": g.quantile(0.95), "max": g.max()})
    return out.reindex([s for s in STATION_ORDER if s in out.index]).round(2)


def n_effective(n, r1):
    """Effective number of independent observations for a series whose lag-1
    autocorrelation is r1 (AR(1) approximation): n * (1 - r1) / (1 + r1)."""
    return n * (1 - r1) / (1 + r1)


def p_with_n_effective(test, ratio):
    """Re-compute a test's p-value as if it had only n_eff = ratio * n independent
    rows. A chi-square statistic grows in proportion to n and a t statistic with
    sqrt(n), so we shrink them by the same factor."""
    if "chi2" in test:
        return stats.chi2.sf(test["chi2"] * ratio, test["dof"])
    return 2 * stats.t.sf(abs(test["t"]) * np.sqrt(ratio), max(test["df"] * ratio, 1))


def partial_spearman(df, x, y, control):
    """Rank correlation of x and y after removing the straight-line effect of
    `control` from both (on ranks). Answers: is there a link left once we know `control`?"""
    r = df[[x, y, control]].dropna().rank()

    def resid(col):
        slope, icept = np.polyfit(r[control], r[col], 1)
        return r[col] - (slope * r[control] + icept)
    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def two_proportions(y, group, names=("group A", "group B")):
    """Compare P(y = 1) between group == True and group == False.
    Returns both rates, their difference with a 95% z-interval, and the
    chi-square test of independence on the 2x2 table."""
    m = y.notna() & group.notna()
    y, group = y[m].astype(int), group[m].astype(bool)
    a, b = y[group], y[~group]
    p1, p2 = a.mean(), b.mean()
    se = np.sqrt(p1 * (1 - p1) / len(a) + p2 * (1 - p2) / len(b))
    table = pd.crosstab(group, y)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"n " + names[0]: len(a), "rate " + names[0]: round(p1, 4),
            "n " + names[1]: len(b), "rate " + names[1]: round(p2, 4),
            "difference": round(p1 - p2, 4),
            "95% CI": (round(p1 - p2 - 1.96 * se, 4), round(p1 - p2 + 1.96 * se, 4)),
            "chi2": round(chi2, 1), "dof": dof, "p_value": p}


def chi2_independence(y, category):
    """Chi-square test: does P(y = 1) depend on a category with k levels?"""
    m = y.notna() & category.notna()
    table = pd.crosstab(category[m], y[m].astype(int))
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"table": table, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
            "min_expected": round(expected.min(), 1)}


def welch(x, y, names=("x", "y")):
    """Welch's two-sample t-test (no equal-variance assumption) with a 95% CI for
    the difference in means, using the Welch-Satterthwaite degrees of freedom."""
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    t, p = stats.ttest_ind(x, y, equal_var=False)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    diff, se = x.mean() - y.mean(), np.sqrt(vx + vy)
    half = stats.t.ppf(0.975, df) * se
    return {"n " + names[0]: len(x), "mean " + names[0]: round(x.mean(), 3),
            "n " + names[1]: len(y), "mean " + names[1]: round(y.mean(), 3),
            "difference": round(diff, 3), "95% CI": (round(diff - half, 3), round(diff + half, 3)),
            "t": round(t, 2), "df": round(df, 1), "p_value": p}


def spearman_table(df, cols, target="pm25_next", seasons=None):
    """Spearman correlation of each column with the target, for the whole year and
    optionally within seasons ({name: [months]}). Rank correlation is used because
    PM2.5 is skewed and the relationships need not be straight lines."""
    seasons = {"all year": list(range(1, 13)), **(seasons or {})}
    rows = []
    for col in cols:
        row = {"feature": col}
        for name, months in seasons.items():
            d = df[df["date"].dt.month.isin(months)][[col, target]].dropna()
            r, p = stats.spearmanr(d[col], d[target])
            row[name] = round(r, 3)
            row[f"p ({name})"] = p
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def autocorr(df, col="pm25", max_lag=14):
    """Autocorrelation of log PM2.5 at lags 1..max_lag days, per station."""
    out = {}
    for st, g in df.groupby("station"):
        s = np.log(g.set_index("date")[col])
        out[st] = [s.corr(s.shift(k)) for k in range(1, max_lag + 1)]
    return pd.DataFrame(out, index=pd.Index(range(1, max_lag + 1), name="lag (days)"))[
        [s for s in STATION_ORDER if s in out]]


# ---------------------------------------------------------------- figures
def _station_label(ax, text):
    ax.set_title(text, fontsize=10, color=INK)


def fig_distribution(tr):
    """fig06: raw vs log PM2.5, and per-station box plots."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    pm = tr["pm25"].dropna()
    axes[0].hist(pm, bins=60, color="#7a99c2", edgecolor="white", linewidth=0.4)
    axes[0].axvline(LIMIT, color=INK, ls="--", lw=1)
    axes[0].text(LIMIT + 3, axes[0].get_ylim()[1] * 0.9, "limit 60", fontsize=8, color=INK2)
    axes[0].set(title="PM2.5 is right-skewed", xlabel="daily PM2.5 (µg/m³)", ylabel="station-days")
    axes[1].hist(np.log(pm), bins=60, color="#7a99c2", edgecolor="white", linewidth=0.4)
    axes[1].axvline(np.log(LIMIT), color=INK, ls="--", lw=1)
    axes[1].set(title="log(PM2.5) is close to symmetric", xlabel="log of daily PM2.5", ylabel="")
    ticks = [5, 10, 20, 40, 60, 100, 200]
    sec = axes[1].secondary_xaxis("top", functions=(np.exp, lambda v: np.log(np.clip(v, 1e-9, None))))
    sec.set_xticks(ticks, [str(t) for t in ticks], fontsize=7, color=MUTED)
    data = [tr.loc[tr.station == s, "pm25"].dropna() for s in STATION_ORDER]
    bp = axes[2].boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                         flierprops={"marker": "o", "ms": 2.5, "mfc": "none", "mec": MUTED, "alpha": 0.6},
                         medianprops={"color": INK, "lw": 1.4})
    for patch, s in zip(bp["boxes"], STATION_ORDER):
        patch.set(facecolor=STATION_COLOR[s], alpha=0.35, edgecolor=STATION_COLOR[s])
    axes[2].set_yticks(range(1, 5), [f"{s} (n={len(d)})" for s, d in zip(STATION_ORDER, data)])
    axes[2].invert_yaxis()
    axes[2].axvline(LIMIT, color=INK, ls="--", lw=1)
    axes[2].set(title="By station (train 2021-23)", xlabel="daily PM2.5 (µg/m³)")
    fig.suptitle("Fig 6 · Distribution of daily PM2.5, training years", x=0.02,
                 ha="left", fontweight="bold")
    return _save(fig, "fig06_pm25_distribution.png")


def fig_seasonality(tr):
    """fig07: P(tomorrow exceeds) by month and station, as a heat map."""
    t = tr.dropna(subset=["y"]).assign(month=lambda d: d.target_date.dt.month)
    rate = t.pivot_table(index="station", columns="month", values="y", aggfunc="mean")
    rate = rate.reindex(STATION_ORDER)
    rate.loc["All stations"] = t.groupby("month")["y"].mean()
    n = t.pivot_table(index="station", columns="month", values="y", aggfunc="count").reindex(STATION_ORDER)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f4f3ef", "#f2c9b8", "#e0805f", "#b13a22", "#6e1d0f"])
    im = ax.imshow(rate.values, cmap=cmap, vmin=0, vmax=0.4, aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(12):
            v = rate.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            few = i < len(STATION_ORDER) and n.values[i, j] < 20
            ax.text(j, i, f"{v:.0%}" + ("*" if few else ""), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.22 else INK)
    ax.axhline(len(STATION_ORDER) - 0.5, color="white", lw=3)
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(rate)), rate.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=plt.matplotlib.ticker.PercentFormatter(1))
    cb.outline.set_visible(False)
    cb.set_label("share of days over 60 µg/m³")
    ax.set_title("Fig 7 · Exceedance days cluster in Oct-Jan (month of the target day, train 2021-23)",
                 fontsize=11, pad=10)
    fig.text(0.06, -0.04, "* fewer than 20 labelled days in that cell (mostly Kodungaiyur, after the drift exclusion)",
             fontsize=8, color=INK2)
    return _save(fig, "fig07_seasonality_heatmap.png")


def fig_persistence(tr, acf):
    """fig08: today vs tomorrow scatter (quadrants = the persistence baseline) and ACF."""
    t = tr.dropna(subset=["pm25", "pm25_next"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    col = np.where(t.pm25_next > LIMIT, EXCEED_COLOR, CALM_COLOR)
    ax.scatter(t.pm25, t.pm25_next, s=7, c=col, alpha=0.55, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [5, 10, 20, 40, 60, 100, 200]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.axvline(LIMIT, color=INK, ls="--", lw=0.9); ax.axhline(LIMIT, color=INK, ls="--", lw=0.9)
    a, b = t.pm25 > LIMIT, t.pm25_next > LIMIT
    quad = {"TP": (a & b).sum(), "FP": (a & ~b).sum(), "FN": (~a & b).sum(), "TN": (~a & ~b).sum()}
    kw = dict(fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
    ax.text(0.97, 0.97, f"both days over\n{quad['TP']} (persistence hit)", transform=ax.transAxes, ha="right", va="top", **kw)
    ax.text(0.97, 0.03, f"today over, tomorrow not\n{quad['FP']} (false alarm)", transform=ax.transAxes, ha="right", va="bottom", **kw)
    ax.text(0.03, 0.97, f"today under, tomorrow over\n{quad['FN']} (missed)", transform=ax.transAxes, ha="left", va="top", **kw)
    ax.set(xlabel="PM2.5 today, day D (µg/m³, log scale)", ylabel="PM2.5 tomorrow, day D+1",
           title="Today predicts tomorrow, but misses the onset days")
    ax = axes[1]
    for st in acf.columns:
        ax.plot(acf.index, acf[st], color=STATION_COLOR[st], lw=2, marker="o", ms=4, label=st)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set(xlabel="lag (days)", ylabel="correlation of log PM2.5 with k days earlier",
           title="Memory fades over about a week", ylim=(0, 1), xlim=(0.5, acf.index[-1] + 0.5))
    ax.set_xticks(acf.index)
    fig.suptitle("Fig 8 · Persistence: how much does today's PM2.5 say about tomorrow? (train 2021-23)",
                 x=0.06, y=1.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig08_persistence.png")


def fig_station_sync(tr):
    """fig09: stations move together (rank correlation, shared exceedance days)."""
    wide = tr.pivot(index="date", columns="station", values="pm25")[STATION_ORDER]
    corr = wide.corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f3ef", "#9ec5f4", "#3987e5", "#184f95"])
    ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if corr.values[i, j] > 0.7 else INK)
    ax.set_xticks(range(4), STATION_ORDER, rotation=20); ax.set_yticks(range(4), STATION_ORDER)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Spearman correlation of daily PM2.5")
    ax = axes[1]
    over = wide > LIMIT
    rows = []
    for st in STATION_ORDER:
        days = over.index[over[st]]
        others = over.drop(columns=st).loc[days]
        avail = wide.drop(columns=st).loc[days].notna().sum(axis=1)
        k = others.sum(axis=1)[avail >= 2]          # need at least two neighbours reporting
        rows.append({"station": st, "n": len(k), "share_with_neighbour": (k >= 1).mean()})
    r = pd.DataFrame(rows)
    ax.barh(r.station, r.share_with_neighbour, color=[STATION_COLOR[s] for s in r.station], height=0.55)
    for i, row in r.iterrows():
        ax.text(row.share_with_neighbour + 0.01, i, f"{row.share_with_neighbour:.0%}  (of {row.n} days)",
                va="center", fontsize=8.5, color=INK2)
    ax.invert_yaxis(); ax.set_xlim(0, 1.15); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set_title("When a station is over 60, is a neighbour over too?")
    ax.set_xlabel("share of its exceedance days where ≥ 1 other station also exceeds")
    fig.suptitle("Fig 9 · Bad-air days are mostly city-wide, not local (train 2021-23)",
                 x=0.06, y=1.03, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig09_station_sync.png")


def fig_weather_by_class(tr, cols):
    """fig10: day-D weather on days before an exceedance vs before a normal day."""
    t = tr.dropna(subset=["y"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, col in zip(axes.flat, cols):
        groups = [t.loc[t.y == 0, col].dropna(), t.loc[t.y == 1, col].dropna()]
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.4})
        for patch, c in zip(bp["boxes"], (CALM_COLOR, EXCEED_COLOR)):
            patch.set(facecolor=c, alpha=0.55, edgecolor=c)
        ax.set_xticks([1, 2], [f"tomorrow ≤ 60\n(n={len(groups[0])})", f"tomorrow > 60\n(n={len(groups[1])})"])
        ax.set_title(WEATHER_LABELS.get(col, col), fontsize=10)
        if col == "w_rain_mm":
            ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=0)
    fig.suptitle("Fig 10 · Today's weather before an exceedance day vs before a normal day "
                 "(box = middle 50%, line = median; train 2021-23)", x=0.06, y=1.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig10_weather_by_class.png")


def fig_simpson(sp):
    """fig11: Spearman r of each weather variable with tomorrow's PM2.5,
    whole year vs within each season. Large shrinkage = mostly a season effect."""
    sp = sp.sort_values("all year")
    y = np.arange(len(sp))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(sp) + 1.4))
    styles = [("all year", dict(marker="o", s=60, facecolors="white", edgecolors=INK, lw=1.6)),
              ("Oct-Feb", dict(marker="o", s=55, color="#3b5b8a")),
              ("Mar-Sep", dict(marker="D", s=40, color="#9fb3cf"))]
    for i, f in enumerate(sp.index):
        vals = [sp.loc[f, k] for k, _ in styles]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=0)
    for name, kw in styles:
        ax.scatter(sp[name], y, label=name, zorder=3, **kw)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [WEATHER_LABELS.get(f, f) for f in sp.index])
    ax.set_xlabel("Spearman correlation with tomorrow's PM2.5")
    ax.set_xlim(-0.7, 0.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="computed on", title_fontsize=8.5)
    ax.set_title("Fig 11 · Do weather links survive inside a season? (train 2021-23)", pad=10)
    return _save(fig, "fig11_simpson_season_check.png")


def fig_calendar(df_all, tr):
    """fig12: festival spikes (Diwali, Bhogi) and the weekday pattern."""
    wide = df_all.pivot(index="date", columns="station", values="pm25")
    city = wide.mean(axis=1)                     # mean of whichever stations report
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    train_years = C.SPLIT_YEARS["train"]
    for ax, fest in zip(axes[:2], ["Diwali", "Bhogi"]):
        for d in C.FESTIVALS[fest]:
            d = pd.Timestamp(d)
            if d.year not in train_years:
                continue
            seg = city.reindex(pd.date_range(d - pd.Timedelta(days=4), d + pd.Timedelta(days=4)))
            ax.plot(range(-4, 5), seg.values, lw=1.8, marker="o", ms=3.5, color="#3b5b8a",
                    alpha=0.35 + 0.3 * (d.year - 2021), label=str(d.year))
        ax.axhline(LIMIT, color=INK, ls="--", lw=0.9)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(-4, 5))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.set(title=f"{fest} (day 0), city mean PM2.5", xlabel="days from festival", xlim=(-4.5, 4.5))
    axes[0].set_ylabel("µg/m³")
    t = tr.dropna(subset=["y"]).assign(wd=lambda d: d.target_date.dt.day_of_week)
    g = t.groupby("wd")["y"].agg(["mean", "count"])
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["count"])
    ax = axes[2]
    ax.bar(g.index, g["mean"], color="#9fb3cf", width=0.6)
    ax.errorbar(g.index, g["mean"], yerr=1.96 * se, fmt="none", ecolor=INK2, capsize=3, lw=1)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set(title="Exceedance rate by weekday (±95% CI)", xlabel="weekday of the target day")
    fig.suptitle("Fig 12 · Calendar effects: festival spikes are real; weekday differences are within noise (train 2021-23)",
                 x=0.06, y=1.03, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig12_calendar_effects.png")


def fig_base_rate(df):
    """fig13: share of exceedance days per target year, coloured by split."""
    t = df.dropna(subset=["y"]).assign(year=lambda d: d.target_date.dt.year)
    g = t.groupby("year").agg(rate=("y", "mean"), n=("y", "size"), split=("split", "first"))
    se = np.sqrt(g.rate * (1 - g.rate) / g.n)
    shade = {"train": "#c9d6e8", "val": "#8fa9cc", "test": "#4f73a6", "monitor": "#26446e"}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(g.index, g.rate, color=[shade[s] for s in g.split], width=0.62)
    ax.errorbar(g.index, g.rate, yerr=1.96 * se, fmt="none", ecolor=INK2, capsize=3, lw=1)
    for yr, row in g.iterrows():
        ax.text(yr, row.rate + 1.96 * se[yr] + 0.006, f"{row.rate:.1%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(g.index, [f"{y}\n{row.split}, n={row.n}" for y, row in g.iterrows()])
    ax.set_yticks(np.arange(0, 0.20, 0.04))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(g.rate + 1.96 * se) + 0.04)
    ax.set_ylabel("share of days over 60 µg/m³")
    ax.set_title("Fig 13 · The base rate roughly doubles in 2025-26 (bars ±95% CI)", pad=10)
    return _save(fig, "fig13_base_rate_by_year.png")


# ---------------------------------------------------------------- stage 5 figures
ROLE_COLOR = {"train": "#c9d6e8", "validate": "#8fa9cc", "threshold": "#8fa9cc", "test": "#4f73a6", "monitor": "#26446e"}


def fig_cv_timeline():
    """fig14: which years each step trains / validates / tests on."""
    rows = [("CV fold 1 (tuning)", {2021: "train", 2022: "validate"}),
            ("CV fold 2 (tuning)", {2021: "train", 2022: "train", 2023: "validate"}),
            ("Model choice + threshold", {2021: "train", 2022: "train", 2023: "train", 2024: "validate"}),
            ("Final test (once)", {2021: "train", 2022: "train", 2023: "train", 2024: "threshold", 2025: "test"}),
            ("Monitoring", {2026: "monitor"})]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, (name, years) in enumerate(rows):
        for yr in range(2021, 2027):
            role = years.get(yr)
            if role is None:
                ax.barh(i, 0.94, left=yr - 0.47, height=0.6, color="white", edgecolor=GRID, lw=1)
                continue
            ax.barh(i, 0.94, left=yr - 0.47, height=0.6, color=ROLE_COLOR[role], lw=0)
            ax.text(yr, i, role, ha="center", va="center", fontsize=8.5,
                    color="white" if role in ("test", "monitor") else INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xticks(range(2021, 2027), [str(y) for y in range(2021, 2027)])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Fig 14 · Every step trains on the past and checks on a later year (by year of the target day)", pad=10)
    fig.text(0.02, -0.04, "\"Final test\": the frozen model is trained on 2021-23 only; 2024 sets its alert "
             "threshold and is not used for training (decision recorded in stage 6).", fontsize=8, color=INK2)
    return _save(fig, "fig14_cv_timeline.png")


def fig_feature_corr(train_rows, features, labels=None):
    """fig15: correlation between features (training rows); only strong pairs are labelled."""
    c = train_rows[features].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#b13a22", "#e0a088", "#efeeea", "#8fb0dc", "#1f5aa6"])
    cmap.set_bad("#d9d8d2")          # undefined, e.g. the missing-flag vs the value it flags
    im = ax.imshow(c.values, cmap=cmap, vmin=-1, vmax=1)
    n = len(features)
    for i in range(n):
        for j in range(n):
            v = c.values[i, j]
            if i != j and abs(v) >= 0.7:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.8 else INK)
    names = [labels.get(f, f) if labels else f for f in features]
    ax.set_xticks(range(n), names, rotation=90, fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation (train 2021-23)")
    ax.set_title("Fig 15 · Feature correlations: numbers shown where |r| ≥ 0.7 (near-duplicates)", pad=10)
    return _save(fig, "fig15_feature_correlation.png")
