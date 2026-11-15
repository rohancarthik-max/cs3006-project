"""Figures for the data-audit stage. One function per figure; every function
returns the saved path. Station colours are fixed (colour follows the station)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

STATION_ORDER = ["Manali", "Kodungaiyur", "Arumbakkam", "Perungudi"]   # north -> south
STATION_COLOR = {"Manali": "#2a78d6", "Kodungaiyur": "#eb6834",
                 "Arumbakkam": "#1baf7a", "Perungudi": "#6250d6"}
INK, INK2, MUTED, GRID = "#1f1f1e", "#55544f", "#8a897f", "#e6e5df"
EXCLUDED_FILL = "#d9d8d2"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade_exclusions(ax, station, col="pm25"):
    for ex in C.MANUAL_EXCLUSIONS:
        if C.STATIONS[ex["station_id"]]["name"] == station and col in ex["columns"]:
            ax.axvspan(pd.Timestamp(ex["start"]), pd.Timestamp(ex["end"]),
                       color=EXCLUDED_FILL, alpha=0.6, lw=0, zorder=0)
            ax.text(pd.Timestamp(ex["start"]), 0.97, " excluded", transform=ax.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=INK2)


def fig_availability(raw, clean):
    """Share of days per month with a usable PM2.5 value, before and after cleaning."""
    def monthly(df):
        w = df.pivot(index="date", columns="station", values="pm25")[STATION_ORDER]
        return w.notna().resample("MS").mean()
    before, after = monthly(raw), monthly(clean)
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f3ef", "#9ec5f4", "#3987e5", "#184f95"])
    for ax, m, title in zip(axes, (before, after), ("Raw CPCB files", "After sensor-health cleaning")):
        im = ax.imshow(m.T.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=[mdates.date2num(m.index[0]), mdates.date2num(m.index[-1] + pd.offsets.MonthEnd()),
                               len(STATION_ORDER) - 0.5, -0.5])
        ax.set_yticks(range(len(STATION_ORDER)), STATION_ORDER)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    axes[-1].xaxis_date()
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("share of days with PM2.5")
    cb.outline.set_visible(False)
    fig.suptitle("PM2.5 data availability by month", x=0.06, ha="left", fontweight="bold")
    return _save(fig, "fig01_pm25_availability.png")


def fig_peer_ratio(ratio):
    """Rolling 90-day median of station / median(other stations)."""
    lo, hi = C.PEER_RATIO_LIMITS
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, st in zip(axes, STATION_ORDER):
        ax.axhspan(lo, hi, color="#f1f0eb", lw=0, zorder=0)
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        _shade_exclusions(ax, st)
        ax.plot(ratio.index, ratio[st], color=STATION_COLOR[st], lw=1.6)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0.2, 5)
        ax.set_title(st, fontsize=10, color=INK)
    axes[0].text(1.0, 1.25, f"shaded band = allowed range {lo}-{hi}", transform=axes[0].transAxes,
                 ha="right", fontsize=8, color=INK2)
    fig.supylabel("station ÷ median of other stations\n(90-day rolling median, log scale)", color=INK2)
    fig.suptitle("Peer check: a healthy sensor tracks its neighbours (ratio near 1)",
                 x=0.06, ha="left", fontweight="bold")
    return _save(fig, "fig02_peer_ratio_drift.png")


def fig_pm_ratio(raw):
    """Monthly median PM2.5/PM10 - the second, independent signal for the exclusions."""
    r = raw.assign(r=raw["pm25"] / raw["pm10"]).pivot(index="date", columns="station", values="r")
    enough = r.notna().resample("MS").sum() >= 10          # need >= 10 days for a monthly median
    m = r.resample("MS").median().where(enough)[STATION_ORDER]
    fig, axes = plt.subplots(4, 1, figsize=(11, 6.5), sharex=True, sharey=True)
    others_med = {st: m.drop(columns=st).median(axis=1) for st in STATION_ORDER}
    for ax, st in zip(axes, STATION_ORDER):
        _shade_exclusions(ax, st)
        ax.plot(m.index, others_med[st], color=MUTED, lw=1.2, ls="--", label="other stations (median)")
        ax.plot(m.index, m[st], color=STATION_COLOR[st], lw=1.8, marker="o", ms=3.5, label=st)
        ax.set_title(st, fontsize=10)
        ax.set_ylim(0, 0.9)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.supylabel("PM2.5 / PM10 (monthly median)", color=INK2)
    fig.suptitle("Second signal: PM2.5 share of PM10 vs the other stations",
                 x=0.06, ha="left", fontweight="bold")
    return _save(fig, "fig03_pm25_pm10_ratio.png")


def fig_stuck_examples(raw, clean):
    """Two concrete failure modes caught by the rules."""
    cases = [("Arumbakkam", "ws", "2022-04-01", "2022-10-31", "Wind speed stuck at the sensor floor (0.30 m/s)"),
             ("Kodungaiyur", "ozone", "2025-06-01", "2026-11-14", "Ozone barely moving: flatline rule")]
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 5.2))
    for ax, (st, col, a, b, title) in zip(axes, cases):
        r = raw[(raw.station == st) & raw.date.between(a, b)].set_index("date")[col]
        c = clean[(clean.station == st) & clean.date.between(a, b)].set_index("date")[col]
        removed = r[r.notna() & c.isna()]
        ax.plot(r.index, r, color=MUTED, lw=1.0, label="raw")
        ax.plot(c.index, c, color=STATION_COLOR[st], lw=1.8, label="kept")
        ax.scatter(removed.index, removed, s=14, facecolors="white", edgecolors=INK, lw=0.8,
                   zorder=3, label=f"removed ({len(removed)} days)")
        ax.set_title(f"{st} - {title}", fontsize=10)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    fig.suptitle("Examples of frozen sensors removed by the health rules", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig04_stuck_sensor_examples.png")


def fig_clean_pm25(clean):
    """Cleaned daily PM2.5 per station with the NAAQS 24-hour limit."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, st in zip(axes, STATION_ORDER):
        s = clean[clean.station == st].set_index("date")["pm25"]
        ax.plot(s.index, s, color=STATION_COLOR[st], lw=0.9)
        ax.axhline(C.NAAQS_PM25_24H, color=INK, lw=0.9, ls="--")
        n = int((s > C.NAAQS_PM25_24H).sum())
        ax.set_title(f"{st}  ·  {n} days above limit", fontsize=10)
        ax.set_ylim(0, 220)
    fig.supylabel("daily PM2.5 (µg/m³)", color=INK2)
    fig.suptitle("Cleaned daily PM2.5, Chennai 2021-2026", x=0.06, y=1.0, ha="left", fontweight="bold")
    fig.text(0.06, 0.955, "dashed line = India NAAQS 24-hour limit, 60 µg/m³",
             ha="left", fontsize=9, color=INK2)
    return _save(fig, "fig05_clean_pm25_timeseries.png")
