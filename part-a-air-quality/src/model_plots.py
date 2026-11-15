"""Figures for stages 6-7 (models, evaluation): fig16-fig24."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config as _C
from .plots import GRID, INK, INK2, MUTED, _save

C_LIMIT = _C.NAAQS_PM25_24H

MODEL_STYLE = {
    "logreg":      {"label": "Logistic regression", "color": "#2f6db5", "ls": "-", "marker": "o"},
    "boosting":    {"label": "Boosted trees", "color": "#d9822b", "ls": "-", "marker": "D"},
    "pm25_score":  {"label": "Baseline: today's PM2.5", "color": "#55544f", "ls": "--", "marker": "s"},
    "persistence": {"label": "Baseline: persistence", "color": INK, "ls": ":", "marker": "X"},
}


def fig_cv_tuning(cv_tables, baseline_folds):
    """fig16: every setting tried, its two fold scores and their mean, vs the baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))       # own y-axis each: labels differ
    for ax, (family, table) in zip(axes, cv_tables.items()):
        st = MODEL_STYLE[family]
        t = table.reset_index(drop=True)
        labels = [", ".join(f"{k.rstrip('_')}={v}" for k, v in p.items()) for p in t["params"]]
        y = np.arange(len(t))
        ax.hlines(y, t["PR-AUC fold 1 (2022)"], t["PR-AUC fold 2 (2023)"], color=GRID, lw=3, zorder=0)
        ax.scatter(t["PR-AUC fold 1 (2022)"], y, s=30, facecolors="white", edgecolors=st["color"], lw=1.4,
                   label="fold 1 (2022)", zorder=2)
        ax.scatter(t["PR-AUC fold 2 (2023)"], y, s=30, color=st["color"], alpha=0.45, lw=0,
                   label="fold 2 (2023)", zorder=2)
        ax.scatter(t["mean PR-AUC"], y, s=70, color=st["color"], marker="|", lw=2.5, label="mean", zorder=3)
        ax.axvline(np.mean(baseline_folds), color=MODEL_STYLE["pm25_score"]["color"], ls="--", lw=1.2)
        ax.text(np.mean(baseline_folds), -0.55, " baseline mean", fontsize=8, color=INK2, va="bottom")
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("PR-AUC on the validation fold")
        ax.set_title(f"{st['label']} (best on top)", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Fig 16 · Tuning on forward-chaining folds: the folds disagree, so the grid stays small",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig16_cv_tuning.png")


def fig_pr_curves(y, scores, thresholds, persistence_point, base_rate, target_recall):
    """fig17: precision-recall curves on 2024 with each model's chosen alert point."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axhline(base_rate, color=MUTED, lw=1, ls="-.")
    ax.text(0.01, base_rate + 0.01, f"no-skill level = base rate {base_rate:.3f}", fontsize=8, color=INK2)
    ax.axvline(target_recall, color=MUTED, lw=0.9)
    ax.text(target_recall + 0.005, 0.03, f"target recall {target_recall}", fontsize=8, color=INK2)
    for name in ["pm25_score", "boosting", "logreg"]:
        st = MODEL_STYLE[name]
        p, r, t = precision_recall_curve(y, scores[name])
        ax.plot(r, p, color=st["color"], ls=st["ls"], lw=2, label=st["label"], drawstyle="steps-post")
        a = scores[name] >= thresholds[name]
        rc, pc = (a & (y == 1)).sum() / y.sum(), (a & (y == 1)).sum() / max(a.sum(), 1)
        ax.scatter([rc], [pc], s=70, color=st["color"], marker=st["marker"], edgecolors="white", lw=1.2, zorder=4)
    st = MODEL_STYLE["persistence"]
    ax.scatter([persistence_point[0]], [persistence_point[1]], s=80, color=st["color"], marker=st["marker"],
               zorder=4, label=st["label"] + " (one fixed point)")
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="recall: share of bad days caught",
           ylabel="precision: share of alerts that were right")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Fig 17 · Precision-recall on 2024 (markers = chosen alert threshold)", pad=10)
    return _save(fig, "fig17_pr_curves_2024.png")


def fig_differences(diffs):
    """fig18: forest plot of PR-AUC differences vs the baseline, with 95% block-bootstrap CIs.
    `diffs` rows: comparison, model, point, lo, hi."""
    d = diffs.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(d) + 1.4))
    y = np.arange(len(d))
    for i, row in d.iterrows():
        st = MODEL_STYLE[row["model"]]
        ax.hlines(i, row["lo"], row["hi"], color=st["color"], lw=2.2)
        ax.scatter(row["point"], i, s=60, color=st["color"], marker=st["marker"], zorder=3,
                   edgecolors="white", lw=1)
        ax.text(max(row["hi"], 0) + 0.008, i, f"{row['point']:+.3f}  [{row['lo']:+.3f}, {row['hi']:+.3f}]",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [f"{r.comparison}: {MODEL_STYLE[r.model]['label']}" for r in d.itertuples()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(d["lo"].min() - 0.03, d["hi"].max() + 0.2)
    ax.set_xlabel("PR-AUC of the model minus PR-AUC of the baseline (today's PM2.5)")
    ax.set_title("Fig 18 · Is a model really better than the baseline? (95% block-bootstrap intervals; "
                 "crossing 0 = not clear)", fontsize=10.5, pad=10)
    return _save(fig, "fig18_model_vs_baseline.png")


def fig_interpretation(coefs, perm_lr, perm_tree, labels=None, top=10):
    """fig19: what the models rely on."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    c = coefs.head(top)[::-1]
    names = lambda idx: [labels.get(i, i) if labels else i for i in idx]
    axes[0].barh(names(c.index), c.values, color=["#2f6db5" if v > 0 else "#9fb3cf" for v in c.values],
                 height=0.6)
    axes[0].axvline(0, color=INK, lw=0.8)
    axes[0].set(title="Logistic regression coefficients\n(per +1 standard deviation)", xlabel="change in log-odds")
    for ax, perm, fam in ((axes[1], perm_lr, "logreg"), (axes[2], perm_tree, "boosting")):
        p = perm.head(top)[::-1]
        ax.barh(names(p.index), p["mean drop in PR-AUC"], xerr=p["std"], color=MODEL_STYLE[fam]["color"],
                height=0.6, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        ax.set(title=f"{MODEL_STYLE[fam]['label']}: permutation importance\n(2024)",
               xlabel="drop in PR-AUC when the feature is shuffled")
        ax.set_axisbelow(True)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_axisbelow(True)
    fig.suptitle("Fig 19 · What the models rely on: today's PM2.5 first, then season, trend and northerly wind",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig19_model_interpretation.png")


# ---------------------------------------------------------------- stage 7 figures
YEAR_FILL = {2024: "#8fa9cc", 2025: "#4f73a6", 2026: "#26446e"}     # same shades as fig13/fig14


def fig_test_results(boot_by_year, base_by_year, labels):
    """fig20: PR-AUC with 95% block-bootstrap CI, per method and year, vs each year's no-skill level."""
    years = list(boot_by_year)
    methods = ["logreg", "pm25_score", "persistence"]
    fig, ax = plt.subplots(figsize=(9, 1.0 * 3 + 0.6 * len(years) + 0.8))
    off = np.linspace(-0.18, 0.18, len(years)) if len(years) > 1 else [0]
    for k, yr in enumerate(years):
        b = boot_by_year[yr]
        for i, m in enumerate(methods):
            row = b.loc[m]
            ax.hlines(i + off[k], row["2.5%"], row["97.5%"], color=YEAR_FILL[yr], lw=2.4)
            ax.scatter(row["PR-AUC (point)"], i + off[k], s=55, color=YEAR_FILL[yr],
                       marker=MODEL_STYLE[m]["marker"], edgecolors="white", lw=1, zorder=3,
                       label=str(yr) if i == 0 else None)
            ax.text(row["97.5%"] + 0.01, i + off[k], f"{row['PR-AUC (point)']:.2f}", va="center",
                    fontsize=8, color=INK2)
        ax.axvline(base_by_year[yr], color=YEAR_FILL[yr], ls=":", lw=1.4)
        ax.text(base_by_year[yr], -0.45 - 0.22 * k, f" no-skill {yr}: {base_by_year[yr]:.3f}",
                fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
    ax.set_ylim(len(methods) - 0.5, -0.5 - 0.25 * len(years))     # top room for the no-skill labels
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC (95% block-bootstrap interval)")
    ax.legend(title="year", loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Fig 20 · Ranking quality of the frozen methods, validation vs test", pad=22)
    return _save(fig, "fig20_test_results.png")


def fig_missed(recall_tables):
    """fig21: recall of the primary model by day type and by how bad tomorrow was."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    for ax, key, title in ((axes[0], "day type", "By day type"),
                           (axes[1], "tomorrow band", "By how bad tomorrow was (India AQI band)")):
        years = list(recall_tables)
        cats = list(dict.fromkeys(c for yr in years for c in recall_tables[yr][key].index))
        w = 0.8 / len(years)
        for k, yr in enumerate(years):
            t = recall_tables[yr][key].reindex(cats)
            x = np.arange(len(cats)) + (k - (len(years) - 1) / 2) * w
            ax.bar(x, t["recall"], width=w * 0.92, color=YEAR_FILL[yr], label=str(yr))
            for xi, (_, r) in zip(x, t.iterrows()):
                if pd.notna(r["recall"]):
                    ax.text(xi, r["recall"] + 0.02, f"{int(r['caught'])}/{int(r['bad_days'])}",
                            ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(cats)), [c.replace(" (", "\n(") for c in cats], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.7, color=MUTED, lw=0.9, ls="--")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.set_ylabel("share of bad days caught (recall)")
        ax.set_title(title, fontsize=10)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"year {x}" for x in l], loc="lower center", ncol=len(l), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Fig 21 · Which bad days does the primary model miss? (labels = caught / bad days; "
                 f"dashed = 70% target)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig21_missed_days.png")


def fig_timeline(a, start, end, stations):
    """fig22: tomorrow's PM2.5 and what the primary model did, Oct-Dec of the evaluated year."""
    fig, axes = plt.subplots(len(stations), 1, figsize=(11.5, 2.0 * len(stations) + 0.6), sharex=True, sharey=True)
    marks = {"caught": dict(marker="o", s=26, color="#2f6db5", label="bad day, alert given (caught)"),
             "missed": dict(marker="o", s=30, facecolors="white", edgecolors="#c4472d", lw=1.4,
                            label="bad day, no alert (missed)"),
             "false alarm": dict(marker="x", s=26, color=INK, lw=1.2, label="alert, day was fine (false alarm)")}
    for ax, st in zip(np.atleast_1d(axes), stations):
        d = a[(a.station == st) & a.target_date.between(start, end)].sort_values("target_date")
        ax.plot(d.target_date, d.pm25_next, color=MUTED, lw=1.1)
        ax.axhline(C_LIMIT, color=INK, ls="--", lw=0.8)
        for name, kw in marks.items():
            e = d[d.outcome == name]
            ax.scatter(e.target_date, e.pm25_next.fillna(C_LIMIT), zorder=3, **kw)
        ax.set_title(st, fontsize=9.5, loc="left")
    window = a[a.station.isin(stations) & a.target_date.between(start, end)]
    np.atleast_1d(axes)[0].set_ylim(0, window.pm25_next.max() * 1.08)   # shared y: show every peak
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.supylabel("PM2.5 on the forecast day (µg/m³)", color=INK2)
    fig.suptitle(f"Fig 22 · The primary model day by day, {start[:7]} to {end[:7]} (dashed = 60 limit)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, "fig22_timeline.png")


def fig_reliability(rel_by_name):
    """fig23: predicted probability vs observed share of bad days."""
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(0.62, 0.66, "perfect calibration", rotation=41, fontsize=8, color=INK2)
    styles = {}
    for i, (name, t) in enumerate(rel_by_name.items()):
        yr = int(name[:4])
        corrected = "corrected" in name
        ax.plot(t["mean_predicted"], t["observed_rate"], color=YEAR_FILL[yr], lw=2,
                ls=":" if corrected else "-", marker="s" if corrected else "o", ms=5, label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability (bin)",
           ylabel="observed share of bad days (bin)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Fig 23 · Can the probabilities be trusted?", pad=10)
    return _save(fig, "fig23_reliability.png")


def fig_upper_bound(ub_by_year):
    """fig24: PR-AUC of the main model vs the same model given tomorrow's actual weather."""
    rows = [(yr, name) for yr in ub_by_year for name in ["logreg", "with D+1 weather"]]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.3))
    ax.set_ylim(len(rows) - 0.5, -0.6)
    for i, (yr, name) in enumerate(rows):
        r = ub_by_year[yr].loc[name]
        c = YEAR_FILL[yr]
        ax.hlines(i, r["2.5%"], r["97.5%"], color=c, lw=2.4)
        ax.scatter(r["PR-AUC (point)"], i, s=60, color=c if name == "logreg" else "white",
                   edgecolors=c, lw=1.6, zorder=3)
        ax.text(r["97.5%"] + 0.01, i, f"{r['PR-AUC (point)']:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [f"{yr}: {'main model (weather of day D)' if n == 'logreg' else 'upper bound (+ actual weather of D+1)'}"
                                     for yr, n in rows], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC (95% block-bootstrap interval)")
    ax.set_title("Fig 24 · What would a perfect weather forecast add? (experiment, not a result)", pad=10)
    return _save(fig, "fig24_upper_bound.png")
