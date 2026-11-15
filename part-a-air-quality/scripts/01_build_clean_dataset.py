"""Stage 2-3 of the pipeline: load raw CPCB files -> sensor-health cleaning ->
data/processed/daily_clean.csv  (+ audit log and figures).

Run from the project root:   python scripts/01_build_clean_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C            # noqa: E402
from src import plots                  # noqa: E402
from src.health import clean, peer_ratio  # noqa: E402
from src.load import load_raw          # noqa: E402


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_all = load_raw()
    print(f"Loaded {len(raw_all):,} station-days from {raw_all['station'].nunique()} stations")

    raw = raw_all[raw_all["date"] >= C.STUDY_START].reset_index(drop=True)
    print(f"Study window {C.STUDY_START} .. {raw['date'].max().date()}: {len(raw):,} station-days")

    cleaned, audit, drift, spikes, summary = clean(raw)
    ratio = peer_ratio(raw)  # computed on raw data so the figure shows the drift itself

    keep = ["date", "station_id", "station"] + C.POLLUTANTS + C.STATION_WEATHER
    cleaned[keep].to_csv(C.PROCESSED_DIR / "daily_clean.csv", index=False, lineterminator="\n")
    audit.to_csv(C.PROCESSED_DIR / "audit_log.csv", index=False, lineterminator="\n")
    drift.to_csv(C.PROCESSED_DIR / "peer_drift_flags.csv", index=False, lineterminator="\n")
    spikes.to_csv(C.PROCESSED_DIR / "isolated_spike_flags.csv", index=False, lineterminator="\n")
    summary.to_csv(C.PROCESSED_DIR / "cleaning_summary.csv", lineterminator="\n")
    pd.DataFrame(C.MANUAL_EXCLUSIONS).to_csv(C.PROCESSED_DIR / "manual_exclusions.csv", index=False, lineterminator="\n")
    pd.DataFrame(C.REVIEWED_AND_KEPT).to_csv(C.PROCESSED_DIR / "reviewed_and_kept.csv", index=False, lineterminator="\n")

    for f in (plots.fig_availability(raw, cleaned), plots.fig_peer_ratio(ratio),
              plots.fig_pm_ratio(raw), plots.fig_stuck_examples(raw, cleaned),
              plots.fig_clean_pm25(cleaned)):
        print("figure:", f.relative_to(C.ROOT))

    pm = summary.loc["pm25"]
    print(f"PM2.5 valid days: {pm.valid_before:,} raw -> {pm.valid_after:,} clean "
          f"({pm.removed:,} removed, {pm.removed / pm.valid_before:.1%})")
    print("Saved data/processed/daily_clean.csv")


if __name__ == "__main__":
    main()
