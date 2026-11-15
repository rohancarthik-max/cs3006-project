# CS3006 Introduction to Data Science for Engineers: Project

Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram

**Team:** Rohan Karthik K (EC24B1015) · Vasanth M S (EC24B1059)

This repository holds both parts of the course project. Each part is self-contained in its own folder,
with its own README, report and instructions.

| | Part A: Data science project (10 marks) | Part B: Learning game (5 marks) |
|---|---|---|
| **Title** | Tomorrow's Air, Today: next-day PM2.5 exceedance alerts for Chennai | Mayor for a Year: correlation does not imply causation |
| **Folder** | [`part-a-air-quality/`](part-a-air-quality/) | [`part-b-mayor-game/`](part-b-mayor-game/) |
| **Report** | [Report (PDF)](part-a-air-quality/reports/CS3006_PM25_Chennai_Report.pdf) | [Report (PDF)](part-b-mayor-game/report/CS3006_MayorForAYear_Report.pdf) |
| **Try it live** | **[Interactive model demo](https://rohancarthik-max.github.io/cs3006-project/part-a-air-quality/demo/)** | **[Play in the browser](https://rohancarthik-max.github.io/cs3006-project/part-b-mayor-game/)** |
| **Slides** | [Slides (PPTX)](part-a-air-quality/reports/CS3006_PM25_Chennai_Report.pptx) | |
| **How it maps to the rubric** | [Part A README](part-a-air-quality/README.md#how-this-project-meets-the-marking-criteria) | [Part B README](part-b-mayor-game/README.md#how-this-meets-the-marking-criteria) |

## Part A: Tomorrow's Air, Today

Predicts, at the end of each day, whether tomorrow's 24-hour mean PM2.5 will exceed India's NAAQS limit
of 60 µg/m³ at four Central Pollution Control Board stations in Chennai, using official CPCB data
(2021-2026) and ERA5 weather. It follows the full data science pipeline: documented cleaning, exploratory
analysis with hypothesis tests, a leakage-tested feature set, baselines and two models tuned by
forward-chaining cross-validation, a single evaluation on the untouched 2025 test year, error analysis,
and a deployable daily forecast.

**Live demo:** <https://rohancarthik-max.github.io/cs3006-project/part-a-air-quality/demo/>: pick any day and
see each station's forecast, the reasons behind it, what-if changes, and a whole year scored live. The saved
model runs in the browser and matches the Python model on every one of 8,544 station-days.

```bash
cd part-a-air-quality
python -m venv .venv
.venv/Scripts/activate              # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements-lock.txt   # Python 3.12+
python run_all.py                   # rebuilds everything from the raw data (about 2 minutes)
python scripts/03_forecast_tomorrow.py --date 2025-10-19   # demo: the alert for Diwali 2025
```

## Part B: Mayor for a Year

A browser game in which the player, as mayor of a small Indian town, must decide which of several
correlated factors actually causes a problem. Four levels teach confounding, reverse causation,
Simpson's paradox and randomised experiments; a hidden structural causal model generates fresh data
every game.

* **Play online:** <https://rohancarthik-max.github.io/cs3006-project/part-b-mayor-game/>
  (also available at <https://rohancarthik-max.github.io/mayor-game/>)
* **Play offline:** open `part-b-mayor-game/index.html` in a browser. No installation needed.
