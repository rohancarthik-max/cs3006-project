"""Rebuild the whole project from the raw data, in pipeline order, with one command:

    python run_all.py

Steps (each stops the run if it fails):
  1. raw CPCB files -> cleaned dataset + audit log + figures 01-05
  2. generate and execute notebooks 01-05 (EDA, features, models, the 2025 evaluation)
  3. save the frozen model, then generate and execute notebook 06 (deployment checks)
  4. export the live demo page's data (demo/) and, if Node.js is installed, check it matches Python
  5. check that every expected output exists

The weather download (scripts/00_fetch_weather.py) needs the internet and is NOT run here:
its output, data/external/weather_daily.csv, is committed. Add --fetch-weather to re-download.
Every step is deterministic (RANDOM_SEED = 42), so a re-run reproduces the same numbers.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
NOTEBOOKS = [("make_notebook.py", [], "01_problem_and_data_audit.ipynb"),
             ("make_notebook_02.py", [], "02_eda.ipynb"),
             ("make_notebook_03.py", [], "03_features.ipynb"),
             ("make_notebook_04.py", [], "04_models.ipynb"),
             ("make_notebook_05.py", ["--final"], "05_evaluation.ipynb")]


def run(label, args):
    t = time.time()
    print(f"-> {label} ...", flush=True)
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        sys.exit(f"FAILED: {label}")
    print(f"   done in {time.time() - t:.0f} s", flush=True)


def notebook(maker, extra, name):
    run(f"generate {name}", [PY, f"notebooks/{maker}", *extra])
    run(f"execute  {name}", [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                             "--inplace", "--ExecutePreprocessor.timeout=1800", f"notebooks/{name}"])


def main():
    start = time.time()
    if "--fetch-weather" in sys.argv:
        run("download weather (internet)", [PY, "scripts/00_fetch_weather.py"])
    run("build clean dataset", [PY, "scripts/01_build_clean_dataset.py"])
    for maker, extra, name in NOTEBOOKS:
        notebook(maker, extra, name)
    run("save frozen model", [PY, "scripts/02_save_model.py"])
    notebook("make_notebook_06.py", [], "06_deployment.ipynb")
    run("export live demo data", [PY, "scripts/04_export_demo.py"])
    if shutil.which("node"):
        run("check demo page matches Python", ["node", "demo/test_demo.js"])
    else:
        print("-> skipped demo check (Node.js not installed)")

    expected = ([ROOT / "data/processed/daily_clean.csv", ROOT / "reports/model_choice.json",
                 ROOT / "models/primary_logreg.joblib", ROOT / "demo/data.js"]
                + [ROOT / "reports/figures" / f for f in [
                    "fig01_pm25_availability.png", "fig13_base_rate_by_year.png",
                    "fig15_feature_correlation.png", "fig19_model_interpretation.png",
                    "fig24_upper_bound.png"]])
    missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    if missing:
        sys.exit(f"FAILED: missing outputs {missing}")
    print(f"\nAll steps passed in {(time.time() - start) / 60:.1f} min. "
          f"Notebooks 01-06 are executed with their outputs saved.")


if __name__ == "__main__":
    main()
