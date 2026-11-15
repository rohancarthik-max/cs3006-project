"""Stage 8: fit the frozen primary model (reports/model_choice.json) on the training
years and save it to models/ with a metadata file. Deterministic: re-running gives
the same model.

Run from the project root:   python scripts/02_save_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deploy import META_FILE, MODEL_FILE, save_model  # noqa: E402

if __name__ == "__main__":
    model, meta = save_model()
    print("saved", MODEL_FILE.relative_to(MODEL_FILE.parents[1]))
    print("saved", META_FILE.relative_to(META_FILE.parents[1]))
    print(f"trained on {meta['trained_on_target_years']}, {meta['training_rows']} rows, "
          f"{meta['training_bad_days']} bad days; alert threshold {meta['threshold_probability']:.3f}")
