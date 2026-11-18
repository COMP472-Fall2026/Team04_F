"""One-time checkpoint restoration and safe subsequent resumption.

Place this file and the recovered CSV next to tune_tree_recovery.py.
Run python resume_tree_from_log.py --restore for the initial recovery.
Afterward run python resume_tree_from_log.py (without --restore).
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import trees.tune_tree_recovery as tuning

COUNT = 1199
SOURCE = Path(__file__).resolve().with_name(
    "tree_grid_image_5fold_nonredundant_recovered.csv"
)


def check_settings():
    expected = {
        "TASK": "image", "CV_FOLDS": 5, "SEED": 42,
        "DEPTH_VALUES": [10, 20, 30, 40, 60],
        "SPLIT_VALUES": [10, 20, 40, 80],
        "LEAF_VALUES": [5, 10, 20, 40],
        "CRITERION_VALUES": ["gini", "entropy"],
        "FEATURE_VALUES": [100, 200, 250, 350, None],
        "INCLUDE_REDUNDANT": False,
        "TREE_SEEDS": [42, 123, 456, 789, 1024],
        "FINALIST_COUNT": 3,
    }
    for name, value in expected.items():
        if getattr(tuning, name) != value:
            raise RuntimeError(f"Restore original setting {name} = {value!r} first.")
    tuning.RESUME = True
    tuning.RECOVERY_MIN_COMPLETED = COUNT


def restore():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Recovered CSV not found: {SOURCE}")
    target = Path(tuning.experiment_prefix(tuning.TASK) + "_runs.csv")
    print(f"Checkpoint: {target}", flush=True)
    if target.exists():
        existing = pd.read_csv(target)
        if len(existing) >= COUNT:
            print("Existing checkpoint has at least 1,199 rows; not overwriting it.")
            return
        if len(existing) > 6:
            raise RuntimeError(
                "Existing checkpoint has more than six rows but fewer than 1,199. "
                "Back it up and move it aside before requesting restoration."
            )

    saved = pd.read_csv(SOURCE, dtype={
        "max_features": str, "logged_training_accuracy_percent": str,
        "logged_validation_accuracy_percent": str, "logged_gap_percent": str,
    })
    configs = tuning.build_configurations()
    if len(configs) != 500 or len(saved) != COUNT:
        raise RuntimeError("Unexpected configuration or recovered row count.")
    if set(saved["experiment_signature"]) != {"UNVERIFIED_LOG_RECOVERY"}:
        raise RuntimeError("The recovered source CSV was changed.")
    for i, row in enumerate(saved.to_dict("records")):
        config = configs[i // 5]
        params = config["parameters"]
        if (row["config_id"], row["tree_seed"], row["fold"]) != (
            config["id"], 42, i % 5 + 1
        ):
            raise RuntimeError(f"Unexpected run order at row {i + 1}")
        for key, value in params.items():
            if key == "max_features":
                value = "all" if value is None else str(value)
            if row[key] != value:
                raise RuntimeError(f"Parameter mismatch at row {i + 1}: {key}")
        if (row["task"], row["split_seed"], row["training_samples"],
            row["validation_samples"]) != ("image", 42, 16000, 4000):
            raise RuntimeError(f"Data metadata mismatch at row {i + 1}")
        if (not np.isfinite(row["training_accuracy"]) or
            not np.isfinite(row["validation_accuracy"]) or
            not np.isfinite(row["training_time_seconds"]) or
            not 0 <= row["training_accuracy"] <= 1 or
            not 0 <= row["validation_accuracy"] <= 1 or
            row["training_time_seconds"] < 0):
            raise RuntimeError(f"Invalid metric at row {i + 1}")

    print("Validated 1,199 recovered runs against the original grid.")
    print("The original experiment hash was not recorded in the log.")
    print("Confirm that the dataset, row order, tree code and CV logic are unchanged.")
    if input("Type YES to verify and restore the recovered checkpoint: ").strip() != "YES":
        raise SystemExit("Cancelled. No checkpoint changed.")

    folds = tuning.prepare_folds(tuning.TASK)
    if len(folds) != 5 or any((len(f[1]), len(f[3])) != (16000, 4000) for f in folds):
        raise RuntimeError("Current fold sizes do not match the original log.")

    # Check three actual fits against the recorded metrics, before writing.
    # Spot checks are evidence of compatibility, not proof of identical inputs.
    for i in (0, 599, 1198):
        row = saved.iloc[i]
        print(f"Verifying recovered iteration {i + 1:,} by retraining it once...", flush=True)
        actual = tuning.evaluate_configuration(
            tuning.TASK, configs[i // 5], int(row["fold"]),
            int(row["tree_seed"]), folds[int(row["fold"]) - 1]
        )
        if (abs(actual["validation_accuracy"] - row["validation_accuracy"]) > 1e-12 or
            actual["training_accuracy"] < row["training_accuracy_min"] - 1e-12 or
            actual["training_accuracy"] > row["training_accuracy_max"] + 1e-12):
            raise RuntimeError(
                f"Spot check failed at iteration {i + 1}. "
                "Current data or implementation may differ. No checkpoint changed."
            )
        # Retain original recorded timing, but recover the exact training
        # accuracy for these rows from the verified repeat.
        saved.loc[i, "training_accuracy"] = actual["training_accuracy"]
        saved.loc[i, "accuracy_gap"] = actual["accuracy_gap"]
        saved.loc[i, "training_accuracy_exact"] = True
        saved.loc[i, "training_accuracy_min"] = actual["training_accuracy"]
        saved.loc[i, "training_accuracy_max"] = actual["training_accuracy"]

    # Assign a NEW signature for the confirmed current experiment. This is
    # not the lost original signature; provenance remains in the CSV.
    saved["experiment_signature"] = tuning.experiment_signature(
        tuning.TASK, configs, folds
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(
            target.name + ".before_restore_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        shutil.copy2(target, backup)
        print(f"Previous checkpoint copied to {backup}", flush=True)
    tuning.save_csv(saved, str(target))
    print(f"Restored {len(pd.read_csv(target)):,} runs. Original source CSV unchanged.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Restore the log-derived checkpoint once")
    args = parser.parse_args()
    check_settings()
    if args.restore:
        restore()
    # Existing recovery module refuses to train unless all first 1,199
    # expected runs are present and the current signature matches.
    tuning.run_experiment(tuning.TASK)


if __name__ == "__main__":
    main()
