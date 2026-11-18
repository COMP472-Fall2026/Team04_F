"""Recover the original unseeded Decision Tree grid search safely.

Place this file and the recovered CSV next to tune_tree_recovery.py.
Run python resume_tree_legacy.py --restore for the initial recovery.
Afterward run python resume_tree_legacy.py (without --restore).
"""
import argparse
import hashlib
import inspect
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import trees.tune_tree_recovery as tuning

COUNT = 1199
# The latest project archive matches the user's diagnostic fingerprints.
EXPECTED_HASHES = {
    "common.py": "a91c2371962501e891dc833b55a0ffbb805346db3cd6312f3e803927674681c2",
    "tree_forest.py": "52c604d0fac4d1a0e51c84937ee6a71141e3d28827184c954a00cd8dfc64a97d",
    "data/image_train.csv": "41e0ef407a3825200f967a2fbfb629a88c50cd40714c9e654dee0c1d8e1c614d",
}
ROOT = Path(__file__).resolve().parent
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
    # Do not change the tree's RNG halfway through the historical experiment.
    if inspect.signature(tuning.DecisionTree).parameters["random_state"].default is not None:
        raise RuntimeError("Expected the original random_state=None behavior.")
    for name, expected_hash in EXPECTED_HASHES.items():
        actual_hash = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"{name} differs from the inspected project version. No checkpoint changed.")
    expected_functions = {'build_configurations': 'cc91910b6685f3bede02419452baf5da954113fb7c136b95d33bedf4c2f3c99b', 'stratified_folds': '58f33369d0a1a1989debe95094f739182b4da6e233f1b3667e94e29ff6dde9f1', 'prepare_folds': 'd4183b3b0012416db6023edf2b40782bafdc58c47cdcde097c51f9d112e4e74a', 'evaluate_configuration': '21713ef0dc21d71d7fb90f71dda38f478f476bedb5454eae5f05c4a6f2c5124f'}
    for name, expected_hash in expected_functions.items():
        actual_hash = hashlib.sha256(inspect.getsource(getattr(tuning, name)).rstrip("\n").encode()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Tuning function {name} changed. No checkpoint changed.")
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
            if "recovery_verification" not in existing or not (
                existing["recovery_verification"] == "deterministic_anchors_matched_legacy_rng"
            ).sum() == COUNT:
                raise RuntimeError("Existing checkpoint has 1,199+ rows but is not marked as this recovery. No checkpoint changed.")
            print("Existing checkpoint already restored; not overwriting it.")
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
    print("The original tree used random_state=None: logged tree seeds were nominal labels, not actual RNG seeds.")
    print("This continuation deliberately preserves that behavior. Deterministic full-feature fits can be checked.")
    print("Confirm that the dataset, row order, tree code and CV logic are unchanged since the original run.")
    if input("Type YES to verify and restore the original unseeded experiment: ").strip() != "YES":
        raise SystemExit("Cancelled. No checkpoint changed.")

    folds = tuning.prepare_folds(tuning.TASK)
    if len(folds) != 5 or any((len(f[1]), len(f[3])) != (16000, 4000) for f in folds):
        raise RuntimeError("Current fold sizes do not match the original log.")

    # Full-feature trees do not sample features. Randomized historical fits
    # cannot be exactly replayed because their actual entropy was not saved.
    for i in (24, 599):
        row = saved.iloc[i]
        if configs[i // 5]["parameters"]["max_features"] is not None:
            raise RuntimeError("Compatibility check requires all features.")
        print(f"Verifying deterministic iteration {i + 1:,} (all features)...", flush=True)
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
        print(f"  MATCH: training={actual['training_accuracy']:.6%}, validation={actual['validation_accuracy']:.6%}", flush=True)
        # Retain original recorded timing, but recover the exact training
        # accuracy for these rows from the verified repeat.
        saved.loc[i, "training_accuracy"] = actual["training_accuracy"]
        saved.loc[i, "accuracy_gap"] = actual["accuracy_gap"]
        saved.loc[i, "training_accuracy_exact"] = True
        saved.loc[i, "training_accuracy_min"] = actual["training_accuracy"]
        saved.loc[i, "training_accuracy_max"] = actual["training_accuracy"]

    # Assign a NEW signature for the confirmed current experiment. This is
    # not the lost original signature; provenance remains in the CSV.
    saved["recovery_verification"] = "deterministic_anchors_matched_legacy_rng"
    saved["tree_rng_mode"] = "unseeded_random_state_none"
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
    original_evaluate = tuning.evaluate_configuration
    def evaluate_new_run(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        result["recovered_from_log"] = False
        result["recovery_verification"] = "new_fit_legacy_rng"
        result["tree_rng_mode"] = "unseeded_random_state_none"
        result["training_accuracy_exact"] = True
        result["training_accuracy_min"] = result["training_accuracy"]
        result["training_accuracy_max"] = result["training_accuracy"]
        return result
    tuning.evaluate_configuration = evaluate_new_run
    print("Continuing original unseeded search; tree_seed is a nominal repetition label.", flush=True)
    tuning.run_experiment(tuning.TASK)


if __name__ == "__main__":
    main()
