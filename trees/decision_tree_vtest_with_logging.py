"""Exhaustive Decision Tree tuning using the project's own implementation."""

import hashlib
import inspect
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR,
    SEED,
    Timer,
    accuracy,
    get_task_data,
    load_task,
    set_seed,
)
from trees.decision_tree import DecisionTree


# ============================================================
# SETTINGS
# ============================================================

TASK = "sentiment"  # Change to "sentiment" or "image" for the other dataset.
CV_FOLDS = 5    # Set to 1 to use common.py's fixed 80/20 split instead.

DEPTH_VALUES = [10, 20, 30, 40, 60]
SPLIT_VALUES = [10, 20, 40, 80]
LEAF_VALUES = [5, 10, 20, 40]
CRITERION_VALUES = ["gini", "entropy"]
FEATURE_VALUES = [100, 200, 250, 350, None]  # None = all features.

# False: 500 non-redundant configurations. True: all 800 literal combinations.
INCLUDE_REDUNDANT = False

TREE_SEEDS = [42, 123, 456, 789, 1024]
# Every configuration is tested with the first seed on every fold.
# The best 3 are also tested with the remaining seeds on the same folds.
# Set to None to test EVERY configuration with EVERY seed.
FINALIST_COUNT = 3

# Resume from saved results if the experiment settings and data are unchanged.
RESUME = True


# ============================================================
# BUILD ALL HYPERPARAMETER COMBINATIONS
# ============================================================


def build_configurations():
    configurations = []

    combinations = product(
        DEPTH_VALUES,
        SPLIT_VALUES,
        LEAF_VALUES,
        CRITERION_VALUES,
        FEATURE_VALUES,
    )

    for depth, minimum_split, minimum_leaf, criterion, max_features in combinations:
        # The tree already requires at least 2 * min_samples_leaf samples
        # before splitting. Lower min_samples_split values are redundant.
        if not INCLUDE_REDUNDANT and minimum_split < 2 * minimum_leaf:
            continue

        configurations.append({
            "id": f"C{len(configurations) + 1:03d}",
            "parameters": {
                "max_depth": depth,
                "min_samples_split": minimum_split,
                "min_samples_leaf": minimum_leaf,
                "criterion": criterion,
                "max_features": max_features,
            },
        })

    return configurations


# ============================================================
# CREATE FIXED STRATIFIED VALIDATION FOLDS
# ============================================================


def stratified_folds(y, number_of_folds, seed=SEED):
    """Return reproducible (training indices, validation indices) pairs."""
    if number_of_folds < 2:
        raise ValueError("Cross-validation requires at least two folds")

    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(number_of_folds)]
    offset = 0

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if len(class_indices) < number_of_folds:
            raise ValueError("Each class must have at least one sample per fold")
        rng.shuffle(class_indices)

        # Rotate the allocation of any extra samples across folds.
        for i, indices in enumerate(np.array_split(class_indices, number_of_folds)):
            fold_indices[(offset + i) % number_of_folds].append(indices)
        offset = (offset + len(class_indices) % number_of_folds) % number_of_folds

    for parts in fold_indices:
        validation_indices = np.sort(np.concatenate(parts))
        training_mask = np.ones(len(y), dtype=bool)
        training_mask[validation_indices] = False
        yield np.flatnonzero(training_mask), validation_indices


def prepare_folds(task):
    """Use all labeled examples for CV, or the team's shared 80/20 split."""
    if CV_FOLDS == 1:
        X_train, y_train, X_val, y_val, _, _ = get_task_data(task)
        return [(X_train, y_train, X_val, y_val)]

    # load_task returns raw features. The unlabeled Kaggle set is never used.
    X, y, _, _ = load_task(task)
    return [
        (X[train], y[train], X[val], y[val])
        for train, val in stratified_folds(y, CV_FOLDS, seed=SEED)
    ]


# ============================================================
# TRAIN AND EVALUATE ONE TREE
# ============================================================


def evaluate_configuration(task, configuration, fold_number, tree_seed, fold):
    X_train, y_train, X_val, y_val = fold
    parameters = configuration["parameters"]

    set_seed(tree_seed)
    tree = DecisionTree(**parameters, random_state=tree_seed)

    with Timer() as training_timer:
        tree.fit(X_train, y_train)

    training_accuracy = accuracy(y_train, tree.predict(X_train))
    validation_accuracy = accuracy(y_val, tree.predict(X_val))

    return {
        "task": task,
        "config_id": configuration["id"],
        **parameters,
        "max_features": "all" if parameters["max_features"] is None else parameters["max_features"],
        "fold": fold_number,
        "split_seed": SEED,
        "tree_seed": tree_seed,
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "accuracy_gap": training_accuracy - validation_accuracy,
        "training_time_seconds": training_timer.elapsed,
    }


# ============================================================
# SAVE AND RANK EXPERIMENT RESULTS
# ============================================================


def save_csv(table, path):
    """Replace the CSV atomically so an interruption preserves the last save."""
    temporary_path = path + ".tmp"
    table.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def experiment_signature(task, configurations, folds):
    """Prevent accidentally resuming a different experiment from an old CSV."""
    settings = {
        "task": task,
        "split_seed": SEED,
        "folds": CV_FOLDS,
        "configurations": configurations,
        "tree_seeds": TREE_SEEDS,
        "finalist_count": FINALIST_COUNT,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(inspect.getsource(DecisionTree).encode())
    for fold in folds:
        for array in fold:
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def summarize_results(results, configurations, seeds, fold_count):
    """Rank only configurations with all requested seed/fold runs completed."""
    expected_runs = len(seeds) * fold_count
    rows = []
    for configuration in configurations:
        runs = [
            row for row in results
            if row["config_id"] == configuration["id"] and row["tree_seed"] in seeds
        ]
        if len(runs) != expected_runs:
            continue

        validation = np.array([row["validation_accuracy"] for row in runs])
        training = np.array([row["training_accuracy"] for row in runs])
        times = np.array([row["training_time_seconds"] for row in runs])
        parameters = configuration["parameters"]
        rows.append({
            "config_id": configuration["id"],
            **parameters,
            "max_features": "all" if parameters["max_features"] is None else parameters["max_features"],
            "runs": expected_runs,
            "mean_training_accuracy": float(training.mean()),
            "mean_validation_accuracy": float(validation.mean()),
            "std_validation_accuracy": float(validation.std()),
            "mean_accuracy_gap": float((training - validation).mean()),
            "mean_training_time_seconds": float(times.mean()),
            "total_training_time_seconds": float(times.sum()),
        })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    # Validation accuracy is the primary selection metric. Break ties using
    # stability, smaller overfitting gap, runtime, and finally a stable ID.
    table["absolute_mean_gap"] = table["mean_accuracy_gap"].abs()
    return table.sort_values(
        ["mean_validation_accuracy", "std_validation_accuracy", "absolute_mean_gap",
         "mean_training_time_seconds", "config_id"],
        ascending=[False, True, True, True, True],
    ).drop(columns="absolute_mean_gap").reset_index(drop=True)


# ============================================================
# RUN THE EXHAUSTIVE SEARCH
# ============================================================


class TeeOutput:
    """Write console output to both the terminal and a text log immediately."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def experiment_prefix(task):
    mode = "fixed_split" if CV_FOLDS == 1 else f"{CV_FOLDS}fold"
    grid = "all" if INCLUDE_REDUNDANT else "nonredundant"
    return os.path.join(RESULTS_DIR, f"tree_grid_{task}_{mode}_{grid}")


def run_experiment(task):
    """Run tuning while mirroring stdout and stderr to a persistent text file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = experiment_prefix(task) + "_output.txt"
    # Append on resume; overwrite when explicitly starting a fresh experiment.
    with open(log_path, "a" if RESUME else "w", encoding="utf-8") as log_file:
        with redirect_stdout(TeeOutput(sys.stdout, log_file)), redirect_stderr(
            TeeOutput(sys.stderr, log_file)
        ):
            print(f"\n{'=' * 65}\nSession started: {datetime.now().isoformat(timespec='seconds')}", flush=True)
            print(f"Output log: {log_path}", flush=True)
            try:
                result = _run_experiment(task)
            except KeyboardInterrupt:
                print("\nExperiment interrupted. Completed fits remain in the CSV checkpoint.", flush=True)
                raise
            except Exception:
                # The terminal receives the normal uncaught traceback on exit.
                # Save a copy here before restoring the original streams.
                traceback.print_exc(file=log_file)
                log_file.flush()
                raise
            print(f"Session completed: {datetime.now().isoformat(timespec='seconds')}", flush=True)
            return result


def _run_experiment(task):
    configurations = build_configurations()
    if not configurations or not TREE_SEEDS or len(set(TREE_SEEDS)) != len(TREE_SEEDS):
        raise ValueError("Provide at least one configuration and unique tree seeds")
    if FINALIST_COUNT is not None and FINALIST_COUNT < 1:
        raise ValueError("FINALIST_COUNT must be positive or None")

    folds = prepare_folds(task)
    fold_count = len(folds)
    finalist_count = len(configurations) if FINALIST_COUNT is None else min(
        FINALIST_COUNT, len(configurations)
    )
    total_fits = fold_count * (
        len(configurations) + finalist_count * (len(TREE_SEEDS) - 1)
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    prefix = experiment_prefix(task)
    runs_path = prefix + "_runs.csv"
    grid_path = prefix + "_grid_summary.csv"
    finalists_path = prefix + "_finalists.csv"

    signature = experiment_signature(task, configurations, folds)
    results = []
    if RESUME and os.path.exists(runs_path):
        saved = pd.read_csv(runs_path, float_precision="round_trip")
        if set(saved["experiment_signature"]) != {signature}:
            raise ValueError(
                "Saved experiment differs from current settings, data or tree code. "
                "Move the existing CSV files or set RESUME = False to restart."
            )
        results = saved.to_dict("records")
    else:
        # Avoid leaving stale rankings visible when explicitly restarting.
        for path in (runs_path, grid_path, finalists_path):
            if os.path.exists(path):
                os.remove(path)

    completed = {(r["config_id"], r["tree_seed"], r["fold"]) for r in results}
    if len(completed) != len(results):
        raise ValueError("The saved runs CSV contains duplicate experiments")

    print(f"\nTask: {task} | configurations: {len(configurations)} | folds: {fold_count}", flush=True)
    print(f"Training/validation sizes: {[(len(f[1]), len(f[3])) for f in folds]}", flush=True)
    print(f"Total planned tree fits: {total_fits:,} | already saved: {len(completed):,}", flush=True)
    print(f"Checkpoint: {runs_path}", flush=True)

    def run_configs(selected_configurations, seeds):
        for configuration in selected_configurations:
            for tree_seed in seeds:
                for fold_number, fold in enumerate(folds, start=1):
                    key = (configuration["id"], tree_seed, fold_number)
                    if key in completed:
                        continue

                    print(
                        f"\n[{len(completed) + 1:,}/{total_fits:,}] "
                        f"{configuration['id']} | seed={tree_seed} | fold={fold_number}/{fold_count}\n"
                        f"Parameters: {configuration['parameters']}", flush=True,
                    )
                    result = evaluate_configuration(task, configuration, fold_number, tree_seed, fold)
                    result["experiment_signature"] = signature
                    results.append(result)
                    save_csv(pd.DataFrame(results), runs_path)
                    completed.add(key)
                    print(
                        f"Training: {result['training_accuracy']:.2%} | "
                        f"Validation: {result['validation_accuracy']:.2%} | "
                        f"Gap: {result['accuracy_gap']:.2%} | "
                        f"Time: {result['training_time_seconds']:.2f}s",
                        flush=True,
                    )

    # Stage 1: exhaustively evaluate every configuration on the full data.
    # There is no small-subset screening or elimination before full CV.
    print("\nSTAGE 1: Full grid search", flush=True)
    run_configs(configurations, TREE_SEEDS[:1])
    grid_summary = summarize_results(results, configurations, TREE_SEEDS[:1], fold_count)
    save_csv(grid_summary, grid_path)

    # Stage 2: repeat the best configurations with the remaining seeds.
    finalists_ids = grid_summary.head(finalist_count)["config_id"].tolist()
    by_id = {configuration["id"]: configuration for configuration in configurations}
    finalists = [by_id[config_id] for config_id in finalists_ids]
    print(f"\nSTAGE 2: Multiple seeds for {len(finalists)} finalists", flush=True)
    run_configs(finalists, TREE_SEEDS[1:])

    final_summary = summarize_results(results, finalists, TREE_SEEDS, fold_count)
    save_csv(final_summary, finalists_path)
    winner = by_id[final_summary.iloc[0]["config_id"]]

    print("\nFINAL SELECTED CONFIGURATION", flush=True)
    print(winner, flush=True)
    print(f"Mean validation accuracy: {final_summary.iloc[0]['mean_validation_accuracy']:.2%}")
    print(f"Validation standard deviation: {final_summary.iloc[0]['std_validation_accuracy']:.2%}")
    print(f"All runs: {runs_path}\nGrid ranking: {grid_path}\nFinal ranking: {finalists_path}\nOutput log: {prefix}_output.txt")
    return winner, pd.DataFrame(results)


if __name__ == "__main__":
    best_configuration, experiment_results = run_experiment(TASK)
