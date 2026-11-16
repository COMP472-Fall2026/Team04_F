import os
from itertools import product

import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR,
    SEED,
    Timer,
    accuracy,
    get_task_data,
    set_seed,
)
from tree_forest import DecisionTree


# ============================================================
# SETTINGS
# ============================================================

# Run one task at a time because tree_forest_zay is slow.
# Change this to "sentiment" to test the sentiment dataset.
TASK = "image"


# ============================================================
# BUILD THE HYPERPARAMETER COMBINATIONS
# ============================================================

def build_configurations():
    depth_values = [20, 30, 40, 60, 80]
    split_values = [10, 20, 40]
    leaf_values = [1, 5, 10]

    configurations = []

    combinations = product(
        depth_values,
        split_values,
        leaf_values,
    )

    for depth, minimum_split, minimum_leaf in combinations:
        # A split must be able to create two leaves.
        # Skipping this condition avoids testing redundant settings.
        if minimum_split < 2 * minimum_leaf:
            continue

        configuration = {
            "parameters": {
                "max_depth": depth,
                "min_samples_split": minimum_split,
                "min_samples_leaf": minimum_leaf,
            }
        }

        configurations.append(configuration)

    # Give each configuration a stable ID.
    for number, configuration in enumerate(configurations, start=1):
        configuration["id"] = f"C{number:02}"

    return configurations


# ============================================================
# CREATE A STRATIFIED TRAINING SUBSET
# ============================================================

def stratified_subset(X, y, fraction, seed=SEED):
    """
    Select a fixed percentage from every class.

    Reusing the same seed makes the subsets reproducible and
    nested: the 10% rows are contained in the 25% rows, etc.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    random_generator = np.random.default_rng(seed)
    selected_indices = []

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)

        # Shuffle this class into a fixed, reproducible order.
        random_generator.shuffle(class_indices)

        number_to_select = max(
            1,
            int(round(len(class_indices) * fraction)),
        )

        selected_indices.append(
            class_indices[:number_to_select]
        )

    selected_indices = np.concatenate(selected_indices)

    # Use a consistent row order.
    selected_indices = np.sort(selected_indices)

    return (
        X[selected_indices],
        y[selected_indices],
        selected_indices,
    )


# ============================================================
# TRAIN AND EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    task,
    stage_name,
    fraction,
    configuration,
    X_subset,
    y_subset,
    X_validation,
    y_validation,
):
    """
    Train one brand-new tree and measure its training and
    validation accuracy.
    """
    parameters = configuration["parameters"]

    tree = DecisionTree(**parameters)

    # Measure how long tree.fit() takes.
    with Timer() as training_timer:
        tree.fit(X_subset, y_subset)

    training_predictions = tree.predict(X_subset)
    validation_predictions = tree.predict(X_validation)

    training_accuracy = accuracy(
        y_subset,
        training_predictions,
    )

    validation_accuracy = accuracy(
        y_validation,
        validation_predictions,
    )

    accuracy_gap = training_accuracy - validation_accuracy

    return {
        "task": task,
        "stage": stage_name,
        "training_fraction": fraction,
        "config_id": configuration["id"],
        "max_depth": parameters["max_depth"],
        "min_samples_split": parameters["min_samples_split"],
        "min_samples_leaf": parameters["min_samples_leaf"],
        "training_samples": len(y_subset),
        "validation_samples": len(y_validation),
        "split_seed": SEED,
        "subset_seed": SEED,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "accuracy_gap": accuracy_gap,
        "training_time_seconds": training_timer.elapsed,
    }


# ============================================================
# RUN THE PROGRESSIVE HYPERPARAMETER TOURNAMENT
# ============================================================

def run_experiment(task):
    set_seed(SEED)

    # common.py creates the fixed stratified 80/20 split.
    X_train, y_train, X_val, y_val, _, _ = get_task_data(task)

    # Start with every generated parameter combination.
    surviving_configurations = build_configurations()

    # Each tuple contains:
    # (stage name, training fraction, number allowed to advance)
    stages = [
        ("10_percent", 0.10, 6),
        ("25_percent", 0.25, 3),
        ("50_percent", 0.50, 2),
        ("100_percent", 1.00, 1),
    ]

    all_results = []

    print()
    print(f"Task: {task}")
    print(
        f"Initial configurations: "
        f"{len(surviving_configurations)}"
    )
    print(f"Full training pool: {len(y_train):,}")
    print(f"Fixed validation set: {len(y_val):,}")

    for stage_name, fraction, number_to_keep in stages:
        print()
        print("=" * 65)
        print(f"STAGE: {stage_name}")
        print(
            f"Configurations being tested: "
            f"{len(surviving_configurations)}"
        )
        print("=" * 65)

        # Create this stage's stratified training subset.
        X_subset, y_subset, _ = stratified_subset(
            X_train,
            y_train,
            fraction,
            seed=SEED,
        )

        print(f"Training samples:   {len(y_subset):,}")
        print(f"Validation samples: {len(y_val):,}")

        stage_results = []

        # Train and evaluate each surviving configuration.
        for configuration in surviving_configurations:
            result = evaluate_configuration(
                task=task,
                stage_name=stage_name,
                fraction=fraction,
                configuration=configuration,
                X_subset=X_subset,
                y_subset=y_subset,
                X_validation=X_val,
                y_validation=y_val,
            )

            stage_results.append(
                (result, configuration)
            )

            all_results.append(result)

            print()
            print(f"Configuration:      {result['config_id']}")
            print(
                "Parameters:         "
                f"depth={result['max_depth']}, "
                f"split={result['min_samples_split']}, "
                f"leaf={result['min_samples_leaf']}"
            )
            print(
                "Training accuracy:  "
                f"{result['training_accuracy']:.2%}"
            )
            print(
                "Validation accuracy:"
                f" {result['validation_accuracy']:.2%}"
            )
            print(
                "Accuracy gap:       "
                f"{result['accuracy_gap']:.2%}"
            )
            print(
                "Training time:      "
                f"{result['training_time_seconds']:.2f} seconds"
            )

        # Rank the configurations using:
        # 1. Highest validation accuracy
        # 2. Smallest training-validation gap
        # 3. Shortest training time
        stage_results.sort(
            key=lambda item: (
                -item[0]["validation_accuracy"],
                abs(item[0]["accuracy_gap"]),
                item[0]["training_time_seconds"],
            )
        )

        winners = stage_results[:number_to_keep]

        print()
        print(f"Advancing from {stage_name}:")

        for result, configuration in winners:
            print(
                f"  {configuration['id']}: "
                f"validation={result['validation_accuracy']:.2%}, "
                f"gap={result['accuracy_gap']:.2%}, "
                f"time={result['training_time_seconds']:.2f}s"
            )

        # Only these winners are tested at the next stage.
        surviving_configurations = [
            configuration
            for result, configuration in winners
        ]

    # ========================================================
    # SAVE EVERY EXPERIMENT TO A CSV FILE
    # ========================================================

    results_table = pd.DataFrame(all_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        f"tree_forest_zay_tuning_{task}.csv",
    )

    results_table.to_csv(output_path, index=False)

    final_configuration = surviving_configurations[0]

    print()
    print("=" * 65)
    print("FINAL SELECTED CONFIGURATION")
    print("=" * 65)
    print(f"Configuration ID: {final_configuration['id']}")
    print(
        f"Parameters:       "
        f"{final_configuration['parameters']}"
    )
    print(f"Results saved to: {output_path}")

    return final_configuration, results_table


# ============================================================
# START THE EXPERIMENT
# ============================================================

if __name__ == "__main__":
    best_configuration, experiment_results = run_experiment(
        TASK
    )