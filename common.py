"""
common.py - shared utilities

Everyone imports from here so that all models use the SAME data loading,
SAME train/validation split, and SAME preprocessing.
Tell the group before changing anything in this file.

Usage:
    from common import *
    set_seed()
    X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data("image")
"""

import os
import csv
import time
import random

import numpy as np
import pandas as pd

# Config
SEED = 42
VAL_FRAC = 0.2
TASKS = ["image", "sentiment"]
NUM_CLASSES = {"image": 50, "sentiment": 3}

# Paths are relative to this file, so scripts work from any working directory.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SUBMISSIONS_DIR = os.path.join(ROOT_DIR, "submissions")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

# Reproducibility
def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Data loading
def load_task(task):
    """
    Load raw train/test CSVs for a task ("image" or "sentiment").

    Returns:
        X_train (n, d) float32, y_train (n,) int,
        X_test (m, d) float32, test_ids (m,) str
    Row order matches the CSV files.
    """
    assert task in TASKS, f"task must be one of {TASKS}"
    train = pd.read_csv(os.path.join(DATA_DIR, f"{task}_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, f"{task}_test.csv"))

    feat_cols = [c for c in train.columns if c.startswith("feature_")]
    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int64)
    X_test = test[feat_cols].to_numpy(dtype=np.float32)
    test_ids = test["id"].to_numpy()
    return X_train, y_train, X_test, test_ids

# Train / validation split (stratified, fixed seed)
def train_val_split(X, y, val_frac=VAL_FRAC, seed=SEED):
    """
    Stratified split: each class contributes the same fraction to validation.
    Same seed => same split for every teammate.
    """
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_mask[idx[:n_val]] = True
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]

# Preprocessing (always FIT on train only, then APPLY to val/test)
def fit_standardizer(X_train):
    """Return (mean, std) computed on the training portion only."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0  # avoid divide-by-zero on constant features
    return mu, sd


def apply_standardizer(X, mu, sd):
    return (X - mu) / sd


def standardize(X_train, *others):
    """
    Fit on X_train, apply to X_train and every array in `others`.
    Returns a list: [X_train_std, other1_std, other2_std, ...]
    """
    mu, sd = fit_standardizer(X_train)
    return [apply_standardizer(X, mu, sd) for X in (X_train, *others)]


def fit_l2_normalizer():
    """Row-wise L2 normalization needs no fitted parameters."""
    return l2_normalize


def l2_normalize(X):
    """Scale each row to unit length (useful for cosine distance in k-NN)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def get_task_data(task, standardize_features=False):
    """
    One-call helper: load, split, and (optionally) standardize.

    Returns:
        X_tr, y_tr, X_val, y_val, X_test, test_ids
    If standardize_features=True, the scaler is fit on X_tr only.
    """
    X_full, y_full, X_test, test_ids = load_task(task)
    X_tr, y_tr, X_val, y_val = train_val_split(X_full, y_full)
    if standardize_features:
        X_tr, X_val, X_test = standardize(X_tr, X_val, X_test)
    return X_tr, y_tr, X_val, y_val, X_test, test_ids


# Metrics (accuracy is the Kaggle metric; others via sklearn.metrics if wanted)
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


class Timer:
    """with Timer() as t: ...   then t.elapsed (seconds)."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

# Results logging (shared CSV -> becomes the results table in the report)
RESULT_FIELDS = [
    "task", "method", "hyperparameters",
    "train_acc", "val_acc", "train_time_s", "predict_time_s", "notes",
]


def log_result(task, method, hyperparameters, train_acc=None, val_acc=None,
               train_time_s=None, predict_time_s=None, notes="", results_file=None):
    """
    Append one row to results/results.csv, or to the supplied results_file.

    Example:
        log_result("image", "knn", {"k": 5, "dist": "euclid", "norm": True},
                   train_acc=0.9, val_acc=0.71, predict_time_s=3.2)
    """
    if results_file is None:
        results_file = RESULTS_FILE
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new_file = not os.path.exists(results_file)

    def fmt(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)

    row = {
        "task": task,
        "method": method,
        "hyperparameters": str(hyperparameters),
        "train_acc": fmt(train_acc),
        "val_acc": fmt(val_acc),
        "train_time_s": fmt(train_time_s),
        "predict_time_s": fmt(predict_time_s),
        "notes": notes,
    }
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

# Kaggle submission builder
def build_submission(image_ids, image_preds, sentiment_ids, sentiment_preds,
                     out_name="submission.csv"):
    """
    Merge predictions from both tasks by ID (NOT by row order), validate
    against sample_submission.csv, and write to submissions/<out_name>.

    image_ids / sentiment_ids: the test_ids returned by load_task / get_task_data
    *_preds: integer predictions in the same order as those ids
    """
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    image_results = pd.DataFrame({"id": image_ids, "label": np.asarray(image_preds)})
    sentiment_results = pd.DataFrame(
        {"id": sentiment_ids, "label": np.asarray(sentiment_preds)}
    )
    predictions = pd.concat([image_results, sentiment_results], ignore_index=True)

    submission = sample[["id"]].merge(
        predictions, on="id", how="left", validate="one_to_one"
    )

    # Validation checks from the project handout
    assert len(submission) == 10000, f"expected 10000 rows, got {len(submission)}"
    assert submission["label"].notna().all(), "missing predictions"
    submission["label"] = submission["label"].astype(int)

    image_mask = submission["id"].str.startswith("img_")
    sentiment_mask = submission["id"].str.startswith("text_")
    assert image_mask.sum() == 5000 and sentiment_mask.sum() == 5000
    assert submission.loc[image_mask, "label"].between(0, 49).all()
    assert submission.loc[sentiment_mask, "label"].between(0, 2).all()

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSIONS_DIR, out_name)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path


# Quick sanity check: python common.py
if __name__ == "__main__":
    set_seed()
    for t in TASKS:
        X_tr, y_tr, X_val, y_val, X_test, ids = get_task_data(t, standardize_features=True)
        print(f"[{t}] train {X_tr.shape}, val {X_val.shape}, test {X_test.shape}, "
              f"classes {len(np.unique(y_tr))}")
        # stratification check: class counts in val should be nearly equal
        counts = np.bincount(y_val, minlength=NUM_CLASSES[t])
        print(f"[{t}] val samples per class: min {counts.min()}, max {counts.max()}")
