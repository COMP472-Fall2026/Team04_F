"""Staged COMP 472 Random Forest experiments. Run from the project root:
    python -m random_forest.tune_forest --task image --through 2

Uses the same bootstrap/feature RNG streams as RandomForest.fit, but trains one
DecisionTree at a time so its votes can be checkpointed and reused. No Kaggle
labels or external classifiers are used. Accuracy in CSV files is a fraction.
"""
import argparse
import hashlib
import inspect
import json
import os
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from common import DATA_DIR, SEED, accuracy, confusion_matrix
from trees.decision_tree import DecisionTree
from forests.random_forest import RandomForest

# Change these before starting a new run. A changed configuration is rejected
# if you try to resume into an existing run directory.
BASE = dict(max_depth=40, min_samples_split=20, min_samples_leaf=5,criterion="gini", max_features="sqrt", bootstrap=True)
DEPTHS = [10, 20, 40, 60]
FEATURES = ["log2", "sqrt", 0.1, 0.25]
COUNTS = [10, 25, 50, 100, 200]
SCREEN_COUNT = 25
TOP_STRUCTURE = 3
TOP_HOMOGENEOUS = 2
FOREST_SEED = 42
FOLDS = 5


def stratified_folds(y, number_of_folds=5, seed=SEED):
    """Same algorithm as trees/tune_tree_with_logging.py (raw features)."""
    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(number_of_folds)]
    offset = 0
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if len(class_indices) < number_of_folds:
            raise ValueError("Each class needs at least one row per fold")
        rng.shuffle(class_indices)
        for i, indices in enumerate(np.array_split(class_indices, number_of_folds)):
            fold_indices[(offset + i) % number_of_folds].append(indices)
        offset = (offset + len(class_indices) % number_of_folds) % number_of_folds
    for parts in fold_indices:
        val = np.sort(np.concatenate(parts))
        mask = np.ones(len(y), dtype=bool)
        mask[val] = False
        yield np.flatnonzero(mask), val


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_id(params):
    return hashlib.sha256(canonical(params).encode()).hexdigest()[:16]


def atomic_csv(table, path):
    temp = path.with_suffix(".tmp")
    table.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def spec(params, count):
    """A homogeneous forest represented as a range of tree slots [0,count)."""
    return [{"params": params, "start": 0, "stop": count}]


def mix(a, b, count, fraction):
    """Use disjoint slots across components to avoid paired bootstrap overlap."""
    cut = int(round(count * (1 - fraction)))
    if not 0 < cut < count:
        raise ValueError("Mixed forest needs at least one tree of each type")
    return [{"params": a, "start": 0, "stop": cut},
            {"params": b, "start": cut, "stop": count}]


def n_trees(components):
    return sum(c["stop"] - c["start"] for c in components)


def unique_params(configs):
    return list({canonical(p): p for p in configs}.values())


def ranked(rows):
    return sorted(rows, key=lambda r: (-r["validation_accuracy"], r["n_estimators"],
                                       r["forest_id"]))


class Experiment:
    def __init__(self, X, y, out, task, smoke=False):
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.folds = list(stratified_folds(y, FOLDS))
        self.out, self.task = out, task
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = out / "tree_votes"
        self.cache.mkdir(exist_ok=True)
        self.rows = {}
        self.smoke = smoke
        # Invalidate old checkpoints when data, experiment or classifiers change.
        digest = hashlib.sha256(Path(__file__).read_bytes())
        for cls in (DecisionTree, RandomForest):
            digest.update(inspect.getsource(cls).encode())
        for a in (X, y):
            a = np.ascontiguousarray(a)
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(memoryview(a).cast("B"))
        signature = digest.hexdigest()
        metadata = out / "signature.txt"
        if metadata.exists() and metadata.read_text().strip() != signature:
            raise ValueError("Code/data/settings changed. Choose a new --run-name.")
        if not metadata.exists():
            metadata.write_text(signature + "\n")
        # Keep previously completed stages in the consolidated CSV when rerunning
        # an earlier stage. Tree caches are still the authoritative checkpoints.
        runs_file = out / "runs.csv"
        if runs_file.exists():
            for row in pd.read_csv(runs_file, float_precision="round_trip").to_dict("records"):
                self.rows[(int(row["stage"]), row["name"], int(row["fold"])-1)] = row
        self.data_cache = {}
        self.vote_cache = {}  # in-memory cache of per-forest aggregate results

    def fold_data(self, fold):
        if fold not in self.data_cache:
            # Keep only the current fold's large arrays in memory.
            tr, va = self.folds[fold]
            self.data_cache = {fold: (self.X[tr], self.y[tr], self.X[va], self.y[va])}
        return self.data_cache[fold]

    def tree_votes(self, params, fold, start, stop):
        """Yield cached/new per-tree predictions and timings in slot order.

        Reproduce RandomForest.fit exactly: advance both RNG streams for each
        slot, even if its predictions were already saved. No retraining cached
        trees. The same slot across homogeneous configs uses paired randomness.
        """
        Xtr, ytr, Xva, yva = self.fold_data(fold)
        folder = self.cache / f"fold_{fold + 1}" / config_id(params)
        folder.mkdir(parents=True, exist_ok=True)
        bootstrap_seed, feature_seed = np.random.SeedSequence(FOREST_SEED).spawn(2)
        brng = np.random.default_rng(bootstrap_seed)
        frng = np.random.default_rng(feature_seed)
        tree_params = {k: v for k, v in params.items() if k != "bootstrap"}
        for i in range(stop):
            indices = brng.choice(len(ytr), size=len(ytr), replace=True) if params["bootstrap"] else None
            seed = int(frng.integers(0, 2**63))
            if i < start:
                continue
            path = folder / f"tree_{i + 1:04d}.npz"
            if not path.exists():
                print(f"  fold {fold+1} | {config_id(params)} | tree {i+1}/{stop} | training", flush=True)
                t = perf_counter()
                tree = DecisionTree(**tree_params, random_state=seed)
                tree.fit(Xtr[indices] if indices is not None else Xtr,
                         ytr[indices] if indices is not None else ytr)
                train_s = perf_counter() - t
                t = perf_counter()
                train_pred = tree.predict(Xtr)
                train_predict_s = perf_counter() - t
                t = perf_counter()
                val_pred = tree.predict(Xva)
                val_predict_s = perf_counter() - t
                atomic_npz(path, train=train_pred, val=val_pred,
                           times=np.array([train_s, train_predict_s, val_predict_s]))
                print(f"    saved | fit {train_s:.2f}s", flush=True)
            with np.load(path, allow_pickle=False) as data:
                yield data["train"], data["val"], data["times"]

    def evaluate(self, components, fold, stage, name):
        key = (canonical(components), fold)
        forest_id = hashlib.sha256(key[0].encode()).hexdigest()[:16]
        _, ytr, _, yva = self.fold_data(fold)
        if key not in self.vote_cache:
            votes_tr = np.zeros((len(ytr), len(self.classes)), dtype=np.int32)
            votes_va = np.zeros((len(yva), len(self.classes)), dtype=np.int32)
            it, iv = np.arange(len(ytr)), np.arange(len(yva))
            times = np.zeros(3)
            t = perf_counter()
            for component in components:
                for p_tr, p_va, tree_times in self.tree_votes(component["params"], fold,
                                                              component["start"], component["stop"]):
                    votes_tr[it, np.searchsorted(self.classes, p_tr)] += 1
                    votes_va[iv, np.searchsorted(self.classes, p_va)] += 1
                    times += tree_times
            pred_tr = self.classes[votes_tr.argmax(axis=1)]
            pred_va = self.classes[votes_va.argmax(axis=1)]
            self.vote_cache[key] = (accuracy(ytr, pred_tr), accuracy(yva, pred_va), times, pred_va)
            # Actual evaluation wall time can include new training, cache IO and voting.
            wall = perf_counter() - t
        else:
            wall = 0.0
        train_acc, val_acc, times, pred_va = self.vote_cache[key]
        p = components[0]["params"] if len(components) == 1 else {}
        row = dict(task=self.task, stage=stage, name=name, forest_id=forest_id,
                   fold=fold+1, split_seed=SEED, forest_seed=FOREST_SEED,
                   n_estimators=n_trees(components), kind="mixed" if len(components)>1 else "homogeneous",
                   max_depth=p.get("max_depth"), max_features=p.get("max_features", "mixed") if p else "mixed",
                   min_samples_split=p.get("min_samples_split"), min_samples_leaf=p.get("min_samples_leaf"),
                   criterion=p.get("criterion"), bootstrap=p.get("bootstrap"),
                   components=canonical(components), training_samples=len(ytr), validation_samples=len(yva),
                   training_accuracy=train_acc, validation_accuracy=val_acc,
                   accuracy_gap=train_acc-val_acc, training_time_seconds=float(times[0]),
                   train_prediction_time_seconds=float(times[1]), val_prediction_time_seconds=float(times[2]),
                   evaluation_wall_seconds=wall)
        if p and p["max_features"] is None:
            row["max_features"] = "all"
        existing = self.rows.get((stage, name, fold))
        if existing is not None:
            # Preserve the original measured evaluation wall time on resume.
            row["evaluation_wall_seconds"] = existing["evaluation_wall_seconds"]
        self.rows[(stage, name, fold)] = row
        atomic_csv(pd.DataFrame(self.rows.values()), self.out / "runs.csv")
        print(f"Stage {stage} | {name} | fold {fold+1} | {n_trees(components)} trees | "
              f"train {train_acc:.2%} | val {val_acc:.2%} | gap {train_acc-val_acc:.2%} | "
              f"sum tree fit {times[0]:.2f}s", flush=True)
        return row

    def summary(self, rows, name):
        table = pd.DataFrame(ranked(rows))
        atomic_csv(table, self.out / f"{name}.csv")
        return ranked(rows)

    def parity_check(self):
        """Check cached orchestration against the actual RandomForest class."""
        # A small synthetic 3-class set, independent of the experiment dataset.
        rng = np.random.default_rng(321)
        X = rng.normal(size=(60, 8))
        y = np.argmax(X[:, :3], axis=1)
        for bootstrap in (True, False):
            params = dict(BASE, max_depth=4, min_samples_split=2, min_samples_leaf=1,
                          max_features="sqrt", bootstrap=bootstrap)
            model = RandomForest(n_estimators=5, random_state=FOREST_SEED, **params).fit(X, y)
            b, f = np.random.SeedSequence(FOREST_SEED).spawn(2)
            brng, frng = np.random.default_rng(b), np.random.default_rng(f)
            votes = np.zeros((len(y), len(np.unique(y))), dtype=int)
            classes = np.unique(y)
            for original in model.trees:
                idx = brng.choice(len(y), size=len(y), replace=True) if bootstrap else np.arange(len(y))
                seed = int(frng.integers(0, 2**63))
                tree = DecisionTree(**{k:v for k,v in params.items() if k != "bootstrap"}, random_state=seed)
                tree.fit(X[idx], y[idx])
                pred = tree.predict(X)
                np.testing.assert_array_equal(pred, original.predict(X))
                votes[np.arange(len(y)), np.searchsorted(classes, pred)] += 1
            np.testing.assert_array_equal(classes[votes.argmax(axis=1)], model.predict(X))
        print("RandomForest parity check passed (bootstrap on/off).", flush=True)

    def run(self, through, include_all):
        if through >= 4:
            # The all-features choice may be made when starting stage 4, but
            # must not change later for the same named experiment.
            options_file = self.out / "stage4_options.json"
            options = canonical({"include_all_features": include_all})
            if options_file.exists() and options_file.read_text().strip() != options:
                raise ValueError("Stage 4 options changed. Use the original flag or a new --run-name.")
            if not options_file.exists():
                options_file.write_text(options + "\n")
        self.parity_check()
        counts = [2, 3, 5] if self.smoke else COUNTS
        screen = 3 if self.smoke else SCREEN_COUNT
        if screen not in counts or sorted(set(counts)) != counts or min(counts) < 2:
            raise ValueError("COUNTS must be increasing, unique, >=2, and include SCREEN_COUNT")
        depths = [2, 4] if self.smoke else DEPTHS
        features = ["log2", "sqrt"] if self.smoke else FEATURES
        base = dict(BASE, max_depth=4, min_samples_split=4, min_samples_leaf=2) if self.smoke else BASE
        print(f"Task {self.task} | stage 1..{through} | output {self.out}", flush=True)
        baseline = self.evaluate(spec(base, screen), 0, 1, "baseline")
        self.summary([baseline], "stage1_baseline")
        if through == 1:
            return
        configs = unique_params([base] + [dict(base, max_depth=d, max_features=f)
                                        for d, f in product(depths, features)])
        screening = [self.evaluate(spec(p, screen), 0, 2, f"structure_{i+1:02d}")
                     for i, p in enumerate(configs)]
        screening = self.summary(screening, "stage2_structure")
        if through == 2:
            return
        selected = [json.loads(r["components"])[0]["params"] for r in screening[:TOP_STRUCTURE]]
        size_rows = []
        for i, p in enumerate(selected):
            for count in counts:
                size_rows.append(self.evaluate(spec(p, count), 0, 3, f"structure_{i+1}_n{count}"))
        size_rows = self.summary(size_rows, "stage3_tree_count")
        if through == 3:
            return
        winner = size_rows[0]
        best_p = json.loads(winner["components"])[0]["params"]
        best_n = winner["n_estimators"]
        # Controlled homogeneous changes; avoid redundant split thresholds.
        leaf = best_p["min_samples_leaf"]
        split = best_p["min_samples_split"]
        variants = [best_p] + [dict(best_p, min_samples_split=s) for s in [10, 40, 80]
                              if s >= 2*leaf and s != split]
        variants += [dict(best_p, min_samples_leaf=l) for l in [1, 10]
                     if l != leaf and 2*l <= split]
        variants += [dict(best_p, criterion="entropy" if best_p["criterion"]=="gini" else "gini"),
                     dict(best_p, bootstrap=False)]
        if include_all:
            variants.append(dict(best_p, max_features=None))
        # Include other stage 3 structures at the SAME tree count for fair mixing.
        variants = unique_params(variants + selected)
        hom = [self.evaluate(spec(p, best_n), 0, 4, f"homogeneous_{i+1:02d}")
               for i,p in enumerate(variants)]
        hom = self.summary(hom, "stage4_homogeneous")
        anchor = json.loads(hom[0]["components"])[0]["params"]
        alternatives = [json.loads(r["components"])[0]["params"] for r in hom[1:3]]
        mixed = []
        for i, alt in enumerate(alternatives):
            for fraction in (0.25, 0.5, 0.75):
                if not 0 < round(best_n*(1-fraction)) < best_n:
                    continue
                mixed.append(self.evaluate(mix(anchor, alt, best_n, fraction), 0, 4,
                                           f"mixed_{i+1}_alt{int(fraction*100)}pct"))
        if mixed:
            mixed = self.summary(mixed, "stage4_mixed")
        self.summary(hom + mixed, "stage4_comparison")
        if through == 4:
            return
        # Confirm top homogeneous configurations and best mixed on all five folds.
        # First-fold results were used in selection; this is tuning CV, not an
        # independent test estimate. Compare with a matched single all-feature tree.
        finalists = hom[:TOP_HOMOGENEOUS] + mixed[:1]
        candidate_specs = [("RF_"+r["name"], json.loads(r["components"])) for r in finalists]
        candidate_specs.append(("DT_matched", spec(dict(anchor, bootstrap=False, max_features=None), 1)))
        if self.task == "image" and not self.smoke:
            candidate_specs.append(("DT_C475", spec(dict(max_depth=60, min_samples_split=80,
                         min_samples_leaf=10, criterion="gini", max_features=None, bootstrap=False), 1)))
        cv = []
        for name, components in candidate_specs:
            oof = np.empty_like(self.y)
            for fold in range(FOLDS):
                row = self.evaluate(components, fold, 5, name)
                cv.append(row)
                oof[self.folds[fold][1]] = self.vote_cache[(canonical(components), fold)][3]
            matrix = confusion_matrix(self.y, oof, int(self.classes.max())+1)
            atomic_csv(pd.DataFrame(matrix), self.out / f"confusion_{name}.csv")
            atomic_csv(pd.DataFrame({"row_index": np.arange(len(self.y)), "label": self.y,
                                     "prediction": oof}), self.out / f"oof_{name}.csv")
        df = pd.DataFrame(cv)
        summary = df.groupby(["name", "forest_id", "kind", "n_estimators", "components"], as_index=False).agg(
            runs=("fold", "count"), mean_training_accuracy=("training_accuracy", "mean"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            std_validation_accuracy=("validation_accuracy", lambda s: s.std(ddof=0)),
            mean_accuracy_gap=("accuracy_gap", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            mean_val_prediction_time_seconds=("val_prediction_time_seconds", "mean"))
        summary = summary.sort_values(["mean_validation_accuracy", "n_estimators"], ascending=[False, True])
        atomic_csv(summary, self.out / "stage5_cv_summary.csv")
        print(summary[["name", "n_estimators", "mean_validation_accuracy", "std_validation_accuracy"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["image", "sentiment"], default="image")
    parser.add_argument("--through", type=int, choices=range(1, 6), default=2,
                        help="Run/resume stages 1 through this stage (default: 2)")
    parser.add_argument("--run-name", default="forest_v1", help="Separate named output directory")
    parser.add_argument("--include-all-features", action="store_true",
                        help="Also test a potentially expensive all-feature bagged forest in stage 4")
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic data and small grids; never mixes with real runs")
    args = parser.parse_args()
    if args.smoke:
        rng = np.random.default_rng(123)
        y = np.repeat(np.arange(3), 30)
        X = rng.normal(size=(len(y), 12))
        X[:, 0] += 2*y
    else:
        # Read only labeled training CSV; no Kaggle test file is needed.
        data = pd.read_csv(Path(DATA_DIR) / f"{args.task}_train.csv")
        X = data[[c for c in data.columns if c.startswith("feature_")]].to_numpy(dtype=np.float32)
        y = data["label"].to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("Training features must be finite")
    out = Path(__file__).resolve().parent / "results" / args.run_name / ("smoke_" if args.smoke else "") / args.task
    Experiment(X, y, out, args.task, args.smoke).run(args.through, args.include_all_features)


if __name__ == "__main__":
    main()
