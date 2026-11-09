import os
import numpy as np
import pandas as pd
from common import *

# k-Means from scratch
def assign(X, centres, chunk_size=2000):
    """For every row of X, the index (0..k-1) of its nearest centre.
    Same trick as k-NN:  d² = a·a + b·b - 2(a·b)  (we only need the argmin,
    so no square root is required)."""
    b_dot_b = np.sum(centres ** 2, axis=1)
    labels = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), chunk_size):
        chunk = X[start : start + chunk_size]
        a_col = np.sum(chunk ** 2, axis=1).reshape(-1, 1)
        d2 = a_col + b_dot_b - 2 * (chunk @ centres.T) # (chunk, k)
        labels[start : start + chunk_size] = np.argmin(d2, axis=1)
    return labels


def update(X, labels, k, rng):
    """New centre = mean of the points assigned to it."""
    centres = np.empty((k, X.shape[1]), dtype=np.float64)
    for j in range(k):
        members = X[labels == j]
        if len(members) == 0: # empty cluster: restart it on a random point
            centres[j] = X[rng.integers(len(X))]
        else:
            centres[j] = members.mean(axis=0)
    return centres


def inertia(X, labels, centres):
    """Sum of squared distances from each point to its own centre (lower = tighter)."""
    return float(np.sum((X - centres[labels]) ** 2))


def kmeans(X, k, seed=SEED, max_iter=100, n_init=3):
    """Run k-Means n_init times from different random starts, keep the tightest."""
    best = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centres = X[rng.choice(len(X), size=k, replace=False)].astype(np.float64)
        labels = assign(X, centres)
        for it in range(max_iter):
            centres = update(X, labels, k, rng)
            new_labels = assign(X, centres)
            if np.array_equal(new_labels, labels): # nothing moved -> converged
                break
            labels = new_labels
        score = inertia(X, labels, centres)
        if best is None or score < best[0]:
            best = (score, labels, centres, it + 1)
    return best[1], best[2], best[0], best[3] # labels, centres, inertia, iterations

# Evaluation: compare clusters with the TRUE labels (never used for clustering)
def contingency(y, clusters, k, n_classes):
    """Rows = cluster, columns = true class, entry = how many points."""
    table = np.zeros((k, n_classes), dtype=np.int64)
    for c, t in zip(clusters, y):
        table[c, t] += 1
    return table


def purity(table):
    """Each cluster 'votes' for its most common class; purity = fraction that vote is right."""
    return table.max(axis=1).sum() / table.sum()


def adjusted_rand_index(table):
    """1 = clusters identical to classes, ~0 = no better than random."""
    def pairs(x):
        return x * (x - 1) / 2
    sum_cells = pairs(table).sum()
    sum_rows = pairs(table.sum(axis=1)).sum()
    sum_cols = pairs(table.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / pairs(table.sum())
    maximum = (sum_rows + sum_cols) / 2
    return (sum_cells - expected) / (maximum - expected)


if __name__ == "__main__":
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cluster counts to try. Include the true number of classes.
    ks = {"image": [10, 25, 50, 75, 100], "sentiment": [2, 3, 5, 10, 20]}
    rows = []

    for task in TASKS:
        X, y, _, _ = load_task(task) # all labeled rows; labels used ONLY to evaluate
        n_classes = NUM_CLASSES[task]
        print(f"\n===== {task}: {X.shape[0]} points, {X.shape[1]} features =====")

        for k in ks[task]:
            with Timer() as t:
                labels, centres, inert, iters = kmeans(X, k)
            table = contingency(y, labels, k, n_classes)
            pur, ari = purity(table), adjusted_rand_index(table)
            print(f"k={k:<4} purity={pur:.3f}  ARI={ari:.3f}  inertia={inert:,.0f}  "
                  f"iters={iters}  time={t.elapsed:.1f}s")
            rows.append({"task": task, "k": k, "purity": round(pur, 4), "ARI": round(ari, 4),
                         "inertia": round(inert, 1), "iterations": iters,
                         "time_s": round(t.elapsed, 2)})

            # Show the full cluster-vs-class table when it is small enough to read
            if task == "sentiment" and k == 3:
                print("cluster-vs-class table (rows = cluster, columns = neg, neu, pos):")
                print(table)

            # For images at the true k, show how clean each cluster is
            if task == "image" and k == n_classes:
                top = table.argmax(axis=1)
                share = table.max(axis=1) / table.sum(axis=1)
                order = np.argsort(share)
                print("5 messiest clusters (cluster, most common class, share of cluster):")
                print([(int(c), int(top[c]), round(float(share[c]), 2)) for c in order[:5]])
                print("5 cleanest clusters:")
                print([(int(c), int(top[c]), round(float(share[c]), 2)) for c in order[-5:]])
                print("classes that are the top class of no cluster:",
                      sorted(set(range(n_classes)) - set(top.tolist())))

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "clustering_results.csv"), index=False)
    print("\nSaved results/clustering_results.csv")