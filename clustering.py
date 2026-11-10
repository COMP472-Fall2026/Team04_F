import os
import numpy as np
import pandas as pd
from common import *

# k-Means from scratch
def assign(X, centers, chunk_size=2000):
    """For every row of X, the index (0..k-1) of its nearest center.
    Same trick as k-NN:  d^2 = a·a + b·b - 2(a·b)."""
    b_dot_b = np.sum(centers ** 2, axis=1) # b dot b
    labels = np.empty(len(X), dtype=np.int64) # initialize empty labels array
    for start in range(0, len(X), chunk_size): # do it in chunks so its faster
        chunk = X[start : start + chunk_size]
        a_col = np.sum(chunk ** 2, axis=1).reshape(-1, 1) # a dot a then reshape to column so dot product works
        d2 = a_col + b_dot_b - 2 * (chunk @ centers.T) # (chunk, k), d^2 = a·a + b·b - 2(a·b)
        labels[start : start + chunk_size] = np.argmin(d2, axis=1) # returns the column (index) of the smallest value for each point
    return labels


def update(X, labels, k, rng):
    """New center = mean of the points assigned to it."""
    centers = np.empty((k, X.shape[1]), dtype=np.float64) # create empty array with one row per centre (k), and one column per feature (X.shape[1])
    for j in range(k): # loop through centers
        members = X[labels == j] # put True/False for it a point belongs to that center and X only keeps the ones with True
        if len(members) == 0: # empty cluster: restart it on a random point
            centers[j] = X[rng.integers(len(X))] # reassign center to another random point
        else:
            centers[j] = members.mean(axis=0) # get the average of each COLUMN to calculate the new center
    return centers


def inertia(X, labels, centers):
    """Sum of squared (to remove the negatives) distances from each point to its own center (lower inertia = closer to center -> good)."""
    return float(np.sum((X - centers[labels]) ** 2)) 


def kmeans(X, k, seed=SEED, max_iter=100, n_init=3):
    """
    Run k-Means n_init times from different random starts, keep the tightest.
    Steps:
    1. Pick k different data points at random as the starting centers
    2. Calculate the squared distance between every points to every centers
    3. Give each point the label of its closest centre
    4. Calculate new centers using the mean of the points associated to that center
    5. Check if with new centers if the labels (the center that point was associated with) changed
    6. If labels changed then replace old centers with new ones and repeat steps 4 and 5
    7. If labels stayed the same then calculate inertia
    8. If this inertia is lower than the best so far (or it's the first run), save this run
    9. Repeat steps 1 to 8 for 3 runs in total, each with a different random start, and return the best run
    """
    best = None
    for run in range(n_init): # The starting centres are random, so some starts are unlucky. Running it 3 times and keeping the best protects against a bad random centers.
        rng = np.random.default_rng(seed + run) # random number generator, each run is different but reproducible with same seed and run
        centers = X[rng.choice(len(X), size=k, replace=False)].astype(np.float64) # chose random centers
        labels = assign(X, centers) # get the labels for each point
        for it in range(max_iter):
            centers = update(X, labels, k, rng) # get new centers
            new_labels = assign(X, centers) # get labels for new centers
            if np.array_equal(new_labels, labels): # nothing moved -> converged
                break
            labels = new_labels # using new centers so change labels (if not equal to old labels)
        score = inertia(X, labels, centers)
        if best is None or score < best[0]:
            best = (score, labels, centers, it + 1)
    return best[1], best[2], best[0], best[3] # labels, centers, inertia, iterations

# Evaluation: compare clusters with the TRUE labels (never used for clustering)
def contingency(y, clusters, k, n_classes):
    """Rows = cluster, columns = true class, entry = how many points."""
    table = np.zeros((k, n_classes), dtype=np.int64)
    # creates a cluster-vs-class table 
    for c, t in zip(clusters, y): # zip pairs up, for each point i, (its cluster number, its true class).
        table[c, t] += 1 # increment every time a cluster contains that class
    return table


def purity(table):
    """Each cluster 'votes' for its most common class; purity = fraction that vote is right."""
    return table.max(axis=1).sum() / table.sum() 
    # max(axis=1): for each cluster (row), the number of points in its most common class (the class that cluster "votes" for)
    # .sum(): adds those up = how many points the majority votes get right
    # table.sum() calculates total number of points


def adjusted_rand_index(table):
    """1 = clusters identical to classes, ~0 = no better than random."""
    def pairs(x):
        return x * (x - 1) / 2
    sum_cells = pairs(table).sum() # pairs in the same cluster and same class
    sum_rows = pairs(table.sum(axis=1)).sum() # pairs in the same cluster
    sum_cols = pairs(table.sum(axis=0)).sum() # pairs in the same class
    expected = sum_rows * sum_cols / pairs(table.sum()) # what are the chances that if the clusters and classes had nothing to do with each other?
    maximum = (sum_rows + sum_cols) / 2 # the maximum number of right pairs (same cluster same class) we could get
    return (sum_cells - expected) / (maximum - expected) # see how many are right without the ones we would have gotten with luck


if __name__ == "__main__":
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cluster counts to try. Include the true number of classes.
    ks = {"image": [10, 25, 50, 75, 100], "sentiment": [2, 3, 5, 7, 10, 15, 20, 30, 50]}
    rows = []

    for task in TASKS:
        X, y, _, _ = load_task(task) # all labeled rows; labels used ONLY to evaluate
        n_classes = NUM_CLASSES[task]
        print(f"\n===== {task}: {X.shape[0]} points, {X.shape[1]} features =====")

        for k in ks[task]:
            with Timer() as t:
                labels, centers, inert, iters = kmeans(X, k)
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