import numpy as np
from common import *
import os


class KNN:
    def __init__(self, k=5, metric="euclidean", power=0.0):
        self.k = k
        self.metric = metric
        # Neighbour weight = 1 / distance ** power.
        # power = 0 -> every neighbour has weight 1 (plain unweighted voting).
        self.power = power

    def fit(self, x, y):
        # k-NN has no training step: it just remembers the data.
        self.x_train = x
        self.y_train = y
        self.b_dot_b = np.sum(x ** 2, axis=1) # b·b, computed once

    def _distances(self, x_chunk):
        # distance² = a·a + b·b - 2(a·b), for every (new point, training point) pair
        a_dot_a = np.sum(x_chunk ** 2, axis=1)
        a_dot_b = x_chunk @ self.x_train.T
        a_col = a_dot_a.reshape(-1, 1)
        sums = a_col + self.b_dot_b
        d2 = sums - 2 * a_dot_b
        return np.sqrt(np.maximum(d2, 0))

    def predict(self, x, chunk_size=500):
        predictions = []
        for start in range(0, len(x), chunk_size):
            x_chunk = x[start : start + chunk_size]
            dists = self._distances(x_chunk)
            order = np.argsort(dists, axis=1) # each row sorted closest -> farthest
            nearest = order[:, :self.k]
            nearest_labels = self.y_train[nearest]

            # Distance of each of the k neighbours
            nearest_dists = np.take_along_axis(dists, nearest, axis=1)
            # Closer neighbour -> bigger weight. The max() avoids dividing by 0
            # if a neighbour is an exact duplicate of the query (distance 0)
            weights = 1.0 / np.maximum(nearest_dists.astype(np.float64), 1e-3) ** self.power

            for row, w in zip(nearest_labels, weights):
                # labels: the different classes in this row
                # first_pos: where each class first appears (0 = closest neighbour)
                # inverse: for every neighbour, which entry of labels it belongs to
                labels, first_pos, inverse = np.unique(
                    row, return_index=True, return_inverse=True
                )
                # Total weight of each class (instead of just counting votes)
                totals = np.bincount(inverse, weights=w)
                # Classes sharing the top total (usually just one)
                tied = np.where(totals == totals.max())[0]
                # Tie-break: the class whose neighbour is CLOSEST
                winner_position = tied[np.argmin(first_pos[tied])]
                predictions.append(labels[winner_position])
        return np.array(predictions)

def clear_knn_results():
    """Delete the old k-NN results file so a re-run doesn't duplicate rows."""
    if os.path.exists(KNN_RESULTS_FILE):
        os.remove(KNN_RESULTS_FILE)


def mark_best():
    """Add 'BEST IN GROUP' to the best row of every experiment, and 'BEST OVERALL'
    to the best row of each task, in the notes column of knn-results.csv."""
    table = pd.read_csv(KNN_RESULTS_FILE)
    for _, rows in table.groupby(["task", "notes"]):
        table.loc[rows["val_acc"].idxmax(), "notes"] += " | BEST IN GROUP"
    for _, rows in table.groupby("task"):
        table.loc[rows["val_acc"].idxmax(), "notes"] += " | BEST OVERALL"
    table.to_csv(KNN_RESULTS_FILE, index=False, float_format="%.4f")

def per_class_metrics(cm):
    """cm: rows = true class, columns = predicted class.
    Returns precision, recall and F1, one value per class."""
    correct = cm.diagonal().astype(np.float64)
    precision = correct / np.maximum(cm.sum(axis=0), 1) # of the points predicted as c, how many truly are c
    recall = correct / np.maximum(cm.sum(axis=1), 1) # of the points that truly are c, how many we found
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1

def run_one(task, data, k, power, metric, standardized, with_power, group, lda=None, exponent=None, append=None):
    """Fit on the train split, predict the validation split, log one row. Returns the accuracy.
    group names the experiment (it goes in the notes column)."""
    X_tr, y_tr, X_val, y_val = data
    if metric == "cosine": # cosine = Euclidean on rows of length 1
        X_tr, X_val = l2_normalize(X_tr), l2_normalize(X_val)

    if exponent is not None: # shrink the large feature values, then renormalize each row
        X_tr, X_val = power_normalize(X_tr, exponent), power_normalize(X_val, exponent)

    if append is not None: # append = (dims, shrink, weight): keep the features, add scaled LDA features
        dims, shrink, weight = append
        X_tr, X_val = append_lda(X_tr, y_tr, X_val, NUM_CLASSES[task], dims, shrink, weight)

    if lda is not None: # lda = (dims, shrink): learn on the train split only, project both splits
        dims, shrink = lda
        fitted = fit_lda(X_tr, y_tr, NUM_CLASSES[task], shrink)
        X_tr, X_val = apply_lda(X_tr, fitted, dims), apply_lda(X_val, fitted, dims)

    model = KNN(k=k, metric=metric, power=power)
    model.fit(X_tr, y_tr)
    with Timer() as t:
        preds = model.predict(X_val)
    acc = accuracy(y_val, preds)

    settings = {"k": k, "metric": metric, "standardized": standardized}
    if with_power:
        settings["power"] = power
    if exponent is not None:
        settings["exponent"] = exponent
    if append is not None:
        settings["append_dims"], settings["append_shrink"], settings["append_weight"] = append
    if lda is not None:
        settings["lda_dims"], settings["lda_shrink"] = lda
    log_result(task, "knn", settings, val_acc=acc, predict_time_s=t.elapsed,
               notes=group, results_file=KNN_RESULTS_FILE)
    print(f"{task:<10} k={k:<4} {metric:<9} std={standardized!s:<5} power={power:<3} lda={lda} val acc {acc:.4f}")
    return acc

def fit_lda(X, y, n_classes, shrink):
    """Learn a supervised projection from the training data only.
    Returns the training mean and a matrix W; project with (X - mean) @ W[:, :dims]."""
    X = X.astype(np.float64)
    n, d = X.shape # n = rows (number of class), d = columns (number of features)
    overall_mean = X.mean(axis=0) # get the mean for all class features
    Sw = np.zeros((d, d)) # within-class scatter: how spread out each class is around its own mean
    Sb = np.zeros((d, d)) # between-class scatter: how far the class means are from the overall mean
    for c in range(n_classes):
        Xc = X[y == c] # get points for that class
        class_mean = Xc.mean(axis=0) # get the mean for the the class feature (center of all the data ignoring labels)
        gap = Xc - class_mean # get the difference between the points of the class and the class mean (how far each point is from its class center)
        Sw += gap.T @ gap # How much points spread around their own class mean (we want this to be small)
        diff = (class_mean - overall_mean).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T) # How much class means differ from the overall mean (we want this to be big)
    # normalize the matrices by the number of training examples
    Sw /= n
    Sb /= n
    # Shrinkage: make the LDA calculations more stable and less sensitive to extremely small within-class scatter estimates (prevent overfitting)
    Sw += (shrink * (np.trace(Sw) / d)) * np.eye(d) # trace gives the sum of the diagonal so np.trace(Sw) measures the total within-class spread across all features
    # np.eye(d) give an identity matrix of dxd where d is the number of features
    # Whiten Sw (make within-class spread equal in every direction)
    vals, vecs = np.linalg.eigh(Sw)
    whiten = vecs / np.sqrt(vals)
    # eigenvectors identify the directions of within-class spread, and dividing by the square root of their 
    # eigenvalues scales those directions so their within-class scatter becomes equal.
    # then find which directions separate the different classes the most
    vals2, vecs2 = np.linalg.eigh(whiten.T @ Sb @ whiten)
    order = np.argsort(vals2)[::-1] # directions with the strongest class separation come first
    return overall_mean, whiten @ vecs2[:, order] # return overall_mean and final LDA transformation matrix
    # final LDA transformation matrix combines whitening and class-separating directions to transform the original features into new, more discriminative features

def apply_lda(X, lda, dims):
    mean, W = lda
    # take original data and center it using the mean learned during training
    # multiply it by the first dims columns of W (remember that W's column are ordered by best class separation first) to produce new features that LDA considers useful for distinguishing classes.
    return (X.astype(np.float64) - mean) @ W[:, :dims] 

def power_normalize(X, exponent):
    """Raise every feature to exponent (0.5 = square root), then rescale each row to length 1.
    Only valid for non-negative features (the image features are)."""
    return l2_normalize(X.astype(np.float64) ** exponent)

def append_lda(X_fit, y_fit, X_other, n_classes, dims, shrink, weight):
    """Keep every original feature and add dims LDA features on the right, scaled by weight.
    LDA is learned on X_fit only (the training rows); X_other (validation or test) is just projected."""
    fitted = fit_lda(X_fit, y_fit, n_classes, shrink)
    Z_fit = apply_lda(X_fit, fitted, dims)
    Z_other = apply_lda(X_other, fitted, dims)
    scale = np.sqrt(np.mean(np.sum(Z_fit ** 2, axis=1))) # find scale so that LDA row lengths become 1 so it matches the square-root transformed values and doesn't dominate
    X_fit = np.hstack([X_fit, weight * Z_fit / scale]) # scale LDA rows to 1, multiply by weight, then add to square-root transformed values
    X_other = np.hstack([X_other, weight * Z_other / scale]) # same thing but for validation set
    return X_fit, X_other

# Experiment settings
RUN_SWEEPS = False  
# True: redo every experiment and rewrite results/knn-results.csv
# False: skip them and only run the final model

K_SWEEP = {"image": [1, 3, 5, 10, 20], "sentiment": [1, 3, 5, 10, 20, 50, 100, 200]}
POWER_KS = {"image": [5, 10, 20], "sentiment": [20, 50, 100]}
POWERS = [0, 1, 4, 16, 64]
COSINE_RUNS = [(50, 0), (50, 4), (100, 0), (100, 4)] # (k, power), sentiment only

KNN_RESULTS_FILE = os.path.join(RESULTS_DIR, "knn-results.csv")

# LDA experiment grid. An LDA projection keeps at most (number of classes - 1) dimensions.
LDA_DIMS = {"image": [10, 25, 49], "sentiment": [1, 2]}
LDA_SHRINKS = [0.1, 0.3, 1.0, 3.0, 10.0]
LDA_KS = {"image": [1, 5, 10, 20, 50], "sentiment": [5, 20, 50, 100, 200, 400]}

# Feature power transform (image only): exponent grid, and the k / voting power used with it
EXPONENTS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
EXPONENT_KS = [5, 10, 20]
EXPONENT_POWERS = [16, 64]

# Appended-LDA grid (image): original features + weight * LDA features
APPEND_SHRINKS = [0.3, 3.0, 30.0, 1000.0]
APPEND_WEIGHTS = [0.0, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 2.0] # 0.0 = the sqrt-only model, as a control
APPEND_KS = [10, 20]

# Best settings found on the validation set: (k, power)
FINAL = {"image": (20, 64), "sentiment": (200, 0)}
# Tasks where LDA is applied before k-NN: (dims, shrink). Image is left out because LDA hurt it (see group "lda").
FINAL_LDA = {"sentiment": (2, 0.1)}
# Tasks where features are raised to this exponent (then L2-normalized) before k-NN. Sentiment is left out: its features can be negative.
FINAL_EXPONENT = {"image": 0.5}
# Tasks where LDA features are appended to the (transformed) features: (dims, shrink, weight)
FINAL_APPEND = {"image": (49, 1000.0, 0.5)}

if __name__ == "__main__":
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # experiments
    if RUN_SWEEPS:
        clear_knn_results()

        # k sweep, raw vs standardized features (unweighted voting)
        print("\n--- k sweep: raw vs standardized ---")
        for std in (False, True):
            for task in TASKS:
                X_tr, y_tr, X_val, y_val, _, _ = get_task_data(task, standardize_features=std)
                for k in K_SWEEP[task]:
                    run_one(task, (X_tr, y_tr, X_val, y_val), k, 0.0, "euclidean", std, with_power=False,
                            group="k sweep, " + ("standardized" if std else "raw"))

        # weighted voting: power sweep on raw features
        print("\n--- power sweep ---")
        for task in TASKS:
            X_tr, y_tr, X_val, y_val, _, _ = get_task_data(task)
            for k in POWER_KS[task]:
                for p in POWERS:
                    run_one(task, (X_tr, y_tr, X_val, y_val), k, p, "euclidean", False, with_power=True,
                            group="power sweep")

        # cosine distance on sentiment (on images it equals Euclidean: rows already have length 1)
        print("\n--- cosine (sentiment) ---")
        X_tr, y_tr, X_val, y_val, _, _ = get_task_data("sentiment")
        for k, p in COSINE_RUNS:
            run_one("sentiment", (X_tr, y_tr, X_val, y_val), k, p, "cosine", False, with_power=True,
                    group="cosine")

        # LDA before k-NN, on both tasks (the projection is learned on the train split only)
        print("\n--- LDA + k-NN ---")
        for task in TASKS:
            X_tr, y_tr, X_val, y_val, _, _ = get_task_data(task)
            for dims in LDA_DIMS[task]:
                for shrink in LDA_SHRINKS:
                    for k in LDA_KS[task]:
                        for p in POWERS:
                            run_one(task, (X_tr, y_tr, X_val, y_val), k, p, "euclidean", False,
                                    with_power=True, group="lda", lda=(dims, shrink))
        
        # element-wise power transform on image features (validation split only)
        print("\n--- exponent (image) ---")
        X_tr, y_tr, X_val, y_val, _, _ = get_task_data("image")
        for e in EXPONENTS:
            for k in EXPONENT_KS:
                for p in EXPONENT_POWERS:
                    run_one("image", (X_tr, y_tr, X_val, y_val), k, p, "euclidean", False,
                            with_power=True, group="exponent", exponent=e)

        # sqrt features + appended LDA features (image only, validation split only)
        print("\n--- appended LDA (image) ---")
        X_tr, y_tr, X_val, y_val, _, _ = get_task_data("image")
        for shrink in APPEND_SHRINKS:
            for weight in APPEND_WEIGHTS:
                for k in APPEND_KS:
                    run_one("image", (X_tr, y_tr, X_val, y_val), k, 64, "euclidean", False, with_power=True,
                            group="append lda", exponent=0.5, append=(49, shrink, weight))

        mark_best()

        # Summary: best LDA row vs best non-LDA row, per task
        table = pd.read_csv(KNN_RESULTS_FILE)
        print("\n--- LDA vs no LDA (best validation accuracy per task) ---")
        for task in TASKS:
            rows = table[table["task"] == task]
            is_lda = rows["notes"].str.startswith("lda")
            print(f"{task:<10} without LDA {rows[~is_lda]['val_acc'].max():.4f} | with LDA {rows[is_lda]['val_acc'].max():.4f}")

    # final model, error analysis
    for task, (k, p) in FINAL.items():
        X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data(task)

        if task in FINAL_EXPONENT:
            X_tr, X_val = power_normalize(X_tr, FINAL_EXPONENT[task]), power_normalize(X_val, FINAL_EXPONENT[task])

        if task in FINAL_APPEND:
            dims, shrink, weight = FINAL_APPEND[task]
            X_tr, X_val = append_lda(X_tr, y_tr, X_val, NUM_CLASSES[task], dims, shrink, weight) # LDA learned on the train split only

        if task in FINAL_LDA:
            dims, shrink = FINAL_LDA[task]
            fitted = fit_lda(X_tr, y_tr, NUM_CLASSES[task], shrink) # learned on the train split only
            X_tr, X_val = apply_lda(X_tr, fitted, dims), apply_lda(X_val, fitted, dims)

        model = KNN(k=k, power=p)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        print("\n==", task, "| k =", k, "| power =", p, "| val accuracy:", round(accuracy(y_val, preds), 4))

        # validation predictions, in validation order, so teammates can compare mistakes
        out = os.path.join(RESULTS_DIR, f"val_preds_knn_{task}.csv")
        np.savetxt(out, preds, fmt="%d", header="pred", comments="")

        n = NUM_CLASSES[task]
        cm = confusion_matrix(y_val, preds, n) # rows = true class, columns = predicted
        precision, recall, f1 = per_class_metrics(cm)

        if task == "sentiment":
            print("confusion matrix (rows = true, columns = predicted; 0 neg, 1 neu, 2 pos):")
            print(cm)
            print("                neg    neu    pos")
            print("precision:    ", np.round(precision, 3))
            print("recall:       ", np.round(recall, 3))
            print("F1:           ", np.round(f1, 3))
        else:
            print("macro averages (mean over the 50 classes): precision", round(float(precision.mean()), 3),
            "| recall", round(float(recall.mean()), 3), "| F1", round(float(f1.mean()), 3))
            worst = np.argsort(recall)[:5]
            print("5 hardest classes (class, recall):", [(int(c), round(float(recall[c]), 2)) for c in worst])
            mistakes = cm.copy()
            np.fill_diagonal(mistakes, 0) # ignore the correct predictions
            top = np.argsort(mistakes.ravel())[::-1][:5]
            print("most common confusions (true -> predicted : count):")
            for idx in top:
                t, q = divmod(int(idx), n)
                print(f"   {t} -> {q} : {mistakes[t, q]}")

    # retrain on all labeled data, predict the test set
    print()
    test_preds = {}
    test_ids_by_task = {}
    for task, (k, p) in FINAL.items():
        X_all, y_all, X_test, test_ids = load_task(task) # every labeled row, no split

        if task in FINAL_EXPONENT: # no fitting involved, so the same formula is applied to train and test
            X_all, X_test = power_normalize(X_all, FINAL_EXPONENT[task]), power_normalize(X_test, FINAL_EXPONENT[task])

        if task in FINAL_APPEND:
            dims, shrink, weight = FINAL_APPEND[task]
            X_all, X_test = append_lda(X_all, y_all, X_test, NUM_CLASSES[task], dims, shrink, weight) # labeled data only, never the test set

        if task in FINAL_LDA:
            dims, shrink = FINAL_LDA[task]
            fitted = fit_lda(X_all, y_all, NUM_CLASSES[task], shrink)  # labeled data only, never the test set
            X_all, X_test = apply_lda(X_all, fitted, dims), apply_lda(X_test, fitted, dims)

        model = KNN(k=k, power=p)
        model.fit(X_all, y_all)
        test_preds[task] = model.predict(X_test)
        test_ids_by_task[task] = test_ids

        out = os.path.join(RESULTS_DIR, f"test_preds_knn_{task}.csv")
        pd.DataFrame({"id": test_ids, "label": test_preds[task]}).to_csv(out, index=False)
        print(task, "test predictions:", len(test_preds[task]),
            "| distinct labels used:", len(np.unique(test_preds[task])))

    build_submission(test_ids_by_task["image"], test_preds["image"],
                    test_ids_by_task["sentiment"], test_preds["sentiment"],
                    out_name="sub_knn_v4.csv")