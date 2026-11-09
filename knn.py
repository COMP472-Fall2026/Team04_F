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
        self.b_dot_b = np.sum(x ** 2, axis=1)   # b·b, computed once

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
                # inverse: for every neighbour, which entry of `labels` it belongs to
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


if __name__ == "__main__":
    set_seed()

    # Best settings found on the validation set: (k, power)
    final = {"image": (20, 64), "sentiment": (100, 4)}

    for task, (k, p) in final.items():
        X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data(task)

        model = KNN(k=k, power=p)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        print("\n==", task, "| k =", k, "| power =", p, "| val accuracy:", round(accuracy(y_val, preds), 4))

        # Save predictions (same row order as the validation split) so teammates
        # can compare which mistakes different models share.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"val_preds_knn_{task}.csv")
        np.savetxt(out, preds, fmt="%d", header="pred", comments="")

        n = NUM_CLASSES[task]
        cm = confusion_matrix(y_val, preds, n)          # rows = true class, columns = predicted
        recall = cm.diagonal() / cm.sum(axis=1)         # fraction of each class found

        if task == "sentiment":
            print("confusion matrix (rows = true, columns = predicted; 0 neg, 1 neu, 2 pos):")
            print(cm)
            print("recall per class (neg, neu, pos):", np.round(recall, 3))
        else:
            worst = np.argsort(recall)[:5]
            print("5 hardest classes (class, recall):", [(int(c), round(float(recall[c]), 2)) for c in worst])
            mistakes = cm.copy()
            np.fill_diagonal(mistakes, 0)                # ignore the correct predictions
            top = np.argsort(mistakes.ravel())[::-1][:5]
            print("most common confusions (true -> predicted : count):")
            for idx in top:
                t, q = divmod(int(idx), n)
                print(f"   {t} -> {q} : {mistakes[t, q]}")

        # ---- Final model: retrain on ALL labeled data, then predict the hidden test set ----
        test_preds = {}
        test_ids_by_task = {}
        for task, (k, p) in final.items():
            X_all, y_all, X_test, test_ids = load_task(task)      # every labeled row, no split
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
                        out_name="sub_knn_v1.csv")