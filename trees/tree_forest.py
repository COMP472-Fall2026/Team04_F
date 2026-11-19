import numpy as np
import pandas as pd
import math
import random
from collections import Counter

from common import get_task_data, accuracy, set_seed

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None,right=None,value=None):
        self.feature_index = feature_index
        self.threshold = threshold #Threshold(Symbol: S), is the pivot value for our splitting
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,criterion="gini",max_features=None,random_state=None):
        self.max_depth = max_depth #Deepest we allow the tree to go (Deeper trees typically have more overfitting)
        self.min_samples_split = min_samples_split #Minimum samples required to split a node
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion #How splits are evaluated
        self.max_features = max_features
        self.random_state = random_state #Seed for this tree's own random feature sampling (None = not reproducible)

    def fit(self, X, y):
        if not isinstance(self.criterion, str) or self.criterion.lower() not in ("gini", "entropy"):
            raise ValueError("criterion must be either 'gini' or 'entropy'")
        self.criterion = self.criterion.lower()

        if self.max_depth is not None and (
            not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative integer or None")
        if not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")

        # Remember DataFrame feature names so prediction always uses the same
        # columns in the same order.  In particular, id and label are excluded
        # even when they happen to have numeric dtypes.
        if isinstance(X, pd.DataFrame):
            self.feature_names = [
                column for column in X.columns
                if str(column).lower() not in ("id", "label")
                and pd.api.types.is_numeric_dtype(X[column])
            ]
            if not self.feature_names:
                raise ValueError("X must contain at least one numerical feature column")
            X = X.loc[:, self.feature_names].to_numpy()
        else:
            self.feature_names = None
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_)):
            raise TypeError("X must contain only numerical features")
        if np.issubdtype(X.dtype, np.complexfloating):
            raise TypeError("complex-valued features are not supported")
        if not np.isfinite(X).all():
            raise ValueError("X cannot contain NaN or infinite values")

        y = np.asarray(y)
        if y.ndim == 2 and 1 in y.shape:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(y) != len(X):
            raise ValueError("X and y must contain the same number of samples")
        if pd.isna(y).any():
            raise ValueError("y cannot contain missing labels")

        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        # Resolve the number of candidate features once, then draw that many
        # without replacement at every node.
        if self.max_features is None:
            self._max_features = self.n_features
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                self._max_features = max(1, int(math.sqrt(self.n_features)))
            elif self.max_features == "log2":
                self._max_features = max(1, int(math.log2(self.n_features)))
            else:
                raise ValueError("max_features must be None, an int, a float, 'sqrt', or 'log2'")
        elif isinstance(self.max_features, (int, np.integer)) and not isinstance(self.max_features, bool):
            if self.max_features <= 0:
                raise ValueError("max_features must be positive")
            self._max_features = min(int(self.max_features), self.n_features)
        elif isinstance(self.max_features, (float, np.floating)):
            if not 0 < self.max_features <= 1:
                raise ValueError("a float max_features must be in the interval (0, 1]")
            self._max_features = max(1, int(self.max_features * self.n_features))
        else:
            raise ValueError("max_features must be None, an int, a float, 'sqrt', or 'log2'")

        # Each tree owns its random generator, so its feature draws depend only
        # on random_state, not on how many other trees ran before it.
        self._rng = np.random.default_rng(self.random_state)

        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, "root"):
            raise ValueError("the decision tree must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                missing_columns = [column for column in self.feature_names if column not in X.columns]
                if missing_columns:
                    raise ValueError(f"X is missing training feature columns: {missing_columns}")
                if any(not pd.api.types.is_numeric_dtype(X[column]) for column in self.feature_names):
                    raise TypeError("training feature columns must remain numerical")
                X = X.loc[:, self.feature_names].to_numpy()
            else:
                feature_columns = [
                    column for column in X.columns
                    if str(column).lower() not in ("id", "label")
                    and pd.api.types.is_numeric_dtype(X[column])
                ]
                X = X.loc[:, feature_columns].to_numpy()
        else:
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must contain exactly {self.n_features} features")
        if not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_)):
            raise TypeError("X must contain only numerical features")
        if np.issubdtype(X.dtype, np.complexfloating):
            raise TypeError("complex-valued features are not supported")
        if not np.isfinite(X).all():
            raise ValueError("X cannot contain NaN or infinite values")

        if len(X) == 0:
            return np.asarray([], dtype=self.classes.dtype)
        return np.asarray([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_labels = len(np.unique(y))

        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        if (
            reached_max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split
            or num_samples < 2 * self.min_samples_leaf
        ):
            return Node(value=self._most_common_label(y))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return Node(value=self._most_common_label(y))

        left_indices, right_indices = self._split(X, feature_index, threshold)
        y_left, y_right = y[left_indices], y[right_indices]

        # This is defensive because _best_split already enforces these rules.
        if (
            len(left_indices) < self.min_samples_leaf
            or len(right_indices) < self.min_samples_leaf
            or self._information_gain(y, y_left, y_right) <= 1e-12
        ):
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_indices], y_left, depth + 1)
        right = self._grow_tree(X[right_indices], y_right, depth + 1)
        return Node(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples < 2 * self.min_samples_leaf:
            return None, None

        max_features = getattr(self, "_max_features", num_features)
        if max_features < num_features:
            feature_indices = np.sort(
                self._rng.choice(num_features, max_features, replace=False)
            )
        else:
            feature_indices = np.arange(num_features)

        _, encoded_y = np.unique(y, return_inverse=True)
        total_counts = np.bincount(encoded_y)
        parent_impurity = self._gini(y) if self.criterion == "gini" else self._entropy(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        sample_positions = np.arange(num_samples)

        if self.criterion == "gini":
            total_square_sum = np.sum(total_counts.astype(float) ** 2)
        else:
            # count_log_count[k] stores k * log2(k), with 0 * log2(0) = 0.
            count_log_count = np.zeros(num_samples + 1, dtype=float)
            positive_counts = np.arange(1, num_samples + 1)
            count_log_count[1:] = positive_counts * np.log2(positive_counts)
            total_count_log_sum = np.sum(count_log_count[total_counts])

        for feature_index in feature_indices:
            order = np.argsort(X[:, feature_index], kind="stable")
            sorted_values = X[order, feature_index]

            # Splits between equal adjacent values are invalid because they do
            # not produce the row counts implied by their sorted positions.
            left_sizes = np.arange(1, num_samples)
            right_sizes = num_samples - left_sizes
            valid_splits = (
                (sorted_values[:-1] < sorted_values[1:])
                & (left_sizes >= self.min_samples_leaf)
                & (right_sizes >= self.min_samples_leaf)
            )
            if not np.any(valid_splits):
                continue

            sorted_labels = encoded_y[order]

            # The occurrence rank of each sorted label lets us update impurity
            # in O(n) vectorized work without constructing an n-by-class matrix.
            label_order = np.argsort(sorted_labels, kind="stable")
            grouped_labels = sorted_labels[label_order]
            group_starts = np.empty(num_samples, dtype=bool)
            group_starts[0] = True
            group_starts[1:] = grouped_labels[1:] != grouped_labels[:-1]
            start_positions = np.maximum.accumulate(
                np.where(group_starts, sample_positions, 0)
            )
            grouped_ranks = sample_positions - start_positions + 1
            ranks = np.empty(num_samples, dtype=np.int64)
            ranks[label_order] = grouped_ranks
            class_totals = total_counts[sorted_labels]

            if self.criterion == "gini":
                left_square_sum = np.cumsum(2 * ranks - 1)
                right_square_sum = total_square_sum + np.cumsum(
                    -2 * class_totals + 2 * ranks - 1
                )
                weighted_impurity = (
                    left_sizes - left_square_sum[:-1] / left_sizes
                    + right_sizes - right_square_sum[:-1] / right_sizes
                ) / num_samples
            else:
                left_count_log_sum = np.cumsum(
                    count_log_count[ranks] - count_log_count[ranks - 1]
                )
                remaining_counts = class_totals - ranks
                right_count_log_sum = total_count_log_sum + np.cumsum(
                    count_log_count[remaining_counts]
                    - count_log_count[remaining_counts + 1]
                )
                weighted_impurity = (
                    count_log_count[left_sizes] - left_count_log_sum[:-1]
                    + count_log_count[right_sizes] - right_count_log_sum[:-1]
                ) / num_samples

            gains = parent_impurity - weighted_impurity
            candidate_positions = np.flatnonzero(valid_splits)
            local_position = candidate_positions[np.argmax(gains[candidate_positions])]
            local_gain = float(gains[local_position])

            if local_gain > best_gain + 1e-12:
                best_gain = local_gain
                best_feature = int(feature_index)
                # With <= on the left, the lower observed value represents
                # this boundary exactly. A midpoint can round up for adjacent
                # floats (or lose precision for large integers).
                best_threshold = sorted_values[local_position].item()

        return best_feature, best_threshold

    def _split(self, X, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices

    def _information_gain(self, y, y_left, y_right):
        if len(y) == 0 or len(y_left) == 0 or len(y_right) == 0:
            return 0.0

        impurity = self._gini if self.criterion == "gini" else self._entropy
        parent_impurity = impurity(y)
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)
        child_impurity = left_weight * impurity(y_left) + right_weight * impurity(y_right)
        return float(parent_impurity - child_impurity)

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(1.0 - np.sum(probabilities ** 2))

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _most_common_label(self, y):
        label_counts = Counter(y)
        largest_count = max(label_counts.values())
        # np.unique sorts the project labels, giving deterministic tie-breaking.
        return next(label for label in np.unique(y) if label_counts[label] == largest_count)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    set_seed()

    # Evaluate image first, followed by sentiment when both datasets exist.
    for task in ("image", "sentiment"):
        try:
            X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data(task)
        except FileNotFoundError as error:
            print("Decision Tree Results")
            print("---------------------")
            print(f"Dataset: {task}")
            print(f"Dataset unavailable: {error.filename}")
            print()
            continue

        tree = DecisionTree()
        tree.fit(X_tr, y_tr)

        train_predictions = tree.predict(X_tr)
        val_predictions = tree.predict(X_val)
        test_predictions = tree.predict(X_test)

        train_accuracy = accuracy(y_tr, train_predictions)
        val_accuracy = accuracy(y_val, val_predictions)

        print("Decision Tree Results")
        print("---------------------")
        print(f"Dataset: {task}")
        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print("Test accuracy: N/A (test labels unavailable)")
        print(f"Test predictions generated: {len(test_predictions)}")
        print()