import numpy as np
import pandas as pd

from trees.tree_forest import DecisionTree


class RandomForest:
    """Random Forest classifier built from our own DecisionTree class."""

    def __init__(self, n_estimators=100, max_depth=40, min_samples_split=20, min_samples_leaf=5, criterion="gini", max_features="sqrt",bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def _prepare_features(self, X, fitting=False):
        """Use numerical features in the same order as during training."""
        if isinstance(X, pd.DataFrame):
            if fitting or self.feature_names is None:
                feature_names = [
                    column for column in X.columns
                    if str(column).lower() not in ("id", "label")
                    and pd.api.types.is_numeric_dtype(X[column])
                ]
            else:
                feature_names = self.feature_names
                missing = [column for column in feature_names if column not in X.columns]
                if missing:
                    raise ValueError(f"X is missing training feature columns: {missing}")
                if any(not pd.api.types.is_numeric_dtype(X[column]) for column in feature_names):
                    raise TypeError("training feature columns must remain numerical")
            if fitting:
                self.feature_names = feature_names
            X = X.loc[:, feature_names].to_numpy()
        else:
            if fitting:
                self.feature_names = None
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if X.shape[1] == 0 or (fitting and len(X) == 0):
            raise ValueError("X must contain features and, during fitting, samples")
        if not (np.issubdtype(X.dtype, np.number) or np.issubdtype(X.dtype, np.bool_)):
            raise TypeError("X must contain only numerical features")
        if np.issubdtype(X.dtype, np.complexfloating):
            raise TypeError("complex-valued features are not supported")
        if not np.isfinite(X).all():
            raise ValueError("X cannot contain NaN or infinite values")
        return X

    def fit(self, X, y):
        """Train independent decision trees on bootstrap samples of X and y."""
        self.trees = []
        if (not isinstance(self.n_estimators, (int, np.integer))
                or isinstance(self.n_estimators, (bool, np.bool_))
                or self.n_estimators < 1):
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean")

        X = self._prepare_features(X, fitting=True)
        y = np.asarray(y)
        if y.ndim == 2 and 1 in y.shape:
            y = y.reshape(-1)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match the number of samples in X")
        if pd.isna(y).any():
            raise ValueError("y cannot contain missing labels")

        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        num_samples = len(y)

        # Separate random streams keep tree seeds independent of row sampling.
        # With a fixed seed, the first k trees are reproducible for any forest size.
        bootstrap_seed, feature_seed = np.random.SeedSequence(self.random_state).spawn(2)
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        feature_rng = np.random.default_rng(feature_seed)

        trees = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = bootstrap_rng.choice(num_samples, size=num_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            tree_seed = int(feature_rng.integers(0, 2**63))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_sample, y_sample)
            trees.append(tree)

        self.trees = trees
        return self

    def predict(self, X):
        """Predict each sample's class using one equal vote per tree."""
        if not self.trees:
            raise ValueError("the random forest must be fitted before prediction")

        X = self._prepare_features(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must contain exactly {self.n_features} features")

        # Rows correspond to samples and columns to the forest's class labels.
        votes = np.zeros((len(X), len(self.classes)), dtype=np.int64)
        sample_indices = np.arange(len(X))

        for tree in self.trees:
            predictions = tree.predict(X)
            # Map labels to columns, even if some classes are absent in a tree.
            class_indices = np.searchsorted(self.classes, predictions)
            votes[sample_indices, class_indices] += 1

        # np.unique sorts classes; argmax selects the first in case of a tie.
        return self.classes[np.argmax(votes, axis=1)]
