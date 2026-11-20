import numpy as np
import pandas as pd

from trees.decision_tree import DecisionTree


class RandomForest:
    """Random Forest classifier built from our own DecisionTree class."""

    def __init__(self, n_estimators=100, max_depth=40, min_samples_split=20, min_samples_leaf=5, criterion="gini", max_features="sqrt",bootstrap=True, random_state=None, tree_groups=None):
        self.n_estimators = n_estimators #Number of decision trees we create for the forest
        self.max_depth = max_depth #Deepest we allow each tree to go (Deeper trees typically have more overfitting)
        self.min_samples_split = min_samples_split #Minimum samples required to split a node within any tree
        self.min_samples_leaf = min_samples_leaf #Minimum samples required per potential child for a node to be allowed
        self.criterion = criterion #How the splits within each tree are evaluated
        self.max_features = max_features #The maximum number of features considered when searching for the best split at each node
        self.bootstrap = bootstrap #Whether every tree trains on a random sample taken with replacement
        self.random_state = random_state #Seed used to make our row sampling and individual tree seeds reproducible
        self.trees = [] #Stores every fitted DecisionTree object belonging to our forest
        self.tree_groups = tree_groups

    def _prepare_features(self, X, fitting=False):
        """Used to prepare the numerical features in X in the same order as the training data"""
        #DataFrames can contain identifier, label, or nonnumerical columns that our trees should not use
        if isinstance(X, pd.DataFrame):
            if fitting or self.feature_names is None:
                #When fitting, find every numerical column except those reserved for ids and labels
                feature_names = [
                    column for column in X.columns
                    if str(column).lower() not in ("id", "label")
                    and pd.api.types.is_numeric_dtype(X[column])
                ]
            else:
                feature_names = self.feature_names #Use the exact same features and order we saved while fitting
                missing = [column for column in feature_names if column not in X.columns] #Find any training features that are no longer within X
                if missing:
                    raise ValueError(f"X is missing training feature columns: {missing}")
                if any(not pd.api.types.is_numeric_dtype(X[column]) for column in feature_names):
                    raise TypeError("training feature columns must remain numerical")
            if fitting:
                self.feature_names = feature_names #Keep the names of the numerical features we used during training
            X = X.loc[:, feature_names].to_numpy() #Select the features in their required order and convert them to an np array
        else:
            if fitting:
                self.feature_names = None #Numpy arrays do not provide feature names for us to save
            X = np.asarray(X) #Ensure the collection passed behaves as an np array

        #Make sure X still has the shape and values expected by our decision trees
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
        return X #Return only the prepared feature matrix used by our forest

    def fit(self, X, y):
        """Encapsulate the random forest training process by fitting independent decision trees to samples of X and y"""
        self.trees = [] #Remove any trees left from a previous fit

        #Ensure the values controlling the amount of trees and row sampling are valid
        if (not isinstance(self.n_estimators, (int, np.integer)) or isinstance(self.n_estimators, (bool, np.bool_)) or self.n_estimators < 1):
            raise ValueError("n_estimators must be a positive integer")
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean")

        #Build the parameters for every tree, applying group overrides in the order provided
        tree_parameters = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "max_features": self.max_features
        }
        tree_configurations = None
        if self.tree_groups is not None:
            if not isinstance(self.tree_groups, (list, tuple)):
                raise ValueError("tree_groups must be a list or tuple of dictionaries")

            allowed_group_keys = {"count", *tree_parameters}
            tree_configurations = []
            total_count = 0
            for group_index, group in enumerate(self.tree_groups):
                if not isinstance(group, dict):
                    raise ValueError(f"tree_groups[{group_index}] must be a dictionary")

                unknown_keys = set(group) - allowed_group_keys
                if unknown_keys:
                    names = ", ".join(sorted(str(key) for key in unknown_keys))
                    raise ValueError(f"tree_groups[{group_index}] contains unknown hyperparameter(s): {names}")
                if "count" not in group:
                    raise ValueError(f"tree_groups[{group_index}] must contain a count")

                count = group["count"]
                if (not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)) or count < 1):
                    raise ValueError(f"tree_groups[{group_index}]['count'] must be a positive integer")

                group_parameters = tree_parameters.copy()
                group_parameters.update({key: value for key, value in group.items() if key != "count"})
                tree_configurations.append((int(count), group_parameters))
                total_count += int(count)

            if total_count != self.n_estimators:
                raise ValueError(
                    f"tree group counts must sum to n_estimators ({self.n_estimators}); got {total_count}"
                )
            tree_configurations = [
                parameters
                for count, parameters in tree_configurations
                for _ in range(count)
            ]

        X = self._prepare_features(X, fitting=True) #Prepare and remember the features our trees will be fitted with
        y = np.asarray(y) #Ensure the labels passed behave as an np array
        if y.ndim == 2 and 1 in y.shape:
            y = y.reshape(-1) #Turn a single row or column of labels into a one-dimensional collection

        #X and y must describe the same amount of training entries
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match the number of samples in X")
        if pd.isna(y).any():
            raise ValueError("y cannot contain missing labels")

        self.n_features = X.shape[1] #Number of features every future entry must contain
        self.classes = np.unique(y) #Contains each possible label once and in sorted order
        num_samples = len(y) #Number of entries available for each bootstrap sample

        #Separate the random streams so sampling rows does not affect the seeds given to each tree
        #With the same random state, the first k trees stay identical even if we ask our forest to contain more trees
        bootstrap_seed, feature_seed = np.random.SeedSequence(self.random_state).spawn(2)
        bootstrap_rng = np.random.default_rng(bootstrap_seed) #Random number generator used to select the training rows
        feature_rng = np.random.default_rng(feature_seed) #Random number generator used to create a seed for every tree

        trees = [] #Temporarily stores the trees until the entire forest is fitted

        #Create the requested amount of decision trees
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = bootstrap_rng.choice(num_samples, size=num_samples, replace=True) #Sample row indexes while allowing the same index to appear multiple times
                X_sample, y_sample = X[indices], y[indices] #Retrieve the entries and their labels according to our sampled indexes
            else:
                X_sample, y_sample = X, y #Give the tree every training entry when bootstrap sampling is disabled

            tree_seed = int(feature_rng.integers(0, 2**63)) #Give each tree its own random seed

            #Create a decision tree according to the restrictions given to our forest and its group
            parameters = tree_parameters if tree_configurations is None else tree_configurations[i]
            tree = DecisionTree(
                max_depth=parameters["max_depth"],
                min_samples_split=parameters["min_samples_split"],
                min_samples_leaf=parameters["min_samples_leaf"],
                criterion=parameters["criterion"],
                max_features=parameters["max_features"],
                random_state=tree_seed
            )
            tree.fit(X_sample, y_sample) #Fit the tree using its own collection of training entries
            trees.append(tree) #Save the fitted tree as another voter within our forest

        self.trees = trees #Only keep the complete collection after every tree is successfully fitted
        return self #Not a necessary return, just allows for chaining such as *** forest = RandomForest().fit(X, y) ***

    def predict(self, X):
        """Used to make predictions for every element in X according to the label receiving the most tree votes"""
        if not self.trees:
            raise ValueError("the random forest must be fitted before prediction")

        X = self._prepare_features(X) #Use the same numerical features and order as our training data
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must contain exactly {self.n_features} features")

        #Given a sample index and class index, votes tells us how many trees predicted that class
        votes = np.zeros((len(X), len(self.classes)), dtype=np.int64)
        sample_indices = np.arange(len(X)) #Contains every row index so we can update all their votes at once

        #Allow each tree to make one prediction for every entry in X
        for tree in self.trees:
            predictions = tree.predict(X) #Contains the label selected by this tree for every entry

            #Map every predicted label to its column even if the tree did not see all possible classes while fitting
            class_indices = np.searchsorted(self.classes, predictions) #Returns the index of each prediction within our sorted classes
            votes[sample_indices, class_indices] += 1 #Give the predicted class one additional vote for every entry

        #np.unique() sorts the labels, so np.argmax() returns the first label if there is a tie
        return self.classes[np.argmax(votes, axis=1)]

    
if __name__ == "__main__":
    from pathlib import Path
    from common import load_task, set_seed, SUBMISSIONS_DIR

    set_seed()

    # 1. Select the dataset
    task = "image"  # Change to "sentiment" if needed

    # 2. Retrieve the dataset
    X_train, y_train, X_test, test_ids = load_task(task)

    # 3. Create the Random Forest
    forest = RandomForest(
        n_estimators=200,
        max_depth=60,
        min_samples_split=20,
        min_samples_leaf=5,
        criterion="gini",
        max_features="sqrt",
        bootstrap=False,
        random_state=42
    )

    # 4. Train the Random Forest
    forest.fit(X_train, y_train)

    # 5. Predict the test set
    predictions = forest.predict(X_test)

    # 6. Save predictions to CSV
    output_dir = Path(SUBMISSIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"random_forest_{task}_predictions.csv"

    pd.DataFrame({
        "id": test_ids,
        "label": predictions.astype(int)
    }).to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} predictions to {output_path}")

