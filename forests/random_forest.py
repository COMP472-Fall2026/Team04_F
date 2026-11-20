import numpy as np
import pandas as pd

from trees.decision_tree import DecisionTree


class RandomForest:
    """Random Forest classifier built from our own DecisionTree class."""

    def __init__(self, n_estimators=100, max_depth=40, min_samples_split=20, min_samples_leaf=5, criterion="gini", max_features="sqrt",bootstrap=True, random_state=None):
        self.n_estimators = n_estimators #Number of decision trees we create for the forest
        self.max_depth = max_depth #Deepest we allow each tree to go (Deeper trees typically have more overfitting)
        self.min_samples_split = min_samples_split #Minimum samples required to split a node within any tree
        self.min_samples_leaf = min_samples_leaf #Minimum samples required per potential child for a node to be allowed
        self.criterion = criterion #How the splits within each tree are evaluated
        self.max_features = max_features #The maximum number of features considered when searching for the best split at each node
        self.bootstrap = bootstrap #Whether every tree trains on a random sample taken with replacement
        self.random_state = random_state #Seed used to make our row sampling and individual tree seeds reproducible
        self.trees = [] #Stores every fitted DecisionTree object belonging to our forest

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

            #Create a decision tree according to the restrictions given to our forest
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self.max_features,
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
