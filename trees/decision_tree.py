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
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,max_features=None, criterion="entropy"):
        if not isinstance(criterion, str) or criterion.lower() not in ("gini", "entropy"):
            raise ValueError("criterion must be either 'gini' or 'entropy'")

        self.root = None
        self.max_depth = max_depth #Deepest we allow the tree to go (Deeper trees typically have more overfitting)
        self.min_samples_split = min_samples_split #Minimum samples required to split a node
        self.min_samples_leaf = min_samples_leaf #Minimum samples required per potential child for a node to be allowed
        self.max_features = max_features #The maximum number of features considered when searching for the best split at each node (They change for each possible split)
        self.criterion = criterion.lower() #How splits are evaluated

    def fit(self, X, y):
        """Encapsulate the decision tree training process with the methods within its body"""
        self.root = self._grow_tree(X, y, depth=0)
        return self #Not a neccesary return, just allows for chaining such as *** tree = DecisionTree().fit(X, y) ***

    def predict(self, X):
        """Used to make predictions for every element in the collection X. Assumes the elements of X have the exact same features and label as the training data"""
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction.")

        predictions = [self._traverse_tree(x, self.root) for x in X]

        return np.array(predictions)

    def _grow_tree(self, X, y, depth=0): 
        """Handles tree expansion and children creation according to the necessary conditions"""
        number_of_samples = X.shape[0] #.shape returns a tuple containing (nb_of_entires, nb_of_features_per_entry)
        number_of_labels = len(np.unique(y)) #number of unique labels

        #3 stopping conditions to stop tree growth
        if (depth >= self.max_depth or number_of_labels == 1 or number_of_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature_index, best_threshold = self._best_split(X, y)

        #If the best gain is 0, set the leaf value to that of the most common label we have
        if best_feature_index is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_indices, right_indices = self._split(X, best_feature_index, best_threshold) #save the indicies of the left children and the right children according to our best threshold

        #Recursively run the same process with the two children
        left_child = self._grow_tree(X[left_indices],y[left_indices],depth + 1) 
        right_child = self._grow_tree(X[right_indices],y[right_indices],depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_child, right=right_child) #Return the reference to our root node 

    def _best_split(self, X, y):
        """Based on all possible features with the X passed, what is the best split (feature to use + threshold)"""
        number_of_samples, number_of_features = X.shape #.shape returns a tuple containing (nb_of_entires, nb_of_features_per_entry)

        unique_labels, label_codes = np.unique(y, return_inverse=True) #label_codes = array of length y. The value at a given index indicates the index of its associated unique label from unique_labels
        n_classes = len(unique_labels) #Tells us how many distinct classes there are

        total_counts = np.bincount(label_codes, minlength=n_classes) #Given an index n, the total_counts array will tell us how many times n appeared in the label_codes at index n

        best_gain = 0
        best_feature_index = None
        best_threshold = None

        parent_impurity = self._impurity_from_counts(total_counts)

        for feature_index in range(number_of_features):
            sorted_indices = np.argsort(X[:, feature_index]) #Sorted_indices according to the given feature
            sorted_values = X[sorted_indices, feature_index] #Sorted_values contains the feature values associated to each index in sorted_indices
            sorted_labels = label_codes[sorted_indices] #Sorted_labels contains the label values associated to each index in sorted_indices

            #Initially, all samples are on the right.
            #These arrays are the same structure as total counts. There lengths represent the amount of features and the values at each index indicate how much they appear
            left_counts = np.zeros(n_classes, dtype=int)
            right_counts = total_counts.copy()

            for i in range(number_of_samples - 1): #No need to consider the last entry because we know we cannot leave the right empty

                label = sorted_labels[i]
                left_counts[label] += 1
                right_counts[label] -= 1

                #Equal feature values cannot be separated by a threshold.
                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                n_left = i + 1
                n_right = number_of_samples - n_left

                # Respect the minimum leaf size.
                if (n_left < self.min_samples_leaf or n_right < self.min_samples_leaf):
                    continue

                gain = self._information_gain(parent_impurity, left_counts, right_counts)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

        return best_feature_index, best_threshold

    def _split(self, X, feature_index, threshold):
        """Splits X into two arrays according to some threshold. Returns 2 arrays with each containing the indexes of the element that split accordingly (one contains the elements returning true, the other false)"""
        feature_column = X[:, feature_index] #Give me an np array containing every rows' feature column value

        left_indices = np.where(feature_column <= threshold)[0] #np.where() returns a tuple containing a list with all the indexes that returned true
        right_indices = np.where(feature_column > threshold)[0] #np.where() returns a tuple containing a list with all the indexes that returned false

        return left_indices, right_indices

    def _information_gain(self, parent_impurity, left_counts, right_counts):
        """Returns the impurity reduction for the selected split criterion."""

        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)
        n_total = n_left + n_right

        if n_left == 0 or n_right == 0:
            return 0.0

        left_impurity = self._impurity_from_counts(left_counts)
        right_impurity = self._impurity_from_counts(right_counts)
        weighted_child_impurity = ((n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity)

        return parent_impurity - weighted_child_impurity

    def _gini(self, y):
        """Standard Gini formula used to calculate the impurity of values within y"""
        if len(y) == 0:
            return 0.0

        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """Standard entropy formula used to calculate the randomness of values within y"""
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _impurity_from_counts(self, counts):
        """Impurity calculation from class counts using the selected criterion"""
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) == 0:
            return 0.0

        probabilities = nonzero_counts / nonzero_counts.sum()
        if self.criterion == "gini":
            return 1 - np.sum(probabilities ** 2)

        return -np.sum(probabilities * np.log2(probabilities))

    def _most_common_label(self, y):
        """Finds the name of the most common label entry in y"""
        label_counts = Counter(y) #Creates a counter objects with all the labels being passed
        return label_counts.most_common(1)[0][0] # .most_common(1) returns the most common label in a tuple in a list, [0][0] exracts the label itself

    def _traverse_tree(self, x, node):
        """Recursive method designed to find which node an element x would belong to. Used to make predictions"""

        if node.value is not None: #Stop when you reach a leaf node
            return node.value

        if x[node.feature_index] <= node.threshold: #Does x belong in the left child?
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right) #Does x belong in the right child?





if __name__ == "__main__":
    set_seed()

    # 1. Select the dataset
    task = "image"  # Change to "sentiment" if needed

    # 2. Retrieve the dataset
    X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data(task)

    # 3. Create the decision tree
    tree = DecisionTree(
        max_depth=60,
        min_samples_split=80,
        min_samples_leaf=10,
        criterion="gini",
        max_features=None
    )

    # 4. Train the decision tree on all labeled data
    X_train = np.concatenate((X_tr, X_val), axis=0)
    y_train = np.concatenate((y_tr, y_val), axis=0)

    tree.fit(X_train, y_train)

    # 5. Predict the test set
    predictions = tree.predict(X_test)

    # 6. Save predictions to CSV
    from pathlib import Path
    import pandas as pd

    output_dir = Path("submissions")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"decision_tree_{task}_predictions.csv"

    pd.DataFrame({
        "id": test_ids,
        "label": predictions.astype(int)
    }).to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} predictions to {output_path}")



