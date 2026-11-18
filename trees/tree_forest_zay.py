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
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,max_features=None):
        self.root = None
        self.max_depth = max_depth #Deepest we allow the tree to go (Deeper trees typically have more overfitting)
        self.min_samples_split = min_samples_split #Minimum samples required to split a node
        self.min_samples_leaf = min_samples_leaf #Minimum samples required per potential child for a node to be allowed
        self.max_features = max_features #The maximum number of features considered when searching for the best split at each node (They change for each possible split)

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
        """Based on all possible features with the X passed, what is roughly (only rough cuz thresholds are not gone through exhaustively) the best split (feature to use + threshold)"""
        number_of_features = X.shape[1] #.shape returns a tuple containing (nb_of_entires, nb_of_features_per_entry)

        best_gain = 0
        best_feature_index = None
        best_threshold = None

        for feature_index in range(number_of_features):
            feature_column = X[:, feature_index] #Give me an np array containing every rows' feature column value
            unique_values = np.unique(feature_column) #Gives up an np array with ONLY unique values

            if len(unique_values) < 2: #If the feature has < 2 entered values among all the training sample records, then its a useless split since one child will be empty
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2 #Generates appropriate threshold values (better than just trying every threshold between min and max) that will be systematically tested to find the best one

            for threshold in thresholds: #Testing every threshold
                left_indices, right_indices = self._split(X, feature_index, threshold)

                if (len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf): #Condition check ensuring the minimum amount of samples per node is met
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain: #Iterate through each threshold and find the best gain (We save the value of the gain, the feature used, and the threshold used)
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _split(self, X, feature_index, threshold):
        """Splits X into two arrays according to some threshold. Returns 2 arrays with each containing the indexes of the element that split accordingly (one contains the elements returning true, the other false)"""
        feature_column = X[:, feature_index] #Give me an np array containing every rows' feature column value

        left_indices = np.where(feature_column <= threshold)[0] #np.where() returns a tuple containing a list with all the indexes that returned true
        right_indices = np.where(feature_column > threshold)[0] #np.where() returns a tuple containing a list with all the indexes that returned false

        return left_indices, right_indices

    def _information_gain(self, y, y_left, y_right):
        """Returns the information gained according to the split of labels in y_left and y_right with y being their parent"""
        #y_left and y_right returns each have the LABELS of the elements that are associated to the indexes found in left_indicies and right_indicies
        
        if len(y_left) == 0 or len(y_right) == 0: #Basically saying that there's literally no value in the split because one of the children is dead
            return 0

        parent_entropy = self._entropy(y)

        left_weight = len(y_left) / len(y) #Fraction representing how many of our entries are going left
        right_weight = len(y_right) / len(y) #Fraction representing how many of our entries are going right 

        conditional_entropy = (left_weight * self._entropy(y_left) + right_weight * self._entropy(y_right))

        information_gain = parent_entropy - conditional_entropy

        return information_gain

    def _entropy(self, y):
        """Standard entropy formula used to calculate the randomness of values within y"""
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
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

    for task in ["image", "sentiment"]:
        #1. Retrieve the dataset
        X_tr, y_tr, X_val, y_val, X_test, test_ids = get_task_data(task)

        #2. Create the decision tree
        tree = DecisionTree(max_depth=10)

        #3. Train the decision tree
        tree.fit(X_tr, y_tr)

        #4. Evaluate on the training data
        training_predictions = tree.predict(X_tr)
        train_accuracy = accuracy(y_tr, training_predictions)

        #5. Evaluate on the validation data
        val_predictions = tree.predict(X_val)
        val_accuracy = accuracy(y_val, val_predictions)

        #6. Display both results
        print(f"\nResults for {task}:")
        print(f"Training accuracy:   {train_accuracy}")
        print(f"Validation accuracy: {val_accuracy}")
