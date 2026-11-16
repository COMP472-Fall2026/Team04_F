import numpy as np

from tree_forest_zay import DecisionTree

if __name__ == "__main__":
    X = np.array([
        [1],
        [2],
        [8],
        [9]
    ])

    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(X)

    print("Predictions:", predictions)
    print("Expected:   ", y)

    assert np.array_equal(predictions, y)

    print("Test passed!")