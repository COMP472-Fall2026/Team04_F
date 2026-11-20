import unittest

import numpy as np

from trees.decision_tree import DecisionTree


class TestDecisionTreeCriterion(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1],
            [2],
            [8],
            [9]
        ])
        self.y = np.array([0, 0, 1, 1])

    def test_entropy_tree_and_default_behavior(self):
        tree = DecisionTree(max_depth=3, criterion="entropy")
        tree.fit(self.X, self.y)

        self.assertEqual(tree.criterion, "entropy")
        self.assertTrue(np.array_equal(tree.predict(self.X), self.y))
        self.assertAlmostEqual(tree._entropy(self.y), 1.0)
        self.assertAlmostEqual(DecisionTree()._entropy(np.array([0, 0, 0, 1])), 0.8112781244591328)

    def test_gini_tree(self):
        tree = DecisionTree(max_depth=3, criterion="gini")
        tree.fit(self.X, self.y)

        self.assertTrue(np.array_equal(tree.predict(self.X), self.y))
        self.assertAlmostEqual(tree._gini(self.y), 0.5)

    def test_information_gain_uses_selected_criterion(self):
        left_counts = np.array([2, 0])
        right_counts = np.array([0, 2])

        entropy_tree = DecisionTree(criterion="entropy")
        gini_tree = DecisionTree(criterion="gini")

        self.assertAlmostEqual(entropy_tree._information_gain(1.0, left_counts, right_counts), 1.0)
        self.assertAlmostEqual(gini_tree._information_gain(0.5, left_counts, right_counts), 0.5)

    def test_invalid_criterion_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "gini.*entropy"):
            DecisionTree(criterion="invalid")

    def test_optimized_split_search_handles_repeated_values(self):
        X = np.array([[1], [1], [2], [2]])
        y = np.array(["left", "left", "right", "right"])

        for criterion in ("entropy", "gini"):
            tree = DecisionTree(max_depth=1, criterion=criterion)
            feature_index, threshold = tree._best_split(X, y)

            self.assertEqual(feature_index, 0)
            self.assertEqual(threshold, 1.5)
            self.assertTrue(np.array_equal(tree.fit(X, y).predict(X), y))


if __name__ == "__main__":
    unittest.main()
