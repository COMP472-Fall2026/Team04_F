import unittest

import numpy as np

from forests.random_forest import RandomForest


class TestRandomForestTreeGroups(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [2, 0, 0],
            [2, 1, 1],
            [3, 0, 1],
            [3, 1, 0],
            [4, 0, 1],
            [4, 1, 0],
            [5, 0, 0],
            [5, 1, 1]
        ], dtype=float)
        self.y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0])

    def _forest(self, tree_groups=None):
        return RandomForest(
            n_estimators=5,
            max_depth=4,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion="gini",
            max_features="sqrt",
            bootstrap=True,
            random_state=123,
            tree_groups=tree_groups
        )

    def test_homogeneous_forest_matches_equivalent_single_group(self):
        homogeneous = self._forest().fit(self.X, self.y)
        single_group = self._forest([{"count": 5}]).fit(self.X, self.y)

        self.assertEqual(len(homogeneous.trees), 5)
        np.testing.assert_array_equal(homogeneous.predict(self.X), single_group.predict(self.X))
        self.assertEqual(
            [tree.random_state for tree in homogeneous.trees],
            [tree.random_state for tree in single_group.trees]
        )
        for homogeneous_tree, grouped_tree in zip(homogeneous.trees, single_group.trees):
            np.testing.assert_array_equal(
                homogeneous_tree.predict(self.X),
                grouped_tree.predict(self.X)
            )

    def test_mixed_forest_applies_group_parameters_in_order(self):
        tree_groups = [
            {"count": 2, "max_depth": 1, "min_samples_split": 3},
            {"count": 1, "min_samples_leaf": 2, "criterion": "entropy", "max_features": None},
            {"count": 2, "max_depth": 3, "max_features": 1}
        ]
        forest = RandomForest(
            n_estimators=5,
            max_depth=4,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion="gini",
            max_features="sqrt",
            bootstrap=False,
            random_state=7,
            tree_groups=tree_groups
        ).fit(self.X, self.y)

        self.assertIs(forest.tree_groups, tree_groups)
        self.assertEqual(len(forest.trees), 5)
        actual_parameters = [
            (
                tree.max_depth,
                tree.min_samples_split,
                tree.min_samples_leaf,
                tree.criterion,
                tree.max_features
            )
            for tree in forest.trees
        ]
        self.assertEqual(actual_parameters, [
            (1, 3, 1, "gini", "sqrt"),
            (1, 3, 1, "gini", "sqrt"),
            (4, 2, 2, "entropy", None),
            (3, 2, 1, "gini", 1),
            (3, 2, 1, "gini", 1)
        ])

    def test_mixed_forest_prediction_uses_tree_majority_vote(self):
        forest = self._forest([
            {"count": 2, "max_depth": 1},
            {"count": 3, "max_depth": 4, "criterion": "entropy"}
        ]).fit(self.X, self.y)

        tree_predictions = np.array([tree.predict(self.X) for tree in forest.trees])
        expected = []
        for sample_predictions in tree_predictions.T:
            labels, counts = np.unique(sample_predictions, return_counts=True)
            expected.append(labels[np.argmax(counts)])

        np.testing.assert_array_equal(forest.predict(self.X), np.asarray(expected))

    def test_identical_mixed_forests_are_reproducible(self):
        groups_a = [
            {"count": 3, "min_samples_split": 2},
            {"count": 2, "min_samples_split": 5, "max_features": None}
        ]
        groups_b = [dict(group) for group in groups_a]

        forest_a = self._forest(groups_a).fit(self.X, self.y)
        forest_b = self._forest(groups_b).fit(self.X, self.y)

        np.testing.assert_array_equal(forest_a.predict(self.X), forest_b.predict(self.X))
        self.assertEqual(
            [tree.random_state for tree in forest_a.trees],
            [tree.random_state for tree in forest_b.trees]
        )
        for tree_a, tree_b in zip(forest_a.trees, forest_b.trees):
            np.testing.assert_array_equal(tree_a.predict(self.X), tree_b.predict(self.X))

    def test_invalid_tree_groups_raise_value_error(self):
        invalid_configurations = [
            ({"count": 3}, "list or tuple"),
            ([None], r"tree_groups\[0\].*dictionary"),
            ([{"max_depth": 2}], r"tree_groups\[0\].*count"),
            ([{"count": 0}], "positive integer"),
            ([{"count": -1}], "positive integer"),
            ([{"count": 1.5}], "positive integer"),
            ([{"count": True}], "positive integer"),
            ([{"count": np.bool_(True)}], "positive integer"),
            ([{"count": 2}], "sum to n_estimators"),
            ([{"count": 10**12}], "sum to n_estimators"),
            ([{"count": 3, "bootstrap": False}], "unknown hyperparameter.*bootstrap"),
            ([], "sum to n_estimators")
        ]

        for tree_groups, message_pattern in invalid_configurations:
            with self.subTest(tree_groups=tree_groups):
                with self.assertRaisesRegex(ValueError, message_pattern):
                    RandomForest(n_estimators=3, tree_groups=tree_groups).fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
