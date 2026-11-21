"""Check score voting, checkpoint handling, and task routing."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import common
import ensemble
import knn
import neural_net


class EnsembleTests(unittest.TestCase):
    def test_knn_scores_preserve_original_weighting_and_ties(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
        y = np.array([2, 1, 0, 2, 1])
        queries = np.array([[0, 0], [0.5, 0.5], [1.5, 1], [0, 0.8]])
        for k in (1, 2, 4):
            for power in (0, 4, 64):
                with self.subTest(k=k, power=power):
                    model = knn.KNN(k=k, power=power)
                    model.fit(X, y)
                    scores, predictions = ensemble.knn_class_scores(model, queries, 4, chunk_size=2)
                    np.testing.assert_array_equal(predictions, model.predict(queries, chunk_size=2))
                    np.testing.assert_allclose(scores.sum(axis=1), 1)
                    np.testing.assert_array_equal(scores[:, 3], 0)

    def test_blend_weights_endpoints_ties_and_invalid_inputs(self):
        result = ensemble.blend_predictions(
            np.array([[0.6, 0.4]]), np.array([[0.2, 0.8]]), np.array([0]), 0.3)
        np.testing.assert_array_equal(result, [1])
        tied = np.array([[0.5, 0.5]])
        # KNN can select class 1 on an equal vote because its neighbour is closer.
        for weight in (0, 0.3, 0.5):
            np.testing.assert_array_equal(
                ensemble.blend_predictions(tied, tied, np.array([1]), weight), [1])
        np.testing.assert_array_equal(
            ensemble.blend_predictions(tied, tied, np.array([1]), 1), [0])
        for weight in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                ensemble.blend_predictions(tied, tied, np.array([0]), weight)
        with self.assertRaises(ValueError):
            ensemble.blend_predictions(tied, np.ones((1, 3)), np.array([0]), 0.3)
        with self.assertRaises(ValueError):
            ensemble.blend_predictions(tied, np.array([[np.nan, 0]]), np.array([0]), 0.3)

    def test_nn_scores_restore_scaling_and_reject_wrong_training_scope(self):
        torch.set_num_threads(1)
        X = np.array([[1, 5], [2, 7], [3, 9]], dtype=np.float32)
        (scaled,), scaler = neural_net.prepare_features(X)
        config = neural_net.Config(hidden=(4,), dropout=0.5)
        model = neural_net.NeuralNetwork(2, 50, config).eval()
        with torch.inference_mode():
            expected = model(torch.from_numpy(scaled)).softmax(dim=1).numpy()
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'image.pt'
            neural_net.save_checkpoint(
                checkpoint, model, 'image', config, scaler, 13, 42, 'all_labeled_data')
            actual = ensemble.image_nn_scores(checkpoint, X, 'all_labeled_data')
            np.testing.assert_allclose(actual, expected)
            with self.assertRaisesRegex(ValueError, 'train_split'):
                ensemble.image_nn_scores(checkpoint, X, 'train_split')
            with self.assertRaisesRegex(ValueError, 'matching input features'):
                ensemble.image_nn_scores(checkpoint, np.zeros((3, 3)), 'all_labeled_data')

    def test_submission_routes_image_to_blend_and_sentiment_to_knn(self):
        X = np.array([[0, 0], [10, 10]], dtype=np.float32)
        image_ids = np.array(['img_a', 'img_b'])
        sentiment_ids = np.array(['text_b', 'text_a'])
        tasks = {
            'image': (X, np.array([2, 4]), X, image_ids),
            'sentiment': (X, np.array([1, 2]), X, sentiment_ids),
        }
        votes = np.zeros((2, 50))
        votes[[0, 1], [2, 4]] = 1
        nn_scores = np.zeros((2, 50))
        nn_scores[[0, 1], [1, 3]] = 1
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'image.pt'
            checkpoint.touch()
            with (patch.object(common, 'load_task', side_effect=tasks.__getitem__) as load,
                  patch.object(ensemble, 'prepare_knn', side_effect=lambda task, fit, y, other: (fit, other)),
                  patch.object(ensemble, 'knn_class_scores', return_value=(votes, np.array([2, 4]))),
                  patch.object(ensemble, 'image_nn_scores', return_value=nn_scores) as nn,
                  patch.object(common, 'build_submission', return_value='submission.csv') as build,
                  redirect_stdout(io.StringIO())):
                ensemble.main(['--image-checkpoint', str(checkpoint)])
            self.assertEqual([call.args[0] for call in load.call_args_list], ['image', 'sentiment'])
            nn.assert_called_once()
            self.assertEqual(nn.call_args.args[2], 'all_labeled_data')
            args = build.call_args.args
            np.testing.assert_array_equal(args[0], image_ids)
            np.testing.assert_array_equal(args[1], [2, 4])
            np.testing.assert_array_equal(args[2], sentiment_ids)
            np.testing.assert_array_equal(args[3], [1, 2])

    def test_validation_fits_preprocessing_on_training_rows_and_saves_sweep(self):
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = np.repeat([0, 1], 5)
        train, _, validation, _ = common.train_val_split(np.arange(len(y)), y)
        scores = np.zeros((len(validation), 50))
        scores[:, 0] = 1
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'image.pt'
            checkpoint.touch()
            results = Path(folder) / 'weights.csv'
            with (patch.object(common, 'load_task', return_value=(X, y, X, np.arange(len(X)))),
                  patch.object(ensemble, 'prepare_knn', side_effect=lambda task, fit, labels, other: (fit, other)) as prepare,
                  patch.object(ensemble, 'knn_class_scores', return_value=(scores, np.zeros(len(validation), dtype=int))),
                  patch.object(ensemble, 'image_nn_scores', return_value=scores) as nn,
                  patch.object(common, 'build_submission') as build,
                  redirect_stdout(io.StringIO())):
                ensemble.main(['--image-checkpoint', str(checkpoint), '--validate', '--sweep',
                               '--results-file', str(results)])
            for call in prepare.call_args_list:
                np.testing.assert_array_equal(call.args[1], X[train])
                np.testing.assert_array_equal(call.args[2], y[train])
                np.testing.assert_array_equal(call.args[3], X[validation])
            self.assertEqual(nn.call_args.args[2], 'train_split')
            build.assert_not_called()
            table = pd.read_csv(results)
            self.assertEqual(len(table), 12)
            self.assertTrue((table.validation_images == len(validation)).all())


if __name__ == '__main__':
    unittest.main()
