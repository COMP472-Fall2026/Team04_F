"""Neural-network tests only; run with unittest (no extra test dependency)."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch

import common
import neural_net as net


class NeuralNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_task_output_dimensions_and_logits(self):
        for task in common.TASKS:
            model = net.NeuralNetwork(net.FEATURE_DIMS[task], common.NUM_CLASSES[task], net.Config())
            self.assertEqual(tuple(model(torch.zeros(4, net.FEATURE_DIMS[task])).shape),
                             (4, common.NUM_CLASSES[task]))
            self.assertFalse(any(isinstance(layer, torch.nn.Softmax) for layer in model.modules()))

    def test_standardization_uses_training_rows_only(self):
        train = np.array([[-1, 4], [1, 4]], dtype=np.float32)
        validation = np.array([[100, 8]], dtype=np.float32)
        test = np.array([[-200, 0]], dtype=np.float32)
        (scaled_train, scaled_val, scaled_test), (mean, std) = net.prepare_features(train, validation, test)
        np.testing.assert_array_equal(mean, [0, 4])
        np.testing.assert_array_equal(std, [1, 1])
        np.testing.assert_array_equal(scaled_train.mean(axis=0), [0, 0])
        np.testing.assert_array_equal(scaled_val, [[100, 4]])
        np.testing.assert_array_equal(scaled_test, [[-200, -4]])
        (unchanged,), scaler = net.prepare_features(train, enabled=False)
        np.testing.assert_array_equal(unchanged, train)
        self.assertEqual(scaler, (None, None))

    def test_learning_and_seed_reproducibility(self):
        rng = np.random.default_rng(10)
        centers = np.array([[-3, -3], [0, 3], [3, -3]], dtype=np.float32)
        y = np.repeat(np.arange(3), 50)
        X = (centers[y] + rng.normal(0, 0.25, (150, 2))).astype(np.float32)
        X_train, y_train, X_val, y_val = common.train_val_split(X, y)
        config = net.Config(hidden=(16,), learning_rate=0.02, dropout=0, epochs=20, batch_size=32)
        model, history, best_epoch, _ = net.train_model(
            X_train, y_train, X_val, y_val, 3, config, verbose=False)
        self.assertGreater(net.evaluate(model, X_val, y_val)["accuracy"], 0.95)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertGreater(best_epoch, 0)
        again, repeated, _, _ = net.train_model(X_train, y_train, X_val, y_val, 3, config, verbose=False)
        self.assertEqual(history, repeated)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, again.state_dict()[key], rtol=0, atol=0)

    def test_early_stopping_restores_best_epoch(self):
        epoch = 0

        def train_epoch(model, *args):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(epoch)

        values = [(0.8, 0.5), (0.9, 0.4), (0.7, 0.6), (0.6, 0.7)]

        def evaluate(*args, **kwargs):
            acc, loss = values[epoch - 1]
            return {"accuracy": acc, "loss": loss}

        X, y = np.ones((6, 2), np.float32), np.zeros(6, np.int64)
        with patch.object(net, "train_epoch", side_effect=train_epoch), patch.object(net, "evaluate", side_effect=evaluate):
            model, history, best_epoch, _ = net.train_model(
                X, y, X, y, 3, net.Config(hidden=(4,), epochs=10, patience=2), verbose=False)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(len(history), 4)
        for parameter in model.parameters():
            self.assertTrue(torch.all(parameter == 2))

    def test_loss_is_weighted_correctly_for_partial_batches(self):
        logits = np.array([[5, 0], [4, 0], [3, 0], [0, 4], [0, 9]], np.float32)
        labels = np.array([0, 0, 0, 1, 0])
        actual = net.evaluate(torch.nn.Identity(), logits, labels, batch_size=2)
        expected = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
        self.assertAlmostEqual(actual["loss"], expected, places=6)
        self.assertEqual(actual["accuracy"], 0.8)

    def test_loss_stopping_restores_earlier_weights_despite_rising_accuracy(self):
        epoch = 0

        def train_epoch(model, *args):
            nonlocal epoch
            epoch += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(epoch)

        losses = [0.5, 0.4, 0.6, 0.7]

        def evaluate(*args, **kwargs):
            return {"accuracy": 0.5 + epoch / 10, "loss": losses[epoch - 1]}

        X, y = np.ones((6, 2), np.float32), np.zeros(6, np.int64)
        config = net.Config(hidden=(4,), epochs=10, patience=2, early_stopping_metric="loss")
        with patch.object(net, "train_epoch", side_effect=train_epoch), patch.object(net, "evaluate", side_effect=evaluate):
            model, history, best_epoch, _ = net.train_model(X, y, X, y, 3, config, verbose=False)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(len(history), 4)
        self.assertGreater(history[-1]["val_acc"], history[best_epoch - 1]["val_acc"])
        for parameter in model.parameters():
            self.assertTrue(torch.all(parameter == 2))

    def test_checkpoint_restores_preprocessing_and_eval_mode(self):
        raw = np.array([[1, 5], [2, 7], [3, 9]], np.float32)
        (scaled,), scaler = net.prepare_features(raw)
        config = net.Config(hidden=(8,), dropout=0.5)
        model = net.NeuralNetwork(2, 3, config)
        expected = net.evaluate(model, scaled)["predictions"]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            net.save_checkpoint(path, model, "sentiment", config, scaler, 3, 42, "train_split")
            restored, metadata = net.load_checkpoint(path)
            self.assertFalse(restored.training)
            self.assertEqual(metadata["training_scope"], "train_split")
            np.testing.assert_array_equal(net.predict_checkpoint(path, raw), expected)

    def test_class_metrics_handle_never_predicted_classes(self):
        cm, report = net.class_report(np.array([0, 1, 2]), np.array([0, 0, 0]), "sentiment")
        self.assertEqual(cm.sum(), 3)
        self.assertEqual(report.name.tolist(), ["Negative", "Neutral", "Positive"])
        self.assertEqual(report.loc[1, "precision"], 0)
        self.assertTrue(np.isfinite(report[["precision", "recall", "f1"]]).all().all())

    def test_batch_norm_training_singleton_batch_and_saved_inference(self):
        common.set_seed(42)
        raw = np.random.default_rng(8).normal(size=(5, 3)).astype(np.float32)
        (X,), scaler = net.prepare_features(raw)
        y = np.array([0, 1, 2, 0, 1])
        config = net.Config(hidden=(8,), batch_norm=True, batch_size=2, dropout=0)
        model = net.NeuralNetwork(3, 3, config)
        original_weights = model.layers[0].weight.detach().clone()
        net.train_epoch(model, net.make_loader(X, y, config, 42), net.make_optimizer(model, config), "cpu")
        self.assertFalse(torch.equal(original_weights, model.layers[0].weight))
        layer = next(item for item in model.modules() if isinstance(item, torch.nn.BatchNorm1d))
        self.assertEqual(layer.num_batches_tracked.item(), 2)
        self.assertTrue(layer.training)
        running_mean = layer.running_mean.clone()
        expected = net.evaluate(model, X)["predictions"]
        torch.testing.assert_close(layer.running_mean, running_mean)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "normalized_model.pt"
            net.save_checkpoint(path, model, "sentiment", config, scaler, 1, 42, "train_split")
            np.testing.assert_array_equal(net.predict_checkpoint(path, raw), expected)
            restored, checkpoint = net.load_checkpoint(path)
            self.assertTrue(checkpoint["config"]["batch_norm"])
            self.assertFalse(restored.training)

    def test_pipeline_preserves_ids_logs_results_and_refits_all_rows(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(250, 500)).astype(np.float32)
        y = np.repeat(np.arange(50), 5)
        y.setflags(write=False)  # Pandas 3 can return read-only label arrays.
        test = rng.normal(size=(7, 500)).astype(np.float32)
        test_ids = np.array([f"img_{i:06}" for i in (5, 2, 6, 0, 1, 4, 3)])
        train_ids = np.array([f"img_train_{i:06}" for i in range(len(y))])
        config = net.Config(hidden=(8,), epochs=2, batch_size=64)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"id": train_ids}).to_csv(root / "image_train.csv", index=False)
            with patch.object(common, "load_task", return_value=(X, y, test, test_ids)), \
                 patch.object(common, "DATA_DIR", str(root)), \
                 patch.object(common, "RESULTS_DIR", str(root)), \
                 patch.object(common, "RESULTS_FILE", str(root / "results.csv")), \
                 redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                row, (ids, predictions) = net.run_task(
                    "image", {"baseline": config}, root / "image", 42, "cpu", finalize=True, plots=False)
            self.assertEqual(row["train_samples"], 200)
            self.assertEqual(row["val_samples"], 50)
            np.testing.assert_array_equal(ids, test_ids)
            self.assertEqual(predictions.dtype.kind, "i")
            self.assertTrue(((predictions >= 0) & (predictions < 50)).all())
            final_model = root / "image" / "final_model.pt"
            np.testing.assert_array_equal(net.predict_checkpoint(final_model, test), predictions)
            _, checkpoint = net.load_checkpoint(final_model)
            np.testing.assert_allclose(checkpoint["mean"].numpy(), X.mean(axis=0))
            self.assertEqual(checkpoint["training_scope"], "all_labeled_data")
            self.assertEqual(json.loads((root / "image" / "final_training.json").read_text())["training_samples"], 250)
            splits = pd.read_csv(root / "image" / "split.csv")
            expected_train, _, expected_val, _ = common.train_val_split(np.arange(len(y)), y)
            self.assertEqual(set(splits.loc[splits.split == "validation", "id"]), set(train_ids[expected_val]))
            self.assertEqual(set(splits.loc[splits.split == "train", "id"]), set(train_ids[expected_train]))
            self.assertEqual(pd.read_csv(root / "results.csv").method.tolist(), ["neural_net"])
            validation = pd.read_csv(root / "image" / "validation_predictions.csv")
            self.assertEqual(len(validation), 50)
            self.assertTrue((root / "image" / "errors.csv").exists())

    def test_bad_hyperparameters_are_rejected(self):
        for kwargs in ({"hidden": ()}, {"epochs": 0}, {"dropout": 1}, {"learning_rate": -1},
                       {"weight_decay": -1}, {"activation": "unknown"}, {"optimizer": "unknown"},
                       {"batch_norm": True, "batch_size": 1}, {"early_stopping_metric": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                net.Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
