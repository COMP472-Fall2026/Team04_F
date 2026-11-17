"""Train feed-forward neural networks on the supplied COMP472 feature vectors.

Examples (run from the project directory):
    .venv/bin/python neural_net.py
    .venv/bin/python neural_net.py --sweep --finalize
    .venv/bin/python neural_net.py --task sentiment --hidden 128 --epochs 30

See NEURAL_NETWORK.md for the experiment procedure and artifact descriptions.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import common


FEATURE_DIMS = {"image": 500, "sentiment": 768}
SENTIMENT_NAMES = ("Negative", "Neutral", "Positive")


@dataclass(frozen=True)
class Config:
    hidden: tuple[int, ...] = (256, 64)
    activation: str = "relu"
    learning_rate: float = 0.001
    optimizer: str = "adam"
    batch_size: int = 128
    epochs: int = 40
    patience: int = 8
    weight_decay: float = 0.0001
    dropout: float = 0.2
    standardize: bool = True
    batch_norm: bool = False
    early_stopping_metric: str = "accuracy"

    def __post_init__(self):
        if not self.hidden or any(width <= 0 for width in self.hidden):
            raise ValueError("Provide at least one positive hidden-layer size.")
        if self.activation not in ("relu", "tanh", "gelu"):
            raise ValueError("activation must be relu, tanh, or gelu")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        if min(self.batch_size, self.epochs, self.patience) <= 0:
            raise ValueError("batch size, epochs, and patience must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_norm and self.batch_size < 2:
            raise ValueError("batch normalization requires a batch size of at least 2")
        if self.early_stopping_metric not in ("accuracy", "loss"):
            raise ValueError("early stopping metric must be accuracy or loss")


class NeuralNetwork(nn.Module):
    """An MLP whose output is raw class logits, as CrossEntropyLoss expects."""

    def __init__(self, input_dim, num_classes, config):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        layers = []
        previous = input_dim
        for width in config.hidden:
            layers.append(nn.Linear(previous, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(activations[config.activation]())
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def prepare_features(X_train, *others, enabled=True):
    """Fit preprocessing on these training rows only; return arrays and scaler."""
    if enabled:
        mean, std = common.fit_standardizer(X_train)
        arrays = [common.apply_standardizer(X, mean, std) for X in (X_train, *others)]
    else:
        mean = std = None
        arrays = (X_train, *others)
    return [np.require(X, dtype=np.float32, requirements=["C", "W"]) for X in arrays], (mean, std)


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is not available; use --device cpu or auto.")
    return torch.device(name)


def make_loader(X, y, config, seed):
    return DataLoader(
        # Pandas may expose labels as read-only arrays; copy before sharing with torch.
        TensorDataset(torch.from_numpy(X), torch.tensor(y, dtype=torch.long)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def make_optimizer(model, config):
    kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)


def train_epoch(model, loader, optimizer, device):
    model.train()  # Re-enable dropout after validation.
    loss_fn = nn.CrossEntropyLoss()
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        # A final minibatch can have one row. Use running statistics for that
        # row so BatchNorm can still train its affine parameters without
        # discarding a training example or estimating variance from one value.
        singleton_layers = [layer for layer in model.modules()
                            if len(features) == 1 and isinstance(layer, nn.BatchNorm1d) and layer.training]
        for layer in singleton_layers:
            layer.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; check features and learning rate.")
        loss.backward()
        optimizer.step()
        for layer in singleton_layers:
            layer.train()


@torch.inference_mode()
def evaluate(model, X, y=None, batch_size=512, device="cpu"):
    """Batched inference; dropout is disabled. Loss is weighted by batch size."""
    if len(X) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    model.eval()
    predictions, confidences = [], []
    total_loss = 0.0
    for start in range(0, len(X), batch_size):
        logits = model(torch.as_tensor(X[start:start + batch_size], device=device))
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        predictions.append(predicted.cpu().numpy())
        confidences.append(confidence.cpu().numpy())
        if y is not None:
            labels = torch.as_tensor(y[start:start + batch_size], dtype=torch.long, device=device)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
    result = {
        "predictions": np.concatenate(predictions),
        "confidence": np.concatenate(confidences),
    }
    if y is not None:
        result.update(loss=total_loss / len(y), accuracy=common.accuracy(y, result["predictions"]))
    return result


def train_model(X_train, y_train, X_val, y_val, num_classes, config, seed=common.SEED,
                device="cpu", verbose=True):
    """Restore the best validation checkpoint using the configured stopping metric."""
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation splits must both be nonempty.")
    common.set_seed(seed)
    model = NeuralNetwork(X_train.shape[1], num_classes, config).to(device)
    optimizer = make_optimizer(model, config)
    loader = make_loader(X_train, y_train, config, seed)
    history, best_state = [], None
    best_score, best_epoch, stale = (-float("inf"), -float("inf")), 0, 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_epoch(model, loader, optimizer, device)
        train_metrics = evaluate(model, X_train, y_train, device=device)
        val_metrics = evaluate(model, X_val, y_val, device=device)
        history.append({
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"], "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
        })
        if config.early_stopping_metric == "loss":
            score = (-val_metrics["loss"], val_metrics["accuracy"])
        else:
            score = (val_metrics["accuracy"], -val_metrics["loss"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            # Clone: state_dict() alone would keep references to changing weights.
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if verbose and (epoch == 1 or epoch % 5 == 0 or stale >= config.patience):
            print(f"  epoch {epoch:2d}: train={train_metrics['accuracy']:.4f}, "
                  f"val={val_metrics['accuracy']:.4f}, val_loss={val_metrics['loss']:.4f}", flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, time.perf_counter() - start


def save_checkpoint(path, model, task, config, scaler, epochs, seed, scope):
    """Store weights, architecture, and fitted preprocessing together."""
    mean, std = scaler
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "task": task, "input_dim": model.layers[0].in_features,
        "num_classes": model.layers[-1].out_features, "config": asdict(config),
        "mean": None if mean is None else torch.from_numpy(mean.copy()),
        "std": None if std is None else torch.from_numpy(std.copy()),
        "epochs": epochs, "seed": seed, "training_scope": scope,
    }, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    model = NeuralNetwork(checkpoint["input_dim"], checkpoint["num_classes"], config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_checkpoint(path, X, device="cpu"):
    """Predict on raw features using the preprocessing saved with the weights."""
    model, checkpoint = load_checkpoint(path, device)
    if checkpoint["mean"] is not None:
        X = common.apply_standardizer(X, checkpoint["mean"].numpy(), checkpoint["std"].numpy())
    return evaluate(model, np.ascontiguousarray(X, dtype=np.float32), device=device)["predictions"]


def experiment_configs(base):
    """Change one factor at a time relative to the user's baseline."""
    return {
        "baseline": base,
        "one_hidden_layer": replace(base, hidden=(128,)),
        "lower_learning_rate": replace(base, learning_rate=base.learning_rate / 10),
        "alternate_activation": replace(base, activation="tanh" if base.activation != "tanh" else "relu"),
        "dropout_comparison": replace(base, dropout=0.0 if base.dropout else 0.2),
        "scaling_comparison": replace(base, standardize=not base.standardize),
    }


def regularization_configs(base):
    """Follow-up study of capacity and regularization after the initial sweep.

    The first variants isolate individual factors; the final variants combine
    smaller hidden layers and stronger weight decay. Selection still uses only
    the shared validation split, never the public Kaggle score.
    """
    return {
        "reference": base,
        "hidden_64": replace(base, hidden=(64,)),
        "hidden_32": replace(base, hidden=(32,)),
        "weight_decay_001": replace(base, weight_decay=0.01),
        "weight_decay_01": replace(base, weight_decay=0.1),
        "dropout_05": replace(base, dropout=0.5),
        "small_regularized": replace(base, hidden=(64,), weight_decay=0.01, dropout=0.3),
        "small_strong_decay": replace(base, hidden=(32,), weight_decay=0.1),
        "unscaled_small": replace(base, hidden=(64,), standardize=False),
        "unscaled_regularized": replace(base, hidden=(64,), weight_decay=0.01, standardize=False),
    }


def optimization_configs(base):
    """Compare moderate settings and alternatives to the initial Adam MLP."""
    return {
        "reference": base,
        "moderate_decay": replace(base, weight_decay=0.001),
        "intermediate_lr": replace(base, learning_rate=0.0003),
        "light_dropout": replace(base, dropout=0.1),
        "sgd_momentum": replace(base, optimizer="sgd", learning_rate=0.05),
        "batch_norm": replace(base, batch_norm=True),
        "batch_norm_raw": replace(base, batch_norm=True, standardize=False),
        "wide_batch_norm": replace(base, batch_norm=True, hidden=(512, 256), dropout=0.2),
    }


def moderate_regularization_configs(base):
    """Hold architecture fixed while comparing a modest dropout/decay grid."""
    configs = {"reference": base}
    for dropout in (0.2, 0.3, 0.4):
        for decay in (0.0003, 0.001, 0.003):
            name = f"dropout_{int(dropout * 100)}_decay_{decay:g}"
            configs[name] = replace(base, dropout=dropout, weight_decay=decay)
    return configs


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def class_report(y_true, predictions, task):
    cm = common.confusion_matrix(y_true, predictions, common.NUM_CLASSES[task])
    tp = np.diag(cm).astype(float)
    support, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp),
                   where=(precision + recall) != 0)
    labels = SENTIMENT_NAMES if task == "sentiment" else [str(i) for i in range(len(cm))]
    report = pd.DataFrame({"class": np.arange(len(cm)), "name": labels, "precision": precision,
                           "recall": recall, "f1": f1, "support": support})
    return cm, report


def make_plots(directory, history, cm=None, task=None):
    # Headless plotting; cache stays inside the project rather than the user's home.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(common.ROOT_DIR) / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(frame.epoch, frame[f"train_{metric}"], label="Training")
        ax.plot(frame.epoch, frame[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Accuracy" if metric == "acc" else "Cross-entropy loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(cm, cmap="Blues")
        ax.set(xlabel="Predicted class", ylabel="True class", title=f"{task}: validation confusion matrix")
        if task == "sentiment":
            ax.set_xticks(range(3), SENTIMENT_NAMES)
            ax.set_yticks(range(3), SENTIMENT_NAMES)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(directory / "confusion_matrix.png", dpi=150)
        plt.close(fig)


def load_data(task):
    X, y, X_test, test_ids = common.load_task(task)
    if X.shape[1] != FEATURE_DIMS[task] or X_test.shape[1] != FEATURE_DIMS[task]:
        raise ValueError(f"{task} requires {FEATURE_DIMS[task]} input features.")
    if not np.isfinite(X).all() or not np.isfinite(X_test).all():
        raise ValueError(f"{task} contains missing or nonfinite features.")
    if not np.isin(y, np.arange(common.NUM_CLASSES[task])).all():
        raise ValueError(f"{task} has out-of-range labels.")
    # Passing indices to the shared splitter gives exactly the same rows as get_task_data.
    train_idx, _, val_idx, _ = common.train_val_split(np.arange(len(y)), y)
    train_ids = pd.read_csv(Path(common.DATA_DIR) / f"{task}_train.csv", usecols=["id"])["id"].to_numpy()
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError(f"{task} contains duplicate sample IDs.")
    return X, y, X_test, test_ids, train_ids, train_idx, val_idx


def finalize_model(task, X, y, X_test, test_ids, config, epochs, seed, device, directory):
    """Refit from scratch on all labeled rows for the validation-selected epoch count."""
    (X_scaled, test_scaled), scaler = prepare_features(X, X_test, enabled=config.standardize)
    common.set_seed(seed)
    model = NeuralNetwork(X.shape[1], common.NUM_CLASSES[task], config).to(device)
    optimizer = make_optimizer(model, config)
    loader = make_loader(X_scaled, y, config, seed)
    print(f"[{task}] refit on all {len(y):,} labeled rows for {epochs} epochs", flush=True)
    start = time.perf_counter()
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    predictions = evaluate(model, test_scaled, device=device)["predictions"]
    predict_time = time.perf_counter() - start
    pd.DataFrame({"id": test_ids, "label": predictions}).to_csv(directory / "test_predictions.csv", index=False)
    save_checkpoint(directory / "final_model.pt", model, task, config, scaler, epochs, seed, "all_labeled_data")
    write_json(directory / "final_training.json", {
        "epochs": epochs, "training_samples": len(y), "test_samples": len(test_ids),
        "train_time_s": train_time, "predict_time_s": predict_time,
        "note": "Validation selected the configuration and epoch count; test labels were never used.",
    })
    return test_ids, predictions


def run_task(task, configs, directory, seed, device, finalize=False, plots=True):
    directory.mkdir()
    X, y, X_test, test_ids, train_ids, train_idx, val_idx = load_data(task)
    pd.DataFrame({"id": train_ids, "split": np.where(np.isin(np.arange(len(y)), val_idx), "validation", "train")
                  }).to_csv(directory / "split.csv", index=False)
    rows, best = [], None
    for name, config in configs.items():
        print(f"[{task}/{name}] train={len(train_idx):,}, validation={len(val_idx):,}", flush=True)
        trial_dir = directory / name
        trial_dir.mkdir()
        (X_train, X_val), scaler = prepare_features(X[train_idx], X[val_idx], enabled=config.standardize)
        model, history, best_epoch, train_time = train_model(
            X_train, y[train_idx], X_val, y[val_idx], common.NUM_CLASSES[task], config, seed, device,
        )
        train_metrics = evaluate(model, X_train, y[train_idx], device=device)
        start = time.perf_counter()
        val_metrics = evaluate(model, X_val, y[val_idx], device=device)
        predict_time = time.perf_counter() - start
        row = {
            "task": task, "experiment": name, "best_epoch": best_epoch,
            "early_stopping_metric": config.early_stopping_metric,
            "epochs_run": len(history), "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"], "val_loss": val_metrics["loss"],
            "generalization_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
            "train_time_s": train_time, "predict_time_s": predict_time,
            "train_samples": len(train_idx), "val_samples": len(val_idx),
            "hyperparameters": json.dumps(asdict(config), sort_keys=True),
        }
        rows.append(row)
        pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
        write_json(trial_dir / "metrics.json", row)
        save_checkpoint(trial_dir / "validation_model.pt", model, task, config, scaler, best_epoch, seed, "train_split")
        if plots:
            make_plots(trial_dir, history)
        common.log_result(task, "neural_net", {**asdict(config), "seed": seed, "best_epoch": best_epoch},
                          train_acc=row["train_acc"], val_acc=row["val_acc"], train_time_s=train_time,
                          predict_time_s=predict_time, notes=f"experiment={name}; artifacts={trial_dir}")
        print(f"[{task}/{name}] best epoch={best_epoch}, validation accuracy={row['val_acc']:.4f}", flush=True)
        if best is None or (row["val_acc"], -row["val_loss"]) > (best["row"]["val_acc"], -best["row"]["val_loss"]):
            best = {"row": row, "config": config, "metrics": val_metrics, "history": history,
                    "checkpoint": trial_dir / "validation_model.pt"}

    pd.DataFrame(rows).to_csv(directory / "experiments.csv", index=False)
    write_json(directory / "best.json", best["row"])
    predictions = best["metrics"]["predictions"]
    cm, report = class_report(y[val_idx], predictions, task)
    report.to_csv(directory / "class_report.csv", index=False)
    pd.DataFrame(cm, index=range(len(cm)), columns=range(len(cm))).to_csv(directory / "confusion_matrix.csv", index_label="true_class")
    validation = pd.DataFrame({"id": train_ids[val_idx], "true_label": y[val_idx], "predicted_label": predictions,
                               "confidence": best["metrics"]["confidence"]})
    validation.to_csv(directory / "validation_predictions.csv", index=False)
    errors = validation[validation.true_label != validation.predicted_label].sort_values("confidence", ascending=False)
    errors.to_csv(directory / "errors.csv", index=False)
    confused = cm.copy()
    np.fill_diagonal(confused, 0)
    pairs = [{"true_class": i, "predicted_class": j, "count": int(confused[i, j])}
             for i, j in zip(*np.nonzero(confused))]
    pd.DataFrame(pairs, columns=["true_class", "predicted_class", "count"]).sort_values(
        "count", ascending=False).to_csv(directory / "confused_classes.csv", index=False)
    if plots:
        make_plots(directory, best["history"], cm, task)
    write_json(directory / "analysis.json", {
        "macro_f1": float(report.f1.mean()), "validation_errors": len(errors),
        "selected_experiment": best["row"]["experiment"],
        "generalization_gap": best["row"]["generalization_gap"],
    })
    final = None
    if finalize:
        final = finalize_model(task, X, y, X_test, test_ids, best["config"], best["row"]["best_epoch"],
                               seed, device, directory)
    return best["row"], final


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["both", *common.TASKS], default="both")
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--sweep", action="store_true", help="Run six controlled hyperparameter experiments per task.")
    study.add_argument("--regularization-sweep", action="store_true",
                       help="Run ten follow-up experiments on network size, weight decay, dropout, and scaling.")
    study.add_argument("--optimization-sweep", action="store_true",
                       help="Compare eight settings including SGD and batch normalization.")
    study.add_argument("--moderate-sweep", action="store_true",
                       help="Compare the reference with nine moderate dropout/weight-decay settings.")
    parser.add_argument("--finalize", action="store_true", help="Refit winners on all labeled data and save test predictions.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--early-stopping-metric", choices=["accuracy", "loss"], default="accuracy",
                        help="Choose and restore checkpoints by validation accuracy or validation loss.")
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-norm", action="store_true", help="Normalize hidden activations before activation functions.")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=common.SEED, help="Model seed; shared data split remains seed 42.")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads; small networks often need only 1-2.")
    parser.add_argument("--output-dir", type=Path, help="New artifact directory; existing directories are never overwritten.")
    parser.add_argument("--no-plots", action="store_true")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        config = Config(hidden=tuple(args.hidden), activation=args.activation,
                        learning_rate=args.learning_rate, optimizer=args.optimizer,
                        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                        weight_decay=args.weight_decay, dropout=args.dropout,
                        standardize=not args.no_standardize, batch_norm=args.batch_norm,
                        early_stopping_metric=args.early_stopping_metric)
        if args.threads <= 0 or args.seed < 0 or args.seed >= 2**32:
            raise ValueError("threads must be positive and seed must be in [0, 2**32).")
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    # CPU defaults keep runs practical and reproducible on ordinary laptops.
    torch.use_deterministic_algorithms(device.type == "cpu")
    if args.moderate_sweep:
        configs = moderate_regularization_configs(config)
    elif args.optimization_sweep:
        configs = optimization_configs(config)
    elif args.regularization_sweep:
        configs = regularization_configs(config)
    else:
        configs = experiment_configs(config) if args.sweep else {"baseline": config}
    tasks = common.TASKS if args.task == "both" else [args.task]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.output_dir or Path(common.RESULTS_DIR) / "neural_net" / stamp
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("numpy", "pandas", "torch", "matplotlib")},
        "platform": platform.platform(), "device": str(device), "threads": args.threads,
        "model_seed": args.seed, "split_seed": common.SEED, "validation_fraction": common.VAL_FRAC,
        "selection": "Configurations are ranked by validation accuracy, then loss; checkpoints use each configuration's early_stopping_metric.",
        "configs": {name: asdict(item) for name, item in configs.items()},
        "data_sha256": {},
    }
    for task in tasks:
        for suffix in ("train", "test"):
            filename = f"{task}_{suffix}.csv"
            with open(Path(common.DATA_DIR) / filename, "rb") as handle:
                metadata["data_sha256"][filename] = hashlib.file_digest(handle, "sha256").hexdigest()
    write_json(directory / "environment.json", metadata)
    print(f"Device: {device}. Artifacts: {directory}", flush=True)
    best_rows, finals = [], {}
    for task in tasks:
        row, final = run_task(task, configs, directory / task, args.seed, device, args.finalize, not args.no_plots)
        best_rows.append(row)
        if final is not None:
            finals[task] = final
    pd.DataFrame(best_rows).to_csv(directory / "summary.csv", index=False)
    summary = {"best": best_rows}
    if len(best_rows) == 2:
        summary["mean_validation_accuracy"] = float(np.mean([row["val_acc"] for row in best_rows]))
        print(f"Mean validation accuracy: {summary['mean_validation_accuracy']:.4f}", flush=True)
    if set(finals) == set(common.TASKS):
        submission = common.build_submission(*finals["image"], *finals["sentiment"],
                                             out_name=f"neural_net_{directory.name}.csv")
        summary["submission"] = submission
    write_json(directory / "summary.json", summary)
    print(f"Finished. Results: {directory / 'summary.csv'}", flush=True)
    return directory


if __name__ == "__main__":
    main()
