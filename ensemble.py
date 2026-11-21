"""Blend image KNN/NN class scores and use KNN for sentiment.

Example:
    python ensemble.py --image-checkpoint path/to/image/final_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import knn
import neural_net


IMAGE_NN_WEIGHTS = (0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1)


def prepare_knn(task, X_fit, y_fit, X_other):
    """Apply the final KNN preprocessing, fitting projections on X_fit only."""
    if task in knn.FINAL_EXPONENT:
        exponent = knn.FINAL_EXPONENT[task]
        X_fit = knn.power_normalize(X_fit, exponent)
        X_other = knn.power_normalize(X_other, exponent)
    if task in knn.FINAL_APPEND:
        X_fit, X_other = knn.append_lda(
            X_fit, y_fit, X_other, common.NUM_CLASSES[task], *knn.FINAL_APPEND[task])
    if task in knn.FINAL_LDA:
        dimensions, shrinkage = knn.FINAL_LDA[task]
        projection = knn.fit_lda(X_fit, y_fit, common.NUM_CLASSES[task], shrinkage)
        X_fit = knn.apply_lda(X_fit, projection, dimensions)
        X_other = knn.apply_lda(X_other, projection, dimensions)
    return X_fit, X_other


def knn_class_scores(model, X, num_classes, chunk_size=500):
    """Return normalized class votes and preserve KNN's nearest-neighbour tie rule."""
    scores = np.zeros((len(X), num_classes), dtype=np.float64)
    predictions = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), chunk_size):
        distances = model._distances(X[start:start + chunk_size])
        neighbours = np.argsort(distances, axis=1)[:, :model.k]
        labels = model.y_train[neighbours]
        distances = np.take_along_axis(distances, neighbours, axis=1)
        weights = 1.0 / np.maximum(distances.astype(np.float64), 1e-3) ** model.power
        for offset, (row, weight) in enumerate(zip(labels, weights)):
            classes, first_position, inverse = np.unique(
                row, return_index=True, return_inverse=True)
            totals = np.bincount(inverse, weights=weight)
            scores[start + offset, classes] = totals / totals.sum()
            tied = np.flatnonzero(totals == totals.max())
            winner = tied[np.argmin(first_position[tied])]
            predictions[start + offset] = classes[winner]
    return scores, predictions


@torch.inference_mode()
def image_nn_scores(checkpoint_path, X, expected_scope):
    """Load an image checkpoint and apply its saved scaler before inference."""
    model, metadata = neural_net.load_checkpoint(checkpoint_path)
    if (metadata['task'] != 'image' or metadata['input_dim'] != X.shape[1]
            or metadata['num_classes'] != common.NUM_CLASSES['image']):
        raise ValueError('The checkpoint must be an image NN with matching input features.')
    if metadata['training_scope'] != expected_scope:
        raise ValueError(f'Expected a checkpoint trained on {expected_scope}.')
    if metadata['mean'] is not None:
        X = common.apply_standardizer(
            X, metadata['mean'].numpy(), metadata['std'].numpy())
    X = np.ascontiguousarray(X, dtype=np.float32)
    return np.concatenate([
        model(torch.as_tensor(X[start:start + 512])).softmax(dim=1).numpy()
        for start in range(0, len(X), 512)
    ])


def blend_predictions(knn_scores, nn_scores, knn_predictions, nn_weight):
    """Combine corresponding class scores; exact ties favour the KNN prediction."""
    if not np.isfinite(nn_weight) or not 0 <= nn_weight <= 1:
        raise ValueError('NN weight must be between 0 and 1.')
    if knn_scores.ndim != 2 or knn_scores.shape != nn_scores.shape:
        raise ValueError('Models must supply matching rows and class columns.')
    if len(knn_predictions) != len(knn_scores):
        raise ValueError('KNN predictions must match the score rows.')
    if not np.isfinite(knn_scores).all() or not np.isfinite(nn_scores).all():
        raise ValueError('Class scores must be finite.')
    if nn_weight == 0:
        return knn_predictions.copy()
    if nn_weight == 1:
        return nn_scores.argmax(axis=1)
    scores = (1 - nn_weight) * knn_scores + nn_weight * nn_scores
    predictions = scores.argmax(axis=1)
    tied = scores[np.arange(len(scores)), knn_predictions] == scores.max(axis=1)
    predictions[tied] = knn_predictions[tied]
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-checkpoint', type=Path, required=True,
                        help='Full-data image checkpoint for submissions; train-split checkpoint for validation.')
    parser.add_argument('--nn-weight', type=float, default=0.3,
                        help='Image NN score weight; KNN receives the remaining weight (default: 0.3).')
    parser.add_argument('--output-name', default='ensemble_reproduced.csv',
                        help='Output filename under submissions/.')
    parser.add_argument('--validate', action='store_true',
                        help='Evaluate on the shared 80/20 split instead of generating a submission.')
    parser.add_argument('--sweep', action='store_true',
                        help='With --validate, compare the 12 recorded image weight ratios.')
    parser.add_argument('--results-file', type=Path,
                        default=Path(common.RESULTS_DIR) / 'ensemble_validation.csv')
    args = parser.parse_args(argv)
    if not args.image_checkpoint.is_file():
        parser.error('Image checkpoint not found. Train the selected image NN first; see README.md.')
    if not np.isfinite(args.nn_weight) or not 0 <= args.nn_weight <= 1:
        parser.error('--nn-weight must be between 0 and 1.')
    if args.sweep and not args.validate:
        parser.error('--sweep requires --validate.')
    torch.set_num_threads(2)
    finals, comparisons = {}, []
    sentiment_accuracy = None

    # Task routing is explicit: each model receives features from its own CSV.
    for task in ('image', 'sentiment'):
        X, y, X_test, test_ids = common.load_task(task)
        if args.validate:
            train, _, validation, _ = common.train_val_split(np.arange(len(y)), y)
            X_fit, y_fit, X_other = X[train], y[train], X[validation]
            truth = y[validation]
        else:
            X_fit, y_fit, X_other = X, y, X_test
        knn_fit, knn_other = prepare_knn(task, X_fit, y_fit, X_other)
        k, power = knn.FINAL[task]
        model = knn.KNN(k=k, power=power)
        model.fit(knn_fit, y_fit)

        if task == 'image':
            votes, knn_predictions = knn_class_scores(model, knn_other, common.NUM_CLASSES[task])
            scope = 'train_split' if args.validate else 'all_labeled_data'
            probabilities = image_nn_scores(args.image_checkpoint, X_other, scope)
            predictions = blend_predictions(votes, probabilities, knn_predictions, args.nn_weight)
            if args.validate:
                weights = IMAGE_NN_WEIGHTS if args.sweep else (args.nn_weight,)
                for weight in weights:
                    result = blend_predictions(votes, probabilities, knn_predictions, weight)
                    comparisons.append({
                        'knn_weight': round(1 - weight, 6), 'nn_weight': weight,
                        'image_validation_accuracy': common.accuracy(truth, result),
                        'correct_images': int(np.sum(result == truth)),
                        'validation_images': len(truth),
                    })
        else:
            # The selected submission uses sentiment KNN without a sentiment NN.
            predictions = model.predict(knn_other)
        if args.validate:
            score = common.accuracy(truth, predictions)
            print(f'{task}: validation accuracy {score:.6f}', flush=True)
            if task == 'sentiment':
                sentiment_accuracy = score
        else:
            finals[task] = (test_ids, predictions)
            print(f'{task}: generated {len(predictions)} test predictions', flush=True)

    if args.validate:
        table = pd.DataFrame(comparisons)
        table['sentiment_validation_accuracy'] = sentiment_accuracy
        table['mean_validation_accuracy'] = (
            table['image_validation_accuracy'] + sentiment_accuracy) / 2
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.results_file, index=False)
        print(f'Saved validation results to {args.results_file}', flush=True)
        return args.results_file
    return common.build_submission(
        *finals['image'], *finals['sentiment'], out_name=args.output_name)


if __name__ == '__main__':
    main()
