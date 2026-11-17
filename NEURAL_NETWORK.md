# Neural network component

This document covers `neural_net.py` only. The implementation uses the supplied
feature vectors, a feed-forward PyTorch network trained from random weights, and
the shared functions in `common.py`. It does not train an encoder or use external
training data. The other classification and clustering components are separate.

## Requirements covered

| Project requirement | Neural network implementation |
| --- | --- |
| Evaluate both tasks | Independent networks with 500 inputs / 50 outputs for images and 768 inputs / 3 outputs for sentiment. |
| Train the network yourselves | Explicit minibatch forward pass, cross-entropy loss, backpropagation, and optimizer updates using permitted `torch.nn` and `torch.optim` tools. |
| Use local validation | The same stratified 80/20 split as `common.get_task_data()`, with split seed 42. |
| Prevent preprocessing leakage | Fit the optional standardizer on training rows only; save its mean and standard deviation with the model. |
| Investigate hyperparameters | Six controlled experiments with `--sweep`, plus CLI options for further experiments. |
| Explain overfitting and errors | Training curves, train/validation gap, confusion matrices, per-class precision/recall/F1, and misclassified sample IDs. |
| Reproduce results | Saved configuration, model seed, split membership, Python/library versions, CPU thread count, device, data SHA-256 hashes, and weights. |
| Predict every test sample | `--finalize` refits each winner on all labeled rows; when running both tasks, the shared submission builder matches all 10,000 predictions to template IDs. |

## Environment and commands

Use Python 3.11 or newer. The existing project requirements pin the dependencies.
The neural network needs NumPy, Pandas, Matplotlib, and PyTorch; it does not import
scikit-learn. In the current project `.venv`, those four packages are installed at
the versions in `requirements.txt`.

To install just the neural network dependencies in another environment:

```bash
python -m pip install numpy==2.4.6 pandas==3.1.1 matplotlib==3.11.2 torch==2.15.0
```

Run one baseline for both datasets:

```bash
.venv/bin/python neural_net.py
```

Run the six-experiment sweep for both datasets, select each winner using validation,
and refit to produce a Kaggle-format CSV locally:

```bash
.venv/bin/python neural_net.py --sweep --finalize
```

Run a separate follow-up study targeting overfitting, with a larger epoch budget:

```bash
.venv/bin/python neural_net.py --regularization-sweep --epochs 80 --patience 12 \
    --finalize --output-dir results/neural_net/regularization_experiments
```

The ten configurations compare the reference network with one-layer networks of
64 and 32 neurons, weight decay of 0.01 and 0.1, dropout of 0.5, and combinations
of smaller networks, regularization, and unstandardized inputs. Some combined
configurations change several factors; use the individual-factor comparisons
when explaining their effects. `--sweep` and `--regularization-sweep` are mutually
exclusive. Compare any winner with the initial study before choosing a submission;
a new experiment does not automatically replace an earlier, better model.

An additional image-focused study compares moderate parameter changes, SGD with
momentum, and optional hidden-layer batch normalization:

```bash
.venv/bin/python neural_net.py --task image --optimization-sweep --dropout 0 \
    --epochs 80 --patience 12 --finalize \
    --output-dir results/neural_net/optimization_experiments
```

Its reference is the initial winning image configuration: hidden layers 256 and
64, ReLU, Adam at 0.001, weight decay 0.0001, standardized inputs, and no dropout.
The eight candidates are the reference; weight decay 0.001; learning rate 0.0003;
dropout 0.1; SGD at learning rate 0.05 with momentum 0.9; batch normalization;
batch normalization on raw features; and batch normalization with hidden layers
512 and 256 and dropout 0.2. Some candidates change several settings, so their
results measure a configuration rather than isolate one setting's effect.
The sweep flags are mutually exclusive.

`--batch-norm` enables `Linear -> BatchNorm1d -> activation -> optional dropout`
in each hidden layer. This remains a feed-forward network. Input standardization
uses training-set statistics once; batch normalization instead tracks hidden
activation statistics during training and uses saved running statistics for
validation and prediction. If the last training minibatch contains one example,
that minibatch uses running statistics for batch normalization rather than
discarding the example. The other network parameters still receive gradients.
Batch normalization is an optional experiment, not a guaranteed overfitting fix.
Existing checkpoints without this setting load with batch normalization disabled.

To tune regularization on the wider image architecture while keeping its other
settings fixed:

```bash
.venv/bin/python neural_net.py --task image --hidden 512 256 --batch-norm \
    --dropout 0.2 --weight-decay 0.0001 --moderate-sweep --epochs 80 --patience 8 \
    --output-dir results/neural_net/moderate_regularization
```

This compares the reference with a grid of dropout 0.2/0.3/0.4 and weight decay
0.0003/0.001/0.003. Learning rate, architecture, batch size, and preprocessing
are fixed. Use a new output directory name for each repeated run.

Early stopping can monitor either validation accuracy (the default) or validation
cross-entropy loss. For example, add `--early-stopping-metric loss --patience 3`
to restore the checkpoint with the lowest validation loss and stop after three
successive epochs without improvement. A loss-selected checkpoint can occur
earlier and have a smaller train/validation gap while obtaining lower accuracy.
Compare both accuracy and loss rather than treating a smaller gap as sufficient.

Run a particular configuration on sentiment only:

```bash
.venv/bin/python neural_net.py --task sentiment --hidden 128 --activation tanh \
    --learning-rate 0.001 --batch-size 128 --epochs 40 --dropout 0.2
```

`--task image` selects the image task. `--optimizer sgd` uses SGD with momentum
0.9, while the default is Adam. `--weight-decay`, `--patience`,
`--no-standardize`, and `--seed` control additional experiments. See `--help`.
Changing `--seed` changes model initialization and minibatch order; it deliberately
keeps the team's shared train/validation split fixed.

The default device is CPU with two threads. No GPU is required. `--device auto`
can select CUDA or Apple MPS when available. Reasonably reproducible results are
expected on the same environment; scores can differ across devices and versions.

## Training and model selection

The baseline is:

```text
Input (500 or 768)
  -> Linear(256) -> ReLU -> Dropout(0.2)
  -> Linear(64)  -> ReLU -> Dropout(0.2)
  -> Linear(50 or 3)
```

The final layer returns raw logits. `CrossEntropyLoss` applies the required
log-softmax internally, so the model has no final softmax layer during training.
Softmax is used only to report prediction confidence during evaluation.
Labels are integer class indices. The optimizer is Adam with learning rate
0.001 and weight decay 0.0001, using minibatches of 128 and up to 40 epochs.

Validation is performed after each epoch with dropout disabled. By default, the
best epoch has the highest validation accuracy, with lower validation
cross-entropy used to break ties. With `--early-stopping-metric loss`, the best
epoch instead has the lowest validation loss, with higher accuracy breaking ties.
Training stops after eight successive epochs without improvement by default;
`--patience` changes this limit.
The best weights are restored before reporting accuracy or saving a checkpoint.
Training and validation loss curves are both measured in evaluation mode.
Across experiments, configurations are still ranked by validation accuracy,
then validation loss, regardless of the checkpoint criterion used within a run.

The default sweep changes one factor relative to this baseline:

| Experiment | Change |
| --- | --- |
| `baseline` | Baseline configuration. |
| `one_hidden_layer` | Use one hidden layer of 128 neurons. |
| `lower_learning_rate` | Divide the baseline learning rate by 10. |
| `alternate_activation` | Use tanh rather than ReLU. |
| `dropout_comparison` | Remove dropout. |
| `scaling_comparison` | Use the supplied features without standardization. |

If CLI options change the baseline, variants are derived from that baseline.
For a tanh baseline the alternate activation is ReLU; a zero-dropout baseline
is compared with dropout 0.2. Epoch count and patience are equal across the sweep.
This is a small controlled study, not an exhaustive search. In particular, a
smaller learning rate may need more epochs than the common training budget.

The two tasks select configurations independently. With `--finalize`, each
selected architecture is initialized again, the scaler is fit on **all labeled
training rows**, and training runs for its validation-selected number of epochs.
Final refitting uses no validation or hidden-test labels. Validation scores in
the summary describe the earlier models trained on the 80% training split;
they are not measured scores for the final refitted models.

## Saved outputs

The compact Git submission package is in
[results/neural_net/final/](results/neural_net/final/README.md). It contains the
selected models' supporting evidence and exact reproduction commands. The
aggregate `results/results.csv` retains all trial results. Other generated
neural-network run folders remain local and are ignored by Git; rerunning
experiments does not automatically replace the curated package.

Each invocation creates a new directory under `results/neural_net/`. An explicit
`--output-dir` must name a new directory, so earlier runs are not overwritten.

- `environment.json`: runtime versions, experiment settings, and dataset hashes.
- `summary.csv` / `summary.json`: winning validation configuration per task and,
  when both tasks are run, their equally weighted mean validation accuracy.
- `<task>/experiments.csv`: all configurations, training/validation accuracy,
  generalization gap, best epoch, and approximate timings.
- `<task>/<experiment>/history.csv`, `learning_curves.png`, and `metrics.json`:
  epoch-level evidence for each experiment.
- `<task>/<experiment>/validation_model.pt`: best validation-selected weights
  together with architecture and training-split scaler.
- `<task>/split.csv`: the shared training/validation sample IDs.
- `<task>/class_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`,
  `confused_classes.csv`, `validation_predictions.csv`, and `errors.csv`:
  error-analysis material for the winning configuration. Rows in a confusion
  matrix are true labels; columns are predicted labels. Errors are sorted by
  descending model confidence. Confidence is not a calibrated probability.
- `<task>/final_model.pt`, `final_training.json`, and `test_predictions.csv`:
  created only with `--finalize`.

Validation experiments also append rows with `method=neural_net` to the shared
`results/results.csv` through the existing `common.log_result()` function.
The code does not modify `common.py` or another teammate's algorithm files.

When **both tasks** are finalized in the same run, the validated combined file
is written to `submissions/neural_net_<run-directory-name>.csv`. A single-task
run writes its task predictions only. Files are created locally; the script
does not upload to Kaggle. These final files and model weights are excluded by
the existing `.gitignore`, so retain or share them separately if needed.

Reload a saved network and make predictions directly from raw feature arrays:

```python
from neural_net import predict_checkpoint

predictions = predict_checkpoint("path/to/final_model.pt", X_raw)
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_neural_net.py' -v
```

The neural network tests check both output dimensions, learning on separable
synthetic data, seed reproducibility, preprocessing leakage, partial minibatch
loss weighting, early stopping and best-weight restoration, checkpoint/scaler
reloads, class metrics, and the experiment/refit pipeline with preserved IDs.
They also check batch-normalization statistics, a singleton final minibatch,
saved-model inference with batch normalization enabled, and loss-based early
stopping when accuracy and loss favor different epochs.
Synthetic samples are used only in tests, never as competition training data.

## Presentation and assistance

Use the saved experiments and curves to explain the effect of each changed
hyperparameter and the training/validation gap. For sentiment, compare the
Negative, Neutral, and Positive rows in the class report. For images, inspect
the largest off-diagonal class pairs. IDs and numerical features support error
analysis; this dataset does not provide the original images or text.

The same validation set is used for tuning and model selection, so these scores
are development estimates rather than an independent final evaluation. Compare
other algorithms using the team's shared split. The private Kaggle labels remain
unavailable, so no hidden-test accuracy is claimed here.

Development assistance: OpenAI Codex/ChatGPT assisted with this neural network
implementation, documentation, and tests. Include that assistance in the team's
report as required by the project overview.

Technical references:

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
