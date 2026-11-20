# Neural network

`neural_net.py` trains separate feed-forward classifiers for the two datasets.
Each model is initialized from random weights and trained on the supplied
labeled features. The image network has 500 inputs and 50 output logits; the
sentiment network has 768 inputs and 3 output logits.

## Training

`common.py` provides the data loader, stratified split, standardizer, metrics,
result logger, and submission builder. The split uses seed 42 and reserves 20%
of each class for validation. Standardization is optional and is fitted on the
training portion before being applied to validation and test features.

The network consists of linear hidden layers with ReLU or Tanh activations,
optional batch normalization, and optional dropout. The output layer returns
logits for `CrossEntropyLoss`. Each minibatch performs a forward pass, loss
calculation, backpropagation, and an Adam or SGD update. SGD uses momentum 0.9.

After each epoch, training and validation metrics are measured in evaluation
mode. Dropout is disabled and batch normalization uses its running statistics.
A singleton final training batch also uses running batch-normalization
statistics, while its parameters still receive gradients.

By default, checkpoint selection maximizes validation accuracy and breaks ties
using lower validation loss. `--early-stopping-metric loss` instead minimizes
validation loss and breaks ties using higher accuracy. Training stops after
`--patience` epochs without improvement, and the selected weights are restored.
Experiments are compared by validation accuracy, then loss.

`--finalize` initializes the selected configuration again and trains on all
labeled rows for the selected number of epochs. Its scaler is refitted on all
labeled rows. Those final models produce test predictions.

## Run commands

Use the environment and data layout in [README.md](README.md). The dependencies
are NumPy, Pandas, Matplotlib, and PyTorch at the versions in `requirements.txt`.
The neural-network script does not import scikit-learn classifiers.

```bash
python neural_net.py
python neural_net.py --task sentiment --hidden 64 --no-standardize --finalize
python neural_net.py --help
```

The baseline uses hidden layers 256/64, ReLU, Adam at 0.001, batch size 128,
weight decay 0.0001, dropout 0.2, up to 40 epochs, and patience 8. Input
standardization is enabled by default. CPU with two threads and model seed 42
are the defaults. `--device auto` selects an
available accelerator. `--seed` changes initialization and minibatch order;
the shared data split remains fixed.

Important options:

| Option | Purpose |
| --- | --- |
| `--task image`, `sentiment`, or `both` | Select the dataset(s). |
| `--hidden 512 256` | Set hidden-layer widths. |
| `--activation relu` or `tanh` | Set the activation function. |
| `--optimizer adam` or `sgd` | Set the optimizer. |
| `--learning-rate`, `--batch-size`, `--epochs` | Set training parameters. |
| `--dropout`, `--weight-decay`, `--batch-norm` | Set regularization and hidden-layer normalization. |
| `--early-stopping-metric`, `--patience` | Set checkpoint selection and stopping. |
| `--output-dir` | Choose a new directory for the run. |
| `--finalize` | Refit and save test predictions. |

The exact commands for the selected image and sentiment models, including
combining their predictions, are in
[results/neural_net/final/README.md](results/neural_net/final/README.md).

## Reproduce the experiments

Run commands from the repository root. Output directory names must be new;
use another name when repeating a local run. All recorded runs used CPU,
two threads, model seed 42, and the shared split seed 42.

```bash
# Initial comparisons on both datasets: 12 runs.
python neural_net.py --sweep --finalize --output-dir results/neural_net/initial_experiments

# Network size, stronger regularization, and scaling: 20 runs.
python neural_net.py --regularization-sweep --epochs 80 --patience 12 \
    --finalize --output-dir results/neural_net/regularization_experiments

# Image optimization and batch normalization: 8 runs.
python neural_net.py --task image --optimization-sweep --dropout 0 \
    --epochs 80 --patience 12 --finalize \
    --output-dir results/neural_net/optimization_experiments

# Image dropout/weight-decay grid, including a reference: 10 runs.
python neural_net.py --task image --hidden 512 256 --batch-norm \
    --dropout 0.2 --weight-decay 0.0001 --moderate-sweep --epochs 80 --patience 8 \
    --output-dir results/neural_net/moderate_regularization
```

The sweep definitions are in `experiment_configs`, `regularization_configs`,
`optimization_configs`, and `moderate_regularization_configs`. Only one sweep
flag can be used per command.

Three image runs also tested raw features at higher learning rates. Use
`--task image --no-standardize --epochs 80 --patience 12` with these settings:

| Directory name | Hidden layers | Learning rate |
| --- | --- | ---: |
| `raw_lr003` | 256/64 | 0.003 |
| `raw_lr001` | 256/64 | 0.01 |
| `raw_wide_lr001` | 512/256 | 0.01 |

For each run, add `--hidden`, `--learning-rate`, and
`--output-dir results/neural_net/<directory name>` to that command.

Five image runs compared stopping rules. Use
`--task image --hidden 512 256 --batch-norm --epochs 80` with the following
options and a separate output directory under
`results/neural_net/early_stopping_comparison/`:

| Run | Dropout | Weight decay | Stopping metric | Patience |
| --- | ---: | ---: | --- | ---: |
| `reference_loss_p3` | 0.2 | 0.0001 | loss | 3 |
| `regularized_loss_p3` | 0.2 | 0.001 | loss | 3 |
| `regularized_loss_p8` | 0.2 | 0.001 | loss | 8 |
| `regularized_accuracy_p3` | 0.2 | 0.001 | accuracy | 3 |
| `stronger_loss_p3` | 0.4 | 0.003 | loss | 3 |

These commands reproduce the 42 image and 16 sentiment training runs in the
aggregate log. Some runs repeat reference configurations with different stopping
limits. Several combined configurations change more than one parameter.

## Results and files

[NEURAL_NETWORK_RESULTS.md](NEURAL_NETWORK_RESULTS.md) explains the findings.
`results/results.csv` stores all trial configurations and summary measurements.
Selected-model evidence is kept in
[results/neural_net/final/](results/neural_net/final/README.md): metrics, histories,
curves, class reports, confusion matrices, and environment records.

Raw run folders remain local. Each run saves its settings and data hashes in
`environment.json`, epoch metrics in `history.csv`, and selected weights with
architecture and preprocessing statistics in `validation_model.pt`. Refitting
also saves `final_model.pt`, `final_training.json`, and `test_predictions.csv`.
Per-example validation predictions, split IDs, and errors are saved locally.

A run that finalizes both tasks uses `common.build_submission()` to write a
combined file in `submissions/`, matching predictions to template IDs. A run
with one task saves that task's predictions. Model weights, datasets, prediction
files, and raw runs are ignored by Git.

Saved weights can be used on raw feature arrays:

```python
from neural_net import predict_checkpoint
predictions = predict_checkpoint('path/to/final_model.pt', X_raw)
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_neural_net.py'
```

The 11 tests cover output dimensions, learning on synthetic test data,
reproducibility, training-only scaling, minibatch loss weighting, both stopping
criteria, batch-normalization behavior, checkpoint reloads, class metrics, and
the training/refit workflow. Synthetic data is used only in tests.

## References

- [PyTorch cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch batch normalization](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
