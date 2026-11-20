# COMP472 Group 04

Classification of the supplied image features and sentiment embeddings.
The datasets contain numerical vectors: 500 features and 50 classes for images,
and 768 features and 3 classes for sentiment.

## Setup

Python 3.11+ is required by the neural-network script. The experiments were run
with Python 3.11.7; library versions are pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the competition files from the Kaggle Data tab and extract them into:

```text
data/
  image_train.csv
  image_test.csv
  sentiment_train.csv
  sentiment_test.csv
  sample_submission.csv
```

The shared functions in `common.py` load these files and create a stratified
80/20 train/validation split with seed 42. Fit preprocessing on the training
split when comparing models.

## Neural network

Run the baseline on both datasets:

```bash
python neural_net.py
```

[NEURAL_NETWORK.md](NEURAL_NETWORK.md) describes the model, training options,
and experiment commands. [NEURAL_NETWORK_RESULTS.md](NEURAL_NETWORK_RESULTS.md)
contains the results and error analysis. Commands for reproducing the selected
models are in [the selected-results folder](results/neural_net/final/README.md).

```bash
python -m unittest discover -s tests -p 'test_neural_net.py'
```

The other classifiers and the clustering experiment are integrated from the
other team members' branches. Their code and results must be included in the
team submission.

## Files to submit

Keep source code, `requirements.txt`, component documentation, the aggregate
`results/results.csv`, and selected supporting results under version control.
The neural-network results kept in Git are under `results/neural_net/final/`.
Datasets, virtual environments, caches, model weights, predictions, and raw
experiment folders are ignored.

Kaggle receives a separate CSV with columns `id,label` and 10,000 prediction
rows. The provided template determines the IDs and their order. The selected
individual neural networks' prepared file is
`submissions/neural_net_optimization_best.csv`; it can also be regenerated using
the selected-model commands. The team chooses its final predictors after
comparing all required methods on the shared validation split.

The joint report and presentation must cover all algorithms, clustering, error
analysis, the final model choice, and division of work. Neural-network references
are recorded in `NEURAL_NETWORK.md`.
