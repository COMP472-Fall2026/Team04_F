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

The shared functions in `common.py` load these files, create a stratified
80/20 train/validation split with seed 42, calculate metrics, log results,
and build Kaggle submissions. Fit preprocessing on the training split when
comparing models. The tree and forest tuning scripts also support cross-validation;
record the evaluation protocol alongside each score.

## Recorded validation results

| Method | Image accuracy | Sentiment accuracy | Evaluation |
| --- | ---: | ---: | --- |
| k-NN | 60.225% | 67.458% | Shared stratified 80/20 split, seed 42 |
| Neural network | 36.650% | 68.042% | Shared stratified 80/20 split, seed 42 |
| Decision tree | 12.93% | 48.26% | Five-fold means across five tree seeds |
| Random forest | Summary missing | Summary missing | Staged tuning; final CSV summaries are not in this checkout |

The KNN accuracies are calculated from the saved validation predictions using
the shared split. NN measurements are in
[the selected-model summary](results/neural_net/final/summary.csv). The decision-tree
means are reported in the completed [image log](trees/results/image/tree_grid_image_5fold_nonredundant_output.txt)
and [sentiment log](trees/results/sentiment/tree_grid_sentiment_5fold_nonredundant_output.txt).
The tree logs report standard deviations of 0.38 and 1.22 percentage points,
respectively. Use the same validation protocol when comparing methods for the
final model choice. These scores were used during tuning and selection.

## k-NN (Ethan Elkaim)

`knn.py` implements k-NN and its preprocessing in NumPy. It stores the
preprocessed training vectors and labels; there are no epochs or learning rate.

Run validation on both tasks, then fit on all labeled rows and predict the test sets:

```bash
python knn.py
```

The selected preprocessing and settings are:

| Task | Preprocessing | k | Voting power |
| --- | --- | ---: | ---: |
| Image | Square root of each feature, then row L2 normalization; append 49 LDA dimensions with weight 0.5 and shrinkage 1000 | 20 | 64 |
| Sentiment | Replace the input with 2 LDA dimensions, using shrinkage 0.1 | 200 | 0 |

LDA is fitted on the training split for validation, and on all labeled rows
for the final test predictions. The recorded validation accuracies are 0.6022
for image and 0.6746 for sentiment.

Set `RUN_SWEEPS = True` in `knn.py` to repeat the slower experiments over k,
voting power, standardization, cosine distance, feature powers, and LDA.
This is off by default. Sweep results are written to `results/knn-results.csv`.
The shared `results/results.csv` retains the earlier KNN sweeps alongside the NN runs.

Validation and test predictions are saved under `results/` as
`val_preds_knn_<task>.csv` and `test_preds_knn_<task>.csv`.
The combined Kaggle file is `submissions/sub_knn_v4.csv`.

## Neural network

Run the baseline on both datasets:

```bash
python neural_net.py
```

[NEURAL_NETWORK.md](NEURAL_NETWORK.md) describes the model, training options,
and experiment commands. [NEURAL_NETWORK_RESULTS.md](NEURAL_NETWORK_RESULTS.md)
contains the results and error analysis. Exact commands for the selected models
and their combined prediction file are in
[NEURAL_NETWORK.md](NEURAL_NETWORK.md#reproduce-the-selected-models).

```bash
python -m unittest discover -s tests -p 'test_neural_net.py'
```

## Decision tree and random forest

The NumPy implementations are in `trees/decision_tree.py` and
`forests/random_forest.py`. Run their prediction entry points from the project root:

```bash
python -m trees.decision_tree
python -m forests.random_forest
```

Both entry points currently select `task = "image"`. These commands fit all
labeled rows and write per-task prediction files under `submissions/`;
combine both tasks in template order before uploading to Kaggle.

The completed decision-tree tuning logs select the following configurations:

| Task | Config ID | max_depth | min_samples_split | min_samples_leaf | criterion | max_features |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Image | C475 | 60 | 80 | 10 | gini | None (all features) |
| Sentiment | C095 | 10 | 80 | 40 | gini | None (all features) |

Set `task` and the `DecisionTree(...)` arguments in the prediction entry point
to the appropriate row. Its current arguments match the image configuration.
Changing only `task` does not load the sentiment configuration.

The random-forest prediction entry point currently uses 25 trees, depth 40,
minimum split size 20, minimum leaf size 5, Gini impurity, `max_features="sqrt"`,
bootstrap sampling, and seed 42. These are its default settings; the selected
forest configuration needs to be documented from the missing experiment summaries.

For decision-tree validation experiments:

```bash
python -m trees.decision_tree_vtest_with_logging
```

Configure `TASK`, the parameter grid, and `CV_FOLDS` in that script before
running. It defaults to sentiment and five folds; `CV_FOLDS = 1` uses the
shared 80/20 split. It saves run logs, grid summaries, and finalist summaries
under `results/` and supports resuming matching experiments.

For the staged forest study:

```bash
python -m forests.tune_forest --task image --through 2
```

Use `--task sentiment` for the other task and `--help` for the stage options.
Stages 1-4 use the first fold for screening; `--through 5` evaluates selected
configurations on all five folds. This is tuning cross-validation because its
first fold also participates in selection. Results and reusable tree votes are
saved under `forests/results/<run-name>/<task>/`. These searches can take much
longer than the individual prediction runs. Compare cross-validation means
separately from scores measured on the fixed 80/20 split.

## Clustering

`clustering.py` implements k-means in NumPy and compares several cluster counts
on each dataset:

```bash
python clustering.py
```

It uses three initializations per cluster count, reports inertia, purity,
adjusted Rand index, and class-pair analysis, and writes
`results/clustering_results.csv`. Labels are used to evaluate the clusters.

The saved results use the supplied features without additional scaling. At the
true class counts, image clustering with k=50 has purity 0.0982 and ARI 0.0237;
sentiment with k=3 has purity 0.3977 and ARI 0.0166. The low ARI values indicate
weak agreement between these k-means clusters and the class labels.

Image purity rises to 0.1295 at k=100. Sentiment purity reaches 0.4909 at k=50,
while its highest recorded ARI is 0.0234 at k=10. More clusters can increase
purity by dividing the data into smaller groups, so purity alone does not show
better recovery of the classes. These results describe this k-means setup;
they do not establish a limit on supervised classification accuracy. Full
measurements are in [results/clustering_results.csv](results/clustering_results.csv).

## Files to submit

Keep source code, `requirements.txt`, component documentation, experiment logs,
and selected supporting results under version control. The shared log is
`results/results.csv`; KNN, tree, forest, and clustering studies also have their
own logs. The selected neural-network evidence kept in Git is under
`results/neural_net/final/`. Datasets, virtual environments, NN model weights,
submission files, and raw neural-network experiment folders are ignored.

Kaggle receives a separate CSV with columns `id,label` and 10,000 prediction
rows. The provided template determines the IDs and their order. The selected
individual neural networks' prepared file is
`submissions/neural_net_optimization_best.csv`; it can also be regenerated using
the selected-model commands. The team chooses its final predictors after
comparing all required methods on the shared validation split.

The joint report and presentation must cover all algorithms, clustering, error
analysis, the final model choice, and division of work. Neural-network references
are recorded in `NEURAL_NETWORK.md`.

Before the team submission, add the missing tree grid/finalist CSV summaries
and forest stage summaries, then complete the forest row above. The ignore
rules allow these selected summaries under `trees/results/<task>/` and
`forests/results/<run-name>/<task>/`; tree-vote caches and per-example forest
predictions remain ignored. Document the final image and sentiment predictors,
their combined submission command and score, and the team members' contributions
in the joint report. The final model choice remains open.
