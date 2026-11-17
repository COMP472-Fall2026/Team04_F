# Initial neural network results

The curated evidence for Git is in
[results/neural_net/final/](results/neural_net/final/README.md), and all trial
aggregates are retained in `results/results.csv`. Historical run folders named
below remain local and can be regenerated using the documented commands.

Run command:

```bash
.venv/bin/python neural_net.py --sweep --finalize --output-dir results/neural_net/initial_experiments
```

These results use only the competition's supplied features and labels. They are
validation measurements, not Kaggle scores. Each task uses the team's shared,
stratified split with seed 42: 16,000/4,000 training/validation image rows and
9,600/2,400 sentiment rows. Every image class contributes 80 validation examples;
every sentiment class contributes 800.

Environment: Python 3.11.7, NumPy 2.4.6, Pandas 3.1.1, PyTorch 2.15.0,
Matplotlib 3.11.2, CPU, two PyTorch threads. Model and minibatch seed: 42.
Full settings and input hashes are in the run's `environment.json`.

## Controlled experiments

Accuracy is measured after restoring each experiment's best validation epoch.
The baseline has hidden layers `(256, 64)`, ReLU, Adam at learning rate 0.001,
batch size 128, weight decay 0.0001, dropout 0.2, and standardized features.
All runs allow up to 40 epochs with early-stopping patience 8.

| Experiment | Image train accuracy | Image validation accuracy | Image best epoch | Sentiment train accuracy | Sentiment validation accuracy | Sentiment best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 92.83% | 32.80% | 16 | 81.45% | 67.17% | 4 |
| One hidden layer of 128 | 82.93% | 32.63% | 14 | 74.26% | 66.96% | 2 |
| Learning rate 0.0001 | 60.34% | 30.93% | 39 | 74.22% | 67.50% | 7 |
| Tanh activation | 71.24% | 29.53% | 11 | 76.59% | 67.46% | 4 |
| No dropout | 83.89% | **33.28%** | 6 | 76.47% | 67.17% | 2 |
| No standardization | 66.09% | 30.15% | 34 | 69.93% | **67.58%** | 3 |

The selected image model removes dropout; the selected sentiment model uses the
original unstandardized features. The equally weighted mean validation accuracy
is **50.43%**. These are the best configurations within this small sweep. Small
differences on one split and model seed do not establish a general ranking.

## Observations for analysis

- **Images overfit strongly.** Even the selected checkpoint has an 83.89%
  training accuracy versus 33.28% validation accuracy, a gap of 50.61 percentage
  points. The baseline's best checkpoint has a larger gap of 60.03 points.
  More epochs often raise training accuracy while validation accuracy stagnates
  and validation loss rises. The saved learning curves show this directly.
- **A smaller architecture is competitive.** One 128-neuron hidden layer obtains
  32.63% image validation accuracy, close to the two-layer baseline, with lower
  training accuracy. This run provides little evidence that adding the second
  layer improves generalization.
- **Learning-rate effects depend on the task.** Reducing the learning rate to
  0.0001 improves sentiment slightly, but the image model is still learning near
  the 40-epoch cap. Its lower score cannot establish that the smaller learning
  rate would remain worse with a larger training budget.
- **Scaling has different effects across representations.** Standardization is
  helpful for the image models in this sweep. On sentiment, the unstandardized
  model has the best validation accuracy and a smaller training/validation gap
  of 2.34 percentage points. This is an empirical observation; the experiment
  does not establish a causal explanation about individual embedding dimensions.
- **Removing dropout is not uniformly better.** It produces a small image
  validation improvement but no sentiment improvement over baseline. It also
  fits training examples rapidly; early stopping remains important.

## Error analysis of the selected validation models

Image macro F1: **0.3287**. There are **2,669 errors out of 4,000** examples.
Some of the most frequent directed confusions are:

| True image class | Predicted image class | Count |
| ---: | ---: | ---: |
| 39 | 37 | 14 |
| 4 | 2 | 13 |
| 44 | 32 | 12 |
| 35 | 36 | 10 |

Class 35 has the lowest recall at 10%. Image class names and original images are
not supplied here, so these are numerical class observations.

Sentiment macro F1: **0.6727**. There are **778 errors out of 2,400** examples.

| Sentiment class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.6832 | 0.8088 | 0.7407 |
| Neutral | 0.6168 | 0.5613 | 0.5877 |
| Positive | 0.7255 | 0.6575 | 0.6898 |

Neutral is the hardest class for this model: 194 neutral examples are predicted
negative and 157 are predicted positive. The numerical embeddings and labels
establish these confusions, but the original sentences are unavailable for
interpreting individual cases. All erroneous IDs, labels, and confidence values
are saved in each task's `errors.csv`.

## Final training and verification

After validation selected configurations and epoch counts, the image winner was
trained again on all 20,000 labeled image rows for 6 epochs. The sentiment winner
was trained again on all 12,000 labeled sentiment rows for 3 epochs. Standardizer
statistics were refit on all labeled image rows; sentiment uses no scaler.

The combined file is `submissions/neural_net_initial_experiments.csv`. It contains
predictions for 5,000 images and 5,000 sentiment examples. It has been checked
against the sample template for exact ID order, unique coverage, no missing
values, integer labels, and valid class ranges. No Kaggle upload was performed.

Nine neural network tests pass. Separately, both saved validation checkpoints
reproduce their reported validation accuracies, and both final checkpoints
reproduce every saved test prediction when reloaded from disk.

Validation scores use the same split used to select configurations; they should
be treated as development estimates. Further work could study stronger
regularization, additional random seeds, and a longer low-learning-rate image
run. The required comparisons with teammates' methods belong in the joint
report; this document covers neural networks only.

The implementation, tests, and documentation were developed with assistance
from OpenAI Codex/ChatGPT, to be acknowledged in the team's report.

## Follow-up after the first Kaggle submission

The initial submission received a **public Kaggle score of 0.499**, as reported
by the team. The reported mid and high baselines are **0.599** and **0.650**.
The score is 10.0 percentage points below the mid baseline and 15.1 points below
the high baseline. It is close to the initial mean validation accuracy of
0.5043, so the observed test result is consistent with the development estimate.
It does not establish a private leaderboard score or a course grade.

### Capacity and regularization study

To investigate image overfitting, ten configurations were evaluated on each task
using the same shared validation split and seed 42. These runs allowed 80 epochs
and early-stopping patience 12. Configurations were selected using validation
accuracy; the public leaderboard was not used to rank configurations.

```bash
.venv/bin/python neural_net.py --regularization-sweep --epochs 80 --patience 12 \
    --finalize --output-dir results/neural_net/regularization_experiments
```

| Configuration | Image validation accuracy | Sentiment validation accuracy |
| --- | ---: | ---: |
| Reference: initial baseline, longer training budget | 32.80% | 67.17% |
| One hidden layer of 64 | 29.85% | 67.42% |
| One hidden layer of 32 | 25.90% | 67.50% |
| Weight decay 0.01 | 31.58% | 67.00% |
| Weight decay 0.1 | 2.00% | 66.21% |
| Dropout 0.5 | 30.33% | 67.21% |
| Hidden 64, weight decay 0.01, dropout 0.3 | 32.48% | 67.58% |
| Hidden 32, weight decay 0.1 | 7.88% | 67.29% |
| Hidden 64, unstandardized features | 29.90% | **68.04%** |
| Hidden 64, weight decay 0.01, unstandardized features | 2.00% | 67.13% |

Smaller networks did not improve image accuracy. The very strong weight decay
caused severe underfitting on images. Neither additional dropout nor the larger
training budget beat the original image winner. The sentiment winner improved
from 67.58% to 68.04%, using one hidden layer of 64, unstandardized features,
dropout 0.2, and its best epoch 9. This difference is small on a single validation
split and does not establish a statistically reliable improvement.

Detailed outputs are in `results/neural_net/regularization_experiments/`.
The combined file produced by that run uses the winners within that study; it
does not automatically include the stronger image model from the initial study.

### Image feature-scale checks

The supplied image vectors have unit L2 length. Three further image-only runs
checked whether larger learning rates would improve training on the original
unstandardized vectors. They used Adam, dropout 0.2, weight decay 0.0001, up to
80 epochs, and patience 12:

| Hidden layers | Learning rate | Validation accuracy | Output directory |
| --- | ---: | ---: | --- |
| 256, 64 | 0.003 | 29.73% | `results/neural_net/raw_lr003/` |
| 256, 64 | 0.01 | 28.78% | `results/neural_net/raw_lr001/` |
| 512, 256 | 0.01 | 30.15% | `results/neural_net/raw_wide_lr001/` |

For example, the first run can be reproduced with:

```bash
.venv/bin/python neural_net.py --task image --no-standardize --learning-rate 0.003 \
    --epochs 80 --patience 12 --output-dir results/neural_net/raw_lr003
```

The other two commands use `--learning-rate 0.01`; the wider model also uses
`--hidden 512 256`. Choose fresh output directory names when rerunning experiments.
These three checks also failed to improve on the original image winner.

### Best models after the regularization and feature-scale studies

Comparing the initial and follow-up studies still selects the original image
model, at **33.28%** validation accuracy, and the new 64-neuron sentiment model,
at **68.04%**. Their mean validation accuracy is **50.66%**, only 0.23 percentage
points above the initial mean. The new experiments therefore do **not** close
the gap to the reported mid baseline.

Both selected models were refitted on all their respective labeled training
examples. Their combined predictions are saved locally in
`submissions/neural_net_best_validation.csv`, with the selection recorded in
`results/neural_net/best_validation/summary.json`. Saved validation scores and
all 10,000 final predictions have been checked by reloading the selected models.
No new file has been uploaded to Kaggle by this code.

These findings support comparing the team's other image classifiers on the
shared validation split before choosing the final competition models. All
required methods still need to be reported, even when a different method is
selected for a task's final predictions.

## Optimization and hidden-layer normalization study

The next image-only study examined settings not covered by the earlier runs:
intermediate learning rate, moderate weight decay, light dropout, SGD with
momentum, and batch normalization. It used the same split and seed 42, with up
to 80 epochs and patience 12. The reference exactly reproduces the initial
winning image configuration and its validation score.

```bash
.venv/bin/python neural_net.py --task image --optimization-sweep --dropout 0 \
    --epochs 80 --patience 12 --finalize \
    --output-dir results/neural_net/optimization_experiments
```

| Configuration | Training accuracy | Validation accuracy | Best epoch |
| --- | ---: | ---: | ---: |
| Reference: hidden 256/64, Adam 0.001, no dropout | 83.89% | 33.28% | 6 |
| Weight decay 0.001 | 99.98% | 33.78% | 16 |
| Learning rate 0.0003 | 98.86% | 33.40% | 26 |
| Dropout 0.1 | 96.41% | 32.80% | 13 |
| SGD, learning rate 0.05, momentum 0.9 | 88.17% | 28.80% | 9 |
| Batch normalization, standardized inputs | 91.95% | 30.60% | 6 |
| Batch normalization, original inputs | 98.49% | 30.90% | 8 |
| Hidden 512/256, batch normalization, dropout 0.2 | **99.89%** | **36.65%** | **13** |

The wider batch-normalized model improves image validation accuracy by **3.38
percentage points**, but its training/validation gap remains large. Because
this configuration jointly changes width, normalization, and dropout relative
to the reference, the gain cannot be attributed to batch normalization alone.
The narrower batch-normalized variants do not improve validation accuracy.

These runs refine the overfitting diagnosis: capacity alone is not sufficient
to explain the results. Smaller networks and stronger regularization can shrink
the gap by lowering training accuracy without raising validation accuracy. A
wider network can instead improve validation accuracy while still fitting the
training examples almost perfectly. The objective is better validation
performance, rather than a small gap obtained through underfitting.

The intermediate learning rate slows fitting but produces only a small gain.
Moderate weight decay produces a small gain, while light dropout and the
particular SGD setting tested here do not improve the score. These findings
describe the tested settings, not a general ranking of optimizers or a proof
that other configurations cannot help.

Batch normalization remains part of a feed-forward architecture. Its running
statistics are learned on training minibatches, frozen during validation, and
saved with the model. Tests cover evaluation behavior, checkpoint reloads,
and a singleton final minibatch without discarding the training example.
There are now **ten passing neural-network tests**.

The selected image network was retrained on all 20,000 labeled rows for 13
epochs. Reloading its validation checkpoint reproduces 36.65% accuracy, and
reloading the final checkpoint reproduces every saved image test prediction.
The existing 64-neuron sentiment model still supplies the sentiment predictions.

The current best neural-network combination is:

| Task | Training accuracy | Validation accuracy |
| --- | ---: | ---: |
| Images: hidden 512/256 with batch normalization and dropout 0.2 | 99.89% | 36.65% |
| Sentiment: hidden 64, original features, dropout 0.2 | 71.51% | 68.04% |

The mean validation accuracy is **52.35%**. Its 10,000 predictions are in
`submissions/neural_net_optimization_best.csv`, with selection details in
`results/neural_net/optimization_selection/summary.json`. Earlier submission
files are preserved. No new Kaggle submission was made by this code.

The team's reported image KNN validation accuracy of 52.48% remains higher than
this neural network's 36.65%. The neural network improvement therefore does not
change the current validation-based recommendation to use KNN for image
predictions and the neural network for sentiment, assuming the shared split was
used for both. Repeated use of the validation split for tuning can make its best
score optimistic; further testing on independent data would be needed to establish
the generality of these findings.

## Moderate regularization and early stopping

A controlled image-only study kept hidden layers 512/256, batch normalization,
ReLU, Adam at learning rate 0.001, minibatches of 128, standardized inputs, and
seed 42 fixed. The shared split remains 16,000 training and 4,000 validation
examples. Ten runs compared the reference with dropout 0.2/0.3/0.4 crossed with
weight decay 0.0003/0.001/0.003. These runs allowed 80 epochs and used
accuracy-based checkpoint selection with patience 8.

```bash
.venv/bin/python neural_net.py --task image --hidden 512 256 --batch-norm \
    --dropout 0.2 --weight-decay 0.0001 --moderate-sweep --epochs 80 --patience 8 \
    --output-dir results/neural_net/moderate_regularization
```

Five follow-up runs compared loss-based stopping with patience 3 and 8, shorter
accuracy-based patience, and a stronger regularized setting. These runs use
`--early-stopping-metric loss` or `accuracy`; this controls both which weights
are restored and which improvements reset patience. Configurations are compared
using their restored checkpoints, rather than the final epoch before stopping.

Selected results are below. Gaps are training minus validation accuracy in
percentage points. The full 15-run comparison is saved in
`results/neural_net/moderate_regularization/experiments.csv`.

| Dropout | Weight decay | Checkpoint metric | Patience | Best epoch | Train accuracy | Validation accuracy | Gap | Validation loss |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.0001 | Accuracy: reference | 8 | 13 | 99.89% | **36.65%** | 63.24 | 3.0959 |
| 0.2 | 0.001 | Accuracy | 8 | 18 | 99.96% | 36.60% | 63.36 | 2.9536 |
| 0.4 | 0.001 | Accuracy | 8 | 11 | 93.61% | 36.33% | 57.28 | 2.4964 |
| 0.4 | 0.003 | Accuracy | 8 | 11 | 92.09% | 36.03% | 56.07 | 2.4924 |
| 0.2 | 0.0001 | Loss | 3 | 4 | 82.72% | 33.58% | 49.14 | 2.5656 |
| 0.2 | 0.001 | Loss | 3 or 8 | 4 | 81.65% | 33.70% | 47.95 | 2.5528 |
| 0.2 | 0.001 | Accuracy | 3 | 8 | 98.41% | 35.80% | 62.61 | 2.6898 |
| 0.4 | 0.003 | Loss | 3 | 11 | 92.09% | 36.03% | 56.07 | 2.4924 |

No tested setting beats the reference's **36.65%** validation accuracy. However,
dropout 0.4 with weight decay 0.001 lowers the gap from 63.24 to 57.28 points
and validation cross-entropy from 3.0959 to 2.4964, while sacrificing only
0.325 percentage points of accuracy (13 validation examples). This is a useful
tradeoff to report, not evidence of a proven improvement on hidden test data.

Loss-based stopping does act earlier: with dropout 0.2 and weight decay 0.001,
it restores epoch 4 instead of accuracy selection's epoch 18. The gap drops to
47.95 points, but validation accuracy drops to 33.70%. Patience 3 stops at epoch
7; patience 8 stops at epoch 12. Both restore the same epoch-4 weights and have
identical measured scores. Shorter patience therefore saves computation in
this comparison without changing its selected model. Shorter accuracy-based
patience instead stops before a later accuracy improvement, lowering the score.

With dropout 0.4 and weight decay 0.003, loss and accuracy select the same epoch
11; the shorter loss-based patience stops training earlier without changing
the selected weights. These examples distinguish stopping the training loop
from selecting which checkpoint to use.

The study reduces some signs of overfitting but does not resolve the image
network's generalization problem or establish a higher accuracy. A lower loss
can accompany lower accuracy because loss also depends on confidence in wrong
and correct predictions. The single reused split and seed limit conclusions
about differences between nearby configurations.

The previous image winner and its existing predictions remain selected by
validation accuracy. Sentiment was not retrained, and no new submission file was
generated. All 15 new checkpoints were reloaded and checked against their
reported accuracy, loss, selected epoch, and checkpoint-selection criterion.
The neural-network suite now has **11 passing tests**, including loss-based
stopping when accuracy and loss favor different epochs. Experiment settings,
curves, and a summary are in `results/neural_net/moderate_regularization/` and
`results/neural_net/early_stopping_comparison/`.
