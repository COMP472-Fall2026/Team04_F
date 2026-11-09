# Initial neural network results

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
