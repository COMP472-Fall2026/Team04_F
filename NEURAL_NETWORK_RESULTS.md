# Neural-network results

## Selected models

| Task | Hidden layers | Input standardization | Batch normalization | Train accuracy | Validation accuracy | Selected epoch |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Images | 512/256 | Yes | Yes | 99.89% | 36.65% | 13 |
| Sentiment | 64 | No | No | 71.51% | 68.04% | 9 |

Both models use ReLU, Adam at learning rate 0.001, dropout 0.2, weight decay
0.0001, and minibatches of 128. The equally weighted mean validation accuracy
is 52.35%. Full metrics and plots are in
[results/neural_net/final/](results/neural_net/final/README.md).

## Validation procedure

The shared stratified 80/20 split uses seed 42. Images have 16,000 training and
4,000 validation rows, including 80 validation examples per class. Sentiment has
9,600 training and 2,400 validation rows, including 800 validation examples per
class. Preprocessing is fitted on training rows. Checkpoint selection uses
validation accuracy, with loss breaking ties, except in the loss-based stopping
experiments described below.

The experiments used Python 3.11.7, NumPy 2.4.6, Pandas 3.1.1, PyTorch 2.15.0,
Matplotlib 3.11.2, CPU, and two PyTorch threads. All model seeds were 42.
Reproduction commands are in [NEURAL_NETWORK.md](NEURAL_NETWORK.md).

## Experiments

The aggregate log [results/results.csv](results/results.csv) contains 58 training
runs: 42 image and 16 sentiment runs, including repeated references. Training
batch size was always 128. The study varied hidden layers, activation, optimizer,
learning rate, scaling, batch normalization, dropout, weight decay, epoch limits,
and stopping rules. It did not compare batch sizes or multiple initialization
seeds.

| Study | Image runs | Sentiment runs | Main comparison |
| --- | ---: | ---: | --- |
| Initial sweep | 6 | 6 | Architecture, learning rate, activation, dropout, scaling |
| Regularization sweep | 10 | 10 | Smaller networks and stronger regularization |
| Raw-feature checks | 3 | 0 | Higher learning rates on original image features |
| Optimization sweep | 8 | 0 | SGD, moderate settings, batch normalization, wider network |
| Moderate regularization and stopping | 15 | 0 | Dropout/decay combinations and checkpoint criteria |

The initial baseline used hidden layers 256/64, ReLU, Adam at 0.001, dropout
0.2, weight decay 0.0001, and standardized inputs. These initial comparisons
used a maximum of 40 epochs and patience 8:

| Change from baseline | Image validation | Sentiment validation |
| --- | ---: | ---: |
| Baseline | 32.80% | 67.17% |
| One hidden layer of 128 | 32.63% | 66.96% |
| Learning rate 0.0001 | 30.93% | 67.50% |
| Tanh activation | 29.53% | 67.46% |
| No dropout | 33.28% | 67.17% |
| No standardization | 30.15% | 67.58% |

Smaller image networks with one layer of 64 or 32 achieved 29.85% and 25.90%.
Strong weight decay of 0.1 caused severe underfitting in the tested image
configurations. Increasing the learning rate on raw image features to 0.003
or 0.01 also failed to improve the result. The particular SGD setting tested,
learning rate 0.05 and momentum 0.9, achieved 28.80%; other SGD rates were not
investigated.

The wider 512/256 network with batch normalization and dropout 0.2 achieved
36.65%, improving on the earlier 33.28% model. Width, normalization, and dropout
changed together, so the gain cannot be attributed to batch normalization alone.
The image model still fitted the training data almost perfectly.

For sentiment, a single 64-neuron layer using the original features achieved
68.04%. Its training accuracy was 71.51%, a gap of 3.47 percentage points.
The initial two-layer standardized model had a larger gap without better
validation accuracy.

## Image regularization and early stopping

The final image study kept the 512/256 architecture, batch normalization,
ReLU, Adam at 0.001, standardized inputs, and minibatches of 128 fixed. It
compared the reference with dropout 0.2/0.3/0.4 and weight decay
0.0003/0.001/0.003. Five further runs compared stopping criteria and patience.
All allowed up to 80 epochs.

| Setting | Checkpoint criterion | Train accuracy | Validation accuracy | Gap, percentage points | Validation loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Reference: dropout 0.2, decay 0.0001 | Accuracy, patience 8 | 99.89% | 36.65% | 63.24 | 3.0959 |
| Dropout 0.4, decay 0.001 | Accuracy, patience 8 | 93.61% | 36.33% | 57.28 | 2.4964 |
| Dropout 0.4, decay 0.003 | Accuracy, patience 8 | 92.09% | 36.03% | 56.07 | 2.4924 |
| Dropout 0.2, decay 0.001 | Loss, patience 3 or 8 | 81.65% | 33.70% | 47.95 | 2.5528 |

[The regularization plot](results/neural_net/final/regularization_tradeoff.png)
shows the tradeoff. Dropout 0.4 with decay 0.001 reduced the gap and validation
loss, with an accuracy decrease of 0.325 percentage points, or 13 validation
examples. None of the new settings exceeded 36.65% accuracy.

Loss-based stopping with dropout 0.2 and decay 0.001 restored epoch 4. Patience
3 stopped training at epoch 7; patience 8 stopped at epoch 12. Both restored
the same weights and produced 33.70% accuracy. Accuracy-based selection with
patience 8 instead restored epoch 18 at 36.60%. Shorter accuracy-based patience
3 restored epoch 8 at 35.80%.

These results show that a smaller train/validation gap can come at an accuracy
cost. Stopping by loss limits fitting earlier, while stopping by accuracy can
retain a later model that classifies more examples correctly but makes more
confident errors.

## Error analysis of the selected models

The image model has 2,534 errors among 4,000 validation examples and macro F1
of 0.3617. Its most frequent directed confusions include:

| True class | Predicted class | Count |
| ---: | ---: | ---: |
| 4 | 2 | 15 |
| 3 | 2 | 12 |
| 28 | 11 | 10 |
| 1 | 2 | 10 |

Classes 6, 9, 29, and 31 have recall of 20%. Class names and original images
are unavailable in these CSVs, so the analysis uses numerical class labels.
The full matrix and per-class measurements are in
[the image results](results/neural_net/final/image/class_report.csv).

The sentiment model has 767 errors among 2,400 validation examples and macro F1
of 0.6763:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.7218 | 76.9% | 0.7446 |
| Neutral | 0.6293 | 52.6% | 0.5732 |
| Positive | 0.6792 | 74.6% | 0.7111 |

Neutral has the lowest recall: 216 neutral examples were predicted positive
and 163 negative. The original sentences are unavailable, so individual errors
can be identified by ID and label but cannot be interpreted from the text.
See [the sentiment results](results/neural_net/final/sentiment/class_report.csv).

## Interpretation and limits

The image model overfits strongly: its training accuracy is 99.89%, while
validation accuracy is 36.65%. Greater capacity may contribute, but the wider
network also improved validation performance. The experiments therefore do not
support parameter count as a complete explanation. The feature relationships
learned by this architecture may transfer poorly. Feature noise, class overlap,
and label problems remain unverified possibilities.

The sentiment network has a much smaller gap. Both tasks use numerical feature
vectors, but their representations and class structures differ. A configuration
that helps one task does not necessarily help the other.

The same validation split was used for tuning and model selection. Small score
differences may reflect this split or initialization, and the best validation
scores may be optimistic. They are development measurements rather than an
independent final evaluation. The joint report must compare all required methods
on the shared split before selecting the team's final competition predictors.

## Final refitting and prediction files

The selected image model was refitted on all 20,000 labeled rows for 13 epochs;
the sentiment model used all 12,000 labeled rows for 9 epochs. Image scaler
statistics were refitted on all labeled image rows. Sentiment used no scaler.
The resulting 5,000 predictions per task are combined by ID using the sample
template in `common.build_submission()`.

The prepared individual-NN CSV is
`submissions/neural_net_optimization_best.csv`. It has 10,000 unique template
IDs, columns `id,label`, and valid integer labels. The initial, earlier NN
submission received a reported public score of 0.499. The selected models above
have an equally weighted local validation accuracy of 52.35%; an updated public
score is not recorded here.

The optional [ensemble record](results/neural_net/final/ensemble_probe.json)
contains a separate validation-only probability-averaging check. It reached
38.625% on images, while sentiment blends
did not improve the selected individual model. Those ensembles were not refitted
into the prepared prediction CSV.

Library references are listed in `NEURAL_NETWORK.md`.
