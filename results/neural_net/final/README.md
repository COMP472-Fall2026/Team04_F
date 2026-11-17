# Neural-network submission evidence

This folder is the compact evidence package for the neural-network component.
The full experiment log, including unsuccessful trials, is in
[results/results.csv](../../results.csv). Implementation instructions and the
analysis are in [NEURAL_NETWORK.md](../../../NEURAL_NETWORK.md) and
[NEURAL_NETWORK_RESULTS.md](../../../NEURAL_NETWORK_RESULTS.md).

## Selected individual networks

| Task | Architecture | Training accuracy | Validation accuracy | Selected epoch |
| --- | --- | ---: | ---: | ---: |
| Images | 500 -> 512 -> 256 -> 50, batch normalization, ReLU, dropout 0.2 | 99.89% | 36.65% | 13 |
| Sentiment | 768 -> 64 -> 3, ReLU, dropout 0.2 | 71.51% | 68.04% | 9 |

Both use Adam with learning rate 0.001, weight decay 0.0001, minibatches of 128,
and model seed 42. Images use training-fitted input standardization; sentiment
uses the supplied features without standardization. Validation uses the shared
stratified 80/20 split with seed 42. The mean validation accuracy is 52.35%.
These validation scores precede the final refit on all labeled examples and are
not hidden-test measurements.

`image/` and `sentiment/` contain the selected model's metrics, training history,
learning curves, per-class report, confusion matrix, frequent class confusions,
runtime/data metadata, and final-training record. `sources.json` records the
source files and SHA-256 hashes of copied evidence. Original experiment folders
are retained locally and excluded from Git.

`regularization_tradeoff.png` supports the regularization analysis. The compact
`ensemble_probe.json` records a validation-only averaging experiment: its best
image accuracy was 38.625%, while the sentiment blends did not improve the best
individual network. Those ensemble members have not been refitted and packaged
as an ensemble submission; the selected individual-model reports above remain
the basis of the prepared CSV.

## Reproduce the selected networks

Run from the repository root with Python 3.11+ and the dependencies in
`requirements.txt`. Place `image_train.csv`, `image_test.csv`,
`sentiment_train.csv`, `sentiment_test.csv`, and `sample_submission.csv` in
`data/`. Output directories must be new; choose different names for repeated runs.

```bash
.venv/bin/python neural_net.py --task image --hidden 512 256 --batch-norm \
    --dropout 0.2 --weight-decay 0.0001 --learning-rate 0.001 --batch-size 128 \
    --epochs 80 --patience 12 --early-stopping-metric accuracy --seed 42 \
    --device cpu --threads 2 --finalize --output-dir results/neural_net/reproduce_image

.venv/bin/python neural_net.py --task sentiment --hidden 64 --no-standardize \
    --dropout 0.2 --weight-decay 0.0001 --learning-rate 0.001 --batch-size 128 \
    --epochs 80 --patience 12 --early-stopping-metric accuracy --seed 42 \
    --device cpu --threads 2 --finalize --output-dir results/neural_net/reproduce_sentiment
```

Combine the two refitted models' predictions in template ID order:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from common import build_submission

image = pd.read_csv('results/neural_net/reproduce_image/image/test_predictions.csv')
sentiment = pd.read_csv('results/neural_net/reproduce_sentiment/sentiment/test_predictions.csv')
build_submission(image['id'], image['label'], sentiment['id'], sentiment['label'],
                 out_name='neural_net_reproduced.csv')
PY
```

The existing prepared upload is `submissions/neural_net_optimization_best.csv`.
It contains predictions from an image NN and a sentiment NN, not a KNN/NN
combination. Prediction files, model weights, datasets, and caches stay local.
The script creates files locally; uploading to Kaggle is a separate action.

Test the implementation with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_neural_net.py' -v
```

AI assistance is acknowledged in the neural-network documentation and analysis
and should be included in the team's final report.
