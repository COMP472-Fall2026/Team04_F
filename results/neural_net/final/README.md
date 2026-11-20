# Selected neural-network results

This folder contains the selected individual models' supporting results.
[NEURAL_NETWORK_RESULTS.md](../../../NEURAL_NETWORK_RESULTS.md) contains the
analysis, and [results/results.csv](../../results.csv) retains all trial
summaries, including unsuccessful configurations.

| Task | Hidden layers | Train accuracy | Validation accuracy | Selected epoch |
| --- | --- | ---: | ---: | ---: |
| Images | 512/256, batch normalization | 99.89% | 36.65% | 13 |
| Sentiment | 64 | 71.51% | 68.04% | 9 |

Both use ReLU, Adam at 0.001, weight decay 0.0001, dropout 0.2, batch size 128,
and model seed 42. Images use training-fitted standardization; sentiment uses
original features. The shared stratified validation split uses seed 42 and 20%
of each class. The equally weighted mean validation accuracy is 52.35%.

Each task folder contains `best.json`, `history.csv`, learning curves, a class
report, confusion matrix, frequent class confusions, environment details, and
final-training measurements. `sources.json` records the copied files' origins
and hashes. Historical run folders remain local.

`regularization_tradeoff.png` supports the regularization analysis. The optional
`ensemble_probe.json` records an exploratory validation-only averaging check.
The prepared prediction CSV uses the two individual models above.

## Reproduce the models

Follow the setup in [README.md](../../../README.md), then run these commands
from the repository root. Output directories must be new.

```bash
python neural_net.py --task image --hidden 512 256 --batch-norm \
    --dropout 0.2 --weight-decay 0.0001 --learning-rate 0.001 --batch-size 128 \
    --epochs 80 --patience 12 --early-stopping-metric accuracy --seed 42 \
    --device cpu --threads 2 --finalize --output-dir results/neural_net/reproduce_image

python neural_net.py --task sentiment --hidden 64 --no-standardize \
    --dropout 0.2 --weight-decay 0.0001 --learning-rate 0.001 --batch-size 128 \
    --epochs 80 --patience 12 --early-stopping-metric accuracy --seed 42 \
    --device cpu --threads 2 --finalize --output-dir results/neural_net/reproduce_sentiment
```

Validation selects the epoch count; refitting then uses all labeled rows.
The image model trains for 13 epochs and sentiment for 9 in the recorded runs.
Combine their test predictions in template order:

```bash
python - <<'PYTHON'
import pandas as pd
from common import build_submission

image = pd.read_csv('results/neural_net/reproduce_image/image/test_predictions.csv')
sentiment = pd.read_csv('results/neural_net/reproduce_sentiment/sentiment/test_predictions.csv')
build_submission(image['id'], image['label'], sentiment['id'], sentiment['label'],
                 out_name='neural_net_reproduced.csv')
PYTHON
```

The existing prepared upload is `submissions/neural_net_optimization_best.csv`.
Datasets, weights, predictions, and caches stay local. Library references are
recorded in `NEURAL_NETWORK.md`.
