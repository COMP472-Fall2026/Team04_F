## k-NN (Ethan Elkaim)

### Contents
- `knn.py`: from-scratch k-NN (NumPy), the preprocessing variants, the validation
  experiments, and the final train/predict run.
- `common.py`: shared code. Loads the CSVs, makes the stratified train/validation
  split, L2 normalization, standardization, accuracy, confusion matrix,
  result logging, and Kaggle submission writing.
- `data/`: Kaggle CSV files (not modified).
- `results/`: experiment logs (`knn-results.csv`, `knn-sqrt-lda-all.csv`) and
  saved predictions.
- `submissions/`: Kaggle submission files (`sub_knn_v4.csv` is the final one).

### Requirements
Python 3 with `numpy` and `pandas`. No scikit-learn model code is used.

### Data
Place `image_train.csv`, `image_test.csv`, `sentiment_train.csv`,
`sentiment_test.csv` and `sample_submission.csv` in `data/`.

### Pre-processing (done inside knn.py)
1. Stratified 80/20 train/validation split (seed 42, set in `common.py`).
2. Image only: square root of each feature, then L2 normalization of each row.
3. LDA is fitted on the training split only, then applied to validation/test.
   Image: 49 LDA dimensions appended to the 500 features (weight 0.5, shrinkage 1000).
   Sentiment: the features are replaced by 2 LDA dimensions (shrinkage 0.1).

### Train
python knn.py
k-NN has no training step: it stores the (preprocessed) training data.
There are no epochs, learning rate or loss function.

### Evaluate
The same run prints validation accuracy for both tasks (images 0.6022,
sentiment 0.6746) using k=20, p=64 (image) and k=200, p=0 (sentiment).
Set `RUN_SWEEPS = True` at the top of `knn.py` to rerun the hyperparameter
sweeps (k, voting power, standardization, cosine, square root, LDA); this
is slow and off by default. Results go to `results/knn-results.csv`.

### Apply
The run then refits on the full training set, predicts the test set and
writes `submissions/sub_knn_v4.csv`.