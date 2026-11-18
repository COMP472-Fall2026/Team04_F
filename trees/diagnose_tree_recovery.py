"""Read-only diagnostic for the recovered COMP472 decision-tree grid.
Run beside tune_tree_recovery.py and resume_tree_from_log.py.
Does not write, restore or delete checkpoint files.
"""
import hashlib
import inspect
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import common
import trees.tune_tree_recovery as tuning
import trees.resume_tree_from_log as recovery


def sha256(path, normalize_newlines=False):
    data = Path(path).read_bytes()
    if normalize_newlines:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def main():
    recovery.check_settings()
    print('Python:', sys.version)
    print('Platform:', platform.platform())
    print('NumPy:', np.__version__)
    print('Pandas:', pd.__version__)
    print('DecisionTree module:', tuning.DecisionTree.__module__)
    for label, path, normalize in [
        ('common.py', common.__file__, True),
        ('DecisionTree source file', inspect.getfile(tuning.DecisionTree), True),
        ('image_train.csv', Path(common.DATA_DIR) / 'image_train.csv', True),
    ]:
        print(f'{label}: {path}\n  SHA256 (LF-normalized): {sha256(path, normalize)}', flush=True)
    print('DecisionTree class hash:', hashlib.sha256(inspect.getsource(tuning.DecisionTree).encode()).hexdigest())
    print('Recovered CSV:', recovery.SOURCE)
    saved = pd.read_csv(recovery.SOURCE)
    configs = tuning.build_configurations()
    folds = tuning.prepare_folds(tuning.TASK)
    print('Fold sizes:', [(len(f[1]), len(f[3])) for f in folds])
    print('Current experiment signature:', tuning.experiment_signature(tuning.TASK, configs, folds))
    target = Path(tuning.experiment_prefix(tuning.TASK) + '_runs.csv')
    if target.exists():
        checkpoint = pd.read_csv(target)
        print('Current checkpoint rows:', len(checkpoint))
        if 'experiment_signature' in checkpoint:
            print('Current checkpoint signatures:', checkpoint['experiment_signature'].unique().tolist())
    print('\nComparing four fits. No results will be saved.', flush=True)
    for iteration in (1, 2, 26, 1):
        i = iteration - 1
        row = saved.iloc[i]
        fold = int(row['fold'])
        config = configs[i // 5]
        assert config['id'] == row['config_id']
        actual = tuning.evaluate_configuration(tuning.TASK, config, fold, int(row['tree_seed']), folds[fold - 1])
        train = actual['training_accuracy']
        val = actual['validation_accuracy']
        train_ok = row['training_accuracy_min'] - 1e-12 <= train <= row['training_accuracy_max'] + 1e-12
        val_ok = abs(val - row['validation_accuracy']) <= 1e-12
        print(f"Iteration {iteration} | {config['id']} | fold {fold} | {config['parameters']}")
        print(f"  Original training: {row['training_accuracy']:.6%} (allowed {row['training_accuracy_min']:.6%}–{row['training_accuracy_max']:.6%})")
        print(f"  Current training:  {train:.6%}")
        print(f"  Original validation: {row['validation_accuracy']:.6%}")
        print(f"  Current validation:  {val:.6%}")
        print(f"  {'MATCH' if train_ok and val_ok else 'MISMATCH'}", flush=True)
    print('\nDiagnostics finished. No checkpoint changed.')


if __name__ == '__main__':
    main()
