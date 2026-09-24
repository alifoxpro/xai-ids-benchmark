# -*- coding: utf-8 -*-
"""
recipe_experiment.py
====================
Constructive contribution: a simple, dataset-agnostic "balanced-IDS recipe"
(leakage-free split + class-weighted training) vs the naive baseline, on all three
datasets. Shows the recipe consistently trades a little accuracy for a large
balanced-accuracy / macro-F1 gain on the rare classes. (On CIC, representation-
level early fusion adds a further +6.7 balanced-accuracy points, see fusion expt.)

Output: results_2026_03_03/generalization/recipe_comparison.json
"""
import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))
from generalize_illusion import (NSL_COLS, NSL_CAT, nsl_category, encode)

DATA = ROOT / 'data'
MC = ROOT / 'results_2026_03_03' / 'multiclass'
OUT = MC.parent / 'generalization'; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def metrics(y, p):
    return {'accuracy': round(float(accuracy_score(y, p)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y, p)), 4),
            'f1_macro': round(float(f1_score(y, p, average='macro', zero_division=0)), 4)}


def fit_xgb(Xtr, ytr, n_cls, weighted):
    w = None
    if weighted:
        c = np.bincount(ytr, minlength=n_cls)
        w = (c.sum() / (n_cls * np.maximum(c, 1)))[ytr]
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            tree_method='hist', device='cuda',
                            objective='multi:softmax', num_class=n_cls,
                            eval_metric='mlogloss', verbosity=0)
    clf.fit(Xtr, ytr, sample_weight=w)
    return clf


def load_cic():
    d = np.load(MC / 'stage1_cache' / 'arrays.npz', allow_pickle=True)
    Xtr, ytr = d['X_train'].astype(np.float32), d['y_train'].astype(int)
    Xte, yte = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xtr), min(150_000, len(Xtr)), replace=False)
    te = rng.choice(len(Xte), min(150_000, len(Xte)), replace=False)
    return Xtr[idx], ytr[idx], Xte[te], yte[te], len(np.unique(ytr))


def load_unsw():
    tr = pd.read_csv(DATA / 'UNSW-NB15' / 'UNSW_NB15_training-set.csv')
    te = pd.read_csv(DATA / 'UNSW-NB15' / 'UNSW_NB15_testing-set.csv')
    tr.columns = [c.strip().lstrip('﻿') for c in tr.columns]
    te.columns = [c.strip().lstrip('﻿') for c in te.columns]
    tr, te = encode(tr, te, ['proto', 'service', 'state'])
    feats = [c for c in tr.columns if c not in ('id', 'label', 'attack_cat')]
    le = LabelEncoder(); le.fit(pd.concat([tr['attack_cat'], te['attack_cat']]).astype(str))
    return (tr[feats].to_numpy(np.float32), le.transform(tr['attack_cat'].astype(str)),
            te[feats].to_numpy(np.float32), le.transform(te['attack_cat'].astype(str)),
            len(le.classes_))


def load_nsl():
    tr = pd.read_csv(DATA / 'nsl-kdd' / 'train.txt', names=NSL_COLS)
    te = pd.read_csv(DATA / 'nsl-kdd' / 'test.txt', names=NSL_COLS)
    tr, te = encode(tr, te, list(NSL_CAT))
    feats = [c for c in NSL_COLS if c not in ('label', 'difficulty')]
    cmap = {c: i for i, c in enumerate(['normal', 'DoS', 'Probe', 'R2L', 'U2R'])}
    ytr = tr['label'].map(nsl_category).map(cmap).to_numpy()
    yte = te['label'].map(nsl_category).map(cmap).to_numpy()
    return tr[feats].to_numpy(np.float32), ytr, te[feats].to_numpy(np.float32), yte, 5


def main():
    loaders = {'CIC IoT-DIAD': load_cic, 'UNSW-NB15': load_unsw, 'NSL-KDD': load_nsl}
    results = {}
    for name, load in loaders.items():
        print(f'=== {name} ===')
        Xtr, ytr, Xte, yte, n = load()
        naive = metrics(yte, fit_xgb(Xtr, ytr, n, weighted=False).predict(Xte))
        recipe = metrics(yte, fit_xgb(Xtr, ytr, n, weighted=True).predict(Xte))
        results[name] = {'naive': naive, 'recipe': recipe,
                         'bal_acc_gain': round(recipe['balanced_accuracy'] - naive['balanced_accuracy'], 3),
                         'acc_cost': round(naive['accuracy'] - recipe['accuracy'], 3)}
        print(f"  naive : {naive}")
        print(f"  recipe: {recipe}")
        print(f"  -> bal-acc gain +{results[name]['bal_acc_gain']}, acc cost -{results[name]['acc_cost']}")
    json.dump(results, open(OUT / 'recipe_comparison.json', 'w'), indent=2)

    print('\n' + '=' * 70)
    print(f"{'Dataset':<16}{'Variant':<9}{'Acc':>8}{'BalAcc':>9}{'F1mac':>8}")
    for ds, r in results.items():
        print(f"{ds:<16}{'naive':<9}{r['naive']['accuracy']:>8.3f}{r['naive']['balanced_accuracy']:>9.3f}{r['naive']['f1_macro']:>8.3f}")
        print(f"{'':<16}{'recipe':<9}{r['recipe']['accuracy']:>8.3f}{r['recipe']['balanced_accuracy']:>9.3f}{r['recipe']['f1_macro']:>8.3f}")
    print(f"\n[DONE] {OUT / 'recipe_comparison.json'}")


if __name__ == '__main__':
    main()
