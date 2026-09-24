# -*- coding: utf-8 -*-
"""
run_smote_leakage_demo.py
=========================
Demonstrate the inflation caused by applying SMOTE BEFORE the train/test split
(the common but incorrect protocol) versus the correct leakage-free protocol.

Protocol A (LEAKY):   SMOTE on the whole dataset -> then split -> train -> test
Protocol B (CORRECT): split -> SMOTE on TRAIN ONLY -> train -> test on untouched test
Protocol C (BASELINE): split -> class-weighted training (our protocol), no SMOTE

Same classifier (XGBoost) and same final test set size for a fair accuracy read.
Output: results_2026_03_03/multiclass/leakage_demo/smote_leakage.json
"""
import io, os, sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='backslashreplace', line_buffering=True)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / 'results_2026_03_03' / 'multiclass' / 'stage1_cache' / 'arrays.npz'
OUT = ROOT / 'results_2026_03_03' / 'multiclass' / 'leakage_demo'
OUT.mkdir(parents=True, exist_ok=True)

SUBSAMPLE = 120_000   # keep SMOTE + XGBoost tractable while preserving imbalance
SEED = 42


def metrics(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p)),
            'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}


def fit_xgb(Xtr, ytr, weight=None):
    n_cls = len(np.unique(ytr))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                            tree_method='hist', device='cuda',
                            objective='multi:softmax', num_class=n_cls,
                            eval_metric='mlogloss', n_jobs=-1, verbosity=0)
    clf.fit(Xtr, ytr, sample_weight=weight)
    return clf


def main():
    print('=' * 70)
    print('SMOTE LEAKAGE DEMONSTRATION (XGBoost, CIC IoT-DIAD 2024 multiclass)')
    print('=' * 70)
    d = np.load(NPZ, allow_pickle=True)
    X = np.concatenate([d['X_train'], d['X_test']])
    y = np.concatenate([d['y_train'], d['y_test']]).astype(int)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(SUBSAMPLE, len(X)), replace=False)
    X, y = X[idx].astype(np.float32), y[idx]
    print(f'[INFO] working set: {X.shape}, classes={np.bincount(y)}')

    results = {}

    # ---------------- Protocol A: LEAKY (SMOTE before split) ----------------
    print('\n[A] LEAKY: SMOTE on all data, THEN split')
    t0 = time.time()
    Xall, yall = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X, y)
    XtrA, XteA, ytrA, yteA = train_test_split(Xall, yall, test_size=0.3,
                                              random_state=SEED, stratify=yall)
    clfA = fit_xgb(XtrA, ytrA)
    mA = metrics(yteA, clfA.predict(XteA))
    mA['note'] = 'test set contains SMOTE-synthetic points correlated with train'
    results['A_leaky_smote_before_split'] = mA
    print(f'    acc={mA["accuracy"]:.4f} bal={mA["balanced_accuracy"]:.4f} '
          f'f1m={mA["f1_macro"]:.4f}  ({time.time()-t0:.0f}s)')

    # ---------------- correct split (shared by B and C) ----------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)

    # ---------------- Protocol B: CORRECT (SMOTE on train only) ----------------
    print('\n[B] CORRECT: split first, SMOTE on TRAIN ONLY, test untouched')
    t0 = time.time()
    XtrB, ytrB = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xtr, ytr)
    clfB = fit_xgb(XtrB, ytrB)
    mB = metrics(yte, clfB.predict(Xte))
    results['B_correct_smote_train_only'] = mB
    print(f'    acc={mB["accuracy"]:.4f} bal={mB["balanced_accuracy"]:.4f} '
          f'f1m={mB["f1_macro"]:.4f}  ({time.time()-t0:.0f}s)')

    # ---------------- Protocol C: our protocol (class weights, no SMOTE) ----------------
    print('\n[C] OURS: split first, class-weighted training, no SMOTE')
    t0 = time.time()
    cw = len(ytr) / (len(np.unique(ytr)) * np.bincount(ytr)[ytr])
    clfC = fit_xgb(Xtr, ytr, weight=cw)
    mC = metrics(yte, clfC.predict(Xte))
    results['C_ours_class_weighted'] = mC
    print(f'    acc={mC["accuracy"]:.4f} bal={mC["balanced_accuracy"]:.4f} '
          f'f1m={mC["f1_macro"]:.4f}  ({time.time()-t0:.0f}s)')

    results['inflation'] = {
        'accuracy_A_minus_B': round(mA['accuracy'] - mB['accuracy'], 4),
        'f1_macro_A_minus_B': round(mA['f1_macro'] - mB['f1_macro'], 4),
    }
    json.dump(results, open(OUT / 'smote_leakage.json', 'w'), indent=2)
    print('\n' + '=' * 70)
    print(f'INFLATION from leakage:  +{results["inflation"]["accuracy_A_minus_B"]:.4f} acc, '
          f'+{results["inflation"]["f1_macro_A_minus_B"]:.4f} macro-F1')
    print(f'[DONE] {OUT / "smote_leakage.json"}')


if __name__ == '__main__':
    main()
