# -*- coding: utf-8 -*-
"""
generalize_illusion.py
======================
Show the "accuracy illusion" generalizes BEYOND CIC IoT-DIAD 2024: train
leakage-free, class-weighted XGBoost on UNSW-NB15 and NSL-KDD (their official
train/test splits), and report accuracy vs balanced-accuracy vs macro-F1 for
binary and multi-class. If high accuracy coincides with much lower macro-F1 /
balanced accuracy on these independent datasets too, the finding is a general
property of imbalanced IDS evaluation, not an artifact of one dataset.

Output: results_2026_03_03/generalization/illusion_multi_dataset.json
"""
import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'results_2026_03_03' / 'generalization'
OUT.mkdir(parents=True, exist_ok=True)

NSL_COLS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty']
NSL_CAT = {'protocol_type','service','flag'}
NSL_DOS = {'back','land','neptune','pod','smurf','teardrop','mailbomb','apache2',
    'processtable','udpstorm','worm'}
NSL_PROBE = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
NSL_R2L = {'ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient',
    'warezmaster','sendmail','named','snmpgetattack','snmpguess','xlock','xsnoop',
    'httptunnel'}
NSL_U2R = {'buffer_overflow','loadmodule','perl','rootkit','ps','sqlattack','xterm'}


def nsl_category(lbl):
    if lbl == 'normal': return 'normal'
    if lbl in NSL_DOS: return 'DoS'
    if lbl in NSL_PROBE: return 'Probe'
    if lbl in NSL_R2L: return 'R2L'
    if lbl in NSL_U2R: return 'U2R'
    return 'other'


def metrics(y, p):
    return {'accuracy': round(float(accuracy_score(y, p)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y, p)), 4),
            'f1_macro': round(float(f1_score(y, p, average='macro', zero_division=0)), 4),
            'f1_weighted': round(float(f1_score(y, p, average='weighted', zero_division=0)), 4)}


def train_eval(Xtr, ytr, Xte, yte, n_cls):
    counts = np.bincount(ytr, minlength=n_cls)
    w = (counts.sum() / (n_cls * np.maximum(counts, 1)))[ytr]
    obj = 'binary:logistic' if n_cls == 2 else 'multi:softmax'
    kw = {} if n_cls == 2 else {'num_class': n_cls}
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            tree_method='hist', device='cuda', objective=obj,
                            eval_metric='logloss', n_jobs=-1, verbosity=0, **kw)
    clf.fit(Xtr, ytr, sample_weight=w)
    return metrics(yte, clf.predict(Xte))


def encode(df_tr, df_te, cat_cols):
    """Joint label-encode categoricals so train/test share category codes."""
    df_tr, df_te = df_tr.copy(), df_te.copy()
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([df_tr[c], df_te[c]]).astype(str))
        df_tr[c] = le.transform(df_tr[c].astype(str))
        df_te[c] = le.transform(df_te[c].astype(str))
    return df_tr, df_te


def run_unsw():
    tr = pd.read_csv(DATA / 'UNSW-NB15' / 'UNSW_NB15_training-set.csv')
    te = pd.read_csv(DATA / 'UNSW-NB15' / 'UNSW_NB15_testing-set.csv')
    tr.columns = [c.strip().lstrip('﻿') for c in tr.columns]
    te.columns = [c.strip().lstrip('﻿') for c in te.columns]
    drop = ['id', 'label', 'attack_cat']
    cat = ['proto', 'service', 'state']
    tr, te = encode(tr, te, cat)
    feats = [c for c in tr.columns if c not in drop]
    Xtr, Xte = tr[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
    out = {}
    # binary
    out['binary'] = train_eval(Xtr, tr['label'].to_numpy(int),
                               Xte, te['label'].to_numpy(int), 2)
    # multiclass (attack_cat)
    le = LabelEncoder(); le.fit(pd.concat([tr['attack_cat'], te['attack_cat']]).astype(str))
    out['multiclass'] = train_eval(Xtr, le.transform(tr['attack_cat'].astype(str)),
                                   Xte, le.transform(te['attack_cat'].astype(str)),
                                   len(le.classes_))
    out['n_classes_multiclass'] = int(len(le.classes_))
    out['n_train'], out['n_test'] = int(len(tr)), int(len(te))
    return out


def run_nsl():
    tr = pd.read_csv(DATA / 'nsl-kdd' / 'train.txt', names=NSL_COLS)
    te = pd.read_csv(DATA / 'nsl-kdd' / 'test.txt', names=NSL_COLS)
    tr, te = encode(tr, te, list(NSL_CAT))
    feats = [c for c in NSL_COLS if c not in ('label', 'difficulty')]
    Xtr, Xte = tr[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)
    out = {}
    # binary
    ytr_b = (tr['label'] != 'normal').astype(int).to_numpy()
    yte_b = (te['label'] != 'normal').astype(int).to_numpy()
    out['binary'] = train_eval(Xtr, ytr_b, Xte, yte_b, 2)
    # multiclass (5 categories)
    cats = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
    cmap = {c: i for i, c in enumerate(cats)}
    ytr_m = tr['label'].map(nsl_category).map(cmap).to_numpy()
    yte_m = te['label'].map(nsl_category).map(cmap).to_numpy()
    out['multiclass'] = train_eval(Xtr, ytr_m, Xte, yte_m, 5)
    out['n_classes_multiclass'] = 5
    out['n_train'], out['n_test'] = int(len(tr)), int(len(te))
    return out


def main():
    results = {}
    print('=== UNSW-NB15 ==='); results['UNSW-NB15'] = run_unsw()
    print('  binary:', results['UNSW-NB15']['binary'])
    print('  multiclass:', results['UNSW-NB15']['multiclass'])
    print('=== NSL-KDD ==='); results['NSL-KDD'] = run_nsl()
    print('  binary:', results['NSL-KDD']['binary'])
    print('  multiclass:', results['NSL-KDD']['multiclass'])
    json.dump(results, open(OUT / 'illusion_multi_dataset.json', 'w'), indent=2)

    print('\n' + '=' * 64)
    print('ACCURACY ILLUSION ACROSS DATASETS (acc vs bal-acc vs macro-F1)')
    print('=' * 64)
    print(f"{'Dataset':<14}{'Mode':<12}{'Acc':>8}{'BalAcc':>9}{'F1mac':>8}{'gap':>8}")
    for ds in results:
        for mode in ['binary', 'multiclass']:
            m = results[ds][mode]
            gap = round(m['accuracy'] - m['f1_macro'], 3)
            print(f"{ds:<14}{mode:<12}{m['accuracy']:>8.3f}{m['balanced_accuracy']:>9.3f}"
                  f"{m['f1_macro']:>8.3f}{gap:>8.3f}")
    print(f"\n[DONE] {OUT / 'illusion_multi_dataset.json'}")


if __name__ == '__main__':
    main()
