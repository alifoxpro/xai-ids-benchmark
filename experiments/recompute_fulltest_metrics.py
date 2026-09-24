# -*- coding: utf-8 -*-
"""
recompute_fulltest_metrics.py
=============================
Reconcile Tables 1 and 2: compute overall AND per-class metrics for the best
single model (FT-Transformer) on the FULL multi-class test set (not a subsample),
so the headline G-Mean and the per-class breakdown come from the same data.
Adds MCC (overall + per-class one-vs-rest), Cohen's kappa, and full-test support.

Output: results_2026_03_03/multiclass/fulltest/ft_transformer_fulltest.json
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import torch
from scipy.stats import gmean
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score, recall_score,
                             precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
MC = ROOT / 'results_2026_03_03' / 'multiclass'
NPZ = MC / 'stage1_cache' / 'arrays.npz'
META = json.load(open(MC / 'stage1_cache' / 'metadata.json'))
MODEL = ROOT / 'models' / 'multiclass' / 'stage3' / 'FT_Transformer.pkl'
OUT = MC / 'fulltest'; OUT.mkdir(parents=True, exist_ok=True)

CLASSES = list(META['label_mapping'].keys())   # index-ordered


def main():
    print('[INFO] loading full test set + FT-Transformer')
    d = np.load(NPZ, allow_pickle=True)
    Xte, yte = d['X_test'], d['y_test'].astype(int)
    print(f'[INFO] full test = {Xte.shape}, class counts = {np.bincount(yte)}')

    obj = pickle.load(open(MODEL, 'rb'))
    net = (obj.model if hasattr(obj, 'model') else obj).cuda().eval()
    # arrays.npz is ALREADY standardized (the models were trained on it directly);
    # do NOT re-apply the scaler or the input is double-scaled.
    Xs = Xte.astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 16384):
            out = net(torch.as_tensor(Xs[i:i+16384], device='cuda'))
            if isinstance(out, (tuple, list)):
                out = out[-1]
            preds.append(out.argmax(-1).cpu().numpy())
    p = np.concatenate(preds)

    # ---- overall
    recalls = recall_score(yte, p, average=None, zero_division=0)
    overall = {
        'model': 'FT_Transformer',
        'n_test': int(len(yte)),
        'accuracy': float(accuracy_score(yte, p)),
        'balanced_accuracy': float(balanced_accuracy_score(yte, p)),
        'f1_macro': float(f1_score(yte, p, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(yte, p, average='weighted', zero_division=0)),
        'g_mean': float(gmean(np.clip(recalls, 1e-12, None))),
        'mcc': float(matthews_corrcoef(yte, p)),
        'cohen_kappa': float(cohen_kappa_score(yte, p)),
    }

    # ---- per-class (one-vs-rest)
    per_class = []
    cm = confusion_matrix(yte, p, labels=list(range(len(CLASSES))))
    for c, name in enumerate(CLASSES):
        y_bin = (yte == c).astype(int)
        p_bin = (p == c).astype(int)
        support = int((yte == c).sum())
        rec = float(recall_score(y_bin, p_bin, zero_division=0))
        per_class.append({
            'class': name,
            'precision': float(precision_score(y_bin, p_bin, zero_division=0)),
            'recall': rec,
            'f1': float(f1_score(y_bin, p_bin, zero_division=0)),
            'fnr': float(1 - rec),
            'mcc': float(matthews_corrcoef(y_bin, p_bin)) if support > 0 else 0.0,
            'support': support,
        })

    result = {'overall': overall, 'per_class': per_class}
    json.dump(result, open(OUT / 'ft_transformer_fulltest.json', 'w'), indent=2)

    print('\n=== OVERALL (full test) ===')
    for k, v in overall.items():
        print(f'  {k}: {v}')
    print('\n=== PER-CLASS (full test) ===')
    print(f"{'Class':<12}{'Prec':>8}{'Rec':>8}{'F1':>8}{'FNR':>8}{'MCC':>8}{'Support':>10}")
    for r in per_class:
        print(f"{r['class']:<12}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['fnr']:>8.3f}{r['mcc']:>8.3f}{r['support']:>10,}")
    print(f"\n[DONE] {OUT / 'ft_transformer_fulltest.json'}")


if __name__ == '__main__':
    main()
