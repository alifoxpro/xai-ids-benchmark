# -*- coding: utf-8 -*-
"""
audit_binary_metrics.py
=======================
Definitive audit of the binary-mode numbers: evaluate EVERY binary checkpoint on
the FULL binary test set (2.54M flows) and report accuracy, balanced accuracy,
per-class precision/recall, and the operating point. Resolves the discrepancy
between the paper's §4.3 (94.3%) and the independently verified 74.4%.

Output: results_2026_03_03/binary/fulltest/binary_fulltest_audit.json
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
CACHE = ROOT / 'results_2026_03_03' / 'binary' / 'stage1_cache'
MODELS = ROOT / 'models' / 'binary' / 'stage3'
OUT = ROOT / 'results_2026_03_03' / 'binary' / 'fulltest'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = np.load(CACHE / 'arrays.npz', allow_pickle=True)
    X, y = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    counts = np.bincount(y)
    print(f'[INFO] FULL binary test: {len(y):,} flows  benign={counts[0]:,} attack={counts[1]:,}')
    print(f'[INFO] majority-class baseline accuracy = {counts.max()/len(y):.4f}\n')

    results = {'n_test': int(len(y)),
               'class_counts': {'benign': int(counts[0]), 'attack': int(counts[1])},
               'majority_baseline_acc': float(counts.max() / len(y)),
               'models': {}}

    for mf in sorted(MODELS.glob('*.pkl')):
        if mf.stat().st_size == 0:
            print(f'{mf.stem:<18} EMPTY checkpoint, skip'); continue
        try:
            obj = pickle.load(open(mf, 'rb'))
            net = (obj.model if hasattr(obj, 'model') else obj).cuda().eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), 16384):
                    o = net(torch.as_tensor(X[i:i+16384], device='cuda'))
                    o = o[-1] if isinstance(o, (tuple, list)) else o
                    preds.append(o.argmax(-1).cpu().numpy())
            p = np.concatenate(preds)
            m = {
                'accuracy': round(float(accuracy_score(y, p)), 4),
                'balanced_accuracy': round(float(balanced_accuracy_score(y, p)), 4),
                'f1_weighted': round(float(f1_score(y, p, average='weighted')), 4),
                'attack_recall': round(float(recall_score(y, p, pos_label=1)), 4),
                'attack_precision': round(float(precision_score(y, p, pos_label=1)), 4),
                'benign_recall': round(float(recall_score(y, p, pos_label=0)), 4),
                'benign_precision': round(float(precision_score(y, p, pos_label=0)), 4),
            }
            results['models'][mf.stem] = m
            print(f"{mf.stem:<18} acc={m['accuracy']:.4f}  bal={m['balanced_accuracy']:.4f}  "
                  f"att-rec={m['attack_recall']:.4f}  att-prec={m['attack_precision']:.4f}  "
                  f"ben-rec={m['benign_recall']:.4f}")
            del net, obj; torch.cuda.empty_cache()
        except Exception as e:
            print(f'{mf.stem:<18} ERROR {e}')
            results['models'][mf.stem] = {'error': str(e)}

    json.dump(results, open(OUT / 'binary_fulltest_audit.json', 'w'), indent=2)
    print(f"\n[DONE] {OUT / 'binary_fulltest_audit.json'}")


if __name__ == '__main__':
    main()
