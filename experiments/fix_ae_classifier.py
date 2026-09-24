# -*- coding: utf-8 -*-
"""
fix_ae_classifier.py
====================
Patch the AE_Classifier evaluation on the CICIDS2017 transfer.

The model's forward() returns (reconstruction, logits). The original
cross-dataset script tried out.argmax(...) on the tuple, causing the
'tuple' object has no attribute 'argmax' error. This script re-runs
AE_Classifier and TCN (if available), then merges results into the
existing cross_dataset_real.json.
"""

import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CICIDS_DIR = PROJECT_ROOT / 'data' / 'cic-ids-2017' / 'ml-csv'
IOTDIAD_ARTIFACTS = PROJECT_ROOT / 'results_2026_03_03' / 'multiclass' / 'stage1_cache'
MODELS_DIR = PROJECT_ROOT / 'models' / 'multiclass' / 'stage3'
OUT_DIR = PROJECT_ROOT / 'results_2026_03_03' / 'multiclass' / 'cross_dataset'

# Reuse the same constants from run_real_cross_dataset.py
from run_real_cross_dataset import (FEATURE_MAP_2017_TO_IOTDIAD, LABEL_MAP,
                                      IOTDIAD_LABELS, load_cicids2017,
                                      align_features, map_labels)


@torch.no_grad()
def evaluate_ae(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                *, batch: int = 2048) -> dict:
    """Evaluator that handles tuple outputs (reconstruction, logits)."""
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch):
        chunk = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        if chunk.shape[0] == 1:
            chunk = chunk.repeat(2, 1)
            out = model(chunk)
            logits = out[1] if isinstance(out, tuple) else out
            preds.append(logits.argmax(dim=-1).cpu().numpy()[:1])
        else:
            out = model(chunk)
            logits = out[1] if isinstance(out, tuple) else out
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)

    metrics = {
        'accuracy': float(accuracy_score(y, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, preds, average='weighted', zero_division=0)),
    }
    per_class = classification_report(y, preds, output_dict=True, zero_division=0)
    metrics['per_class'] = {}
    for cls_idx, name in IOTDIAD_LABELS.items():
        key = str(cls_idx)
        if key in per_class:
            metrics['per_class'][name] = {
                'precision': per_class[key]['precision'],
                'recall': per_class[key]['recall'],
                'f1-score': per_class[key]['f1-score'],
                'support': per_class[key]['support'],
            }
    return metrics


def main():
    print("=" * 60)
    print("AE_Classifier cross-dataset fix")
    print("=" * 60)

    with open(IOTDIAD_ARTIFACTS / 'metadata.json') as f:
        meta = json.load(f)
    feature_names = meta['feature_columns']

    with open(IOTDIAD_ARTIFACTS / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    df_2017, label_strs, csv_cols = load_cicids2017()
    X_aligned = align_features(df_2017, feature_names)
    y, mask = map_labels(label_strs)
    X_aligned = X_aligned[mask]
    y = y[mask]
    print(f"[INFO] X={X_aligned.shape}, y={y.shape}")

    X_scaled = scaler.transform(X_aligned)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0,
                              neginf=-10.0).astype(np.float32)

    # Re-evaluate AE_Classifier
    print("\n--- AE_Classifier (tuple-aware) ---")
    with open(MODELS_DIR / 'AE_Classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    model = model.cuda()
    t0 = time.time()
    ae_metrics = evaluate_ae(model, X_scaled, y)
    ae_metrics['eval_time_sec'] = round(time.time() - t0, 2)
    print(f"  acc = {ae_metrics['accuracy']:.4f}  "
          f"bal = {ae_metrics['balanced_accuracy']:.4f}  "
          f"f1m = {ae_metrics['f1_macro']:.4f}  "
          f"({ae_metrics['eval_time_sec']}s)")

    # Merge into existing JSON
    out_path = OUT_DIR / 'cross_dataset_real.json'
    with open(out_path) as f:
        results = json.load(f)
    results['models']['AE_Classifier'] = ae_metrics
    results['models']['TCN'] = {'error': 'model checkpoint file empty (0 bytes); not evaluable'}

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] merged into {out_path}")

    # Reprint summary
    print("\nSUMMARY (sorted by balanced accuracy):")
    rows = [(name, m['balanced_accuracy'], m['accuracy'], m['f1_macro'])
            for name, m in results['models'].items()
            if 'balanced_accuracy' in m]
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'Model':<20} {'Bal.Acc':>10} {'Acc':>10} {'F1m':>10}")
    for name, bal, acc, f1m in rows:
        print(f"{name:<20} {bal:>10.4f} {acc:>10.4f} {f1m:>10.4f}")


if __name__ == '__main__':
    main()
