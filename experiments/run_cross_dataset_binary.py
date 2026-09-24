# -*- coding: utf-8 -*-
"""
run_cross_dataset_binary.py
===========================
CLEAN cross-dataset transfer: CIC IoT-DIAD 2024 (binary) -> CICIDS2017 (binary).

Binary mapping is unambiguous: BENIGN -> 0, every attack -> 1. This removes the
fragile 8-class label mapping (Bot->Mirai, Infiltration->Spoofing, ...) that
weakens the multi-class transfer, so the measured generalization gap reflects
detection ability rather than taxonomy mismatch.

Output: results_2026_03_03/binary/cross_dataset/cross_dataset_binary.json
"""
import io, os, sys, json, pickle, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='backslashreplace', line_buffering=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, classification_report)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
CICIDS_DIR = ROOT / 'data' / 'cic-ids-2017' / 'ml-csv'
BIN_CACHE = ROOT / 'results_2026_03_03' / 'binary' / 'stage1_cache'
MODELS = ROOT / 'models' / 'binary' / 'stage3'
OUT = ROOT / 'results_2026_03_03' / 'binary' / 'cross_dataset'
OUT.mkdir(parents=True, exist_ok=True)

# reuse the validated feature map from the multiclass script
from run_real_cross_dataset import FEATURE_MAP_2017_TO_IOTDIAD, align_features, load_cicids2017


@torch.no_grad()
def evaluate(model, X, y, batch=8192):
    dev = next(model.parameters()).device
    model.eval(); preds = []
    for i in range(0, len(X), batch):
        out = model(torch.as_tensor(X[i:i+batch], dtype=torch.float32, device=dev))
        if isinstance(out, (tuple, list)):
            out = out[-1]
        preds.append(out.argmax(-1).cpu().numpy())
    p = np.concatenate(preds)
    rep = classification_report(y, p, output_dict=True, zero_division=0)
    # Model/dataset label space: Attack = 0, Benign = 1 (per stage1 metadata).
    return {
        'accuracy': float(accuracy_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'f1_macro': float(f1_score(y, p, average='macro', zero_division=0)),
        'attack_recall': float(rep.get('0', {}).get('recall', 0.0)),
        'attack_precision': float(rep.get('0', {}).get('precision', 0.0)),
        'benign_recall': float(rep.get('1', {}).get('recall', 0.0)),
    }


def main():
    print('=' * 70)
    print('CLEAN BINARY CROSS-DATASET: CIC IoT-DIAD 2024 -> CICIDS2017')
    print('=' * 70)
    meta = json.load(open(BIN_CACHE / 'metadata.json'))
    feature_names = meta['feature_columns']
    scaler = pickle.load(open(BIN_CACHE / 'scaler.pkl', 'rb'))
    print(f'[INFO] binary features = {len(feature_names)}, labels = {meta["label_mapping"]}')

    df_2017, label_strs, _ = load_cicids2017()
    # BINARY mapping matching the MODEL's label space (metadata: Attack=0, Benign=1):
    # CICIDS2017 BENIGN -> 1, every attack -> 0. Unambiguous, no rows dropped.
    y = np.where(np.char.upper(label_strs.astype(str)) == 'BENIGN', 1, 0).astype(np.int64)
    X = align_features(df_2017, feature_names)
    print(f'[INFO] CICIDS2017 binary: benign={int((y==0).sum()):,}, attack={int((y==1).sum()):,}')

    Xs = scaler.transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    results = {'dataset': 'CICIDS2017', 'mapping': 'binary (BENIGN=0, attack=1)',
               'n_samples': int(len(y)), 'models': {}}
    for mf in sorted(MODELS.glob('*.pkl')):
        if mf.stat().st_size == 0:
            print(f'--- {mf.stem}: empty, skip'); continue
        name = mf.stem
        try:
            obj = pickle.load(open(mf, 'rb'))
            net = (obj.model if hasattr(obj, 'model') else obj).cuda()
            t0 = time.time()
            m = evaluate(net, Xs, y)
            m['eval_time_sec'] = round(time.time() - t0, 2)
            results['models'][name] = m
            print(f"--- {name}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
                  f"f1m={m['f1_macro']:.4f} attack_recall={m['attack_recall']:.4f}")
            del net, obj; torch.cuda.empty_cache()
        except Exception as e:
            print(f'--- {name}: ERROR {e}')
            results['models'][name] = {'error': str(e)}

    json.dump(results, open(OUT / 'cross_dataset_binary.json', 'w'), indent=2)
    print(f"\n[DONE] {OUT / 'cross_dataset_binary.json'}")
    print('\nSUMMARY (sorted by balanced accuracy)')
    rows = [(n, m['balanced_accuracy'], m['accuracy'], m['f1_macro'], m['attack_recall'])
            for n, m in results['models'].items() if 'balanced_accuracy' in m]
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'Model':<18}{'Bal':>9}{'Acc':>9}{'F1m':>9}{'AttRec':>9}")
    for n, b, a, f, ar in rows:
        print(f"{n:<18}{b:>9.4f}{a:>9.4f}{f:>9.4f}{ar:>9.4f}")


if __name__ == '__main__':
    main()
