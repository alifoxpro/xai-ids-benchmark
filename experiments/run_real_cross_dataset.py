# -*- coding: utf-8 -*-
"""
run_real_cross_dataset.py
=========================
Real cross-dataset evaluation:
  Train on CIC IoT-DIAD 2024 -> Test on CICIDS2017

This script:
  1. Loads CICIDS2017 from cic-ids-2017/ml-csv/ (8 CSVs, 2.8M rows)
  2. Aligns CICIDS2017 features (78) to CIC IoT-DIAD feature names (81)
  3. Applies the IoT-DIAD StandardScaler to the projected features
  4. Maps CICIDS2017 labels to IoT-DIAD label space
  5. Loads each trained DL model from models/multiclass/stage3/*.pkl
  6. Evaluates accuracy, balanced accuracy, F1, per-class recall on target

Output: results_2026_03_03/multiclass/cross_dataset/cross_dataset_real.json
"""

import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Force UTF-8 stdout to avoid cp1256 encoding errors on Arabic-locale Windows
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
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Feature-name mapping: CICIDS2017 column name -> CIC IoT-DIAD feature name
# Built by manual inspection of both datasets' CICFlowMeter outputs.
FEATURE_MAP_2017_TO_IOTDIAD = {
    'Flow Duration': 'Flow Duration',
    'Total Fwd Packets': 'Total Fwd Packet',
    'Total Backward Packets': 'Total Bwd packets',
    'Total Length of Fwd Packets': 'Total Length of Fwd Packet',
    'Total Length of Bwd Packets': 'Total Length of Bwd Packet',
    'Fwd Packet Length Max': 'Fwd Packet Length Max',
    'Fwd Packet Length Min': 'Fwd Packet Length Min',
    'Fwd Packet Length Mean': 'Fwd Packet Length Mean',
    'Fwd Packet Length Std': 'Fwd Packet Length Std',
    'Bwd Packet Length Max': 'Bwd Packet Length Max',
    'Bwd Packet Length Min': 'Bwd Packet Length Min',
    'Bwd Packet Length Mean': 'Bwd Packet Length Mean',
    'Bwd Packet Length Std': 'Bwd Packet Length Std',
    'Flow Bytes/s': 'Flow Bytes/s',
    'Flow Packets/s': 'Flow Packets/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Total': 'Fwd IAT Total',
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Total': 'Bwd IAT Total',
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': 'Bwd PSH Flags',
    'Fwd URG Flags': 'Fwd URG Flags',
    'Bwd URG Flags': 'Bwd URG Flags',
    'Fwd Header Length': 'Fwd Header Length',
    'Bwd Header Length': 'Bwd Header Length',
    'Fwd Packets/s': 'Fwd Packets/s',
    'Bwd Packets/s': 'Bwd Packets/s',
    'Min Packet Length': 'Packet Length Min',
    'Max Packet Length': 'Packet Length Max',
    'Packet Length Mean': 'Packet Length Mean',
    'Packet Length Std': 'Packet Length Std',
    'Packet Length Variance': 'Packet Length Variance',
    'FIN Flag Count': 'FIN Flag Count',
    'SYN Flag Count': 'SYN Flag Count',
    'RST Flag Count': 'RST Flag Count',
    'PSH Flag Count': 'PSH Flag Count',
    'ACK Flag Count': 'ACK Flag Count',
    'URG Flag Count': 'URG Flag Count',
    'CWE Flag Count': 'CWR Flag Count',  # CWE in 2017 = CWR in IoT-DIAD
    'ECE Flag Count': 'ECE Flag Count',
    'Down/Up Ratio': 'Down/Up Ratio',
    'Average Packet Size': 'Average Packet Size',
    'Avg Fwd Segment Size': 'Fwd Segment Size Avg',
    'Avg Bwd Segment Size': 'Bwd Segment Size Avg',
    'Fwd Avg Bytes/Bulk': 'Fwd Bytes/Bulk Avg',
    'Fwd Avg Packets/Bulk': 'Fwd Packet/Bulk Avg',
    'Fwd Avg Bulk Rate': 'Fwd Bulk Rate Avg',
    'Bwd Avg Bytes/Bulk': 'Bwd Bytes/Bulk Avg',
    'Bwd Avg Packets/Bulk': 'Bwd Packet/Bulk Avg',
    'Bwd Avg Bulk Rate': 'Bwd Bulk Rate Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Packets',
    'Subflow Fwd Bytes': 'Subflow Fwd Bytes',
    'Subflow Bwd Packets': 'Subflow Bwd Packets',
    'Subflow Bwd Bytes': 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward': 'FWD Init Win Bytes',
    'Init_Win_bytes_backward': 'Bwd Init Win Bytes',
    'act_data_pkt_fwd': 'Fwd Act Data Pkts',
    'min_seg_size_forward': 'Fwd Seg Size Min',
    'Active Mean': 'Active Mean',
    'Active Std': 'Active Std',
    'Active Max': 'Active Max',
    'Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': 'Idle Std',
    'Idle Max': 'Idle Max',
    'Idle Min': 'Idle Min',
}


# CICIDS2017 label -> CIC IoT-DIAD label index
# IoT-DIAD label_mapping: {'Benign': 0, 'BruteForce': 1, 'DDOS': 2, 'DOS': 3,
#                         'Mirai': 4, 'Recon': 5, 'Spoofing': 6, 'Web-Based': 7}
LABEL_MAP = {
    'BENIGN': 0,                          # Benign
    'FTP-Patator': 1, 'SSH-Patator': 1,   # BruteForce
    'DDoS': 2,                            # DDOS
    'DoS GoldenEye': 3, 'DoS Hulk': 3,
    'DoS Slowhttptest': 3, 'DoS slowloris': 3,
    'Heartbleed': 3,                      # DOS
    'Bot': 4,                             # Mirai (closest analogue: both botnets)
    'PortScan': 5,                        # Recon
    'Infiltration': 6,                    # Spoofing (closest analogue)
    'Web Attack \x96 Brute Force': 7,
    'Web Attack \x96 XSS': 7,
    'Web Attack \x96 Sql Injection': 7,    # Web-Based
}

IOTDIAD_LABELS = {0: 'Benign', 1: 'BruteForce', 2: 'DDOS', 3: 'DOS',
                  4: 'Mirai', 5: 'Recon', 6: 'Spoofing', 7: 'Web-Based'}


# -------------------------------------------------------------- loaders

def load_cicids2017() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all CICIDS2017 CSVs and return (raw_features_df, labels_str, columns)."""
    print(f"[INFO] Loading CICIDS2017 from {CICIDS_DIR}")
    frames = []
    for csv in sorted(CICIDS_DIR.glob('*.csv')):
        print(f"  -> {csv.name}")
        df = pd.read_csv(csv, low_memory=False, encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]
        # Identify label column (the last column, named " Label" or "Label")
        label_col = next((c for c in df.columns if c.lower() == 'label'), None)
        if label_col is None:
            print(f"    [WARN] no label column, skipping")
            continue
        # Drop Destination Port (not in IoT-DIAD) and any header rows
        if 'Destination Port' in df.columns:
            df = df.drop(columns=['Destination Port'])
        # Remove rows where label is non-string (artifact rows)
        df = df[df[label_col].apply(lambda x: isinstance(x, str))]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Concatenated: {full.shape}")

    # Strip whitespace from string labels
    full[label_col] = full[label_col].str.strip()

    # Replace inf/-inf with NaN, then drop NaN rows
    feature_cols = [c for c in full.columns if c != label_col]
    full[feature_cols] = full[feature_cols].apply(pd.to_numeric, errors='coerce')
    full = full.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[INFO] After NaN/inf removal: {full.shape}")

    return full[feature_cols].copy(), full[label_col].values, feature_cols


def align_features(df_2017: pd.DataFrame,
                    iotdiad_features: list[str]) -> np.ndarray:
    """Reorder + rename CICIDS2017 columns to match IoT-DIAD 81-feature layout.

    Features not present in CICIDS2017 (the 5 derived feat_* features and
    duplicated Fwd Header Length) are filled with zeros.
    """
    # Apply mapping
    df = df_2017.rename(columns=FEATURE_MAP_2017_TO_IOTDIAD)

    # Output matrix: (N, 81) initialised to 0 to handle missing features
    n_rows = len(df)
    X = np.zeros((n_rows, len(iotdiad_features)), dtype=np.float32)
    matched = 0
    for j, feat in enumerate(iotdiad_features):
        if feat in df.columns:
            # Some columns may appear twice (e.g. duplicated Fwd Header Length) -- take first
            col = df[feat]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            X[:, j] = col.to_numpy(dtype=np.float32)
            matched += 1
    print(f"[INFO] Matched {matched}/{len(iotdiad_features)} features")
    return X


def map_labels(label_strs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map CICIDS2017 string labels to IoT-DIAD int labels. Return (y, mask)."""
    y = np.full(len(label_strs), -1, dtype=np.int64)
    for i, s in enumerate(label_strs):
        if s in LABEL_MAP:
            y[i] = LABEL_MAP[s]
    mask = y != -1
    return y, mask


# -------------------------------------------------------------- evaluator

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                    *, batch: int = 2048) -> dict:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch):
        chunk = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        try:
            logits = model(chunk)
        except RuntimeError:
            # BatchNorm needs batch > 1; pad if necessary
            if chunk.shape[0] == 1:
                chunk = chunk.repeat(2, 1)
                logits = model(chunk)[:1]
            else:
                raise
        # Some models return a tuple (e.g. AE-Classifier -> (reconstruction, logits));
        # the classification logits are the last element.
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)

    out = {
        'accuracy': float(accuracy_score(y, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, preds, average='weighted', zero_division=0)),
    }

    # per-class recall
    per_class = classification_report(y, preds, output_dict=True,
                                       zero_division=0)
    out['per_class'] = {}
    for cls_idx, name in IOTDIAD_LABELS.items():
        key = str(cls_idx)
        if key in per_class:
            out['per_class'][name] = {
                'precision': per_class[key]['precision'],
                'recall': per_class[key]['recall'],
                'f1-score': per_class[key]['f1-score'],
                'support': per_class[key]['support'],
            }
    return out


# -------------------------------------------------------------- main

def main():
    print("=" * 70)
    print("REAL CROSS-DATASET EVALUATION: CIC IoT-DIAD 2024 -> CICIDS2017")
    print("=" * 70)

    # ---- IoT-DIAD metadata + scaler
    with open(IOTDIAD_ARTIFACTS / 'metadata.json') as f:
        meta = json.load(f)
    feature_names = meta['feature_columns']
    label_mapping = meta['label_mapping']
    print(f"[INFO] IoT-DIAD features = {len(feature_names)}")
    print(f"[INFO] IoT-DIAD labels = {label_mapping}")

    with open(IOTDIAD_ARTIFACTS / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # ---- Load CICIDS2017
    t0 = time.time()
    df_2017, label_strs, csv_cols = load_cicids2017()
    print(f"[INFO] Load time: {time.time()-t0:.1f}s")

    # Show label distribution
    from collections import Counter
    print("\n[INFO] CICIDS2017 label distribution:")
    for lbl, count in Counter(label_strs).most_common():
        # Replace any non-ASCII characters for safe printing
        safe_lbl = lbl.encode('ascii', 'replace').decode('ascii')
        mapped = LABEL_MAP.get(lbl, 'UNMAPPED')
        if mapped != 'UNMAPPED':
            mapped_name = IOTDIAD_LABELS[mapped]
            print(f"  {safe_lbl:>40} -> {mapped_name:<12} ({count:>8,})")
        else:
            print(f"  {safe_lbl:>40} -> UNMAPPED      ({count:>8,})")

    # ---- Feature alignment
    print("\n[INFO] Aligning features...")
    X_2017_aligned = align_features(df_2017, feature_names)
    print(f"  aligned shape = {X_2017_aligned.shape}")

    # ---- Label mapping & subset
    y_2017, mask = map_labels(label_strs)
    X_2017_aligned = X_2017_aligned[mask]
    y_2017 = y_2017[mask]
    print(f"[INFO] After label filter: X={X_2017_aligned.shape}, y={y_2017.shape}")

    # ---- Apply IoT-DIAD scaler
    print("[INFO] Applying IoT-DIAD scaler...")
    X_2017_scaled = scaler.transform(X_2017_aligned)
    X_2017_scaled = np.nan_to_num(X_2017_scaled, nan=0.0,
                                    posinf=10.0, neginf=-10.0).astype(np.float32)

    # ---- Evaluate trained models
    results = {'dataset': 'CICIDS2017',
               'n_samples': int(len(y_2017)),
               'n_features_matched': int(len([f for f in feature_names
                                              if f in FEATURE_MAP_2017_TO_IOTDIAD.values()])),
               'models': {}}

    model_files = sorted(MODELS_DIR.glob('*.pkl'))
    for mf in model_files:
        if 'ensemble' in mf.name.lower() or 'voting' in mf.name.lower():
            continue   # voting ensemble is meta-model
        name = mf.stem
        print(f"\n--- {name} ---")
        try:
            with open(mf, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, 'cuda'):
                model = model.cuda()
            t0 = time.time()
            metrics = evaluate_model(model, X_2017_scaled, y_2017)
            metrics['eval_time_sec'] = round(time.time() - t0, 2)
            results['models'][name] = metrics
            print(f"  acc = {metrics['accuracy']:.4f}  "
                  f"bal = {metrics['balanced_accuracy']:.4f}  "
                  f"f1m = {metrics['f1_macro']:.4f}  "
                  f"({metrics['eval_time_sec']}s)")
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results['models'][name] = {'error': str(exc)}

    out_path = OUT_DIR / 'cross_dataset_real.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (CICIDS2017 target, sorted by balanced accuracy)")
    print("=" * 70)
    rows = [(name, m['balanced_accuracy'], m['accuracy'], m['f1_macro'])
            for name, m in results['models'].items()
            if 'balanced_accuracy' in m]
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'Model':<20} {'Bal.Acc':>10} {'Acc':>10} {'F1m':>10}")
    for name, bal, acc, f1m in rows:
        print(f"{name:<20} {bal:>10.4f} {acc:>10.4f} {f1m:>10.4f}")


if __name__ == '__main__':
    main()
