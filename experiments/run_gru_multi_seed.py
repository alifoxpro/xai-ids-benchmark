"""
run_gru_multi_seed.py
=====================
DL multi-seed proof-of-concept: train GRU on the cached preprocessed data with
5 random seeds and report mean ± std for accuracy, balanced accuracy, F1.

Self-contained: does not depend on the project's stage3 module. Uses the same
cached arrays as the ML baseline script.

Expected wall time on RTX 5070 Laptop GPU: ~4-6 hours total (5 seeds × ~60 min).
With reduced subsample (300K), runs in ~90 minutes total.

Usage:
    python experiments/run_gru_multi_seed.py
    python experiments/run_gru_multi_seed.py --subsample 300000 --epochs 8
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score)


SEEDS = [17, 42, 123, 2024, 31337]
DATA_PATH = Path("results_2026_03_03/multiclass/stage1_cache/arrays.npz")
OUT_DIR = Path("results_2026_03_03/multiclass/q1_variance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- GRU model

class GRUClassifier(nn.Module):
    """3-layer GRU classifier matching the architecture in the paper."""

    def __init__(self, in_features: int = 81, hidden: int = 128,
                 num_layers: int = 3, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape (N, d) -> (N, 1, d) for the recurrent layer
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ------------------------------------------------------------- helpers

def set_all_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    return {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'balanced_accuracy': float(balanced_accuracy_score(all_labels, all_preds)),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro',
                                    zero_division=0)),
        'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted',
                                        zero_division=0)),
        'precision_weighted': float(precision_score(all_labels, all_preds,
                                                     average='weighted',
                                                     zero_division=0)),
    }


def train_one_seed(seed: int, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray, *,
                    epochs: int, batch_size: int, lr: float,
                    device: str) -> dict:
    set_all_seeds(seed)

    # Loaders
    X_tr_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_train, dtype=torch.long)
    X_te_t = torch.as_tensor(X_test, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_test, dtype=torch.long)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 4, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model & class weights
    num_classes = int(max(y_train.max(), y_test.max())) + 1
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_w = (class_counts.sum() / (num_classes * np.maximum(class_counts, 1)))
    class_w_t = torch.as_tensor(class_w, dtype=torch.float32, device=device)

    model = GRUClassifier(in_features=X_train.shape[1],
                           hidden=128, num_layers=3,
                           num_classes=num_classes, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=class_w_t)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    train_start = time.time()
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(X)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
        epoch_loss = total_loss / max(1, len(train_loader))
        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        print(f"    epoch {epoch+1:>2}/{epochs}  loss={epoch_loss:.4f}")

    train_time = time.time() - train_start

    # Inference
    inf_start = time.time()
    metrics = evaluate_model(model, test_loader, device)
    inf_time = (time.time() - inf_start) / X_test.shape[0] * 1000

    metrics.update({'train_time': train_time, 'inference_ms': inf_time,
                    'seed': seed})

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return metrics


def aggregate(per_seed: list[dict]) -> dict:
    keys = [k for k in per_seed[0].keys()
            if isinstance(per_seed[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n': len(vals),
        }
    return out


# ------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subsample', type=int, default=500_000,
                        help='Sample N train+test rows for tractable runtime')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] device = {device}")
    if device == 'cuda':
        print(f"[INFO] gpu = {torch.cuda.get_device_name(0)}")

    print(f"[INFO] Loading {DATA_PATH} ...")
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train_full, y_train_full = d['X_train'], d['y_train']
    X_test_full, y_test_full = d['X_test'], d['y_test']
    print(f"[INFO] full train = {X_train_full.shape}, test = {X_test_full.shape}")

    per_seed_results = []
    for seed in seeds:
        print(f"\n{'='*60}\n  GRU seed = {seed}\n{'='*60}")
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(X_train_full),
                            size=min(args.subsample, len(X_train_full)),
                            replace=False)
        te_idx = rng.choice(len(X_test_full),
                            size=min(args.subsample, len(X_test_full)),
                            replace=False)
        X_tr = X_train_full[tr_idx]
        y_tr = y_train_full[tr_idx]
        X_te = X_test_full[te_idx]
        y_te = y_test_full[te_idx]
        print(f"  sampled train={X_tr.shape}, test={X_te.shape}")

        result = train_one_seed(
            seed=seed,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=device)
        per_seed_results.append(result)
        print(f"  -> acc = {result['accuracy']:.4f}  "
              f"bal = {result['balanced_accuracy']:.4f}  "
              f"f1m = {result['f1_macro']:.4f}  "
              f"train = {result['train_time']:.1f}s")

    summary = {'per_seed': per_seed_results,
               'aggregate': aggregate(per_seed_results),
               'config': vars(args)}
    out_path = OUT_DIR / 'gru_seeds.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {out_path}")
    print(f"GRU multi-seed: mean acc = "
          f"{summary['aggregate']['accuracy']['mean']:.4f} ± "
          f"{summary['aggregate']['accuracy']['std']:.4f}")
    print(f"               mean bal = "
          f"{summary['aggregate']['balanced_accuracy']['mean']:.4f} ± "
          f"{summary['aggregate']['balanced_accuracy']['std']:.4f}")


if __name__ == '__main__':
    main()
