# -*- coding: utf-8 -*-
"""
run_3model_multi_seed.py
========================
Multi-seed training for 3 priority DL models:
  - FT-Transformer
  - ResNet1D
  - Transformer

5 seeds per model, multi-class CIC IoT-DIAD, 300K samples, 8 epochs.

Wall time on RTX 5070: ~1.5-2 hours total.
"""

import io, json, os, random, sys, time
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)

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


# --------------------------------------------------------- Models

class FTTransformer(nn.Module):
    """Lightweight FT-Transformer matching the paper architecture."""
    def __init__(self, in_features=81, dim=64, heads=8, depth=3,
                 num_classes=8, dropout=0.1):
        super().__init__()
        # Feature tokenizer: each feature -> a token via a 1x1 linear
        self.feature_tokenizer = nn.Linear(1, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, in_features + 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):                  # x: (B, d)
        b, d = x.shape
        tokens = self.feature_tokenizer(x.unsqueeze(-1))  # (B, d, dim)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_emb[:, :tokens.size(1)]
        out = self.encoder(tokens)
        return self.head(out[:, 0])  # cls token output


class ResNet1D(nn.Module):
    """Compact 1D-ResNet matching the paper architecture."""
    def __init__(self, in_features=81, filters=64, kernel=3, num_classes=8):
        super().__init__()
        self.conv0 = nn.Conv1d(1, filters, kernel, padding=kernel // 2)
        self.bn0 = nn.BatchNorm1d(filters)
        self.blocks = nn.ModuleList([
            self._block(filters, filters, kernel) for _ in range(4)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(filters, num_classes)

    @staticmethod
    def _block(in_c, out_c, k):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, padding=k // 2),
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, k, padding=k // 2),
            nn.BatchNorm1d(out_c))

    def forward(self, x):                  # x: (B, d)
        x = x.unsqueeze(1)                 # (B, 1, d)
        x = torch.relu(self.bn0(self.conv0(x)))
        for blk in self.blocks:
            res = x
            x = blk(x)
            x = torch.relu(x + res)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


class TransformerNet(nn.Module):
    """Standard Transformer encoder on tabular tokens."""
    def __init__(self, in_features=81, dim=32, heads=4, depth=2,
                 num_classes=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):                  # x: (B, d) -> (B, 1, dim)
        x = self.proj(x).unsqueeze(1)
        out = self.encoder(x)
        return self.head(out[:, 0])


MODEL_REGISTRY = {
    'FT_Transformer': lambda: FTTransformer(in_features=81, dim=64,
                                              heads=8, depth=3, num_classes=8),
    'ResNet1D':       lambda: ResNet1D(in_features=81, filters=64, num_classes=8),
    'Transformer':    lambda: TransformerNet(in_features=81, dim=32, heads=4,
                                              depth=2, num_classes=8),
}


# --------------------------------------------------------- training

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    preds, ys = np.asarray(preds), np.asarray(ys)
    return {
        'accuracy': float(accuracy_score(ys, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(ys, preds)),
        'f1_macro': float(f1_score(ys, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(ys, preds, average='weighted', zero_division=0)),
        'precision_weighted': float(precision_score(ys, preds, average='weighted', zero_division=0)),
    }


def train_one(model_name, seed, X_tr, y_tr, X_te, y_te,
                *, epochs=8, batch_size=256, lr=1e-3, device='cuda'):
    set_all_seeds(seed)
    model = MODEL_REGISTRY[model_name]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
    counts = np.bincount(y_tr, minlength=8)
    class_w = torch.as_tensor(counts.sum() / (8 * np.maximum(counts, 1)),
                                dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    scaler = torch.amp.GradScaler('cuda')

    train_ds = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),
                              torch.as_tensor(y_tr, dtype=torch.long))
    test_ds = TensorDataset(torch.as_tensor(X_te, dtype=torch.float32),
                             torch.as_tensor(y_te, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 4, shuffle=False,
                              pin_memory=True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train(); tot = 0.0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tot += float(loss.item())
        avg = tot / max(1, len(train_loader))
        scheduler.step(avg)
        print(f"    epoch {epoch+1:>2}/{epochs}  loss={avg:.4f}")
    train_time = time.time() - t0

    metrics = evaluate(model, test_loader, device)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    del model
    torch.cuda.empty_cache()
    return metrics


def aggregate(per_seed):
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] device={device}")
    print(f"[INFO] loading {DATA_PATH} ...")
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train_full, y_train_full = d['X_train'], d['y_train']
    X_test_full, y_test_full = d['X_test'], d['y_test']
    print(f"[INFO] full train={X_train_full.shape}, test={X_test_full.shape}")

    SUBSAMPLE = 300_000
    all_results = {}

    for model_name in ['FT_Transformer', 'ResNet1D', 'Transformer']:
        print(f"\n{'='*70}\n  Model: {model_name}\n{'='*70}")
        per_seed = []
        for seed in SEEDS:
            print(f"\n--- {model_name} seed={seed} ---")
            rng = np.random.default_rng(seed)
            tr_idx = rng.choice(len(X_train_full),
                                 size=min(SUBSAMPLE, len(X_train_full)),
                                 replace=False)
            te_idx = rng.choice(len(X_test_full),
                                 size=min(SUBSAMPLE, len(X_test_full)),
                                 replace=False)
            X_tr = X_train_full[tr_idx]; y_tr = y_train_full[tr_idx]
            X_te = X_test_full[te_idx];  y_te = y_test_full[te_idx]
            res = train_one(model_name, seed, X_tr, y_tr, X_te, y_te,
                              epochs=8, batch_size=256, lr=1e-3, device=device)
            per_seed.append(res)
            print(f"  -> acc={res['accuracy']:.4f}  "
                  f"bal={res['balanced_accuracy']:.4f}  "
                  f"f1m={res['f1_macro']:.4f}  "
                  f"t={res['train_time']:.1f}s")
        agg = aggregate(per_seed)
        all_results[model_name] = {'per_seed': per_seed, 'aggregate': agg}
        # incremental save
        with open(OUT_DIR / 'dl_3models_seeds.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[SAVE] partial -> {OUT_DIR/'dl_3models_seeds.json'}")
        print(f">>> {model_name}: acc={agg['accuracy']['mean']:.4f}"
              f" ± {agg['accuracy']['std']:.4f}")

    print("\n" + "="*70 + "\n  ALL DONE\n" + "="*70)
    for name, r in all_results.items():
        a = r['aggregate']
        print(f"  {name:<20}  acc={a['accuracy']['mean']:.4f}±{a['accuracy']['std']:.4f}  "
              f"bal={a['balanced_accuracy']['mean']:.4f}±{a['balanced_accuracy']['std']:.4f}")


if __name__ == '__main__':
    main()
