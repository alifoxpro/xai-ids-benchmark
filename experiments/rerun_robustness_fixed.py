# -*- coding: utf-8 -*-
"""
rerun_robustness_fixed.py
=========================
Fix the PGD under-powering bug (step_size was fixed at 0.01, so 10 steps could
not reach an epsilon ball larger than 0.1) and add a concept-drift sweep.
PGD now uses the standard step = 2.5*epsilon/n_steps, so PGD is at least as
strong as FGSM, as theory requires.

Model: GRU (multiclass). Test: 50K stratified subsample.
Output: results_2026_03_03/multiclass/stage8_robustness/GRU_pgd_fixed.csv,
        GRU_concept_drift.csv, robustness_scores_fixed.json
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
MC = ROOT / 'results_2026_03_03' / 'multiclass'
NPZ = MC / 'stage1_cache' / 'arrays.npz'
MODEL = ROOT / 'models' / 'multiclass' / 'stage3' / 'GRU.pkl'
OUT = MC / 'stage8_robustness'; OUT.mkdir(parents=True, exist_ok=True)
DEV = 'cuda'


def load():
    d = np.load(NPZ, allow_pickle=True)
    Xte, yte = d['X_test'], d['y_test'].astype(int)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xte), min(50_000, len(Xte)), replace=False)
    obj = pickle.load(open(MODEL, 'rb'))
    net = (obj.model if hasattr(obj, 'model') else obj).to(DEV).eval()
    # arrays.npz is already standardized; use directly (no re-scaling).
    Xs = Xte[idx].astype(np.float32)
    return net, Xs, yte[idx]


def predict(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            o = net(torch.as_tensor(X[i:i+16384], dtype=torch.float32, device=DEV))
            if isinstance(o, (tuple, list)):
                o = o[-1]
            out.append(o.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def acc(net, X, y):
    return float(accuracy_score(y, predict(net, X)))


def pgd_attack(net, X, y, epsilon, n_steps=10):
    """Standard PGD with step = 2.5*epsilon/n_steps (reaches the epsilon ball)."""
    step = 2.5 * epsilon / n_steps
    lossf = torch.nn.CrossEntropyLoss()
    advs = []
    torch.backends.cudnn.enabled = False
    for i in range(0, len(X), 4096):
        xb0 = torch.as_tensor(X[i:i+4096], dtype=torch.float32, device=DEV)
        yb = torch.as_tensor(y[i:i+4096], dtype=torch.long, device=DEV)
        xb = xb0.clone() + torch.empty_like(xb0).uniform_(-epsilon, epsilon)
        for _ in range(n_steps):
            xb.requires_grad_(True)
            out = net(xb)
            if isinstance(out, (tuple, list)):
                out = out[-1]
            loss = lossf(out, yb)
            grad, = torch.autograd.grad(loss, xb)
            xb = xb.detach() + step * grad.sign()
            xb = xb0 + torch.clamp(xb - xb0, -epsilon, epsilon)
        advs.append(xb.detach().cpu().numpy())
    return np.concatenate(advs)


def fgsm_attack(net, X, y, epsilon):
    lossf = torch.nn.CrossEntropyLoss()
    torch.backends.cudnn.enabled = False
    advs = []
    for i in range(0, len(X), 4096):
        xb = torch.as_tensor(X[i:i+4096], dtype=torch.float32, device=DEV)
        yb = torch.as_tensor(y[i:i+4096], dtype=torch.long, device=DEV)
        xb.requires_grad_(True)
        out = net(xb)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        loss = lossf(out, yb)
        grad, = torch.autograd.grad(loss, xb)
        advs.append((xb + epsilon * grad.sign()).detach().cpu().numpy())
    return np.concatenate(advs)


def main():
    net, X, y = load()
    base = acc(net, X, y)
    print(f'[INFO] clean accuracy = {base:.4f}')

    rows = []
    # ---- FGSM (same baseline/subsample for a consistent table)
    for eps in [0.1, 0.3, 0.5]:
        a = acc(net, fgsm_attack(net, X, y, eps), y)
        rows.append({'attack': 'FGSM', 'parameter': f'epsilon={eps}',
                     'baseline_accuracy': base, 'adversarial_accuracy': a,
                     'accuracy_drop': base - a, 'robustness_ratio': a / base})
        print(f'  FGSM eps={eps}: acc={a:.4f} (drop {base-a:.4f})')
    # ---- Gaussian noise
    rng_n = np.random.default_rng(1)
    for sigma in [0.05, 0.10, 0.15]:
        Xn = X + rng_n.normal(0, sigma, X.shape).astype(np.float32)
        a = acc(net, Xn, y)
        rows.append({'attack': 'Noise', 'parameter': f'sigma={sigma}',
                     'baseline_accuracy': base, 'adversarial_accuracy': a,
                     'accuracy_drop': base - a, 'robustness_ratio': a / base})
        print(f'  noise sigma={sigma}: acc={a:.4f} (drop {base-a:.4f})')
    # ---- PGD fixed (step = 2.5*eps/steps)
    pgd_rows = []
    for eps in [0.1, 0.3, 0.5]:
        a = acc(net, pgd_attack(net, X, y, eps), y)
        r = {'attack': 'PGD', 'parameter': f'epsilon={eps},steps=10',
             'baseline_accuracy': base, 'adversarial_accuracy': a,
             'accuracy_drop': base - a, 'robustness_ratio': a / base}
        rows.append(r); pgd_rows.append(r)
        print(f'  PGD eps={eps}: acc={a:.4f} (drop {base-a:.4f})')
    pd.DataFrame(rows).to_csv(OUT / 'GRU_robustness_consistent.csv', index=False)

    # ---- concept drift (gradual covariate shift: add scaled gaussian per step)
    drift_rows = []
    rng = np.random.default_rng(7)
    direction = rng.standard_normal(X.shape[1]).astype(np.float32)
    direction /= np.linalg.norm(direction)
    for mag in [0.1, 0.3, 0.5]:
        Xd = X + mag * direction[None, :]    # systematic distribution shift
        a = acc(net, Xd, y)
        f1 = float(f1_score(y, predict(net, Xd), average='macro', zero_division=0))
        drift_rows.append({'magnitude': mag, 'accuracy': a, 'f1_macro': f1,
                           'accuracy_drop': base - a})
        print(f'  drift mag={mag}: acc={a:.4f} f1m={f1:.4f}')
    pd.DataFrame(drift_rows).to_csv(OUT / 'GRU_concept_drift.csv', index=False)

    def mean_ratio(att):
        v = [r['robustness_ratio'] for r in rows if r['attack'] == att]
        return float(np.mean(v)) if v else None
    json.dump({'baseline': base,
               'fgsm_ratio': mean_ratio('FGSM'),
               'noise_ratio': mean_ratio('Noise'),
               'pgd_ratio': mean_ratio('PGD')},
              open(OUT / 'robustness_scores_fixed.json', 'w'), indent=2)
    print(f"\n[DONE] ratios — FGSM={mean_ratio('FGSM'):.3f} "
          f"Noise={mean_ratio('Noise'):.3f} PGD={mean_ratio('PGD'):.3f}")


if __name__ == '__main__':
    main()
