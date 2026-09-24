# -*- coding: utf-8 -*-
"""
run_xai_stability_real.py
=========================
Real XAI stability evaluation on the trained models.

Computes two metrics per (model, method) cell:
  * Lipschitz constant L_phi  (Eq. 6 in the paper)
  * Top-k Jaccard stability J (Eq. 7 in the paper)

For: SHAP (KernelExplainer) and LIME on:
  - FT_Transformer (multi-class)
  - GRU             (binary)

Total wall time on RTX 5070: ~1.5-2 hours.
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
import torch

import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'results_2026_03_03' / 'xai_stability'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reduced sample counts for tractable runtime
N_SAMPLES = 15           # paper says 30; halved for time budget
N_PERTURB = 10           # paper says 20; halved
N_RUNS_FOR_JACCARD = 3   # paper says 5; reduced
TOP_K = 10
RADIUS = 0.05


# --------------------------------------------------------- utilities

def lipschitz_local(explain_fn, x: np.ndarray, *, radius=RADIUS,
                     n_perturb=N_PERTURB, rng) -> float:
    phi_x = explain_fn(x.reshape(1, -1)).reshape(-1)
    ratios = []
    for _ in range(n_perturb):
        eta = rng.normal(scale=radius / np.sqrt(x.size), size=x.shape).astype(np.float32)
        x_prime = (x + eta).astype(np.float32)
        denom = float(np.linalg.norm(eta))
        if denom < 1e-9:
            continue
        phi_xp = explain_fn(x_prime.reshape(1, -1)).reshape(-1)
        num = float(np.linalg.norm(phi_x - phi_xp))
        ratios.append(num / denom)
    return float(np.max(ratios)) if ratios else float('nan')


def topk_jaccard(explain_fn, x: np.ndarray, *, k=TOP_K,
                  n_runs=N_RUNS_FOR_JACCARD) -> float:
    runs = []
    for _ in range(n_runs):
        phi = explain_fn(x.reshape(1, -1)).reshape(-1)
        topk = set(np.argsort(np.abs(phi))[-k:].tolist())
        runs.append(topk)
    intersections, unions = [], []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            inter = len(runs[i] & runs[j])
            uni = len(runs[i] | runs[j])
            if uni > 0:
                intersections.append(inter)
                unions.append(uni)
    if not intersections:
        return float('nan')
    return float(np.mean([i / u for i, u in zip(intersections, unions)]))


def bootstrap_ci(values, confidence=0.95, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if len(arr) < 2:
        return None, None
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean())
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    return lo, hi


# --------------------------------------------------------- core runner

def evaluate_one(model_name: str, mode: str) -> dict:
    print(f"\n{'='*60}\n  {model_name} ({mode})\n{'='*60}")

    # ---- load model
    model_path = PROJECT_ROOT / 'models' / mode / 'stage3' / f'{model_name}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model = model.cuda().eval()

    # ---- load test data
    data_path = PROJECT_ROOT / 'results_2026_03_03' / mode / 'stage1_cache' / 'arrays.npz'
    d = np.load(data_path, allow_pickle=True)
    X_test = d['X_test'].astype(np.float32)
    y_test = d['y_test']

    rng = np.random.default_rng(42)
    # Background for SHAP (small for KernelExplainer)
    bg_idx = rng.choice(X_test.shape[0], size=50, replace=False)
    background = X_test[bg_idx]
    # Sample test points
    sample_idx = rng.choice(X_test.shape[0], size=N_SAMPLES, replace=False)

    # ---- prediction wrapper
    def predict_fn(x_array: np.ndarray) -> np.ndarray:
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        with torch.no_grad():
            tens = torch.as_tensor(x_array, dtype=torch.float32, device='cuda')
            if tens.shape[0] == 1:
                tens = tens.repeat(2, 1)
                out = model(tens)
                if isinstance(out, tuple):
                    out = out[1]
                probs = torch.softmax(out, dim=-1).cpu().numpy()[:1]
            else:
                out = model(tens)
                if isinstance(out, tuple):
                    out = out[1]
                probs = torch.softmax(out, dim=-1).cpu().numpy()
        return probs

    n_classes = predict_fn(background[:2]).shape[1]
    print(f"  n_classes = {n_classes}")

    # ---- SHAP and LIME explainers
    print(f"  building SHAP KernelExplainer (background={background.shape[0]}) ...")
    shap_exp = shap.KernelExplainer(predict_fn, background, silent=True)

    print(f"  building LIME explainer ...")
    lime_exp = LimeTabularExplainer(
        training_data=background, mode='classification',
        feature_names=[f'f{i}' for i in range(background.shape[1])])

    def shap_attr(x_arr):
        out = shap_exp.shap_values(x_arr, nsamples=64, silent=True)
        if isinstance(out, list):
            cls = int(np.argmax(predict_fn(x_arr)))
            return np.asarray(out[cls]).reshape(-1)
        return np.asarray(out).reshape(-1)

    def lime_attr(x_arr):
        exp = lime_exp.explain_instance(
            x_arr.reshape(-1), predict_fn, num_features=x_arr.shape[-1],
            num_samples=200)
        cls = int(np.argmax(predict_fn(x_arr)))
        out = np.zeros(x_arr.shape[-1])
        for feat_idx, weight in exp.local_exp.get(cls, []):
            out[feat_idx] = weight
        return out

    metrics = {'shap': {'lipschitz': [], 'jaccard': []},
               'lime': {'lipschitz': [], 'jaccard': []}}

    for i, idx in enumerate(sample_idx):
        x = X_test[idx].astype(np.float32)
        t0 = time.time()

        try:
            metrics['shap']['lipschitz'].append(
                lipschitz_local(shap_attr, x, rng=rng))
            metrics['shap']['jaccard'].append(topk_jaccard(shap_attr, x))
        except Exception as e:
            print(f"    SHAP error: {e}")
            metrics['shap']['lipschitz'].append(float('nan'))
            metrics['shap']['jaccard'].append(float('nan'))

        try:
            metrics['lime']['lipschitz'].append(
                lipschitz_local(lime_attr, x, rng=rng))
            metrics['lime']['jaccard'].append(topk_jaccard(lime_attr, x))
        except Exception as e:
            print(f"    LIME error: {e}")
            metrics['lime']['lipschitz'].append(float('nan'))
            metrics['lime']['jaccard'].append(float('nan'))

        elapsed = time.time() - t0
        print(f"  {i+1:>3}/{N_SAMPLES}  "
              f"SHAP L={metrics['shap']['lipschitz'][-1]:.3f} J={metrics['shap']['jaccard'][-1]:.3f}  "
              f"LIME L={metrics['lime']['lipschitz'][-1]:.3f} J={metrics['lime']['jaccard'][-1]:.3f}  "
              f"({elapsed:.1f}s)")

    # ---- aggregate
    summary = {}
    for method in ('shap', 'lime'):
        summary[method] = {}
        for metric in ('lipschitz', 'jaccard'):
            vals = [v for v in metrics[method][metric] if not np.isnan(v)]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals)
            summary[method][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'ci95_low': lo, 'ci95_high': hi,
                'n': len(vals),
            }

    del model
    torch.cuda.empty_cache()

    return {'config': {'n_samples': N_SAMPLES, 'n_perturb': N_PERTURB,
                       'n_runs_jaccard': N_RUNS_FOR_JACCARD,
                       'top_k': TOP_K, 'radius': RADIUS},
            'raw': metrics, 'summary': summary}


def main():
    print("XAI STABILITY EVALUATION")
    print("=" * 60)
    print(f"N_SAMPLES={N_SAMPLES}, N_PERTURB={N_PERTURB}, "
          f"N_RUNS_J={N_RUNS_FOR_JACCARD}, TOP_K={TOP_K}, RADIUS={RADIUS}")

    results = {}
    for model_name, mode in [('FT_Transformer', 'multiclass'),
                              ('GRU', 'binary')]:
        results[f'{model_name}_{mode}'] = evaluate_one(model_name, mode)
        # Save incrementally so partial progress is preserved
        with open(OUT_DIR / 'xai_stability_real.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  -> partial save to {OUT_DIR / 'xai_stability_real.json'}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        print(f"\n{key}:")
        for method in ('shap', 'lime'):
            s = res['summary'].get(method, {})
            if 'lipschitz' in s and 'jaccard' in s:
                print(f"  {method:<5} L = {s['lipschitz']['mean']:.3f} "
                      f"[{s['lipschitz']['ci95_low']:.3f}, {s['lipschitz']['ci95_high']:.3f}]"
                      f"  J = {s['jaccard']['mean']:.3f} "
                      f"[{s['jaccard']['ci95_low']:.3f}, {s['jaccard']['ci95_high']:.3f}]")


if __name__ == '__main__':
    main()
