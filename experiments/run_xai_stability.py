"""
run_xai_stability.py
====================
Compute the two stability metrics defined in the paper:

  * Lipschitz constant   L_phi  (Eq. 6 in the paper)
        L_phi(x) = sup_{x' in B_r(x)}  || phi(x) - phi(x') ||_2 / || x - x' ||_2

  * Top-k Jaccard       J(S1, S2) = |S1 ∩ S2| / |S1 ∪ S2|

Reports mean and bootstrap 95% confidence intervals for each metric, separately
for SHAP and LIME, across 30 multi-class test samples.

Usage
-----
python experiments/run_xai_stability.py --mode multiclass --model FT-Transformer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# SHAP / LIME
import shap
from lime.lime_tabular import LimeTabularExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import MODEL_REGISTRY
    from src.stage1_preprocess import load_preprocessed_data
    from src.stage3_dl_models import load_trained_model
except ImportError as exc:
    print(f"[WARN] Project modules not importable ({exc}).")
    MODEL_REGISTRY = ['GRU', 'FT-Transformer']
    load_preprocessed_data = None
    load_trained_model = None


# -------------------------------------------------------------------- helpers

def lipschitz_local(explain_fn, x: np.ndarray, *, radius: float = 0.05,
                    n_perturb: int = 20, rng: np.random.Generator) -> float:
    """Estimate the local Lipschitz constant of an explanation function at x.

    explain_fn : callable mapping (1, d) -> (d,) feature-attribution vector
    """
    phi_x = explain_fn(x.reshape(1, -1)).reshape(-1)
    ratios = []
    for _ in range(n_perturb):
        eta = rng.normal(scale=radius / np.sqrt(x.size), size=x.shape)
        x_prime = x + eta
        denom = np.linalg.norm(eta)
        if denom < 1e-9:
            continue
        phi_xp = explain_fn(x_prime.reshape(1, -1)).reshape(-1)
        num = np.linalg.norm(phi_x - phi_xp)
        ratios.append(num / denom)
    return float(np.max(ratios)) if ratios else float('nan')


def topk_jaccard(explain_fn, x: np.ndarray, *, k: int = 10,
                 n_runs: int = 2, rng: np.random.Generator) -> float:
    """Average Jaccard similarity between top-k features across n_runs."""
    runs = []
    for _ in range(n_runs):
        # Re-seed Sampling RNG if explainer uses internal randomness
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


def bootstrap_ci(values: list[float], confidence: float = 0.95,
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = [float(rng.choice(values, size=len(values), replace=True).mean())
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    return lo, hi


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['binary', 'multiclass'],
                        default='multiclass')
    parser.add_argument('--model', default='FT-Transformer',
                        choices=MODEL_REGISTRY)
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--radius', type=float, default=0.05)
    parser.add_argument('--n-perturb', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of independent explainer runs for Jaccard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                        default='results_2026_03_03')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if load_trained_model is None:
        print("[FATAL] Project modules unavailable; aborting.")
        sys.exit(1)

    # ---- load model, data, background
    data = load_preprocessed_data(mode=args.mode)
    model = load_trained_model(args.model, mode=args.mode)
    model.eval()

    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    background = X_test[rng.choice(X_test.shape[0], size=100, replace=False)]

    # ---- explainer wrappers
    def predict_fn(x_array: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tens = torch.as_tensor(x_array, dtype=torch.float32, device=next(model.parameters()).device)
            probs = torch.softmax(model(tens), dim=-1).cpu().numpy()
        return probs

    shap_explainer = shap.KernelExplainer(predict_fn, background)
    lime_explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=data.get('class_names'),
        mode='classification',
    )

    def shap_attr(x_arr: np.ndarray) -> np.ndarray:
        # SHAP returns list of arrays for multiclass; we use predicted class
        vals = shap_explainer.shap_values(x_arr, nsamples=100, silent=True)
        if isinstance(vals, list):
            cls = int(np.argmax(predict_fn(x_arr)))
            return np.asarray(vals[cls])
        return np.asarray(vals)

    def lime_attr(x_arr: np.ndarray) -> np.ndarray:
        out = np.zeros(x_arr.shape[-1])
        exp = lime_explainer.explain_instance(
            x_arr.reshape(-1), predict_fn, num_features=x_arr.shape[-1])
        cls = int(np.argmax(predict_fn(x_arr)))
        for feat_idx, weight in exp.local_exp[cls]:
            out[feat_idx] = weight
        return out

    # ---- sample test points
    sample_idx = rng.choice(X_test.shape[0], size=args.n_samples, replace=False)

    metrics = {'shap': {'lipschitz': [], 'jaccard': []},
               'lime': {'lipschitz': [], 'jaccard': []}}

    print(f"[INFO] Evaluating XAI stability over {args.n_samples} samples ...")
    for i, idx in enumerate(sample_idx):
        x = X_test[idx]
        t0 = time.time()

        metrics['shap']['lipschitz'].append(
            lipschitz_local(shap_attr, x, radius=args.radius,
                            n_perturb=args.n_perturb, rng=rng))
        metrics['shap']['jaccard'].append(
            topk_jaccard(shap_attr, x, k=args.top_k,
                         n_runs=args.n_runs, rng=rng))

        metrics['lime']['lipschitz'].append(
            lipschitz_local(lime_attr, x, radius=args.radius,
                            n_perturb=args.n_perturb, rng=rng))
        metrics['lime']['jaccard'].append(
            topk_jaccard(lime_attr, x, k=args.top_k,
                         n_runs=args.n_runs, rng=rng))

        print(f"  sample {i+1:>3}/{args.n_samples}  "
              f"SHAP L={metrics['shap']['lipschitz'][-1]:.3f} "
              f"J={metrics['shap']['jaccard'][-1]:.3f}  "
              f"LIME L={metrics['lime']['lipschitz'][-1]:.3f} "
              f"J={metrics['lime']['jaccard'][-1]:.3f}  "
              f"({time.time()-t0:.1f}s)")

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
                'std': float(np.std(vals, ddof=1)),
                'ci95_low': lo,
                'ci95_high': hi,
                'n': len(vals),
            }

    out_dir = Path(args.output_dir) / args.mode / 'xai_stability'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.model}_xai_stability.json'
    with open(out_path, 'w') as f:
        json.dump({'config': vars(args), 'raw': metrics, 'summary': summary},
                  f, indent=2)
    print(f"\n[DONE] Written {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
