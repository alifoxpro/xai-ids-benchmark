# -*- coding: utf-8 -*-
"""
generate_xai_misclassified.py
=============================
The novel diagnostic figure: WHY minority attacks are missed.

Panel A: SHAP mean|value| (top features) for minority-attack flows that the model
         misclassifies as Benign — what the model attends to.
Panel B: distribution of the #1 SHAP feature (Packet Length Max) for
         Benign vs correctly-detected minority attacks vs missed (misclassified) minority
         attacks — showing the missed ones overlap the benign region.

Output: paper/figures/fig_xai_misclassified.png (+ paper_submission/fig copy)
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
MC = ROOT / 'results_2026_03_03' / 'multiclass'
NPZ = MC / 'stage1_cache' / 'arrays.npz'
META = json.load(open(MC / 'stage1_cache' / 'metadata.json'))
WRAP = MC / 'stage3_cache' / 'model_wrappers' / 'GRU.pkl'

FEATS = META['feature_columns']
PLMAX = FEATS.index('Packet Length Max')   # 38
MINORITY = {1: 'BruteForce', 6: 'Spoofing', 7: 'Web-Based'}  # rare attacks

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11})


def main():
    print('[INFO] loading data + model')
    d = np.load(NPZ, allow_pickle=True)
    Xte, yte = d['X_test'], d['y_test']
    # subsample for speed
    rng = np.random.default_rng(42)
    if len(Xte) > 400_000:
        idx = rng.choice(len(Xte), 400_000, replace=False)
        Xte, yte = Xte[idx], yte[idx]

    obj = pickle.load(open(WRAP, 'rb'))
    net = (obj.model if hasattr(obj, 'model') else obj).cuda().eval()
    # arrays.npz is already standardized; do NOT re-apply the scaler.
    Xs = Xte.astype(np.float32)

    # predict
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            logits = net(torch.as_tensor(Xs[i:i+8192], device='cuda'))
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]
            preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    is_minor = np.isin(yte, list(MINORITY))
    missed = is_minor & (preds == 0)            # minority predicted Benign (missed)
    caught = is_minor & (preds == yte)          # minority predicted correctly
    benign = (yte == 0)                         # all true benign (feature reference)
    benign_pp = (yte == 0) & (preds == 0)       # benign predicted benign (SHAP bg)
    print(f'[INFO] missed minority={missed.sum()}, caught minority={caught.sum()}, '
          f'benign(true)={benign.sum()}, benign_pp={benign_pp.sum()}')

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---------------- Panel A: SHAP on missed-minority samples ----------------
    try:
        import shap
        torch.backends.cudnn.enabled = False    # RNN backward works in eval mode without cudnn
        bg_pool = np.where(benign_pp)[0] if benign_pp.sum() >= 20 else np.where(benign)[0]
        bg_idx = rng.choice(bg_pool, min(100, len(bg_pool)), replace=False)
        ex_idx = rng.choice(np.where(missed)[0], min(200, missed.sum()), replace=False)
        background = torch.as_tensor(Xs[bg_idx], device='cuda')
        explain = torch.as_tensor(Xs[ex_idx], device='cuda')
        expl = shap.GradientExplainer(net, background)
        sv = expl.shap_values(explain)
        # sv: list per class OR array (n, f, c)
        if isinstance(sv, list):
            sv0 = np.asarray(sv[0])             # class 0 = Benign attribution
        else:
            sv0 = np.asarray(sv)[..., 0]
        mean_abs = np.abs(sv0).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:10][::-1]
        axA.barh([FEATS[i][:22] for i in order], mean_abs[order], color='#e34a33')
        axA.set_xlabel('mean |SHAP value| toward "Benign"')
        axA.set_title('(a) SHAP for missed minority attacks\n(features pushing them to "Benign")')
    except Exception as e:
        print(f'[WARN] SHAP panel failed ({e}); drawing fallback importance')
        axA.text(0.5, 0.5, f'SHAP unavailable:\n{e}', ha='center', va='center')

    # ---------------- Panel B: feature overlap (Packet Length Max) ------------
    def vals(mask):
        v = Xte[mask, PLMAX].astype(float)
        return v[np.isfinite(v)]
    vb, vc, vm = vals(benign), vals(caught), vals(missed)
    # robust clip for display
    hi = np.percentile(np.concatenate([vb, vc, vm]) if len(vc) else np.concatenate([vb, vm]), 99)
    bins = np.linspace(0, max(hi, 1), 60)
    axB.hist(vb, bins=bins, density=True, alpha=0.5, label='Benign', color='#888888')
    if len(vc):
        axB.hist(vc, bins=bins, density=True, alpha=0.5, label='Minority — detected', color='#2c7fb8')
    axB.hist(vm, bins=bins, density=True, alpha=0.5, label='Minority — MISSED', color='#e34a33')
    axB.set_xlabel('Packet Length Max (top SHAP feature)')
    axB.set_ylabel('Density')
    axB.set_title('(b) Missed minority attacks overlap the Benign\nfeature region')
    axB.legend()

    fig.tight_layout()
    for outdir in [ROOT / 'paper' / 'figures', ROOT / 'paper_submission' / 'fig']:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'fig_xai_misclassified.png', bbox_inches='tight')
        fig.savefig(outdir / 'fig_xai_misclassified.pdf', bbox_inches='tight')
    plt.close(fig)
    print('[DONE] fig_xai_misclassified written')


if __name__ == '__main__':
    main()
