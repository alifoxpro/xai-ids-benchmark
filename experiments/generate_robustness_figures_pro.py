# -*- coding: utf-8 -*-
"""
generate_robustness_figures_pro.py
==================================
Publication-quality robustness figures from CORRECTED, consistent data.
Re-runs binary robustness (FGSM/PGD/noise) so the binary plots are correct, reuses
the corrected multi-class CSV, and renders professional academic figures
(consistent palette, baseline reference, annotations, grid, 300 dpi, PNG+PDF).

Regenerates: fig14_fgsm_attack, fig15_noise_attack, fig_pgd_attack,
fig_robustness_degradation, fig_fgsm_binary, fig_fgsm_binary_ieee,
fig_fgsm_comparison_ieee  (into paper/figures + paper_submission/fig).
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
OUTDIRS = [ROOT / 'paper' / 'figures', ROOT / 'paper_submission' / 'fig']
DEV = 'cuda'

# ---- professional, colour-blind-safe palette + consistent rcParams
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 12, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.35, 'grid.linestyle': '--',
    'axes.axisbelow': True, 'axes.linewidth': 0.9,
    'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.7',
})
C = {'PGD': '#d62728', 'FGSM': '#1f77b4', 'Noise': '#2ca02c', 'Drift': '#9467bd',
     'binary': '#ff7f0e', 'multi': '#1f77b4'}


def save(fig, name):
    for d in OUTDIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f'{name}.png', bbox_inches='tight')
        fig.savefig(d / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {name}')


# ---------------------------------------------------------------- attacks
def predict(net, X, bs=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = net(torch.as_tensor(X[i:i+bs], dtype=torch.float32, device=DEV))
            if isinstance(o, (tuple, list)):
                o = o[-1]
            out.append(o.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def acc(net, X, y):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y, predict(net, X)))


def fgsm(net, X, y, eps):
    lf = torch.nn.CrossEntropyLoss(); torch.backends.cudnn.enabled = False
    adv = []
    for i in range(0, len(X), 4096):
        xb = torch.as_tensor(X[i:i+4096], dtype=torch.float32, device=DEV).requires_grad_(True)
        yb = torch.as_tensor(y[i:i+4096], dtype=torch.long, device=DEV)
        o = net(xb); o = o[-1] if isinstance(o, (tuple, list)) else o
        g, = torch.autograd.grad(lf(o, yb), xb)
        adv.append((xb + eps * g.sign()).detach().cpu().numpy())
    return np.concatenate(adv)


def pgd(net, X, y, eps, steps=10):
    step = 2.5 * eps / steps; lf = torch.nn.CrossEntropyLoss()
    torch.backends.cudnn.enabled = False; adv = []
    for i in range(0, len(X), 4096):
        x0 = torch.as_tensor(X[i:i+4096], dtype=torch.float32, device=DEV)
        yb = torch.as_tensor(y[i:i+4096], dtype=torch.long, device=DEV)
        xb = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        for _ in range(steps):
            xb.requires_grad_(True)
            o = net(xb); o = o[-1] if isinstance(o, (tuple, list)) else o
            g, = torch.autograd.grad(lf(o, yb), xb)
            xb = (x0 + torch.clamp(xb.detach() + step * g.sign() - x0, -eps, eps))
        adv.append(xb.detach().cpu().numpy())
    return np.concatenate(adv)


def run_robustness(mode):
    cache = ROOT / 'results_2026_03_03' / mode / 'stage1_cache' / 'arrays.npz'
    mdl = ROOT / 'models' / mode / 'stage3' / 'GRU.pkl'
    d = np.load(cache, allow_pickle=True)
    Xte, yte = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xte), min(50_000, len(Xte)), replace=False)
    X, y = Xte[idx], yte[idx]
    obj = pickle.load(open(mdl, 'rb'))
    net = (obj.model if hasattr(obj, 'model') else obj).to(DEV).eval()
    base = acc(net, X, y)
    eps = [0.1, 0.3, 0.5]; sig = [0.05, 0.10, 0.15]
    res = {'baseline': base, 'eps': eps, 'sigma': sig,
           'FGSM': [acc(net, fgsm(net, X, y, e), y) for e in eps],
           'PGD':  [acc(net, pgd(net, X, y, e), y) for e in eps]}
    rng2 = np.random.default_rng(1)
    res['Noise'] = [acc(net, X + rng2.normal(0, s, X.shape).astype(np.float32), y) for s in sig]
    print(f'[{mode}] base={base:.3f} FGSM={res["FGSM"]} PGD={res["PGD"]} Noise={res["Noise"]}')
    return res


# ---------------------------------------------------------------- single-attack plot
def attack_plot(name, base, xs, ys, color, xlabel, title, ratio):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(base, ls='--', lw=1.6, color='0.4', label=f'Clean baseline ({base:.3f})')
    xx = [0] + list(xs); yy = [base] + list(ys)
    ax.plot(xx, yy, marker='o', ms=8, lw=2.4, color=color, label=f'Under attack (ratio {ratio:.2f})')
    for x, yv in zip(xs, ys):
        ax.annotate(f'{yv:.3f}', (x, yv), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=10)
    ax.fill_between(xx, yy, base, alpha=0.12, color=color)
    ax.set_xlabel(xlabel); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.02)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    return fig


def main():
    mc = run_robustness('multiclass')
    bn = run_robustness('binary')

    def ratio(base, ys):
        return float(np.mean([v / base for v in ys]))

    # F14 multiclass FGSM
    save(attack_plot('fgsm', mc['baseline'], mc['eps'], mc['FGSM'], C['FGSM'],
                     r'FGSM perturbation budget $\varepsilon$',
                     'FGSM Adversarial Robustness (GRU, multi-class)',
                     ratio(mc['baseline'], mc['FGSM'])), 'fig14_fgsm_attack')
    # F15 multiclass noise
    save(attack_plot('noise', mc['baseline'], mc['sigma'], mc['Noise'], C['Noise'],
                     r'Gaussian noise level $\sigma$',
                     'Gaussian-Noise Robustness (GRU, multi-class)',
                     ratio(mc['baseline'], mc['Noise'])), 'fig15_noise_attack')
    # PGD multiclass
    save(attack_plot('pgd', mc['baseline'], mc['eps'], mc['PGD'], C['PGD'],
                     r'PGD perturbation budget $\varepsilon$ (10 steps)',
                     'PGD Adversarial Robustness (GRU, multi-class)',
                     ratio(mc['baseline'], mc['PGD'])), 'fig_pgd_attack')
    # binary FGSM (two names, same pro figure)
    for nm in ['fig_fgsm_binary', 'fig_fgsm_binary_ieee']:
        save(attack_plot('fgsmb', bn['baseline'], bn['eps'], bn['FGSM'], C['binary'],
                         r'FGSM perturbation budget $\varepsilon$',
                         'FGSM Adversarial Robustness (GRU, binary)',
                         ratio(bn['baseline'], bn['FGSM'])), nm)

    # combined degradation (all attacks, multi-class)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(mc['baseline'], ls='--', lw=1.5, color='0.4',
               label=f'Clean baseline ({mc["baseline"]:.3f})')
    for att, xs, col, mk in [('PGD', mc['eps'], C['PGD'], 'o'),
                             ('FGSM', mc['eps'], C['FGSM'], 's'),
                             ('Noise', mc['sigma'], C['Noise'], '^')]:
        ax.plot([0] + list(xs), [mc['baseline']] + mc[att], marker=mk, ms=7, lw=2.2,
                color=col, label=f'{att} (ratio {ratio(mc["baseline"], mc[att]):.2f})')
    ax.set_xlabel(r'Perturbation strength ($\varepsilon$ / $\sigma$)')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.02)
    ax.set_title('Robustness Degradation under Adversarial and Noise Attacks\n'
                 '(GRU, multi-class)', fontweight='bold')
    ax.legend(loc='upper right', title='Attack (lower = more damaging)')
    save(fig, 'fig_robustness_degradation')

    # FGSM multi vs binary comparison
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([0] + mc['eps'], [mc['baseline']] + mc['FGSM'], marker='o', ms=8, lw=2.4,
            color=C['multi'], label=f'Multi-class (clean {mc["baseline"]:.3f})')
    ax.plot([0] + bn['eps'], [bn['baseline']] + bn['FGSM'], marker='s', ms=8, lw=2.4,
            color=C['binary'], label=f'Binary (clean {bn["baseline"]:.3f})')
    ax.set_xlabel(r'FGSM perturbation budget $\varepsilon$')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.02)
    ax.set_title('FGSM Robustness: Multi-class vs.\\ Binary (GRU)', fontweight='bold')
    ax.legend(loc='upper right')
    save(fig, 'fig_fgsm_comparison_ieee')

    print('\n[DONE] 7 professional robustness figures regenerated (PNG+PDF)')


if __name__ == '__main__':
    main()
