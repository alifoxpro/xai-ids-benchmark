# -*- coding: utf-8 -*-
"""
generate_all_ieee_figures.py
============================
Regenerate ALL paper figures with strict IEEEtran-compliant styling and
two fixed sizes:
    - IEEE_SINGLE = 3.5 x 2.6 in   (single column)
    - IEEE_DOUBLE = 7.16 x 4.5 in  (double column)

Settings (applied globally):
    DPI=600, Times New Roman 10pt, STIX math
    Title 11pt bold, labels 10pt bold
    Dashed grid 0.5pt alpha=0.25, ticks inward
    Legend framed alpha=0.9, partial spines (no top/right)
    Padding 0.05 in
"""

import io, json, os, sys
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
RES_DIR = PROJECT_ROOT / 'results_2026_03_03'
FIG_DIR.mkdir(parents=True, exist_ok=True)

IEEE_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.25,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',
    'legend.fancybox': False,
}
plt.rcParams.update(IEEE_RC)

IEEE_SINGLE = (3.5, 2.6)
IEEE_DOUBLE = (7.16, 4.5)

COLOR_BLUE = '#1f77b4'
COLOR_ORANGE = '#ff7f0e'
COLOR_GREEN = '#2ca02c'
COLOR_RED = '#d62728'
COLOR_PURPLE = '#9467bd'
COLOR_BROWN = '#8c564b'
COLOR_PINK = '#e377c2'
COLOR_GRAY = '#7f7f7f'
PALETTE = [COLOR_BLUE, COLOR_RED, COLOR_GREEN, COLOR_ORANGE,
           COLOR_PURPLE, COLOR_BROWN, COLOR_PINK, COLOR_GRAY]


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f'  {name}')


# ---------------------------------------------------- model comparison

def load_dl_results(mode):
    d = {}
    base = RES_DIR / mode / 'stage3_ml_models'
    for f in base.glob('*_results.json'):
        name = f.stem.replace('_results', '')
        with open(f) as fh:
            d[name] = json.load(fh)
    return d


def _get_test(r):
    """Models store metrics under r['test'] in this project's pipeline."""
    return r.get('test', r)


def fig_model_comparison(mode, fname, title):
    d = load_dl_results(mode)
    if not d:
        print(f'  [SKIP] no data for {mode}')
        return
    rows = []
    for name, r in d.items():
        t = _get_test(r)
        acc = t.get('accuracy', t.get('Accuracy', None))
        bal = t.get('balanced_accuracy', t.get('Balanced Accuracy', None))
        f1m = t.get('f1_macro', t.get('F1 Macro', None))
        if acc is None: continue
        rows.append((name.replace('_', '-'), acc, bal or 0, f1m or 0))
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    bal = [r[2] for r in rows]
    f1m = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, acc, w, label='Accuracy', color=COLOR_BLUE)
    ax.bar(x, bal, w, label='Balanced Acc.', color=COLOR_RED)
    ax.bar(x + w, f1m, w, label='F1 (macro)', color=COLOR_GREEN)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis='y')
    ax.legend(loc='lower right', ncol=3)
    save(fig, fname)


# ---------------------------------------------------- metric heatmap

def fig_metrics_heatmap(mode, fname, title):
    d = load_dl_results(mode)
    if not d: return
    metric_keys = ['accuracy', 'balanced_accuracy', 'f1_macro',
                    'f1', 'precision', 'recall', 'mcc', 'roc_auc']
    labels = ['Acc', 'Bal.Acc', 'F1$_\\text{m}$', 'F1$_\\text{w}$',
               'Prec', 'Rec', 'MCC', 'ROC-AUC']
    matrix = []
    rows = []
    for name, r in d.items():
        t = _get_test(r)
        vals = []
        for k in metric_keys:
            v = t.get(k)
            if v is None: v = t.get(k.title().replace('_', ' '))
            vals.append(v if isinstance(v, (int, float)) else np.nan)
        matrix.append(vals); rows.append(name.replace('_', '-'))
    M = np.array(matrix, dtype=float)
    # Sort by accuracy
    idx = np.argsort(M[:, 0])[::-1]
    M = M[idx]; rows = [rows[i] for i in idx]
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    im = ax.imshow(M, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                txt = '--'; color = 'k'
            else:
                txt = f'{v:.3f}'
                color = 'k' if 0.35 < v < 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    ax.tick_params(top=False, right=False)
    save(fig, fname)


# ---------------------------------------------------- per-class

def fig_per_class(mode, fname, title):
    """Per-class recall = 1 - FNR_per_class. Built from project's fnr_per_class arrays."""
    d = load_dl_results(mode)
    if not d: return
    # Class names from any model's confusion matrix order
    if mode == 'multiclass':
        class_names = ['Benign', 'BruteForce', 'DDOS', 'DOS',
                        'Mirai', 'Recon', 'Spoofing', 'Web-Based']
    else:
        class_names = ['Benign', 'Attack']
    matrix = []
    models = []
    for name, r in d.items():
        t = _get_test(r)
        fnr = t.get('fnr_per_class')
        if fnr is None: continue
        recall = [1 - x if isinstance(x, (int, float)) else np.nan
                   for x in fnr]
        # pad/truncate to expected class count
        recall = (recall + [np.nan] * len(class_names))[:len(class_names)]
        matrix.append(recall); models.append(name.replace('_', '-'))
    if not matrix: return
    M = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    im = ax.imshow(M, aspect='auto', cmap='YlOrRd_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                color = 'k' if v > 0.5 else 'white'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=7, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Recall')
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    ax.tick_params(top=False, right=False)
    save(fig, fname)


# ---------------------------------------------------- feature importance

def fig_feature_importance():
    d = {}
    for method in ['SHAP', 'mRMR', 'PERM', 'IG']:
        p = RES_DIR / 'multiclass' / 'stage2_feature_selection' / f'{method}_features.json'
        if p.exists():
            with open(p) as f:
                d[method] = json.load(f)
    if not d: return
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    plotted = 0
    for i, (method, payload) in enumerate(d.items()):
        if isinstance(payload, list):
            # list of [feature, score] pairs or dicts
            if len(payload) == 0: continue
            if isinstance(payload[0], dict):
                feats = [r.get('feature', r.get('name', str(idx)))
                          for idx, r in enumerate(payload)][:10]
                scores = [r.get('score', r.get('importance', 0))
                           for r in payload][:10]
            elif isinstance(payload[0], (list, tuple)):
                feats = [str(r[0]) for r in payload][:10]
                scores = [r[1] for r in payload][:10]
            else:
                continue
        else:
            feats = payload.get('top_features', payload.get('features', []))[:10]
            scores = payload.get('scores', payload.get('values', []))[:10]
        if not feats or not scores: continue
        s = np.array(scores, dtype=float)
        if s.max() > 0:
            s = s / s.max()
        x = np.arange(len(feats))
        ax.plot(x, s, marker='o', linewidth=1.0, markersize=4,
                 label=method, color=PALETTE[i])
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel('Top-10 feature rank')
    ax.set_ylabel('Normalised importance')
    ax.set_title('Feature Importance Across Selection Methods (Multi-class)')
    ax.grid(axis='y')
    ax.legend(loc='upper right')
    save(fig, 'fig_feature_importance_ieee.png')


# ---------------------------------------------------- XAI fidelity

def fig_xai_fidelity():
    rows = []
    for mode in ('multiclass', 'binary'):
        p = RES_DIR / mode / 'stage5_xai_quality' / 'xai_quality_summary.json'
        if not p.exists(): continue
        with open(p) as f:
            d = json.load(f)
        # Best-effort extract: fidelity for SHAP and LIME
        shap_fid = d.get('shap_fidelity') or d.get('SHAP', {}).get('fidelity')
        lime_fid = d.get('lime_fidelity') or d.get('LIME', {}).get('fidelity')
        shap_t = d.get('shap_time_ms') or d.get('SHAP', {}).get('time_ms')
        lime_t = d.get('lime_time_ms') or d.get('LIME', {}).get('time_ms')
        rows.append((mode, shap_fid, lime_fid, shap_t, lime_t))
    if not rows: return
    fig, (axF, axT) = plt.subplots(1, 2, figsize=IEEE_DOUBLE)
    modes = [r[0].capitalize() for r in rows]
    shap_fid = [r[1] for r in rows]; lime_fid = [r[2] for r in rows]
    shap_t   = [r[3] for r in rows]; lime_t   = [r[4] for r in rows]
    x = np.arange(len(modes)); w = 0.35
    # Fidelity
    if all(v is not None for v in shap_fid + lime_fid):
        axF.bar(x - w/2, shap_fid, w, label='SHAP', color=COLOR_BLUE)
        axF.bar(x + w/2, lime_fid, w, label='LIME', color=COLOR_ORANGE)
        axF.set_xticks(x); axF.set_xticklabels(modes)
        axF.set_ylabel('Fidelity'); axF.set_ylim(0, 1.05)
        axF.set_title('(a) Top-$k$ fidelity')
        axF.grid(axis='y'); axF.legend(loc='upper left')
    # Time
    if all(v is not None for v in shap_t + lime_t):
        axT.bar(x - w/2, shap_t, w, label='SHAP', color=COLOR_BLUE)
        axT.bar(x + w/2, lime_t, w, label='LIME', color=COLOR_ORANGE)
        axT.set_xticks(x); axT.set_xticklabels(modes)
        axT.set_ylabel('Explanation time (ms)')
        axT.set_yscale('log')
        axT.set_title('(b) Wall time per sample')
        axT.grid(axis='y'); axT.legend(loc='upper left')
    plt.tight_layout()
    save(fig, 'fig_xai_quality_ieee.png')


# ---------------------------------------------------- robustness (FGSM)

def fig_fgsm():
    for mode, fname in [('multiclass', 'fig_fgsm_multiclass_ieee.png'),
                          ('binary',     'fig_fgsm_binary_ieee.png')]:
        p = RES_DIR / mode / 'stage8_robustness' / 'GRU_fgsm.csv'
        if not p.exists(): continue
        df = pd.read_csv(p)
        eps_col = next((c for c in df.columns if 'eps' in c.lower() or 'epsilon' in c.lower()), df.columns[0])
        acc_col = next((c for c in df.columns if 'acc' in c.lower()), df.columns[-1])
        fig, ax = plt.subplots(figsize=IEEE_SINGLE)
        ax.plot(df[eps_col], df[acc_col], marker='o', linewidth=1.5,
                 color=COLOR_RED if mode == 'multiclass' else COLOR_BLUE)
        ax.set_xlabel(r'FGSM perturbation $\epsilon$')
        ax.set_ylabel('Accuracy retention')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'FGSM Robustness ({mode})')
        ax.grid(True)
        save(fig, fname)


# ---------------------------------------------------- feature perturbation

def fig_perturbation():
    p = RES_DIR / 'multiclass' / 'stage8_robustness' / 'GRU_feature_perturbation.csv'
    if not p.exists(): return
    df = pd.read_csv(p)
    k_col = df.columns[0]
    fig, ax = plt.subplots(figsize=IEEE_SINGLE)
    for i, c in enumerate(df.columns[1:]):
        ax.plot(df[k_col], df[c], marker='o', linewidth=1.2,
                 label=c, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel('Top-$k$ features zeroed')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Feature Perturbation Impact (GRU)')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save(fig, 'fig_feature_perturbation_ieee.png')


# ---------------------------------------------------- p-value heatmap

def fig_pvalue():
    p = RES_DIR / 'multiclass' / 'stage9_statistical' / 'wilcoxon_tests.csv'
    if not p.exists(): return
    df = pd.read_csv(p)
    # try to find Model_A, Model_B, p-value columns
    cols = {c.lower(): c for c in df.columns}
    a_col = cols.get('model_a') or cols.get('model1') or df.columns[0]
    b_col = cols.get('model_b') or cols.get('model2') or df.columns[1]
    p_col = next((c for c in df.columns if 'p' in c.lower() and 'val' in c.lower()), None)
    if p_col is None: return
    models = sorted(set(df[a_col]).union(df[b_col]))
    n = len(models); P = np.full((n, n), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    for _, row in df.iterrows():
        i = idx[row[a_col]]; j = idx[row[b_col]]
        P[i, j] = P[j, i] = row[p_col]
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    # log10 for visual
    L = np.log10(np.clip(P, 1e-20, 1))
    im = ax.imshow(L, cmap='viridis', aspect='auto')
    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r'$\log_{10}(p)$'); cb.ax.tick_params(labelsize=8)
    ax.set_title('Wilcoxon Pairwise $p$-values (Multi-class)')
    ax.tick_params(top=False, right=False)
    save(fig, 'fig_pvalue_heatmap_ieee.png')


# ---------------------------------------------------- main

def main():
    print('Comprehensive IEEE figure regeneration')
    print('=' * 60)

    fig_model_comparison('multiclass', 'fig_model_comparison_multi_ieee.png',
                          'Multi-class Model Comparison (Full Dataset)')
    fig_model_comparison('binary', 'fig_model_comparison_binary_ieee.png',
                          'Binary Model Comparison (Full Dataset)')
    fig_metrics_heatmap('multiclass', 'fig_metrics_heatmap_multi_ieee.png',
                         'Multi-class Metric Heatmap')
    fig_metrics_heatmap('binary', 'fig_metrics_heatmap_binary_ieee.png',
                         'Binary Metric Heatmap')
    fig_per_class('multiclass', 'fig_perclass_multi_ieee.png',
                   'Per-class Recall Across Models (Multi-class)')
    fig_feature_importance()
    fig_xai_fidelity()
    fig_fgsm()
    fig_perturbation()
    fig_pvalue()

    print('=' * 60)
    print('done.')


if __name__ == '__main__':
    main()
