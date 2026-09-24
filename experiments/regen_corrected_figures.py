# -*- coding: utf-8 -*-
"""
regen_corrected_figures.py
==========================
Regenerate the three figures that became inconsistent with the corrected tables:
  (1) robustness degradation curves  -> match Table 5 (corrected PGD/FGSM/noise)
  (2) per-class bars (FT-Transformer, FULL test) -> match Table 2
  (3) confusion matrix (FT-Transformer, FULL test) -> match Table 2
Writes into paper/figures/ AND paper_submission/fig/ with the existing filenames
referenced by main.tex (fig12_robustness_multiclass, fig19_perclass_multiclass,
fig7_cm_voting_multiclass) so no LaTeX edits are needed.
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
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
MC = ROOT / 'results_2026_03_03' / 'multiclass'
OUTDIRS = [ROOT / 'paper' / 'figures', ROOT / 'paper_submission' / 'fig']
CLASSES = list(json.load(open(MC / 'stage1_cache' / 'metadata.json'))['label_mapping'].keys())
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11})


def save(fig, name):
    for d in OUTDIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f'{name}.png', bbox_inches='tight')
        fig.savefig(d / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {name}')


# ---------------- (1) robustness curves ----------------
def robustness():
    df = pd.read_csv(MC / 'stage8_robustness' / 'GRU_robustness_consistent.csv')
    drift = pd.read_csv(MC / 'stage8_robustness' / 'GRU_concept_drift.csv')
    base = float(df['baseline_accuracy'].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for att, mark, col in [('PGD', 'o', '#d62728'), ('FGSM', 's', '#ff7f0e'),
                           ('Noise', '^', '#2ca02c')]:
        sub = df[df['attack'] == att]
        levels = [float(p.split('=')[1].split(',')[0]) for p in sub['parameter']]
        xs = [0] + levels
        ys = [base] + list(sub['adversarial_accuracy'])
        ax.plot(xs, ys, marker=mark, color=col, linewidth=2, label=att)
    ax.plot([0] + list(drift['magnitude']), [base] + list(drift['accuracy']),
            marker='d', color='#9467bd', linewidth=2, linestyle='--',
            label='Concept drift')
    ax.set_xlabel('Perturbation strength (ε / σ / drift magnitude)')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_title('Robustness degradation (GRU, multi-class)')
    ax.legend()
    save(fig, 'fig12_robustness_multiclass')


# ---------------- FT-Transformer full-test predictions ----------------
def ft_predictions():
    d = np.load(MC / 'stage1_cache' / 'arrays.npz', allow_pickle=True)
    Xte, yte = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    net = pickle.load(open(ROOT / 'models' / 'multiclass' / 'stage3' / 'FT_Transformer.pkl', 'rb'))
    net = (net.model if hasattr(net, 'model') else net).cuda().eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 16384):
            o = net(torch.as_tensor(Xte[i:i+16384], device='cuda'))
            if isinstance(o, (tuple, list)):
                o = o[-1]
            preds.append(o.argmax(-1).cpu().numpy())
    return yte, np.concatenate(preds)


def perclass(pc):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pc)); w = 0.27
    ax.bar(x - w, pc['precision'], w, label='Precision', color='#2c7fb8')
    ax.bar(x, pc['recall'], w, label='Recall', color='#e34a33')
    ax.bar(x + w, pc['f1'], w, label='F1', color='#fdae61')
    ax.set_xticks(x); ax.set_xticklabels(pc['class'], rotation=30, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis='y')
    ax.set_title('Per-class precision/recall/F1 — FT-Transformer (full test)')
    ax.legend()
    save(fig, 'fig19_perclass_multiclass')


def cmatrix(y, p):
    cm = confusion_matrix(y, p, labels=list(range(len(CLASSES))))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right'); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion matrix — FT-Transformer (full test, normalized)')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f'{cmn[i,j]:.2f}', ha='center', va='center',
                        color='white' if cmn[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, 'fig7_cm_voting_multiclass')


def main():
    robustness()
    y, p = ft_predictions()
    pc = pd.DataFrame(json.load(open(MC / 'fulltest' / 'ft_transformer_fulltest.json'))['per_class'])
    perclass(pc)
    cmatrix(y, p)
    print('\n[DONE] 3 figures regenerated to match corrected tables')


if __name__ == '__main__':
    main()
