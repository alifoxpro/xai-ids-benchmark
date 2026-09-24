# -*- coding: utf-8 -*-
"""
generate_architecture_diagrams.py
=================================
Publication-quality architecture schematics for the ten evaluated models, drawn as
a 2x5 grid of vertical layer stacks with a shared colour legend. Replaces the
text-only architecture appendix (which overflowed the column) with a clean figure.

Output: paper/figures/fig_architectures.png/.pdf (+ paper_submission/fig)
"""
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUTDIRS = [ROOT / 'paper' / 'figures', ROOT / 'paper_submission' / 'fig']

plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

# layer category -> colour (muted academic palette)
PAL = {
    'io':    '#E8E8E8',   # input/output
    'lin':   '#4C72B0',   # linear / embedding
    'conv':  '#55A868',   # convolution
    'rec':   '#DD8452',   # recurrent (GRU/LSTM)
    'attn':  '#8172B3',   # attention / transformer
    'norm':  '#C7C7C7',   # norm / pool
    'head':  '#C44E52',   # classification head
    'spec':  '#4FA1A1',   # special block (tokenizer, GLU, AE, voting)
}
LIGHT = {'io', 'norm'}   # categories needing dark text

# each model: list of (label, category), bottom(input) -> top(output)
MODELS = {
 'GRU': [('Input (78)', 'io'), ('Embed FC+BN', 'lin'), ('Pseudo-seq (8)', 'norm'),
         ('Bi-GRU x2 (128)', 'rec'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'ResNet1D': [('Input (78)', 'io'), ('Embed FC+BN', 'lin'), ('ResBlock x3 (128)', 'conv'),
              ('Global Avg-Pool', 'norm'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'FT-Transformer': [('Input (78)', 'io'), ('Feature tokenizer', 'spec'), ('+ [CLS] token', 'norm'),
                    ('Encoder x3 (d=64)', 'attn'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'CNN-LSTM': [('Input (78)', 'io'), ('Conv1d x2 (64,128)', 'conv'), ('LSTM (128)', 'rec'),
              ('LSTM (64)', 'rec'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'BiLSTM-Attn': [('Input (78)', 'io'), ('Embed FC+BN', 'lin'), ('Bi-LSTM x2 (128)', 'rec'),
                 ('Self-attention', 'attn'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'TabNet': [('Input (78)', 'io'), ('Feature BN', 'norm'), ('Attentive step x3', 'spec'),
            ('GLU transform', 'conv'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'TCN': [('Input (78)', 'io'), ('Pseudo-seq', 'norm'), ('Dilated Conv x3', 'conv'),
         ('(dil. 1/2/4)', 'conv'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'Transformer': [('Input (78)', 'io'), ('Embed FC (256)', 'lin'), ('Reshape (16,16)', 'norm'),
                 ('Encoder x2 (d=16)', 'attn'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'AE-Classifier': [('Input (78)', 'io'), ('Encoder 64-32-16', 'lin'), ('Latent (16)', 'spec'),
                   ('Decoder (recon.)', 'lin'), ('FC head', 'head'), ('Logits (C)', 'io')],
 'VotingEnsemble': [('GRU | ResNet1D', 'io'), ('| FT-Transformer', 'io'), ('Soft voting', 'spec'),
                    ('(mean softmax)', 'spec'), ('Logits (C)', 'io')],
}

NROW, NCOL = 2, 5
BW, BH, VG = 2.6, 0.62, 0.42       # box width/height, vertical gap
CW, CH = 4.2, 6.2                  # cell width/height


def draw_box(ax, cx, cy, text, cat):
    fc = PAL[cat]
    box = FancyBboxPatch((cx - BW/2, cy - BH/2), BW, BH,
                         boxstyle='round,pad=0.02,rounding_size=0.08',
                         linewidth=0.7, edgecolor='#333333', facecolor=fc)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            color='#222222' if cat in LIGHT else 'white',
            fontsize=7.2, fontweight='medium')


def main():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, NCOL * CW); ax.set_ylim(0, NROW * CH + 0.6)
    ax.axis('off')

    for idx, (name, layers) in enumerate(MODELS.items()):
        r, c = divmod(idx, NCOL)
        x0 = c * CW + CW / 2
        y_top = (NROW - 1 - r) * CH + CH - 0.55
        ax.text(x0, y_top + 0.33, name, ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#1a1a1a')
        n = len(layers)
        for i, (label, cat) in enumerate(layers):     # layers[0]=input at top
            cy = y_top - i * (BH + VG)
            draw_box(ax, x0, cy, label, cat)
            if i < n - 1:
                y1 = cy - BH/2
                y2 = cy - VG - BH/2
                ax.add_patch(FancyArrowPatch((x0, y1), (x0, y2 + BH),
                             arrowstyle='-|>', mutation_scale=8,
                             linewidth=0.7, color='#555555'))

    # legend
    handles = [plt.Line2D([0], [0], marker='s', linestyle='', markersize=9,
                          markerfacecolor=PAL[k], markeredgecolor='#333',
                          label=v) for k, v in [
                  ('lin', 'Linear/Embed'), ('conv', 'Convolution'),
                  ('rec', 'Recurrent'), ('attn', 'Attention/Transformer'),
                  ('spec', 'Special block'), ('head', 'Classifier head'),
                  ('norm', 'Norm/Pool/Reshape'), ('io', 'Input/Output')]]
    ax.legend(handles=handles, loc='lower center', ncol=8, frameon=False,
              bbox_to_anchor=(0.5, -0.02), fontsize=8, handletextpad=0.3,
              columnspacing=1.0)
    fig.suptitle('', y=0.99)
    fig.tight_layout()
    for d in OUTDIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'fig_architectures.png', dpi=300, bbox_inches='tight')
        fig.savefig(d / 'fig_architectures.pdf', bbox_inches='tight')
    plt.close(fig)
    print('[DONE] fig_architectures written to', [str(d) for d in OUTDIRS])


if __name__ == '__main__':
    main()
