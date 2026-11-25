# -*- coding: utf-8 -*-
"""plot_final_curves.py —— 把 results/curve_knee_final_s*.csv 画成 val 曲线 + 拐点标注图。
只读 CSV，不训练、不改任何结果文件。用法：python plot_final_curves.py
"""
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
curves = sorted(glob.glob(os.path.join(RES, 'curve_knee_final_s*.csv')))
if not curves:
    raise SystemExit('没有找到 curve_knee_final_s*.csv')

# 读 results_final.csv 拿每个种子的 knee / best-val epoch
meta = {}
rp = os.path.join(RES, 'results_final.csv')
if os.path.isfile(rp):
    for r in csv.DictReader(open(rp, encoding='utf-8')):
        meta[int(r['seed'])] = r

fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.6), squeeze=False)
axes = axes[0]
for ax, f in zip(axes, curves):
    sd = int(os.path.basename(f).split('_s')[-1].split('.')[0])
    rows = [r for r in csv.DictReader(open(f, encoding='utf-8')) if r.get('val')]
    ep = np.array([int(r['ep']) for r in rows])
    val = np.array([float(r['val']) for r in rows])
    d1 = np.array([float(r['d1']) if r['d1'] else np.nan for r in rows])

    ax.plot(ep, val, 'o-', color='#2563eb', lw=1.6, ms=3.5, label='val macro-AUROC')
    m = meta.get(sd)
    if m:
        k, b = int(m['knee_ep']), int(m['best_ep'])
        ax.axvline(k, color='#16a34a', ls='--', lw=1.4,
                   label=f'knee detector -> ep{k}')
        ax.plot([b], [val[ep == b][0]], '*', color='#dc2626', ms=14,
                label=f'selected ckpt ep{b} (TEST {float(m["test"]):.4f})')
    ax.set_title(f'seed {sd}', fontsize=11)
    ax.set_xlabel('epoch')
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, loc='lower right')
    if ax is axes[0]:
        ax.set_ylabel('val macro-AUROC')

fig.suptitle('Knee early stopping (final): val curve, detected knee, and selected checkpoint',
             fontsize=12)
fig.tight_layout()
out = os.path.join(RES, 'final_knee_curves.png')
fig.savefig(out, dpi=170)
print('saved:', out)
