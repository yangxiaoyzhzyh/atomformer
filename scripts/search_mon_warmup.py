"""search_mon_warmup.py
固定默认最优参数（lr=3e-4, dp=0.1, dim=256, layers=2, batch=64），
搜索 warmup × epochs（12 配置）。阶段1: fold-0 筛选；阶段2: Top3 × 5 折。
"""
import itertools
import subprocess
import os
import re
import statistics

DATA_ROOT = os.environ.get('MON_DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'MON'))
SCRIPT = os.environ.get('MON_SCRIPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_mon.py'))
PY = os.environ.get('PYTHON', 'python')
LOG_DIR = os.environ.get('MON_LOG_DIR', 'logs_search_warmup')

BASE = {'lr': 3e-4, 'dropout': 0.1, 'dim': 256, 'layers': 2,
        'nhead': 4, 'wd': 1e-6, 'batch': 64}
WARMUPS = [1, 3, 5, 8]
EPOCHS = [40, 60, 80]

os.makedirs(LOG_DIR, exist_ok=True)


def run(folds, warmup, epochs, tag):
    log = os.path.join(LOG_DIR, f'{tag}.log')
    cmd = [PY, SCRIPT, '--data-root', DATA_ROOT, '--folds'] + \
          [str(f) for f in folds] + \
          ['--dim', str(BASE['dim']), '--nhead', str(BASE['nhead']),
           '--layers', str(BASE['layers']), '--dropout', str(BASE['dropout']),
           '--lr', str(BASE['lr']), '--wd', str(BASE['wd']),
           '--batch', str(BASE['batch']), '--warmup', str(warmup),
           '--epochs', str(epochs)]
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    txt = open(log).read()
    if len(folds) == 1:
        m = re.search(r'>> Fold 0: best ep \d+ \| test R ([\d.-]+) MAE ([\d.]+)', txt)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)
    rs, ms = [], []
    for fm in re.finditer(r'>> Fold (\d): best ep \d+ \| test R ([\d.-]+) MAE ([\d.]+)', txt):
        rs.append(float(fm.group(2)))
        ms.append(float(fm.group(3)))
    if len(rs) == len(folds):
        return (statistics.mean(rs), statistics.mean(ms))
    return (None, None)


# 阶段 1: fold-0
print('=== 阶段 1: fold-0（按 MAE 排）===', flush=True)
results = []
for i, (wu, ep) in enumerate(itertools.product(WARMUPS, EPOCHS)):
    tag = f's1_{i:02d}_wu{wu}_ep{ep}'
    r, mae = run([0], wu, ep, tag)
    results.append((wu, ep, r, mae))
    print(f'  {tag}: R={r} MAE={mae}', flush=True)

results.sort(key=lambda x: x[3] if x[3] is not None else 999)
print('\n=== Top 5（按 MAE）===', flush=True)
for wu, ep, r, mae in results[:5]:
    print(f'  warmup={wu} epochs={ep}: R={r:.4f} MAE={mae:.4f}', flush=True)

# 阶段 2: Top 3 × 5 折
print('\n=== 阶段 2: Top 3 × 5 折 ===', flush=True)
for wu, ep, r, mae in results[:3]:
    tag = f's2_wu{wu}_ep{ep}'
    r5, m5 = run([0, 1, 2, 3, 4], wu, ep, tag)
    print(f'  warmup={wu} epochs={ep}: 5折 R={r5} MAE={m5}', flush=True)
