"""search_mon.py
MON 超参搜索：24 个配置 × fold-0 快速筛选 → Top 3 配置 × 5 折验证。
每个配置调用 train_mon.py（--folds 0 或 --folds 0 1 2 3 4）。
输出: 阶段1排名表 + 阶段2 5折结果。
"""
import itertools
import subprocess
import sys
import os

DATA_ROOT = os.environ.get('MON_DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'MON'))
SCRIPT = os.environ.get('MON_SCRIPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_mon.py'))
PY = os.environ.get('PYTHON', 'python')
LOG_DIR = os.environ.get('MON_LOG_DIR', 'logs_search')

# 搜索空间（LR×dropout×dim×batch，layers 固定 2）
LRS = [1e-3, 3e-4, 1e-4]
DROPS = [0.0, 0.1, 0.2]
DIMS = [128, 256]
BATCHES = [32, 64, 128, 256]
WDS = [1e-6]

configs = []
for lr, dp, dim, bs in itertools.product(LRS, DROPS, DIMS, BATCHES):
    # 裁剪：lr=1e-3 只跑 dp=0.1；dim=128 只跑 dp=0.1
    if lr == 1e-3 and dp != 0.1:
        continue
    if dim == 128 and dp != 0.1:
        continue
    configs.append({'lr': lr, 'dropout': dp, 'dim': dim, 'layers': 2,
                    'nhead': 4, 'wd': WDS[0], 'batch': bs})
print(f'配置数: {len(configs)}', flush=True)

os.makedirs(LOG_DIR, exist_ok=True)


def run(folds, cfg, tag):
    log = os.path.join(LOG_DIR, f'{tag}.log')
    cmd = [PY, SCRIPT, '--data-root', DATA_ROOT, '--folds'] + \
          [str(f) for f in folds] + \
          ['--dim', str(cfg['dim']), '--nhead', str(cfg['nhead']),
           '--layers', str(cfg['layers']), '--dropout', str(cfg['dropout']),
           '--lr', str(cfg['lr']), '--wd', str(cfg['wd']),
           '--batch', str(cfg['batch'])]
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    # 提取结果
    import re
    txt = open(log).read()
    if folds == [0]:
        m = re.search(r'>> Fold 0: best ep \d+ \| test R ([\d.-]+) MAE ([\d.]+)', txt)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)
    else:
        rs, ms = [], []
        for fm in re.finditer(r'>> Fold (\d): best ep \d+ \| test R ([\d.-]+) MAE ([\d.]+)', txt):
            rs.append(float(fm.group(2))); ms.append(float(fm.group(3)))
        if len(rs) == len(folds):
            import statistics
            return (statistics.mean(rs), statistics.mean(ms))
        return (None, None)


# 阶段 1: fold-0 筛选
print('=== 阶段 1: fold-0 筛选 ===', flush=True)
results = []
for i, cfg in enumerate(configs):
    tag = f"s1_{i:02d}_lr{cfg['lr']:.0e}_dp{cfg['dropout']}_d{cfg['dim']}_b{cfg['batch']}"
    r, mae = run([0], cfg, tag)
    results.append((cfg, r, mae))
    print(f'  {tag}: R={r} MAE={mae}', flush=True)

# 按 val 排序（fold-0 test R 作为代理）
results.sort(key=lambda x: -(x[1] if x[1] is not None else -9))
print('\n=== 阶段 1 排名 (Top 6) ===', flush=True)
for cfg, r, mae in results[:6]:
    print(f'  R={r:.4f} MAE={mae:.4f} | lr={cfg['lr']} '
          f'dropout={cfg['dropout']} dim={cfg['dim']} layers={cfg['layers']}',
          flush=True)

# 阶段 2: Top 3 × 5 折
print('\n=== 阶段 2: Top 3 × 5 折 ===', flush=True)
for cfg, r, mae in results[:3]:
    tag = (f's2_lr{cfg['lr']:.0e}_dp{cfg['dropout']}'
           f'_d{cfg['dim']}_b{cfg['batch']}')
    r5, m5 = run([0, 1, 2, 3, 4], cfg, tag)
    print(f'  {tag}: 5折 R={r5} MAE={m5}', flush=True)
