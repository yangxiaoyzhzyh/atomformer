# -*- coding: utf-8 -*-
"""标签歧义的中点验证（补充材料 S3, Table S9 + 几何保证）。

1) Table S9：对 1,000 个冲突 AB/BA 对，用 full 模型推理（A+B 与 B+A 图同构，输出 z 相同），
   计算 z 与语义中点、两个端点标签嵌入的余弦；统计 z 更接近中点的比例。
2) 检索验证：z 的 top-2 最近邻命中两个冲突标签的比例。
3) 中点几何保证：任意两端点嵌入的中点，top-2 最近邻恰为两端点（全量枚举）。

语义中点 = 归一化嵌入的均值 (e1+e2)/2（训练 target 为归一化嵌入，MSE 最优解即此）。
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'model_tgsc_bl_best.pt')

# 复用 train_tgsc_bl_31dim.py 的模型与特征（截断到数据加载之前，只保留定义）
sys.argv = ['verify_midpoint', '--variant', 'full']
_code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'train_tgsc_bl_31dim.py'), encoding='utf-8').read()
_ns = {'__name__': 'verify_midpoint',
       '__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_tgsc_bl_31dim.py')}
exec(compile(_code[:_code.index('# ─── 数据 ───')], 'train_tgsc_bl_31dim.py', 'exec'), _ns)
T = _ns


def torch_load(path):
    import torch
    return torch.load(path, map_location='cpu')


def main():
    import torch
    m = T['Model']()
    m.load_state_dict(torch_load(CKPT))
    m.eval()

    cache = json.load(open(os.path.join(DATA, 'odor_group_1024dim_cache.json')))

    rows = []
    for line in open(os.path.join(DATA, 'blender_train.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        rows.append((d['smiles_a'], d['smiles_b'], d['odor_group'].split(',')[0].strip()))
    labs = defaultdict(set)
    for a, b, g in rows:
        labs[(a, b)].add(g)

    conflicts = []
    for (a, b), gs in labs.items():
        if (b, a) in labs and labs[(b, a)] != gs and len(gs) == 1 and len(labs[(b, a)]) == 1:
            g1, g2 = list(gs)[0], list(labs[(b, a)])[0]
            if g1 in cache and g2 in cache:
                conflicts.append(((a, b), g1, g2))
    print('冲突对: %d' % len(conflicts))

    rng = np.random.default_rng(0)
    sample = [conflicts[i] for i in rng.choice(len(conflicts), 1000, replace=False)]

    zs, g1s, g2s, n1s, n2s = [], [], [], [], []
    with torch.no_grad():
        for (a, b), g1, g2 in sample:
            d = T['build']([a, b])
            z = m(d.x, d.edge_index, d.batch)
            e1, e2 = cache.get(g1), cache.get(g2)
            zs.append(z.numpy()[0]); g1s.append(e1); g2s.append(e2)
            n1s.append(g1); n2s.append(g2)
    ok = [(z, g1, g2) for z, g1, g2 in zip(zs, g1s, g2s) if g1 is not None and g2 is not None]
    okn = [(n1, n2) for n1, n2, g1, g2 in zip(n1s, n2s, g1s, g2s) if g1 is not None and g2 is not None]
    print('有效: %d' % len(ok))

    z_n = np.array([z / np.linalg.norm(z) for z, _, _ in ok])
    g1 = np.array([g / np.linalg.norm(g) for _, g, _ in ok])
    g2 = np.array([g / np.linalg.norm(g) for _, _, g in ok])
    mid_raw = np.array([(np.array(g1_) + np.array(g2_)) / 2 for z_, g1_, g2_ in ok])
    mid_n = mid_raw / np.linalg.norm(mid_raw, axis=1, keepdims=True)

    c_mid = (z_n * mid_n).sum(1)
    c_a = (z_n * g1).sum(1)
    c_b = (z_n * g2).sum(1)
    c_ab = (g1 * g2).sum(1)
    print('Mean   cosine: z->mid %.4f | z->A %.4f | z->B %.4f | A<->B %.4f'
          % (c_mid.mean(), c_a.mean(), c_b.mean(), c_ab.mean()))
    print('Median cosine: z->mid %.4f | z->A %.4f | z->B %.4f | A<->B %.4f'
          % (np.median(c_mid), np.median(c_a), np.median(c_b), np.median(c_ab)))
    closer_mid = (c_mid > c_a) & (c_mid > c_b)
    print('z 更接近中点: %.1f%%' % (closer_mid.mean() * 100))

    gn_list = sorted(cache)
    gidx = {n: i for i, n in enumerate(gn_list)}
    gn_all = np.array([cache[n] for n in gn_list])
    gn_all = gn_all / np.linalg.norm(gn_all, axis=1, keepdims=True)
    both = one = 0
    for i in range(len(ok)):
        top2 = set(np.argsort(-(gn_all @ z_n[i]))[:2])
        ia, ib = gidx[okn[i][0]], gidx[okn[i][1]]
        if top2 == {ia, ib}:
            both += 1
        if ia in top2 or ib in top2:
            one += 1
    n = len(ok)
    print('z 检索 top-2 恰为两冲突标签: %.1f%% | 含至少一个: %.1f%%' % (both / n * 100, one / n * 100))


def verify_midpoint_geometry():
    "中点几何验证（补充材料 S3）：任意两端点嵌入的中点，top-2 最近邻恰为两端点。"
    cache = json.load(open(os.path.join(DATA, 'odor_group_1024dim_cache.json')))
    names = sorted(cache)
    emb = np.array([cache[n] for n in names])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(names)
    ok = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            m = emb[i] + emb[j]
            m /= np.linalg.norm(m)
            total += 1
            if set(np.argsort(-(emb @ m))[:2]) == {i, j}:
                ok += 1
    print('中点几何验证: %d 对全量枚举 | top-2 == 两端点: %.1f%%' % (total, 100.0 * ok / total))


if __name__ == '__main__':
    main()
    verify_midpoint_geometry()
