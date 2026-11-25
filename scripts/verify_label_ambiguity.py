# -*- coding: utf-8 -*-
"""标签歧义统计（补充材料 S3 / 正文 2.1）：条目级冲突统计。

口径：一条 "A+B → 气味描述" 为一个条目；总条目数为分母。
冲突条目 = 存在另一个条目共享同一无序分子对（A+B / B+A 在模型中不可区分）
但气味描述不同的条目。同时报告冲突标签嵌入的余弦相似度与随机基线。
"""
import json
import os
import numpy as np
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def main():
    rows = []
    for line in open(os.path.join(DATA, 'blender_train.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        key = tuple(sorted([d['smiles_a'], d['smiles_b']]))
        rows.append((key, d['odor_group'].split(',')[0].strip()))

    total = len(rows)
    key_labs = defaultdict(set)
    for key, g in rows:
        key_labs[key].add(g)
    conflict = sum(1 for key, g in rows if len(key_labs[key]) > 1)
    frac = conflict / total
    print(f'总条目: {total}')
    print(f'冲突条目: {conflict} | 占比: {frac:.3f} ({frac*100:.1f}%)')

    # 冲突标签对的嵌入余弦 vs 随机基线
    cache = json.load(open(os.path.join(DATA, 'odor_group_1024dim_cache.json')))
    names = list(cache.keys())
    emb = np.array([cache[n] for n in names])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    idx = {n: i for i, n in enumerate(names)}

    seen = set(); sims = []
    for key, gs in key_labs.items():
        gs = sorted(gs)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                if gs[i] in idx and gs[j] in idx:
                    p = (gs[i], gs[j])
                    if p not in seen:
                        seen.add(p)
                        sims.append(float(emb[idx[gs[i]]] @ emb[idx[gs[j]]]))
    print(f'冲突标签 cosine: {np.mean(sims):.3f} ({len(sims)} 对)')

    rng = np.random.default_rng(0)
    base = []
    for _ in range(50000):
        i, j = rng.integers(0, len(names)), rng.integers(0, len(names))
        if i != j:
            base.append(float(emb[i] @ emb[j]))
    print(f'随机标签基线 cosine: {np.mean(base):.3f}')


if __name__ == '__main__':
    main()
