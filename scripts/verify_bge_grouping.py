# -*- coding: utf-8 -*-
"""验证 BGE-M3 嵌入空间对气味描述的区分度（补充材料 S3）。

将 TGSC 数据按描述的第一个词（主描述符）分组，
用组质心作为原型做 one-vs-rest 分类，并计算组内/组间余弦。
输出：组数、one-vs-rest AUROC、mean intra-group cosine、mean inter-group cosine。
"""
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_groups():
    groups = defaultdict(list)
    with open(os.path.join(DATA, 'tgsc_train_bge.jsonl'), encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            desc = d.get('description', '').strip()
            if not desc:
                continue
            first = desc.split()[0]
            groups[first].append(np.array(d['embedding'], dtype=np.float64))
    return {g: np.stack(v) for g, v in groups.items()}


def main():
    groups = load_groups()
    names = sorted(groups)
    emb = np.vstack([groups[g] for g in names])
    n = len(names)
    print(f'组数: {n} | 样本数: {len(emb)}')

    centroids = np.stack([groups[g].mean(0) for g in names])
    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cen_n = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sim = emb_n @ cen_n.T

    aucs, offset = [], 0
    for i, g in enumerate(names):
        k = len(groups[g])
        y = np.zeros(len(emb)); y[offset:offset + k] = 1
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, sim[:, i]))
        offset += k
    print(f'one-vs-rest AUROC: {np.mean(aucs):.4f} ({len(aucs)} 组)')

    # 样本-质心口径（与 one-vs-rest AUROC 一致）：
    # intra = 组内样本与自身质心的 cosine；inter = 组内样本与他组质心的 cosine
    intra, inter, offset = [], [], 0
    for i, g in enumerate(names):
        k = len(groups[g])
        s = sim[offset:offset + k]
        intra.append(float(s[:, i].mean()))
        inter.append(float(np.delete(s, i, axis=1).mean()))
        offset += k
    intra_m, inter_m = np.mean(intra), np.mean(inter)
    print(f'mean intra-group cosine (sample-centroid): {intra_m:.4f}')
    print(f'mean inter-group cosine (sample-other-centroid): {inter_m:.4f} '
          f'(separation {intra_m - inter_m:.4f})')


if __name__ == '__main__':
    main()
