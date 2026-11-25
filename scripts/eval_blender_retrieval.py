# -*- coding: utf-8 -*-
"""Blender 测试集检索评估（补充材料 S3）。

对 blender test 的 6,260 对：full 模型推理 z（A+B 无序图），
计算 z 与目标标签嵌入的 cosine（绝对位置）与 top-K 检索命中。
输出：mean/median cosine、top-1/top-3 命中率。
"""
import json
import os
import sys
import numpy as np
import torch

sys.argv = ['eval_blender_retrieval', '--variant', 'full']
_code = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'train_tgsc_bl_31dim.py'), encoding='utf-8').read()
_ns = {'__name__': 'eval_blender_retrieval',
       '__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_tgsc_bl_31dim.py')}
exec(compile(_code[:_code.index('# ─── 数据 ───')], 'x', 'exec'), _ns)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'model_tgsc_bl_best.pt')


def main():
    m = _ns['Model']()
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    m.eval()

    cache = json.load(open(os.path.join(DATA, 'odor_group_1024dim_cache.json')))
    names = sorted(cache)
    gi = {n: i for i, n in enumerate(names)}
    emb = np.array([cache[n] for n in names])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    cosines, top1, top3 = [], 0, 0
    n = 0
    for line in open(os.path.join(DATA, 'blender_test.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        g = d['odor_group'].split(',')[0].strip()
        if g not in gi:
            continue
        dd = _ns['build']([d['smiles_a'], d['smiles_b']])
        if dd is None:
            continue
        with torch.no_grad():
            z = m(dd.x, dd.edge_index, dd.batch).numpy()[0]
        z = z / np.linalg.norm(z)
        cosines.append(float(z @ emb[gi[g]]))
        n += 1
        order = np.argsort(-(emb @ z))
        if order[0] == gi[g]:
            top1 += 1
        if gi[g] in order[:3]:
            top3 += 1

    print('blender test %d 对:' % n)
    print('  z 与目标标签 cosine: mean %.4f | median %.4f' % (np.mean(cosines), np.median(cosines)))
    print('  top-1 命中目标标签: %.1f%% | top-3: %.1f%%' % (top1 / n * 100, top3 / n * 100))


if __name__ == '__main__':
    main()
