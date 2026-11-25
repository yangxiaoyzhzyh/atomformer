import os, sys, json
import torch, torch.nn.functional as F
from collections import Counter

sys.argv = ['x', 'C:/Users/yangx/v3-mixture-token/atomformer_paper/repro/checkpoints/model_tgsc_bl_best.pt', 'full']
exec(open('eval_tgsc_31dim.py').read().split("if MODE == 'monomer':")[0])
DATA = 'C:/Users/yangx/v3-mixture-token/atomformer_paper/repro/data'

m = Model(VARIANT).to(DEVICE)
m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True)); m.eval()

test_data = [json.loads(l) for l in open(os.path.join(DATA, 'tgsc_test_bge.jsonl'))]
lib_texts, lib_embs = [], []
for line in open(os.path.join(DATA, 'tgsc_train_bge.jsonl')):
    r = json.loads(line)
    lib_texts.append(r['description']); lib_embs.append(r['embedding'])
lib_embs = F.normalize(torch.tensor(lib_embs), dim=1)
samples = [(d, r['description']) for d, r in
           [(build(r['smiles']), r) for r in test_data] if d is not None]

fw = lambda s: s.split()[0] if s else ''
miss = []
for d, desc in samples:
    z = embed(m, d)
    cos = torch.mm(z.unsqueeze(0), lib_embs.t())
    vals, idxs = cos[0].topk(100)
    w = F.softmax(vals / 0.1, dim=0).numpy()
    wc = {}
    for j, wt in zip(idxs.tolist(), w):
        wd = fw(lib_texts[j]); wc[wd] = wc.get(wd, 0) + wt
    ranked = sorted(wc.items(), key=lambda x: -x[1])
    preds = [r[0] for r in ranked[:3]]
    gt = fw(desc)
    if gt not in preds[:3]:
        miss.append((gt, preds))

print('未命中样本: %d/%d' % (len(miss), len(samples)))

gd = json.load(open(os.path.join(DATA, 'odor_group_1024dim_cache.json')))
names = sorted(gd)
emb = F.normalize(torch.tensor([gd[n] for n in names]), dim=1)
name2i = {n: i for i, n in enumerate(names)}

gt_in = 0; near80 = 0; near90 = 0; near70 = 0
samples_list = []
for gt, preds in miss:
    if gt not in name2i:
        continue
    gt_in += 1
    best = (0.0, None)
    for p in preds:
        if p in name2i:
            c = float((emb[name2i[gt]] * emb[name2i[p]]).sum())
            if c > best[0]: best = (c, p)
    samples_list.append((gt, preds, best[0], best[1]))
    if best[0] >= 0.90: near90 += 1
    if best[0] >= 0.80: near80 += 1
    if best[0] >= 0.70: near70 += 1

print('未命中且 gt 在 128 词表: %d' % gt_in)
print('gt 与某预测词 cosine>=0.90: %d' % near90)
print('gt 与某预测词 cosine>=0.80: %d (%.1f%%)' % (near80, near80/max(gt_in,1)*100))
print('gt 与某预测词 cosine>=0.70: %d (%.1f%%)' % (near70, near70/max(gt_in,1)*100))
print()
print('最高 cosine 样本（gt, top3, max_cos, 最近词）:')
for gt, preds, c, bp in sorted(samples_list, key=lambda x: -x[2])[:15]:
    print('  %s  <-  %s  (cos=%.3f, 最近=%s)' % (gt, preds, c, bp))
print()
oov = Counter(gt for gt, _ in miss if gt not in name2i)
print('gt 不在 128 词表的未命中样本首词示例（%d 个样本）:' % sum(oov.values()))
for g, n in oov.most_common(20):
    print('  %s x%d' % (g, n))
