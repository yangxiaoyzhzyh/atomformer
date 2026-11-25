# -*- coding: utf-8 -*-
"""GS-CLS：pyrfume/GoodScents 单体多标签分类（MIST135 词表）。

架构复用 MON：GIN×3(64) → proj(64→256) → TFM×2(256, nhead4, ff512) → mean pool → Linear(256→135) + BCE。
训练细节复用 MON（lr=3e-4, batch 64, warmup 3, 60 ep, cosine, best-val 选点）。
数据：本地 gs_train/gs_test.jsonl（sweet 版）→ text2y(MIST135) → 分子级去重分割 300 验证 / 600 测试 / 其余训练。
"""
import json
import os
import random
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data', 'gs_cls')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

MIST_LABELS = ['almond', 'amber', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic',
    'banana', 'beefy', 'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery',
    'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy', 'cherry',
    'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac',
    'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic',
    'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut',
    'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy',
    'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky',
    'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion',
    'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine', 'pineapple',
    'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy',
    'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato', 'tropical',
    'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody']
N = len(MIST_LABELS)
L2I = {l: i for i, l in enumerate(MIST_LABELS)}
print(f'MIST classes: {N}')

# ── 图构建（31 维特征，train_tgsc_bl 同款）──
FEAT_DIM = 31
_gc = {}


def smi2g(smi):
    """返回 {'x': np.ndarray (n,64), 'ei': (2,E)}，前 31 维为原子特征。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    nf, ei = [], []
    for a in mol.GetAtoms():
        nf.append([a.GetAtomicNum() == 6, a.GetAtomicNum() == 7, a.GetAtomicNum() == 8,
                   a.GetAtomicNum() == 9, a.GetAtomicNum() == 15, a.GetAtomicNum() == 16,
                   a.GetAtomicNum() == 17, a.GetAtomicNum() == 35, a.GetAtomicNum() == 53,
                   a.GetFormalCharge() == 1, a.GetFormalCharge() == -1,
                   a.GetHybridization() == Chem.HybridizationType.SP,
                   a.GetHybridization() == Chem.HybridizationType.SP2,
                   a.GetHybridization() == Chem.HybridizationType.SP3,
                   a.GetIsAromatic(),
                   a.GetDegree() == 1, a.GetDegree() == 2, a.GetDegree() == 3, a.GetDegree() == 4,
                   a.GetTotalNumHs() == 0, a.GetTotalNumHs() == 1, a.GetTotalNumHs() == 2,
                   a.GetTotalNumHs() == 3, a.IsInRing(),
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]) > 0,
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]) > 0,
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]) > 0,
                   a.GetMass() > 30, a.GetMass() > 60, a.GetMass() > 90, a.GetMass() > 120])
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        ei.append([i, j]); ei.append([j, i])
    if not nf:
        return None
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}


def build(smi):
    if smi not in _gc:
        g = smi2g(smi)
        _gc[smi] = g if g else None
    if _gc[smi] is None:
        return None
    g = _gc[smi]
    x = torch.tensor(g['x']).float()[:, :FEAT_DIM]
    return Data(x=x, edge_index=torch.tensor(g['ei']))


# ── Model（MON 架构 + 分类头）──
class GINConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class Model(nn.Module):
    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - FEAT_DIM, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                         dim_feedforward=2 * dim,
                                         batch_first=True, dropout=dropout)
        self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(dim, N)

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        x = x[m]
        x = scatter(x, batch, dim=0, reduce='mean')
        return self.head(x)


# ── 数据 ──
def text2y(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    y = np.zeros(N, dtype=np.float32)
    for w in words:
        if w in L2I:
            y[L2I[w]] = 1.0
    return y if y.sum() > 0 else None


def load_rows(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        y = text2y(r['text'])
        if y is None:
            continue
        rows.append({'smiles': r['smiles'], 'y': y})
    return rows


def split_rows(rows, n_val=300, n_test=600):
    random.seed(SEED)
    by_smi = {}
    for r in rows:
        by_smi.setdefault(r['smiles'], []).append(r)
    smis = list(by_smi.keys())
    random.shuffle(smis)
    val = set(smis[:n_val]); test = set(smis[n_val:n_val + n_test])
    tr, va, te = [], [], []
    for smi in smis:
        if smi in val:
            va += by_smi[smi]
        elif smi in test:
            te += by_smi[smi]
        else:
            tr += by_smi[smi]
    return tr, va, te


def make_graphs(rows):
    out = []
    for r in rows:
        d = build(r['smiles'])
        if d:
            d.lbl = torch.tensor(r['y']).unsqueeze(0)
            out.append(d)
    return out


def evaluate(model, loader):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            z = model(b.x, b.edge_index, b.batch)
            zs.append(z.cpu()); ys.append(b.lbl.cpu())
    Z = torch.cat(zs).numpy(); Y = torch.cat(ys).numpy()
    aucs = [roc_auc_score(Y[:, i], Z[:, i]) for i in range(N)
            if 0 < Y[:, i].sum() < len(Y)]
    return float(np.mean(aucs)), len(aucs)


def main():
    torch.manual_seed(SEED)
    rows = load_rows(os.path.join(DATA, 'gs_train.jsonl')) + \
           load_rows(os.path.join(DATA, 'gs_test.jsonl'))
    print(f'有效样本(含≥1 MIST 标签): {len(rows)}')
    tr, va, te = split_rows(rows)
    tr_d, va_d, te_d = make_graphs(tr), make_graphs(va), make_graphs(te)
    print(f'train {len(tr_d)} | val {len(va_d)} | test {len(te_d)}')

    tr_loader = DataLoader(tr_d, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_d, batch_size=128)
    te_loader = DataLoader(te_d, batch_size=128)

    m = Model().to(DEVICE)
    print(f'Params: {sum(p.numel() for p in m.parameters()):,}')

    epochs, lr, warmup = 60, 3e-4, 3
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup)

    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        if ep < warmup:
            for pg in opt.param_groups:
                pg['lr'] = lr * (ep + 1) / warmup
        m.train()
        tl = 0
        for b in tr_loader:
            b = b.to(DEVICE)
            z = m(b.x, b.edge_index, b.batch)
            loss = F.binary_cross_entropy_with_logits(z, b.lbl)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        if ep >= warmup:
            sch.step()
        vauc, _ = evaluate(m, va_loader)
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            best_ep = ep
        print(f'  ep {ep:2d} | loss {tl/len(tr_loader):.4f} | val AUROC {vauc:.4f} | best {best_auc:.4f}@{best_ep}', flush=True)

    m.load_state_dict(best_state)
    t_auc, n_auc = evaluate(m, te_loader)
    print(f'>> best ep {best_ep} | TEST macro-AUROC = {t_auc:.4f} ({n_auc}/{N} classes)')
    torch.save(best_state, os.path.join(BASE, 'gs_cls_best.pt'))


if __name__ == '__main__':
    main()
