"""verify_odor_claims.py
验证讨论段三个可检验断言（零成本，用现有 ckpt）：
A. 单体 conR@3 按重原子数分层 —— 预测：分子越大，TFM-only 相对 full 越差
B. blender AUROC 按总重原子数分层 —— 预测：图越大，TFM-only 相对 full 越差
C. 骨架混淆分组（属性包相似但 Murcko 骨架不同）—— 预测：混淆度越高，
   TFM-only 相对 full/gin_only 的差距越大
用法: python verify_odor_claims.py
"""
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
from collections import defaultdict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = os.environ.get('ODOR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
CKPT_DIR = os.environ.get('ODOR_CKPT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints'))
CKPTS = {k: os.path.join(CKPT_DIR, v) for k, v in {
    'full': 'model_tgsc_bl_best.pt',
    'gin_only': 'model_tgsc_bl_gin_only_best.pt',
    'tfm_only': 'model_tgsc_bl_tfm_only_best.pt',
}.items()}
print('Device:', DEVICE)

class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
                                 nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))

class Model(nn.Module):
    def __init__(self, variant='full'):
        super().__init__()
        self.variant = variant
        self.pad = nn.ZeroPad2d((0, 64 - 31, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
        if variant != 'gin_only':
            enc = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True, dropout=0.1)
            self.tfm = nn.TransformerEncoder(enc, 2)
    def forward(self, x, ei, batch):
        x = self.pad(x)
        if self.variant != 'tfm_only':
            for c in self.convs: x = c(x, ei)
        x = self.proj(x); x, m = to_dense_batch(x, batch)
        if self.variant == 'gin_only':
            me = m.unsqueeze(-1).float()
            return F.normalize((x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1), dim=1)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        return F.normalize((x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1), dim=1)

_gc = {}
def build(smi):
    if smi not in _gc:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        t = [6,7,8,9,16,17,35,53]
        c = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
        nf = []
        for a in mol.GetAtoms():
            f = [1 if a.GetAtomicNum() == x else 0 for x in t]
            f += [1 if a.GetChiralTag() == x else 0 for x in c]
            deg = min(a.GetDegree(), 5); f += [1.0 if deg == i else 0.0 for i in range(6)]
            h = min(a.GetTotalNumHs(), 4); f += [1.0 if h == i else 0.0 for i in range(5)]
            hyb = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                   Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                   Chem.HybridizationType.SP3D2]
            f += [1.0 if a.GetHybridization() == y else 0.0 for y in hyb]
            f += [1.0 if a.GetIsAromatic() else 0.0]
            fc = int(a.GetFormalCharge())
            f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0, 1.0 if fc > 0 else 0.0]
            f += [1.0 if a.IsInRing() else 0.0]
            nf.append(f[:31])
        ei = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei += [[i, j], [j, i]]
        if not ei: ei = [[0, 0]]
        _gc[smi] = {'x': np.array(nf, dtype=np.float32),
                    'ei': np.array(ei, dtype=np.int64).T}
    g = _gc[smi]
    return Data(x=torch.tensor(g['x']).float()[:, :31],
                edge_index=torch.tensor(g['ei']))

def embed(m, d):
    with torch.no_grad():
        return m(d.x.to(DEVICE), d.edge_index.to(DEVICE),
                 torch.zeros(d.x.size(0)).long().to(DEVICE)).cpu().squeeze(0)

def heavies(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol else 0

def propbag(smi):
    """31 维特征均值向量（属性包近似）。"""
    d = build(smi)
    return d.x.mean(dim=0).numpy() if d is not None else None

def murcko(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None

def fw(t): return t.strip().split()[0].strip(',;.')

# ── 加载 ──
with open(os.path.join(DATA, 'odor_group_1024dim_cache.json')) as f:
    gd = json.load(f); gn = list(gd.keys())
gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1)
gi = {n: i for i, n in enumerate(gn)}

mon_test = [json.loads(l) for l in open(os.path.join(DATA, 'tgsc_test_bge.jsonl'))]
lib_texts, lib_embs = [], []
for line in open(os.path.join(DATA, 'tgsc_train_bge.jsonl')):
    r = json.loads(line)
    lib_texts.append(r['description']); lib_embs.append(r['embedding'])
lib_embs = F.normalize(torch.tensor(lib_embs), dim=1)

bl_test = [json.loads(l) for l in open(os.path.join(DATA, 'blender_test.jsonl'))]

def load_models():
    ms = {}
    for v, ck in CKPTS.items():
        m = Model(v).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=False)
        m.eval()
        ms[v] = m
        print(f'Loaded {v} ({sum(p.numel() for p in m.parameters()):,})')
    return ms

def monomer_conr3_by_size(m, samples, bins=[15, 30, 999]):
    """按重原子数分层 conR@3。"""
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])  # (hit, n)
    for d, desc, smi in samples:
        h = heavies(smi)
        b = 0 if h <= bins[0] else (1 if h <= bins[1] else 2)
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
        buckets[b][1] += 1
        if gt in preds[:3]: buckets[b][0] += 1
    return {f'<= {bins[0]}': (buckets[0][0], buckets[0][1]),
            f'{bins[0]+1}-{bins[1]}': (buckets[1][0], buckets[1][1]),
            f'> {bins[1]}': (buckets[2][0], buckets[2][1])}

def blend_auroc_by_size(m, samples, bins=[40, 60, 999]):
    """按总重原子数分层 blender AUROC。"""
    from collections import defaultdict
    buckets = defaultdict(lambda: ([], []))  # (y_true, cos)
    for d, g, n_atoms in samples:
        b = 0 if n_atoms <= bins[0] else (1 if n_atoms <= bins[1] else 2)
        z = embed(m, d)
        cos = torch.mm(z.unsqueeze(0), gv.t()).numpy()[0]
        y = np.zeros(len(gn)); y[g] = 1
        buckets[b][0].append(y); buckets[b][1].append(cos)
    out = {}
    for k, (ys, cs) in buckets.items():
        Y = np.array(ys); C = np.array(cs)
        aucs = [roc_auc_score(Y[:, i], C[:, i]) for i in range(len(gn))
                if 0 < Y[:, i].sum() < len(Y)]
        out[k] = (np.mean(aucs), len(aucs), len(Y))
    return out

def main():
    ms = load_models()

    # ── A. 单体分层 ──
    print('\n===== A. 单体 conR@3 按重原子数分层 =====')
    mon_samples = []
    for r in mon_test:
        d = build(r['smiles'])
        if d: mon_samples.append((d, r['description'], r['smiles']))
    print(f'单体: {len(mon_samples)}')
    for v, m in ms.items():
        res = monomer_conr3_by_size(m, mon_samples)
        print(f'  {v:10s}: ' + ' | '.join(f'{k}: {h}/{n}={h/n*100:.1f}%'
              for k, (h, n) in res.items()))

    # ── B. blender 原子数分层 ──
    print('\n===== B. blender AUROC 按总重原子数分层 =====')
    bl_samples = []
    for r in bl_test:
        g = r['odor_group'].split(',')[0].strip()
        if g not in gi: continue
        na = heavies(r['smiles_a']) + heavies(r['smiles_b'])
        da, db = build(r['smiles_a']), build(r['smiles_b'])
        if da is None or db is None: continue
        off = da.x.size(0)
        d = Data(x=torch.cat([da.x, db.x]),
                 edge_index=torch.cat([da.edge_index, db.edge_index + off], dim=1))
        bl_samples.append((d, gi[g], na))
    print(f'blender: {len(bl_samples)}')
    for v, m in ms.items():
        res = blend_auroc_by_size(m, bl_samples)
        print(f'  {v:10s}: ' + ' | '.join(f'{k}: {a:.4f}({ng}组/{n}样本)'
              for k, (a, ng, n) in res.items()))

    # ── C. 易混淆且气味不同（双条件筛选）──
    print('===== C. 易混淆且气味不同的子集（属性包相似、骨架不同、组不同）=====')
    # 单体训练库：属性包 + 骨架（混淆度用）
    lib_pb, lib_scaf = [], []
    for line in open(os.path.join(DATA, 'tgsc_train_bge.jsonl')):
        smi = json.loads(line)['smiles']
        pb = propbag(smi); sc = murcko(smi)
        if pb is not None and sc is not None:
            lib_pb.append(pb); lib_scaf.append(sc)
    lib_pb = np.array(lib_pb); lib_scaf = np.array(lib_scaf)
    tree_lib = cKDTree(lib_pb)
    print(f'库: {len(lib_pb)} (属性包+骨架)')

    # blender test 全集：属性包树（气味冲突度用）+ 组标签
    bl_smis = [(r['smiles_a'], r['smiles_b'],
                gi[r['odor_group'].split(',')[0].strip()])
               for r in bl_test if r['odor_group'].split(',')[0].strip() in gi]
    test_pb, test_smi_list = [], []
    for a, b, g in bl_smis:
        for s in (a, b):
            pb = propbag(s)
            if pb is not None:
                test_pb.append(pb); test_smi_list.append(s)
    test_pb = np.array(test_pb)
    tree_test = cKDTree(test_pb)
    # 每个分子的组集合（blender 对级别组；同分子跨对取并集）
    smi2g = defaultdict(set)
    for a, b, g in bl_smis:
        smi2g[a].add(g); smi2g[b].add(g)
    print(f'test 分子: {len(test_pb)}')

    def confusion_and_conflict(smi, k=20):
        """返回 (混淆度, 气味冲突度)：属性包 top-k 邻居中骨架不同占比、
        以及（在 test 集中）气味组不同的占比。"""
        pb = propbag(smi)
        if pb is None: return None, None
        _, idx = tree_lib.query(pb, k=min(k, len(lib_pb)))
        idx = np.atleast_1d(idx)
        sc = murcko(smi)
        if sc is None: return None, None
        conf = sum(1 for i in idx if lib_scaf[i] != sc) / len(idx)
        # 气味冲突：test 集属性包近邻中组不同的占比
        _, tidx = tree_test.query(pb, k=min(k, len(test_pb)))
        tidx = np.atleast_1d(tidx)
        gs = smi2g.get(smi)
        if gs is None: return None, None
        conflict = sum(1 for i in tidx if not (smi2g.get(test_smi_list[i], set()) & gs)) / len(tidx)
        return conf, conflict

    scored = []
    for a, b, g in bl_smis:
        c1, f1 = confusion_and_conflict(a)
        c2, f2 = confusion_and_conflict(b)
        if c1 is None or c2 is None: continue
        da, db = build(a), build(b)
        if da is None or db is None: continue
        off = da.x.size(0)
        d = Data(x=torch.cat([da.x, db.x]),
                 edge_index=torch.cat([da.edge_index, db.edge_index + off], dim=1))
        conf = (c1 + c2) / 2
        confl = (f1 + f2) / 2
        scored.append((conf * confl, (d, g, conf, confl)))
    scored.sort(key=lambda x: -x[0])
    n = len(scored)
    print(f'有效 blender 对: {n}')
    sub = scored[:n // 3]  # 易混淆且气味冲突最高的 1/3
    print(f'易混淆且气味不同子集: {len(sub)} 对 '
          f'(混淆度 {np.mean([s[1][2] for s in sub]):.3f}, '
          f'气味冲突 {np.mean([s[1][3] for s in sub]):.3f})')
    for v, m in ms.items():
        ys, cs = [], []
        for _, (d, g, _, _) in sub:
            z = embed(m, d)
            cos = torch.mm(z.unsqueeze(0), gv.t()).numpy()[0]
            y = np.zeros(len(gn)); y[g] = 1
            ys.append(y); cs.append(cos)
        Y = np.array(ys); C = np.array(cs)
        aucs = [roc_auc_score(Y[:, i], C[:, i]) for i in range(len(gn))
                if 0 < Y[:, i].sum() < len(Y)]
        print(f'  {v:10s}: AUROC {np.mean(aucs):.4f} ({len(aucs)}组/{len(Y)}样本)')

if __name__ == '__main__':
    main()
