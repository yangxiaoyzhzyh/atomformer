"""eval_tgsc_31dim.py
31 维版统一评估（对齐 train_tgsc_bl_31dim.py 的 Model/forward）：
  单体模式：conR@1/@3 + cosR@1/@3（TGSC monomer test vs 训练库）
  blender 模式：AUROC（blender test，108 groups，cosine vs 类别嵌入）
用法：
  python eval_tgsc_31dim.py <ckpt> full|gin_only|tfm_only [monomer|blender]
"""
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

CKPT = sys.argv[1]
VARIANT = sys.argv[2] if len(sys.argv) > 2 else 'full'
MODE = sys.argv[3] if len(sys.argv) > 3 else 'monomer'
DATA = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
print(f'Variant: {VARIANT} | Mode: {MODE} | ckpt: {os.path.basename(CKPT)}')

class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
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

def fw(t): return t.strip().split()[0].strip(',;.')

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

# ── 模型 ──
m = Model(VARIANT).to(DEVICE)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True), strict=False)
m.eval()

if MODE == 'monomer':
    test_data = [json.loads(l) for l in open(os.path.join(DATA, 'tgsc_test_bge.jsonl'))]
    lib_texts, lib_embs = [], []
    for line in open(os.path.join(DATA, 'tgsc_train_bge.jsonl')):
        r = json.loads(line)
        lib_texts.append(r['description']); lib_embs.append(r['embedding'])
    lib_embs = F.normalize(torch.tensor(lib_embs), dim=1)
    print(f'Test: {len(test_data)}, Lib: {len(lib_texts)}')
    samples = [(build(r['smiles']), r['description']) for r in test_data]
    samples = [(d, desc) for d, desc in samples if d is not None]
    print(f'Graphs: {len(samples)}')
    h1 = h3 = 0; ch1 = ch3 = 0
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
        if preds and preds[0] == gt: h1 += 1
        if gt in preds[:3]: h3 += 1
        cos_idxs = cos[0].topk(3).indices
        cos_preds = [fw(lib_texts[j]) for j in cos_idxs.tolist()]
        if cos_preds and cos_preds[0] == gt: ch1 += 1
        if gt in cos_preds: ch3 += 1
    n = len(samples)
    print(f'conR@1={h1/n*100:.1f}% conR@3={h3/n*100:.1f}% ({h1}/{h3}/{n})')
    print(f'cosR@1={ch1/n*100:.1f}% cosR@3={ch3/n*100:.1f}% ({ch1}/{ch3}/{n})')
else:  # blender AUROC
    with open(os.path.join(DATA, 'odor_group_1024dim_cache.json')) as f:
        gd = json.load(f); gn = list(gd.keys())
    gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1)
    gi = {n: i for i, n in enumerate(gn)}
    print(f'Groups: {len(gn)}')
    bl_test = [json.loads(l) for l in open(os.path.join(DATA, 'blender_test.jsonl'))]
    samples, gts = [], []
    for r in bl_test:
        g = r['odor_group'].split(',')[0].strip()
        if g not in gi: continue
        d = build(r['smiles_a'] + '..' + r['smiles_b']) if False else None
        # blender 是双分子：拼接两个分子图
        da, db = build(r['smiles_a']), build(r['smiles_b'])
        if da is None or db is None: continue
        off = da.x.size(0)
        d = Data(x=torch.cat([da.x, db.x]),
                 edge_index=torch.cat([da.edge_index, db.edge_index + off], dim=1))
        samples.append(d); gts.append(gi[g])
    print(f'Blender test graphs: {len(samples)}')
    all_z, y_true = [], np.zeros((len(samples), len(gn)))
    for i, d in enumerate(samples):
        all_z.append(embed(m, d)); y_true[i, gts[i]] = 1
    all_z = F.normalize(torch.stack(all_z), dim=1)
    cos = torch.mm(all_z, gv.t()).numpy()
    aucs = [roc_auc_score(y_true[:, i], cos[:, i]) for i in range(len(gn))
            if 0 < y_true[:, i].sum() < len(y_true)]
    print(f'BL AUROC = {np.mean(aucs):.4f} ({len(aucs)} groups)')
