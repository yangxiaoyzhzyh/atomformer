"""eval_per_group_31dim.py: 31 维 full 模型的 blender test per-group AUROC（按训练对频率分层）"""
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = r'C:\Users\yangx\v3-mixture-token\atomformer_paper\_clean\data'
CKPT = r'C:\Users\yangx\v3-mixture-token\binding_predict\checkpoints\model_tgsc_bl_best.pt'

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
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei += [[u, v], [v, u]]
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

# 训练集频率
freq = {}
for line in open(os.path.join(DATA, 'blender_train.jsonl')):
    r = json.loads(line)
    g = r['odor_group'].rstrip(', ')
    freq[g] = freq.get(g, 0) + 1

with open(os.path.join(DATA, 'odor_group_1024dim_cache.json')) as f:
    gd = json.load(f); gn = list(gd.keys())
gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1)
gi = {n: i for i, n in enumerate(gn)}

bl_test = [json.loads(l) for l in open(os.path.join(DATA, 'blender_test.jsonl'))]
samples, gts = [], []
for r in bl_test:
    g = r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    da, db = build(r['smiles_a']), build(r['smiles_b'])
    if da is None or db is None: continue
    off = da.x.size(0)
    d = Data(x=torch.cat([da.x, db.x]),
             edge_index=torch.cat([da.edge_index, db.edge_index + off], dim=1))
    samples.append(d); gts.append(gi[g])
print(f'Test: {len(samples)}')

m = Model('full').to(DEVICE)
m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True), strict=False)
m.eval()

cos_all = np.zeros((len(samples), len(gn)))
for i, d in enumerate(samples):
    z = embed(m, d)
    cos_all[i] = torch.mm(z.unsqueeze(0), gv.t()).numpy()[0]
    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{len(samples)}', flush=True)

y_true = np.zeros((len(samples), len(gn)))
for i, g in enumerate(gts): y_true[i, g] = 1

# per-group AUROC
aucs = {}
for i, g in enumerate(gn):
    if 0 < y_true[:, i].sum() < len(y_true):
        aucs[g] = roc_auc_score(y_true[:, i], cos_all[:, i])

# 分层
rare = {g: a for g, a in aucs.items() if freq.get(g, 0) < 100}
mid = {g: a for g, a in aucs.items() if 100 <= freq.get(g, 0) <= 1000}
common = {g: a for g, a in aucs.items() if freq.get(g, 0) > 1000}

# 全表输出
os.makedirs('results', exist_ok=True)
with open('results/per_group_full_31dim.csv', 'w') as f:
    f.write('group,AUROC,train_count\n')
    for g, a in sorted(aucs.items(), key=lambda kv: -freq.get(kv[0], 0)):
        f.write(f'{g},{a:.4f},{freq.get(g, 0)}\n')
print('全表已存 results/per_group_full_31dim.csv')

print(f'\n===== per-group 分层结果（32 维 full）=====')
print(f'Rare (<100 对): {len(rare)} 组, mean AUROC = {np.mean(list(rare.values())):.4f}')
print(f'Mid (100-1K):   {len(mid)} 组, mean AUROC = {np.mean(list(mid.values())):.4f}')
print(f'Common (>1K):   {len(common)} 组, mean AUROC = {np.mean(list(common.values())):.4f}')
print(f'Overall:        {len(aucs)} 组, mean AUROC = {np.mean(list(aucs.values())):.4f}')

# 频率分布统计
import collections
fc = collections.Counter(freq.values())
print(f'\n训练对总数: {sum(freq.values()):,} | 组总数: {len(freq)}')
print('常见组示例:', sorted(common, key=lambda g: -freq[g])[:8])
print('稀有组示例:', sorted(rare, key=lambda g: -freq[g])[:8])
