"""超额焓 Hᴱ Ensemble 评估 — v2 架构

v2 模型约定（与 train_heatvap_5fold.py 的 32 维原子特征一致）：
- 摩尔分数作为每个分子的原子特征第32维（在 GIN 输入层面拼接），而非 pool 后拼接
- 原子特征 31 维增强 + 1 维摩尔分数 = 32 维 → ZeroPad2d 到 64
- regressor: 256 → 256 → 64 → 1，forward(x, ei, batch) 无 extra 参数

读取 checkpoints/heatvap_32dim/ 下所有 model_heatvap_seed*.pt（兼容 model_he_seed*.pt），
对 data/HE/HE_test.csv 做集成预测。数据和权重目录可用 HE_DATA_DIR / HE_CKPT_DIR 覆盖。
"""
import os, torch, csv
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('HE_DATA_DIR', os.path.join(BASE, '..', 'data', 'HE'))
CKPT_DIR = os.environ.get('HE_CKPT_DIR',
                          os.path.join(BASE, '..', 'checkpoints', 'heatvap_32dim'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cmp_smiles = {}
with open(os.path.join(DATA, 'HE_compounds.csv')) as f:
    reader = csv.DictReader(f)
    for r in reader:
        cmp_smiles[int(float(r['compound_id']))] = r['smiles']

class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei; o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))

class Model(nn.Module):
    """v2 架构：摩尔分数作为原子第32维（与 train_he_32dim.py 一致）。"""
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 32, 0, 0))  # 原子特征 32 维（增强 + 摩尔分数）
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 256)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.tfm = nn.TransformerEncoder(enc, 2)
        self.regressor = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x); x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        return self.regressor(x).squeeze(1)

class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y',): return None
        return super().__cat_dim__(key, value, *a, **kw)

_gc = {}
def build(cmp_ids, mol_fracs):
    """v2：每个分子的所有原子都带上该分子的摩尔分数作为第32维。"""
    ax, aei = [], []; off = 0
    for ci, cid in enumerate(cmp_ids):
        cid_int = int(float(cid))
        smi = cmp_smiles.get(cid_int)
        if smi is None: return None
        if smi not in _gc:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None
            t=[6,7,8,9,16,17,35,53]; c=[Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
            hyb=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,
                 Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]
            nf=[]
            for a in mol.GetAtoms():
                f=[1 if a.GetAtomicNum()==x else 0 for x in t]
                f+=[1 if a.GetChiralTag()==x else 0 for x in c]
                deg=min(a.GetDegree(),5)
                f+=[1.0 if deg==i else 0.0 for i in range(6)]
                h=min(a.GetTotalNumHs(),4)
                f+=[1.0 if h==i else 0.0 for i in range(5)]
                f+=[1.0 if a.GetHybridization()==y else 0.0 for y in hyb]
                f+=[1.0 if a.GetIsAromatic() else 0.0]
                fc=int(a.GetFormalCharge())
                f+=[1.0 if fc<0 else 0.0, 1.0 if fc==0 else 0.0, 1.0 if fc>0 else 0.0]
                f+=[1.0 if a.IsInRing() else 0.0]
                nf.append(f[:31])
            ei=[]
            for b in mol.GetBonds(): i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei+=[[i,j],[j,i]]
            if not ei: ei=[[0,0]]
            _gc[smi]={'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
        g=_gc[smi]; x=torch.tensor(g['x']).float()[:,:31]
        mf = mol_fracs[ci]
        mf_col = torch.full((x.size(0), 1), mf, dtype=torch.float32)
        x = torch.cat([x, mf_col], dim=1)  # [N, 32]
        ax.append(x); aei.append(torch.tensor(g['ei'])+off); off+=x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

# Load test data (official CheMixHub Fold 0 split — no shuffling)
test_data = []
df = pd.read_csv(os.path.join(DATA, 'HE_test.csv'))
for _, r in df.iterrows():
    cmp_ids = eval(r['cmp_ids']); mol_fracs = eval(r['cmp_mole_fractions'])
    d = build(cmp_ids, mol_fracs)
    if d is None: continue
    d.y = torch.tensor(r['value'], dtype=torch.float32)
    test_data.append(d)

# Normalize using training statistics
train_df = pd.read_csv(os.path.join(DATA, 'HE_train.csv'))
train_ys = torch.tensor(train_df['value'].values, dtype=torch.float32)
y_mean, y_std = train_ys.mean(), train_ys.std()
for d in test_data: d.nraw = d.y.clone(); d.y = (d.y - y_mean) / y_std

test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

# Find all seed checkpoints — 兼容 heatvap 与 he 两种命名
SEED_PREFIXES = ('model_heatvap_seed', 'model_he_seed')
seed_files = sorted([f for f in os.listdir(CKPT_DIR)
                     if f.endswith('.pt') and f.startswith(SEED_PREFIXES)])
print(f'Found {len(seed_files)} seeds: {seed_files}')
assert seed_files, f'No model_heatvap_seed*/model_he_seed*.pt found in {CKPT_DIR}'

models = []
for sf in seed_files:
    m = Model().to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, sf), map_location=DEVICE, weights_only=True))
    m.eval()
    models.append(m)

# Ensemble prediction
all_preds = []
with torch.no_grad():
    for b in test_loader:
        b = b.to(DEVICE)
        preds_seeds = []
        for m in models:
            z = m(b.x, b.edge_index, b.batch)
            preds_seeds.append((z * y_std + y_mean).cpu().numpy())
        all_preds.append(np.mean(preds_seeds, axis=0))

preds = np.concatenate(all_preds)
trues = torch.stack([d.nraw for d in test_data]).numpy()

mae = np.mean(np.abs(preds - trues))
r, _ = pearsonr(preds, trues)
print(f'\nEnsemble ({len(models)} seeds): MAE={mae:.4f} kJ/mol, R={r:.4f}')
