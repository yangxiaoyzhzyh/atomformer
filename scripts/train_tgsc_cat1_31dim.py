"""train_tgsc_cat1_31dim.py
Category 2 对照：带分子边界 + 分子内 pooling + 组合 + MSE→BGE。
与主模型 train_tgsc_bl_31dim.py 完全同数据/同特征(31维)/同损失/同评估协议。
分子内聚合：mean；分子间组合：sum / mean 两变体（--combine）。
"""
import sys, os, time, json, math
import torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

COMBINE = sys.argv[1] if len(sys.argv) > 1 else 'sum'  # sum / mean
print(f'Combine: {COMBINE}')

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
    def __init__(self, combine='sum'):
        super().__init__()
        self.combine = combine
        self.pad = nn.ZeroPad2d((0, 64 - 31, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
    def forward(self, x, ei, batch, mol_id, n_mol):
        # 分子边界内 GIN（边仅分子内共价键）
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x)
        # 分子级 mean 池化：key = batch*2 + mol_id（每样本最多 2 分子）
        key = batch * 2 + mol_id
        emb = scatter(x, key, dim=0, dim_size=2 * batch.max().item() + 2, reduce='mean')
        a = emb[0::2]  # 分子 A（monomer 样本唯一分子）
        b = emb[1::2]  # 分子 B（monomer 样本为零向量）
        if self.combine == 'sum':
            out = a + b
        else:  # mean
            out = (a + b) / n_mol.unsqueeze(-1).clamp(min=1)
        return out  # [n_samples, 1024]

class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'n_mol'): return None
        return super().__cat_dim__(key, value, *a, **kw)

_gc = {}
def build(smis):
    ax, aei, amid = [], [], []
    off = 0
    for i, s in enumerate(smis):
        if s not in _gc:
            mol = Chem.MolFromSmiles(s)
            t = [6,7,8,9,16,17,35,53]
            c = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
            nf = []
            for a in mol.GetAtoms():
                f = [1 if a.GetAtomicNum() == x else 0 for x in t]
                f += [1 if a.GetChiralTag() == x else 0 for x in c]
                deg = min(a.GetDegree(), 5); f += [1.0 if deg == i2 else 0.0 for i2 in range(6)]
                h = min(a.GetTotalNumHs(), 4); f += [1.0 if h == i2 else 0.0 for i2 in range(5)]
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
            _gc[s] = {'x': np.array(nf, dtype=np.float32),
                      'ei': np.array(ei, dtype=np.int64).T}
        g = _gc[s]
        ax.append(g['x']); aei.append(g['ei'] + off)
        amid += [i] * len(g['x'])
        off += len(g['x'])
    if not ax: return None
    x = torch.tensor(np.concatenate(ax), dtype=torch.float)
    ei = torch.tensor(np.concatenate(aei, axis=1), dtype=torch.long)
    return GData(x=x, edge_index=ei, mol_id=torch.tensor(amid, dtype=torch.long),
                 n_mol=torch.tensor(len(smis), dtype=torch.float))

# ─── 数据（与主模型完全一致）───
with open(os.path.join(BASE, '..', 'data', 'odor_group_1024dim_cache.json')) as f:
    gd = json.load(f); gn = list(gd.keys())
gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1).to(DEVICE)
gi = {n: i for i, n in enumerate(gn)}
print(f'Classes: {len(gn)}')

data = []
for line in open(os.path.join(BASE, '..', 'data', 'tgsc_train_bge.jsonl')):
    r = json.loads(line)
    d = build([r['smiles']])
    if d: d.y = F.normalize(torch.tensor(r['embedding']), dim=0); data.append(d)
gs_n = len(data)
print(f'Monomer: {gs_n}')

bl_n = 0
for line in open(os.path.join(BASE, '..', 'data', 'blender_train.jsonl')):
    r = json.loads(line); g = r['odor_group'].rstrip(', ')
    if g not in gi: continue
    d = build([r['smiles_a'], r['smiles_b']])
    if d: d.y = gv[gi[g]].cpu(); data.append(d); bl_n += 1
print(f'Blender train: {bl_n} | Total: {len(data)}')

bl_val_samples, bl_val_gts = [], []
for line in open(os.path.join(BASE, '..', 'data', 'blender_val.jsonl')):
    r = json.loads(line); g = r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    d = build([r['smiles_a'], r['smiles_b']])
    if d: bl_val_samples.append(d); bl_val_gts.append(gi[g])

bl_test = [json.loads(l) for l in open(os.path.join(BASE, '..', 'data', 'blender_test.jsonl'))]
bl_samples, bl_gts = [], []
for r in bl_test:
    g = r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    d = build([r['smiles_a'], r['smiles_b']])
    if d: bl_samples.append(d); bl_gts.append(gi[g])

def eval_auc(model, samples, gts):
    model.eval(); batch = Batch.from_data_list(samples).to(DEVICE)
    with torch.no_grad():
        z = model(batch.x, batch.edge_index, batch.batch, batch.mol_id, batch.n_mol)
    cos = torch.mm(z.cpu(), gv.cpu().t()).numpy()
    y_true = np.zeros((len(gts), len(gn)))
    for i, g in enumerate(gts): y_true[i, g] = 1
    aucs = [roc_auc_score(y_true[:, i], cos[:, i]) for i in range(len(gn))
            if 0 < y_true[:, i].sum() < len(gts)]
    return np.mean(aucs), len(aucs)

# ─── 训练（与主模型协议一致）───
m = Model(COMBINE).to(DEVICE)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
w = [100] * gs_n + [1] * bl_n
s = torch.utils.data.WeightedRandomSampler(w, len(data), replacement=True)
loader = DataLoader(data, batch_size=1024, sampler=s, num_workers=0)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
WARMUP_EPOCHS, TOTAL_EPOCHS = 3, 60
warmup_lrs = [1e-3 * (ep + 1) / WARMUP_EPOCHS for ep in range(WARMUP_EPOCHS)]

t0 = time.time(); best_val_auc = -1; best_ep = 0
for ep in range(TOTAL_EPOCHS):
    if ep < WARMUP_EPOCHS:
        for pg in opt.param_groups: pg['lr'] = warmup_lrs[ep]
    m.train(); tl = 0
    for b in loader:
        b = b.to(DEVICE)
        z = m(b.x, b.edge_index, b.batch, b.mol_id, b.n_mol)
        loss = F.mse_loss(z, b.y.float())
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    if ep >= WARMUP_EPOCHS: sch.step()
    al = tl / len(loader)
    val_auc, n_groups = eval_auc(m, bl_val_samples, bl_val_gts)
    improved = val_auc > best_val_auc
    if improved:
        best_val_auc = val_auc; best_ep = ep
        torch.save(m.state_dict(), f'best_cat1_{COMBINE}.pt')
    print(f'Ep{ep:2d} | lr={[p["lr"] for p in opt.param_groups][0]:.1e} | '
          f'loss={al:.6f} | Val_AUROC={val_auc:.4f}({n_groups}grp) '
          f'{"★" if improved else ""} | {time.time()-t0:.0f}s', flush=True)

print(f'\nBest val AUROC: {best_val_auc:.4f} @ Ep{best_ep}')
m.load_state_dict(torch.load(f'best_cat1_{COMBINE}.pt', weights_only=True, map_location=DEVICE))
test_auc, n = eval_auc(m, bl_samples, bl_gts)
print(f'Final test AUROC: {test_auc:.4f} ({n} groups)')
