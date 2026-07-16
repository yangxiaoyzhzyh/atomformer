"""训练: GIN-only (无 Transformer), TGSC 单体 + Blender, 以 BL AUROC 为早停标准"""
import sys, os, time, random, json, torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(42); random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = os.path.dirname(os.path.abspath(__file__))
print(f'Device: {DEVICE}')

class GINConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei; o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return torch.relu(self.mlp((1 + self.eps) * x + o))

class GINOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 9, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x)
        out = scatter(x, batch, dim=0, reduce='mean')
        return F.normalize(out, dim=1)

FEAT_DIM = 9
_gc = {}
class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y',): return None
        return super().__cat_dim__(key, value, *a, **kw)

def smi2g(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    t = [6, 7, 8, 9, 16, 17, 35, 53]
    c = [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in t]
        f += [1 if a.GetChiralTag() == x else 0 for x in c]
        f += [a.GetDegree() / 5, a.GetFormalCharge() / 5, 1 if a.IsInRing() else 0, 0]
        nf.append(f[:9])
    ei = []
    for b in mol.GetBonds(): i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); ei += [[i, j], [j, i]]
    if not ei: ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}

def build(smis):
    ax, aei = [], []; off = 0
    for s in smis:
        if s not in _gc:
            g = smi2g(s)
            if g is None: return None
            _gc[s] = g
        g = _gc[s]; x = torch.tensor(g['x']).float()
        if x.size(1) > FEAT_DIM: x = x[:, :FEAT_DIM]
        ax.append(x); aei.append(torch.tensor(g['ei']) + off)
        off += x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

# ─── 数据 ───
print('Loading monomer training...')
data = []
for line in open(os.path.join(BASE, 'tgsc_train_bge.jsonl')):
    r = json.loads(line)
    d = build([r['smiles']])
    if d: d.y = F.normalize(torch.tensor(r['embedding']), dim=0); data.append(d)
gs_n = len(data)
print(f'  Monomer: {gs_n}')

print('Loading blender training...')
with open(os.path.join(BASE, 'odor_group_1024dim_cache.json')) as f:
    gd = json.load(f); gn = list(gd.keys())
    gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1).to(DEVICE)
gi = {n: i for i, n in enumerate(gn)}

bl_n = 0
for line in open(os.path.join(BASE, 'blender_train.jsonl')):
    r = json.loads(line); g = r['odor_group'].rstrip(', ')
    if g not in gi: continue
    d = build([r['smiles_a'], r['smiles_b']])
    if d: d.y = gv[gi[g]].cpu(); data.append(d); bl_n += 1
print(f'  Blender: {bl_n}')
print(f'  Total: {len(data)}')

# ─── 评估 ───
print('Loading blender test...')
bl_test = [json.loads(line) for line in open(os.path.join(BASE, 'blender_test.jsonl'))]
print(f'  Blender test: {len(bl_test)}')

print('Building blender test graphs...')
bl_samples, bl_gts = [], []
for r in bl_test:
    g = r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    d = build([r['smiles_a'], r['smiles_b']])
    if d: bl_samples.append(d); bl_gts.append(gi[g])
print(f'  Graphs: {len(bl_samples)}')

def eval_bl_auroc(model):
    model.eval()
    batch = Batch.from_data_list(bl_samples).to(DEVICE)
    with torch.no_grad():
        z = model(batch.x, batch.edge_index, batch.batch)
    cos = torch.mm(z.cpu(), gv.cpu().t()).numpy()
    y_true = np.zeros((len(bl_gts), len(gn)))
    for i, g in enumerate(bl_gts): y_true[i, g] = 1
    aucs = [roc_auc_score(y_true[:, i], cos[:, i]) for i in range(len(gn))
            if 0 < y_true[:, i].sum() < len(bl_gts)]
    return np.mean(aucs), len(aucs)

# ─── 训练 ───
m = GINOnly().to(DEVICE)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
w = [100] * gs_n + [1] * bl_n
s = torch.utils.data.WeightedRandomSampler(w, len(data), replacement=True)
loader = DataLoader(data, batch_size=1024, sampler=s, num_workers=0)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

WARMUP_EPOCHS = 3
TOTAL_EPOCHS = 60
warmup_lrs = [1e-3 * (ep + 1) / WARMUP_EPOCHS for ep in range(WARMUP_EPOCHS)]

t0 = time.time(); best_auc = -1
for ep in range(TOTAL_EPOCHS):
    if ep < WARMUP_EPOCHS:
        for pg in opt.param_groups: pg['lr'] = warmup_lrs[ep]
    m.train(); tl = 0
    for b in loader:
        b = b.to(DEVICE); z = m(b.x, b.edge_index, b.batch)
        loss = F.mse_loss(z, b.y.float())
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
    if ep >= WARMUP_EPOCHS: sch.step()
    al = tl / len(loader)

    bl_auc, n_groups = eval_bl_auroc(m)
    torch.save(m.state_dict(), os.path.join(BASE, f'model_gin_only_ep{ep}.pt'))

    improved = bl_auc > best_auc
    if improved: best_auc = bl_auc
    tag = '★' if improved else ' '
    clr = opt.param_groups[0]['lr']
    elapsed = time.time() - t0
    print(f'Ep{ep:2d} | lr={clr:.1e} | loss={al:.6f} | BL_AUROC={bl_auc:.4f}({n_groups}grp) {tag} | {elapsed:.0f}s')

print(f'\nDone! Best BL AUROC: {best_auc:.4f}')
