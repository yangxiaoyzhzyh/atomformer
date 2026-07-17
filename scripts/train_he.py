"""超额焓 Hᴱ 多种子训练, GIN+TFM"""
import sys, os, random, torch, csv
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load compound SMILES
cmp_smiles = {}
with open(BASE + '/HE_compounds.csv') as f:
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
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 9, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 256)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1)
        self.tfm = nn.TransformerEncoder(enc, 2)
        self.regressor = nn.Sequential(
            nn.Linear(256 + 1, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1))
    def forward(self, x, ei, batch, extra=None):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x); x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        if extra is not None: x = torch.cat([x, extra], dim=1)
        return self.regressor(x).squeeze(1)

class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y', 'extra'): return None
        return super().__cat_dim__(key, value, *a, **kw)

_gc = {}
def build(cmp_ids):
    ax, aei = [], []; off = 0
    for cid in cmp_ids:
        cid_int = int(float(cid))
        smi = cmp_smiles.get(cid_int)
        if smi is None: return None
        if smi not in _gc:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None
            t=[6,7,8,9,16,17,35,53]; c=[Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
            nf=[]
            for a in mol.GetAtoms():
                f=[1 if a.GetAtomicNum()==x else 0 for x in t]
                f+=[1 if a.GetChiralTag()==x else 0 for x in c]
                f+=[a.GetDegree()/5,a.GetFormalCharge()/5,1 if a.IsInRing() else 0,0]; nf.append(f[:9])
            ei=[]
            for b in mol.GetBonds(): i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei+=[[i,j],[j,i]]
            if not ei: ei=[[0,0]]
            _gc[smi]={'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
        g=_gc[smi]; x=torch.tensor(g['x']).float()[:,:9]
        ax.append(x); aei.append(torch.tensor(g['ei'])+off); off+=x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

def train_seed(seed):
    print(f'\n{"="*40}\nSeed {seed}\n{"="*40}')
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    # Load data — use official CheMixHub Fold 0 split (no random shuffling)
    train_data = []
    df = pd.read_csv(BASE + '/HE_train.csv')
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids']); mol_fracs = eval(r['cmp_mole_fractions'])
        d = build(cmp_ids)
        if d is None: continue
        d.y = torch.tensor(r['value'], dtype=torch.float32)
        d.extra = torch.tensor([mol_fracs[0]], dtype=torch.float32)
        train_data.append(d)

    test_data = []
    df = pd.read_csv(BASE + '/HE_test.csv')
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids']); mol_fracs = eval(r['cmp_mole_fractions'])
        d = build(cmp_ids)
        if d is None: continue
        d.y = torch.tensor(r['value'], dtype=torch.float32)
        d.extra = torch.tensor([mol_fracs[0]], dtype=torch.float32)
        test_data.append(d)

    # Z-score normalize
    all_y = torch.stack([d.y for d in train_data])
    y_mean, y_std = all_y.mean(), all_y.std()
    for d in train_data: d.y = (d.y - y_mean) / y_std
    for d in test_data: d.nraw = d.y.clone(); d.y = (d.y - y_mean) / y_std

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    m = Model().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=37)

    best_r, best_ep = -1, 0
    for ep in range(40):
        if ep < 3:
            for pg in opt.param_groups: pg['lr'] = 1e-4 * (ep + 1) / 3
        m.train(); tl = 0
        for b in train_loader:
            b = b.to(DEVICE); z = m(b.x, b.edge_index, b.batch, b.extra)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        if ep >= 3: sch.step()
        al = tl / len(train_loader)

        m.eval()
        with torch.no_grad():
            preds, trues = [], []
            for b in test_loader:
                b = b.to(DEVICE)
                z = m(b.x, b.edge_index, b.batch, b.extra)
                preds.append((z * y_std + y_mean).cpu())
                trues.append(b.nraw.cpu())
            preds = torch.cat(preds).numpy()
            trues = torch.cat(trues).numpy()
        mae = np.mean(np.abs(preds - trues))
        r, _ = pearsonr(preds, trues) if len(preds) > 2 else (0, 0)
        print(f'  Ep{ep:2d} | loss={al:.6f} | MAE={mae:.4f} | R={r:.4f}')

        if r > best_r:
            best_r, best_ep = r, ep
            torch.save(m.state_dict(), f"{BASE}/model_he_seed{seed}.pt")

    print(f'  >>> Seed {seed} best: Ep{best_ep} R={best_r:.4f}')
    return best_r, best_ep

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(10))
    print(f'Device: {DEVICE}, Seeds: {seeds}')
    results = []
    for seed in seeds:
        r, ep = train_seed(seed)
        results.append((seed, r, ep))

    print('\n' + '='*40)
    print('All seeds done:')
    for seed, r, ep in results:
        print(f'  Seed {seed:2d}: best R={r:.4f} @ Ep{ep}')
    rs = [r for _, r, _ in results]
    print(f'  Mean R={np.mean(rs):.4f} ± {np.std(rs):.4f}')
