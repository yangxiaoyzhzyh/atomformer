"""超额焓 Hᴱ 多种子训练 — GIN+TFM，浓度作为原子特征第10维（官方 CheMixHub Fold 0 split）

设计（v2，取代旧的 regressor 拼接方案）：
- 旧版 (v1): 摩尔分数在 masked mean pooling 后拼接 [B, 256+1] → MLP
- 本版 (v2): 每个分子的所有原子都带上该分子的摩尔分数作为第10维 → [N, 10]
  在 GIN 输入层面保留浓度信息，GIN 和 TFM 可以学习浓度如何影响跨分子原子交互

用法:
    python scripts/train_he.py            # 默认 seeds 0-9
    python scripts/train_he.py 0,1,2,...,19  # 显式 seed 列表
"""
import sys, os, time, random, json, torch, csv
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(BASE, '..', 'data', 'HE')
CKPT = os.path.join(BASE, '..', 'checkpoints')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load compound SMILES
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
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 10, 0, 0))  # 原子特征 10 维（含摩尔分数）
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
    ax, aei = [], []; off = 0
    for ci, cid in enumerate(cmp_ids):
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
            for b in mol.GetBonds(): ia,ib=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei+=[[ia,ib],[ib,ia]]
            if not ei: ei=[[0,0]]
            _gc[smi]={'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
        g=_gc[smi]; x=torch.tensor(g['x']).float()[:,:9]
        # 拼接该分子的摩尔分数作为第10维
        mf = mol_fracs[ci]
        mf_col = torch.full((x.size(0), 1), mf, dtype=torch.float32)
        x = torch.cat([x, mf_col], dim=1)  # [N, 10]
        ax.append(x); aei.append(torch.tensor(g['ei'])+off); off+=x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

def load_split(fname):
    data = []
    df = pd.read_csv(os.path.join(DATA, fname))
    for _, r in df.iterrows():
        cmp_ids = eval(r['cmp_ids']); mol_fracs = eval(r['cmp_mole_fractions'])
        d = build(cmp_ids, mol_fracs)
        if d is None: continue
        d.y = torch.tensor(r['value'], dtype=torch.float32)
        data.append(d)
    return data

def train_seed(seed):
    print(f'\n{"="*40}\nSeed {seed}\n{"="*40}', flush=True)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    train_data = load_split('HE_train.csv')
    val_data = load_split('HE_val.csv')
    test_data = load_split('HE_test.csv')
    print(f'Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}', flush=True)

    all_y = torch.stack([d.y for d in train_data])
    y_mean, y_std = all_y.mean(), all_y.std()
    for d in train_data: d.y = (d.y - y_mean) / y_std
    for d in val_data: d.nraw = d.y.clone(); d.y = (d.y - y_mean) / y_std
    for d in test_data: d.nraw = d.y.clone(); d.y = (d.y - y_mean) / y_std

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=0)
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
            b = b.to(DEVICE); z = m(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(z, b.y)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        if ep >= 3: sch.step()
        al = tl / len(train_loader)

        m.eval()
        with torch.no_grad():
            preds, trues = [], []
            for b in val_loader:
                b = b.to(DEVICE)
                z = m(b.x, b.edge_index, b.batch)
                preds.append((z * y_std + y_mean).cpu())
                trues.append(b.nraw.cpu())
            preds = torch.cat(preds).numpy()
            trues = torch.cat(trues).numpy()
        mae = np.mean(np.abs(preds - trues))
        r, _ = pearsonr(preds, trues) if len(preds) > 2 else (0, 0)
        print(f'  Ep{ep:2d} | loss={al:.6f} | val_MAE={mae:.4f} | val_R={r:.4f}', flush=True)

        if r > best_r:
            best_r, best_ep = r, ep
            torch.save(m.state_dict(), os.path.join(CKPT, f'model_he_seed{seed}.pt'))

    # Final test evaluation
    print(f'  >>> Seed {seed} best: Ep{best_ep} val_R={best_r:.4f}', flush=True)
    m.load_state_dict(torch.load(os.path.join(CKPT, f'model_he_seed{seed}.pt'), weights_only=True, map_location=DEVICE))
    m.eval()
    with torch.no_grad():
        preds, trues = [], []
        for b in test_loader:
            b = b.to(DEVICE)
            z = m(b.x, b.edge_index, b.batch)
            preds.append((z * y_std + y_mean).cpu())
            trues.append(b.nraw.cpu())
        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
    test_mae = np.mean(np.abs(preds - trues))
    test_r, _ = pearsonr(preds, trues) if len(preds) > 2 else (0, 0)
    print(f'  >>> Seed {seed} test: MAE={test_mae:.4f} R={test_r:.4f}', flush=True)
    return best_r, best_ep, test_mae, test_r

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(10))
    print(f'Device: {DEVICE}, Seeds: {seeds}', flush=True)
    results = []
    for seed in seeds:
        r_val, ep, mae_test, r_test = train_seed(seed)
        results.append((seed, r_val, ep, mae_test, r_test))

    print('\n' + '='*40, flush=True)
    print('All seeds done:', flush=True)
    for seed, r_val, ep, mae_test, r_test in results:
        print(f'  Seed {seed:2d}: val_R={r_val:.4f} @ Ep{ep} | test_MAE={mae_test:.4f} test_R={r_test:.4f}', flush=True)
    r_vals = [r for _, r, _, _, _ in results]
    r_tests = [r for _, _, _, _, r in results]
    mae_tests = [m for _, _, _, m, _ in results]
    print(f'  Mean val_R:  {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}', flush=True)
    print(f'  Mean test_R: {np.mean(r_tests):.4f} ± {np.std(r_tests):.4f}', flush=True)
    print(f'  Mean test_MAE: {np.mean(mae_tests):.4f} ± {np.std(mae_tests):.4f}', flush=True)

    # ── 最佳 seed 另存为 model_he_best.pt（供 eval_he.py 使用）──
    best_seed = max(results, key=lambda t: t[1])[0]  # 按 val_R 选
    best_ckpt = os.path.join(CKPT, f'model_he_seed{best_seed}.pt')
    torch.save(torch.load(best_ckpt, weights_only=True, map_location='cpu'),
               os.path.join(CKPT, 'model_he_best.pt'))
    print(f'\nBest seed by val_R: {best_seed} → saved as model_he_best.pt', flush=True)

    # ── Ensemble 评估 ──
    print('\n' + '='*40, flush=True)
    print('Ensemble over all seeds:', flush=True)
    test_data = load_split('HE_test.csv')
    train_df = pd.read_csv(os.path.join(DATA, 'HE_train.csv'))
    y_mean = torch.tensor(train_df['value'].values, dtype=torch.float32).mean()
    y_std = torch.tensor(train_df['value'].values, dtype=torch.float32).std()
    for d in test_data: d.nraw = d.y.clone(); d.y = (d.y - y_mean) / y_std
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)
    models = []
    for seed, _, _, _, _ in results:
        mm = Model().to(DEVICE)
        mm.load_state_dict(torch.load(os.path.join(CKPT, f'model_he_seed{seed}.pt'), weights_only=True, map_location=DEVICE))
        mm.eval(); models.append(mm)
    all_preds = []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(DEVICE)
            p_seeds = []
            for mm in models:
                z = mm(b.x, b.edge_index, b.batch)
                p_seeds.append((z * y_std + y_mean).cpu().numpy())
            all_preds.append(np.mean(p_seeds, axis=0))
    preds = np.concatenate(all_preds)
    trues = torch.stack([d.nraw for d in test_data]).numpy()
    mae = np.mean(np.abs(preds - trues))
    r, _ = pearsonr(preds, trues)
    print(f'  Ensemble ({len(models)} seeds): MAE={mae:.4f} kJ/mol, R={r:.4f}', flush=True)
