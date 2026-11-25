"""GIN-only 6 pooling methods, full 60 epoch training"""
import sys, os, time, json, torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter
from torch_geometric.nn import Set2Set, GATConv
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

class GINConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei): r,c=ei;o=scatter(x[c],r,dim=0,dim_size=x.size(0),reduce='sum');return F.relu(self.mlp((1+self.eps)*x+o))

class GINPoolModel(nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()
        self.pad = nn.ZeroPad2d((0,64-31,0,0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
        self.pool = pool
        if pool == 'weighted':
            self.attn = nn.Linear(1024, 1)
        elif pool == 'set2set':
            self.s2s = Set2Set(1024, 3); self.proj2 = nn.Linear(2048, 1024)
        elif pool == 'gat':
            self.gat = GATConv(1024, 512, 1); self.attn_proj = nn.Linear(512, 1)
    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x)
        if self.pool == 'mean': out = scatter(x, batch, dim=0, reduce='mean')
        elif self.pool == 'max': out = scatter(x, batch, dim=0, reduce='max')
        elif self.pool == 'sum': out = scatter(x, batch, dim=0, reduce='sum')
        elif self.pool == 'weighted':
            w = torch.sigmoid(self.attn(x))
            out = (scatter(w*x, batch, dim=0, reduce='sum') / scatter(w, batch, dim=0, reduce='sum').clamp(min=1))
        elif self.pool == 'set2set': out = self.proj2(self.s2s(x, batch))
        elif self.pool == 'gat':
            w = torch.sigmoid(self.attn_proj(self.gat(x, ei)))
            out = (scatter(w*x, batch, dim=0, reduce='sum') / scatter(w, batch, dim=0, reduce='sum').clamp(min=1))
        return F.normalize(out, dim=1)

print('Loading data...')
_gc = {}; FEAT_DIM = 31

class GData(Data):
    def __cat_dim__(self,key,value,*a,**kw):
        if key in ('y',): return None
        return super().__cat_dim__(key,value,*a,**kw)

def build(smis):
    ax,aei=[],[]; off=0
    for s in smis:
        if s not in _gc:
            mol=Chem.MolFromSmiles(s);t=[6,7,8,9,16,17,35,53];c=[Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW];nf=[]
            for a in mol.GetAtoms():
                f=[1 if a.GetAtomicNum()==x else 0 for x in t]
                f+=[1 if a.GetChiralTag()==x else 0 for x in c]
                deg=min(a.GetDegree(),5); f+=[1.0 if deg==i else 0.0 for i in range(6)]
                h=min(a.GetTotalNumHs(),4); f+=[1.0 if h==i else 0.0 for i in range(5)]
                hyb=[Chem.HybridizationType.SP,Chem.HybridizationType.SP2,Chem.HybridizationType.SP3,
                     Chem.HybridizationType.SP3D,Chem.HybridizationType.SP3D2]
                f+=[1.0 if a.GetHybridization()==y else 0.0 for y in hyb]
                f+=[1.0 if a.GetIsAromatic() else 0.0]
                fc=int(a.GetFormalCharge())
                f+=[1.0 if fc<0 else 0.0,1.0 if fc==0 else 0.0,1.0 if fc>0 else 0.0]
                f+=[1.0 if a.IsInRing() else 0.0];nf.append(f[:31])
            ei=[]
            for b in mol.GetBonds():i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();ei+=[[i,j],[j,i]]
            if not ei:ei=[[0,0]]
            _gc[s]={'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
        g=_gc[s]; x=torch.tensor(g['x']).float()[:,:FEAT_DIM]
        ax.append(x); aei.append(torch.tensor(g['ei'])+off); off+=x.size(0)
    return GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

with open(os.path.join(BASE, 'odor_group_1024dim_cache.json')) as f:
    gd=json.load(f); gn=list(gd.keys())
    gv=F.normalize(torch.tensor([gd[n] for n in gn]), dim=1).to(DEVICE)
gi={n:i for i,n in enumerate(gn)}

data=[]
for line in open(os.path.join(BASE, 'tgsc_train_bge.jsonl')):
    r=json.loads(line); d=build([r['smiles']])
    if d: d.y=F.normalize(torch.tensor(r['embedding']), dim=0); data.append(d)
gs_n=len(data)
for line in open(os.path.join(BASE, 'blender_train.jsonl')):
    r=json.loads(line); g=r['odor_group'].rstrip(', ')
    if g not in gi: continue
    d=build([r['smiles_a'],r['smiles_b']])
    if d: d.y=gv[gi[g]].cpu(); data.append(d)
bl_n=len(data)-gs_n
print(f'Monomer: {gs_n}, Blender: {bl_n}, Total: {len(data)}')

bl_val_samples,bl_val_gts=[],[]
for line in open(os.path.join(BASE, 'blender_val.jsonl')):
    r=json.loads(line); g=r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    d=build([r['smiles_a'],r['smiles_b']])
    if d: bl_val_samples.append(d); bl_val_gts.append(gi[g])

bl_test=[json.loads(l) for l in open(os.path.join(BASE, 'blender_test.jsonl'))]
bl_samples,bl_gts=[],[]
for r in bl_test:
    g=r['odor_group'].split(',')[0].strip()
    if g not in gi: continue
    d=build([r['smiles_a'],r['smiles_b']])
    if d: bl_samples.append(d); bl_gts.append(gi[g])

def eval_auc(model, samples, gts):
    model.eval(); batch=Batch.from_data_list(samples).to(DEVICE)
    with torch.no_grad(): z=model(batch.x,batch.edge_index,batch.batch)
    cos=torch.mm(z.cpu(),gv.cpu().t()).numpy()
    y_true=np.zeros((len(gts),len(gn)))
    for i,g in enumerate(gts): y_true[i,g]=1
    aucs=[roc_auc_score(y_true[:,i],cos[:,i]) for i in range(len(gn)) if 0<y_true[:,i].sum()<len(gts)]
    return np.mean(aucs), len(aucs)

configs = sys.argv[1:2] or ['mean','max','sum','weighted','gat','set2set']
results = []

for pool in configs:
    name=f'GINx3_{pool}'
    print(f'\n{"="*50}\n{name}\n{"="*50}')
    m = GINPoolModel(pool=pool).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
    w = [100]*gs_n + [1]*bl_n
    loader = DataLoader(data, batch_size=1024, sampler=torch.utils.data.WeightedRandomSampler(w, len(data), replacement=True), num_workers=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_val = -1; t0 = time.time()
    for ep in range(60):
        if ep < 3:
            for pg in opt.param_groups: pg['lr']=1e-3*(ep+1)/3
        m.train(); tl=0
        for b in loader:
            b=b.to(DEVICE); z=m(b.x,b.edge_index,b.batch)
            loss=F.mse_loss(z,b.y.float())
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
        if ep>=3: sch.step()
        val_auc,n_grp=eval_auc(m,bl_val_samples,bl_val_gts)
        if val_auc>best_val:
            best_val=val_auc
            torch.save(m.state_dict(), f'best_gin3_{pool}.pt')
        print(f'  Ep{ep:2d} | loss={tl/len(loader):.6f} | Val_AUROC={val_auc:.4f}({n_grp}) | {time.time()-t0:.0f}s')

    m.load_state_dict(torch.load(f'best_gin3_{pool}.pt', weights_only=True, map_location=DEVICE))
    test_auc,n_grp=eval_auc(m,bl_samples,bl_gts)
    results.append((pool, best_val, test_auc))
    print(f'  >>> {name}: best_val={best_val:.4f} test={test_auc:.4f}')

print('\n'+'='*50)
print('Summary:')
for pool, val, test in results:
    print(f'  GINx3_{pool:>10s}: val={val:.4f} test={test:.4f}')
