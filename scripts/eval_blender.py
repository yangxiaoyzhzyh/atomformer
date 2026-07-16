"""完整评估：清洗 + 去泄漏后的 blender 测试集"""
import os, json, requests, torch, torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

# ── 编码缺失组 ──
missing = ['radish', 'cucumber', 'hay', 'cabbage']
with open(os.path.join(BASE, 'data', 'odor_group_1024dim_cache.json')) as f: gd = json.load(f)
for g in missing:
    if g not in gd:
        resp = requests.post('http://localhost:1234/v1/embeddings',
            json={'model':'BGE-M3','input':g},timeout=30)
        gd[g] = resp.json()['data'][0]['embedding']
        print(f'Encoded: {g}')
with open(os.path.join(BASE, 'data', 'odor_group_1024dim_cache.json'),'w') as f:
    json.dump(gd, f)
print(f'Group cache: {len(gd)} groups')

gn = list(gd.keys()); gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1).to(DEVICE)
gi = {n:i for i,n in enumerate(gn)}

# ── 加载清洗后的训练集（用于泄漏检测）───
train_pairs = set()
for line in open(os.path.join(BASE, 'data', 'blender_train.jsonl')):
    r = json.loads(line)
    train_pairs.add((r['smiles_a'], r['smiles_b']))  # already canonical + sorted
print(f'Train pairs: {len(train_pairs)}')

# ── 加载清洗后的测试集（排除泄漏）───
class GINConv(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.mlp=torch.nn.Sequential(torch.nn.Linear(dim,dim),torch.nn.BatchNorm1d(dim),torch.nn.ReLU(),torch.nn.Linear(dim,dim),torch.nn.BatchNorm1d(dim),torch.nn.ReLU())
        self.eps=torch.nn.Parameter(torch.zeros(1))
    def forward(self,x,ei):
        r,c=ei;o=scatter(x[c],r,dim=0,dim_size=x.size(0),reduce='sum')
        return torch.relu(self.mlp((1+self.eps)*x+o))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad=torch.nn.ZeroPad2d((0,64-9,0,0))
        self.convs=torch.nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj=torch.nn.Linear(64,1024)
        enc=torch.nn.TransformerEncoderLayer(1024,8,batch_first=True,dropout=0.1)
        self.tfm=torch.nn.TransformerEncoder(enc,2)
    def forward(self,x,ei,batch):
        x=self.pad(x)
        for c in self.convs:x=c(x,ei)
        x=self.proj(x);x,m=to_dense_batch(x,batch)
        x=self.tfm(x,src_key_padding_mask=~m if m is not None else None)
        me=m.unsqueeze(-1).float()
        return F.normalize((x*me).sum(1)/me.sum(1).clamp(min=1),1)

_gc={}
def smi2g(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None:return None
    t=[6,7,8,9,16,17,35,53];c=[Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
    nf=[]
    for a in mol.GetAtoms():
        f=[1 if a.GetAtomicNum()==x else 0 for x in t]
        f+=[1 if a.GetChiralTag()==x else 0 for x in c]
        f+=[a.GetDegree()/5,a.GetFormalCharge()/5,1 if a.IsInRing() else 0,0];nf.append(f[:9])
    ei=[]
    for b in mol.GetBonds():i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();ei+=[[i,j],[j,i]]
    if not ei:ei=[[0,0]]
    return {'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
def build(smis):
    ax,aei=[],[];off=0
    for s in smis:
        if s not in _gc:
            g=smi2g(s)
            if g is None:return None
            _gc[s]=g
        g=_gc[s];x=torch.tensor(g['x']).float()[:,:9]
        ax.append(x);aei.append(torch.tensor(g['ei'])+off);off+=x.size(0)
    return Data(x=torch.cat(ax),edge_index=torch.cat(aei,dim=1))

samples,gts=[],[]
for line in open(os.path.join(BASE, 'data', 'blender_test.jsonl')):
    r=json.loads(line)
    g=r['odor_group'].split(',')[0].strip()
    d=build([r['smiles_a'],r['smiles_b']])
    if d:samples.append(d);gts.append(gi[g])

print(f'Clean test (from blender_test.jsonl): {len(samples)}')

batch=Batch.from_data_list(samples).to(DEVICE)
m=Model().to(DEVICE)
m.load_state_dict(torch.load(os.path.join(BASE, 'checkpoints', 'model_tgsc_bl_ep38.pt'),map_location=DEVICE),strict=False)
m.eval()

with torch.no_grad():
    z=m(batch.x,batch.edge_index,batch.batch)
cos=torch.mm(z.cpu(),gv.cpu().t()).numpy()
gt_np=np.array(gts)

# AUROC
y_true=np.zeros((len(gts),len(gn)))
for i,g in enumerate(gts):y_true[i,g]=1
aucs=[roc_auc_score(y_true[:,i],cos[:,i]) for i in range(len(gn)) if 0<y_true[:,i].sum()<len(gts)]
print(f'AUROC: {np.mean(aucs):.4f} ({len(aucs)}/{len(gn)} groups)')

# R@1/R@3
top3=cos.argsort(axis=1)[:,-3:][:,::-1]
n=len(samples)
r1=sum(1 for i,g in enumerate(gt_np) if top3[i,0]==g)/n*100
r3=sum(1 for i,g in enumerate(gt_np) if g in top3[i,:3])/n*100
print(f'R@1={r1:.1f}% R@3={r3:.1f}%')
