"""单体 conR@K 评估（温度加权 T=0.1）"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GINConv(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d,d),nn.BatchNorm1d(d),nn.ReLU(),nn.Linear(d,d),nn.BatchNorm1d(d),nn.ReLU())
        self.eps=nn.Parameter(torch.zeros(1))
    def forward(self,x,ei):
        r,c=ei;o=scatter(x[c],r,dim=0,dim_size=x.size(0),reduce='sum')
        return torch.relu(self.mlp((1+self.eps)*x+o))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad=nn.ZeroPad2d((0,64-9,0,0))
        self.convs=nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj=nn.Linear(64,1024)
        enc=nn.TransformerEncoderLayer(d_model=1024,nhead=8,batch_first=True,dropout=0.1)
        self.tfm=nn.TransformerEncoder(enc,2)
    def forward(self,x,ei,batch):
        x=self.pad(x)
        for c in self.convs:x=c(x,ei)
        x=self.proj(x);x,m=to_dense_batch(x,batch)
        x=self.tfm(x,src_key_padding_mask=~m if m is not None else None)
        me=m.unsqueeze(-1).float()
        return F.normalize((x*me).sum(dim=1)/me.sum(dim=1).clamp(min=1),dim=1)

def fw(t): return t.strip().split()[0].strip(',;.')

# Load data
test_data=[json.loads(l) for l in open(os.path.join(BASE, 'data', 'tgsc_test_bge.jsonl'))]
lib_texts=[];lib_embs=[]
for line in open(os.path.join(BASE, 'data', 'tgsc_train_bge.jsonl')):
    r=json.loads(line);lib_texts.append(r['description']);lib_embs.append(r['embedding'])
lib_embs=F.normalize(torch.tensor(lib_embs),dim=1)
print(f'Test: {len(test_data)}, Lib: {len(lib_texts)}')

_gc={}
def build(smi):
    if smi not in _gc:
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
        _gc[smi]={'x':np.array(nf,dtype=np.float32),'ei':np.array(ei,dtype=np.int64).T}
    g=_gc[smi];x=torch.tensor(g['x']).float()[:,:9]
    return Data(x=x,edge_index=torch.tensor(g['ei']))

samples=[(build(r['smiles']),r['description']) for r in test_data]
samples=[(d,desc) for d,desc in samples if d is not None]
print(f'Graphs: {len(samples)}')

# Load model
m=Model().to(DEVICE)
m.load_state_dict(torch.load(os.path.join(BASE, 'checkpoints', 'model_tgsc_bl_ep38.pt'),map_location=DEVICE,weights_only=True),strict=False)
m.eval()

h1=h3=0; ch1=ch3=0
for d,desc in samples:
    with torch.no_grad():
        z=m(d.x.to(DEVICE),d.edge_index.to(DEVICE),torch.zeros(d.x.size(0)).long().to(DEVICE))
    cos=torch.mm(z.cpu(),lib_embs.t())
    vals,idxs=cos[0].topk(100)
    # ── conR: temperature-weighted consensus ──
    w=F.softmax(vals/0.1,dim=0).numpy()
    wc={}
    for j,wt in zip(idxs.tolist(),w):
        wd=fw(lib_texts[j]);wc[wd]=wc.get(wd,0)+wt
    ranked=sorted(wc.items(),key=lambda x:-x[1])
    preds=[r[0] for r in ranked[:3]]
    gt=fw(desc)
    if preds and preds[0]==gt:h1+=1
    if gt in preds[:3]:h3+=1
    # ── cosR: raw cosine retrieval ──
    cos_idxs = cos[0].topk(3).indices
    cos_preds = [fw(lib_texts[j]) for j in cos_idxs.tolist()]
    if cos_preds and cos_preds[0]==gt: ch1+=1
    if gt in cos_preds: ch3+=1
n=len(samples)
print(f'conR@1={h1/n*100:.1f}% conR@3={h3/n*100:.1f}% ({h1}/{h3}/{n})')
print(f'cosR@1={ch1/n*100:.1f}% cosR@3={ch3/n*100:.1f}% ({ch1}/{ch3}/{n})')
