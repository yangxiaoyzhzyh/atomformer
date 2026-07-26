"""验证冲突对：模型z是否落在标签中点附近——用训练数据直接测"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F, random
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, 'data', 'odor_group_1024dim_cache.json')) as f: gd=json.load(f)
gn=list(gd.keys());gv=F.normalize(torch.tensor([gd[n] for n in gn]),dim=1)
gi={n:i for i,n in enumerate(gn)}

# 收集训练数据中的冲突对
pair_tags=defaultdict(set)
for line in open(os.path.join(BASE, 'data', 'blender_train.jsonl')):
    r=json.loads(line)
    og=r['odor_group'].split(',')[0].strip()
    if og not in gi:continue
    pair_tags[tuple(sorted([r['smiles_a'],r['smiles_b']]))].add(og)

conflicts=[(k,v) for k,v in pair_tags.items() if len(v)>=2]
random.seed(42);random.shuffle(conflicts)
print(f'冲突对总数: {len(conflicts)}')

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
    return np.array(nf,dtype=np.float32),np.array(ei,dtype=np.int64).T
def build_p(a,b):
    xa,ea=smi2g(a);xb,eb=smi2g(b)
    if xa is None or xb is None:return None
    x=np.concatenate([xa,xb]);ei=np.concatenate([ea,eb+xa.shape[0]],axis=1)
    return Data(x=torch.tensor(x).float()[:,:9],edge_index=torch.tensor(ei))

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
        enc=nn.TransformerEncoderLayer(1024,8,batch_first=True,dropout=0.1)
        self.tfm=nn.TransformerEncoder(enc,2)
    def forward(self,x,ei,batch):
        x=self.pad(x)
        for c in self.convs:x=c(x,ei)
        x=self.proj(x);x,m=to_dense_batch(x,batch)
        x=self.tfm(x,src_key_padding_mask=(~m if m is not None else None))
        me=m.unsqueeze(-1).float()
        return F.normalize((x*me).sum(1)/me.sum(1).clamp(min=1),1)

m=Model()
m.load_state_dict(torch.load(os.path.join(BASE, 'checkpoints', 'model_tgsc_bl_ep38.pt'),map_location='cpu'),strict=False)
m.eval()

c_ab,c_ba,c_mid,c_abba=[],[],[],[]
cnt=0
for smis,ogs in conflicts:
    tags=list(ogs)
    ga=tags[0];gb=tags[1]
    a,b=smis
    d=build_p(a,b)
    if d is None:continue
    ba=Batch.from_data_list([d])
    with torch.no_grad():z=m(ba.x,ba.edge_index,ba.batch)
    va=F.normalize(gv[gi[ga]].unsqueeze(0),dim=1)
    vb=F.normalize(gv[gi[gb]].unsqueeze(0),dim=1)
    mid=F.normalize((va+vb)/2,dim=1)
    c_ab.append(F.cosine_similarity(z,va).item())
    c_ba.append(F.cosine_similarity(z,vb).item())
    c_mid.append(F.cosine_similarity(z,mid).item())
    c_abba.append(F.cosine_similarity(va,vb).item())
    cnt+=1
    if cnt>=1000:break

print(f'\n测试: {cnt} 个冲突对')
labels=['z -> labelA','z -> labelB','z -> midpoint','labelA <-> labelB']
data=[c_ab,c_ba,c_mid,c_abba]
print(f'{"指标":20s} | {"mean":6s} | {"median":6s} | {">0.5":5s} | {">0.7":5s}')
print('-'*50)
def s(arr):
    return f'{np.mean(arr):.4f} | {np.median(arr):.4f} | {sum(1 for x in arr if x>0.5)/len(arr)*100:.0f}% | {sum(1 for x in arr if x>0.7)/len(arr)*100:.0f}%'
for l,d in zip(labels,data):
    print(f'{l:20s} | {s(d)}')
bet=sum(1 for i in range(cnt) if c_mid[i] > max(c_ab[i],c_ba[i]))
print(f'\nz到中点 > z到两端任一: {bet}/{cnt} = {bet/cnt*100:.1f}%')
bet2=sum(1 for i in range(cnt) if c_mid[i] > (c_ab[i]+c_ba[i])/2)
print(f'z到中点 > z到两端平均: {bet2}/{cnt} = {bet2/cnt*100:.1f}%')
