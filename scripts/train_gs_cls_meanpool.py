"""GINx3 + TFMx2 + masked mean pool + BCE, GS monomers only"""
import sys,os,time,random,json,torch,re
import torch.nn as nn,torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter,to_dense_batch
from rdkit import Chem,RDLogger;RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

torch.manual_seed(42);random.seed(42)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

MIST_LABELS=['almond','amber','animal','anisic','apple','apricot','aromatic','balsamic',
    'banana','beefy','bergamot','berry','bitter','black currant','brandy','burnt',
    'buttery','cabbage','camphoreous','caramellic','cedar','celery','chamomile',
    'cheesy','cherry','chocolate','cinnamon','citrus','clean','clove','cocoa',
    'coconut','coffee','cognac','cooked','cooling','cortex','coumarinic','creamy',
    'cucumber','dairy','dry','earthy','ethereal','fatty','fermented','fishy',
    'floral','fresh','fruit skin','fruity','garlic','gassy','geranium','grape',
    'grapefruit','grassy','green','hawthorn','hay','hazelnut','herbal','honey',
    'hyacinth','jasmin','juicy','ketonic','lactonic','lavender','leafy','leathery',
    'lemon','lily','malty','meaty','medicinal','melon','metallic','milky','mint',
    'muguet','mushroom','musk','musty','natural','nutty','odorless','oily','onion',
    'orange','orangeflower','orris','ozone','peach','pear','phenolic','pine',
    'pineapple','plum','popcorn','potato','powdery','pungent','radish','raspberry',
    'ripe','roasted','rose','rummy','sandalwood','savory','sharp','smoky','soapy',
    'solvent','sour','spicy','strawberry','sulfurous','sweaty','sweet','tea',
    'terpenic','tobacco','tomato','tropical','vanilla','vegetable','vetiver',
    'violet','warm','waxy','weedy','winey','woody']
N=len(MIST_LABELS);L2I={l:i for i,l in enumerate(MIST_LABELS)}
print(f'Classes: {N}')

def text2y(text):
    words=set(re.findall(r'[a-z]+',text.lower()));y=torch.zeros(N)
    for w in words:
        if w in L2I:y[L2I[w]]=1.0
    return y if y.sum()>0 else None

class GINConv(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(d,d),nn.BatchNorm1d(d),nn.ReLU(),nn.Linear(d,d),nn.BatchNorm1d(d),nn.ReLU())
        self.eps=nn.Parameter(torch.zeros(1))
    def forward(self,x,ei):
        r,c=ei;o=scatter(x[c],r,dim=0,dim_size=x.size(0),reduce='sum')
        return F.relu(self.mlp((1+self.eps)*x+o))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad=nn.ZeroPad2d((0,64-9,0,0))
        self.convs=nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj=nn.Linear(64,1024)
        enc=nn.TransformerEncoderLayer(d_model=1024,nhead=8,batch_first=True,dropout=0.1)
        self.tfm=nn.TransformerEncoder(enc,2)
        self.classifier=nn.Linear(1024,N)
    def forward(self,x,ei,batch):
        x=self.pad(x)
        for c in self.convs:x=c(x,ei)
        x=self.proj(x);x,m=to_dense_batch(x,batch)
        x=self.tfm(x,src_key_padding_mask=~m if m is not None else None)
        me=m.unsqueeze(-1).float()
        x=(x*me).sum(dim=1)/me.sum(dim=1).clamp(min=1)
        return self.classifier(x)

class GData(Data):
    def __cat_dim__(self,key,value,*a,**kw):
        if key in ('y',):return None
        return super().__cat_dim__(key,value,*a,**kw)

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

_gc={}
def build(smi):
    if smi not in _gc:
        g=smi2g(smi);_gc[smi]=g if g else None
    if _gc[smi] is None:return None
    g=_gc[smi];x=torch.tensor(g['x']).float()[:,:9]
    return GData(x=x,edge_index=torch.tensor(g['ei']))

BASE=os.path.dirname(os.path.abspath(__file__))
print('Loading data...')
dt=lambda p:os.path.join(BASE,'data/processed/goodscents',p)
train_data=[]
for line in open(dt('gs_train_nosweet.jsonl')):
    r=json.loads(line);y=text2y(r['text'])
    if y is None:continue
    d=build(r['smiles'])
    if d:d.y=y;train_data.append(d)
print(f'Train: {len(train_data)}')
ev_data=[]
for line in open(dt('gs_test_nosweet.jsonl')):
    r=json.loads(line);y=text2y(r['text'])
    if y is None:continue
    d=build(r['smiles'])
    if d:d.y=y;ev_data.append(d)
print(f'Test: {len(ev_data)}')
loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)

m=Model().to(DEVICE)
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
opt=torch.optim.Adam(m.parameters(),lr=1e-3)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)

for ep in range(30):
    m.train();tl=0
    for b in loader:
        b=b.to(DEVICE);z=m(b.x,b.edge_index,b.batch)
        loss=F.binary_cross_entropy_with_logits(z,b.y.float())
        opt.zero_grad();loss.backward();opt.step();tl+=loss.item()
    sch.step();al=tl/len(loader)
    m.eval()
    with torch.no_grad():
        all_z,all_y=[],[]
        for d in ev_data:
            z=m(d.x.to(DEVICE),d.edge_index.to(DEVICE),torch.zeros(d.x.size(0)).long().to(DEVICE))
            all_z.append(z.cpu());all_y.append(d.y)
        all_z=torch.cat(all_z).numpy();all_y=torch.stack(all_y).numpy()
    aucs=[roc_auc_score(all_y[:,i],all_z[:,i]) for i in range(N) if 0<all_y[:,i].sum()<len(all_y)]
    print(f'Ep{ep:2d} | loss={al:.6f} | AUROC={np.mean(aucs):.4f} ({len(aucs)}/{N})')
