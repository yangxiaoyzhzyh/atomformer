"""Classification model GS test evaluation: macro-AUROC + first-word R@1

Supports two architectures:
  --model_type gin    : GINx5+Set2Set (model_gin_cls_best.pt)
  --model_type gin_tfm: GINx3+TFMx2+cls  (model_nosweet_cls135_auc_best.pt)
"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F, sys, re
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import Set2Set
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIST_LABELS = [
    'almond','amber','animal','anisic','apple','apricot','aromatic','balsamic',
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
N = len(MIST_LABELS)
L2I = {l: i for i, l in enumerate(MIST_LABELS)}

def text2y(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    y = torch.zeros(N)
    for w in words:
        if w in L2I: y[L2I[w]] = 1.0
    return y

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
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    if not ei: ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}


# ── Model architectures ──

class GINConv64(nn.Module):
    """GINConv (dim=64) for TFM variant"""
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
                                 nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei; o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))

class GINx5_Cls(nn.Module):
    """GINx5 (dim=200) + Set2Set + classifier"""
    def __init__(self):
        super().__init__()
        class GConv(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(),
                                         nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU())
                self.eps = nn.Parameter(torch.zeros(1))
            def forward(self, x, ei):
                r, c = ei; o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
                return F.relu(self.mlp((1 + self.eps) * x + o))
        self.pad = nn.ZeroPad2d((0, 200 - 9, 0, 0))
        self.convs = nn.ModuleList([GConv(200) for _ in range(5)])
        self.set2set = Set2Set(200, processing_steps=3)
        self.head = nn.Sequential(nn.Linear(400, 200), nn.ReLU(), nn.Linear(200, N))

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        g = self.set2set(x, batch)
        return self.head(g)

class GINx3_TFM_Cls(nn.Module):
    """GINx3 (dim=64) + TFMx2 + classifier"""
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 9, 0, 0))
        self.convs = nn.ModuleList([GINConv64(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
        enc = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True, dropout=0.1)
        self.tfm = nn.TransformerEncoder(enc, 2)
        self.classifier = nn.Linear(1024, N)

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x); x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        return self.classifier(x)


# ── Main ──
if __name__ == '__main__':
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'gin'
    assert model_type in ('gin', 'gin_tfm'), "Usage: python eval_cls_gs.py [gin|gin_tfm]"

    ckpt_map = {
        'gin': 'model_gin_cls_best.pt',
        'gin_tfm': 'model_nosweet_cls135_auc_best.pt',
    }
    ckpt = os.path.join(BASE, 'checkpoints', ckpt_map[model_type])
    print(f'Model: {model_type} | Checkpoint: {os.path.basename(ckpt)}')

    # Load model
    if model_type == 'gin':
        m = GINx5_Cls().to(DEVICE)
    else:
        m = GINx3_TFM_Cls().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
    m.eval()
    print(f'Params: {sum(p.numel() for p in m.parameters()):,}')

    # Load test data
    _gc = {}
    test_data = []
    for line in open(os.path.join(BASE, 'data', 'gs_test_nosweet.jsonl')):
        r = json.loads(line)
        y = text2y(r['text'])
        if y.sum() == 0: continue
        mol = Chem.MolFromSmiles(r['smiles'])
        if mol is None: continue
        g = smi2g(r['smiles'])
        if g is None: continue
        x = torch.tensor(g['x']).float()[:, :9]
        d = Data(x=x, edge_index=torch.tensor(g['ei']))
        d.y_gt = y; d.gt_text = r['text']
        test_data.append(d)
    print(f'Test samples: {len(test_data)}')

    # Inference
    all_y_true, all_y_score = [], []
    for d in test_data:
        with torch.no_grad():
            logits = m(d.x.to(DEVICE), d.edge_index.to(DEVICE),
                       torch.zeros(d.x.size(0)).long().to(DEVICE))
        all_y_true.append(d.y_gt.numpy())
        all_y_score.append(torch.sigmoid(logits.cpu()).numpy()[0])
    y_true = np.array(all_y_true)
    y_score = np.array(all_y_score)

    # macro-AUROC
    aucs = []
    for i in range(N):
        s = y_true[:, i].sum()
        if 0 < s < len(y_true):
            try: aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
            except: pass
    macro_auc = np.mean(aucs) if aucs else 0.0
    print(f'macro-AUROC: {macro_auc:.4f} ({len(aucs)}/{N} classes)')

    # first-word R@1
    h1 = 0
    for i, d in enumerate(test_data):
        gt_first = d.gt_text.strip().split()[0].strip(',;.')
        pred_top5 = [MIST_LABELS[j] for j in np.argsort(y_score[i])[-5:][::-1]]
        if pred_top5[0] == gt_first:
            h1 += 1
    r1 = h1 / len(test_data) * 100
    print(f'R@1: {r1:.1f}% ({h1}/{len(test_data)})')
