"""135 标签多标签分类 (BCE), 去 sweet 版
输出 135 维 logits (支持任意 N 分子输入)
"""
import sys, os, time, random, json, torch, re
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42); random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {DEVICE}')

# ── 135 标签 ──
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
print(f'标签: {N} 维')

def text2y(text):
    words = set(re.findall(r'[a-z]+', text.lower()))
    y = torch.zeros(N)
    for w in words:
        if w in L2I: y[L2I[w]] = 1.0
    return y

# ── GINConv ──
class GINConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))

# ── 模型（输出 135 维 logits）───
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - 9, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, 1024)
        enc = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True, dropout=0.1)
        self.tfm = nn.TransformerEncoder(enc, 2)
        self.classifier = nn.Linear(1024, N)

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        x = (x * me).sum(dim=1) / me.sum(dim=1).clamp(min=1)
        return self.classifier(x)

# ── 图构建 ──
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

_gc = {}

class GData(Data):
    def __cat_dim__(self, key, value, *a, **kw):
        if key in ('y',): return None
        return super().__cat_dim__(key, value, *a, **kw)

def build_d(smis, target=None):
    ax, aei = [], []
    off = 0
    for s in smis:
        if s not in _gc:
            g = smi2g(s)
            if g is None: return None
            _gc[s] = g
        g = _gc[s]
        x = torch.tensor(g['x']).float()
        if x.size(1) > 9: x = x[:, :9]
        ax.append(x)
        aei.append(torch.tensor(g['ei']) + off)
        off += x.size(0)
    d = GData(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))
    if target is not None: d.y = target
    return d

# ── 加载数据 ──
print('加载数据...')
all_data = []

with open(BASE + '/data/processed/goodscents/gs_train_nosweet.jsonl') as f:
    for line in f:
        r = json.loads(line)
        y = text2y(r['text'])
        if y.sum() == 0: continue
        d = build_d([r['smiles']], y)
        if d: all_data.append((d, 1))
print(f'单体: {sum(1 for _, m in all_data if m)}')

with open(BASE + '/data/processed/blender/blender_train_nosweet.jsonl') as f:
    for line in f:
        r = json.loads(line)
        y = text2y(r['odor_group'])
        if y.sum() == 0: continue
        d = build_d([r['smiles_a'], r['smiles_b']], y)
        if d: all_data.append((d, 0))
print(f'Blender: {sum(1 for _, m in all_data if not m)}')

tw = [100 if m else 1 for _, m in all_data]
sampler = torch.utils.data.WeightedRandomSampler(tw, len(all_data), replacement=True)
loader = DataLoader([d for d, _ in all_data], batch_size=1024, sampler=sampler, num_workers=0)
print(f'总计: {len(all_data)}')

# ── 测试集 ──
ev_data = []
with open(BASE + '/data/processed/goodscents/gs_test_nosweet.jsonl') as f:
    for line in f:
        r = json.loads(line)
        y = text2y(r['text'])
        if y.sum() == 0: continue
        d = build_d([r['smiles']])
        if d: d.y_gt = y; d.gt_text = r['text']; ev_data.append(d)
print(f'测试集: {len(ev_data)}')

def evaluate(m):
    m.eval()
    all_y_true, all_y_score = [], []
    for d in ev_data:
        with torch.no_grad():
            logits = m(d.x.to(DEVICE), d.edge_index.to(DEVICE),
                       torch.zeros(d.x.size(0)).long().to(DEVICE))
        all_y_true.append(d.y_gt.numpy())
        all_y_score.append(torch.sigmoid(logits.cpu()).numpy()[0])
    y_true = np.array(all_y_true)
    y_score = np.array(all_y_score)
    # macro-AUC（跳过纯0/纯1列）
    aucs = []
    for i in range(N):
        s = y_true[:, i].sum()
        if 0 < s < len(y_true):
            try: aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
            except: pass
    macro_auc = np.mean(aucs) if aucs else 0.0
    # 首词 R@1（取 sigmoid top-1 标签）
    h1 = 0
    for i, d in enumerate(ev_data):
        gt_first = d.gt_text.strip().split()[0].strip(',;.')
        pred_labels = [MIST_LABELS[j] for j in np.argsort(y_score[i])[-5:][::-1]]
        if any(fl == gt_first for fl in [l.split()[0] for l in pred_labels[:1]]):
            h1 += 1
    r1 = h1 / len(ev_data) * 100
    return macro_auc, r1

# ── 训练 ──
m = Model().to(DEVICE)
print(f'参数: {sum(p.numel() for p in m.parameters()):,}')

opt = torch.optim.Adam(m.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

best_auc, best_r1 = 0, 0
t0 = time.time()
print(f'\nepoch | loss     | macro-AUC | R@1    | time  |')
print('-' * 45)
for ep in range(30):
    m.train(); tl = 0
    for b in loader:
        b = b.to(DEVICE)
        logits = m(b.x, b.edge_index, b.batch)
        loss = F.binary_cross_entropy_with_logits(logits, b.y.float())
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()
    sch.step()
    al = tl / len(loader)
    auc, r1 = evaluate(m)
    flag = ''
    if r1 > best_r1:
        best_r1 = r1
        torch.save(m.state_dict(), BASE + '/checkpoints/model_nosweet_cls135_best.pt')
        flag = '★'
    if auc > best_auc:
        best_auc = auc
        torch.save(m.state_dict(), BASE + '/checkpoints/model_nosweet_cls135_auc_best.pt')
    print(f'ep{ep:2d}  | {al:.6f} | {auc:.4f}    | {r1:.1f}%  | {int(time.time()-t0)}s | {flag}')

torch.save(m.state_dict(), BASE + '/checkpoints/model_nosweet_cls135_final.pt')
print(f'\n完成! best R@1={best_r1:.1f}%, best AUC={best_auc:.4f}')
