"""Per-group AUROC breakdown and train/test distribution comparison"""
import json, torch, torch.nn.functional as F, numpy as np, sys, os
from sklearn.metrics import roc_auc_score
from collections import Counter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_dense_batch
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cpu'

# Load group cache
with open(os.path.join(BASE, 'data', 'odor_group_1024dim_cache.json')) as f:
    gd = json.load(f)
gn = list(gd.keys())
gv = F.normalize(torch.tensor([gd[n] for n in gn]), dim=1)
gi = {n: i for i, n in enumerate(gn)}

# Train distribution
train_counts = Counter()
for line in open(os.path.join(BASE, 'data', 'blender_train.jsonl')):
    r = json.loads(line)
    for g in r['odor_group'].split(','):
        train_counts[g.strip()] += 1
total_train = sum(train_counts.values())

# Test data
test_counts = Counter()
test_data = []
for line in open(os.path.join(BASE, 'data', 'blender_test.jsonl')):
    r = json.loads(line)
    for g in r['odor_group'].split(','):
        test_counts[g.strip()] += 1
    test_data.append(r)
total_test = sum(test_counts.values())

# Model
class GINConv(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU())
        self.eps = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x, ei):
        r, c = ei; o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return torch.relu(self.mlp((1 + self.eps) * x + o))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ZeroPad2d((0, 64 - 9, 0, 0))
        self.convs = torch.nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = torch.nn.Linear(64, 1024)
        enc = torch.nn.TransformerEncoderLayer(1024, 8, batch_first=True, dropout=0.1)
        self.tfm = torch.nn.TransformerEncoder(enc, 2)
    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs: x = c(x, ei)
        x = self.proj(x); x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        me = m.unsqueeze(-1).float()
        return F.normalize((x * me).sum(1) / me.sum(1).clamp(min=1), 1)

# Build graphs
_gc = {}
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

def build(smis):
    ax, aei = [], []
    off = 0
    for s in smis:
        if s not in _gc:
            g = smi2g(s)
            if g is None: return None
            _gc[s] = g
        g = _gc[s]
        x = torch.tensor(g['x']).float()[:, :9]
        ax.append(x)
        aei.append(torch.tensor(g['ei']) + off)
        off += x.size(0)
    return Data(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))

# Inference
samples, gts = [], []
for r in test_data:
    g = r['odor_group'].split(',')[0].strip()
    d = build([r['smiles_a'], r['smiles_b']])
    if d:
        samples.append(d)
        gts.append(gi[g])

m = Model().to(DEVICE)
ckpt = os.path.join(BASE, 'checkpoints', 'model_tgsc_bl_ep38.pt')
m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
m.eval()

batch = Batch.from_data_list(samples).to(DEVICE)
with torch.no_grad():
    z = m(batch.x, batch.edge_index, batch.batch)
cos = torch.mm(z.cpu(), gv.cpu().t()).numpy()
gt_np = np.array(gts)

# Per-group AUROC
y_true = np.zeros((len(gts), len(gn)))
for i, g in enumerate(gts):
    y_true[i, g] = 1

results = []
for i, gname in enumerate(gn):
    n_pos = y_true[:, i].sum()
    if 0 < n_pos < len(gts):
        auc = roc_auc_score(y_true[:, i], cos[:, i])
        n_train = train_counts.get(gname, 0)
        n_test = int(n_pos)
        results.append((auc, n_train, n_test, gname))

results.sort(key=lambda x: x[0])

print('=' * 70)
print('Per-group AUROC (sorted ascending = worst first)')
print('=' * 70)
print(f"{'AUROC':>7} {'Train#':>8} {'Test#':>5}  Group")
print('-' * 45)
for auc, n_tr, n_te, gname in results[:15]:
    print(f'{auc:>7.4f} {n_tr:>8} {n_te:>5}  {gname}')
print('  ...')
for auc, n_tr, n_te, gname in results[-8:]:
    print(f'{auc:>7.4f} {n_tr:>8} {n_te:>5}  {gname}')

auc_vals = [r[0] for r in results]
print(f'\nMean AUROC: {np.mean(auc_vals):.4f} across {len(results)} groups')
print(f'Min: {results[0][0]:.4f} ({results[0][3]}, train#{results[0][1]})')
print(f'Max: {results[-1][0]:.4f} ({results[-1][3]}, train#{results[-1][1]})')

# Correlation
tr_counts_arr = [r[1] for r in results]
from scipy.stats import spearmanr
corr, p = spearmanr(tr_counts_arr, auc_vals)
print(f'\nSpearman (train count vs AUROC): r={corr:.4f}, p={p:.4e}')

# Split by frequency
print('\n--- By training frequency ---')
freq_bins = [(0, 100, 'rare (<100)'), (100, 1000, 'mid (100-1K)'), (1000, 9999999, 'common (>1K)')]
for lo, hi, label in freq_bins:
    subset = [r for r in results if lo <= r[1] < hi]
    if subset:
        print(f'  {label:15s}: mean AUROC={np.mean([r[0] for r in subset]):.4f} (n={len(subset)})')

# Distribution table
print('\n' + '=' * 70)
print('Distribution: Train vs Test (top 15 by test frequency)')
print('=' * 70)
print(f"{'Group':>20} {'Train%':>8} {'Test%':>8} {'Ratio(T/M)':>10}")
print('-' * 48)
for gname, _ in test_counts.most_common(15):
    tp = train_counts.get(gname, 0) / total_train * 100
    tep = test_counts[gname] / total_test * 100
    ratio = tep / tp if tp > 0 else 0
    print(f'{gname:>20} {tp:>7.3f}% {tep:>7.3f}% {ratio:>7.3f}')

# Spearman between train% and test%
train_pcts = {g: train_counts[g]/total_train*100 for g in test_counts}
test_pcts = {g: test_counts[g]/total_test*100 for g in test_counts}
common = [g for g in test_counts if g in train_counts]
if len(common) > 2:
    x_vals = [train_pcts[g] for g in common]
    y_vals = [test_pcts[g] for g in common]
    corr2, p2 = spearmanr(x_vals, y_vals)
    print(f'\nSpearman (train% vs test% across {len(common)} groups): r={corr2:.4f}, p={p2:.4e}')
