# -*- coding: utf-8 -*-
"""train_sisson_knee_final.py 

模型：AtomFormer-cls74（wm-cls74）原架构 train_sisson_atomformer_cls.py
      pad(31→64) → GIN×3(64) → proj(256) → 跨分子 TFM×2(nhead 4) → 全体原子 masked-mean
      → Linear(256→74)；损失 74-way BCEWithLogits；指标 macro-AUROC
      数据 trainpairs74/testpairs74（Sisson 分子级 novel-molecule 切分，test 从不参与选择）
      val：pair 级从 train 随机切 1000 对（seed 42），与旧脚本完全同一切分

选模策略（本脚本唯一的策略）：KneeStopper 拐点早停
      val → headroom 变换 h = −log10(1−val)（把饱和段拉直）
          → 因果移动平均（窗口 window）
          → 一阶导 d1 = h_s[t] − h_s[t−1]
          → 当 d1 < ratio × d1_max 连续 streak 个 epoch 且已达 min_epochs
            → 判定拐点，立即停止训练
      → 回载 [0, knee] 区间内 val 最高的权重（= 早停 + best-val 回载）
      → 该权重只评估 test 一次（test 不参与任何选择）


用法：
    python train_sisson_knee_final.py --seeds 42,43,44        # 正式 3 种子
    python train_sisson_knee_final.py --limit 3000 --epochs 6 # 冒烟
    python train_sisson_knee_final.py --eval-only results/xxx.pt  # 回评任意 ckpt
"""
import argparse, os, sys, csv, math, random, time, types as _types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, scatter
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('--odor-pair-dir', default=None,
                help='含 pairing/data.py 的 odor-pair 目录（本地没有则自动 stub 一个最小 PairData）')
ap.add_argument('--pairing-dir', default=None,
                help='数据目录（默认 <脚本目录>/../data，其次 ../dataset）')
ap.add_argument('--suffix', default='74', help="数据文件名后缀，'74'=trainpairs74.pt")
# ── 训练超参（与旧脚本逐行一致）──
ap.add_argument('--epochs', type=int, default=60, help='训练预算上限（拐点通常会提前停止）')
ap.add_argument('--bsz', type=int, default=64)
ap.add_argument('--n-val', type=int, default=1000)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--wd', type=float, default=0.0)
ap.add_argument('--dim', type=int, default=256)
ap.add_argument('--nhead', type=int, default=4)
ap.add_argument('--n-layers', type=int, default=2)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--warmup', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--seeds', default=None, help='多种子，逗号分隔，如 42,43,44（数据只建一次）')
ap.add_argument('--limit', type=int, default=0, help='冒烟：仅取前 N 对 train 与 N//4 对 test')
ap.add_argument('--device', default=None)
ap.add_argument('--tag', default='')
# ── 拐点早停（唯一策略；只读 val）──
ap.add_argument('--knee-window', type=int, default=5, help='因果移动平均窗口')
ap.add_argument('--knee-ratio', type=float, default=0.1, help='一阶导阈值（相对早期峰值斜率）')
ap.add_argument('--knee-streak', type=int, default=2, help='需连续满足的 epoch 数')
ap.add_argument('--min-epochs', type=int, default=10, help='至少训练这么多 epoch 才允许早停')
ap.add_argument('--eval-every', type=int, default=1, help='每 N 个 epoch 跑一次 val')
ap.add_argument('--eval-only', default=None, help='给定 ckpt：只评估 val/test，不训练')
ap.add_argument('--out-dir', default=None, help='结果输出目录（默认 <脚本目录>/../results）')
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = args.out_dir or os.path.join(HERE, '..', 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))


def _resolve_pairing():
    cands = [args.pairing_dir,
             os.path.join(HERE, '..', 'data'),
             os.path.join(HERE, 'data'),
             os.path.join(HERE, '..', 'dataset')]
    if args.odor_pair_dir:
        cands += [os.path.join(args.odor_pair_dir, f'pairing{args.suffix}'),
                  os.path.join(args.odor_pair_dir, 'pairing')]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, f'trainpairs{args.suffix}.pt')):
            return os.path.abspath(c)
    raise SystemExit('找不到数据目录：请用 --pairing-dir 指定含 '
                     f'trainpairs{args.suffix}.pt 的目录')


PAIRING = _resolve_pairing()
print(f'Device: {DEVICE} | pairing: {PAIRING} | out: {OUT}', flush=True)

# ── 注册 PairData（有 odor-pair 就用真的，没有就 stub，unpickle 只需要类名）──
if args.odor_pair_dir:
    sys.path.insert(0, args.odor_pair_dir)
    try:
        import pairing.data  # noqa: F401
        print('pairing.data: imported', flush=True)
    except Exception as e:
        _pkg = sys.modules.setdefault('pairing', _types.ModuleType('pairing'))
        if not hasattr(_pkg, '__path__'):
            _pkg.__path__ = [args.odor_pair_dir]
        _mod = _types.ModuleType('pairing.data')

        class PairData(Data):
            pass

        _mod.PairData = PairData
        sys.modules['pairing.data'] = _mod
        print(f'pairing.data: stub ({type(e).__name__}: {e})', flush=True)
else:
    _pkg = sys.modules.setdefault('pairing', _types.ModuleType('pairing'))
    _pkg.__path__ = []
    _mod = _types.ModuleType('pairing.data')

    class PairData(Data):
        pass

    _mod.PairData = PairData
    sys.modules['pairing.data'] = _mod
    print('pairing.data: stub (未提供 --odor-pair-dir)', flush=True)

with open(os.path.join(PAIRING, f'notes{args.suffix}.csv'), encoding='utf-8') as f:
    NOTES = [x.strip() for x in f.read().split(',') if x.strip()]
N = len(NOTES)
print(f'Sisson classes: {N}', flush=True)

# ═══════════ 图构建（与旧脚本逐行一致：双分子拼接、无边界）═══════════
FEAT_DIM = 31
_gc = {}


def smi2g(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    nf, ei = [], []
    for a in mol.GetAtoms():
        nf.append([a.GetAtomicNum() == 6, a.GetAtomicNum() == 7, a.GetAtomicNum() == 8,
                   a.GetAtomicNum() == 9, a.GetAtomicNum() == 15, a.GetAtomicNum() == 16,
                   a.GetAtomicNum() == 17, a.GetAtomicNum() == 35, a.GetAtomicNum() == 53,
                   a.GetFormalCharge() == 1, a.GetFormalCharge() == -1,
                   a.GetHybridization() == Chem.HybridizationType.SP,
                   a.GetHybridization() == Chem.HybridizationType.SP2,
                   a.GetHybridization() == Chem.HybridizationType.SP3,
                   a.GetIsAromatic(),
                   a.GetDegree() == 1, a.GetDegree() == 2, a.GetDegree() == 3, a.GetDegree() == 4,
                   a.GetTotalNumHs() == 0, a.GetTotalNumHs() == 1, a.GetTotalNumHs() == 2,
                   a.GetTotalNumHs() == 3, a.IsInRing(),
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]) > 0,
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]) > 0,
                   len([b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.AROMATIC]) > 0,
                   a.GetMass() > 30, a.GetMass() > 60, a.GetMass() > 90, a.GetMass() > 120])
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        ei.append([i, j]); ei.append([j, i])
    if not nf:
        return None
    return {'x': np.array(nf, dtype=np.float32), 'ei': np.array(ei, dtype=np.int64).T}


def build_pair(sm_a, sm_b):
    ax, aei, off = [], [], 0
    for s in (sm_a, sm_b):
        if s not in _gc:
            g = smi2g(s)
            _gc[s] = g if g else None
        g = _gc[s]
        if g is None:
            return None
        ax.append(torch.tensor(g['x']).float()[:, :FEAT_DIM])
        aei.append(torch.tensor(g['ei']) + off)
        off += g['x'].shape[0]
    return Data(x=torch.cat(ax), edge_index=torch.cat(aei, dim=1))


def load_pairs(fname):
    data, slices = torch.load(fname, weights_only=False)
    n = data.y.size(0)
    assert data.y.size(1) == N, f'class mismatch: pt has {data.y.size(1)}, notes.csv has {N}'
    out = []
    for i in range(n):
        y = data.y[i].float()
        if y.sum() > 0:
            out.append((data.smiles_s[i], data.smiles_t[i], y))
    return out


print(f'Loading trainpairs{args.suffix}.pt ...', flush=True)
train_raw = load_pairs(os.path.join(PAIRING, f'trainpairs{args.suffix}.pt'))
print(f'  train pairs: {len(train_raw)}', flush=True)
test_raw = load_pairs(os.path.join(PAIRING, f'testpairs{args.suffix}.pt'))
print(f'  test pairs : {len(test_raw)}', flush=True)

if args.limit > 0:
    train_raw = train_raw[:args.limit]
    test_raw = test_raw[:max(1, args.limit // 4)]
    print(f'[smoke] limited to train {len(train_raw)} / test {len(test_raw)}', flush=True)

# val 切分：（pair 级随机 1000 对，seed 42）
idx = list(range(len(train_raw)))
random.Random(42).shuffle(idx)
n_val = min(args.n_val, max(1, len(train_raw) // 2))
va_rows = [train_raw[i] for i in idx[:n_val]]
tr_rows = [train_raw[i] for i in idx[n_val:]]
print(f'split(pair): train {len(tr_rows)} | val {len(va_rows)} | test {len(test_raw)}', flush=True)


def make_graphs(rows):
    out = []
    for sa, sb, y in rows:
        d = build_pair(sa, sb)
        if d is not None:
            d.lbl = y.unsqueeze(0)
            out.append(d)
    return out


print('Building graphs ...', flush=True)
tr_d = make_graphs(tr_rows); va_d = make_graphs(va_rows); te_d = make_graphs(test_raw)
print(f'graphs: train {len(tr_d)} | val {len(va_d)} | test {len(te_d)}', flush=True)
va_loader = DataLoader(va_d, batch_size=256)
te_loader = DataLoader(te_d, batch_size=256)


# ═══════════ 模型═══════════
class GINConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return F.relu(self.mlp((1 + self.eps) * x + o))


class Model(nn.Module):
    def __init__(self, dim=256, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 64 - FEAT_DIM, 0, 0))
        self.convs = nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = nn.Linear(64, dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=2 * dim,
                                         batch_first=True, dropout=dropout)
        self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(dim, N)

    def forward(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        x = self.tfm(x, src_key_padding_mask=~m if m is not None else None)
        x = x[m]
        x = scatter(x, batch, dim=0, reduce='mean')
        return self.head(x)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    zs, ys = [], []
    for b in loader:
        b = b.to(DEVICE)
        zs.append(model(b.x, b.edge_index, b.batch).cpu())
        ys.append(b.lbl.cpu())
    Z = torch.cat(zs).numpy(); Y = torch.cat(ys).numpy()
    aucs = [roc_auc_score(Y[:, i], Z[:, i]) for i in range(N)
            if 0 < Y[:, i].sum() < len(Y)]
    return float(np.mean(aucs)), len(aucs)


def clone_sd(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


m = Model(dim=args.dim, nhead=args.nhead, n_layers=args.n_layers, dropout=args.dropout).to(DEVICE)
n_params = sum(p.numel() for p in m.parameters())
print(f'Params: {n_params:,}', flush=True)

if args.eval_only:
    m.load_state_dict(torch.load(args.eval_only, weights_only=True, map_location=DEVICE))
    v_auc, _ = evaluate(m, va_loader)
    t_auc, n_auc = evaluate(m, te_loader)
    print(f'[eval-only] {args.eval_only}\n  val {v_auc:.4f} | TEST {t_auc:.4f} ({n_auc}/{N} classes)',
          flush=True)
    sys.exit(0)


# ═══════════ 拐点早停═══════════
class KneeStopper:
    """val → headroom 变换 h=-log10(1-val) → 因果 MA(window) → 一阶导 d1。
    当 d1 < ratio × d1_max 连续 streak 个 epoch（且已达 min_epochs）→ 判定拐点。
    与 knee_detect.py 的 d1 准则同构，保证在线/离线判定一致。"""

    def __init__(self, window, ratio, streak, min_epochs, eps=1e-6):
        self.w, self.ratio, self.streak = window, ratio, streak
        self.min_epochs, self.eps = min_epochs, eps
        self.hist, self.prev_hs, self.d1_max = [], None, 0.0
        self.bad, self.knee = 0, None

    def step(self, ep, val):
        h = -math.log10(max(self.eps, 1.0 - val))
        self.hist.append(h)
        hs = float(np.mean(self.hist[-self.w:]))
        d1 = 0.0 if self.prev_hs is None else hs - self.prev_hs
        self.prev_hs = hs
        self.d1_max = max(self.d1_max, d1)
        if ep < self.min_epochs:
            return False, d1, self.d1_max
        if d1 < self.ratio * self.d1_max:
            self.bad += 1
            if self.bad >= self.streak and self.knee is None:
                self.knee = ep
                return True, d1, self.d1_max
        else:
            self.bad = 0
        return False, d1, self.d1_max


def run_one(seed, tag):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    mm = Model(dim=args.dim, nhead=args.nhead, n_layers=args.n_layers,
               dropout=args.dropout).to(DEVICE)
    tr_loader = DataLoader(tr_d, batch_size=args.bsz, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(mm.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - args.warmup))
    WARMUP = args.warmup
    stopper = KneeStopper(args.knee_window, args.knee_ratio, args.knee_streak, args.min_epochs)

    curve_path = os.path.join(OUT, f'curve_{tag}.csv')
    _f = open(curve_path, 'w', newline='', encoding='utf-8')
    _w = csv.DictWriter(_f, fieldnames=['ep', 'lr', 'loss', 'val', 'd1', 'd1_peak', 'sec'])
    _w.writeheader()

    t0 = time.time()
    best_val, best_ep, best_sd = -1.0, -1, None
    knee_ep, stop_ep = None, None

    for ep in range(args.epochs):
        if ep < WARMUP:
            for pg in opt.param_groups:
                pg['lr'] = args.lr * (ep + 1) / max(1, WARMUP)

        mm.train(); tl = 0.0
        for b in tr_loader:
            b = b.to(DEVICE)
            z = mm(b.x, b.edge_index, b.batch)
            loss = F.binary_cross_entropy_with_logits(z, b.lbl)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item()
        if ep >= WARMUP:
            sch.step()

        do_val = (ep % args.eval_every == 0) or (ep == args.epochs - 1)
        if do_val:
            vauc, _ = evaluate(mm, va_loader)
            if vauc > best_val:
                best_val, best_ep = vauc, ep
                best_sd = clone_sd(mm)
        else:
            vauc = float('nan')

        d1 = d1p = ''
        if do_val:
            stopped, d1, d1p = stopper.step(ep, vauc)
            if stopped:
                knee_ep = ep
                stop_ep = ep
                print(f'>> [{tag}] 拐点触发：停于 ep{ep}，回载 [0,{ep}] 内 best-val ckpt ep{best_ep} '
                      f'(val {best_val:.4f}, lr {opt.param_groups[0]["lr"]:.2e})', flush=True)

        msg = (f'[{tag}] ep {ep:3d} | lr {opt.param_groups[0]["lr"]:.1e} | '
               f'loss {tl / len(tr_loader):.4f} | val {vauc:.4f} (best {best_val:.4f}@{best_ep})'
               + (f' | d1 {d1:.4f}/{d1p:.4f}' if d1 != '' else '')
               + f' | {time.time() - t0:.0f}s')
        print(msg, flush=True)
        _w.writerow(dict(ep=ep, lr=f'{opt.param_groups[0]["lr"]:.3e}',
                         loss=round(tl / len(tr_loader), 6),
                         val=round(vauc, 4) if do_val else '',
                         d1=round(d1, 5) if d1 != '' else '',
                         d1_peak=round(d1p, 5) if d1p != '' else '',
                         sec=round(time.time() - t0)))
        _f.flush()

        if stop_ep is not None:
            break

    # ── 拐点早停：回载 [0, knee] 内 best-val 权重，test 只评估这一次 ──
    mm.load_state_dict(best_sd if best_sd is not None else clone_sd(mm))
    t_auc, n_auc = evaluate(mm, te_loader)
    torch.save(best_sd if best_sd is not None else clone_sd(mm),
               os.path.join(OUT, f'{tag}_ckpt.pt'))
    elapsed = time.time() - t0
    _f.close()

    if stop_ep is None:
        print(f'!! [{tag}] 拐点未触发（跑满 {args.epochs} ep），回载全程 best-val ep{best_ep}', flush=True)
    print(f'>> [{tag}] knee ep{knee_ep if knee_ep is not None else "-"} | '
          f'best-val ep{best_ep} (val {best_val:.4f}) | '
          f'TEST {t_auc:.4f} ({n_auc}/{N} classes) | {elapsed:.0f}s\n', flush=True)
    return dict(tag=tag, seed=seed,
                knee_window=args.knee_window, knee_ratio=args.knee_ratio,
                knee_streak=args.knee_streak, min_epochs=args.min_epochs,
                epochs=args.epochs, bsz=args.bsz, lr=args.lr,
                dropout=args.dropout, params=n_params,
                best_val=round(best_val, 4), best_ep=best_ep,
                knee_ep=knee_ep if knee_ep is not None else '',
                stop_ep=stop_ep if stop_ep is not None else args.epochs - 1,
                test=round(t_auc, 4), n_classes=n_auc, seconds=round(elapsed))


SEEDS = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
TAG = args.tag or ('knee_final' if len(SEEDS) == 1 else 'knee_final')
rows = []
for sd in SEEDS:
    _tag = f'{TAG}_s{sd}' if len(SEEDS) > 1 else f'{TAG}_s{sd}'
    rows.append(run_one(sd, _tag))

csv_path = os.path.join(OUT, 'results_final.csv')
write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if write_header:
        wr.writeheader()
    for r in rows:
        wr.writerow(r)

ts = [r['test'] for r in rows]
print('=' * 78, flush=True)
print(f'>> 策略：knee 拐点早停 | seeds={SEEDS} | 预算上限 {args.epochs} ep', flush=True)
for r in rows:
    print(f"   seed {r['seed']}: knee ep{r['knee_ep']} → best-val ep{r['best_ep']} "
          f"(val {r['best_val']:.4f}) | TEST {r['test']:.4f}", flush=True)
if len(rows) > 1:
    print(f'>> TEST = {np.mean(ts):.4f} ± {np.std(ts, ddof=1):.4f} '
          f'(n={len(ts)}, 样本标准差 ddof=1)', flush=True)
print('>> 锚点：Sisson MPNN-GNN 0.7627±0.0047 | GIN-GNN 0.7359±0.0222', flush=True)
print(f'>> 结果：{csv_path}', flush=True)
