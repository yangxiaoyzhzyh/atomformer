# AtomFormer — 复现包

本仓库包含论文《AtomFormer》全部实验的复现代码与数据（脚本中或有绝对路径，读者需自行修改）：
**气味混合物预测**（回归到 BGE 描述符嵌入 + 检索评估）、**超额焓预测**、**辛烷值（MON）预测**，以及消融/验证实验与数据规模拐点实验。

## 目录结构

```
.
├── scripts/          # 全部训练/评估/分析脚本
├── data/             # 数据目录（见下文"数据获取"）
│   ├── blender_train.jsonl / blender_val.jsonl / blender_test.jsonl
│   ├── tgsc_train_bge.jsonl / tgsc_test_bge.jsonl
│   ├── gs_cls/       # GoodScents 单体多标签分类数据（gs_train / gs_test）
│   ├── odor_group_1024dim_cache.json
│   ├── MON/          # processed_MON.csv, compounds.csv, motor_octane_number_splits/
│   ├── HE/           # HE_compounds.csv, HE_train.csv, HE_val.csv, HE_test.csv
│   └── miscible-solvent/  # CheMixHub 原始 processed_data（含 heat_of_vaporization_splits/）
├── checkpoints/      # 论文全部模型的 best ckpt（见下表）
├── logs/             # 实验训练日志
├── results/          # 评估输出（per-group AUROC 等）
├── sisson_final/     # 数据规模拐点实验（见"数据规模拐点实验"小节）
├── LICENSE
└── README.md
```

`.pt` / `.safetensors` / `.jsonl` 等大体积文件由 **Git LFS** 管理，clone 前请先安装并执行 `git lfs install`。
`*.bak`、`__pycache__/` 与超过 GitHub 单文件上限的中间数据缓存不纳入版本控制（见 `.gitignore`）。

**预训练权重（checkpoints/，对应表 2 与 Fig 2）**：

| 文件 | 模型 | Test AUROC |
|---|---|---|
| model_tgsc_bl_best.pt | full（Cat4，GIN+TFM） | 0.9658 |
| model_tgsc_bl_gin_only_best.pt | GIN-only（无 TFM） | 0.8438 |
| model_tgsc_bl_tfm_only_best.pt | TFM-only（无 GIN） | 0.8226 |
| best_cat1norm_sum.pt / best_cat1norm_mean.pt | Cat1（带边界） | 0.7467 / 0.7432 |
| best_gin3_mean/max/sum/weighted/gat.pt | Cat2（无边界池化） | 0.8436 / 0.8561 / 0.8541 / 0.8013 / 0.8107 |
| best_gin3_set2set.pt | Cat3（Set2Set） | 0.9149 |
| mon_32dim/model_mon_full_fold0-4.pt | MON（5 折，warmup=3/ep=60/lr=3e-4） | MAE 4.53 ± 0.54 |
| mon_32dim/model_mon_gin_only_fold0.pt † | MON GIN-only（fold 0，Table 4，lr=3e-4） | MAE 5.75 / R 0.832 |
| mon_32dim/model_mon_tfm_only_fold0.pt † | MON TFM-only（fold 0，Table 4，lr=3e-4） | MAE 6.74 / R 0.792 |

† 未随仓库分发，用 `train_mon.py --folds 0 --variant gin_only|tfm_only --lr 3e-4 --batch 64` 复现。

所有气味实验使用统一训练协议：batch 1024、lr 1e-3、warmup 3、60 epochs、best-val checkpoint 评估（MON 与超额焓协议见各自小节）。

## 依赖

- Python >= 3.10
- torch、torch_geometric（>= 2.4）、rdkit、scikit-learn、scipy、pandas、pyarrow、safetensors
- 建议 GPU（所有脚本默认自动使用 CUDA）

## 数据获取

| 数据 | 来源 | 说明 |
|---|---|---|
| blender_*.jsonl | The Good Scents Company (TGSC) 气味数据库（自官网爬取并清洗，见正文 Materials；pyrfume 亦提供 GoodScents 数据镜像）| 二元混合物对 + odor group 标签；`embedding` 列为 BGE-M3 文本嵌入 |
| gs_cls/gs_train.jsonl, gs_test.jsonl | GoodScents 单分子数据（本地清洗版，保留 sweet；MIST 135 词表见 train_gs_cls.py）| 单体多标签分类（S3.4 对照实验）|
| tgsc_*_bge.jsonl | 同上（TGSC 单体） | 单体 + 描述符 BGE-M3 嵌入 |
| odor_group_1024dim_cache.json | 本文生成 | 128 个气味描述符组的 BGE-M3 嵌入（Blender 全数据集标签，供检索评估） |
| processed_MON.csv, compounds.csv | CheMixHub（Rajaonson et al., NeurIPS 2025） | MON（辛烷值）任务，保留原始许可 |
| HE_*.csv | CheMixHub（同上） | 超额焓任务，CheMixHub fold-0 划分 |
| sisson_final/data/notes74.csv | 本文整理 | 74 个气味描述符组的类名表 |
| sisson_final/data/trainpairs74.pt, testpairs74.pt | odor-pair 数据集 | **未随仓库分发**（各约 100 MB，超出 GitHub 单文件上限）；为 pickle 的 `PairData` 对象，含分子图与 74 维多标签。用 `--pairing-dir` 指向本地构建的同名数据即可 |

- **BGE-M3 嵌入**：`tgsc_*_bge.jsonl` 与 `odor_group_1024dim_cache.json` 中的嵌入由
  BGE-M3（Chen et al., 2024）对描述文本编码生成；如需自行生成，
  用 BGE-M3 对每条 `text`/`description` 编码即可复现。
- 训练/验证/测试划分与正文完全一致（blender 6,260 测试对等）。

## 运行

所有命令在仓库根目录执行。各脚本以「脚本所在目录/../data」定位数据，与当前工作目录无关。

### 1. 气味混合预测（主模型 + 消融）

```bash
# 主模型 full / gin_only / tfm_only（对应表 2 与 Category 4 / B / C）
python scripts/train_tgsc_bl_31dim.py --variant full      # 60 epochs
python scripts/train_tgsc_bl_31dim.py --variant gin_only
python scripts/train_tgsc_bl_31dim.py --variant tfm_only

# 无边界池化系（Category 2/3：mean / max / sum / weighted / gat / set2set）
python scripts/test_pooling_final_31dim.py mean

# 带分子边界（Category 1：sum / mean 组合）
python scripts/train_tgsc_cat1_31dim.py sum
python scripts/train_tgsc_cat1_31dim.py mean
```

### 2. 单体检索与混合评估（conR@3 / BL AUROC）

```bash
# 单体 conR@3 / cosR@3 + blender AUROC（需先训练得到 checkpoints/ 下的模型）
python scripts/eval_tgsc_31dim.py checkpoints/model_tgsc_bl_best.pt full monomer
python scripts/eval_tgsc_31dim.py checkpoints/model_tgsc_bl_best.pt full blender
python scripts/train_gs_cls.py              # GS-CLS：单体多标签分类头（S3.4），TEST macro-AUROC ≈ 0.85
python scripts/eval_blender_retrieval.py   # 检索：z-目标 cosine 0.9312 / top-1 81.6% / top-3 94.4% (S3)
```

### 3. 验证实验（易混淆子集，补充材料 S5）

```bash
python scripts/verify_odor_claims.py   # 易混淆子集验证（S5：gin−tfm +2.1 → +5.0）
```

### 4. MON（辛烷值）

```bash
# 论文配置：lr=3e-4 / batch=64 / warmup=3 / ep=60（Table 4 与 5 折共用）
python scripts/train_mon.py --data-root data/MON --folds 0 1 2 3 4 --variant full --lr 3e-4 --batch 64
python scripts/train_mon.py --data-root data/MON --folds 0 --variant gin_only --lr 3e-4 --batch 64
python scripts/train_mon.py --data-root data/MON --folds 0 --variant tfm_only --lr 3e-4 --batch 64

# 复现论文最终结果（MAE = 4.53 ± 0.54）的两阶段搜索：
# 阶段 1: 40 配置 × fold-0 筛选；阶段 2: Top 3 × 5 折
python scripts/search_mon.py           # 基础超参搜索（lr/dropout/dim/batch）
python scripts/search_mon_warmup.py    # warmup×epochs 搜索（固定最优基础参数）
```

### 5. 超额焓（32 维特征）

论文报告 20-seed ensemble（R = 0.9758, MAE = 0.1583 kJ/mol），20 个 seed 的 best-val 权重已随仓库提供：

```bash
# 读 checkpoints/heatvap_32dim/model_heatvap_seed0-19.pt，对 data/HE/HE_test.csv 集成预测
python scripts/eval_he_ensemble.py
```

5 折协议（补充材料 S6 表：R = 0.9719 ± 0.0026, MAE = 0.1686 kJ/mol）由 CheMixHub 多任务脚本完成，
数据在 `data/miscible-solvent/processed_data/`：

```bash
python scripts/train_heatvap_5fold.py --task he --folds 0 1 2 3 4 --root data
```

逐 seed 训练脚本（`train_he_32dim.py`）为本地超参搜索产物，未随仓库分发；等价设置为
32 维原子特征（31 维增强 + 摩尔分数）、best-val checkpoint、60 epochs。同脚本还支持
`--task heat_vap / density` 等 CheMixHub 任务，用 `--root` 指向含 `<task>/processed_data` 的数据根目录。

### 6. BGE 空间分组验证（补充材料 S3）

```bash
python scripts/verify_bge_grouping.py   # 122 组 first-word：AUROC 0.9824 / intra 0.8697 / inter 0.5517
python scripts/verify_label_ambiguity.py  # 条目级标签冲突 72% / 冲突 cosine 0.50 vs 随机 0.49
python scripts/verify_midpoint.py          # 1000 冲突对：z→mid 0.9707 / 中点更近 76.3% / 检索 74.3%+99.8% / 几何 8128 对 100%
```

### 7. Per-group 评估（补充材料 S1）

```bash
python scripts/eval_per_group_31dim.py   # 输出 108 组 per-group AUROC 表（0.9658）
```

### 8. 数据规模拐点实验（`sisson_final/`）

在 74 类多标签气味分类（odor-pair 数据集）上与图神经网络基线对比，并检验「以 val 曲线拐点早停替代固定 epoch 预算」的稳定性：

```bash
# 3 seeds（42/43/44），knee 拐点早停（因果移动平均 window=5、一阶导阈值 0.1、连续 2 个 epoch）
python sisson_final/scripts/train_sisson_knee_final.py --seeds 42,43,44

python sisson_final/scripts/knee_detect.py        # 拐点判据核对
python sisson_final/scripts/plot_final_curves.py  # 生成 results/final_knee_curves.png
```

- 数据：`sisson_final/data/`（`notes74.csv` 为 74 类类名；`trainpairs74.pt` / `testpairs74.pt` 体积各约 100 MB，**未随仓库分发**，用 `--pairing-dir` 指向本地构建的同名数据）
- 结果：`sisson_final/results/results_final.csv`（逐 seed 拐点、best-val epoch、TEST AUROC）、`curve_knee_final_s{42,43,44}.csv`（逐 epoch 曲线）、`knee_final_s*_ckpt.pt`（best-val 权重）
- TEST AUROC = **0.7603 ± 0.0057**（n=3，样本标准差 ddof=1）；对照 Sisson MPNN-GNN 0.7627 ± 0.0047、GIN-GNN 0.7359 ± 0.0222

## 复现结果（对应论文表/图）

| 实验 | 论文位置 | 复现命令 |
|---|---|---|
| 主模型 full | Table 2 / Fig 2 (Cat 4) | `train_tgsc_bl_31dim.py --variant full` |
| 消融 B/C | Table 2 / 补充 | `--variant gin_only / tfm_only` |
| Category 1-4 | Fig 2 | 上表全部命令（消融数字见 logs/） |
| 单体 conR@3 | Table 2 | `eval_tgsc_31dim.py ... monomer` |
| 易混淆子集验证 | 补充 S5 | `verify_odor_claims.py` |
| MON 三变体 | Table 4 | `train_mon.py --folds 0 --variant ... --lr 3e-4 --batch 64` |
| 超额焓 | 2.5 / 补充 S4 | `eval_he_ensemble.py`（20 seeds ensemble）/ `train_heatvap_5fold.py --task he` |
| 数据规模拐点实验 | 补充 | `sisson_final/scripts/train_sisson_knee_final.py --seeds 42,43,44` |

## 许可

代码：MIT（见仓库根 LICENSE）。
数据：气味数据遵循 pyrfume 许可；MON 与超额焓数据遵循 CheMixHub 许可（Rajaonson et al., 2025），
使用时请引用原文。
