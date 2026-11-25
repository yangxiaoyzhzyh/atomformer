# AtomFormer — 复现包

论文：**Less is more in mixture modeling: every molecular vector is a map, yet the mixture is the territory**
Xiao Yang, Guanghao Song, Feiyu Li, Zihan Xu（Xinqiao Flavors & Fragrances (Zhengzhou) Co., Ltd.）

本仓库提供正文与补充材料全部实验的代码、处理后的数据、训练权重与日志。脚本以「脚本所在目录/../data」解析路径，
与当前工作目录无关。下文所有表/图编号与论文一致（正文 Table 1、Figure 1–2；补充 Tables S1–S8、Texts S1–S6）。

`.pt` / `.safetensors` / `.jsonl` 由 **Git LFS** 管理，clone 前请先 `git lfs install`。
`*.bak`、`__pycache__/` 与超过 GitHub 单文件上限的中间数据不纳入版本控制（见 `.gitignore`）。

## 1. 模型（正文 Materials and Methods 1；补充 Text S1）

所有输入（单分子、二元或多元混合物）的原子被拼成**一个图，无分子边界标记**。

| 阶段 | 气味模型（主模型） | CheMixHub 模型（MON / ΔHvap） | Sisson 变体 |
|---|---|---|---|
| 原子特征 | 31 维（13 intrinsic + 18 local bonding）+ 零填充至 64 | 32 维（+1 维摩尔分数）→ 64 | 同气味模型 |
| GIN | ×3，hidden 64，BN + 可学习 ε 残差 | ×3，hidden 64 | ×3，hidden 64 |
| Transformer | ×2，1024 维，8 头，FFN 2048 | ×2，256 维，4 头，FFN 512 | ×2，256 维，8 头 |
| 位置编码 | 无（保证同一分子不同 SMILES 顺序输出一致） | 无 | 无 |
| 池化 / 读出 | masked mean → L2 归一化 → MSE 回归 BGE-M3 嵌入 | masked mean → 回归 MLP | masked mean → 74 类多标签 BCE |
| 参数量 | 16.9 M | 1.2 M | 1.1 M |

> 正文摘要中的 "1.2 M" 指 CheMixHub 配置（MON / ΔHvap，Transformer 256 维）；气味模型因 Transformer 为 1024 维，
> 参数量为 16.9 M（补充 Table S1）。

31 维原子特征（补充 Text S1.1）：元素 8 维 one-hot（C/N/O/F/P/S/Cl/Br）、手性 2 维、度 6 维、氢数 5 维、
杂化 5 维、芳香性 1 维、形式电荷 3 维、成环 1 维。两个特征组交错排列，均不显式编码连接性——拓扑完全由 GIN 消息传递写入。

## 2. 目录结构

```
.
├── scripts/          # 全部训练 / 评估 / 验证脚本
├── data/             # 数据（见 §5）
├── checkpoints/      # 论文全部模型权重（见 §3）
├── logs/             # 训练日志（各变体最终指标可从日志尾部直接核对）
├── results/          # 评估输出（108 组 per-group AUROC 等）
├── sisson_final/     # Sisson 基准适配与拐点早停实验（见 §6.10）
├── LICENSE
└── README.md
```

## 3. 权重清单

### 3.1 正文 Table 1（单体检索 + blender 预测）

| 文件 | 模型 | 单体 conR@3 | blender AUROC |
|---|---|---|---|
| `model_tgsc_bl_best.pt` | Full（GIN+TFM，Category 4） | 54.7% | **0.9658** |
| `model_tgsc_bl_gin_only_best.pt` | GIN-only（无 Transformer） | 48.7% | 0.8438 |
| `model_tgsc_bl_tfm_only_best.pt` | TFM-only（无 GIN） | 53.8% | 0.8226 |

### 3.2 Figure 2（四类消融，blender macro-AUROC）

| 文件 | 类别 | AUROC |
|---|---|---|
| `best_cat1norm_sum.pt` / `best_cat1norm_mean.pt` | Cat 1 有边界、分子级池化（sum / mean） | 0.8561 / 0.8569 |
| `best_gin3_mean.pt` / `best_gin3_sum.pt` / `best_gin3_max.pt` | Cat 2 简单池化 mean / sum / max | 0.8436 / 0.8541 / 0.8561 |
| `best_gin3_weighted.pt` / `best_gin3_gat.pt` | Cat 2 单步注意力读出 weighted / GAT | 0.8013 / 0.8107 |
| `best_gin3_set2set.pt` | Cat 3 Set2Set 迭代读出 | 0.9149 |
| `model_tgsc_bl_best.pt` | Cat 4 读写式 Transformer（full） | 0.9658 |

### 3.3 CheMixHub 与辅助任务

| 文件 | 任务 | 结果 |
|---|---|---|
| `mon_32dim/model_mon_full_fold0-4.pt` | MON 5 折（Table S3） | R 0.898 ± 0.033 / MAE 4.534 ± 0.539 |
| `heatvap_32dim/model_heatvap_seed0-19.pt` | ΔH_mix 20-seed ensemble（本地集成评估，正文未列表） | R 0.9758 / MAE 0.1583 kJ/mol |
| `gs_cls_best.pt` | GoodScents 单体多标签分类（Text S4.2） | macro-AUROC 0.8514（132/135 类） |

> 目录名 `heatvap_32dim/` 沿用早期命名，其中权重实际对应**混合焓 ΔH_mix** 的 32 维特征模型；
> 汽化焓 ΔH_vap 的 5 折结果由 `scripts/train_ChemixhubTasks.py --task heat_vap` 重新训练（§6.9）。

MON 的 GIN-only / TFM-only 消融权重未随仓库分发，可用 §6.6 的命令复现。

## 4. 训练配置（补充 Table S1）

| 超参 | 气味 | MON | ΔHvap |
|---|---|---|---|
| 优化器 | Adam | Adam | Adam |
| 学习率 | 1×10⁻³ | 1×10⁻⁴ | 1×10⁻⁴ |
| batch | 1024 | 256 | 256 |
| epochs | 60 | 60 | 60 |
| warmup | 3（linear） | 3（linear） | 3（linear） |
| LR schedule | cosine（T_max = 30） | cosine | cosine |
| 采样器 | WeightedRandomSampler（monomer:blender = 100:1） | — | — |
| 输出头 | L2 归一化嵌入（MSE） | 回归 MLP 256→256→256→1 | 回归 MLP in→256→256→1 |

训练环境：单张 RTX 4080 Super；CheMixHub 任务亦可在 CPU 上训练。

## 5. 数据

### 5.1 规模与划分（补充 Table S4 / S5）

| 数据集 | 规模 | 划分 |
|---|---|---|
| TGSC 单体 | 3,666 | 训练 3,430 / 测试 236（测试集与 blender 训练集组分无重叠） |
| TGSC Blender | 469,791 | 训练 462,839 / 验证 692 / 测试 6,260 |
| 气味标签 | 128 组 | 测试集覆盖 108 组（per-group 表见 `results/per_group_full_32dim.csv`） |
| CheMixHub ΔH_mix | 30,061 | `data/HE/`：训练 21,041 / 验证 3,007 / 测试 6,013 |
| CheMixHub MON | 684 | `data/MON/motor_octane_number_splits/`（5 折） |

Blender 训练集 462,839 条可归并为 252,230 个无序分子对，其中 331,455 条（71.6%）与其他条目共享同一分子对但标签不同
（补充 Text S2.2）。数据在 RDKit canonical SMILES 按字母序后按无序对 1% 随机切分，测试集 A/B 顺序随机打乱。

### 5.2 文件与来源

| 文件 | 说明 |
|---|---|
| `data/blender_{train,val,test}.jsonl` | TGSC 二元混合物对 + 气味标签；`embedding` 列为 BGE-M3 文本嵌入 |
| `data/tgsc_{train,test}_bge.jsonl` | TGSC 单分子 + 完整气味描述及其 BGE-M3 嵌入 |
| `data/odor_group_1024dim_cache.json` | 128 个标签的 BGE-M3 嵌入（检索评估用） |
| `data/gs_cls/` | GoodScents 单分子数据（本地清洗版，135 词表见 `train_gs_cls.py`） |
| `data/MON/` | CheMixHub MON：`processed_MON.csv`、`compounds.csv`、`motor_octane_number_splits/` |
| `data/HE/` | CheMixHub 混合焓：`HE_compounds.csv`、`HE_{train,val,test}.csv` |
| `data/miscible-solvent/processed_data/` | CheMixHub 原始 `processed_MiscibleSolventData.csv`（density / ΔH_mix / ΔH_vap 共 90,184 行）+ `heat_of_vaporization_splits/` |
| `sisson_final/data/notes74.csv` | Sisson 74 类标签名 |

**未随仓库分发**（体积或授权原因）：

- `sisson_final/data/trainpairs74.pt`、`testpairs74.pt`（各约 100 MB，超出 GitHub 单文件上限）：pickle 的 `PairData` 对象，
  含分子图与 74 维多标签，按补充 Text S3.2 的重建协议生成后用 `--pairing-dir` 指定。
- CheMixHub 的 ln *η*（NIST / NIST-full）、ln *κ*、ln *η*（IlThermo）四个任务数据，需从 CheMixHub 原仓库获取；
  把 `<task>/processed_data` 放到 `--root` 指向的目录即可用 §6.9 的命令训练。
- CheMixHub 的 `density_splits/`、`enthalpy_of_mixing_splits/` 等划分索引：仓库内只附带 `heat_of_vaporization_splits/`；
  `data/HE/` 是 ΔH_mix 的单次 70/10/20 划分，Table S3 的 5 折数字需从 CheMixHub 原始划分重建。

气味数据自 TGSC 官网爬取并清洗（非 pyrfume 发布版）；BGE-M3 嵌入由 BGE-M3（Chen et al., 2024）对描述文本编码生成，
可按 `text` / `description` 字段重算。

## 6. 复现命令

模型与脚本的对应关系见补充 Table S7。

### 6.1 气味主模型与四类消融（Table 1、Figure 2）

```bash
# 主模型与两个单塔变体（Category 4 / GIN-only / TFM-only，Table 1）
python scripts/train_tgsc_bl_31dim.py --variant full
python scripts/train_tgsc_bl_31dim.py --variant gin_only
python scripts/train_tgsc_bl_31dim.py --variant tfm_only

# Category 2/3：无边界 + 简单池化 / Set2Set 读出
python scripts/test_pooling_final_31dim.py mean
python scripts/test_pooling_final_31dim.py sum
python scripts/test_pooling_final_31dim.py max
python scripts/test_pooling_final_31dim.py weighted
python scripts/test_pooling_final_31dim.py gat
python scripts/test_pooling_final_31dim.py set2set

# Category 1：保留分子边界，分子级池化后组合
python scripts/train_tgsc_cat1_31dim.py sum
python scripts/train_tgsc_cat1_31dim.py mean
```

### 6.2 单体检索与位置精度（Table 1、Text S4.2）

```bash
python scripts/eval_tgsc_31dim.py checkpoints/model_tgsc_bl_best.pt full monomer   # conR@3 = 54.7%
python scripts/eval_tgsc_31dim.py checkpoints/model_tgsc_bl_best.pt full blender   # macro-AUROC = 0.9658
python scripts/eval_blender_retrieval.py   # z–目标 cosine 0.9312（中位 0.9803）/ top-1 81.6% / top-3 94.4%
```

### 6.3 标签歧义与连续嵌入（Text S2、Table S2）

```bash
python scripts/verify_midpoint.py          # 1,000 对冲突 AB/BA：z→mid 0.9707；中点更近 76.3%；top-2 含冲突标签 99.8%（双命中 74.3%）
python scripts/verify_label_ambiguity.py   # 冲突条目 71.6%；冲突标签间 cosine 0.50 vs 随机 0.49
python scripts/verify_bge_grouping.py      # 早期辅助验证（按描述首词分组），最终稿未报告该组数字
```

### 6.4 易混淆子集（Text S6、Table S8）

```bash
python scripts/verify_odor_claims.py
# 2,086 对易混淆子集：full 0.9652 / GIN-only 0.8680 / TFM-only 0.8180
# GIN-only − TFM-only 由全集 +2.1 扩大到 +5.0
```

### 6.5 Per-group 评估（Text S4.3、Table S6）

```bash
python scripts/eval_per_group_31dim.py   # 108 组 per-group AUROC，逐组写入 results/per_group_full_31dim.csv
```

正文分层统计：稀有组（<100 对）0.8666、中频组 0.9908、高频组 0.9813。仓库内 `results/per_group_full_32dim.csv`
是论文 Table S6 所用的成品表（108 行，可直接逐组核对）。

### 6.6 MON（Table S3）

```bash
# 论文配置（Table S1：lr 1e-4 / batch 256 / warmup 3 / 60 epochs，为脚本默认值）
python scripts/train_mon.py --data-root data/MON --folds 0 1 2 3 4 --variant full
python scripts/train_mon.py --data-root data/MON --folds 0 --variant gin_only
python scripts/train_mon.py --data-root data/MON --folds 0 --variant tfm_only
```

5 折结果 R 0.898 ± 0.033 / MAE 4.534 ± 0.539，见 `logs/mon_best5fold.log`。`search_mon.py`、
`search_mon_warmup.py` 为超参搜索脚本（两阶段：40 配置 × fold-0 筛选 → Top 3 × 5 折）。

### 6.7 混合焓 ΔH_mix（Table S3）

```bash
# 20-seed ensemble 权重已随仓库提供，直接在 data/HE/HE_test.csv 上集成评估
python scripts/eval_he_ensemble.py
```

`data/HE/` 为 CheMixHub 的单次划分；Table S3 报告的 ΔH_mix 5 折数字（R 0.972 ± 0.003 / MAE 0.169 ± 0.001 kJ/mol）
需 CheMixHub 的 `enthalpy_of_mixing_splits` 划分，该索引未随仓库分发（见 §5.2）。

### 6.8 单体多标签分类 GS-CLS（Text S4.2）

```bash
python scripts/train_gs_cls.py   # 135 词表，80:10:10 划分，TEST macro-AUROC 0.8514（日志见 logs/gs_cls.log）
```

### 6.9 CheMixHub 多任务（Table S3）

```bash
# ΔHvap 5 折（Table S1：lr 1e-4 / batch 256）
python scripts/train_ChemixhubTasks.py --task heat_vap --folds 0 1 2 3 4 --root data
```

`--root` 需指向含 `<task>/processed_data` 的目录；仓库内 `data/miscible-solvent/processed_data/` 覆盖
density / ΔH_mix / ΔH_vap 三个任务的样本，但只附带了 `heat_of_vaporization_splits/` 划分索引，
因此**开箱可跑的是 ΔHvap**（R 0.999 / MAE 0.085 kcal·mol⁻¹，`logs/heat_vap_5fold.log`：
0.9988 ± 0.0000 / 0.0851 ± 0.0018）。density（R 0.999 / MAE 0.004 g·cm⁻³）与其余四个任务
（NIST、IlThermo）需从 CheMixHub 取得对应的 `*_splits/` 索引与数据后再 `--root` 指向。
`scripts/train_heatvap_5fold.py` 与该脚本内容相同，为 ΔHvap 的早期副本。

### 6.10 Sisson 基准与拐点早停（Text S3.2）

```bash
# 3 seeds，knee 拐点早停（因果移动平均 window=5、一阶导阈值 0.1、连续 2 个 epoch）
python sisson_final/scripts/train_sisson_knee_final.py --seeds 42,43,44
python sisson_final/scripts/knee_detect.py
python sisson_final/scripts/plot_final_curves.py
```

- 协议：按 Sisson et al. 的定义重建 74 类划分（42,158 训练 / 40,461 测试 / 83,319 丢弃），再从训练集留出 1,000 对做早停验证。
- 模型：GIN×3（64）→ Transformer×2（256 维 / 8 头）→ masked mean → 74 类多标签 BCE，1.1 M 参数。
- 结果：TEST AUROC **0.7603 ± 0.0057**（n=3，样本标准差）；对照 Sisson MPNN-GNN 0.7627 ± 0.0047、GIN-GNN 0.7359 ± 0.0222。
- 产物：`sisson_final/results/results_final.csv`（逐 seed 拐点与指标）、`curve_knee_final_s*.csv`、`knee_final_s*_ckpt.pt`、`final_knee_curves.png`。

## 7. 结果对照

| 论文位置 | 结果 | 复现入口 |
|---|---|---|
| Table 1（Full / GIN-only / TFM-only） | 0.9658 / 0.8438 / 0.8226 | §6.1、§6.2 |
| Figure 2（Category 1–4） | 0.8561–0.8569 / 0.8436–0.8561 / 0.9149 / 0.9658 | §6.1 |
| Table S2（标签歧义） | z→mid 0.9707；中点更近 76.3%；top-2 命中 99.8% | §6.3 |
| Table S3（8 个 CheMixHub 任务） | MON R 0.898 / MAE 4.534 等 | §6.6、§6.9 |
| Table S6（108 组 per-group） | 0.9658；稀有/中频/高频 0.8666 / 0.9908 / 0.9813 | §6.5 |
| Table S7（模型–脚本映射） | — | §6.1、§6.9、§6.10 |
| Table S8（易混淆子集） | full 0.9652；GIN − TFM 由 +2.1 → +5.0 | §6.4 |
| Text S3.2（Sisson） | 0.7603 ± 0.0057 vs MPNN-GNN 0.7627 | §6.10 |
| Text S4.2（GS-CLS） | macro-AUROC 0.8514 | §6.8 |

## 8. 许可与引用

- 代码：MIT（见 `LICENSE`）。
- 数据：气味数据遵循 TGSC / pyrfume 的使用条款；MON、ΔH_mix、ΔH_vap 等 CheMixHub 数据遵循
  Rajaonson et al. (NeurIPS 2025) 的原始许可，使用时请引用原文。
- 若使用本仓库，请引用本文正文所列论文。
