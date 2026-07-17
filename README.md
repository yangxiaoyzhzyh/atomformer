## Architecture

```
ZeroPad(9→64) → GINConv×3(hidden=64) → Linear(64→1024)
→ TransformerEncoder×2(heads=8) → Masked Mean Pool → L2 → MSE → BGE-M3-1024
```

## Repository Structure

```
atomformer_paper/
├── README.md
├── requirements.txt
├── paper_draft_v2.7.md       # Full paper draft
├── scripts/                  # Training & evaluation
│   ├── train_tgsc_bl.py      # Full model (GIN+TFM→BGE, TGSC+BL) 🏆
│   ├── train_gin_only.py     # GIN-only regression ablation
│   ├── train_gin_135.py      # GIN-only classification (GIN×5+Set2Set+BCE)
│   ├── train_gs_cls_meanpool.py  # GIN+TFM+cls (BCE, GS only)
│   ├── train_cls135.py       # GIN+TFM+cls (BCE, GS+BL)
│   ├── eval_blender.py       # Blender test: macro-AUROC + R@1/R@3
│   ├── eval_bl_per_group.py  # Per-group AUROC breakdown
│   ├── eval_ginonly_auroc.py # GIN-only regression blender AUROC
│   ├── eval_monomer.py       # Monomer conR@K + cosR@K (temperature-weighted)
│   ├── eval_label_ambiguity.py  # Label ambiguity midpoint verification
│   ├── eval_cls_gs.py        # Classification model GS test (AUROC + R@1)
│   └── supporting_material_auroc.md  # Full per-group AUROC table
│   ├── train_he.py           # Excess enthalpy multi-seed training
│   └── eval_he.py            # Excess enthalpy ensemble evaluation
├── data/
│   ├── tgsc_train_bge.jsonl  # TGSC monomer training (3,430 compounds)
│   ├── tgsc_test_bge.jsonl   # TGSC monomer test (236 compounds)
│   ├── blender_train.jsonl   # Blender pair training (552,816 pairs)
│   ├── blender_test.jsonl    # Blender pair test (6,260 pairs)
│   ├── gs_train_nosweet.jsonl  # GoodScents training (classification)
│   ├── gs_test_nosweet.jsonl   # GoodScents test (classification)
│   ├── odor_group_1024dim_cache.json  # Odor group → BGE-M3 embedding
│   └── HE/                   # Excess enthalpy data (Section 4)
│       ├── HE_compounds.csv
│       ├── HE_train.csv
│       ├── HE_val.csv
│       └── HE_test.csv
├── checkpoints/
│   ├── model_tgsc_bl_ep38.pt       # Full regression model (BL AUROC=0.9347)
│   ├── model_gin_only_ep59.pt      # GIN-only regression (BL AUROC=0.8185)
│   ├── model_gin_cls_best.pt       # GIN-only classification (GS AUROC=0.8326)
│   ├── model_nosweet_cls135_auc_best.pt  # GIN+TFM+cls (BL AUROC=0.7847)
│   ├── model_gin135_nosweet.pt     # GIN135 Sisson replication
│   └── model_he_best.pt           # Excess enthalpy best single-seed
└── data/clean_descriptions.py  # Data preprocessing utilities
    data/encode_descriptions.py
    data/split_data.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torch_geometric
- rdkit
- numpy
- scikit-learn

`pip install torch torch_geometric rdkit numpy scikit-learn`

## Training

```bash
# Full model (TGSC + Blender, regression)
python scripts/train_tgsc_bl.py

# GIN-only regression ablation
python scripts/train_gin_only.py

# GIN-only classification (Sisson replication)
python scripts/train_gin_135.py

# GIN+TFM classification (GS only)
python scripts/train_gs_cls_meanpool.py

# GIN+TFM classification (GS + Blender)
python scripts/train_cls135.py

# Excess enthalpy multi-seed training
python scripts/train_he.py 0,1,2,...,49
```

## Evaluation

```bash
# Full model — blender test (macro-AUROC + R@1/R@3)
python scripts/eval_blender.py

# Per-group AUROC breakdown
python scripts/eval_bl_per_group.py

# GIN-only regression — blender test
python scripts/eval_ginonly_auroc.py

# Monomer retrieval — conR@K + cosR@K (TGSC test)
python scripts/eval_monomer.py

# Label ambiguity midpoint verification
python scripts/eval_label_ambiguity.py

# Classification models — GS test AUROC + R@1
python scripts/eval_cls_gs.py gin       # GIN-only classification
python scripts/eval_cls_gs.py gin_tfm   # GIN+TFM classification

# Excess enthalpy ensemble evaluation
python scripts/eval_he.py
```

## Data

All training data (TGSC monomer and blender, GoodScents) originates from [The Good Scents Company](http://www.thegoodscentscompany.com). Processed data files with BGE-M3 embeddings are included.

**Large files** (>50 MB) are stored with Git LFS in `checkpoints/` and `data/`. Clone with:

```bash
git lfs pull
```

## Citation

```
@article{atomformer2026,
  title={AtomFormer: Breaking Molecular Boundaries for Any-Molecule Odor Prediction},
  ...
}
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0) — see [LICENSE](LICENSE) for details.
