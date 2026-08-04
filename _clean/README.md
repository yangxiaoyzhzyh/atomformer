## AtomFormer

A simple GIN+Transformer backbone for general mixture property prediction. Concatenates all atoms from all constituent molecules into a single graph and processes them through a unified attention mechanism—no per-molecule encoding, no separate fusion module.

## Architecture

```
ZeroPad(9→64) → GINConv×3(hidden=64) → Linear(64→1024)
→ TransformerEncoder×2(heads=8) → Masked Mean Pool → L2 → MSE → BGE-M3-1024
```

For excess enthalpy, each molecule's mole fraction is appended to every atom of that molecule as a 10th atomic feature (ZeroPad(10→64)). The projection reduces to 256-dim, the Transformer uses 4 heads, and the pooled vector feeds a scalar regression head (Linear → scalar).

## Repository Structure

```
atomformer_paper/
├── README.md
├── requirements.txt
├── scripts/                    # Training & evaluation
│   ├── train_tgsc_bl.py        # Odor prediction (GIN+TFM→BGE, TGSC+BL)
│   ├── train_he.py             # Excess enthalpy multi-seed training
│   ├── eval_blender.py         # Blender test: macro-AUROC + R@1/R@3
│   ├── eval_bl_per_group.py    # Per-group AUROC breakdown
│   ├── eval_monomer.py         # Monomer conR@K + cosR@K
│   ├── eval_label_ambiguity.py # Label ambiguity midpoint verification
│   └── eval_he.py              # Excess enthalpy single-seed evaluation
├── data/
│   ├── tgsc_train_bge.jsonl         # TGSC monomer training (3,430 compounds)
│   ├── tgsc_test_bge.jsonl          # TGSC monomer test (236 compounds)
│   ├── blender_train.jsonl          # Blender pair training (547,287 pairs)
│   ├── blender_val.jsonl            # Blender pair validation (5,529 pairs)
│   ├── blender_test.jsonl           # Blender pair test (6,260 pairs)
│   ├── odor_group_1024dim_cache.json # Odor group → BGE-M3 embedding
│   ├── clean_descriptions.py        # Odor description preprocessing
│   ├── encode_descriptions.py       # BGE-M3 embedding generation
│   ├── split_data.py                # Train/val/test split utilities
│   └── HE/                          # Excess enthalpy data
│       ├── HE_compounds.csv
│       ├── HE_train.csv             # 21,041 samples
│       ├── HE_val.csv               # 3,007 samples
│       └── HE_test.csv              # 6,013 samples
├── checkpoints/
│   ├── model_tgsc_bl_best.pt        # Odor prediction
│   └── model_he_best.pt             # Excess enthalpy best single-seed
└── logs/
    └── train_tgsc_bl.log            # Training log (reproducibility)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torch_geometric
- rdkit
- numpy
- scikit-learn

```bash
pip install torch torch_geometric rdkit numpy scikit-learn pandas scipy requests
```

## Training

```bash
# Odor prediction (TGSC + Blender)
python scripts/train_tgsc_bl.py

# Excess enthalpy (default: 10 seeds; pass comma-separated seeds to override)
python scripts/train_he.py           # uses seeds 0–9
python scripts/train_he.py 0,1,2,...,49  # explicit seed list
```

## Evaluation

```bash
# Blender test — macro-AUROC + R@1/R@3
python scripts/eval_blender.py

# Per-group AUROC breakdown
python scripts/eval_bl_per_group.py

# Monomer retrieval — conR@K + cosR@K
python scripts/eval_monomer.py

# Label ambiguity verification
python scripts/eval_label_ambiguity.py

# Excess enthalpy evaluation
python scripts/eval_he.py
```

## Data

All training data (TGSC monomer and blender) originates from [The Good Scents Company](http://www.thegoodscentscompany.com). Processed data files with BGE-M3 embeddings are included.

**Large files** (>50 MB) are stored with Git LFS. Clone with:

```bash
git lfs pull
```

## Citation

```
@article{atomformer2026,
  title={AtomFormer: A Simple Model for Any Mixture Odor Prediction and Beyond},
  ...
}
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0) — see [LICENSE](LICENSE) for details.
