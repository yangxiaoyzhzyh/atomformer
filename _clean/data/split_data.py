"""完整数据分割 — 训练集/测试集 固定下来"""

import sys; sys.path.insert(0, r'C:\Users\yangx\v3-mixture-token')

import json, random, sqlite3, torch
import torch.nn.functional as F
random.seed(42)

DB = r'C:\Users\yangx\Documents\TGSC\tgsc_data.db'
GS_FILE = 'data/metadata/goodscents_library.json'
GROUP_EMB = r'C:\Users\yangx\odor-pair\odor_group_1024dim_cache.json'

TRAIN = 'data/processed/combined/train.jsonl'
TEST = 'data/processed/combined/test.jsonl'

conn = sqlite3.connect(DB)

# 1. Blender 数据
blenders = conn.execute('''SELECT c1.smiles,c2.smiles,b.odor_group FROM blenders b
    JOIN compounds c1 ON b.compound_url=c1.page_url JOIN compounds c2 ON b.blender_url=c2.page_url
    WHERE c1.smiles IS NOT NULL AND c2.smiles IS NOT NULL AND c1.smiles!="" AND c2.smiles!=""
      AND c1.smiles NOT LIKE "%.%" AND c2.smiles NOT LIKE "%.%" AND b.odor_group NOT LIKE "No%" AND LENGTH(c1.smiles)<200''').fetchall()
random.shuffle(blenders)

# 收集出现在 blender 中的分子
blender_smis = set()
for a, b, _ in blenders: blender_smis.add(a); blender_smis.add(b)

# 2. GoodScents 单体（带 BGE-M3 embedding）
with open(GS_FILE) as f:
    lib = json.load(f)
emb_by_smi = {c['smiles']: c['embedding'] for c in lib['compounds'] if c.get('smiles') and c.get('embedding')}
desc_by_smi = {c['smiles']: c['description'] for c in lib['compounds'] if c.get('smiles') and c.get('description','').strip()}
total_emb = len(emb_by_smi)
print(f'GoodScents 有 embedding: {total_emb}')

# 3. 单体分割（同 train_final.py: 不在 blender 中的分子做测试，10% cap）
all_singles = [(smi, desc_by_smi.get(smi,'')) for smi in emb_by_smi if smi in desc_by_smi]
single_ood = [(smi, desc) for smi, desc in all_singles if smi not in blender_smis]  # 不在 blender 中
single_id = [(smi, desc) for smi, desc in all_singles if smi in blender_smis]       # 在 blender 中
random.shuffle(single_ood)
n_te = int(total_emb * 0.1)  # 10% 的 embedding
single_test = single_ood[:n_te]
test_smis = set(s for s,_ in single_test)

# 训练：除去测试集的所有单体（不管在不在 blender 中）
single_train = [(smi, desc_by_smi.get(smi,'')) for smi in emb_by_smi if smi in desc_by_smi and smi not in test_smis]
print(f'单体: train={len(single_train)} test={len(single_test)} (OOD={len(single_ood)})')

# 4. Blender 分割：2% 测试
n_bt = int(len(blenders) * 0.02)
blender_test = blenders[:n_bt]
blender_train = blenders[n_bt:]
print(f'Blender: train={len(blender_train)} test={len(blender_test)}')

# 5. 加载 odor group embedding
with open(GROUP_EMB) as f:
    group_data = json.load(f)
group_names = list(group_data.keys())
group_vecs = {n: torch.tensor(v) for n, v in group_data.items()}
print(f'Odor groups: {len(group_names)}')

# 6. 写出训练集
with open(TRAIN, 'w') as f:
    # GoodScents 单体
    for smi, desc in single_train:
        record = {
            'smiles': smi, 'text': desc, 'type': 'single',
            'embedding': emb_by_smi[smi],
        }
        f.write(json.dumps(record) + '\n')
    # Blender
    for a, b, og in blender_train:
        g = og.rstrip(', ')
        if g not in group_vecs: continue
        record = {
            'smiles_a': a, 'smiles_b': b,
            'text': g, 'type': 'blender',
            'embedding': group_vecs[g].tolist(),
        }
        f.write(json.dumps(record) + '\n')

# 7. 写出测试集
with open(TEST, 'w') as f:
    # GoodScents 单体（不在 blender 中）
    for smi, desc in single_test:
        record = {
            'smiles': smi, 'text': desc, 'type': 'single',
            'embedding': emb_by_smi[smi],
        }
        f.write(json.dumps(record) + '\n')
    # Blender 测试
    for a, b, og in blender_test:
        g = og.rstrip(', ')
        if g not in group_vecs: continue
        record = {
            'smiles_a': a, 'smiles_b': b,
            'text': g, 'type': 'blender',
            'embedding': group_vecs[g].tolist(),
        }
        f.write(json.dumps(record) + '\n')

conn.close()
print(f'\n已保存:')
print(f'  {TRAIN}')
print(f'  {TEST}')

# 统计
import subprocess
for f in [TRAIN, TEST]:
    n = int(subprocess.run(['wc', '-l', f], capture_output=True, text=True).stdout.split()[0])
    print(f'  {f}: {n} 条')
