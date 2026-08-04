"""Complete data split pipeline: monomers + blenders with decontamination.

Pipeline for blender pairs:
  1. Extract all A+B pairs from TGSC database
  2. RDKit canonical SMILES for both molecules
  3. Sort (smiles_a, smiles_b) to canonical order for dedup
  4. Deduplicate by canonical pair key
  5. Three-way random split (train ~97%, val ~1%, test ~2%)
  6. Decontamination: no molecule in test/val appears in any train pair
  7. Output separate jsonl files + odor group embedding cache

Monomer split:
  - TGSC compounds with BGE-M3 embeddings
  - Test set = compounds never seen in any blender (OOD), 10% cap
  - Train set = all remaining
"""

import sys; sys.path.insert(0, r'C:\Users\yangx\v3-mixture-token')

import json, random, sqlite3, os
from collections import defaultdict

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

random.seed(42)

# ─── Paths ───
DB = r'C:\Users\yangx\Documents\TGSC\tgsc_data.db'
GS_FILE = r'C:\Users\yangx\odor-pair\tgsc_bge_embeddings_1024dim_clean.json'
GROUP_EMB = r'C:\Users\yangx\odor-pair\odor_group_1024dim_cache.json'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MONO_TRAIN = os.path.join(OUT_DIR, 'tgsc_train_bge.jsonl')
MONO_TEST = os.path.join(OUT_DIR, 'tgsc_test_bge.jsonl')
BL_TRAIN = os.path.join(OUT_DIR, 'blender_train.jsonl')
BL_VAL = os.path.join(OUT_DIR, 'blender_val.jsonl')
BL_TEST = os.path.join(OUT_DIR, 'blender_test.jsonl')


# ═══════════════════════════════════════════════════════════════════
# 1. Monomer data: load BGE-M3 embedded TGSC library
# ═══════════════════════════════════════════════════════════════════

with open(GS_FILE) as f:
    lib = json.load(f)

emb_by_smi = {}
desc_by_smi = {}
for c in lib['compounds']:
    smi = c.get('smiles', '')
    if smi and c.get('embedding'):
        emb_by_smi[smi] = c['embedding']
        desc_by_smi[smi] = c.get('description', '')

print(f'GoodScents compounds with BGE-M3 embedding: {len(emb_by_smi)}')


# ═══════════════════════════════════════════════════════════════════
# 2. Blender data: extract, canonicalize, dedup, decontaminate
# ═══════════════════════════════════════════════════════════════════

def canonical_smiles(smi):
    """RDKit canonical SMILES; returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


conn = sqlite3.connect(DB)

# Extract raw blender pairs from TGSC database
rows = conn.execute('''
    SELECT c1.smiles, c2.smiles, b.odor_group
    FROM blenders b
    JOIN compounds c1 ON b.compound_url = c1.page_url
    JOIN compounds c2 ON b.blender_url = c2.page_url
    WHERE c1.smiles IS NOT NULL AND c1.smiles != ""
      AND c2.smiles IS NOT NULL AND c2.smiles != ""
      AND c1.smiles NOT LIKE "%.%" AND c2.smiles NOT LIKE "%.%"
      AND b.odor_group NOT LIKE "No%"
      AND LENGTH(c1.smiles) < 200
''').fetchall()
conn.close()

print(f'Raw blender pairs from TGSC: {len(rows)}')

# Step 1: RDKit canonical SMILES
canonicalized = []
bad_smi = 0
for smi_a, smi_b, og in rows:
    ca = canonical_smiles(smi_a)
    cb = canonical_smiles(smi_b)
    if ca is None or cb is None:
        bad_smi += 1
        continue
    canonicalized.append((ca, cb, og))
print(f'After RDKit canonicalization: {len(canonicalized)} ({bad_smi} dropped)')

# Step 2: Sort to canonical pair order (A,B) -> always sort so A <= B
canonical_pairs = []
for smi_a, smi_b, og in canonicalized:
    if smi_a <= smi_b:
        canonical_pairs.append((smi_a, smi_b, og))
    else:
        canonical_pairs.append((smi_b, smi_a, og))

# Step 3: Deduplicate by canonical pair key
dedup = {}
for smi_a, smi_b, og in canonical_pairs:
    key = (smi_a, smi_b)
    if key not in dedup:
        dedup[key] = og

# Convert to list and shuffle
blender_pairs = [(a, b, og) for (a, b), og in dedup.items()]
random.shuffle(blender_pairs)

print(f'After canonical-pair dedup: {len(blender_pairs)} pairs')

# Step 4: Collect all unique molecules in each pair
pair_mols = {}
for smi_a, smi_b, og in blender_pairs:
    pair_mols[(smi_a, smi_b)] = {smi_a, smi_b}

# Step 5: Three-way split with decontamination
# Strategy: first select test and val pairs, then remove any pair
# from train that shares a molecule with test/val.

n_total = len(blender_pairs)
n_test_target = int(n_total * 0.02)
n_val_target = int(n_total * 0.01)

# Select decontaminated test set: iteratively sample pairs, ensuring
# no molecule overlap with already-selected test/val sets.
def select_decontaminated(pairs, n_target, forbidden_mols):
    """Select n_target pairs without molecule overlap with forbidden_mols."""
    selected = []
    selected_mols = set()
    available = [p for p in pairs if not (set(p[:2]) & forbidden_mols)]
    random.shuffle(available)
    
    for smi_a, smi_b, og in available:
        if len(selected) >= n_target:
            break
        pair_set = {smi_a, smi_b}
        if not (pair_set & selected_mols):
            selected.append((smi_a, smi_b, og))
            selected_mols.update(pair_set)
    
    return selected, selected_mols

# Select test first (strict: no overlap with val or train)
test_pairs, test_mols = select_decontaminated(blender_pairs, n_test_target, set())
print(f'Test set: {len(test_pairs)} pairs (target: {n_test_target})')

# Select validation (no overlap with test, will exclude from train)
val_pairs, val_mols = select_decontaminated(blender_pairs, n_val_target, test_mols)
print(f'Val set: {len(val_pairs)} pairs (target: {n_val_target})')

# Train = all remaining pairs that don't share molecules with test/val
test_val_mols = test_mols | val_mols
train_pairs = [(a, b, og) for a, b, og in blender_pairs
               if not ({a, b} & test_val_mols)]
print(f'Train set: {len(train_pairs)} pairs')
print(f'Decontamination: {n_total} -> {len(train_pairs) + len(val_pairs) + len(test_pairs)} '
      f'(dropped {n_total - len(train_pairs) - len(val_pairs) - len(test_pairs)} cross-contaminated)')

# Step 6: Load odor group embeddings
with open(GROUP_EMB) as f:
    group_data = json.load(f)
group_names = list(group_data.keys())
print(f'Odor groups: {len(group_names)}')


# ═══════════════════════════════════════════════════════════════════
# 3. Monomer split (same logic as original: OOD test, 10% cap)
# ═══════════════════════════════════════════════════════════════════

# Collect all molecules appearing in any blender pair
all_blender_mols = set()
for smi_a, smi_b, _ in train_pairs + val_pairs + test_pairs:
    all_blender_mols.add(smi_a)
    all_blender_mols.add(smi_b)

# Monomer split: compounds NOT in any blender -> test pool; rest -> train
all_singles = [(smi, desc_by_smi.get(smi, ''))
               for smi in emb_by_smi if smi in desc_by_smi]
single_ood = [(smi, desc) for smi, desc in all_singles
              if smi not in all_blender_mols]
single_id = [(smi, desc) for smi, desc in all_singles
             if smi in all_blender_mols]

random.shuffle(single_ood)
n_te = min(int(len(emb_by_smi) * 0.1), len(single_ood))
single_test = single_ood[:n_te]
test_smis = set(s for s, _ in single_test)
single_train = [(smi, desc_by_smi.get(smi, ''))
                for smi in emb_by_smi
                if smi in desc_by_smi and smi not in test_smis]

print(f'Monomer: train={len(single_train)} test={len(single_test)} '
      f'(OOD pool={len(single_ood)}, ID pool={len(single_id)})')


# ═══════════════════════════════════════════════════════════════════
# 4. Write output files
# ═══════════════════════════════════════════════════════════════════

# Monomer train
with open(MONO_TRAIN, 'w') as f:
    for smi, desc in single_train:
        record = {'smiles': smi, 'description': desc, 'embedding': emb_by_smi[smi]}
        f.write(json.dumps(record) + '\n')

# Monomer test
with open(MONO_TEST, 'w') as f:
    for smi, desc in single_test:
        record = {'smiles': smi, 'description': desc, 'embedding': emb_by_smi[smi]}
        f.write(json.dumps(record) + '\n')

# Blender files
def write_blender(path, pairs):
    with open(path, 'w') as f:
        for smi_a, smi_b, og in pairs:
            record = {'smiles_a': smi_a, 'smiles_b': smi_b, 'odor_group': og}
            f.write(json.dumps(record) + '\n')

write_blender(BL_TRAIN, train_pairs)
write_blender(BL_VAL, val_pairs)
write_blender(BL_TEST, test_pairs)

# ─── Summary ───
print(f'\nSaved:')
for path in [MONO_TRAIN, MONO_TEST, BL_TRAIN, BL_VAL, BL_TEST]:
    count = sum(1 for _ in open(path))
    print(f'  {os.path.basename(path)}: {count} records')
