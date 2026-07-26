"""生成清洗后的 TGSC 描述并编码（BGE-M3）"""
import sqlite3, json, re, requests, numpy as np
from collections import OrderedDict

DB = r'C:\Users\yangx\Documents\TGSC\tgsc_data.db'
API = "http://127.0.0.1:1234/v1/embeddings"
OUT = r'C:\Users\yangx\odor-pair\tgsc_bge_embeddings_1024dim_clean.json'

STOP = {'at','in','on','the','a','an','of','and','or','for','with','to','from','by','is','it','be','are','was','not','no','as','its'}
BAD = {'dipropylene','glycol','triethyl','citrate','ethanol','propylene','benzyl','benzoate',
       'diethyl','phthalate','triacetin','carbitol','solvent','dilution','diluted',
       'available','product','strength','recommend','recommended','smelling','terpenes',
       'folded','terpeneless','concentrate','replacer','artificial','synthetic','ppm',
       'oakmoss','absolute'}  # "absolute" as in "absolute oil" not odor

def clean(raw):
    if not raw: return ''
    t = raw
    t = re.sub(r'Odor Type:\s*', '', t)
    t = re.sub(r'Odor Strength:\s*(?:medium|high|low|strong|weak)?\s*', '', t)
    t = re.sub(r'Substantivity:?\s*(?:>)?\s*[\d.]+\s*hour\(s\)\s*at\s+[\d.]+\s*%\s*', '', t)
    t = re.sub(r'Substantivity:?\s*(?:>)?\s*[\d.]+\s*hour\(s\)\s*', '', t)
    t = re.sub(r'Odor Description:\s*', '', t)
    t = re.sub(r'Flavor Type:\s*', '', t)
    t = re.sub(r'Taste Description:\s*', '', t)
    t = re.sub(r'Odor and/or flavor descriptions? from others \([^)]*\)\.?\s*', '', t)
    t = re.sub(r'Firmenich\s+[^.]*\.?\s*', '', t)
    t = re.sub(r'at\s+[\d.]+\s*%', '', t)
    t = re.sub(r'in\s+dipropylene\s+glycol', '', t)
    t = re.sub(r'recommend\s+smelling\s+in\s+a?\s*[\d.]+\s*%\s*solution\s+or\s+less', '', t, flags=re.IGNORECASE)
    # names + companies
    t = re.sub(r'luebke[\s,]*william[\s,\d]*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'mosciano[\s,]*gerard[\s,\d]*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'sigma[\s-]*aldrich', '', t, flags=re.IGNORECASE)
    t = re.sub(r'tgsc[\s,\d]*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'odor\s+sample\s+from:?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'p\s*&\s*f\s+no[.\s]*\d*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'flavors?\s*(&\s*)?\s*fragr?ances?', '', t, flags=re.IGNORECASE)
    for co in ['givaudan','firmenich','iff','bedoukian','berje','harrmann reimer',
               'symrise','takasago','quest international','noville',
               'haarmann reimer','rhone poulenc','bush boake allen','fritzsche dodge',
               'v mane fils','roure bertrand dupont','naarden','dragoco','florasynth']:
        t = re.sub(co, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(?:inc|corp|ltd|llc|co|laboratories?|limited|corporation)[\s,]*', ' ', t, flags=re.IGNORECASE)
    # tokenize + filter
    words = t.split()
    result = []
    for w in words:
        w = w.strip(' ,;.()[]')
        wl = w.lower()
        if len(wl) <= 1: continue
        if wl in STOP or wl in BAD: continue
        if re.match(r'^[\d.]+%?$', wl): continue
        if wl.endswith('like') or wl.endswith('nuance'): continue  # "vanilla-like", "citrus nuance"
        result.append(wl)
    # global dedup preserving order
    seen = set()
    out = []
    for w in result:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return ' '.join(out)

# 处理所有化合物
conn = sqlite3.connect(DB)
rows = conn.execute('''
    SELECT id, name, smiles, odor_description FROM compounds
    WHERE smiles IS NOT NULL AND smiles != ""
      AND odor_description IS NOT NULL AND odor_description != ""
      AND odor_description LIKE "%Odor Type%"
''').fetchall()
conn.close()

print(f"处理 {len(rows)} 个化合物...")
compounds = []
empty_count = 0
for rid, name, smi, raw in rows:
    c = clean(raw)
    if not c:
        empty_count += 1
        continue
    compounds.append({'id': rid, 'name': name, 'smiles': smi, 'description': c})

print(f"有效: {len(compounds)}, 清洗后为空: {empty_count}")
print(f"\n示例清洗后描述:")
for c in compounds[:8]:
    print(f"  {c['name'][:35]:<35s} → [{c['description']}]")

# 编码
print(f"\n编码 {len(compounds)} 条描述（BGE-M3）...")
for i in range(0, len(compounds), 10):
    batch = compounds[i:i+10]
    texts = [c['description'] for c in batch]
    try:
        r = requests.post(API, json={"model": "text-embedding-bge-m3", "input": texts}, timeout=60)
        for item, c in zip(r.json()['data'], batch):
            c['embedding'] = item['embedding']
    except Exception as e:
        print(f"  batch {i} 失败: {e}")
    if (i//10) % 10 == 0:
        print(f"  {i+10}/{len(compounds)}...")

# 保存
lib_out = {
    'model': 'text-embedding-bge-m3',
    'dimension': 1024,
    'total': len(compounds),
    'compounds': [{'name': c['name'], 'smiles': c['smiles'], 'description': c['description'], 'embedding': c['embedding']} for c in compounds if c.get('embedding')],
}
with open(OUT, 'w') as f:
    json.dump(lib_out, f, ensure_ascii=False)
print(f"\n已保存: {OUT}")
print(f"  维度: {lib_out['dimension']}")
print(f"  数量: {lib_out['total']}")
print(f"  带嵌入: {len(lib_out['compounds'])}")
