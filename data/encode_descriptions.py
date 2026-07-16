"""用本地 LM Studio Qwen3-4B 编码 TGSC 描述库（2048-dim）"""

import sys; sys.path.insert(0, r'C:\Users\yangx\v3-mixture-token')

import requests, json, time, os

EMB_PATH = r'C:\Users\yangx\odor-pair\tgsc_bge_embeddings_1024dim_backup.json'
GROUP_PATH = r'C:\Users\yangx\odor-pair\odor_group_2048dim_cache.json'
API = "http://127.0.0.1:1234/v1/embeddings"
OUT_PATH = r'C:\Users\yangx\odor-pair\tgsc_qwen3_embeddings_2048dim.json'

# 读取原库
with open(EMB_PATH) as f:
    lib = json.load(f)

# 提取所有 description
descs = [c['description'] for c in lib['compounds'] if c.get('description','').strip()]
print(f"总描述数: {len(descs)}")
print(f"去重数: {len(set(descs))}")

# 先验证 groups 是否存在
if os.path.exists(GROUP_PATH):
    with open(GROUP_PATH) as f:
        g = json.load(f)
    print(f"Group cache: {len(g)} groups, dim={len(list(g.values())[0])}")
else:
    print("⚠️ Group cache 不存在，先编码 groups...")
    with open(r'C:\Users\yangx\odor-pair\odor_group_1024dim_cache.json') as f:
        old_groups = json.load(f)
    group_names = list(old_groups.keys())
    resp = requests.post(API, json={
        "model": "text-embedding-qwen3-embedding-4b",
        "input": group_names
    }, timeout=120)
    data = resp.json()
    group_out = {name: item['embedding'] for name, item in zip(group_names, data['data'])}
    with open(GROUP_PATH, 'w') as f:
        json.dump(group_out, f)
    print(f"  Saved {len(group_out)} groups")

# 编码所有 description
print(f"\n编码 {len(descs)} 条 description (batch=8, 2560-dim)...")
t0 = time.time()
all_embs = []
BATCH = 8

for i in range(0, len(descs), BATCH):
    batch = descs[i:i+BATCH]
    success = False
    for retry in range(3):
        try:
            resp = requests.post(API, json={
                "model": "text-embedding-qwen3-embedding-4b",
                "input": batch
            }, timeout=120)
            data = resp.json()
            if 'data' in data:
                for item in data['data']:
                    all_embs.append(item['embedding'])
                success = True
                break
            else:
                print(f"  重试 {retry+1}: {json.dumps(data)[:200]}")
        except Exception as e:
            print(f"  重试 {retry+1}: {e}")
    if not success:
        print(f"  ❌ batch {i//BATCH} 失败，逐个编码")
        for text in batch:
            for r in range(3):
                try:
                    resp = requests.post(API, json={
                        "model": "text-embedding-qwen3-embedding-4b",
                        "input": [text]
                    }, timeout=120)
                    all_embs.append(resp.json()['data'][0]['embedding'])
                    break
                except:
                    if r == 2: all_embs.append([0.0]*2048)

    if (i // BATCH) % 25 == 0:
        elap = time.time() - t0
        rate = (i + len(batch)) / elap if elap > 0 else 0
        rem = (len(descs) - i - len(batch)) / rate if rate > 0 else 0
        print(f"  [{i+len(batch)}/{len(descs)}] {elap:.0f}s elapsed, ~{rem:.0f}s remaining")

dim = len(all_embs[0]) if all_embs else 0
print(f"\n编码完成: {len(all_embs)} 条, dim={dim}, 总耗时: {time.time()-t0:.0f}s")

# 对应回 compounds
desc_idx = 0
for c in lib['compounds']:
    if c.get('description','').strip():
        c['embedding'] = all_embs[desc_idx]
        desc_idx += 1

# 保存
lib_out = {
    "model": "text-embedding-qwen3-embedding-4b",
    "dimension": dim,
    "total": len(lib['compounds']),
    "compounds": lib['compounds'],
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(OUT_PATH, 'w') as f:
    json.dump(lib_out, f, ensure_ascii=False)

print(f"已保存: {OUT_PATH}")
print(f"  compounds: {len(lib_out['compounds'])}")
print(f"  dimension: {dim}")
print(f"  带嵌入的: {sum(1 for c in lib['compounds'] if c.get('embedding'))}")
