import re

ref_text = {
    '1': 'Chen et al., 2024',
    '2': 'Dosovitskiy et al., 2021',
    '3': 'Tom et al., 2025',
    '4': 'Keller et al., 2017',
    '5': 'Kurfalı et al., 2025',
    '6': 'Lee et al., 2023',
    '7': 'Leenhouts et al., 2025',
    '8': 'Hamel et al., 2024',
    '9': 'Loshchilov & Hutter, 2017',
    '10': 'Magnasco et al., 2015',
    '11': 'Omatu et al., 2012',
    '12': 'Partin et al., 2026',
    '13': 'Poivet et al., 2018',
    '14': 'Sanchez-Lengeling et al., 2019',
    '15': 'Sisson et al., 2025',
    '16': 'Vaswani et al., 2017',
    '17': 'Vinyals et al., 2016',
    '18': 'Yang & Daescu, 2026',
    '19': 'Xu et al., 2019',
    '20': 'Zhu et al., 2024',
    '21': 'Rajaonson et al., 2025',
    '22': 'Chen et al., 2025',
    '23': 'Zhang et al., 2024',
    '24': 'Samanta et al., 2025',
}

with open('paper_draft_v2.8.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Strategy: use regex to find ^[number(s)] and replace with corresponding text
def replace_cite(m):
    inner = m.group(1)  # e.g. "4" or "10,6" or "18,22,24"
    nums = [n.strip() for n in inner.split(',')]
    texts = [ref_text[n] for n in nums]
    return '[' + '; '.join(texts) + ']'

# Replace ^[...] with [...]  (only where ^ precedes [)
# Use negative lookbehind to avoid matching things like "value^2"
new_content = re.sub(r'\^\[([0-9,\s]+)\]', replace_cite, content)

# Count changes
import difflib
count = sum(1 for a, b in zip(content, new_content) if a != b)
print(f"Changed {count} characters")

with open('paper_draft_v2.8.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Done.")
