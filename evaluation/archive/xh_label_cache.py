#!/usr/bin/env python3
"""XH3 標籤快取：train／dev 20 條擷取 → 每格標籤 PNG（0 背景、1 目標、2 ignore；N 圓鈕為背景）＋干擾物遮罩 PNG＋索引。

輸出 evaluation/xh_ckpt/labels/（gitignore；已存在則拒絕覆寫）。test 資產不處理。標註＝dl0_autolabel_v2（PC2 +0.5、資產尺寸）。

    python3 evaluation/xh_label_cache.py
"""
import hashlib
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402

OUT = os.path.join(HERE, 'xh_ckpt', 'labels')
REG = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    os.makedirs(OUT)
    rows, sha = [], hashlib.sha256()
    for a in REG['assets']:
        if a['split'] == 'test':
            continue
        spec = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/xh', f'drawer_unit_xh_{a["id"]}.yaml'))
        for tmpl in ('A_front', 'B_oblique'):
            seq = f'xh_{a["id"]}_{tmpl}'
            for f in V2.frames_of_xh(os.path.join(HERE, 'runs', seq, 'wrist_v0'), spec):
                lab = V2.label_xh(f)
                L = np.zeros((480, 640), np.uint8)
                L[lab['mask']] = 1
                L[lab['ignore'] & ~lab['mask']] = 2
                lp = os.path.join(OUT, f'{seq}_f{f["n"]:02d}.png')
                dp = os.path.join(OUT, f'{seq}_f{f["n"]:02d}_distractor.png')
                Image.fromarray(L).save(lp)
                Image.fromarray(lab['distractor'].astype(np.uint8)).save(dp)
                sha.update(L.tobytes())
                neg_reason = None
                if lab['kind'] == 'negative':
                    neg_reason = 'N_asset' if a['type'] == 'N' else 'no_target_in_view'
                rows.append({'key': f'{seq}:{f["n"]}', 'group': a['id'], 'asset_type': a['type'], 'split': a['split'], 'template': tmpl,
                             'n': f['n'], 'kind': lab['kind'], 'neg_reason': neg_reason, 'dist_m': lab.get('dist_m'),
                             'rgb': os.path.relpath(f['rgb'], HERE), 'label': os.path.relpath(lp, HERE),
                             'distractor': os.path.relpath(dp, HERE), 'distractor_px': int(lab['distractor'].sum())})
    idx = {'schema': 'xh_label_index/1', 'labeler': 'dl0_autolabel_v2.py', 'labeler_sha256': hashlib.sha256(open(V2.__file__, 'rb').read()).hexdigest(),
           'encoding': '0 背景（含 N 圓鈕）、1 目標、2 ignore', 'labels_sha256': sha.hexdigest(), 'rows': rows}
    json.dump(idx, open(os.path.join(OUT, 'index.json'), 'w'), ensure_ascii=False, indent=1)
    from collections import Counter
    print(len(rows), Counter((r['split'], r['asset_type'], r['kind']) for r in rows))


if __name__ == '__main__':
    main()
