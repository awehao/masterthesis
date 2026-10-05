#!/usr/bin/env python3
"""DL0：把自動標籤預先算成三值圖（0 背景、1 前景＝可見遮罩、2 ignore）並建索引。只含訓練／開發群組；**封存測試不得列入**。

輸出（不入版本控制）：evaluation/dl0_ckpt/labels/<group>__<capture>__f<n>.png（uint8 三值）＋ index.json
index 每列：key、split（train／dev）、group、capture、n、rgb、label、flags、dist_m、kind（positive／negative／excluded）
  positive：前景非空；negative：no_handle_in_view；excluded：有預期把手但無可評估前景（例如全被近裁切）

    python3 evaluation/dl0_label_cache.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dl0_autolabel as AL  # noqa: E402

SPLIT = {'static_v0': 'train', 'traj_c0b1_p3r_H5': 'train', 'traj_d1s4b_M': 'train', 'synth_dev_near': 'train',
         'traj_wg4b_f02_P': 'dev', 'traj_mt_b1_02_P': 'dev'}
OUT = os.path.join(HERE, 'dl0_ckpt', 'labels')


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cap, grp, _ in AL.SOURCES:
        assert 'dl0_test' not in cap, '封存測試不得列入'
        if grp not in SPLIT:
            continue
        for fr in AL.frames_of(cap):
            if fr['c'] is None:
                continue
            L = AL.label(fr)
            lab = np.zeros((AL.H, AL.W), np.uint8)
            lab[L['ignore']] = 2
            lab[L['mask']] = 1
            key = f"{grp}__{cap.split('/')[0]}__f{fr['n']}"
            lp = os.path.join(OUT, key + '.png')
            Image.fromarray(lab).save(lp)
            kind = ('negative' if 'no_handle_in_view' in L['flags'] else
                    'positive' if L['mask'].any() else 'excluded')
            rows.append({'key': key, 'split': SPLIT[grp], 'group': grp, 'capture': cap, 'n': fr['n'],
                         'rgb': os.path.relpath(fr['rgb'], HERE), 'label': os.path.relpath(lp, HERE),
                         'flags': L['flags'], 'dist_m': round(L['dist_m'], 4), 'kind': kind})
    idx = {'schema': 'dl0_label_index/1', 'label_code': {'0': 'background', '1': 'handle_bar（可見）', '2': 'ignore'},
           'autolabel_sha256': hashlib.sha256(open(os.path.join(HERE, 'dl0_autolabel.py'), 'rb').read()).hexdigest(),
           'rows': rows}
    json.dump(idx, open(os.path.join(OUT, 'index.json'), 'w'), ensure_ascii=False, indent=1)
    from collections import Counter
    print(Counter((r['split'], r['kind']) for r in rows))
    print(Counter(r['group'] for r in rows))


if __name__ == '__main__':
    main()
