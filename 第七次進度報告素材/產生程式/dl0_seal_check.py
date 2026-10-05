#!/usr/bin/env python3
"""DL0 封存測試序列的技術完整性檢查（不產生標籤、不跑辨識、不讀影像像素）。

只讀 meta.json：影格數、擷取端拒絕、rendering_time 嚴格遞增且不重複、位姿時刻與擷取時刻差、檔案齊全；
並把擷取目錄內全部檔案的 sha256 寫進封存清單。

    python3 evaluation/dl0_seal_check.py runs/dl0_test_lateral runs/dl0_test_view
"""
import hashlib
import json
import os
import sys

out = {'schema': 'dl0_seal/1', 'rule': '封存：模型與門檻凍結前不查看影像與辨識結果；只做技術完整性檢查', 'runs': {}}
lines = []
for R in sys.argv[1:]:
    V = os.path.join(R, 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    fr = m['frames']
    rt = [f['rendering_time'] for f in fr]
    dpose = [abs(f['pose_time'] - f['rendering_time']) for f in fr if f.get('pose_time') is not None]
    miss = [f['n'] for f in fr for suf in ('_rgb.png', '_depth_m.npy')
            if not os.path.exists(os.path.join(V, 'frames', f"f{f['n']:02d}{suf}"))]
    out['runs'][os.path.basename(R.rstrip('/'))] = {
        'n_frames': len(fr), 'capture_rejected': len(m.get('rejected_frames') or []),
        'rendering_time_strictly_increasing': all(b > a for a, b in zip(rt, rt[1:])),
        'rendering_time_unique': len(set(rt)) == len(rt),
        'pose_time_minus_render_max_s': (max(dpose) if dpose else None),
        'missing_files': miss, 'replay_source': (m.get('replay') or {}).get('source')}
    for root, _, files in os.walk(R):
        for f in sorted(files):
            p = os.path.join(root, f)
            lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))}")
    print(os.path.basename(R.rstrip('/')), out['runs'][os.path.basename(R.rstrip('/'))])
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'vision')
json.dump(out, open(os.path.join(base, 'DL0_test_seal_check.json'), 'w'), ensure_ascii=False, indent=1)
open(os.path.join(base, 'seal_dl0_test.sha256'), 'w').write('\n'.join(sorted(lines, key=lambda x: x.split('  ')[1])) + '\n')
print('封存清單', len(lines), '檔')
