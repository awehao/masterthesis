#!/usr/bin/env python3
"""XH2 擷取核對：資產雜湊／幾何、R／N 真值欄位、影格與時間匹配、標註可讀取（v2 frames_of_xh＋label_xh）。

    python3 evaluation/xh_capture_check.py <序列名> [...]     # 逐條印出並寫 runs/xh_<序列名>/analysis/xh_capture_check.json（已存在則拒絕覆寫）
"""
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402

REG = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH2_paths_registration.yaml')))
ASSET_SHA = {l.split()[1]: l.split()[0] for l in open(os.path.join(HERE, 'results', 'vision', 'xh_assets.sha256')) if l.strip()}


def main():
    for seq in sys.argv[1:]:
        r = REG['sequences'][seq]
        D = os.path.join(HERE, 'runs', f'xh_{seq}')
        out_p = os.path.join(D, 'analysis', 'xh_capture_check.json')
        if os.path.exists(out_p):
            raise SystemExit(f'{out_p} 已存在，不覆寫')
        rel = f'src/my_omnibot_description/config/xh/drawer_unit_xh_{r["asset"]}.yaml'
        spec = DA.load(os.path.join(WS, rel))
        sim = open(os.path.join(D, 'sim.log'), errors='replace').read()
        meta = json.load(open(os.path.join(D, 'wrist_v0', 'meta.json')))
        truth = json.load(open(os.path.join(D, 'wrist_v0', 'truth.json')))
        rep = {'seq': seq, 'asset': r['asset'], 'split': r['split']}
        rep['asset_sha_expected'] = ASSET_SHA[rel]
        rep['asset_sha_file_now'] = hashlib.sha256(open(os.path.join(WS, rel), 'rb').read()).hexdigest()
        rep['asset_path_in_sim_log'] = rel in sim
        rep['no_bar_message'] = '無橫桿資產' in sim
        fr = meta['frames']
        ts = [f['rendering_time'] for f in fr]
        rep['frames'] = {'n_design_poses': r['n_design_poses'], 'n_phys_steps': r['n_phys_steps'], 'n_new_frames': len(fr),
                         'n_rejected': len(meta.get('rejected_frames') or []), 'fresh_check': meta.get('fresh_frames_check'),
                         'pose_minus_render_max_s': max(abs(f['pose_time'] - f['rendering_time']) for f in fr) if fr else None,
                         'render_time_range': [min(ts), max(ts)] if ts else None}
        rep['truth_target_kind'] = truth.get('target_kind')
        is_R = spec['drawer'].get('handle') is not None
        exp = (np.array([0.0, 1.45 - 0.285, 0.55]) if is_R else
               np.array([0.0, 1.45 - 0.245 - spec['drawer']['distractors'][0]['protrusion'], 0.55]))
        key = 'handle_center_world_at_capture' if is_R else 'observation_ref_center_world_at_capture'
        vals = np.array([f[key] for f in fr if f.get(key) is not None])
        rep['truth_field'] = key
        rep['truth_field_present'] = f'{len(vals)}/{len(fr)}'
        rep['truth_vs_expected_max_mm'] = float(np.abs(vals - exp).max()) * 1e3 if len(vals) else None
        rep['has_handle_center_field'] = any('handle_center_world_at_capture' in f for f in fr)
        try:
            frs = list(V2.frames_of_xh(os.path.join(D, 'wrist_v0'), spec))
            kinds = collections.Counter()
            npx = []
            for f in frs:
                lab = V2.label_xh(f)
                kinds[lab['kind']] += 1
                npx.append(int(lab['mask'].sum()) if is_R else int(lab['distractor'].sum()))
            rep['labels'] = {'n': len(frs), 'kinds': dict(kinds), 'target_or_distractor_px_median': float(np.median(npx)) if npx else None}
        except V2.EvidenceError as e:
            rep['labels'] = {'evidence_error': str(e)}
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        json.dump(rep, open(out_p, 'w'), ensure_ascii=False, indent=1, default=str)
        print(json.dumps(rep, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
