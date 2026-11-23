#!/usr/bin/env python3
"""XH2：確定性觀測路徑（每資產兩條：A_front、B_oblique）。採集前產生並記雜湊；不因結果換路徑。

沿用凍結的 dl0_paths（**匯入、不改**）：IK（底盤朝向觀測參考中心、六關節最小平方、關節內縮 0.05 rad）、5 Hz 設計姿態、
逐物理步線性內插（不含最後端點）、前距選擇規則＝在候選 (0.25, 0.30, 0.35, 0.40, 0.45, 0.50) m 中**先取 IK 不合格格數最少、
再取最大位置誤差最小；同分保留候選順序中較前者**（dl0_paths.make_seq 以 tuple `<` 比較）。
逐序列設定 dl0_paths 的模組變數：H_C＝觀測參考中心（R：橫桿中心；N：圓鈕外端面中心）、LEN＝橫桿長度（N 以圓鈕直徑作 FK 粗估）、OUT＝本包輸出目錄。
相位 φ = 2π·(seed mod 1000)/1000（seed＝登錄的資產種子）。

    python3 evaluation/xh_paths.py      # 輸出 results/vision/xh_paths/*.json 與 results/vision/XH2_paths_registration.yaml（已存在則拒絕覆寫）
"""
import hashlib
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_paths as P                                   # noqa: E402  凍結版（只設定其模組變數，不改程式）
import drawer_asset_v2 as DA                            # noqa: E402

REG = os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')
ASSET_DIR = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'xh')
OUT = os.path.join(HERE, 'results', 'vision', 'xh_paths')
DRAWER_POSE = (0.0, 1.45)
X_HINT = np.array([1.0, 0.0, 0.0])


def ref_center(spec):
    ox, oy = DRAWER_POSE
    h = spec['drawer'].get('handle')
    if h is not None:
        c = h['bar']['center']
        return np.array([c[0] + ox, c[1] + oy, c[2]]), float(h['bar']['length']), 'R：橫桿中心'
    g = DA.knob_geometry(spec['drawer']['distractors'][0])
    c = g['outer_face_center']
    return np.array([c[0] + ox, c[1] + oy, c[2]]), float(spec['drawer']['distractors'][0]['diameter']), 'N：圓鈕外端面中心'


def templates(ref, seed):
    phi = 2 * math.pi * (seed % 1000) / 1000.0
    sm = P.smooth

    def dA(i):
        return sm(3.0, 0.4, i / 60) if i <= 60 else sm(0.4, 0.2, (i - 60) / 20)

    A = dict(n=81, cam_fn=lambda i: ref + np.array([0.04 * math.sin(2 * math.pi * 0.1 * (0.2 * i) + phi), -dA(i), 0.03]),
             aim_fn=lambda i: ref, x_hint_fn=lambda i: X_HINT, roll_fn=lambda i: 0.0,
             design='正面：d = smooth(3.0→0.4, i/60)（i ≤ 60）、smooth(0.4→0.2, (i−60)/20)；x 偏移 0.04·sin(2π·0.1·t + φ)；高 +0.03 m；全程朝參考中心')
    u = np.array([math.sin(math.radians(30)), -math.cos(math.radians(30)), 0.0])

    def dB(i):
        return 2.0 if i < 15 else sm(2.0, 0.3, (i - 15) / 50)

    def camB(i):
        return ref + dB(i) * u + np.array([0.0, 0.0, 0.06])

    def aimB(i):
        away = camB(i) + 1.5 * np.array([0.6, 0.8, -0.1])
        return sm(away, ref, (i - 10) / 5)
    B = dict(n=66, cam_fn=camB, aim_fn=aimB, x_hint_fn=lambda i: X_HINT, roll_fn=lambda i: 0.0,
             design='斜向：方位 u = (sin30°, −cos30°, 0)（相機在 +x 側前方）；d = 2.0（i < 15）、smooth(2.0→0.3, (i−15)/50)；高 +0.06 m；'
                    '觀看目標 smooth(朝外點, 參考中心, (i−10)/5)，朝外點＝相機 + 1.5·(0.6, 0.8, −0.1)')
    return {'A_front': A, 'B_oblique': B}, phi


def main():
    regp = os.path.join(HERE, 'results', 'vision', 'XH2_paths_registration.yaml')
    if os.path.exists(regp):
        raise SystemExit(f'{regp} 已存在，不覆寫')
    reg = yaml.safe_load(open(REG))
    os.makedirs(OUT, exist_ok=True)
    P.OUT = OUT
    out = {'schema': 'xh2_paths_registration/1', 'date': '2026-10-06',
           'rule': ('沿用凍結 dl0_paths 的 IK、內插與前距選擇（先最少 IK 不合格格數、再最小最大位置誤差、同分取候選順序較前者）；'
                    '設計姿態數 ≠ 物理步數（內插不含最後端點：步數 = (設計姿態數 − 1) × 20）≠ 實際擷取新影格數（另有暖機與拒絕，擷取後分開記錄）'),
           'drawer_pose': list(DRAWER_POSE), 'not_rendered_until_freeze': [a['id'] for a in reg['assets'] if a['split'] == 'test'],
           'sequences': {}}
    for a in reg['assets']:
        spec = DA.load(os.path.join(ASSET_DIR, f'drawer_unit_xh_{a["id"]}.yaml'))
        ref, length, ref_kind = ref_center(spec)
        P.H_C = ref
        P.LEN = length
        T, phi = templates(ref, int(a['seed']))
        for tname, t in T.items():
            name = f'{a["id"]}_{tname}'
            info, frames = P.make_seq(name, t['n'], t['cam_fn'], t['x_hint_fn'], t['roll_fn'], t['aim_fn'])
            info.update({'asset': a['id'], 'split': a['split'], 'template': tname, 'design': t['design'],
                         'observation_ref_center_world': ref.tolist(), 'ref_kind': ref_kind, 'phase_rad': phi,
                         'n_design_poses': t['n'], 'fk_prediction_note': ('N：以圓鈕直徑作虛擬線段的 FK 粗估，只供參考' if a['type'] == 'N' else 'R：橫桿兩端 FK 投影')})
            out['sequences'][name] = info
            print(name, info['n_frames_5hz'], info['n_phys_steps'], 'IK bad', len(info['ik_bad_frames']), 'pos max', info['ik_pos_err_mm_max'])
    yaml.safe_dump(out, open(regp, 'w'), allow_unicode=True, sort_keys=False, width=200)
    lines = [f"{v['sha256']}  {v['file']}" for v in out['sequences'].values()]
    open(os.path.join(HERE, 'results', 'vision', 'xh_paths.sha256'), 'w').write('\n'.join(lines) + '\n')
    print('寫出', regp)


if __name__ == '__main__':
    main()
