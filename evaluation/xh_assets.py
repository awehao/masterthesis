#!/usr/bin/env python3
"""XH2：依封存登錄 XH1_registration.yaml 產生 13 個資產檔（R：drawer_unit/1；N：drawer_unit/2）。

R：由 drawer_unit_bar26.yaml 複製，只改 bar.radius＝d/2、bar.length＝L、支柱中心 x＝∓(L/2 − 0.015)、尺寸 (0.016, 0.040, d − 0.002)；
   其餘欄位（含 grasp_surface 舊欄位）照 bar26 原樣——標註與評估不讀這些欄位。
N：/2，無 handle；distractors 一個圓鈕（中心 x 0、z 0.55、根部貼前板外面 face_y = −0.245、直徑與突出量照登錄）。
輸出 src/my_omnibot_description/config/xh/drawer_unit_xh_<id>.yaml（已存在則拒絕覆寫）＋ results/vision/xh_assets.sha256。

    python3 evaluation/xh_assets.py
"""
import copy
import hashlib
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
SRC = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'drawer_unit_bar26.yaml')
OUT = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'xh')
REG = os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')
FACE_Y = -0.245


def make(base, a):
    s = copy.deepcopy(base)
    s['name'] = f'drawer_unit_xh_{a["id"]}'
    s['xh_asset_id'] = a['id']
    s['xh_split'] = a['split']
    if a['type'] == 'R':
        d, L = a['diameter_mm'] / 1000.0, a['length_mm'] / 1000.0
        h = s['drawer']['handle']
        h['bar']['radius'] = round(d / 2.0, 6)
        h['bar']['length'] = round(L, 6)
        h['posts'] = [['post_l', [round(-(L / 2 - 0.015), 6), -0.265, 0.550], [0.016, 0.040, round(d - 0.002, 6)]],
                      ['post_r', [round(L / 2 - 0.015, 6), -0.265, 0.550], [0.016, 0.040, round(d - 0.002, 6)]]]
        s['schema'] = 'drawer_unit/1'
    else:
        s['schema'] = 'drawer_unit/2'
        del s['drawer']['handle']
        s['drawer']['distractors'] = [{'name': 'knob', 'type': 'knob_cylinder_y', 'center_xz': [0.0, 0.550],
                                       'face_y': FACE_Y, 'diameter': a['knob_diameter_mm'] / 1000.0,
                                       'protrusion': a['protrusion_mm'] / 1000.0}]
    return s


def main():
    reg = yaml.safe_load(open(REG))
    base = yaml.safe_load(open(SRC))
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for a in reg['assets']:
        p = os.path.join(OUT, f'drawer_unit_xh_{a["id"]}.yaml')
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
        s = make(base, a)
        txt = (f'# XH2 自動產生（xh_assets.py）：資產 {a["id"]}（{a["type"]}，{a["split"]}）。由 drawer_unit_bar26.yaml 衍生；'
               f'只改登錄的尺寸／干擾物欄位。不要手改。\n') + yaml.safe_dump(s, allow_unicode=True, sort_keys=False)
        open(p, 'w').write(txt)
        lines.append(f'{hashlib.sha256(txt.encode()).hexdigest()}  {os.path.relpath(p, WS)}')
    open(os.path.join(HERE, 'results', 'vision', 'xh_assets.sha256'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
