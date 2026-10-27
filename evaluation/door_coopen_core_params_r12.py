#!/usr/bin/env python3
"""核心段登錄檔 r12：r11＋control_cost_change.closure_guidance（Codex r35／r36：C2 持握閉合修正引導；參數為 A/B 前固定值，未依結果調整）。r11 不覆寫。
  cfg＝door_coopen_closure_guidance2.CFG（36→40° smoothstep 啟用、τ 1.0 s、輸出變化率範數 2 mm/s、輸出幅度 8 mm、無效輸入沿用上限 0.2 s）；
  依據＝離線 A/B（v3 A／B、v4 B）結果檔 sha256；限制：離線門角由 TCP 推得、不驗承載與內力、不保證接觸實跑收益。

    python3 evaluation/door_coopen_core_params_r12.py   # 寫 results/door/coopen_core_params_48_r12.json（不覆寫）
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_closure_guidance2 as CG2            # noqa: E402

R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r11.json'), os.path.join(R, 'coopen_core_params_48_r12.json')
AB = [os.path.join(R, f) for f in ('door_coopen_wgmpc_v3_A.json', 'door_coopen_wgmpc_v3_B.json', 'door_coopen_wgmpc_v4_B.json')]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    if 'closure_guidance' in p['control_cost_change']:
        raise SystemExit('r11 已含 closure_guidance ⇒ 不建')
    ab = {os.path.basename(f): json.load(open(f)) for f in AB}
    p['control_cost_change'] = dict(p['control_cost_change'], closure_guidance={
        'enabled': True, 'cfg': vars(CG2.CFG), 'module': 'door_coopen_closure_guidance2.py', 'module_sha256': sha(CG2.__file__),
        'rule': '只改引導參考位姿的平移（ShiftedTable）；前饋、偏航、k_lin／k_yaw、v_cap／w_cap 不變；只在 OPEN／SETTLE 且 free、抓取基準已建立時更新',
        'basis': {'source': 'Codex r35（選 C2）／r36（足以進入版本化接線；不保證接觸實跑收益）',
                  'ab': {k: {'sha256': sha(os.path.join(R, k)), 'closure_max_40_48': v.get('closure_max_40_48'), 'D_drop_38_48_m': v.get('D_drop_38_48_m'),
                             'geo_min_mm': v.get('geo_min_mm')} for k, v in ab.items()},
                  'limit': '離線門角由 TCP 推得；無接觸、門動力學、E2／安全層；不驗兩指承載或內力；底盤取樣淨空 72.8→70.5 mm（仍高於門檻）'}})
    p['schema'] = 'coopen_core_params/9'
    p['derived_from_r11'] = {'path': os.path.relpath(SRC, HERE), 'sha256': sha(SRC)}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, p['control_cost_change']['closure_guidance']['cfg'])


if __name__ == '__main__':
    main()
