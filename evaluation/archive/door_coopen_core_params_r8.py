#!/usr/bin/env python3
"""核心段登錄檔 r8：r6＋grasp_init（Howard 2026-10-27 確認先驗證核心動作；Codex r15 補充裁定：指定抓取初態的隔離驗證）。
**由 r6 派生（不含 r7 approach_cost）**：接近段 w_a 0.01 不沿用到持握或開門段。r6、r7 不覆寫；其餘鍵值不變。
  grasp_init.qa_grasp：既有候選的門關閉抓取構型（coopen_core_h0_r4.json qa_grasp；離線開指淨空見 clearance_offline_mm），手指張開生成。
  grasp_init.init_check_timeout_s 15.0：沿用 K3AConfig timeouts['APPROACH']（同一期限，不新增時間）。
  閉爪准入沿用 K2Config grip_tol_m 0.002／grip_settle_s 0.5（只記錄，不在此改）。
  experimental_initialization：跳過自主接近 ⇒ 只能稱「指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證」。

    python3 evaluation/door_coopen_core_params_r8.py   # 寫 results/door/coopen_core_params_48_r8.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r6.json'), os.path.join(R, 'coopen_core_params_48_r8.json')
H0 = os.path.join(R, 'coopen_core_h0_r4.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    h0 = json.load(open(H0))
    qa = [float(v) for v in h0['qa_grasp']]
    if [float(v) for v in h0['stand']] != [float(v) for v in p['start']['stand']]:
        raise SystemExit('h0 停位與 r6 停位不同 ⇒ 不建')
    p['grasp_init'] = {'qa_grasp': qa, 'qa_grasp_cli': ','.join(f'{v:.6f}' for v in qa), 'init_check_timeout_s': 15.0,
                       'source': {'path': os.path.relpath(H0, HERE), 'sha256': hashlib.sha256(open(H0, 'rb').read()).hexdigest(), 'key': 'qa_grasp'},
                       'clearance_offline_mm': h0['grasp_clear_mm'], 'grip_tol_m': 0.002, 'grip_settle_s': 0.5}
    p['grasp_init_basis'] = {'qa_grasp': 'coopen_core_h0_r4.json（候選 0° 抓取構型，grasp_ik_mode continuation）',
                             'init_check_timeout_s': 'K3AConfig timeouts APPROACH 15 s（沿用）',
                             'scope': 'Howard 2026-10-27／Codex r15：指定初態隔離驗證；跳過自主接近，不代表接近已解決'}
    p['start']['experimental_initialization_iso'] = '手臂以 --stow-q 直接生成於 qa_grasp（手指張開）；展開目標＝同一構型（零位移）；不經 ALIGN／APPROACH'
    p['schema'] = 'coopen_core_params/5'
    p['derived_from_r6'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, 'qa_grasp', p['grasp_init']['qa_grasp_cli'])


if __name__ == '__main__':
    main()
