#!/usr/bin/env python3
"""核心段登錄檔 r6：r5＋approach_gate（Codex 20261027_003024 §2：接近進度閘控單一候選）。r5 不覆寫；其餘鍵值不變。
  perp_pause_m 0.003／perp_resume_m 0.002／resume_hold_s 0.5：Codex 登錄單一候選，不掃描；
  along_resume_m 0.005＝K2Config.pre_tol_m；rot_resume_rad 0.02＝求解器 reach_rot_rad；align_confirm_s 0.5＝K2Config.grip_settle_s（既有值）。
  APPROACH 逾時沿用 K3AConfig timeouts['APPROACH']＝15 s，不重置、不放寬。

    python3 evaluation/door_coopen_core_params_r6.py   # 寫 results/door/coopen_core_params_48_r6.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r5.json'), os.path.join(R, 'coopen_core_params_48_r6.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    p['approach_gate'] = {'perp_pause_m': 0.003, 'perp_resume_m': 0.002, 'resume_hold_s': 0.5,
                          'along_resume_m': 0.005, 'rot_resume_rad': 0.02, 'align_confirm_s': 0.5}
    p['approach_gate_basis'] = {'perp_pause_m／perp_resume_m／resume_hold_s': 'Codex 20261027_003024 登錄單一候選',
                                'along_resume_m': 'K2Config.pre_tol_m', 'rot_resume_rad': '求解器 reach_rot_rad', 'align_confirm_s': 'K2Config.grip_settle_s',
                                'approach_timeout_s': 'K3AConfig timeouts APPROACH 15 s（不改）'}
    p['schema'] = 'coopen_core_params/3'
    p['derived_from_r5'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
