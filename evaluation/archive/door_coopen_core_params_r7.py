#!/usr/bin/env python3
"""核心段登錄檔 r7：r6＋approach_cost（Codex 20261027_011742／r13 裁定：接近段單一成本候選）。r6 不覆寫；其餘鍵值不變。
  w_a 0.01：Codex 指定單一試驗選值（非最佳值、非成功保證）；只在 APPROACH 經協同話題 coord[12] 送出，其他相位 NaN（求解端沿用啟動值 0.05）。
  latch_after_exit：本趟一旦離開 APPROACH 即不再送候選值。
  restore_deadline_s 0.55＝求解端 --coord-max-age-s 0.5 ＋ 一個求解週期 0.05（Codex r13 §2 有界恢復）。
  launch_w_a 0.05：求解節點啟動值（run 腳本 --w-a），J5 用以核對恢復。

    python3 evaluation/door_coopen_core_params_r7.py   # 寫 results/door/coopen_core_params_48_r7.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r6.json'), os.path.join(R, 'coopen_core_params_48_r7.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    p['approach_cost'] = {'w_a': 0.01, 'phases': ['APPROACH'], 'latch_after_exit': True, 'coord_index': 12,
                          'launch_w_a': 0.05, 'restore_deadline_s': 0.55}
    p['approach_cost_basis'] = {'w_a': 'Codex 20261027_011742 指定單一候選；合成對照 door_coopen_core_wa_cl_r1／osc_r1',
                                'restore_deadline_s': 'Codex r13 §2：coord_max_age 0.5 s＋一個求解週期 0.05 s',
                                'launch_w_a': '求解節點啟動值（run_door_coopen_core_g2.sh 既有 --w-a）'}
    p['schema'] = 'coopen_core_params/4'
    p['derived_from_r6'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
