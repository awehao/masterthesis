#!/usr/bin/env python3
"""核心段登錄檔 r5：r4（door_coopen_core_build.py 產出）＋沿用鏈所需的 pause／solver_change／executor_change 三段（照抄 w1f_params_60_r2.json，
即 W1f2 鏈通過 door_wide_w1f_60_3 時的設定：p4b 主動保持停止、ADM9 偏移保持、ev5 暫停期手指上限 20 N）。r4 不覆寫；其餘鍵值不變。

    python3 evaluation/door_coopen_core_params_r5.py   # 寫 results/door/coopen_core_params_48_r5.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, BASE = os.path.join(R, 'coopen_core_params_48_r4.json'), os.path.join(R, 'w1f_params_60_r2.json')
OUT = os.path.join(R, 'coopen_core_params_48_r5.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p, b = json.load(open(SRC)), json.load(open(BASE))
    for k in ('pause', 'solver_change', 'executor_change'):
        p[k] = b[k]
    p['schema'] = 'coopen_core_params/2'
    p['derived_from'] = {'r4': os.path.relpath(SRC, HERE), 'r4_sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest(),
                         'chain_settings_from': os.path.relpath(BASE, HERE), 'chain_settings_sha256': hashlib.sha256(open(BASE, 'rb').read()).hexdigest(),
                         'copied_keys': ['pause', 'solver_change', 'executor_change']}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
