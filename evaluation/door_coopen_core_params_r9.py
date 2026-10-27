#!/usr/bin/env python3
"""核心段登錄檔 r9：r8＋bench（Codex r19／r20：台架初態啟動；初始設定點預補償＋求解端偏移估計登錄初值）。r8 不覆寫；其餘鍵值不變。
  bench.q_nominal＝grasp_init.qa_grasp（名目初態，生成時寫入關節位置）；bench.d_init＝待驗證的初始化估計（door_coopen_bench_release_offline_r2.json：
    g3_1 第三次暫停 [25.28, 26.40) s 無接觸無穿透 112 步之 mean(q_arm_meas − arm_sp)；含動態追蹤落後，非已辨識靜態偏移）；
  bench.s_init＝q_nominal − d_init（實際寫入的致動器初始設定點）；bench.uncertainty＝每軸 max(區間 std, 三區間平均最大差)（非已證明上界）。
  bench.release_timeout_wall_s 120：凍結期等待首筆有效解的牆鐘上限（逾時 ⇒ 不釋放、中止）。
  其餘（w_a 0.05、d̂_b 關閉、閘控、15 s、保護）全同 r8／r6。

    python3 evaluation/door_coopen_core_params_r9.py   # 寫 results/door/coopen_core_params_48_r9.json（不覆寫）
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
SRC, OUT = os.path.join(R, 'coopen_core_params_48_r8.json'), os.path.join(R, 'coopen_core_params_48_r9.json')
OFF = os.path.join(R, 'door_coopen_bench_release_offline_r2.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    p = json.load(open(SRC))
    o = json.load(open(OFF))
    qn = [float(v) for v in p['grasp_init']['qa_grasp']]
    d = [float(v) for v in o['d_init']]
    if [round(v, 12) for v in o['q_nominal']] != [round(v, 12) for v in qn]:
        raise SystemExit('離線來源的 q_nominal 與 r8 qa_grasp 不同 ⇒ 不建')
    s_init = [a - b for a, b in zip(qn, d)]
    p['bench'] = {'q_nominal': qn, 'd_init': d, 's_init': s_init, 'uncertainty': [float(v) for v in o['d_init_uncertainty']],
                  'q_nominal_cli': ','.join(f'{v:.9f}' for v in qn), 's_init_cli': ','.join(f'{v:.9f}' for v in s_init),
                  'release_timeout_wall_s': 120.0,
                  'source': {'path': os.path.relpath(OFF, HERE), 'sha256': hashlib.sha256(open(OFF, 'rb').read()).hexdigest(),
                             'window': o['d_init_source'][0], 'cross_check_windows': o['d_init_source'][1:]},
                  'status': '待驗證的初始化估計（含動態追蹤落後；非已辨識靜態偏移）'}
    p['bench_basis'] = {'scope': 'Codex r19／r20：台架初態啟動；不偽稱停車閘門或導航交棒通過；初值不冒充觀測更新',
                        'offset_max_rad': '須 ≤ 求解端 --offset-max-rad 0.05（超界拒跑，不截斷）'}
    p['schema'] = 'coopen_core_params/6'
    p['derived_from_r8'] = {'path': os.path.relpath(SRC, HERE), 'sha256': hashlib.sha256(open(SRC, 'rb').read()).hexdigest()}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT, 's_init', p['bench']['s_init_cli'])


if __name__ == '__main__':
    main()
