#!/usr/bin/env python3
"""往後拉開 48° 登錄 r1（Howard 10/28「不要邊繞，往後拉開」；Codex 20261028_195306：以 W1 正面開門配置為起點，不在 coopen r11 上只換走廊表）。
欄位格式同 coopen_core_params（任務節點 v10／執行端 ev9r8／求解端 ADM20／bench1 跑法讀同一組鍵）；數值來源：
  θ 相關（task_param_extension、protection_scope_change：走廊表 door_wide_corridor_table_90、包絡、觀察器上限）＝door_w1b_params.py 48 推得的 w1b_params_48_r2.json（規則不手調）；
  control_cost_change（引導 v_cap 0.02、w_cap 0.05、OPEN＋SETTLE）＝w1f_params_60_r2（W1 實跑配置）；
  grasp＝正面抓法基準 runs/phf_01_M/task.json（w1f_60_3 所用，無 T_HG_full ⇒ 任務節點走正面 T_HG、a_H (0,−1,0)）；
  start.stand＝w1f_60_3 停位（run.log 登錄 0.6,−0.5332,π/2；與走廊表錨點（W1 起始實測）差 ≤ 1e-3）；k2_ready／q_pre 沿 W1 實跑（台架起始不經過，僅供跑法讀取）；
  grasp_init.qa_grasp＝w1f_60_3 夾持靜止窗 [18.5, 20.4) s（閉爪、雙指接觸、門角 < 0.03°、求解端接手前）實測關節平均；
  bench.d_init＝同窗 mean(q_arm_meas − arm_sp)；uncertainty＝每軸 max(窗內 std, 與閉爪前窗 [15.6, 16.9) 平均差)（非已證明上界）；
  executor_change（暫停期手指上限 20 N）＝W1 實跑配置（正面抓法暫停期開度約 1.67 mm 的位置驅動需求，W1e 依據）；
  pause／solver_change（ADM 偏移保持＋開門段更新）／approach_gate／bench_basis＝coopen r11（求解核心與命令鏈沿用，Codex 同意）。

    python3 evaluation/door_pullback_params_r1.py   # 寫 results/door/pullback_params_48_r1.json（不覆寫）
"""
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
R = os.path.join(HERE, 'results', 'door')
W1B = os.path.join(R, 'w1b_params_48_r2.json')
W1F = os.path.join(R, 'w1f_params_60_r2.json')
R11 = os.path.join(R, 'coopen_core_params_48_r11.json')
GB = os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')
RUN = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')
OUT = os.path.join(R, 'pullback_params_48_r1.json')
WIN, WIN_X = (18.5, 20.4), (15.6, 16.9)
STAND = [0.6, -0.5332, math.pi / 2]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def window(S, I, E, w):
    ks = [k for k, s in enumerate(S) if w[0] <= s[1] < w[1]]
    q = np.array([S[k][I['q_arm_meas']] for k in ks], float)
    d = np.array([np.asarray(S[k][I['q_arm_meas']], float) - np.asarray(E[S[k][I['step']]]['arm_sp'], float) for k in ks], float)
    door = max(abs(math.degrees(S[k][I['door_q_rad']])) for k in ks)
    fn = min(min(S[k][I['finger_contact_n']] or [0.0]) for k in ks)
    return {'window_s': list(w), 'n': len(ks), 'q_mean': q.mean(0), 'q_std': q.std(0), 'd_mean': d.mean(0), 'd_std': d.std(0), 'door_abs_max_deg': door, 'finger_n_min': fn}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    w1b, w1f, r11 = json.load(open(W1B)), json.load(open(W1F)), json.load(open(R11))
    rr = json.load(open(os.path.join(RUN, 'room_run.json')))
    S, C = rr['steps'], rr['steps_cols']
    I = {c: i for i, c in enumerate(C)}
    E = {e['step']: e for e in map(json.loads, open(os.path.join(RUN, 'evidence.jsonl'))) if e.get('arm_sp') is not None}
    a, b = window(S, I, E, WIN), window(S, I, E, WIN_X)
    if a['door_abs_max_deg'] > 0.05 or a['finger_n_min'] < 0.5 or float(np.max(a['q_std'])) > 0.002:
        raise SystemExit(f'夾持靜止窗不合格：門 {a["door_abs_max_deg"]}°、手指 {a["finger_n_min"]} N、std {np.max(a["q_std"])}')
    anchor = json.load(open(os.path.join(WS, w1b['protection_scope_change']['corridor_table'])))['table']['anchor']
    if max(abs(x - y) for x, y in zip(anchor, STAND)) > 1e-3:                        # 錨點＝W1 起始實測（偏航差 0.34 mrad）
        raise SystemExit(f'停位與走廊表錨點不符 {anchor}')
    qn = a['q_mean'].tolist()
    d = a['d_mean'].tolist()
    unc = np.maximum(a['d_std'], np.abs(a['d_mean'] - b['d_mean'])).tolist()
    s_init = [x - y for x, y in zip(qn, d)]
    if max(abs(v) for v in d) > 0.05:
        raise SystemExit('d_init 超過求解端 --offset-max-rad 0.05 ⇒ 不建')
    task = json.load(open(os.path.join(RUN, 'task.json')))['args']
    p = {'schema': r11['schema'], 'variant': 'pullback_w1', 'theta_goal_deg': 48.0,
         'task_param_extension': w1b['task_param_extension'],
         'grasp': {'basis': os.path.relpath(GB, WS), 'basis_sha256': sha(GB), 'kind': 'front（正面抓法；無 T_HG_full）'},
         'protection_scope_change': w1b['protection_scope_change'],
         'control_cost_change': w1f['control_cost_change'],
         'start': {'stand': STAND, 'stand_cli': '0.600000,-0.533200,1.570796', 'k2_ready_cli': r11['start']['k2_ready_cli'],
                   'q_pre_cli': None, 'experimental_initialization': r11['start']['experimental_initialization'],
                   'experimental_initialization_iso': '手臂以 --stow-q 直接生成於 qa_grasp（手指張開）；不經 K2_READY／APPROACH',
                   'w1_run_args': {k: task.get(k) for k in ('grasp_basis', 'corridor_table', 'w1b_params')}},
         'unchanged': r11['unchanged'], 'pause': r11['pause'], 'solver_change': r11['solver_change'],
         'executor_change': w1f['executor_change'], 'approach_gate': r11['approach_gate'], 'approach_gate_basis': r11['approach_gate_basis'],
         'grasp_init': {'qa_grasp': qn, 'qa_grasp_cli': ','.join(f'{v:.6f}' for v in qn), 'init_check_timeout_s': r11['grasp_init']['init_check_timeout_s'],
                        'source': {'path': os.path.relpath(os.path.join(RUN, 'room_run.json'), HERE), 'window_s': list(WIN), 'q_std': a['q_std'].tolist(),
                                   'door_abs_max_deg': a['door_abs_max_deg'], 'finger_n_min': a['finger_n_min']}},
         'grasp_init_basis': {'qa_grasp': 'w1f_60_3 夾持靜止窗實測關節平均（實際抓到把手的構型，含重力下垂）', 'scope': '指定初態隔離驗證；跳過自主接近，不代表接近已解決'},
         'bench': {'q_nominal': qn, 'd_init': d, 's_init': s_init, 'uncertainty': unc,
                   'q_nominal_cli': ','.join(f'{v:.9f}' for v in qn), 's_init_cli': ','.join(f'{v:.9f}' for v in s_init),
                   'release_timeout_wall_s': r11['bench']['release_timeout_wall_s'],
                   'source': {'path': os.path.relpath(os.path.join(RUN, 'evidence.jsonl'), HERE), 'sha256': sha(os.path.join(RUN, 'evidence.jsonl')),
                              'window': list(WIN), 'cross_check_window': list(WIN_X), 'cross_check_d_mean': b['d_mean'].tolist()},
                   'status': '待驗證的初始化估計（W1 求解端接手前、閉爪靜止；非已辨識靜態偏移）'},
         'bench_basis': r11['bench_basis'],
         'derived_from': {'w1b_params_48_r2': sha(W1B), 'w1f_params_60_r2': sha(W1F), 'coopen_core_params_48_r11': sha(R11),
                          'source_note': 'Codex 20261028_195306'}}
    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)
    print('qa_grasp', p['grasp_init']['qa_grasp_cli'], 'd_init', np.round(d, 5).tolist(), 'unc', np.round(unc, 5).tolist())
    print('envelope', w1b['protection_scope_change']['envelope']['cli'], 'obs', w1b['protection_scope_change'].get('observer_door_max_deg'), 'timeout', w1b['task_param_extension']['open_timeout_s'])


if __name__ == '__main__':
    main()
