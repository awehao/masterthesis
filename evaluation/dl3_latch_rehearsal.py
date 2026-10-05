#!/usr/bin/env python3
"""DL3 離線鎖定演練：用既有 S4 實錄 d1s4b_M（凍結 D1 線上結果＋擷取影格），不新增擷取、不用真值選參數。

依 S4 節點紀錄的 processed 事件：t_obs＝擷取 stamp、t_recv＝輸出模擬時間 out_sim_t；t_latch＝task.json 的 ALIGN 進入時刻。
每格以 DL2 估計端（N-obs）產生把手位姿，再以 dl3_latch 規則鎖定。真值只在最後回報誤差。

    python3 evaluation/dl3_latch_rehearsal.py     # 輸出 results/vision/DL3_latch_rehearsal.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import dl0_autolabel as AL                      # noqa: E402
import dl2_eval as EV                           # noqa: E402
import dl2_obs_to_target as E                   # noqa: E402
import dl3_latch as LT                          # noqa: E402
import object_target_geometry as OT             # noqa: E402

RUN = 'd1s4b_M'


def main():
    out_p = os.path.join(HERE, 'results', 'vision', 'DL3_latch_rehearsal.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    D = os.path.join(HERE, 'runs', RUN)
    task = json.load(open(os.path.join(D, 'task.json')))
    t_latch = next(e['sim_t'] for e in task['events'] if e.get('phase') == 'ALIGN')
    ev = [json.loads(l) for l in open(os.path.join(D, 'd1_shadow', 'd1_shadow.jsonl'))]
    proc = [e for e in ev if e.get('ev') == 'processed']
    frs = {f['n']: f for f in AL.frames_of(f'{RUN}/wrist_live')}
    Rg = EV.R_grasp()
    T_HG, a_H = E.baseline_grasp(Rg)
    ests, rows = [], []
    for e in proc:
        t_obs = e['stamp'][0] + e['stamp'][1] * 1e-9
        if not (t_latch - LT.W_S - 1.0 <= t_obs <= t_latch + 1.0):      # 只算鎖定窗附近（含窗外各 1 s 以示排除）
            continue
        fr = frs.get(e['n'])
        obs = {'path': 'G0', 'L1': e['L1'], 'L2': e['L2'], 'reject': e['reject'], 'reject_L2': e['reject_L2'],
               'depth': (AL.load_depth(fr['depth_path']) if fr else None), 'K': (fr['K'] if fr else None),
               'T_cam': (fr['T'] if fr else None), 't_obs': t_obs}
        r = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=t_obs)
        ests.append({'n': e['n'], 't_obs': t_obs, 't_recv': e['out_sim_t'], 'ok': r['ok'], 'why': r.get('why'),
                     'T_WH': (np.asarray(r['T_WH']).tolist() if r['ok'] else None)})
        rows.append({'n': e['n'], 't_obs': t_obs, 't_recv': e['out_sim_t'], 'level': e.get('level'), 'ok': r['ok'], 'why': r.get('why')})
    L = LT.latch(ests, t_latch, Rg, T_HG, a_H)
    res = {'schema': 'dl3_latch_rehearsal/1', 'run': RUN, 't_latch': t_latch, 'params': L['params'],
           'note': '離線演練；參數為登錄的工程候選（一致性篩選，非精度保證）；不自動擴窗；真值只在 eval 欄',
           'frames_near_window': rows, 'latch': {k: (np.asarray(v).tolist() if k in ('T_WH', 'T_WE_s') else v)
                                                 for k, v in L.items()}}
    if L['ok']:
        # 評估（真值只在此）：鎖定時最近一格的真值把手中心（物體靜止）
        tr = [json.loads(l) for l in open(os.path.join(D, 'wrist_live', 'truth.jsonl'))]
        c_true = np.asarray(min(tr, key=lambda t: abs(t['t_cap'] - t_latch))['handle_center_world_at_capture'], float)
        T_true = np.eye(4)
        T_true[:3, 3] = c_true
        Tt = OT.object_target(T_true, T_HG, a_H, 0.03)['T_WE']
        res['eval_truth_only'] = {'center_err_mm': float(np.linalg.norm(L['T_WH'][:3, 3] - c_true) * 1e3),
                                  'target_s_err_mm': float(np.linalg.norm(L['T_WE_s'][:3, 3] - Tt[:3, 3]) * 1e3),
                                  'target_s_err_deg': float(np.degrees(OT.geodesic(L['T_WE_s'][:3, :3], Tt[:3, :3])))}
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res['latch'].get(k) for k in ('ok', 'why', 'n_selected', 'spread', 'selected_n', 'ages_s')}, ensure_ascii=False))
    print('eval', res.get('eval_truth_only'))
    import collections
    print('窗內影格去向', collections.Counter(r.get('excluded', 'selected') for r in L['considered']))


if __name__ == '__main__':
    main()
