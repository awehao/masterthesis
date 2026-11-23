#!/usr/bin/env python3
"""K3-M 停止後狀態的離線取樣（純離線；依 Codex 20261023_210920 第三節「停止可容許條件／減速保持」）。

**中止當下狀態 S0**：門角 q0 ∈ {1, 2, 4, 6, 8}°；底盤＝走廊參考＋內側帶四角或中心（±20 mm、±20 mm、±0.8°）；手臂 IK 以候選構型暖啟動並維持夾持。
**停止後狀態** S1＝S0 加上：
  - 底盤剩餘位移 δ（底盤座標 x、y、偏航，8 個符號組合）：取 door_k3m_offline_stop_r3_inner20.json 各反應情形「超出內帶」的最壞量
    （executor／chain_timeout／policy；位置以 x、y 各取該值、偏航取該值，保守地同時取滿）；
  - 手臂殘餘 Δq：設定點凍結 ≠ 實測關節凍結。模型 q⁺＝q＋α(s−q)＋α·d 的平衡為 q＝s＋d；接觸負載消失時平衡往 s＋d_free 移。
    取三種：0（負載不變）、−(d̂_abort − d̂_open)（只釋放 OPEN 期間增加的負載）、−d̂_abort（保守：全部釋放）；d̂ 取 door_k3ab_ev_1 實錄；
  - 門餘轉 Δc ∈ {0, 0.25, 0.5, 1, 2}°（實錄中止後 3 s 內 +0.87～+1.05°；經驗值，不是上界）。
每個 S1 核：nonfinger（含夾爪殼／相機對橫桿）、finger（不含橫桿）、base、self（door_k3m_geom_monitor 後端，自碰 full），
另列手指—橫桿（設計接觸區，負值＝穿入）。門檻同監看：5／5／20／5 mm。
界線：靜態取樣、近似幾何、門與手臂獨立擺放（沒有接觸力或門被推動的模型）；結果只用來判定原停止方式的幾何餘裕，不是接觸安全證明。

    python3 evaluation/door_k3m_offline_stop_states.py   # 寫 results/door/door_k3m_offline_stop_states.json（已存在則拒絕覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_offline_corridor as CR                 # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

C = GA.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_stop_states.json')
Q0 = [1.0, 2.0, 4.0, 6.0, 8.0]
INNER = (0.020, 0.020, 0.8)
COAST = [0.0, 0.25, 0.5, 1.0, 2.0]
QF = 0.00166
TH = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    stop = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_stop_r3_inner20.json')))
    t_ = json.load(open(os.path.join(CR.RUN, 'task.json')))
    T_hold = np.array(t_['H_frozen']) @ np.array(t_['grasp_basis']['T_HG'])
    ab = os.path.join(HERE, 'runs', 'door_k3ab_ev_1')
    tk, al = json.load(open(os.path.join(ab, 'task.json'))), json.load(open(os.path.join(ab, 'align_solver.json')))
    ph = {e['to']: e['t'] for e in tk['policy_events'] if 'to' in e}
    t_ab = next(e['t'] for e in tk['policy_events'] if 'abort' in e)
    sn = [x['solve_in'] for x in al['log'] if x.get('solve_in') and x['solve_in'].get('offset_d_hat') is not None]
    d_open = np.array([s for s in sn if s['snap_sim_t_exact'] <= ph['OPEN']][-1]['offset_d_hat'])
    d_ab = np.array([s for s in sn if s['snap_sim_t_exact'] <= t_ab][-1]['offset_d_hat'])
    dq_cases = {'none': np.zeros(6), 'load_release': -(d_ab - d_open), 'full_release': -d_ab}
    resid = {k: (max(0.0, v['pos_mm'] - stop['inner_band']['mm']) * 1e-3, max(0.0, v['yaw_deg'] - stop['inner_band']['deg']))
             for k, v in stop['cases'].items()}
    fn = GM.default_backend(self_mode='full')
    ax, ay, ath = rp['anchor']
    ca, sa = math.cos(ath), math.sin(ath)
    rows = []
    for q0 in Q0:
        ref = CR.ref_at(rp['candidate'], q0)
        T = rotate_about_hinge(T_hold, np.zeros(3), CR.AX, math.radians(q0))
        for corner in [(0, 0, 0)] + list(itertools.product((-1, 1), repeat=2)):
            df, dl = corner[0] * INNER[0], corner[1] * INNER[1]
            dy = (corner[0] * corner[1]) * INNER[2] if corner != (0, 0, 0) else 0.0
            fwd, lat = ref['fwd_m'] + df, ref['lat_m'] + dl
            b0 = np.array([ax + ca * fwd - sa * lat, ay + sa * fwd + ca * lat, ath + math.radians(ref['dyaw_deg'] + dy)])
            qf0 = C.ik(b0, T, np.array(ref['q_arm']), only_prev=True)
            if qf0 is None:
                rows.append({'q0': q0, 'corner': corner, 'ik': False})
                continue
            for rn, (rp_, ry) in resid.items():
                for sg in itertools.product((-1, 1), repeat=3):
                    cb, sb = math.cos(b0[2]), math.sin(b0[2])
                    dx_b, dy_b = sg[0] * rp_, sg[1] * rp_
                    b1 = np.array([b0[0] + cb * dx_b - sb * dy_b, b0[1] + sb * dx_b + cb * dy_b, b0[2] + sg[2] * math.radians(ry)])
                    for dn, dq in dq_cases.items():
                        qa = np.asarray(qf0[3:9], float) + dq
                        qf = np.concatenate([b1, qa])
                        for dc in COAST:
                            qd = math.radians(q0 + dc)
                            d = fn(qf, qd, QF)
                            parts, _ = GA.parts_clearance(qf, qd, QF)
                            fbar = min(parts[f]['min_to_bar_m'] for f in GA.FINGERS)
                            ok = all(d[c][0] >= TH[c] for c in TH)
                            rows.append({'q0': q0, 'corner': corner, 'ik': True, 'reaction': rn, 'sign': sg, 'dq': dn, 'coast_deg': dc,
                                         **{c: d[c][0] for c in TH}, 'who_nonfinger': d['nonfinger'][1], 'finger_bar_m': fbar, 'ok': ok})
    ok_rows = [r for r in rows if r['ik']]
    summ = {}
    for rn in resid:
        for dn in dq_cases:
            for dc in COAST:
                xs = [r for r in ok_rows if r['reaction'] == rn and r['dq'] == dn and r['coast_deg'] == dc]
                summ[f'{rn}|{dn}|{dc}'] = {'n': len(xs), 'n_ok': sum(1 for r in xs if r['ok']),
                                           'min': {c: min(r[c] for r in xs) for c in list(TH) + ['finger_bar_m']},
                                           'worst_nonfinger_who': min(xs, key=lambda r: r['nonfinger'])['who_nonfinger']}
    res = {'schema': 'door_k3m_offline_stop_states/1', 'd_hat_open': d_open.tolist(), 'd_hat_abort': d_ab.tolist(),
           'dq_cases': {k: v.tolist() for k, v in dq_cases.items()}, 'base_residual_beyond_inner': resid, 'coast_deg': COAST,
           'thresholds': TH, 'n_states': len(ok_rows), 'n_ik_fail': sum(1 for r in rows if not r['ik']), 'summary': summ, 'rows': rows,
           'note': '靜態取樣、近似幾何、門與手臂獨立擺放（無接觸力、門被推動或夾持變形的模型）；只判定原停止方式（凍結設定點＋底盤減速）的幾何餘裕。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('d̂ open (mrad)', np.round(d_open * 1e3, 2), 'abort', np.round(d_ab * 1e3, 2))
    print('底盤超出內帶的剩餘（m, °）', resid)
    for k, v in summ.items():
        m = v['min']
        print(f"{k:34s} ok {v['n_ok']:3d}/{v['n']:3d}  非手指 {m['nonfinger'] * 1e3:6.1f}（{v['worst_nonfinger_who']}） 手指 {m['finger'] * 1e3:5.1f} "
              f"底盤 {m['base'] * 1e3:5.0f} 自碰 {m['self'] * 1e3:5.1f} 指—桿 {m['finger_bar_m'] * 1e3:6.1f}")


if __name__ == '__main__':
    main()
