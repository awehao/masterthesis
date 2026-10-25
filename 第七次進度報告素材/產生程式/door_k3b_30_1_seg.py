#!/usr/bin/env python3
"""door_wide_k3b_30_1 張爪後分段分析與退離時機離線對照（Codex 20261026_021025 §3；只讀本趟紀錄，不需 Isaac）。

一、分段（room 逐物理步）：
  S0 張爪前基準   [t_open−0.5 s, t_open)
  S1 張爪→求解端 FAIL   [t_open, t_fail)        t_fail＝求解端暫停監督器 FAIL（moved_in_held）
  S2 求解端 FAIL→task 中止  [t_fail, t_abort)
  S3 task 中止→執行端閂鎖   [t_abort, t_latch)    t_latch＝執行端 W2 權威 abort 事件步
  S4 中止後  [t_latch, 紀錄末]
  每段：門角起訖、門速（門角差分；逐步與 0.1 s 窗）、手指開度、各連桿接觸力、腕力、手臂最大關節速（0.1 s 窗）、
  底盤線速／角速（0.1 s 窗）、實際套用命令（底盤命令、手臂命令、套用來源序號、權威模式）。
  門速只由門角差分求得；不擬合、不外推總餘轉角（Codex：此段有接觸與命令變化，不能當作無接觸、固定阻尼的自由衰減）。

二、退離時機對照（離線、重放本趟實測門角與手指開度）：
  A 現行（實測）：手臂不退離，即本趟中止後實際狀態。
  B 確認後立即退離：t_start＝執行端解除確認 t_conf＋延遲 d_B。
  C 張爪同步退離：t_start＝t_open＋一個控制週期。**只作離線比較，不代表准許越過解除准入條件。**
  - 退離量：沿固定把手座標 H(q_rel) 的退離軸 A_H，s(t)＝v·(t − t_start)，v＝0.02 m/s（策略同值），上限 0.10 m。
  - 工具位姿：本步實測 TCP 位姿再平移 s(t)·A_H（保留實測手臂回彈、底盤漂移等誤差），以 IK（only_prev、底盤固定為本步實測）求手臂。
  - 門角、手指開度取本步實測。前提：退離不改變門的運動。中止後 finger1 曾有一筆 0.43 N 接觸，之後的門角含該接觸影響，須揭露。
  - 每 5 物理步（任務端 20 Hz）以任務節點同一登錄設定重算：
    - 一般幾何監看（GeometryMonitor）：nonfinger／finger／self／base 的 d 與 d_pred；
    - 指定配對 PairGeom 的兩指 d 與 d_pred；
    - 門角帶 Δ_pred（0.1 s 差分、T_pred 0.2 s）對上界 band(s)。
    - state_age 取 0.01 s（本趟 trace 中位量級；揭露）。
  - 退出近區時刻：s ≥ 0.02 且兩指配對 d、d_pred ≥ 5 mm 首次成立。

    python3 evaluation/door_k3b_30_1_seg.py   # 寫 evaluation/runs/door_wide_k3b_30_1/analysis/door_k3b_30_1_seg.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402
import door_k3b_pair_geom as PG                        # noqa: E402
import door_k3b_release4 as KR                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_k3b_30_1')
OUT = os.path.join(RUN, 'analysis', 'door_k3b_30_1_seg.json')
V_RET, S_END, T_CYC = 0.02, 0.10, 0.05
STATE_AGE = 0.01
D_B = T_CYC + 0.13 + T_CYC          # 任務端得知確認（一週期）＋請求→提交（本趟 mode 1／2 實測 0.13／0.11 s 取大）＋首筆退離目標生效（一週期）
WIN = 0.1


def ld(f):
    return json.load(open(os.path.join(RUN, f)))


def win_rate(rows, i, f, win=WIN):
    """以 i 往前 ≥ win 的最近一列求 |Δ|/Δt；f(row) 回傳向量或純量。"""
    t1 = rows[i][1]
    j = i
    while j > 0 and t1 - rows[j][1] < win - 1e-9:
        j -= 1
    if j == i:
        return None
    a, b = np.atleast_1d(np.asarray(f(rows[j]), float)), np.atleast_1d(np.asarray(f(rows[i]), float))
    return (b - a) / (t1 - rows[j][1])


def segments(room, task, solver):
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = room['steps']
    rel = task['k3b']['rel']
    t_open = rel['t']['open']
    t_fail = solver['stats']['k3m_pause']['supervisor_outcome']['transitions']['FAIL']
    t_abort = next(e['t'] for e in task['policy_events'] if e.get('abort'))
    ab = next(e for e in room['w2']['auth']['events'] if e['kind'] == 'abort')
    t_latch, k_latch = ab['t'], ab['step']
    t_end = S[-1][1]
    pubs = [e for e in solver['log'] if e.get('published')]
    bounds = [('S0_pre', t_open - 0.5, t_open), ('S1_open_to_solver_fail', t_open, t_fail), ('S2_solver_fail_to_task_abort', t_fail, t_abort),
              ('S3_task_abort_to_exec_latch', t_abort, t_latch), ('S4_after_latch', t_latch, t_end + 1e-6)]
    out = []
    for name, a, b in bounds:
        idx = [i for i, s in enumerate(S) if a - 1e-6 <= s[1] < b - 1e-6]
        if not idx:
            out.append({'seg': name, 't0': a, 't1': b, 'n_steps': 0})
            continue
        qd = [math.degrees(S[i][I['door_q_rad']]) for i in idx]
        step_rate = [abs(math.degrees(S[i][I['door_q_rad']] - S[i - 1][I['door_q_rad']]) / (S[i][1] - S[i - 1][1])) for i in idx if i > 0]
        arm_w = [np.max(np.abs(r)) for r in (win_rate(S, i, lambda s: s[I['q_arm_meas']]) for i in idx) if r is not None]
        bw = [win_rate(S, i, lambda s: s[I['base_xyth']]) for i in idx]
        b_lin = [math.hypot(r[0], r[1]) for r in bw if r is not None]
        b_ang = [abs(r[2]) for r in bw if r is not None]
        door_w = [abs(r[0]) for r in (win_rate(S, i, lambda s: math.degrees(s[I['door_q_rad']])) for i in idx) if r is not None]
        cmax = {}
        for i in idx:
            for link, v in (S[i][I['leaf_contacts']] or {}).items():
                for kk, f in v.items():
                    if isinstance(f, (int, float)) and f > 0:
                        key = f'{link}:{kk}'
                        if f > cmax.get(key, (0.0, None))[0]:
                            cmax[key] = (float(f), S[i][0])
        wr = [S[i][I['wrist_force_n']] for i in idx if isinstance(S[i][I['wrist_force_n']], (int, float))]
        src = sorted({(S[i][I['applied_src']][2], S[i][I['applied_src']][3]) for i in idx})
        bcmd = [S[i][I['base_cmd_body']] for i in idx]
        acmd = [S[i][I['applied_arm']] for i in idx]
        out.append({
            'seg': name, 't0': a, 't1': b, 'step0': S[idx[0]][0], 'step1': S[idx[-1]][0], 'n_steps': len(idx),
            'door_deg': [qd[0], qd[-1]], 'door_delta_deg': qd[-1] - qd[0],
            'door_rate_step_deg_s': {'max': max(step_rate) if step_rate else None, 'mean': (float(np.mean(step_rate)) if step_rate else None)},
            'door_rate_win0.1_deg_s': {'max': max(door_w) if door_w else None, 'last': door_w[-1] if door_w else None},
            'finger_q_m': [S[idx[0]][I['q_finger']], S[idx[-1]][I['q_finger']]],
            'contact_max_n': {k: {'n': v[0], 'step': v[1]} for k, v in sorted(cmax.items())},
            'wrist_n': {'max': max(wr) if wr else None, 'min': min(wr) if wr else None},
            'arm_joint_rate_win0.1_rad_s_max': max(arm_w) if arm_w else None,
            'base_win0.1': {'lin_max_m_s': max(b_lin) if b_lin else None, 'ang_max_rad_s': max(b_ang) if b_ang else None},
            'applied_src_seq_mode': src,
            'base_cmd_body': {'first': bcmd[0], 'last': bcmd[-1], 'max_abs': [max(abs(c[k]) for c in bcmd) for k in range(3)]},
            'arm_cmd_max_abs': max(max(abs(x) for x in c) for c in acmd),
        })
    base0 = next(s for s in S if s[0] == k_latch)[I['base_xyth']]
    base1 = S[-1][I['base_xyth']]
    after = {'from_step': k_latch, 'to_step': S[-1][0], 'base_shift_mm': 1e3 * math.hypot(base1[0] - base0[0], base1[1] - base0[1]),
             'base_yaw_deg': math.degrees(math.atan2(math.sin(base1[2] - base0[2]), math.cos(base1[2] - base0[2])))}
    ev = {'t_open': t_open, 'k_open': rel['step_open'], 't_solver_last_publish': pubs[-1].get('publish_sim_t_exact'),
          'solver_last_publish_source_seq': pubs[-1].get('pause_source_seq'), 't_solver_fail': t_fail, 't_task_abort': t_abort,
          't_exec_latch': t_latch, 'k_exec_latch': k_latch, 't_exec_confirm': room['w2']['release']['t'].get('B_CONFIRMED'),
          't_exec_release_failed': room['w2']['release']['t'].get('failed'), 'record_end': t_end}
    return ev, out, after


def compare(room, task, ev):
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = room['steps']
    by = {s[0]: s for s in S}
    k_open = ev['k_open']
    s0 = by[k_open]
    q_rel = float(s0[I['door_q_rad']])
    Hr = KC._H_of(q_rel)
    E_rel = np.linalg.inv(Hr) @ KC._tcp(s0[I['base_xyth']], s0[I['q_arm_meas']])
    _gc = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    C = V2.C
    starts = {'A_no_retreat_measured': None, 'B_retreat_after_confirm': ev['t_exec_confirm'] + D_B, 'C_retreat_at_open': ev['t_open'] + T_CYC}
    res = {}
    for name, t_start in starts.items():
        gm = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(_gc['v_floor_registered']), ds_err=dict(_gc['downsampling']['ds_err_registered'])),
                                GM.default_backend())
        pg = PG.PairGeom()
        rows, qhist, n_ik_fail, ik_err = [], [], 0, 0.0
        exit_t = None
        for s in S:
            if s[0] < k_open or (s[0] - k_open) % 5:
                continue
            t = float(s[1])
            sv = 0.0 if t_start is None or t < t_start else min(S_END, V_RET * (t - t_start))
            base = list(s[I['base_xyth']])
            qa = np.array(s[I['q_arm_meas']], float)
            if sv > 0.0:
                Tm = KC._tcp(base, qa)
                Tt = Tm.copy()
                Tt[:3, 3] = Tm[:3, 3] + Hr[:3, :3] @ (sv * KR.A_H)
                qf_ik = C.ik(np.array(base), Tt, qa, only_prev=True)            # 以本步實測手臂為唯一種子
                if qf_ik is None:
                    n_ik_fail += 1
                    continue
                qa = np.asarray(qf_ik, float)[C.IDX]
                ik_err = max(ik_err, float(np.linalg.norm(KC._tcp(base, qa)[:3, 3] - Tt[:3, 3])))
            qf = [float(x) for x in base + list(qa)]
            fq = [float(x) for x in s[I['q_finger']]]
            q = float(s[I['door_q_rad']])
            dec, det = gm.step(t, qf, q, float(sum(fq) / 2.0), STATE_AGE)
            pst, pd, pdp, _ = pg.step(t, qf, q, fq, STATE_AGE)
            E = np.linalg.inv(Hr) @ KC._tcp(base, qa)
            sm = float((E[:3, 3] - E_rel[:3, 3]) @ KR.A_H)
            qhist.append((t, q))
            past = [x for x in qhist if x[0] <= t - 0.1 + 1e-9]
            rate = (q - past[-1][1]) / (t - past[-1][0]) if past else 0.0
            dq = math.degrees(q - q_rel)
            dq_pred = dq + math.degrees(max(rate, 0.0)) * KR.REG['t_pred_s']
            up = KR.band_upper_deg(max(sm, 0.0))
            cl = det.get('classes') or {}
            row = {'step': s[0], 't': t, 's_cmd': sv, 's_meas': sm, 'dec': dec, 'why': det.get('why'),
                   'nonfinger_mm': [1e3 * cl['nonfinger']['d'], 1e3 * cl['nonfinger']['d_pred']] if 'nonfinger' in cl else None,
                   'finger_mm': [1e3 * cl['finger']['d'], 1e3 * cl['finger']['d_pred']] if 'finger' in cl else None,
                   'self_mm': [1e3 * cl['self']['d'], 1e3 * cl['self']['d_pred']] if 'self' in cl else None,
                   'pair_mm': None if pd is None else [1e3 * min(pd), 1e3 * min(pdp)], 'pair_each_mm': None if pd is None else [1e3 * x for x in pd],
                   'dq_deg': dq, 'dq_pred_deg': dq_pred, 'band_up_deg': up, 'band_margin_deg': up - dq_pred}
            rows.append(row)
            if exit_t is None and sm >= KR.REG['near_s_m'] and row['pair_mm'] is not None and min(row['pair_mm']) >= 1e3 * KR.REG['pair_exit_m']:
                exit_t = t
        def mn(key, j):
            v = [r[key][j] for r in rows if r[key] is not None]
            return min(v) if v else None
        first_stop = next((r for r in rows if r['dec'] == 'stop'), None)
        first_band = next((r for r in rows if r['band_margin_deg'] < 0), None)
        first_pair_touch = next((r for r in rows if r['pair_mm'] is not None and r['pair_mm'][0] <= 0.0), None)
        res[name] = {
            't_start': t_start, 'n_rows': len(rows), 'n_ik_fail': n_ik_fail, 'ik_pos_err_max_mm': 1e3 * ik_err, 's_max': max(r['s_meas'] for r in rows),
            'min_nonfinger_mm': [mn('nonfinger_mm', 0), mn('nonfinger_mm', 1)], 'min_finger_mm': [mn('finger_mm', 0), mn('finger_mm', 1)],
            'min_self_mm': [mn('self_mm', 0), mn('self_mm', 1)], 'min_pair_mm': [mn('pair_mm', 0), mn('pair_mm', 1)],
            'min_band_margin_deg': min(r['band_margin_deg'] for r in rows),
            'first_geom_stop': None if first_stop is None else {k: first_stop[k] for k in ('step', 't', 'why', 's_meas', 'nonfinger_mm', 'pair_mm')},
            'first_band_violation': None if first_band is None else {k: first_band[k] for k in ('step', 't', 's_meas', 'dq_pred_deg', 'band_up_deg')},
            'first_pair_le_0': None if first_pair_touch is None else {k: first_pair_touch[k] for k in ('step', 't', 's_meas', 'pair_each_mm')},
            'near_exit_t': exit_t, 'rows': rows,
        }
    return {'q_rel_deg': math.degrees(q_rel), 'v_retreat_m_s': V_RET, 'd_B_s': D_B, 'state_age_s': STATE_AGE, 'scenarios': res}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    room, task, solver = ld('room_run.json'), ld('task.json'), ld('align_solver.json')
    ev, seg, after = segments(room, task, solver)
    cmp_ = compare(room, task, ev)
    res = {'version': 'door_k3b_30_1_seg/1', 'events': ev, 'segments': seg, 'after_latch_base': after, 'compare': cmp_,
           'caveats': ['門速只由門角差分求得，不擬合、不外推總餘轉角',
                       '對照 B／C 假設退離不改變門的運動；步 7268 起門角含 finger1 0.43 N 接觸影響',
                       '對照 B／C 以實測 TCP 平移後 IK（底盤取實測），不模擬控制器追蹤落後；state_age 固定 0.01 s',
                       'C（張爪同步退離）只作離線比較，不代表准許越過解除准入條件']}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({'events': ev, 'after_latch_base': after}, ensure_ascii=False, default=float))
    for s in seg:
        print(json.dumps({k: s.get(k) for k in ('seg', 'step0', 'step1', 'door_deg', 'door_rate_step_deg_s', 'arm_joint_rate_win0.1_rad_s_max',
                                                 'base_win0.1', 'contact_max_n', 'applied_src_seq_mode', 'base_cmd_body', 'wrist_n')},
                         ensure_ascii=False, default=float))
    for k, v in cmp_['scenarios'].items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'rows'}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
