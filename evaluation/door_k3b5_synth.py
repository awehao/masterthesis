#!/usr/bin/env python3
"""K3-B r5 完整合成趟產生器（**測試用合成資料，不是實跑結果**；Codex 20261026_034842 §1：最外層 judge() 的完整合成通過例）。

基底＝實跑 door_wide_w2c_30_1（HELD、mode 1 提交 69.71 s 後 3.1 s 的 mode 1 保持段）。
  1. 張爪前（實跑列）：mode 1 提交步 k1 起為卸載段——兩指接觸力依斜坡等比縮放（f/20）；W1e 定期讀回改為斜坡值或 5 N（others_same True）；
     door_k3b_unload2.ExecUnloadMonitor 對這些列逐步產生 w2.unload（含 set_log）與 unload_ctrl；權威列補 finger_target。
  2. 時間軸：執行端 UNLOADED、以 transient2 對 room 列重算卸載收斂＋t_settling＋0.5 得求解端 UNLOADED（＋0.02 s 宣告延遲）；
     mode 2 請求＝兩端完成＋0.03 s、登記＋0.02 s、提交＝請求後第 8 步（k2）。
  3. 張爪後（合成列）：手指 0.31 s 開到 95 %、接觸 0.04 s 歸零、手臂 j2 回彈 0.0015 rad（τ 0.15 s）；
     door_k3b_release4.ExecReleaseMonitor 逐步產生 w2.release；transient2 重算釋放收斂，求解端 RELEASE_SETTLED＝重算＋0.02 s；
     mode 3 請求＝max(B_CONFIRMED, RELEASE_SETTLED)＋0.05 s、提交＋0.1 s；之後任務端每 0.05 s 以 0.02 m/s 推進 s_cmd 至 0.10 m，
     手臂以離線 IK 追蹤上一週期目標（線性內插、底盤固定），回彈偏移於 mode 3 後以 τ 0.3 s 衰減；終點後保持 3.6 s。
  4. 任務端：TaskReleaseMonitor（core r4）以真配對幾何與一般幾何監看逐 20 Hz 樣本產生 k3b.monitor；trace（RELEASE 列、k3b 欄、pub_seq）
     與 trace_log 檔（寫入呼叫端給的目錄）；rel（t、eff_verify、unload_gate、solver_sub_at_req3）；policy_events PAUSE → RELEASE → DONE。
  5. 求解端：監督器 r9 outcome（轉移、兩段瞬態 CONVERGED、t_settling）、w2_retreat 交接、args 旗標；room.w2 的提交、pending、runtime 請求與轉送。
    build(out_dir) → {'task.json', 'room_run.json', 'park_gate.json', 'align_solver.json', 'lw_observer.json', 'meta'}
"""
from __future__ import annotations

import copy
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, 'runs', 'door_wide_w2c_30_1')
FO = 0.0089
DT = 0.01


def _ld(f):
    return json.load(open(os.path.join(RUN_DIR, f)))


def build(out_dir):
    import door_k2_check as KC
    import door_k2_offline_v2 as V2
    import door_k3b_pair_geom as PG
    import door_k3b_release5 as KR
    import door_k3b_transient2 as TT2
    import door_k3b_unload2 as KU
    import door_k3m_check_unload3 as CU3
    import door_k3m_geom_monitor as GM
    import door_w2_mode5 as M5
    task, room, gate, solver, obs = (_ld(f) for f in ('task.json', 'room_run.json', 'park_gate.json', 'align_solver.json', 'lw_observer.json'))
    RUN = task['args']['run_id']
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = room['steps']
    by = {s[0]: s for s in S}
    OFF = int(round(S[0][1] / DT)) - S[0][0]                                 # 實跑：步 0 ↔ 0.02 s
    tk = lambda k: round((k + OFF) * DT, 4)                                 # noqa: E731
    kt = lambda t: int(round(t / DT)) - OFF                                 # noqa: E731
    w2 = room['w2']
    ev = w2['auth']['events']
    c1 = next(e for e in ev if e.get('kind') == 'commit' and e.get('mode_seq') == 1)
    k1, t1 = c1['step'], c1['t']
    rr = KU.REG_UNLOAD
    # ---- 1. 卸載段（實跑列）
    xu = KU.ExecUnloadMonitor()
    xu.start(k1, t1)
    uctrl = []
    for s in S:
        if s[0] < k1:
            continue
        f = KU.f_cmd(rr, t1, s[1])
        for n in ('uflite_finger1', 'uflite_finger2'):
            s[I['leaf_contacts']][n]['bar'] = round(s[I['leaf_contacts']][n]['bar'] * f / 20.0, 4)
        fc = s[I['finger_contact_n']]
        if isinstance(fc, list):
            s[I['finger_contact_n']] = [round(x * f / 20.0, 4) for x in fc]
        if xu.state in ('RAMPING', 'SETTLING', 'UNLOADED'):
            lc = s[I['leaf_contacts']]
            xu.step(s[0], s[1], [f, f], f, [lc[n]['bar'] + lc[n]['ambiguous'] for n in ('uflite_finger1', 'uflite_finger2')], True,
                    s[I['door_q_rad']], s[I['base_xyth']], s[I['q_arm_meas']])
            if s[0] % 10 == 0:
                uctrl.append({'ok': True, 'why': None, 'step': s[0], 't': s[1], 'set': f, 'readback': [f, f]})
        if xu.state == 'UNLOADED' and s[1] >= xu.t['UNLOADED'] + 0.05:
            break
    assert xu.state == 'UNLOADED', (xu.why, xu.events)
    tu_x, t_set = xu.t['UNLOADED'], xu.t['SETTLING']
    i_pre = max(i for i, s in enumerate(S) if s[1] <= t1 + 1e-9)
    i_u = max(i for i, s in enumerate(S) if s[1] <= tu_x + 0.05)
    st8, w8, tc8, wst8 = CU3.recompute_transient(S, I, i_pre, i_u, rr['unload_max_s'])
    assert st8 == 'CONVERGED', (st8, w8)
    tu_s = round(max(tc8, t_set + rr['quiet_s']) + 0.02, 4)
    t_req2 = round(max(tu_x, tu_s) + 0.03, 2)
    t_pend2 = round(t_req2 + 0.02, 2)
    k2 = kt(t_req2) + 8
    t2 = tk(k2)
    assert k2 < S[-1][0], 'mode 2 提交須在實跑段內'
    for s in S:
        if k1 <= s[0] < k2 and xu.state == 'UNLOADED' and s[0] > xu.last_step:
            f = 5.0
            lc = s[I['leaf_contacts']]
            xu.step(s[0], s[1], [f, f], f, [lc[n]['bar'] + lc[n]['ambiguous'] for n in ('uflite_finger1', 'uflite_finger2')], True,
                    s[I['door_q_rad']], s[I['base_xyth']], s[I['q_arm_meas']])
            if s[0] % 10 == 0 or s[0] == k2 - 1:
                uctrl.append({'ok': True, 'why': None, 'step': s[0], 't': s[1], 'set': f, 'readback': [f, f]})
    xu.handoff(k2, t2)
    # ---- 2. 張爪後（合成列）
    base_row = copy.deepcopy(by[k2 - 1])
    base, q0, qd0 = list(base_row[I['base_xyth']]), np.array(base_row[I['q_arm_meas']], float), float(base_row[I['door_q_rad']])
    H = KC._H_of(qd0)
    E_rel = np.linalg.inv(H) @ KC._tcp(base, list(q0))
    C = V2.C
    fq_closed = list(base_row[I['q_finger']])
    fb0 = [base_row[I['leaf_contacts']][n]['bar'] for n in ('uflite_finger1', 'uflite_finger2')]
    S[:] = [s for s in S if s[0] < k2]
    spring = lambda t: 0.0015 * (1 - math.exp(-max(0.0, t - t2) / 0.15))      # noqa: E731
    # 先以不退離的列決定 B_CONFIRMED 與釋放收斂
    xm = KR.ExecReleaseMonitor()
    xm.open(k2, t2)

    def mk_row(k, q_arm, fq, fb):
        r = copy.deepcopy(base_row)
        t = tk(k)
        r[I['step']], r[I['sim_t']] = k, t
        r[I['q_arm_meas']] = [round(float(x), 6) for x in q_arm]
        r[I['q_finger']] = [round(x, 6) for x in fq]
        r[I['finger_contact_n']] = [round(x, 4) for x in fb]
        r[I['contact_held_s']] = 0.0
        r[I['leaf_contacts']] = {'uflite_finger1': {'bar': round(fb[0], 4), 'ambiguous': 0.0, 'panel': 0.0},
                                 'uflite_finger2': {'bar': round(fb[1], 4), 'ambiguous': 0.0, 'panel': 0.0},
                                 'uflite_gripper_link': {'bar': 0.0, 'ambiguous': 0.0, 'panel': 0.0}}
        r[I['wall_contacts']] = {}
        r[I['door_qdot_rad_s']] = 0.0
        return r

    def fingers(t):
        tr = t - t2
        f = min(1.0, 0.2 + 0.75 * tr / 0.31 if tr < 0.31 else 0.95 + (tr - 0.31) * 0.2)
        q = [c + (FO - c) * f for c in fq_closed]
        fb = [max(0.0, x * (1 - tr / 0.04)) for x in fb0]
        return q, fb
    pre = []
    for k in range(k2, k2 + 260):
        t = tk(k)
        qa = q0.copy()
        qa[1] += spring(t)
        fq, fb = fingers(t)
        pre.append(mk_row(k, qa, fq, fb))
    for r in pre:
        xm.step(r[0], r[1], r[I['q_finger']], [x['bar'] for n, x in r[I['leaf_contacts']].items() if n != 'uflite_gripper_link'], True,
                door_q=r[I['door_q_rad']])
        if xm.state == 'B_CONFIRMED':
            break
    t_conf = xm.t['B_CONFIRMED']
    tmp = S + pre
    Itmp = I
    st2, w2_, tc2, wst2 = CU3.recompute_transient(tmp, Itmp, len(S) - 1, len(tmp) - 1, TT2.T_MAX_RELEASE)
    assert st2 == 'CONVERGED', (st2, w2_)
    t_settled = round(tc2 + 0.02, 4)
    t_req3 = round(max(t_conf, t_settled) + 0.05, 2)
    t_pend3 = round(t_req3 + 0.02, 2)
    k3 = kt(t_req3) + 10
    t3 = tk(k3)
    # 任務端週期（偶數步對齊）：mode 3 提交後下一週期起推進
    tasks_t = [round(x * 0.05, 4) for x in range(int(math.ceil((t2 + 0.03) / 0.05)), int((t3 + 12.0) / 0.05))]
    t_seen3 = next(tt for tt in tasks_t if tt >= t3 + 0.03)
    s_cmd, s_hist = 0.0, []
    for tt in tasks_t:
        if tt > t_seen3 + 1e-9:
            s_cmd = min(KR.REG['s_end_m'], s_cmd + 0.02 * 0.05)
        s_hist.append((tt, s_cmd))

    def s_meas(t):                                                           # 一個週期落後的線性內插
        prev = (t2, 0.0)
        for tt, sv in s_hist:
            if tt + 0.05 > t:
                break
            prev = (tt + 0.05, sv)
        nxt = next(((tt + 0.10, sv) for tt, sv in s_hist if tt + 0.10 > t), prev)
        if nxt[0] <= prev[0]:
            return prev[1]
        a = (t - prev[0]) / (nxt[0] - prev[0])
        return prev[1] + max(0.0, min(1.0, a)) * (nxt[1] - prev[1])
    t_full = next(tt for tt, sv in s_hist if sv >= KR.REG['s_end_m'] - 1e-12)
    t_end_room = round(t_full + 0.2 + 3.6 + 1.0, 2)
    k_end = kt(t_end_room)
    xm = KR.ExecReleaseMonitor()
    xm.open(k2, t2)
    prev_q = q0.copy()
    for k in range(k2, k_end + 1):
        t = tk(k)
        sv = s_meas(t) if t > t3 else 0.0
        if sv > 0:
            Tm = np.eye(4)
            Tm[:3, 3] = sv * KR.A_H
            q = C.ik(np.array(base), H @ Tm @ E_rel, prev_q, only_prev=True)
            qa = np.asarray(q, float)[C.IDX] if q is not None else prev_q
        else:
            qa = q0.copy()
        prev_q = qa.copy()
        off = spring(t) if t <= t3 else spring(t3) * math.exp(-(t - t3) / 0.3)
        qa = qa.copy()
        qa[1] += off
        fq, fb = fingers(t)
        r = mk_row(k, qa, fq, fb)
        S.append(r)
        xm.step(k, t, r[I['q_finger']], fb, True, door_q=r[I['door_q_rad']])
    room['steps'] = S
    # ---- 權威列、提交、請求
    rows = [x for x in w2['rows'] if x['step'] < k2]
    last_src = max(x['src'] for x in rows)
    for x in rows:
        if x['step'] >= k1:
            x['finger_target'] = M5.FINGER_OF_SEQ5[x['auth_mode_seq']]
    src = last_src
    for k in range(k2, k_end + 1):
        ms = 2 if k < k3 else 3
        if (k - k2) % 5 == 0:
            src += 1
        if k in (k2, k3):
            src += 1
        rows.append({'step': k, 't': tk(k), 'src': src, 'mode_seq': ms, 'chain_mode': 0, 'profile': M5.PROFILE_OF_SEQ5[ms],
                     'auth_mode_seq': ms, 'finger_target': M5.FINGER_OF_SEQ5[ms], 'sel_src': src, 'fed': 'fed', 'confirm_ok': True,
                     'confirm_why': None, 'u': [0.0027, -0.0017, 0.0004] + ([0.0] * 6 if ms == 2 else [0.001] * 6)})
        if k == k2:
            src2 = src
        if k == k3:
            src3 = src
    w2['rows'] = rows
    w2['auth_rows'] = [{'step': x['step']} for x in rows]
    cs = [{'mode_seq': 1, 'step': k1, 't': t1, 'first_source_seq': c1['first_source_seq'], 'applied_confirmed': True},
          {'mode_seq': 2, 'step': k2, 't': t2, 'first_source_seq': src2, 'applied_confirmed': True},
          {'mode_seq': 3, 'step': k3, 't': t3, 'first_source_seq': src3, 'applied_confirmed': True}]
    p1 = next(e for e in ev if e.get('kind') == 'pending')
    ev[:] = [p1, c1, {'kind': 'pending', 'step': kt(t_pend2), 't': t_pend2, 'req_seq': 2, 'mode_seq': 2},
             {'kind': 'commit', 'step': k2, 't': t2, 'first_source_seq': src2, 'mode_seq': 2},
             {'kind': 'pending', 'step': kt(t_pend3), 't': t_pend3, 'req_seq': 3, 'mode_seq': 3},
             {'kind': 'commit', 'step': k3, 't': t3, 'first_source_seq': src3, 'mode_seq': 3}]
    w2['auth'].update(commits=cs, aborted=None, violations=[])
    names = ['PAUSE_HELD', 'RELEASE_PREP_TEST', 'RELEASE_OPEN', 'RETREAT']
    w2['runtime'] = dict(w2.get('runtime') or {}, requests=[{'req': {'run_id': RUN, 'req_seq': i, 'from_mode': names[i - 1], 'to_mode': names[i],
                                                                      'mode_seq': i}, 'result': 'pending'} for i in (1, 2, 3)],
                         relayed=[{'req_seq': i} for i in (1, 2, 3)])
    w2['release'] = xm.report()
    w2['effort'] = {'ack': None, 'events': [], 'restore': None}
    w2['unload'] = xu.report()
    w2['unload_ctrl'] = uctrl
    # ---- W1e 定期讀回
    pz = room['finger_effort']['pause']
    per = [p for p in pz['periodic'] if p['t'] < t1]
    tp, kp = per[-1]['t'], per[-1]['step']
    t_rows = {s[0]: s[1] for s in S}
    while True:
        kp += 101
        if kp not in t_rows:
            break
        tp = t_rows[kp]
        v = KU.f_cmd(rr, t1, tp) if tp < t_set else 5.0
        per.append({'step': kp, 't': tp, 'readback': [v, v], 'others_same': True, 'ok': True, 'why': None})
    pz['periodic'] = per
    pz['restore'] = None
    # ---- 任務端監看（core r4、真配對幾何、一般幾何監看）
    gcfg = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    gm = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(gcfg['v_floor_registered']), ds_err=dict(gcfg['downsampling']['ds_err_registered'])),
                            GM.default_backend())
    pg = PG.PairGeom()
    mon = KR.TaskReleaseMonitor()
    t_seen2 = next(tt for tt in tasks_t if tt >= t2 + 0.03)
    k_base = max(k for k in t_rows if k < k2 and k % 2 == 0)
    mon.open(t2, k_base, qd0, E_rel)
    byk = {s[0]: s for s in S}
    trace_new, s_prev = [], 0.0
    pub = max([x.get('pub_seq') or 0 for x in task['trace'] if x['t'] < t2] or [0])
    tgt0 = (H @ E_rel)
    t_done = None
    for tt, sv in s_hist:
        k = kt(tt)
        s = byk.get(k)
        if s is None:
            break
        qf = list(s[I['base_xyth']]) + list(s[I['q_arm_meas']])
        dec, det = gm.step(tt, qf, s[I['door_q_rad']], float(sum(s[I['q_finger']]) / 2), 0.01)
        pst, pd, pdp, pdet = pg.step(tt, qf, s[I['door_q_rad']], s[I['q_finger']], 0.01)
        cl = det['classes']
        E = np.linalg.inv(H) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        es = 'B_CONFIRMED' if tt >= t_conf else 'A_OPENING'
        x = {'t': tt, 'src_step': k, 'state_age': 0.01, 'compute_s': 0.02, 'E': E, 's_cmd': s_prev, 'door_q': s[I['door_q_rad']], 'door_qd': 0.0,
             'pair_d': pd, 'pair_d_pred': pdp, 'nonfinger_d': min(cl['nonfinger']['d'], cl['finger']['d']),
             'nonfinger_d_pred': min(cl['nonfinger']['d_pred'], cl['finger']['d_pred']), 'self_d': cl['self']['d'], 'self_d_pred': cl['self']['d_pred'],
             'base_d': cl['base']['d'], 'base_d_pred': cl['base']['d_pred'], 'base_dev_m': 0.0, 'base_dev_yaw': 0.0, 'wrist_n': 3.0,
             'unexpected_n': 0.0, 'exec_state': es, 't_conf': t_conf if tt >= t_conf else None}
        if tt >= t_seen2:
            mon.step(x)
        assert mon.state != 'FAILED', (mon.why, tt, mon.events[-3:])
        T = H.copy()
        Tm = np.eye(4)
        Tm[:3, 3] = sv * KR.A_H
        tgt = H @ Tm @ E_rel
        pub += 1
        row = copy.deepcopy(task['trace'][-1])
        row.update(t=tt, step=float(k), phase='RELEASE' if tt >= t_seen2 else 'PAUSE', state_step=k, target=tgt.reshape(-1).tolist(), pub_seq=pub,
                   grip=FO if tt >= t_seen2 else 0.0, q_door=s[I['door_q_rad']], s=0.0,
                   k3b={'state': mon.state if tt >= t_seen2 else 'IDLE', 's_cmd': sv, 'rel': 'release'}, w2='committed',
                   wall_mono=row['wall_mono'] + (tt - row['t']), loop_seq=row['loop_seq'] + 1 + len(trace_new),
                   trace_seq=row['trace_seq'] + 1 + len(trace_new))
        trace_new.append(row)
        s_prev = sv
        if mon.state == 'COMPLETE':
            mon.end(tt)
            t_done = tt
            break
    assert t_done is not None, (mon.state, mon.events[-3:])
    while S and S[-1][1] > t_done + 0.05 + 1e-9:
        S.pop()
    w2['rows'] = [x for x in w2['rows'] if x['step'] <= S[-1][0]]
    w2['auth_rows'] = [x for x in w2['auth_rows'] if x['step'] <= S[-1][0]]
    pz['periodic'] = [p for p in pz['periodic'] if p['t'] <= S[-1][1]]
    tr_old = [x for x in task['trace'] if x['t'] < t2]
    task['trace'] = tr_old + trace_new
    tc_clear = mon.t['C_CLEAR']
    rb_t = max(p['t'] for p in pz['periodic'] if p['t'] <= tc_clear + 0.05)
    t_rel = t_seen2
    names_ = ['PAUSE_HELD', 'RELEASE_PREP_TEST', 'RELEASE_OPEN', 'RETREAT']
    task['events'] = [e for e in task['events'] if not (isinstance(e, dict) and e.get('t', 0) >= t2)] + \
        [{'t': tq, 'w2_mode_request_published': {'run_id': RUN, 'req_seq': m, 'from_mode': names_[m - 1], 'to_mode': names_[m], 'mode_seq': m}}
         for m, tq in ((2, t_req2), (3, t_req3))]
    task['policy_events'] = [e for e in task['policy_events'] if e.get('to') != 'DONE' and e['t'] < t2] + \
        [{'t': t_rel, 'from': 'PAUSE', 'to': 'RELEASE', 't_open': t2, 'baseline_step': k_base}, {'t': t_done, 'from': 'RELEASE', 'to': 'DONE', 'k3b': 'retreat_complete'}]
    task['final_phase'], task['abort'] = 'DONE', None
    task['n_target_published'] = sum(1 for x in task['trace'] if x.get('pub_seq') is not None)
    task['loop_timing'] = dict(task['loop_timing'], n=len(task['trace']))
    task['args'] = dict(task['args'], w2_release=True, w2_unload=True)
    task['k3b'] = {'version': 'door_k3m_policy_k3b4/r4', 'unload': True,
                   'rel': {'state': 'done', 's_cmd': KR.REG['s_end_m'], 'eff_req': None, 'eff_ack': None, 'eff_done': None,
                           't': {'req_open': t_req2, 'open': t2, 'baseline': tk(k_base), 'req_retreat': t_req3, 'retreat': t_seen3,
                                 'eff_verify_from': tc_clear, 'done': t_done},
                           'eff_verify': {'t': tc_clear, 'readback': [5.0, 5.0], 'readback_t': rb_t, 'ok': True},
                           'unload_gate': {'t': t_req2, 'gate': 'go', 't_c1': t1, 'state_t': round(t_req2 - 0.02, 4), 'solver_t': round(t_req2 - 0.01, 4),
                                           'exec_unload': {'state': 'UNLOADED'}, 'solver_sub': 'unloaded'},
                           'solver_sub_at_req3': 'release_settled', 'q_rel': qd0, 'step_open': k2, 'base_anchor': base,
                           'anchor_record': {'t_open_base': base, 'baseline_step': k_base, 'baseline_t': tk(k_base), 'k_open': k2}},
                   'monitor': mon.report(), 'semantics': 'synthetic'}
    tl = dict(task['trace_log'])
    p = os.path.join(out_dir, 'task_trace.jsonl')
    with open(p, 'w', encoding='utf-8') as fh:
        for n, x in enumerate(task['trace'], 1):
            fh.write(json.dumps({'enq_seq': n, 'enq_wall_mono': x['wall_mono'] + 1e-4, 'row': x}, ensure_ascii=False, default=float) + '\n')
    tl.update(path=p, n_enqueued=len(task['trace']), n_written=len(task['trace']))
    task['trace_log'] = tl
    # ---- 求解端
    o = dict(solver['stats']['k3m_pause']['supervisor_outcome'])
    tr = dict(o.get('transitions') or {})
    tr.update(UNLOAD=round(t1 + 0.02, 4), UNLOADED=tu_s, RELEASE_OPEN=round(t2 + 0.04, 4), RELEASE_SETTLED=t_settled, HANDOVER=round(t3 + 0.02, 4))
    o.update(version='door_w2_mode9/r9', state='HANDED_OVER', held_end='handover', transitions=tr, sub='release_settled', t_settling=t_set,
             transient={'1': {'version': TT2.VERSION, 'state': 'CONVERGED', 't_conv': tc8, 'worst': wst8},
                        '2': {'version': TT2.VERSION, 'state': 'CONVERGED', 't_conv': tc2, 'worst': wst2}})
    solver['stats']['k3m_pause']['supervisor_outcome'] = o
    solver['stats']['w2_retreat'] = {'handover': {'sim_t': round(t3 + 0.02, 4), 'slot': None, 'by': 'applied_candidate'}, 'n_hist_insert': 1,
                                     'n_cand_pub': 1, 'n_cand_fail': 0}
    solver['args'] = dict(solver['args'], w2_retreat=True, w2_hold_r9=True, w2_unload=True, w2_held=True, w2_held_max_s=60.0)
    meta = {'k1': k1, 't1': t1, 'tu_exec': tu_x, 'tu_solver': tu_s, 't_settling': t_set, 't_req2': t_req2, 'k2': k2, 't2': t2, 't_conf': t_conf,
            't_settled': t_settled, 't_conv_release': tc2, 'k3': k3, 't3': t3, 't_done': t_done, 'C_CLEAR': tc_clear}
    return {'task.json': task, 'room_run.json': room, 'park_gate.json': gate, 'align_solver.json': solver, 'lw_observer.json': obs, 'meta': meta}
