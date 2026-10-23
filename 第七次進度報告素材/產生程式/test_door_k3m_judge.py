#!/usr/bin/env python3
"""door_k3m_check.judge（**完整判定**）的合成實錄與反例。

由 test_door_k3a_check 的 K3-A 合成實錄改出：OPEN 起先有 3 列等待切換（目標＝最後 HOLD 目標、無 q_ref），0.15 s 後才開始推進參考
（門角剖面與 SETTLE／END_HOLD／DONE 一併後移）；底盤固定在錨點（走廊表以錨點為中心）；另產生執行端 room.k3m（切換、逐步走廊紀錄）、
求解器紀錄（首筆 free 解前 hold、之後 free 且 park_hold 為 None）、任務 k3m（請求、兩 ack 與到達時刻、confirmed）。

    python3 evaluation/test_door_k3m_judge.py
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check as MC                            # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import dataclasses                                      # noqa: E402
import door_target_k3 as T3                            # noqa: E402
import door_k2_check as KC                             # noqa: E402
import test_door_k3a_check as B                        # noqa: E402

D = 0.15
RUN = 'door_k3m_syn'
GCFG = GM.GeomMonitorConfig(v_floor={'nonfinger': 0.002, 'finger': 0.002, 'base': 0.010, 'self': 0.002},
                            ds_err={'nonfinger': 0.000226, 'finger': 0.0, 'base': 0.0, 'self': 0.001342})
NOMD = {'nonfinger': 0.00717, 'finger': 0.0356, 'base': 0.2026, 'self': 0.00981}


def build():
    T_OPEN = B.T_OPEN
    dtk = T3.DoorTargetK3(B.T_HOLD, B.H0, B._cfg.target_cfg())
    REF, t = [], T_OPEN + D
    while True:
        q_meas = REF[-1][1] if REF else 0.0
        Tg, _ = dtk.step(KC._H_of(q_meas), q_meas, 0.05)
        REF.append((round(t, 4), dtk.q_ref, Tg, q_meas))
        if dtk.reached:
            break
        t += 0.05
    T_SETTLE = round(REF[-1][0] + 0.05, 4)
    T_END = round(T_SETTLE + 1.6, 4)
    T_DONE = round(T_END + 3.0, 4)
    PH = [('UNFOLD', 1.0), ('ALIGN_PRE', 8.0), ('ALIGN_HOLD', 10.0), ('APPROACH', 11.0), ('ENGAGE', 16.0), ('HOLD', 19.5), ('OPEN', T_OPEN),
          ('SETTLE', T_SETTLE), ('END_HOLD', T_END), ('DONE', T_DONE)]

    def door_q(tt):
        if tt < REF[0][0]:
            return 0.0
        if tt >= REF[-1][0]:
            return REF[-1][1]
        k = int((tt - REF[0][0]) / 0.05)
        a = (tt - REF[k][0]) / 0.05
        return (1 - a) * REF[k][1] + a * REF[k + 1][1]

    def ph_of(tt):
        cur = None
        for p, t0 in PH:
            if tt >= t0:
                cur = p
        return cur
    req_t, k_sw, t_sw, t_e, t_s = T_OPEN, 2051, 20.51, 20.52, 20.57
    trace = []
    tt = 19.5
    while tt < T_OPEN - 1e-3 - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'HOLD', 'q_door': 0.0, 'q_ref': None, 'target': B.T_HOLD.reshape(-1).tolist(), 'k3m_mode': 'hold'})
        tt += 0.05
    trace.append({'t': T_OPEN - 1e-3, 'phase': 'OPEN', 'q_door': 0.0, 'q_ref': None, 'target': B.T_HOLD.reshape(-1).tolist(), 'k3m_mode': 'hold'})
    for k in range(3):
        trace.append({'t': round(T_OPEN + 0.05 * k, 4), 'phase': 'OPEN', 'q_door': 0.0, 'q_ref': None, 'target': B.T_HOLD.reshape(-1).tolist(),
                      'k3m_mode': 'waiting'})
    for t_, qr, Tg, qm in REF:
        trace.append({'t': t_, 'phase': 'OPEN', 'q_door': qm, 'q_ref': qr, 'target': Tg.reshape(-1).tolist(), 'k3m_mode': 'free'})
    tt = T_SETTLE
    while tt < T_END - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'SETTLE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist(),
                      'k3m_mode': 'free'})
        tt += 0.05
    while tt < T_DONE - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'END_HOLD', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist(),
                      'k3m_mode': 'free'})
        tt += 0.05
    trace.append({'t': T_DONE, 'phase': 'DONE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': None, 'k3m_mode': 'free'})
    n = 0
    for e in trace:
        e['step'] = e['state_step'] = int(round(e['t'] * 100))
        e['grip'] = 0.0
        if e['target'] is not None:
            n += 1
            e['pub_seq'] = n
        else:
            e['pub_seq'] = None
        e['state_age_wall'] = 0.03
        if e['phase'] in MC.OPENING and e['t'] >= t_e - 1e-9:
            T = GCFG.t_cycle_s + 0.03 + 0.0137 + GCFG.t_stop_s
            e['k3m_geom'] = {'dec': 'ok', 'why': None, 'compute_s': 0.0137,
                             'cls': {c: [NOMD[c], NOMD[c] - GCFG.v_floor[c] * T - GCFG.ds_err[c]] for c in MC.GEOM_CLASSES}}
            e['k3m_corr'] = [0.0, 0.0, 0.0]
    req = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'req_step': 2050, 'req_t': req_t}
    ea = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': k_sw, 'switch_t': t_sw}
    sa = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': k_sw, 'switch_t': t_sw, 'first_free_slot': 412,
          'first_free_sim_t': 20.55, 'publish_sim_t': 20.56, 'published': True}
    ev = [{'t': t0, 'from': 'x', 'to': p} for p, t0 in PH]
    task = {'schema': 'door_k3m_task/1', 'geom_config': dataclasses.asdict(GCFG), 'policy_events': ev, 'abort': None, 'final_phase': 'DONE', 'received_any_sync': True,
            'trace': trace, 'n_target_published': n, 'G_T_H': B.G_T_H.tolist(), 'H_frozen': B.H0.tolist(), 'grasp_basis': {'T_HG': B.T_HG.tolist()},
            'k3m': {'request': req, 'exec_ack': ea, 'solver_ack': sa, 't_exec_recv': t_e, 't_solver_recv': t_s, 'result': 'confirmed'}}
    steps, held, qprev = [], 0.0, 0.0
    for k in range(0, int((T_DONE + 0.5) / 0.01)):
        t = round(k * 0.01, 4)
        ph = ph_of(t)
        closed = ph in ('HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE') or (ph == 'ENGAGE' and t >= 16.5)
        fb = 2.0 if closed else 0.0
        held = held + 0.01 if closed else 0.0
        q = door_q(t)
        qa = B.Q_PRE if ph in (None, 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD') else (B.Q_HOLD if ph in ('APPROACH', 'ENGAGE', 'HOLD') else B.q_arm_of(q))
        qf = [0.0, 0.0] if closed else [0.0089, 0.0089]
        lc = {'uflite_finger1': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}, 'uflite_finger2': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}} if fb else {}
        steps.append([k, t, 1, 'wholebody_applied', [0, 0, 0], [0, 0, 0], [0] * 6, list(B.BASE), list(map(float, qa)), q, (q - qprev) / 0.01,
                      qf, [fb, fb], held, lc, {}, 3.0, 'hold'])
        qprev = q
    C = CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                     {'q_door_deg': 8.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], B.BASE)
    log = []
    for s in steps:
        if s[0] >= k_sw:
            e = C.deviation(s[7], math.degrees(s[9]))
            log.append([s[0], s[1], [round(v, 6) for v in e], round(math.degrees(s[9]), 5), [0.0, 0.0, 0.0]])
    room = {'steps_cols': B.COLS, 'steps': steps, 'events': [{'step': k_sw, 'park': 'hold_end_k3m_switch'}],
            'k3m': {'config': {'corridor_table_json': C.to_json()}, 'active': True, 'switch': dict(ea), 'violation': None, 'events': [], 'log': log}}
    slog = []
    for i, slot in enumerate(range(400, 600)):
        free = slot >= 412
        slog.append({'slot': slot, 'published': True, 'k3m_mode': 'free' if free else 'hold', 'park_hold': None if free else {'u_hold': [0, 0, 0]}})
    solver = {'stats': {'k3m_mode': {'active': True, 'record': {'ack': dict(sa)}}}, 'log': slog}
    return task, room, {'gate': {'anchor': list(B.BASE)}}, solver


_G = build()


def G():
    return copy.deepcopy(_G)


class FullJudge(unittest.TestCase):
    def fails_with(self, mut, flag):
        t, r, g, s = G()
        mut(t, r, g, s)
        res = MC.judge(t, r, g, s)
        self.assertFalse(res['pass'])
        self.assertTrue(any(w.startswith(flag) for w in res['why']), res['why'])

    def test_pass(self):
        res = MC.judge(*G())
        self.assertTrue(res['pass'], (res['why'], {k: res['stats'].get(k) for k in ('mode', 'corridor_worst_mm', 'n_waiting_rows')}))
        self.assertEqual(res['stats']['n_waiting_rows'], 3)

    def test_waiting_target_moved(self):
        def m(t, r, g, s):
            e = next(e for e in t['trace'] if e.get('k3m_mode') == 'waiting')
            T = np.asarray(e['target']).reshape(4, 4)
            T[0, 3] += 1e-6
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(m, 'waiting_target_not_held')

    def test_reference_before_confirm(self):
        def m(t, r, g, s):
            t['k3m']['t_solver_recv'] = 20.70
        self.fails_with(m, 'reference_advanced_before_confirm')

    def test_exec_ack_not_room(self):
        def m(t, r, g, s):
            r['k3m']['switch']['switch_step'] = 2052
        self.fails_with(m, 'mode_exec_ack_not_room_switch')

    def test_solver_ack_not_record(self):
        def m(t, r, g, s):
            s['stats']['k3m_mode']['record']['ack']['first_free_slot'] = 413
        self.fails_with(m, 'mode_solver_ack_not_solver_record')

    def test_late_ack(self):
        def m(t, r, g, s):
            t['k3m']['t_exec_recv'] = 21.6
        self.fails_with(m, 'mode_ack_late_or_missing')

    def test_time_order(self):
        def m(t, r, g, s):
            for a in (t['k3m']['solver_ack'], s['stats']['k3m_mode']['record']['ack']):
                a['first_free_sim_t'] = 20.505
        self.fails_with(m, 'mode_time_order')

    def test_switch_time_not_room(self):
        def m(t, r, g, s):
            for a in (t['k3m']['exec_ack'], r['k3m']['switch'], t['k3m']['solver_ack'], s['stats']['k3m_mode']['record']['ack']):
                a['switch_t'] = 20.515
        self.fails_with(m, 'mode_switch_step_time_mismatch')

    def test_not_confirmed(self):
        def m(t, r, g, s):
            t['k3m']['result'] = ['abort', 'x']
        self.fails_with(m, 'mode_not_confirmed')

    def test_solver_rows(self):
        def m(t, r, g, s):
            s['log'][30]['park_hold'] = {'u_hold': [0, 0, 0]}
        self.fails_with(m, 'solver_mode_rows_inconsistent')

        def m2(t, r, g, s):
            s['log'] = [x for x in s['log'] if x['slot'] != 412]
        self.fails_with(m2, 'solver_first_free_not_published')

    def test_base_drift_before_switch(self):
        def m(t, r, g, s):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            for x in r['steps']:
                if 19.6 <= x[I['sim_t']] < 20.0:
                    x[I['base_xyth']] = [x[I['base_xyth']][0] + 0.021, x[I['base_xyth']][1], x[I['base_xyth']][2]]
        self.fails_with(m, 'base_drift_before_switch')

    def test_corridor_after_switch(self):
        def m(t, r, g, s):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            for x in r['steps']:
                if x[I['step']] >= 2300:
                    x[I['base_xyth']] = [x[I['base_xyth']][0], x[I['base_xyth']][1] + 0.025, x[I['base_xyth']][2]]
        self.fails_with(m, 'base_corridor')

    def test_executor_log_mismatch(self):
        def m(t, r, g, s):
            r['k3m']['log'][100][2][0] += 1e-4
        self.fails_with(m, 'executor_corridor_log_mismatch')

        def m2(t, r, g, s):
            del r['k3m']['log'][100]
        self.fails_with(m2, 'executor_corridor_log_mismatch')

    def test_executor_violation(self):
        def m(t, r, g, s):
            r['k3m']['violation'] = {'why': 'corridor'}
        self.fails_with(m, 'k3m_executor_violation')

    def test_geometry(self):
        def m(t, r, g, s):
            e = [e for e in t['trace'] if e.get('k3m_geom')][-1]                   # 最後一列：重算與紀錄一致的 stop
            T = GCFG.t_cycle_s + 0.03 + 0.0137 + GCFG.t_stop_s
            e['k3m_geom']['cls']['self'] = [0.004, 0.004 - GCFG.v_floor['self'] * T - GCFG.ds_err['self']]
            e['k3m_geom']['dec'] = 'stop'
        self.fails_with(m, 'geometry_stop')

        def m_only_dec(t, r, g, s):
            next(e for e in t['trace'] if e.get('k3m_geom'))['k3m_geom']['dec'] = 'stop'
        self.fails_with(m_only_dec, 'geometry_recompute_mismatch')

        def m2(t, r, g, s):
            e = [e for e in t['trace'] if e.get('k3m_geom')][20]
            del e['k3m_geom']
        self.fails_with(m2, 'geometry_evidence_missing')

    def test_codex_geometry_defects(self):
        """Codex 反例：compute_s＝−1、距離 NaN、cls＝{} 都必須不通過；記錄 dec＝ok 但重算為 stop、d_pred 竄改、年齡超限也不通過。"""
        def row(t):
            return [e for e in t['trace'] if e.get('k3m_geom')][30]

        def neg(t, r, g, s):
            row(t)['k3m_geom']['compute_s'] = -1.0
        self.fails_with(neg, 'geometry_evidence_missing')

        def nan(t, r, g, s):
            row(t)['k3m_geom']['cls']['self'][0] = float('nan')
        self.fails_with(nan, 'geometry_evidence_missing')

        def empty(t, r, g, s):
            row(t)['k3m_geom']['cls'] = {}
        self.fails_with(empty, 'geometry_evidence_missing')

        def lie(t, r, g, s):
            e = row(t)
            T = GCFG.t_cycle_s + 0.03 + 0.0137 + GCFG.t_stop_s
            d = 0.004
            e['k3m_geom']['cls']['self'] = [d, d - GCFG.v_floor['self'] * T - GCFG.ds_err['self']]   # 一致的預測，但決策仍寫 ok
        self.fails_with(lie, 'geometry_recompute_mismatch')

        def dpred(t, r, g, s):
            row(t)['k3m_geom']['cls']['base'][1] += 1e-6
        self.fails_with(dpred, 'geometry_recompute_mismatch')

        def age(t, r, g, s):
            row(t)['state_age_wall'] = 0.195
        self.fails_with(age, 'geometry_evidence_missing')

        def cfg(t, r, g, s):
            del t['geom_config']
        self.fails_with(cfg, 'geometry_config_missing')

    def test_k3a_safety_still_applies(self):
        def m(t, r, g, s):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            r['steps'][2400][I['wrist_force_n']] = 31.0
        self.fails_with(m, 'wrist_force')

    def test_schema(self):
        def m(t, r, g, s):
            t['schema'] = 'door_k3a_task/1'
        self.fails_with(m, 'task_schema')


if __name__ == '__main__':
    unittest.main(verbosity=2)
