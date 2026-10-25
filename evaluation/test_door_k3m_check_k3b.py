#!/usr/bin/env python3
"""door_k3m_check_k3b 反例（K3-B 部分以**測試用合成紀錄**驗；舊鏈組合以實錄 60_3 驗）。
合成：
- 以 door_wide_w2c_30_1 HELD 實測（底盤、手臂、門 30.21°）為起點，底盤固定；
- mode 1／2／3 提交步 100／200／300；張爪 0.31 s 到 95 %、承載 0.04 s 歸零 ⇒ 重算 t_conf＝281；
- 300 步起以 0.02 m/s 退到 0.10 m（離線 IK 逐步求手臂關節）、保持到 12 s；
- 5 N 切換在 C_CLEAR（4.0 s）之後。
  J0 合成正例 ⇒ check_k3b 無理由。
  J1 M 項：提交只有 1、2 ⇒ M1；手指目標在 201 步才轉 ⇒ M2_finger_switch；某列 confirm_ok False ⇒ M2_confirm_coverage；
     某列 profile 錯 ⇒ M2_profile_or_finger_target_mismatch。
  J2 執行端 t_conf 報告 290（重算 281）⇒ K2_t_conf_mismatch_exec；段 A 承載 19 N ⇒ K3_segment_A_force；確認後 0.3 N ⇒ K3_after_confirm…；
     mode 3 提交在 270 ⇒ K2_mode3_before_t_conf。
  J3 5 N：切換在 C_CLEAR 之前 ⇒ K4_restore_before_clear_or_retreat；恢復紀錄 ok False ⇒ K4_restore_record；切換後定期讀回失敗 ⇒ K4_periodic_after_restore。
  J4 幾何：退離中門角 +0.6°（s≈0.025，帶 +0.5°）⇒ K5_geometry_or_band；只退到 0.08 m ⇒ K5_retreat_not_reached。
  J5 底盤離 t_open 位姿 25 mm ⇒ K6_base_anchor；夾爪殼 panel 0.5 N ⇒ K7_non_design_contact。
  J6 任務端 COMPLETE 只保持 2 s ⇒ K8_complete_hold；求解端無交接 ⇒ K9。
  J7 組合：實錄 60_3（舊 DONE、無 K3-B）⇒ 舊判準除取代項外全過、K3-B 只報 M1（無多段提交）。
  J8 取代清單依據：W2c2 合成趟改為釋放軌跡（RELEASE 列、張爪、退離目標）⇒ W1f2 舊鏈理由恰為 K3B_REPLACED 兩項。

    python3 evaluation/test_door_k3m_check_k3b.py
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402
import door_k3b_release2 as KR                         # noqa: E402
import door_k3m_check_k3b as J                         # noqa: E402

FO = 0.0089


def synth():
    v4 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')))
    hs = v4['held_state']
    base, q_arm, q = list(hs['base']), np.array(hs['q_arm']), float(hs['q_door'])
    H = KC._H_of(q)
    E_rel = np.linalg.inv(H) @ KC._tcp(base, list(q_arm))
    C = V2.C
    cols = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'q_finger', 'leaf_contacts', 'wall_contacts']
    steps, rows, arows = [], [], []
    prev = q_arm.copy()
    for k in range(0, 1201):
        t = round(k * 0.01, 4)
        s = 0.0 if k < 300 else min(0.10, 0.02 * (t - 3.0))
        if k >= 300:
            T = np.eye(4)
            T[:3, 3] = s * KR.A_H
            prev = C.ik(np.array(base), H @ T @ E_rel, prev, only_prev=True)[C.IDX]
        tr = (t - 2.0) if k > 200 else None
        f = 0.17 if tr is None else min(1.0, 0.2 + 0.75 * tr / 0.31 if tr < 0.31 else 0.95 + (tr - 0.31) * 0.2)
        fb = 12.0 if tr is None else max(0.0, 12.0 * (1 - tr / 0.04))
        lc = {'uflite_finger1': {'bar': fb, 'panel': 0.0, 'ambiguous': 0.0}, 'uflite_finger2': {'bar': fb, 'panel': 0.0, 'ambiguous': 0.0},
              'uflite_gripper_link': {'bar': 0.0, 'panel': 0.0, 'ambiguous': 0.0}}
        steps.append([k, t, list(base), [float(x) for x in prev], q, [FO * f, FO * f], lc, {}])
        if k >= 100:
            ms = 1 if k < 200 else 2 if k < 300 else 3
            src = {1: 11, 2: 21, 3: 31}[ms] + (k - {1: 100, 2: 200, 3: 300}[ms]) // 5 * 10
            rows.append({'step': k, 't': t, 'src': src, 'mode_seq': ms, 'chain_mode': 0, 'profile': J.M5.PROFILE_OF_SEQ5[ms], 'auth_mode_seq': ms,
                         'finger_target': J.M5.FINGER_OF_SEQ5[ms], 'confirm_ok': True})
            arows.append({'step': k})
    cs = [{'mode_seq': m, 'step': k, 't': k * 0.01, 'first_source_seq': f, 'applied_confirmed': True} for m, k, f in ((1, 100, 11), (2, 200, 21), (3, 300, 31))]
    room = {'steps_cols': cols, 'steps': steps,
            'w2': {'auth': {'commits': cs, 'aborted': None, 'violations': []}, 'rows': rows, 'auth_rows': arows,
                   'release': {'events': [{'t': 2.0, 'step': 200, 'to': 'A_OPENING'}, {'t': 2.81, 'step': 281, 'to': 'B_CONFIRMED'}]},
                   'effort': {'restore': {'ok': True, 'from': 20.0, 'to': 5.0, 't': 4.1, 'step': 410}}},
            'finger_effort': {'pause': {'periodic': [{'t': 4.5, 'ok': True}, {'t': 5.5, 'ok': True}]}}}
    task = {'final_phase': 'DONE', 'k3b': {'monitor': {'why': None, 'events': [{'to': 'ARMED', 't': 2.0}, {'to': 'NEAR', 't': 2.85},
                                                                             {'to': 'C_CLEAR', 't': 4.0, 's_meas': 0.02, 'pair': 0.0137},
                                                                             {'to': 'RETREAT_DONE', 't': 8.5}, {'to': 'COMPLETE', 't': 11.5}],
                                                        't': {'ARMED': 2.0, 'NEAR': 2.85, 'C_CLEAR': 4.0, 'RETREAT_DONE': 8.5, 'COMPLETE': 11.5}},
                                           'rel': {'state': 'done', 'eff_done': {'readback_ok': True}, 't': {}}}}
    solver = {'stats': {'w2_retreat': {'handover': {'sim_t': 3.0}, 'n_hist_insert': 1, 'n_cand_pub': 1},
                        'k3m_pause': {'supervisor_outcome': {'held_end': 'handover'}}}}
    return task, room, solver


BASE = synth()


def run(f=None):
    task, room, solver = copy.deepcopy(BASE)
    if f:
        f(task, room, solver)
    st = {}
    return J.check_k3b(task, room, None, solver, st), st


def I(room, c):
    return room['steps_cols'].index(c)


class TJ(unittest.TestCase):
    def assertWhy(self, f, tag):
        why, _ = run(f)
        self.assertTrue(any(w.startswith(tag) for w in why), (tag, why))

    def test_j0_pass(self):
        why, st = run()
        self.assertEqual(why, [], st)
        self.assertEqual(st['k3b_exec']['k_conf_recomputed'], 281)
        self.assertGreaterEqual(st['k3b_geom']['s_max'], 0.095)

    def test_j1_multi(self):
        self.assertWhy(lambda t, r, s: r['w2']['auth'].update(commits=r['w2']['auth']['commits'][:2]), 'M1_commits')

        def late_switch(t, r, s):
            for x in r['w2']['rows']:
                if x['step'] == 200:
                    x['finger_target'] = 'closed'
        self.assertWhy(late_switch, 'M2_')

        def bad_conf(t, r, s):
            r['w2']['rows'][50]['confirm_ok'] = False
        self.assertWhy(bad_conf, 'M2_confirm_coverage')

        def bad_prof(t, r, s):
            r['w2']['rows'][150]['profile'] = 'pause_hold'
        self.assertWhy(bad_prof, 'M2_profile_or_finger_target_mismatch')

    def test_j2_exec(self):
        self.assertWhy(lambda t, r, s: r['w2']['release']['events'][1].update(step=290), 'K2_t_conf_mismatch_exec')

        def segA(t, r, s):
            r['steps'][205][I(r, 'leaf_contacts')]['uflite_finger1']['bar'] = 19.0
        self.assertWhy(segA, 'K3_segment_A_force')

        def post(t, r, s):
            r['steps'][500][I(r, 'leaf_contacts')]['uflite_finger2']['ambiguous'] = 0.3
        self.assertWhy(post, 'K3_after_confirm')

        def m3early(t, r, s):
            r['w2']['auth']['commits'][2]['step'] = 270
            for x in r['w2']['rows']:
                if x['step'] == 270:
                    x.update(src=31, mode_seq=3)
        self.assertWhy(m3early, 'K2_mode3_before_t_conf')

    def test_j3_effort(self):
        self.assertWhy(lambda t, r, s: r['w2']['effort']['restore'].update(t=3.9, step=390), 'K4_restore_before_clear_or_retreat')
        self.assertWhy(lambda t, r, s: r['w2']['effort']['restore'].update(ok=False), 'K4_restore_record')
        self.assertWhy(lambda t, r, s: r['finger_effort']['pause']['periodic'].append({'t': 6.5, 'ok': False}), 'K4_periodic_after_restore')

    def test_j4_geom(self):
        def door(t, r, s):
            for x in r['steps']:
                if 420 <= x[0] <= 430:
                    x[I(r, 'door_q_rad')] += math.radians(0.6)
        self.assertWhy(door, 'K5_geometry_or_band')

        def short(t, r, s):
            hold = None
            for x in r['steps']:
                if x[0] == 700:
                    hold = list(x[I(r, 'q_arm_meas')])
                if x[0] > 700:
                    x[I(r, 'q_arm_meas')] = hold
        self.assertWhy(short, 'K5_retreat_not_reached')

    def test_j5_base_contact(self):
        def base(t, r, s):
            for x in r['steps']:
                if x[0] >= 600:
                    b = list(x[I(r, 'base_xyth')])
                    b[0] += 0.025
                    x[I(r, 'base_xyth')] = b
        self.assertWhy(base, 'K6_base_anchor')

        def shell(t, r, s):
            r['steps'][700][I(r, 'leaf_contacts')]['uflite_gripper_link']['panel'] = 0.5
        self.assertWhy(shell, 'K7_non_design_contact')

    def test_j6_task_solver(self):
        self.assertWhy(lambda t, r, s: t['k3b']['monitor']['t'].update(COMPLETE=10.5), 'K8_complete_hold')
        self.assertWhy(lambda t, r, s: s['stats']['w2_retreat'].update(handover=None), 'K9_solver')

    def test_j7_composition_60_3(self):
        run_d = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')
        ld = (lambda f: json.load(open(os.path.join(run_d, f))) if os.path.exists(os.path.join(run_d, f)) else None)
        res = J.judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
        self.assertTrue(res['old_pass_excluding_replaced'], res['why'])
        self.assertEqual(res['replaced_old_reasons'], [])
        self.assertEqual(res['why'], ['M1_commits_not_1_2_3_confirmed'])


class TJ8(unittest.TestCase):
    def test_j8_old_chain_on_release_trace(self):
        """取代清單依據：W2c2 合成趟（60_3＋合成 W2）t_open 後改 RELEASE 列、夾爪張開、目標退離 ⇒ W1f2 舊鏈理由 ⊆ K3B_REPLACED。"""
        import door_k3m_check_w1f2 as F2
        import test_door_k3m_check_w2c2 as T2
        d = copy.deepcopy(T2.SYN)
        task = d['task.json']
        tr = task['trace']
        t_open = task['w2']['t']['committed'] + 3.2
        for x in tr:
            if x['t'] >= t_open and x['phase'] == 'PAUSE':
                x['phase'], x['grip'] = 'RELEASE', FO
                if x.get('target') is not None:
                    tg = np.array(x['target'], float).reshape(4, 4)
                    tg[1, 3] -= 0.006 * (x['t'] - t_open)
                    x['target'] = tg.reshape(-1).tolist()
        with open(task['trace_log']['path'], 'w', encoding='utf-8') as fh:
            for n, x in enumerate(tr, 1):
                fh.write(json.dumps({'enq_seq': n, 'enq_wall_mono': x['wall_mono'] + 1e-4, 'row': x}, ensure_ascii=False, default=float) + '\n')
        r = F2.judge(task, d['room_run.json'], d['park_gate.json'], d['align_solver.json'], d['lw_observer.json'])
        self.assertTrue(set(r['why']) <= set(J.K3B_REPLACED), r['why'])
        self.assertEqual(sorted(r['why']), sorted(J.K3B_REPLACED))


if __name__ == '__main__':
    unittest.main(verbosity=1)
