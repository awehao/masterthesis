#!/usr/bin/env python3
"""核心段判定器 door_coopen_check_core 的反例（不啟動模擬器、不用 ROS）。
  D1 rate_constants：區塊內 w1b 重建門角參考的設定 r_max_dps＝0.6，離開後還原（原版缺省 1.0）；
  D2 check_coopen：合成 task.json（登錄 r5、基準 r4）正確 ⇒ 無理由；a_H 不符 ⇒ coopen_a_H_mismatch；門速不符 ⇒ coopen_r_max_mismatch；
     登錄雜湊不符 ⇒ coopen_params_sha_mismatch；w1b 登錄不是同一檔 ⇒ coopen_w1b_params_not_same；
  D3 report_a2 幾何一致性：以 q48 候選 0–48° 的底盤／手臂構型與門角合成 room_run（door_q_rad＝候選門角）與取證（T_WH＝H(q)），
     TCP 推得門角與門關節角之差 ≤ 0.05°、相對位姿殘差位置 ≤ 1 mm、姿態 ≤ 0.1°（候選為 IK 解，誤差在 IK 門檻內）；出口進展可算出。

    python3 evaluation/test_door_coopen_check_core.py
"""
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core as J                     # noqa: E402
import door_k2_check as KC                             # noqa: E402
import door_k3m_check_w1b as W1B                       # noqa: E402

PRM_REL = 'evaluation/results/door/coopen_core_params_48_r5.json'
PRM = os.path.join(WS, PRM_REL)
CAND = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2_phi60_s0.85_q48.json')


def good_task():
    prm = json.load(open(PRM))
    gb = json.load(open(os.path.join(WS, prm['grasp']['basis'])))['args']
    sha = hashlib.sha256(open(PRM, 'rb').read()).hexdigest()
    return {'coopen': {'params_path': PRM_REL, 'params_sha256': sha, 'grasp_basis_has_full': True, 'T_HG': gb['T_HG_full'], 'a_H': gb['a_H'],
                       'r_max_dps': 0.6}, 'w1b': {'params_path': PRM_REL}, 'config': {'r_max_dps': 0.6}}


class D(unittest.TestCase):
    def test_d1_rate_constants(self):
        with J.rate_constants(0.6):
            with W1B.target_constants(48.0, 1.0, (0, 0, 0, 0, 0)):
                self.assertAlmostEqual(W1B.KC3.CFG.r_max_dps, 0.6)
                self.assertAlmostEqual(W1B.KC3.THETA, math.radians(48.0))
        with W1B.target_constants(48.0, 1.0, (0, 0, 0, 0, 0)):
            self.assertAlmostEqual(W1B.KC3.CFG.r_max_dps, 1.0)

    def test_d2_check_coopen(self):
        why, st = [], {}
        self.assertIsNotNone(J.check_coopen(good_task(), why, st))
        self.assertEqual(why, [])
        for mut, want in ((lambda t: t['coopen'].__setitem__('a_H', [0.0, -1.0, 0.0]), 'coopen_a_H_mismatch'),
                          (lambda t: t['coopen'].__setitem__('r_max_dps', 1.0), 'coopen_r_max_mismatch'),
                          (lambda t: t['config'].__setitem__('r_max_dps', 1.0), 'coopen_r_max_mismatch'),
                          (lambda t: t['coopen'].__setitem__('params_sha256', '0' * 64), 'coopen_params_sha_mismatch'),
                          (lambda t: t['w1b'].__setitem__('params_path', 'x.json'), 'coopen_w1b_params_not_same'),
                          (lambda t: t['coopen'].__setitem__('grasp_basis_has_full', False), 'coopen_grasp_basis_not_full')):
            t = copy.deepcopy(good_task())
            mut(t)
            why = []
            J.check_coopen(t, why, {})
            self.assertIn(want, why)

    def test_d3_report_geometry(self):
        tr = json.load(open(CAND))['full'][0]['trajectory']
        cols = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'door_qdot_rad_s', 'base_cmd_body', 'applied_src']
        steps, ev = [], []
        for i, p in enumerate(tr):
            q = math.radians(p['q_deg'])
            steps.append([i, float(i), list(p['base']), list(p['qa']), q, math.radians(0.6), [0.01, 0.0, 0.0], 1])
            ev.append({'step': i, 'T_WH': KC._H_of(q).reshape(-1).tolist()})
        room = {'steps_cols': cols, 'steps': steps}
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
            for e in ev:
                f.write(json.dumps(e) + '\n')
            path = f.name
        try:
            a2 = J.report_a2(room, path, json.load(open(PRM)))
        finally:
            os.remove(path)
        self.assertTrue(a2['available'])
        self.assertLessEqual(a2['door_joint_vs_tcp_deg']['absmax'], 0.05)
        self.assertLessEqual(max(a2['tcp_handle_residual']['pos_absmax_mm_handle_xyz']), 1.0)
        self.assertLessEqual(a2['tcp_handle_residual']['rot_max_deg'], 0.1)
        self.assertIsNotNone(a2['exit_progress']['drop_m'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
