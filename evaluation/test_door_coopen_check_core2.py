#!/usr/bin/env python3
"""核心段判定器 r2 的最小反例（Codex 20261027_002013 §1：正常、缺取證、門速資料無效）。不啟動模擬器、不用 ROS。
合成資料：q48 候選 0–48° 的底盤／手臂構型（每 1° 一步、物理步 0.01 s 間隔的 sim_t），門關節角＝候選門角，取證 T_WH＝H(q)；
任務 trace：state_step 0 → HOLD、1 起 OPEN、48 起 SETTLE。
  C1 正常：core_evidence＝sufficient；OPEN 段有效殘差配對 > 0、把手座標殘差 ≤ 1 mm、≤ 0.1°；出口進展（OPEN）可算；
     門速有效（步距 0.01 s、窗 0.1 s）；grasp_entry 有 HOLD 與 OPEN；
  C2 缺取證 ⇒ insufficient，理由含 evidence_missing 與 open_residual_pairs_zero；
  C3 門速資料無效（相鄰步 sim_t 間隔 0.05 s > gap 0.03 s）⇒ 門速全無效、不當零：n_drop_rows_rate_invalid＝n_drop_rows、占比為 None；
  C4 殘差座標：對 T_tcp 施加把手座標 +2 mm x 的位移 ⇒ HOLD 段把手座標殘差 x ≈ 2 mm（驗 M2 為把手座標，不是夾爪座標）。

    python3 evaluation/test_door_coopen_check_core2.py
"""
import json
import math
import os
import sys
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core2 as J                    # noqa: E402
import door_k2_check as KC                             # noqa: E402

PRM = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r5.json')
CAND = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2_phi60_s0.85_q48.json')
COLS = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'base_cmd_body', 'applied_src']


def synth(dt=0.01, with_ev=True):
    tr = json.load(open(CAND))['full'][0]['trajectory']
    steps, ev = [], []
    for i, p in enumerate(tr):
        q = math.radians(p['q_deg'])
        steps.append([i, round(1.0 + i * dt, 6), list(p['base']), list(p['qa']), q, 0.0, [0.0016, 0.0016], [2.0, 2.0], [0.0, 0.0, 0.0], 1])
        ev.append({'step': i, 'T_WH': KC._H_of(q).reshape(-1).tolist()})
    task = {'trace': [{'state_step': 0, 'phase': 'HOLD'}, {'state_step': 1, 'phase': 'OPEN'}, {'state_step': 48, 'phase': 'SETTLE'}]}
    path = None
    if with_ev:
        f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
        for e in ev:
            f.write(json.dumps(e) + '\n')
        f.close()
        path = f.name
    return {'steps_cols': COLS, 'steps': steps}, path, task


class C(unittest.TestCase):
    def setUp(self):
        self.prm = json.load(open(PRM))

    def test_c1_normal(self):
        room, path, task = synth()
        try:
            r = J.report_a2(room, path, self.prm, task)
        finally:
            os.remove(path)
        self.assertEqual(r['core_evidence'], 'sufficient', r['core_evidence_why'])
        o = r['residual_by_segment']['OPEN']
        self.assertGreater(o['n_valid_pairs'], 0)
        self.assertLessEqual(o['pos_norm_max_mm'], 1.0)
        self.assertLessEqual(o['rot_max_deg'], 0.1)
        self.assertIsNotNone(r['exit_progress_open']['drop_m'])
        self.assertIn('HOLD', r['grasp_entry'])
        self.assertIn('OPEN', r['grasp_entry'])

    def test_c2_missing_evidence(self):
        room, _, task = synth(with_ev=False)
        r = J.report_a2(room, '/nonexistent/evidence.jsonl', self.prm, task)
        self.assertEqual(r['core_evidence'], 'insufficient')
        self.assertIn('evidence_missing', r['core_evidence_why'])
        self.assertIn('open_residual_pairs_zero', r['core_evidence_why'])

    def test_c3_rate_invalid_not_zero(self):
        room, path, task = synth(dt=0.05)
        try:
            r = J.report_a2(room, path, self.prm, task)
        finally:
            os.remove(path)
        e = r['exit_progress_open']
        self.assertEqual(r['door_rate']['n_invalid'], r['door_rate']['n_rows'])
        self.assertEqual(e['n_drop_rows_rate_invalid'], e['n_drop_rows'])
        self.assertIsNone(e['drop_rows_while_opening_share_of_rate_valid'])

    def test_c4_handle_frame_residual(self):
        room, path, task = synth()
        orig = KC._tcp
        def shifted(b, qa):
            T = orig(b, qa)
            k = next(i for i, s in enumerate(room['steps']) if s[2] == b and s[3] == qa)
            H = KC._H_of(room['steps'][k][4])
            T2 = T.copy()
            T2[:3, 3] = T[:3, 3] + H[:3, 0] * 0.002
            return T2
        KC._tcp = shifted
        try:
            r = J.report_a2(room, path, self.prm, task)
        finally:
            KC._tcp = orig
            os.remove(path)
        hx = r['residual_by_segment']['HOLD']['pos_absmax_mm_handle_xyz']
        self.assertAlmostEqual(hx[0], 2.0, delta=0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
