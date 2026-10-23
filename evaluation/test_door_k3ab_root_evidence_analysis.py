#!/usr/bin/env python3
"""根取證分析器反例（合成資料）：完整時 evidence_complete；缺步／重複／缺欄／寫檔未完成／缺快照 ⇒ 不完整並列原因；
handle 計數；api 參數核對（yaw 轉換）；B 回讀差；D_k 對 A_{k+1}；零命令段速度；門扇 −ω_z 積分對 Δq（理想一致 ⇒ 比 1）；
C_recorded 對 C 讀值。

    python3 evaluation/test_door_k3ab_root_evidence_analysis.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_root_evidence_analysis as A          # noqa: E402


def quat_z(th):
    return [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]


def make(n=400, latch=200, drift=0.003, wz_door=0.004, yaw=math.pi / 2):
    rows, steps = [], []
    x, y, q = 0.6, -0.53, 0.01
    for k in range(n):
        t = round(0.01 * k, 6)
        cmd = [0.003, -0.002, 0.0] if k < latch else [0.0, 0.0, 0.0]
        wx = cmd[0] * math.cos(yaw) - cmd[1] * math.sin(yaw)
        wy = cmd[0] * math.sin(yaw) + cmd[1] * math.cos(yaw)
        vlin = [0.0, 0.0] if k < latch else [drift, 0.0]
        A_pose = [x, y, 0.0] + quat_z(yaw)
        A_vel = vlin + [0.0, 0.0, 0.0, 0.0]
        leaf = [0.4, 0.0, 1.0] + quat_z(-q) + [0, 0, 0, 0, 0, -wz_door]
        r = {'schema': 'x', 'step': k, 't': t, 'handle_leaf_A': True, 'handle_robot_A': True, 'A_root_pre': A_pose, 'A_root_vel_pre': A_vel,
             'A_leaf_pre': leaf, 'api_args': [wx, wy, 0.0, 0.0, 0.0, cmd[2]], 'api_yaw': [yaw], 'cmd_body': cmd,
             'B_root_vel_post_write': [wx, wy, 0.0, 0.0, 0.0, cmd[2]], 'handle_robot_B': True, 'handle_leaf_C': True,
             'C_leaf_pre_step': leaf, 'C_root_pre_step': A_pose + A_vel, 'C_recorded': [q, wz_door]}
        x += vlin[0] * 0.01
        q += wz_door * 0.01
        D_pose = [x, y, 0.0] + quat_z(yaw)
        r.update({'handle_leaf_D': True, 'handle_robot_D': True, 'D_t': [t + 0.01], 'D_root_post_step': D_pose + A_vel,
                  'D_leaf_post_step': [0.4, 0.0, 1.0] + quat_z(-q) + [0, 0, 0, 0, 0, -wz_door], 'evidence_s': [0.0004], 'missing': []})
        rows.append(r)
        steps.append([k, t])
    room = {'steps': steps, 'events': [{'k2_abort_latched': {'step': latch, 'sim_t': 0.01 * latch}}],
            'root_evidence': {'complete': True, 'n': n, 'n_err': {}, 'errors': []}}
    task = {'policy_events': [{'to': 'OPEN', 't': 0.5}, {'t': 1.97, 'abort': 'x'}]}
    snap = {'modules': {'objects': {'RigidPrim': {'file': 'x'}}}}
    return rows, room, task, snap


class T(unittest.TestCase):
    def test_complete_and_consistent(self):
        R = A.analyse(*make())
        self.assertTrue(R['completeness']['evidence_complete'], R['completeness'])
        self.assertEqual(R['handle']['handle_leaf_D']['true'], 400)
        self.assertLess(R['api_vs_cmd_maxabs']['max'], 1e-12)
        self.assertLess(R['B_readback_vs_api_maxabs']['max'], 1e-12)
        self.assertLess(R['D_k_vs_A_k1']['pose_maxabs']['max'], 1e-12)
        z = R['zero_command_after_latch']
        self.assertEqual(z['n'], 200)
        self.assertAlmostEqual(z['D_root_lin_speed']['median'], 0.003)
        self.assertAlmostEqual(z['D_vel_over_pose_diff']['median'], 1.0, places=6)
        for v in R['door']['D_omega_z_integral_vs_dq'].values():
            self.assertAlmostEqual(v['ratio'], 1.0, places=3)
        self.assertLess(R['door']['C_recorded_vs_C_read']['qdot_absdiff']['max'], 1e-12)

    def test_incomplete_reasons(self):
        rows, room, task, snap = make()
        del rows[50]
        rows.append(dict(rows[10]))
        rows[20]['missing'] = ['B_root_vel_post_write']
        room['root_evidence']['complete'] = False
        R = A.analyse(rows, room, task, None)
        rs = R['completeness']['incomplete_reasons']
        self.assertFalse(R['completeness']['evidence_complete'])
        for k in ('missing_steps_1', 'duplicate_steps_1', 'rows_with_missing_or_errors_1', 'writer_not_complete', 'physics_snapshot_missing'):
            self.assertIn(k, rs)

    def test_handle_false_counted_and_api_mismatch(self):
        rows, room, task, snap = make()
        rows[5]['handle_leaf_C'] = False
        rows[6]['handle_leaf_C'] = None
        rows[7]['api_args'] = [1.0] + rows[7]['api_args'][1:]
        R = A.analyse(rows, room, task, snap)
        self.assertEqual((R['handle']['handle_leaf_C']['false'], R['handle']['handle_leaf_C']['none']), (1, 1))
        self.assertGreater(R['api_vs_cmd_maxabs']['max'], 0.9)

    def test_velocity_inconsistent_door(self):
        rows, room, task, snap = make()
        for r in rows:                                              # 記錄的角速度是實際門角變化的 4 倍
            r['D_leaf_post_step'][12] *= 4
        R = A.analyse(rows, room, task, snap)
        for v in R['door']['D_omega_z_integral_vs_dq'].values():
            self.assertAlmostEqual(v['ratio'], 4.0, places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
