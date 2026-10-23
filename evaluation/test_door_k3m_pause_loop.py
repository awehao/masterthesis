#!/usr/bin/env python3
"""door_k3m_pause_loop 的核對：(1) 每輪發布、不改寫時與 CL.run 逐元素相同；(2) 停止發布後不入列，在途命令照常套用、
來源序號沿管線傳遞，年齡 > 0.2 s 才走 limit9_timeout（src＝None）；(3) 預測外推不產生套用回報（回報步序連續、每輪恰 n_phys 筆）。

    python3 evaluation/test_door_k3m_pause_loop.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_loop as PL                       # noqa: E402

RP, CL = PL.RP, PL.CL


def setup():
    ident = json.load(open(RP.IDENT))
    t = json.load(open(os.path.join(RP.RUN, 'task.json')))
    r = json.load(open(os.path.join(RP.RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    s_open = int(next(e['step'] for e in t['trace'] if e.get('phase') == 'OPEN'))
    so = next(x for x in r['steps'] if int(x[I['step']]) == s_open)
    q0 = np.array(list(so[I['base_xyth']]) + list(so[I['q_arm_meas']]), float)
    s0 = q0[3:9] - np.asarray(ident['bias_rad']) / np.asarray(ident['alpha'])
    K = CL.WholeBodyKinematics.from_urdf_file(RP.C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05)
    T = K.fk(q0, 'link_tcp') if hasattr(K, 'fk') else None
    if T is None:
        T = RP.KC._tcp(list(q0[:3]), list(q0[3:9]))
    T = np.array(T, float).copy()
    T[0, 3] -= 0.03                                     # 移動目標，讓命令非零
    return cfg, K, q0, s0, T


class Loop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.K, cls.q0, cls.s0, cls.T = setup()

    def test_equivalent_to_cl_run(self):
        a = CL.run(self.cfg, self.K, self.q0, self.s0, self.T, t_end=1.5, hold_s=1e9, delay_cycles=1.4, d_pub_cycles=0.6,
                   comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)['rec']
        cfg2 = CL.make_cfg(json.load(open(RP.IDENT)), w_s=0.001, w_a=0.05, w_s_arm=0.05)
        b = PL.run_loop(cfg2, self.K, self.q0, self.s0, lambda t: self.T, t_end=1.5)
        n = min(len(a['t']), len(b['t']))
        self.assertGreater(n, 25)
        np.testing.assert_array_equal(np.asarray(a['q'])[:n], b['q'][:n])
        np.testing.assert_array_equal(np.asarray(a['u_app'])[:n], b['u_app'][:n])
        self.assertTrue(np.abs(b['u_app']).max() > 1e-4)

    def test_equivalent_with_moving_target_pre(self):
        def mk():
            st = {'T': self.T.copy()}

            def pre(t, q, cfg_):
                st['T'] = self.T.copy()
                st['T'][1, 3] += 0.02 * t                                  # pre 逐輪改目標（下一輪生效）
            return st, pre
        st1, pre1 = mk()
        a = CL.run(self.cfg, self.K, self.q0, self.s0, lambda t: st1['T'], t_end=1.5, hold_s=1e9, delay_cycles=1.4, d_pub_cycles=0.6,
                   comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0, pre_solve=pre1)['rec']
        st2, pre2 = mk()
        cfg2 = CL.make_cfg(json.load(open(RP.IDENT)), w_s=0.001, w_a=0.05, w_s_arm=0.05)
        b = PL.run_loop(cfg2, self.K, self.q0, self.s0, lambda t: st2['T'], t_end=1.5, pre=pre2)
        n = min(len(a['t']), len(b['t']))
        np.testing.assert_array_equal(np.asarray(a['q'])[:n], b['q'][:n])

    def test_stop_publishing_semantics(self):
        cfg2 = CL.make_cfg(json.load(open(RP.IDENT)), w_s=0.001, w_a=0.05, w_s_arm=0.05)
        T_STOP = 0.8
        rep = []
        b = PL.run_loop(cfg2, self.K, self.q0, self.s0, lambda t: self.T, t_end=1.6,
                        decide=lambda t, r, a: (t < T_STOP - 1e-9, None), skip_solve=lambda t: t >= T_STOP - 1e-9,
                        on_applied=lambda j, t, u, src: rep.append((j, t, u, src)))
        js = [x[0] for x in rep]
        self.assertEqual(js, list(range(len(js))))                         # 只有實際步回報、連續
        self.assertEqual(len(js), 5 * len(b['t']))
        last_seq = max(x for x in b['seq'] if x is not None)
        after = [x for x in rep if x[1] >= T_STOP - 1e-9]
        # 在途命令照常套用：停止發布後前幾步仍是最後幾筆序號
        self.assertTrue(any(x[3] == last_seq for x in after))
        held_until = max(x[1] for x in rep if x[3] == last_seq)
        t_pub_last = (T_STOP - 0.05) + 0.6 * 0.05
        self.assertLessEqual(held_until - t_pub_last, 0.2 + 1e-9)
        to = [x for x in after if x[3] is None]
        self.assertTrue(to and min(x[1] for x in to) - t_pub_last > 0.2 - 1e-9)   # 逾時才 src=None
        self.assertTrue(all(np.all(x[2][3:] == 0.0) for x in to))          # 逾時：手臂立即 0
        self.assertNotIn(None, [x[3] for x in rep if x[1] < held_until])


if __name__ == '__main__':
    unittest.main(verbosity=2)
