#!/usr/bin/env python3
"""台架初態啟動測試（Codex r20 §3；無 Isaac）。
  A1 BenchConfig：登錄 r9 合格；d_init 超 0.05 ⇒ 拒（不截斷）；s_init ≠ q_nominal − d_init ⇒ 拒；五軸、NaN、bool ⇒ 拒；check_cli 不符 ⇒ 拒。
  A2 BenchRelease：run_id 不符、source_seq 0／1.5／True、sim_t ≠ t0 ⇒ 拒收宣告；宣告前到達的封裝只計數不釋放；seq 不同 ⇒ 不釋放；
     seq 相同 ⇒ released；逾時 ⇒ aborted（之後不再釋放）；宣告衝突 ⇒ 拒。
  A3 真 E2 鏈（wb_cmd_chain_e2.CmdChainE2，參數同 ev5）：以 (0, s_init) 建立種子 ⇒ 第一筆命令後設定點＝s_init＋r·dt（不被實測 q_nominal 覆寫）；
     重複種子 ⇒ RuntimeError；不給種子（旗標關）⇒ 設定點由實測建立（原行為）。
  A4 任務端起始相位（bench_grasp(init_grasp2(gated2(K3MP2Policy)))，合成世界 test_door_coopen_init_grasp2.InitP2Sim）：
     凍結期（模擬時間停在 t0、bench_state frozen）每輪發布目標 T(H,0)、solver_start，不進 INIT_CHECK、不呼叫監看（注入手指—橫桿力不中止，計 monitor_skipped）；
     釋放且時間前進 ⇒ INIT_CHECK → ENGAGE → … → DONE；相位紀錄無 WAIT_HANDOVER／UNFOLD／ALIGN／APPROACH；
     釋放後注入手指—橫桿力 ⇒ INIT_CHECK 中止；bench_state aborted ⇒ bench_release_failed。
  A5 求解端初值（offset_interval.IntervalOffsetObserver 既有 d0 介面）：d0＝d_init ⇒ d＝d_init、n_upd 0；初值與 d_init 一致性。

    python3 evaluation/test_door_bench_start.py
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import door_bench_start as B                           # noqa: E402

PRM9 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')
REG = json.load(open(PRM9))['bench']


class A1(unittest.TestCase):
    def test_config(self):
        c = B.BenchConfig(REG, 0.05)
        np.testing.assert_allclose(np.subtract(c.q_nominal, c.d_init), c.s_init, atol=1e-12)
        bad = []
        r = copy.deepcopy(REG)
        r['d_init'][1] = -0.06
        r['s_init'][1] = r['q_nominal'][1] + 0.06
        bad.append(r)
        r = copy.deepcopy(REG)
        r['s_init'][0] += 1e-6
        bad.append(r)
        for k in ('q_nominal', 's_init', 'd_init'):
            r = copy.deepcopy(REG)
            r[k] = r[k][:5]
            bad.append(r)
            r = copy.deepcopy(REG)
            r[k][2] = float('nan')
            bad.append(r)
            r = copy.deepcopy(REG)
            r[k][2] = True
            bad.append(r)
        for r in bad + [None]:
            with self.assertRaises(ValueError):
                B.BenchConfig(r, 0.05)
        c.check_cli([float(v) for v in REG['q_nominal_cli'].split(',')], [float(v) for v in REG['s_init_cli'].split(',')])
        with self.assertRaises(ValueError):
            c.check_cli([float(v) for v in REG['q_nominal_cli'].split(',')], [float(v) + 1e-6 for v in REG['s_init_cli'].split(',')])


class A2(unittest.TestCase):
    def test_release(self):
        r = B.BenchRelease('run', 0.0, 10.0, 100.0)
        for msg in ({'run_id': 'x', 'source_seq': 1, 'sim_t': 0.0}, {'run_id': 'run', 'source_seq': 0, 'sim_t': 0.0},
                    {'run_id': 'run', 'source_seq': 1.5, 'sim_t': 0.0}, {'run_id': 'run', 'source_seq': True, 'sim_t': 0.0},
                    {'run_id': 'run', 'source_seq': 1, 'sim_t': 0.05}, 'bad'):
            self.assertFalse(r.on_announce(msg))
        self.assertEqual(r.try_release(3, 101.0), 'frozen')
        self.assertEqual(r.n_env_before_announce, 1)
        self.assertTrue(r.on_announce({'run_id': 'run', 'source_seq': 4, 'sim_t': 0.0}))
        self.assertFalse(r.on_announce({'run_id': 'run', 'source_seq': 5, 'sim_t': 0.0}))
        self.assertEqual(r.try_release(3, 101.1), 'frozen')
        self.assertEqual(r.try_release(4, 101.2), 'released')
        self.assertEqual(r.released['source_seq'], 4)
        r2 = B.BenchRelease('run', 0.0, 10.0, 100.0)
        self.assertEqual(r2.try_release(None, 111.0), 'aborted')
        r2.on_announce({'run_id': 'run', 'source_seq': 1, 'sim_t': 0.0})
        self.assertEqual(r2.try_release(1, 111.5), 'aborted')


def mk_chain():
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    from wb_cmd_chain_e2 import CmdChainE2
    from wb_wheel_limit import WheelLimitConfig
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    lim = np.array(K.joint_limits())
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=0.5, wheel_ok=lambda vx, vy, wz: (True, None),
                      joint_lower=tuple(float(x) for x in lim[0, 3:]), joint_upper=tuple(float(x) for x in lim[1, 3:]),
                      joint_margin=0.05, mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=0.5, dt_max=0.05))


class A3(unittest.TestCase):
    def test_e2_chain_seed(self):
        c = mk_chain()
        s_init, qn = np.asarray(REG['s_init']), np.asarray(REG['q_nominal'])
        c.seed_from_handover([0.0] * 9, list(s_init), 0.0, -1, source='bench_start')
        r = np.array([0.0, 0.0, 0.0, 0.01, -0.02, 0.0, 0.0, 0.0, 0.0])
        c.receive(list(r), 0.0)
        out = c.step(0.0, 0.01, list(qn))
        sp = np.asarray(out[1] if isinstance(out, tuple) else out.arm_setpoint, float)
        np.testing.assert_allclose(sp, s_init + r[3:] * 0.01, atol=1e-12)
        self.assertGreater(float(np.abs(sp - qn).max()), 0.01)
        with self.assertRaises(RuntimeError):
            c.seed_from_handover([0.0] * 9, list(s_init), 0.01, 0, source='again')
        c2 = mk_chain()
        c2.receive([0.0] * 9, 0.0)
        out2 = c2.step(0.0, 0.01, list(qn))
        sp2 = np.asarray(out2[1] if isinstance(out2, tuple) else out2.arm_setpoint, float)
        np.testing.assert_allclose(sp2, qn, atol=1e-12)


class A4(unittest.TestCase):
    def mk(self, frozen_n=20, bench_abort=False, inject=None):
        import door_coopen_approach_gate as AG
        import door_coopen_approach_gate2 as AG2
        import door_coopen_bench_grasp as BG
        import door_coopen_init_grasp2 as IG2
        import door_k3m_geom_monitor as GM
        import door_k3m_policy_p2 as KP2
        import test_door_coopen_init_grasp2 as T2
        import test_door_k3a_policy as TK
        import test_door_k3m_policy as TM

        class BSim(T2.InitP2Sim):
            def __init__(s):
                super().__init__()
                gfn = (lambda q, a, b: {k: (v, 'x') for k, v in TM.NOM.items()})
                s.pol = BG.bench_grasp(IG2.init_grasp2(AG2.gated2(KP2.K3MP2Policy)))(s.cfg, TK.T_HG(), TM.RUN, TM.COR,
                                                                                  GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
                s.pol.gate_init(AG.ApproachGateConfig())
                s.n_meas, s.t0 = 0, None
                s.owner = 'wholebody'

            def meas(s):
                m = super().meas()
                s.n_meas += 1
                s.t0 = m.t if s.t0 is None else s.t0
                if s.n_meas <= frozen_n:
                    m.t, m.bench_state = s.t0, ('aborted' if bench_abort and s.n_meas == frozen_n else 'frozen')
                else:
                    m.bench_state = 'released'
                m.bench_t0 = s.t0
                if inject:
                    inject(s, m)
                return m
        return BSim()

    def test_nominal(self):
        s = self.mk()
        pol = s.run(t_max=200.0)
        ph = s.phases()
        self.assertEqual(ph[0], 'BENCH_START')
        for p in ('INIT_CHECK', 'ENGAGE', 'HOLD', 'OPEN'):
            self.assertIn(p, ph)
        for p in ('WAIT_HANDOVER', 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            self.assertNotIn(p, ph)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        fr = [(t, p, a) for (t, p, a, m) in s.acts if p == 'BENCH_START']
        self.assertGreaterEqual(len(fr), 19)
        self.assertTrue(all(a['target'] is not None and a['solver_start'] for (_, _, a) in fr))
        self.assertGreaterEqual(pol.bench_stats['monitor_skipped_cycles'], 20)

    def test_contact_frozen_vs_released(self):
        def inj(s, m):
            m.finger_bar_max = (0.2, 0.0) if s.pol.phase in ('BENCH_START', 'INIT_CHECK') else m.finger_bar_max
        s = self.mk(inject=inj)
        pol = s.run(t_max=60.0)
        self.assertEqual(pol.abort, 'finger_bar_outside_allowed_phase')
        self.assertEqual(next(e for e in pol.events if e.get('abort'))['phase'], 'INIT_CHECK')

    def test_bench_abort(self):
        s = self.mk(bench_abort=True)
        pol = s.run(t_max=60.0)
        self.assertEqual(pol.abort, 'bench_release_failed')
        self.assertNotIn('INIT_CHECK', s.phases())


class A5(unittest.TestCase):
    def test_observer_d0(self):
        import offset_interval as OINT
        idn = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
        o = OINT.IntervalOffsetObserver(idn['alpha'], idn['phys_dt_measured_s'], 0.5, 0.05, d0=REG['d_init'])
        np.testing.assert_allclose(o.d, REG['d_init'], atol=0)
        self.assertEqual(o.n_upd, 0)
        self.assertTrue(math.isfinite(float(np.max(np.abs(o.d)))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
