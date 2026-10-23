#!/usr/bin/env python3
"""封裝進鏈與套用來源（door_k3m_applied_src）＋真的 CmdChainE2 與 K2AbortLatch（無 ROS、無 Isaac）：
  延遲：封裝晚到 ⇒ 到達前套用的是前一筆的序號；重送：同序號重收不改來源語意；停更：超過 max_cmd_age ⇒ 鏈逾時、來源 −1；
  中止優先：閂鎖後封裝不進鏈、鏈內命令過期 ⇒ 來源 −1；非全身控制 ⇒ −1；
  舊命令重新生效：零速確認、停止發布後，晚到的舊非零封裝被鏈套用 ⇒ PauseSupervisor 判 old_command_after_stop。

    python3 evaluation/test_door_k3m_applied_src.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_io as KIO                                 # noqa: E402
import door_k3m_applied_src as A                         # noqa: E402
import door_k3m_pause_sup as PS                          # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                   # noqa: E402
from wb_wheel_limit import WheelLimitConfig              # noqa: E402

DT = 0.01
Q0 = [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0]


def mk_chain():
    c = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=lambda vx, vy, wz: (True, None),
                   joint_lower=(-6.28,) * 6, joint_upper=(6.28,) * 6, joint_margin=0.0, mode='solver_drawer',
                   wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05), base_accel_max_lin=0.5, base_accel_max_ang=2.0)
    c.seed_from_handover([0.0] * 9, Q0, 0.0, 0, source='test')
    return c


def env(seq, u):
    return {'run_id': 1.0, 'source_seq': seq, 'u': list(u)}


U1 = [0.002, 0.0, 0.0] + [0.0] * 6
Z = [0.0] * 9


class T(unittest.TestCase):
    def test_delay_resend_stale(self):
        c, I = mk_chain(), A.EnvIntake()
        I.receive(env(1, U1), c, 0.0)
        out = []
        for k in range(40):
            t = round(k * DT, 6)
            if k == 3:                                        # 第 2 筆晚到（延遲）
                I.receive(env(2, U1), c, t)
            if k == 4:                                        # 重送第 2 筆
                I.receive(env(2, U1), c, t)
            c.step(t, DT, Q0)
            out.append(A.applied_source(c, I.map, True))
        self.assertEqual([s for s, m in out[:3]], [1, 1, 1])
        self.assertEqual(out[3], (2, 0))
        self.assertEqual(out[4], (2, 0))
        self.assertEqual(out[30], (-1, 1))                    # 停更 > 0.2 s ⇒ 逾時、來源 −1
        self.assertEqual(A.applied_source(c, I.map, False)[0], -1)   # 非全身控制 ⇒ −1

    def test_abort_priority(self):
        c, I, L = mk_chain(), A.EnvIntake(), KIO.K2AbortLatch()
        I.receive(env(1, U1), c, 0.0, filt=L.filter_wb)
        c.step(0.0, DT, Q0)
        self.assertEqual(A.applied_source(c, I.map, True)[0], 1)
        L.on_msg('test_abort', 1, 0.01, 0.0)
        L.expire_chain(c)
        self.assertFalse(I.receive(env(2, U1), c, 0.02, filt=L.filter_wb))   # 閂鎖後不進鏈
        c.step(0.02, DT, Q0)
        self.assertEqual(A.applied_source(c, I.map, True), (-1, 1))

    def test_old_command_after_stop(self):
        sup = PS.PauseSupervisor()
        s = sup.seq
        s.state, s.zero_seq0, s.t_stop = 'STOPPED', 10, 5.0                # 已確認零速、停止發布
        c, I = mk_chain(), A.EnvIntake()
        I.receive(env(10, Z), c, 5.0)
        c.step(5.0, DT, Q0)
        src, mc = A.applied_source(c, I.map, True)
        sup.applied(500, 5.0, Z[:3], Z[3:], src)
        self.assertEqual(sup.state, 'STOPPED')                              # 在途零速命令可接受
        I.receive(env(7, U1), c, 5.01)                                      # 晚到的舊非零命令
        c.step(5.01, DT, Q0)
        src, mc = A.applied_source(c, I.map, True)
        self.assertEqual(src, 7)
        sup.applied(501, 5.01, U1[:3], U1[3:], src)
        self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'old_command_after_stop'))
        sup2 = PS.PauseSupervisor()
        sup2.seq.state, sup2.seq.zero_seq0 = 'STOPPED', 10
        sup2.applied(600, 6.0, Z[:3], Z[3:], 9)                             # 數值為零但序號舊 ⇒ 仍失敗
        self.assertEqual(sup2.seq.why, 'old_command_after_stop')
        sup3 = PS.PauseSupervisor()
        sup3.seq.state, sup3.seq.zero_seq0 = 'STOPPED', 10
        sup3.applied(700, 7.0, Z[:3], Z[3:], -1)                            # 逾時零值 ⇒ 接受
        self.assertEqual(sup3.state, 'STOPPED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
