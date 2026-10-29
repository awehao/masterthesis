"""dual_source_executor_r2＋wb_cmd_chain_e2t＋door_k3m_applied_src_t：全身 → 導航 → 全身往返（離線，真命令鏈）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_authority import AUTH_NAV, AUTH_WHOLEBODY          # noqa: E402
from door_k3m_applied_src_t import EnvIntakeT                    # noqa: E402
from door_nav_envelope_monitor import EnvelopeMonitor           # noqa: E402
from dual_source_executor_r2 import DualSourceExecutorR2        # noqa: E402
from wb_cmd_chain_e2t import CmdChainE2T                        # noqa: E402
from wb_wheel_limit import WheelLimitConfig                     # noqa: E402

STOW = (0.0, 0.0, 0.15, 0.0, -1.10, 0.0)
F = (0.0089, 0.0089)
DT = 0.01


def chain():
    return CmdChainE2T(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=lambda *a: (True, ''),
                       joint_lower=(-3.0,) * 6, joint_upper=(3.0,) * 6, joint_margin=0.05, mode='solver_drawer',
                       wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05))


class World:
    def __init__(self, rho=0.37, nav_ready_from=0):
        self.ch = chain()
        self.closed, self.opened, self.viol = [], [], []
        self.nav_ready_from = nav_ready_from
        self.env = EnvelopeMonitor(lambda q, f: (self.rho, 'x'))
        self.rho = rho
        self.ex = DualSourceExecutorR2(self.ch, initial=AUTH_NAV, stow_setpoint=STOW, max_seed_age_s=0.05,
                                       envelope=self.env, nav_ready_fn=lambda t: ((t >= self.nav_ready_from), '導航未備妥'),
                                       on_tenure_close=lambda k, t: self.closed.append((k, t)),
                                       on_tenure_open=lambda k, t: self.opened.append((k, t)),
                                       on_envelope_violation=lambda first, t: self.viol.append(t),
                                       nav_accel_max=(1.5, 1.0, 2.0))
        self.k = 0
        self.q = list(STOW)

    def run(self, n, nav=(0.03, 0.0, 0.0), wb=None):
        out = []
        for _ in range(n):
            t = self.k * DT
            if wb is not None:
                self.ch.receive(list(wb), t)
            r = self.ex.step(self.k, t, DT, nav_cmd=nav, q_arm_measured=self.q, meas_vb=(0.03, 0.0, 0.0), fingers=F)
            out.append(r)
            self.k += 1
        return out


WB1 = (0.03, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0)      # j3 以 0.2 rad/s 動 ⇒ 設定點離開 STOW
WB2 = (0.03, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0)


def first_entry(w):
    w.run(10)
    assert w.ex.request_handover(AUTH_WHOLEBODY, w.k, w.k * DT, window_steps=20).ok
    w.run(5, wb=WB1)
    assert w.ex.owner == AUTH_WHOLEBODY and w.ch.tenure == 0 and w.ch.tenure_open
    w.run(40, wb=WB1)


def test_round_trip():
    w = World(nav_ready_from=0.80)
    first_entry(w)
    sp_r1 = list(w.ch.setpoint)
    assert abs(sp_r1[2] - STOW[2]) > 0.05
    v = w.ex.request_nav(w.k, w.k * DT, w.q, F, window_steps=40)
    assert v is not None and v.ok
    res = w.run(20, wb=WB1)                       # 導航 0.80 s 前未備妥 ⇒ 順延、全身繼續
    assert w.ex.owner == AUTH_WHOLEBODY and w.ex.n_nav_not_ready > 0
    res = w.run(15)                               # 0.80 s 起備妥 ⇒ 提交
    assert w.ex.owner == AUTH_NAV
    lg = w.ex.loop_log[-1]
    assert lg['what'] == 'wb_to_nav' and w.closed == [(1, lg['sim_t'])]
    hold = tuple(lg['arm_hold'])
    assert hold != STOW and not w.ch.tenure_open and w.ch.snap is None
    assert all(r.arm_setpoint == hold for r in res if r.owner == AUTH_NAV)
    w.run(30)
    # 再次進入：新任期命令開始送 ⇒ 承接導航最後套用＋保持設定點
    assert w.ex.request_handover(AUTH_WHOLEBODY, w.k, w.k * DT, window_steps=20).ok
    nav_last = w.ex.direct_applied
    w.run(5, wb=WB2)
    assert w.ex.owner == AUTH_WHOLEBODY and w.ch.tenure == 1 and w.ch.tenure_open
    assert w.opened and w.opened[-1][0] == 1
    su = w.ex.seed_used
    assert tuple(su['arm_setpoint']) == hold
    assert np.allclose(su['u_applied'][:3], nav_last[0][:3], atol=0.02)
    assert w.ex.vetted_matched is True
    assert w.ch.tenures[0]['setpoint'] == list(hold)


def test_nav_never_ready_cancels():
    w = World(nav_ready_from=99.0)
    first_entry(w)
    assert w.ex.request_nav(w.k, w.k * DT, w.q, F, window_steps=10).ok
    w.run(20, wb=WB1)
    assert w.ex.owner == AUTH_WHOLEBODY and w.ex.n_nav_cancelled == 1 and w.ch.tenure_open


def test_nav_request_refused_by_envelope():
    w = World(rho=0.386)
    first_entry(w)
    assert w.ex.request_nav(w.k, w.k * DT, w.q, F) is None
    w.rho = 0.37
    assert w.ex.request_nav(w.k, w.k * DT, w.q, None) is None          # 手指讀值缺 ⇒ 失效封閉


def test_envelope_violation_decelerates():
    w = World()
    first_entry(w)
    assert w.ex.request_nav(w.k, w.k * DT, w.q, F, window_steps=10).ok
    w.run(10, nav=(0.25, 0.0, 0.0))
    assert w.ex.owner == AUTH_NAV
    w.run(20, nav=(0.25, 0.0, 0.0))
    v_before = w.ex.direct_applied[0][0]
    w.rho = 0.395
    res = w.run(30, nav=(0.25, 0.0, 0.0))
    assert w.viol and w.ex.env_violated
    vx = [r.base_cmd[0] for r in res]
    assert vx[0] < v_before and vx[-1] == 0.0 and all(b <= a + 1e-12 for a, b in zip(vx, vx[1:]))


def test_intake_rejects_stale_tenure():
    ch = chain()
    it = EnvIntakeT(run_id_num=None, stage_id=2, kinds=(0,))
    env = lambda seq, k: {'stage_id': 2, 'kind': 0, 'source_seq': seq, 'derived': True, 'u': list(WB1), 'tenure': k}
    assert it.receive(env(1, 0), ch, 0.0) == 'fed'
    it.set_expected(1, 0.1)
    assert it.receive(env(2, 0), ch, 0.11) == 'stale_tenure' and it.n_stale == 1
    assert it.receive(env(3, 1), ch, 0.12) == 'fed'
    no_tag = {k: v for k, v in env(4, 1).items() if k != 'tenure'}
    assert it.receive(no_tag, ch, 0.13) == 'stale_tenure'
    try:
        it.set_expected(0, 0.2)
        assert False
    except ValueError:
        pass


def test_chain_rebase_guards():
    ch = chain()
    ch.seed_from_handover([0.0] * 9, STOW, 0.0, 0)
    try:
        ch.rebase_from_handover([0.0] * 9, STOW, 0.1, 1)
        assert False
    except RuntimeError:
        pass
    ch.close_tenure(0.2, 2)
    ch.fail = 'x'
    try:
        ch.seed_from_handover([0.0] * 9, STOW, 0.3, 3)
        assert False
    except RuntimeError:
        pass


def test_envelope_v4_roundtrip():
    import cmd_envelope_v4 as E4
    import wgmpc_cmd_envelope as V2
    d = E4.encode4(7.0, 12, V2.ST_SOLVER, 12, 12, True, 3.5, [0.01] * 9, 4, 1)
    e = E4.decode4(d)
    assert e['tenure'] == 1 and e['mode_seq'] == 4 and e['source_seq'] == 12 and len(d) == E4.NFIELD4
    for bad in (d[:-1], d[:-1] + [-1.0], d[:-1] + [0.5]):
        try:
            E4.decode4(bad)
            assert False
        except ValueError:
            pass
