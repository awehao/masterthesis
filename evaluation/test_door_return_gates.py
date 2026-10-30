#!/usr/bin/env python3
"""door_return_gates 純邏輯測試。python3 evaluation/test_door_return_gates.py"""
import math

import numpy as np

import door_return_gates as G

OPEN, CLOSED = 0.0089, 0.0
DT = 0.01


def rec(pts=()):
    return {'status': 'ok', 'complete': True, 'points': [{'f': f, 'robot_link': lk, 'class': cl, 'evidence_ok': True} for lk, cl, f in pts]}


def grip(cl='bar_in', f=(2.0, 2.5)):
    return rec([('uflite_finger1', cl, f[0]), ('uflite_finger2', cl, f[1])])


class Sim:
    def __init__(self, **kw):
        self.g = G.ReturnGate(**kw)
        self.t, self.q = 100.0, 0.03

    def run(self, s, r=None, qf=(OPEN, OPEN), tcp=0.001, wrist=5.0, hb=0.1, ramp=False, dq=0.0, ramp_at=None, rot=0.001, rel=None):
        n = int(round(s / DT))
        for _ in range(n):
            self.t += DT
            self.q += dq * DT
            rd = ramp or (ramp_at is not None and self.t >= ramp_at)
            ab = self.g.step(self.t, r if r is not None else rec(), self.q, qf, tcp, wrist, hb, rd, tcp_rot=rot, rel=np.eye(4) if rel is None else rel)
            if ab:
                return ab
        return None


def full_cycle(s, bar='in', op=1):
    cl = G.BAR[bar]
    assert s.g.arm(op, bar, s.t)[0]
    s.g.request('close', s.t)
    assert s.run(0.02) is None and s.g.state == 'CLOSING'
    assert s.run(2.2, grip(cl), qf=(0.002, 0.002), ramp=True) is None and s.g.state == 'GRASPED'
    s.g.request('move', s.t)
    assert s.run(0.1, grip(cl), qf=(0.002, 0.002), dq=0.01) is None and s.g.state == 'MOVING'
    s.g.request('open', s.t)
    assert s.run(1.0, grip(cl), qf=(0.002, 0.002)) is None and s.g.state == 'OPENING'
    assert s.run(0.7, rec(), qf=(OPEN, OPEN)) is None and s.g.state == 'RELEASED'
    s.g.request('retreat', s.t)
    assert s.run(0.05) is None and s.g.state == 'RETREAT'
    s.g.request('done', s.t)
    assert s.run(0.05) is None and s.g.state == 'FREE'


def t_full_two_ops():
    s = Sim()
    full_cycle(s, 'in', 1)
    full_cycle(s, 'out', 2)
    assert s.g.abort is None and s.g.n_ops == 2


def t_close_refused_tcp():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    assert s.run(0.2, tcp=0.02) is None and s.g.state == 'FREE'
    assert s.run(0.02, tcp=0.002) is None and s.g.state == 'CLOSING'


def t_close_refused_not_open():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    assert s.run(0.2, qf=(0.004, 0.004)) is None and s.g.state == 'FREE'


def t_wrong_bar_aborts():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    ab = s.run(0.5, grip('bar_out'), qf=(0.002, 0.002))
    assert ab and ab['why'] == 'contact_not_designated' and ab['cls'] == 'bar_out'


def t_panel_aborts():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    ab = s.run(0.2, rec([('uflite_finger1', 'bar_in', 2.0), ('uflite_finger2', 'panel', 1.0)]))
    assert ab and ab['cls'] == 'panel'


def t_nonfinger_aborts():
    s = Sim()
    ab = s.run(0.05, rec([('link5', 'panel', 1.0)]))
    assert ab and ab['why'] == 'contact_not_designated'


def t_armed_free_allows_designated_finger():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    assert s.run(0.2, rec([('uflite_finger1', 'bar_in', 1.0)])) is None
    ab = s.run(0.05, rec([('uflite_finger1', 'bar_out', 1.0)]))
    assert ab and ab['cls'] == 'bar_out'


def t_armed_heartbeat_checked():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    assert s.run(0.1, hb=2.0)['why'] == 'task_heartbeat_stale'


def t_free_contact_aborts():
    s = Sim()
    ab = s.run(0.05, rec([('uflite_finger1', 'bar_in', 1.0)]))
    assert ab and ab['why'] == 'contact_not_designated'


def t_one_finger_no_evidence_timeout():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    ab = s.run(11.0, rec([('uflite_finger1', 'bar_in', 3.0)]), qf=(0.002, 0.002), ramp=True)
    assert ab and ab['why'] == 'grasp_no_contact' and s.g.state == 'CLOSING'


def t_deadline_not_reset_by_resend():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.run(6.0, rec(), qf=(0.002, 0.002), ramp=True)
    s.g.request('close', s.t)
    ab = s.run(4.5, rec(), qf=(0.002, 0.002), ramp=True)
    assert ab and ab['why'] == 'grasp_no_contact'


def t_flicker_resets_evidence():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    for _ in range(3):
        assert s.run(1.5, grip(), qf=(0.002, 0.002), ramp=True) is None
        assert s.run(0.02, rec(), qf=(0.002, 0.002), ramp=True) is None
    assert s.g.state == 'CLOSING'


def t_move_before_grasp_ignored():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('move', s.t)
    assert s.run(0.1) is None and s.g.state == 'FREE'


def t_open_needs_settle():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.run(2.2, grip(), qf=(0.002, 0.002), ramp=True)
    s.g.request('move', s.t)
    s.run(0.02, grip(), qf=(0.002, 0.002))
    s.g.request('open', s.t)
    assert s.run(1.0, grip(), qf=(0.002, 0.002), dq=0.01) is None and s.g.state == 'MOVING'
    assert s.run(0.7, grip(), qf=(0.002, 0.002)) is None and s.g.state == 'OPENING'


def t_close_refused_rotation():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    assert s.run(0.2, rot=0.05) is None and s.g.state == 'FREE'
    assert s.run(0.02) is None and s.g.state == 'CLOSING'


def _grasped():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.run(2.2, grip(), qf=(0.002, 0.002), ramp=True)
    assert s.g.state == 'GRASPED'
    return s


def t_drift_pos_aborts_and_no_rebase():
    s = _grasped()
    T = np.eye(4)
    T[0, 3] = 0.004
    assert s.run(1.0, grip(), qf=(0.002, 0.002), rel=T) is None
    T[0, 3] = 0.006
    ab = s.run(0.05, grip(), qf=(0.002, 0.002), rel=T)
    assert ab and ab['why'] == 'grasp_drift'


def t_drift_rot_aborts():
    s = _grasped()
    a = math.radians(3.0)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    ab = s.run(0.05, grip(), qf=(0.002, 0.002), rel=T)
    assert ab and ab['why'] == 'grasp_drift'


def t_incomplete_evidence_no_accumulate():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    bad = dict(grip(), complete=False)
    for _ in range(4):
        assert s.run(0.6, grip(), qf=(0.002, 0.002), ramp=True) is None
        assert s.run(0.1, bad, qf=(0.002, 0.002), ramp=True) is None
    assert s.g.state == 'CLOSING'


def t_point_not_evidence_ok_invalid():
    r = grip()
    r['points'][0]['evidence_ok'] = False
    assert G.forces(r) is None


def t_missing_evidence_counts_as_finger_loss():
    s = _grasped()
    ab = s.run(0.3, dict(grip(), complete=False), qf=(0.002, 0.002))
    assert ab and ab['why'] == 'finger_loss'


def t_finger_loss():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.run(2.2, grip(), qf=(0.002, 0.002), ramp=True)
    assert s.run(0.15, rec([('uflite_finger1', 'bar_in', 2.0)]), qf=(0.002, 0.002)) is None
    ab = s.run(0.1, rec([('uflite_finger1', 'bar_in', 2.0)]), qf=(0.002, 0.002))
    assert ab and ab['why'] == 'finger_loss'


def t_release_needs_no_force():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.run(2.2, grip(), qf=(0.002, 0.002), ramp=True)
    s.g.request('move', s.t)
    s.run(0.02, grip(), qf=(0.002, 0.002))
    s.g.request('open', s.t)
    s.run(0.7, grip(), qf=(0.002, 0.002))
    assert s.g.state == 'OPENING'
    assert s.run(1.0, grip(f=(0.5, 0.5)), qf=(OPEN, OPEN)) is None and s.g.state == 'OPENING'
    assert s.run(0.6, rec(), qf=(OPEN, OPEN)) is None and s.g.state == 'RELEASED'


def t_heartbeat_stale():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    ab = s.run(0.1, hb=1.5)
    assert ab and ab['why'] == 'task_heartbeat_stale'


def t_heartbeat_ignored_when_free():
    s = Sim()
    assert s.run(0.1, hb=5.0) is None


def t_door_rate_wrist_invalid():
    s = Sim()
    assert s.run(0.1, dq=0.1)['why'] == 'door_rate'
    s = Sim()
    assert s.run(0.02, wrist=31.0)['why'] == 'wrist_force'
    s = Sim()
    assert s.run(0.1, wrist=float('nan'))['why'] == 'wrist_force'
    s = Sim()
    assert s.run(0.3, {'status': 'query_failed'})['why'] == 'contact_record_invalid'


def t_abort_latches_first():
    s = Sim()
    a1 = s.run(0.02, wrist=31.0)
    s.g.request('close', s.t)
    a2 = s.run(0.1, rec([('link5', 'panel', 5.0)]))
    assert a2 is a1 and a1['why'] == 'wrist_force' and s.g.state == 'FREE'


def t_op_sequence_forward_only():
    s = Sim()
    assert s.g.arm(2, 'in', s.t)[0]
    assert not s.g.arm(3, 'out', s.t)[0] or s.g.state == 'FREE'
    full = Sim()
    full_cycle(full, 'in', 2)
    assert not full.g.arm(1, 'out', full.t)[0]
    assert not full.g.arm(3, 'x', full.t)[0]


def t_finger_target():
    s = Sim()
    assert s.g.finger_target() == OPEN
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    assert s.g.finger_target() == CLOSED


def t_latched_freezes_finger():
    s = Sim()
    s.g.arm(1, 'in', s.t)
    s.g.request('close', s.t)
    s.run(0.02)
    s.g.step(s.t + DT, rec([('link5', 'panel', 5.0)]), s.q, (0.004, 0.004), 0.001, 5.0, 0.1, False, fcmd=0.004, tcp_rot=0.0, rel=np.eye(4))
    assert s.g.abort and s.g.finger_target() == 0.004


def t_second_op_requires_arm():
    s = Sim()
    full_cycle(s, 'in', 1)
    s.g.request('close', s.t)
    assert s.run(0.1) is None and s.g.state == 'FREE' and s.g.abort is None


def t_arm_idempotent_and_second_op_fresh_timers():
    s = Sim()
    full_cycle(s, 'in', 1)
    assert s.g.arm(2, 'out', s.t)[0] and s.g.arm(2, 'out', s.t)[0] and s.g.n_ops == 2 and s.g.t_ramp is None
    assert not s.g.arm(1, 'out', s.t)[0]
    s.g.request('close', s.t)
    assert s.run(0.2, ramp=True) is None and s.g.state == 'CLOSING'
    assert s.run(3.0, grip('bar_out'), qf=(0.002, 0.002), ramp=True) is None and s.g.state == 'GRASPED'


if __name__ == '__main__':
    fs = [(k, v) for k, v in list(globals().items()) if k.startswith('t_') and callable(v)]
    for k, f in fs:
        f()
    print(f'{len(fs)} 項全過')
