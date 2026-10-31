#!/usr/bin/env python3
"""瓶子 N1 鏈路純邏輯驗收（無 ROS、無 Isaac）：命令時序同 pipe、逾時不重用舊命令、掃描斷流、壞資料、序號不符。
    python3 evaluation/test_bottle_ros_chain.py
"""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_ros_chain as RC                           # noqa: E402

BAD, N = [], [0]


def chk(name, cond, info=''):
    N[0] += 1
    if not cond:
        BAD.append(f'{name} {info}')


class Tx:
    def __init__(self):
        self.scans, self.cmds = [], []

    def pub_scan(self, st, r):
        self.scans.append(st)

    def pub_cmd(self, u):
        self.cmds.append(np.array(u, float))


class Judge:
    stop = None

    def _stop(self, t, why, **kw):
        self.stop = dict(t=t, why=why, **kw)


def rig():
    ctrl = NS(u_last=np.zeros(9), judge=Judge(), t=0.0)
    ch = RC.RosChain(Tx(), ctrl)
    ch.ping()
    ch.cb_cmd((0.0, 0.0, 0.0))
    ch.cb_diag(diag(1, np.zeros(3), np.zeros(3)))
    ch.cb_obs([1.0, 0.9, 0.0, 0.0])
    ch.start(1.0)
    return ctrl, ch


def diag(c, u_in, u_out):
    a, b = np.broadcast_to(u_in, (3,)) * np.array([1.0, 1.0, 1.0]), np.broadcast_to(u_out, (3,))
    return [float(c), 0.0, 0.0, -1.0, 0.0, *a, *b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def run(ctrl, ch, n, u_of, loopback=True, k0=0):
    """模擬主迴圈：tick 算命令；回覆在下一物理步排空時到。回傳實際套用值。"""
    out, pipe = [], []
    if not hasattr(ch, '_pipe'):
        ch._pipe = [np.zeros(3) for _ in range(RC.D_STEPS)]
    pipe = ch._pipe
    for i in range(k0, k0 + n):
        t = i * RC.PHYS
        ctrl.t = t
        if i % RC.NPH == 0:
            ctrl.u_last = np.r_[u_of(t), np.zeros(6)]
        pipe.append(ctrl.u_last[:3].copy())
        ua = pipe.pop(0)
        n0 = len(ch.sent)
        out.append(ch.shield(t, ua))
        ch.set_now(1.0 + t + RC.PHYS)
        if loopback and len(ch.sent) > n0:
            e = ch.sent[-1]
            ch.cb_cmd(e['u'])
            ch.cb_diag(diag(ch.last_cycle + 1, e['u'], e['u']))
    return np.array(out), pipe


def t_timing():
    ctrl, ch = rig()
    u = lambda t: np.array([0.1 * t, 0.0, 0.0])         # noqa: E731
    out, _ = run(ctrl, ch, 60, u)
    want = [np.zeros(3) if k < RC.D_STEPS else u(((k - RC.D_STEPS) // RC.NPH) * RC.NPH * RC.PHYS) for k in range(60)]
    chk('命令時序同 pipe：第 k 步套用 tick k−8 的命令', np.abs(out - np.array(want)).max() < 1e-12, np.abs(out - np.array(want)).max())
    chk('正常流程不停止', ctrl.judge.stop is None, ctrl.judge.stop)
    chk('序號核對全過', ch.n_seq_bad == 0 and ch.n_pipe_bad == 0, (ch.n_seq_bad, ch.n_pipe_bad))
    r = ch.report()['ros']
    chk('往返與注入分列，總和＝8 步', abs(r['cmd_rtt_sim_ms']['p50'] + r['injected_fill_ms']['p50'] - 80.0) < 1e-6, r)


def t_cmd_timeout():
    ctrl, ch = rig()
    out, _ = run(ctrl, ch, 30, lambda t: np.array([0.3, 0.0, 0.2]))
    chk('斷流前達命令值', abs(out[-1][0] - 0.3) < 1e-12)
    out2, _ = run(ctrl, ch, 60, lambda t: np.array([0.3, 0.0, 0.2]), loopback=False, k0=30)
    chk('回覆斷流走受控停止', ctrl.judge.stop is not None and ctrl.judge.stop['why'] == 'cmd_timeout', ctrl.judge.stop)
    d = np.abs(out2[-1])
    chk('斷流後不重用舊命令：速度降到零', d.max() < 1e-12, d)
    sp = out2[:, 0]
    i0 = int(np.argmax(sp < 0.3))
    chk('減速有界且單調', (np.diff(sp[i0 - 1:]) >= -RC.A_BASE[0] * RC.PHYS - 1e-12).all() and (np.diff(sp[i0 - 1:]) <= 1e-12).all())
    chk('減速第一步就低於舊命令', out2[RC.D_STEPS + RC.NPH + 1][0] < 0.3, out2[RC.D_STEPS + RC.NPH + 1])


def t_bad_cmd():
    ctrl, ch = rig()
    run(ctrl, ch, 12, lambda t: np.array([0.2, 0.0, 0.0]))
    ch.k = 15
    ch.shield(0.15, np.array([0.2, 0.0, 0.0]))
    ch.cb_cmd((float('nan'), 0.0, 0.0))
    chk('非有限回覆不採用', ch.n_bad_cmd == 1 and ch.sent[-1]['out'] is None, ch.n_bad_cmd)
    ch.cb_cmd((0.0, 0.0, 0.0))
    ch.cb_cmd((0.0, 0.0, 0.0))
    chk('多出的回覆被計數，不對應到別筆命令', ch.n_extra >= 1, ch.n_extra)


def t_seq_mismatch():
    ctrl, ch = rig()
    ctrl.u_last = np.r_[0.2, np.zeros(8)]
    u = np.array([0.2, 0.0, 0.0])
    ch.shield(0.0, np.zeros(3))
    ch.cb_cmd(u)
    ch.cb_diag(diag(ch.last_cycle + 1, np.array([0.05, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])))     # Shield 的輸入與送出的不同：回覆錯位
    for k in range(1, RC.D_STEPS):
        ch.shield(k * RC.PHYS, np.zeros(3))
        if k % RC.NPH == 0:
            ch.cb_cmd(u)
            ch.cb_diag(diag(ch.last_cycle + 1, u, u))
    ch.shield(RC.D_STEPS * RC.PHYS, u)
    chk('序號不符走受控停止', ctrl.judge.stop is not None and ctrl.judge.stop['why'] == 'cmd_seq_mismatch', ctrl.judge.stop)


def t_scan():
    _, ch = rig()
    chk('開始前的障礙時間只看來源 stamp', abs(ch.age(0.0) - 0.1) < 1e-9, ch.age(0.0))
    ch.set_now(2.0)
    ch.cb_obs([2.0, 1.5, 5.0, 5.0, 1.0, 2.0, 0.0, 0.5, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0])
    chk('動態與靜態都接入', len(ch.dyn) == 1 and len(ch.stat) == 1 and ch.stat[0]['static'] and ch.stat[0]['margin'] == RC.M_STATIC)
    chk('來源時間算年齡，不看收到時間', abs(ch.age(1.0) - 0.5) < 1e-9, ch.age(1.0))
    o = ch.obstacles(1.0)
    chk('動態前推到生效時刻，靜態不動', abs(o[0]['x'] - 1.25) < 1e-9 and o[1]['x'] == 3.0, o)
    a = ch.age(1.2)
    ch.cb_obs([3.0, 2.2, -1.0, -1.0])
    chk('追蹤器沒輸出時來源時間不前進', ch.age(1.2) == a and ch.n_no_output == 1)
    ch.cb_obs([4.0, 2.3, 5.0, 0.0, float('nan'), 0.0, 0.0, 0.0, 0.0])
    chk('非有限障礙不採用', ch.n_bad_obs == 1 and ch.age(1.2) == a)
    ch.cb_obs([5.0, 2.2, 5.0, 5.0, 1, 2, 0, 0, 0])
    chk('長度不符不採用', ch.n_bad_obs == 2 and ch.age(1.2) == a)
    ch.cb_obs([6.0, 1.4, 0.0, 0.0])
    chk('舊幀不倒退', ch.n_old_obs == 1 and ch.age(1.2) == a)
    chk('掃描斷流時年齡超過門檻', ch.age(1.0 + RC.OBS_MAX_AGE + 0.2) > RC.OBS_MAX_AGE)


def t_ready():
    ch = RC.RosChain(Tx(), NS(u_last=np.zeros(9), judge=Judge(), t=0.0))
    for i in range(3):
        ch.ping()
        ch.cb_cmd((0.0, 0.0, 0.0))
        d = diag(i + 1, np.zeros(3), np.zeros(3))
        d[12] = 1.0
        ch.cb_diag(d)
        ch.cb_obs([float(i + 1), 0.5 + i, 0.0, 0.0])
    chk('Shield 掃描過期時握手不算就緒', not ch.ready(), ch.n_fresh_diag)
    for i in range(3, 6):
        ch.ping()
        ch.cb_cmd((0.0, 0.0, 0.0))
        ch.cb_diag(diag(i + 1, np.zeros(3), np.zeros(3)))
    chk('Shield 連續新鮮後才就緒', ch.ready(), ch.n_fresh_diag)
    ch.ping()
    chk('還有未回覆命令時不就緒', not ch.ready())


def t_constants():
    chk('掃描常數同 bottle_nav_chain', (RC.SCAN_N, RC.SCAN_MIN, RC.SCAN_MAX, RC.RANGE_MIN, RC.RANGE_MAX, RC.SCAN_HZ, RC.OBS_MAX_AGE, RC.M_STATIC) == (NC.SCAN_N, NC.SCAN_MIN, NC.SCAN_MAX, NC.RANGE_MIN, NC.RANGE_MAX, NC.SCAN_HZ, NC.OBS_MAX_AGE, NC.M_STATIC))


def main():
    for f in (t_constants, t_ready, t_timing, t_cmd_timeout, t_bad_cmd, t_seq_mismatch, t_scan):
        f()
    print(f'{N[0] - len(BAD)}／{N[0]} 通過')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
