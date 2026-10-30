#!/usr/bin/env python3
"""返程門操作的執行端順向准入（Codex 20261030 m6e 第 2–4 點；規格 results/specs/door_return_phase3_r1.md）。純邏輯，無 ROS／Isaac。
不用 W2 模式表、不倒放；每次門操作一個 op（bar＝'in' 推開／'out' 推關），序列只能順向：
  FREE → (request close＋准入) → CLOSING → (指定橫桿兩指連續 2 s) → GRASPED → (request move) → MOVING
       → (request open＋門停穩) → OPENING → (張開＋橫桿無力連續 0.5 s) → RELEASED → (request retreat) → RETREAT → (request done) → FREE
准入與監看：
  - 閉爪：張爪就緒、TCP 誤差 ≤ tcp_tol、無非指定接觸；
  - 接觸：未武裝 FREE 不准任何門接觸；arm 後至 done 前手指可碰指定橫桿（滑入接近含在內）；CLOSING／RETREAT／OPENING 只准手指碰指定橫桿；GRASPED／MOVING 同，且雙指連續低於門檻逾 finger_loss_s ⇒ 中止；非手指連桿或門板／立柱力 > contact_abort_n ⇒ 中止；
  - 抓持期限：閉爪斜坡完成後 grasp_timeout_s 未建立證據 ⇒ 中止；每個 op 只起算一次，不因請求重送重設；
  - 門速 > door_rate_abort、腕力 ≥ wrist_abort_n、接觸紀錄無效 > invalid_max_s、任務心跳逾 hb_max_s（牆鐘）⇒ 中止；
  - 中止第一原因閂鎖；同輪中止優先於任何轉移；閂鎖後 finger_target 凍結最後實際值。
r2（Codex m6g）：證據只採 status ok、complete 且 evidence_ok；缺證據步重置所有連續計時並算入失抓；閉爪另核姿態；抓持成立固定 E0（把手系下 TCP），持握期核 5 mm／2° 不重設；失抓 0.2 s（同去程）；停穩用 0.1 s 跨距位置差分。
力只算 evidence_ok 的點。時間皆模擬秒（心跳例外，呼叫端給牆鐘年齡）。
"""
from __future__ import annotations

import math

import numpy as np

from door_regrasp_close_settle2 import RateFromPos

VERSION = 'door_return_gates/r2'
FINGERS = ('uflite_finger1', 'uflite_finger2')
BAR = {'in': 'bar_in', 'out': 'bar_out'}


def forces(rec):
    """接觸紀錄 → {'fingers': {finger: 指定類別→力}, 'other': [(link, class, f)]}；status 非 ok 回 None。"""
    if not isinstance(rec, dict) or rec.get('status') != 'ok' or rec.get('complete') is not True:
        return None
    fg = {f: {} for f in FINGERS}
    other = {}
    for p in rec.get('points') or []:
        f = float(p.get('f', 0.0) or 0.0)
        if not math.isfinite(f) or p.get('evidence_ok') is not True:
            return None
        lk, cl = p.get('robot_link'), p.get('class')
        if lk in fg:
            fg[lk][cl] = fg[lk].get(cl, 0.0) + f
        else:
            other[(lk, cl)] = other.get((lk, cl), 0.0) + f
    return {'fingers': fg, 'other': other}


def pre_release_offset(i, i_end, n_rows, m):
    """移動段末預退量（m）：列 i_end−n_rows→i_end 由 0 平滑升到 m（端點速度 0），之後維持 m。"""
    if m <= 0.0:
        return 0.0
    f = min(1.0, max(0.0, (i - (i_end - n_rows)) / n_rows))
    return m * (10 * f ** 3 - 15 * f ** 4 + 6 * f ** 5)


class ReturnGate:
    def __init__(self, contact_min_n=0.5, contact_abort_n=0.1, hold_s=2.0, grasp_timeout_s=10.0, release_hold_s=0.5, finger_loss_s=0.2,
                 door_rate_abort=0.05, wrist_abort_n=30.0, invalid_max_s=0.2, hb_max_s=1.0, tcp_tol=0.005, finger_open=0.0089, finger_closed=0.0,
                 settle_rate=0.002, settle_hold_s=0.5, open_tol=1e-4, tcp_rot_tol=math.radians(2.0), drift_pos=0.005, drift_rot=math.radians(2.0)):
        self.c = dict(contact_min_n=contact_min_n, contact_abort_n=contact_abort_n, hold_s=hold_s, grasp_timeout_s=grasp_timeout_s, release_hold_s=release_hold_s,
                      finger_loss_s=finger_loss_s, door_rate_abort=door_rate_abort, wrist_abort_n=wrist_abort_n, invalid_max_s=invalid_max_s, hb_max_s=hb_max_s, tcp_tol=tcp_tol,
                      finger_open=finger_open, finger_closed=finger_closed, settle_rate=settle_rate, settle_hold_s=settle_hold_s, open_tol=open_tol,
                      tcp_rot_tol=tcp_rot_tol, drift_pos=drift_pos, drift_rot=drift_rot)
        self.state = 'FREE'
        self.op = None
        self.bar = None
        self.abort = None
        self.req = None
        self.events = []
        self.cmd = float(finger_open)
        self.t_ev0 = self.t_low = self.t_rel0 = self.t_ramp = self.t_bad = self.t_settle0 = None
        self.q_prev = None
        self.n_ops = 0
        self.armed = False
        self.E0 = None
        self.drift_max = (0.0, 0.0)
        self.rate_pos = RateFromPos(span_s=0.1, gap_max_s=0.05)

    def _ev(self, t, **kw):
        self.events.append(dict(kw, t=float(t), op=self.op, state=self.state))

    def _abort(self, why, t, **kw):
        if self.abort is None:
            self.abort = dict(kw, why=str(why), t=float(t), state=self.state, op=self.op)
            self._ev(t, abort=why)
        return self.abort

    def arm(self, op, bar, t):
        """開始新 op；只准在 FREE；序號必須遞增。"""
        if self.armed and self.op == int(op) and self.bar == bar and self.abort is None:
            return True, None                                                   # 重送：冪等（任務端等確認前可重送），不重設計時
        if self.abort is not None or self.state != 'FREE':
            return False, f'不可 arm（state {self.state}，abort {self.abort}）'
        if bar not in BAR or not (self.op is None or int(op) > int(self.op)):
            return False, f'op／bar 不合法（{op}, {bar}）'
        self.op, self.bar = int(op), bar
        self.t_ev0 = self.t_low = self.t_rel0 = self.t_ramp = self.t_settle0 = None
        self.E0 = None
        self.n_ops += 1
        self.armed = True
        self._ev(t, armed=bar)
        return True, None

    def request(self, kind, t):
        """任務端請求；實際轉移在 step。重送不重設任何計時。"""
        self.req = str(kind)

    def finger_target(self):
        """閉爪斜坡由呼叫端 ramp；此處只給目標。閂鎖後凍結最後實際值（呼叫端 cmd）。"""
        if self.abort is not None:
            return self.cmd
        return self.c['finger_closed'] if self.state in ('CLOSING', 'GRASPED', 'MOVING') else self.c['finger_open']

    def step(self, t, rec, q_door, qf, tcp_err, wrist_n, hb_age, ramp_done, fcmd=None, tcp_rot=None, rel=None):
        """每物理步一次。回傳 self.abort（None＝無）。ramp_done＝閉爪命令已到全閉。"""
        c = self.c
        t = float(t)
        if fcmd is not None:
            self.cmd = float(fcmd)
        if self.abort is not None:
            return self.abort
        if hb_age is None or not math.isfinite(float(hb_age)) or float(hb_age) > c['hb_max_s']:
            if self.state != 'FREE' or self.armed:
                return self._abort('task_heartbeat_stale', t, age=hb_age)
        if not (isinstance(wrist_n, (int, float)) and math.isfinite(wrist_n)) or wrist_n >= c['wrist_abort_n']:
            return self._abort('wrist_force', t, n=wrist_n)
        rate = None
        if self.q_prev is not None and t > self.q_prev[0]:
            rate = (float(q_door) - self.q_prev[1]) / (t - self.q_prev[0])
        self.q_prev = (t, float(q_door))
        if rate is not None and abs(rate) > c['door_rate_abort']:
            return self._abort('door_rate', t, rate=rate)
        v_pos = self.rate_pos.feed(t, float(q_door))
        F = forces(rec)
        if F is None:
            self.t_bad = self.t_bad if self.t_bad is not None else t
            self.t_ev0 = self.t_rel0 = self.t_settle0 = None            # 缺證據不累計
            if self.state in ('GRASPED', 'MOVING'):
                self.t_low = self.t_low if self.t_low is not None else t
                if t - self.t_low > c['finger_loss_s'] + 1e-9:
                    return self._abort('finger_loss', t, why2='no_evidence')
            if t - self.t_bad > c['invalid_max_s'] + 1e-9:
                return self._abort('contact_record_invalid', t)
            return None
        self.t_bad = None
        cls = None if self.bar is None else BAR[self.bar]
        allow = (cls,) if (cls is not None and (self.state != 'FREE' or self.armed)) else ()
        for (lk, cl), f in F['other'].items():
            if f > c['contact_abort_n']:
                return self._abort('contact_not_designated', t, link=lk, cls=cl, n=f)
        for fg, d in F['fingers'].items():
            for cl, f in d.items():
                if cl not in allow and f > c['contact_abort_n']:
                    return self._abort('contact_not_designated', t, link=fg, cls=cl, n=f)
        fb = [F['fingers'][fg].get(cls, 0.0) if cls else 0.0 for fg in FINGERS]
        both = min(fb) >= c['contact_min_n']
        none = max(fb) < c['contact_abort_n']
        r, st = self.req, self.state
        if st == 'FREE':
            if r == 'close' and self.armed:                                     # 未武裝不准閉爪（rt_10：B 的 arm 遺失卻沿用 A 的計時）
                if min(qf) < c['finger_open'] - c['open_tol'] * 10:
                    return None
                if not (isinstance(tcp_err, (int, float)) and tcp_err <= c['tcp_tol']):
                    return None
                if not (isinstance(tcp_rot, (int, float)) and tcp_rot <= c['tcp_rot_tol']):
                    return None
                self.state = 'CLOSING'
                self.req = None
                self._ev(t, to='CLOSING', tcp_err=tcp_err)
        elif st == 'CLOSING':
            if ramp_done and self.t_ramp is None:
                self.t_ramp = t
                self._ev(t, ramp_done=True)                                     # 抓持期限起算點（只記錄，不改判定）
            self.t_ev0 = (self.t_ev0 if self.t_ev0 is not None else t) if both else None
            if self.t_ev0 is not None and t - self.t_ev0 >= c['hold_s'] - 1e-9:
                if rel is None:
                    return self._abort('no_relative_pose', t)
                self.state = 'GRASPED'
                self.t_low = None
                self.E0 = np.array(rel, float)                                  # 固定基準，持握期間不重設
                self._ev(t, to='GRASPED', fb=fb, remaining_s=None if self.t_ramp is None else c['grasp_timeout_s'] - (t - self.t_ramp))
            elif self.t_ramp is not None and t - self.t_ramp >= c['grasp_timeout_s'] - 1e-9:
                return self._abort('grasp_no_contact', t, fb=fb)
        elif st in ('GRASPED', 'MOVING'):
            self.t_low = None if both else (self.t_low if self.t_low is not None else t)
            if self.t_low is not None and t - self.t_low > c['finger_loss_s'] + 1e-9:
                return self._abort('finger_loss', t, fb=fb)
            if rel is None:
                return self._abort('no_relative_pose', t)
            Tr = np.asarray(rel, float)
            dp = float(np.linalg.norm(Tr[:3, 3] - self.E0[:3, 3]))
            dr = float(np.arccos(np.clip((np.trace(self.E0[:3, :3].T @ Tr[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
            self.drift_max = (max(self.drift_max[0], dp), max(self.drift_max[1], dr))
            if dp > c['drift_pos'] or dr > c['drift_rot']:
                return self._abort('grasp_drift', t, dp=dp, dr=dr)
            if st == 'GRASPED' and r == 'move':
                self.state = 'MOVING'
                self.req = None
                self._ev(t, to='MOVING')
            elif st == 'MOVING' and r == 'open':
                ok = v_pos is not None and math.isfinite(v_pos) and abs(v_pos) <= c['settle_rate']
                self.t_settle0 = (self.t_settle0 if self.t_settle0 is not None else t) if ok else None
                if self.t_settle0 is not None and t - self.t_settle0 >= c['settle_hold_s'] - 1e-9:
                    self.state = 'OPENING'
                    self.req = None
                    self._ev(t, to='OPENING', q=q_door)
        elif st == 'OPENING':
            opn = min(qf) >= c['finger_open'] - c['open_tol'] * 10
            self.t_rel0 = (self.t_rel0 if self.t_rel0 is not None else t) if (opn and none) else None
            if self.t_rel0 is not None and t - self.t_rel0 >= c['release_hold_s'] - 1e-9:
                self.state = 'RELEASED'
                self._ev(t, to='RELEASED')
        elif st == 'RELEASED':
            if r == 'retreat':
                self.state = 'RETREAT'
                self.req = None
                self._ev(t, to='RETREAT')
        elif st == 'RETREAT':
            if r == 'done':
                self.state = 'FREE'
                self.armed = False
                self.req = None
                self._ev(t, to='FREE')
        return None

    def report(self):
        return {'version': VERSION, 'cfg': self.c, 'state': self.state, 'op': self.op, 'bar': self.bar, 'abort': self.abort, 'events': self.events[:200], 'n_ops': self.n_ops, 'drift_max': self.drift_max}
