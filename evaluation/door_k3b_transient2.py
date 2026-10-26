#!/usr/bin/env python3
"""K3-B 卸載／釋放瞬態監看 r2（door_k3b_transient.py〔r1〕的版本化複本；Codex 20261026_034842 §2）。r2 差異：
  - 任何無效樣本（非有限、維度錯、tcp_fn 失敗）**中斷收斂證據**：低速計時歸零、速度窗清空（之後須重新成窗、重新計時）；
    自最後一筆有效樣本起連續無效 > invalid_max_s ⇒ FAILED data_invalid（於下一筆樣本〔有效或無效〕到達時判）。
  - feed_many(samples)：依序重放已收到的樣本（供延遲得知提交時補核提交至得知之間的證據）。
以下為 r1 說明：
K3-B 卸載／釋放瞬態監看（純邏輯；Codex 20261026_023226 §2、20261026_025647 §4）。
用途：卸載（mode 1）與張爪（mode 2）後，手臂在零速保持命令下可能回彈；此監看**取代該期間的手臂停止門檻**，但以有界規則約束，
與停止確認（HELD 舊門檻）明確區分。不放寬既有停止門檻：收斂後即回到舊門檻。

規則（相對起點＝提交時刻前最後一筆有效量測）：
  - 位移界：TCP 位置偏移 ≤ disp_max_m、姿態偏移 ≤ rot_max_rad（＝既有夾持漂移中止界 drift_abort_m 5 mm、drift_rot_abort_rad 2°）；
  - 速度界：TCP 線速（win_s 0.1 s 窗）≤ v_max_m_s（＝策略登錄退離速度 0.02 m/s：瞬態不得快於有意的退離）；
  - 持續時間：起點後 t_max_s 內必須收斂（卸載：unload_max_s 4.0；釋放：ExecReleaseMonitor t_conf_max_s 1.5，皆既有登錄），逾時 ⇒ FAILED timeout；
  - 收斂：手臂最大關節速（arm_win_s 0.5 s 窗）≤ 舊 HELD 手臂門檻 0.001 rad/s，且 TCP 線速 ≤ 舊門檻同量級 0.001 m/s，連續 quiet_s 0.5 s ⇒ CONVERGED；
  - 失效：樣本非有限、步號或時間不遞增、缺樣本 > invalid_max_s ⇒ FAILED data_invalid；
  - 收斂後不再以瞬態規則放行（呼叫端回到舊門檻）。
tcp_fn(base3, arm6) → 4×4（世界座標 TCP 位姿）由呼叫端注入（求解端用自身 WholeBodyKinematics；離線用 door_k2_check._tcp）。
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'door_k3b_transient/2'
REG_TRANSIENT = {'disp_max_m': 0.005, 'rot_max_rad': math.radians(2.0), 'v_max_m_s': 0.02, 'win_s': 0.1, 'arm_win_s': 0.5,
                 'quiet_arm': 1e-3, 'quiet_tcp_m_s': 1e-3, 'quiet_s': 0.5, 'invalid_max_s': 0.2}
T_MAX_UNLOAD, T_MAX_RELEASE = 4.0, 1.5


def _fin(*xs):
    try:
        return all(isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


class TransientMonitor:
    def __init__(self, tcp_fn, t_max_s, reg=None, name=''):
        self.r = dict(REG_TRANSIENT, **(reg or {}))
        self.tcp_fn, self.t_max, self.name = tcp_fn, float(t_max_s), str(name)
        self.state = 'IDLE'
        self.why = None
        self.t0 = self.T0 = None
        self.hist = []                       # (t, p3, arm6)
        self.t_quiet0 = None
        self.t_conv = None
        self.last_t = None
        self.worst = {'disp_m': 0.0, 'rot_rad': 0.0, 'v_m_s': 0.0}

    def _tcp(self, base, arm):
        T = np.asarray(self.tcp_fn(list(base), list(arm)), float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError('tcp_fn 輸出無效')
        return T

    def start(self, t, base, arm):
        if self.state != 'IDLE':
            return False
        try:
            if not (_fin(t) and _fin(*base) and _fin(*arm)) or len(list(base)) != 3 or len(list(arm)) != 6:
                raise ValueError('start 輸入無效')
            self.T0 = self._tcp(base, arm)
        except (TypeError, ValueError):
            self.state, self.why = 'FAILED', 'start_invalid'
            return False
        self.t0 = self.last_t = float(t)
        self.hist = [(float(t), self.T0[:3, 3].copy(), [float(x) for x in arm])]
        self.state = 'ACTIVE'
        return True

    def _fail(self, why):
        self.state, self.why = 'FAILED', why
        return why

    def feed(self, t, base, arm):
        """回傳 None（正常）或失敗原因。CONVERGED／FAILED 後不再處理。"""
        if self.state != 'ACTIVE':
            return self.why if self.state == 'FAILED' else None
        r = self.r
        try:
            ok = _fin(t) and len(list(base)) == 3 and len(list(arm)) == 6 and _fin(*base) and _fin(*arm)
            T = self._tcp(base, arm) if ok else None
        except (TypeError, ValueError):
            ok, T = False, None
        if not ok:
            self.t_quiet0 = None                                                # r2：無效樣本中斷收斂證據
            self.hist = self.hist[-1:]                                          # 速度窗須重新成窗
            self.n_invalid = getattr(self, 'n_invalid', 0) + 1
            if _fin(t) and t - self.last_t > r['invalid_max_s'] + 1e-9:
                return self._fail('data_invalid')
            self._broken = True
            return None
        if not t > self.last_t:
            return self._fail('time_order')
        if t - self.last_t > r['invalid_max_s'] + 1e-9:
            return self._fail('data_invalid')
        if getattr(self, '_broken', False):                                     # r2：恢復有效後自此重新成窗、重新計時
            self._broken = False
            self.hist = []
            self.t_quiet0 = None
        self.last_t = float(t)
        p = T[:3, 3].copy()
        disp = float(np.linalg.norm(p - self.T0[:3, 3]))
        Rm = T[:3, :3] @ self.T0[:3, :3].T
        rot = math.acos(max(-1.0, min(1.0, (float(np.trace(Rm)) - 1.0) / 2.0)))
        self.worst['disp_m'], self.worst['rot_rad'] = max(self.worst['disp_m'], disp), max(self.worst['rot_rad'], rot)
        if disp > r['disp_max_m'] or rot > r['rot_max_rad']:
            return self._fail('displacement')
        self.hist.append((float(t), p, [float(x) for x in arm]))
        while len(self.hist) > 2 and t - self.hist[0][0] > r['arm_win_s'] + 0.05:
            self.hist.pop(0)

        def back(win):
            for h in reversed(self.hist[:-1]):
                if t - h[0] >= win - 1e-9:
                    return h
            return None
        a = back(r['win_s'])
        v = None if a is None else float(np.linalg.norm(p - a[1])) / (t - a[0])
        if v is not None:
            self.worst['v_m_s'] = max(self.worst['v_m_s'], v)
            if v > r['v_max_m_s']:
                return self._fail('tcp_speed')
        if t - self.t0 > self.t_max + 1e-9:
            return self._fail('timeout')
        aa = back(r['arm_win_s'])
        varm = None if aa is None else max(abs(x - y) for x, y in zip(arm, aa[2])) / (t - aa[0])
        if v is not None and varm is not None and v <= r['quiet_tcp_m_s'] and varm <= r['quiet_arm']:
            self.t_quiet0 = self.t_quiet0 if self.t_quiet0 is not None else float(t)
            if t - self.t_quiet0 >= r['quiet_s'] - 1e-9:
                self.state, self.t_conv = 'CONVERGED', float(t)
        else:
            self.t_quiet0 = None
        return None

    def feed_many(self, samples):
        """samples：[(t, base3, arm6)]，依序餵入；回傳最後的失敗原因或 None。"""
        w = None
        for t, b, a in samples:
            w = self.feed(t, b, a)
            if self.state != 'ACTIVE':
                break
        return w

    def report(self):
        return {'version': VERSION, 'name': self.name, 'state': self.state, 'why': self.why, 't0': self.t0, 't_conv': self.t_conv,
                't_max_s': self.t_max, 'worst': dict(self.worst), 'reg': dict(self.r), 'n_invalid': getattr(self, 'n_invalid', 0)}
