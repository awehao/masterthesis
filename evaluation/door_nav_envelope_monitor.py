#!/usr/bin/env python3
"""導航段整機包絡線上核對（Codex 20261029_145654：交棒及導航期間都核**實際構型**；20261029_151139 §4：失效封閉）。
r2（同檔，r1 未被任何實跑使用）：
  - 同時核**實測構型**與**導航期間保持的實際設定點**（兩者取大）；
  - 兩指各用實際讀值（finger1、finger2 分開，不假定 mimic 完全相等）；
  - 缺資料、維度錯、非有限、計算例外 ⇒ ρ＝inf（不准入／違反），理由寫明。
幾何同 door_nav_envelope_v1（URDF 碰撞幾何全部頂點、圓柱外接）；每連桿預取凸包頂點（水平半徑最大值必在凸包頂點）。
ρ＝max 手臂＋夾爪＋手指幾何的水平半徑（底盤中心＝base_footprint 原點）。
**界線**：逐週期檢查是越界偵測，不證明取樣之間與停止延遲內不越界；准入差額 5 mm 未經證明足夠。
EnvelopeMonitor：
  admit(q_meas, q_sp, fingers)    交權前准入：ρ ≤ r_env − admit_slack；
  step(q_meas, q_sp, fingers, t)  導航期間逐週期：ρ > r_env ⇒ 鎖存違反（first 記錄 t、ρ、理由）；不自行解除。
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'door_nav_envelope_monitor/2'
R_ENV = 0.39
ADMIT_SLACK = 0.005
FINGER_NAMES = ('uflite_finger1', 'uflite_finger2')


def hull_backend():
    from scipy.spatial import ConvexHull
    import door_nav_envelope_v1 as E
    import door_offline_check_v1 as C
    CL = E.full_clouds(open(C.URDF).read())
    arm = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
    Hv = {n: CL[n][ConvexHull(CL[n]).vertices] for n in arm}

    def rho(q_arm, fingers):
        """fingers＝(finger1, finger2) 各自的稜柱位移。"""
        qf, best = [0.0, 0.0, 0.0] + [float(v) for v in q_arm], (0.0, None)
        fmap = dict(zip(FINGER_NAMES, fingers))
        for n in arm:
            T = np.asarray(C.K.fk(qf, n))
            P = Hv[n] + (C.FINGER_AXIS[n] * fmap[n] if n in C.FINGER_AXIS else 0.0)
            W = (T[:3, :3] @ P.T).T + T[:3, 3]
            r = float(np.hypot(W[:, 0], W[:, 1]).max())
            if r > best[0]:
                best = (r, n)
        return best
    return rho


def _vec(x, n):
    """長度 n 的有限數列，否則回 (None, 理由)。"""
    if x is None:
        return None, '缺資料'
    try:
        v = [float(e) for e in x]
    except (TypeError, ValueError):
        return None, f'不是數列：{x!r}'
    if len(v) != n:
        return None, f'維度 {len(v)} ≠ {n}'
    if not all(math.isfinite(e) for e in v):
        return None, '含非有限值'
    return v, None


class EnvelopeMonitor:
    def __init__(self, rho_fn=None, r_env=R_ENV, admit_slack=ADMIT_SLACK):
        self.rho_fn = rho_fn if rho_fn is not None else hull_backend()
        self.r_env, self.admit_slack = float(r_env), float(admit_slack)
        self.violated = False
        self.first = None
        self.n = 0
        self.rho_max = 0.0
        self.n_fail_closed = 0

    def evaluate(self, q_meas, q_sp, fingers):
        """回傳 (ρ, 細節)；任何缺陷 ⇒ ρ＝inf。"""
        f, why = _vec(fingers, 2)
        if f is None:
            return math.inf, {'why': f'手指讀值{why}'}
        out = {}
        for tag, q in (('meas', q_meas), ('setpoint', q_sp)):
            v, why = _vec(q, 6)
            if v is None:
                return math.inf, {'why': f'{tag} 構型{why}'}
            try:
                r, n = self.rho_fn(v, f)
            except Exception as e:                       # 計算失敗不得准入
                return math.inf, {'why': f'{tag} 計算失敗：{e!r}'}
            if not (isinstance(r, float) and math.isfinite(r)):
                return math.inf, {'why': f'{tag} 計算結果非有限：{r!r}'}
            out[tag] = {'rho_m': r, 'link': n}
        k = max(out, key=lambda t: out[t]['rho_m'])
        return out[k]['rho_m'], {'why': None, 'worst': k, 'link': out[k]['link'], 'meas_m': out['meas']['rho_m'], 'setpoint_m': out['setpoint']['rho_m']}

    def admit(self, q_meas, q_sp, fingers):
        r, d = self.evaluate(q_meas, q_sp, fingers)
        if d['why'] is not None:
            self.n_fail_closed += 1
        lim = self.r_env - self.admit_slack
        return r <= lim, dict(d, rho_m=r, limit_m=lim)

    def step(self, q_meas, q_sp, fingers, t):
        r, d = self.evaluate(q_meas, q_sp, fingers)
        self.n += 1
        if d['why'] is not None:
            self.n_fail_closed += 1
        else:
            self.rho_max = max(self.rho_max, r)
        if r > self.r_env and not self.violated:
            self.violated = True
            self.first = dict(d, t=float(t), rho_m=r)
        return not self.violated, r

    def report(self):
        return {'version': VERSION, 'r_env_m': self.r_env, 'admit_slack_m': self.admit_slack, 'n': self.n, 'rho_max_m': self.rho_max,
                'n_fail_closed': self.n_fail_closed, 'violated': self.violated, 'first': self.first,
                'limits': '逐週期越界偵測；不證明取樣之間與停止延遲內不越界；准入差額 5 mm 未經證明足夠'}
