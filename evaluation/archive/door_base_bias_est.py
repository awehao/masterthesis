#!/usr/bin/env python3
"""底盤速度偏差 d̂_b 的**因果估計器**（純邏輯；Codex 20261027_010258 §2-3～§2-5）。求解節點只轉發訊息、在快照時取值。

模型（與 wgmpc_core_sp_db.plant_phys_step 同一式）：每物理步 j，p_{j+1} ＝ p_j ＋ dt_p·R(θ_j)·(u_j ＋ b)，θ 同式（R 第三列為恆等）。
區間：相鄰兩筆 /odom（50 Hz ⇒ 2 物理步）[t_a, t_b)；作用其間的命令是 sim_t ∈ [t_a, t_b) 的 /coman/applied_cmd 各步（**位姿 k→k＋1 配命令 k**；
  模擬端 meas_vb_body 是後向差分，不用）。區間觀測 b_obs 依上式精確反解：
  b_ω ＝ Δθ/(m·dt_p) − mean(u_ω)；θ_j ＝ θ_a ＋ dt_p·Σ_{i<j}(u_iω ＋ b_ω)；b_xy ＝ (Σ_j R_xy(θ_j))⁻¹·(Δp_xy/dt_p − Σ_j R_xy(θ_j)·u_j,xy)。
區間合格（否則該區間不用、計 why）：
  - 兩端位姿有限、t_b > t_a；m ＝ round((t_b − t_a)/dt_p) ∈ [1, M_MAX]，|t_b − t_a − m·dt_p| ≤ 1e-6；
  - [t_a, t_b) 內恰 m 筆命令、相鄰間距 dt_p、首筆時刻＝t_a（1e-6）；每筆 exec_mode＝0、api_applied、值有限；
  - |b_obs| 逐軸 ≤ CAP（超界即不採用，記 out_of_bounds）。
單調：位姿以時刻為鍵，t ≤ 已處理高水位 ⇒ 不重開舊區間；同時刻再到且值同 ⇒ dup 忽略、值不同 ⇒ conflict（該時刻作廢、不形成區間）。
  命令以步號為鍵，同步號再到值同 ⇒ dup、值不同 ⇒ 該步衝突（含它的區間不用）。
登錄常數（單一組，不掃描）：WIN_S 0.5 s 窗內區間觀測的**算術平均**；N_MIN 10 個區間（0.2 s）才有效；MAX_AGE_S 0.1 s（最新合格區間終點到快照）；
  CAP（0.005 m/s, 0.005 m/s, 0.010 rad/s）約為 g_1 所量殘差的 4–5 倍、亦涵蓋 B0 零命令殘差 2–3 mm/s。
啟用：set_enabled(flag, t)。只有啟用中才接受新區間，區間不跨啟用切換；**停用即清空窗**（再啟用從零累積，不沿用舊值）。estimate(t_now) 在停用、樣本不足、
  過期、平均超界時回傳 None（求解端即不加偏差 ⇒ 與原核心同一預測；停用時預測位移跳變 ≤ dt·CAP）。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

WIN_S = 0.5
N_MIN = 10
MAX_AGE_S = 0.1
CAP = (0.005, 0.005, 0.010)
M_MAX = 4
T_TOL = 1e-6
MODE_NORMAL = 0


def _fin(x, n):
    try:
        v = np.asarray(x, float).reshape(n)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v).all() else None


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def interval_bias(pa, pb, us, dtp):
    """位姿 pa→pb（x, y, θ）、其間各步本體命令 us（m×3）⇒ 依模型精確反解的 b（3,）。"""
    us = np.asarray(us, float).reshape(-1, 3)
    m = len(us)
    bw = _wrap(float(pb[2]) - float(pa[2])) / (m * dtp) - float(us[:, 2].mean())
    th = float(pa[2])
    S = np.zeros((2, 2))
    v = np.zeros(2)
    for j in range(m):
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        S += R
        v += R @ us[j, :2]
        th += dtp * (float(us[j, 2]) + bw)
    dp = np.array([float(pb[0]) - float(pa[0]), float(pb[1]) - float(pa[1])])
    bxy = np.linalg.solve(S, dp / dtp - v)
    return np.array([bxy[0], bxy[1], bw])


class BaseBiasEstimator:
    def __init__(self, dtp, win_s=WIN_S, n_min=N_MIN, max_age_s=MAX_AGE_S, cap=CAP, m_max=M_MAX, cmd_cap=512):
        self.dtp = float(dtp)
        self.win_s, self.n_min, self.max_age_s = float(win_s), int(n_min), float(max_age_s)
        self.cap = np.asarray(cap, float).reshape(3)
        self.m_max, self.cmd_cap = int(m_max), int(cmd_cap)
        self.enabled = False
        self.win = deque()                       # (t_b, b_obs)
        self.cmds = {}                           # step → (t, u3, mode, api)；None ⇒ 衝突
        self.pose_prev = None                    # (t, p3)
        self.high_water = None
        self.n_used = 0
        self.n_dup = 0
        self.n_conflict = 0
        self.rejects = {}
        self.n_clear = 0

    def _rej(self, why):
        k = why.split(':')[0]
        self.rejects[k] = self.rejects.get(k, 0) + 1
        return {'used': False, 'why': why}

    def set_enabled(self, flag, t=None):
        flag = bool(flag)
        if self.enabled and not flag:
            self.win.clear()
            self.n_clear += 1
        if self.enabled != flag:
            self.pose_prev = None                # 區間不跨啟用切換：啟用後第一筆位姿只設錨點
        self.enabled = flag

    def on_cmd(self, step, t, u3, mode, api):
        try:
            st = int(step)
            if st != step or isinstance(step, bool):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        v = (float(t), None if _fin(u3, 3) is None else tuple(_fin(u3, 3).tolist()), mode, api)
        if st in self.cmds:
            old = self.cmds[st]
            if old is None:
                return False
            if old == v:
                self.n_dup += 1
                return False
            self.cmds[st] = None
            self.n_conflict += 1
            return False
        self.cmds[st] = v
        if len(self.cmds) > self.cmd_cap:
            for k in sorted(self.cmds)[:len(self.cmds) - self.cmd_cap]:
                del self.cmds[k]
        return True

    def _cmds_in(self, ta, tb):
        out = []
        for st in sorted(self.cmds):
            v = self.cmds[st]
            if v is None:
                continue
            if ta - T_TOL <= v[0] < tb - T_TOL:
                out.append((st, v))
        return out

    def on_pose(self, t, p3):
        """一筆 /odom（stamp＝模擬時刻）。回傳本次區間紀錄。"""
        p = _fin(p3, 3)
        try:
            t = float(t)
        except (TypeError, ValueError):
            t = float('nan')
        if p is None or not math.isfinite(t):
            self.pose_prev = None
            return self._rej('pose_invalid')
        if self.high_water is not None and t <= self.high_water + T_TOL:
            if self.pose_prev is not None and abs(t - self.pose_prev[0]) <= T_TOL:
                if np.array_equal(p, self.pose_prev[1]):
                    self.n_dup += 1
                    return {'used': False, 'why': 'dup'}
                self.n_conflict += 1
                self.pose_prev = None
                return self._rej('pose_conflict')
            return self._rej('pose_not_advancing')
        prev = self.pose_prev
        self.pose_prev = (t, p)
        self.high_water = t
        if not self.enabled:
            return {'used': False, 'why': 'disabled'}
        if prev is None:
            return {'used': False, 'why': 'anchor_set'}
        ta, pa = prev
        dt = t - ta
        m = int(round(dt / self.dtp))
        if m < 1 or m > self.m_max or abs(dt - m * self.dtp) > T_TOL:
            return self._rej(f'interval_len:{dt:.6f}')
        cs = self._cmds_in(ta, t)                    # 衝突步不列入 ⇒ 筆數不足而拒
        if len(cs) != m:
            return self._rej(f'cmd_count:{len(cs)}/{m}')
        steps = [st for st, _ in cs]
        if steps != list(range(steps[0], steps[0] + m)):
            return self._rej('cmd_steps')
        ts = [v[0] for _, v in cs]
        if abs(ts[0] - ta) > T_TOL or any(abs((b - a) - self.dtp) > T_TOL for a, b in zip(ts, ts[1:])):
            return self._rej('cmd_spacing')
        for _, v in cs:
            if v[1] is None:
                return self._rej('cmd_nonfinite')
            if v[2] != MODE_NORMAL or not v[3]:
                return self._rej(f'exec_mode:{v[2]}/{v[3]}')
        b = interval_bias(pa, p, [v[1] for _, v in cs], self.dtp)
        if not np.isfinite(b).all() or (np.abs(b) > self.cap).any():
            return self._rej('out_of_bounds')
        self.win.append((t, b))
        while self.win and self.win[0][0] < t - self.win_s + T_TOL:
            self.win.popleft()
        self.n_used += 1
        for st in [k for k in self.cmds if k < steps[0]]:
            del self.cmds[st]
        return {'used': True, 'why': None, 't0': ta, 't1': t, 'm': m, 'b_obs': b.tolist()}

    def estimate(self, t_now):
        """回傳 (b 或 None, why, 統計)。"""
        info = {'n': len(self.win), 'age_s': None}
        if not self.enabled:
            return None, 'disabled', info
        if len(self.win) < self.n_min:
            return None, 'too_few', info
        age = float(t_now) - self.win[-1][0]
        info['age_s'] = age
        if not math.isfinite(age) or age > self.max_age_s or age < -T_TOL:
            return None, 'stale', info
        b = np.mean([x[1] for x in self.win], axis=0)
        if (np.abs(b) > self.cap).any():
            return None, 'out_of_bounds', info
        return b, None, info
