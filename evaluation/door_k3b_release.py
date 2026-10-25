#!/usr/bin/env python3
"""K3-B 受控釋放的判準核心（純邏輯；規格 K3B_release_clear_spec_draft2.md；Codex 20261025_230812）。
執行端、任務端、判定器與離線／假世界測試共用同一份；本檔不發命令，只回傳狀態與失敗理由。

狀態（單向，失敗即閂鎖）：
  IDLE ─open()→ A_OPENING ─確認→ B_CONFIRMED ─退出近區→ C_CLEAR ─讀回成功→ C_RESTORED ─s≥0.095→ RETREAT_DONE
       ─收臂前置核對→ STOW_READY
  任一步不符 ⇒ FAILED（why）；之後一律走既有失效路徑（§5：K2 中止閂鎖，不續退、不閉爪、不宣稱安全停止）。

每步輸入 sample（呼叫端組成；缺欄或非有限 ⇒ 該步無效）：
  t, step（int）, finger_q（2，m）, pair_force（2，N：該窗口手指—橫桿最大值，ambiguous 計入）,
  contacts_age（s）, contacts_invalid（int）, door_q（rad）, door_qd（rad/s）, E（4×4：工具在 H(q_rel) 把手座標）, s_cmd（m）,
  pair_d（2，m）, pair_d_pred（2，m）, nonfinger_d, nonfinger_d_pred, self_d, self_d_pred, base_d, base_d_pred（m）,
  base_dev_m, base_dev_yaw（相對錨點）, wrist_n, unexpected_n, effort_readback_ok（None／True／False）
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'door_k3b_release/1'

REG = {
    'finger_open_m': 0.0089, 'open_frac': 0.95, 'f_small_n': 0.1, 'confirm_hold_s': 0.5, 't_conf_max_s': 1.5,
    'invalid_max_s': 0.2, 'contacts_age_max_s': 0.1, 'f_design_max_n': 18.0,
    'near_s_m': 0.02, 'near_exit_max_s': 2.0, 'pair_exit_m': 0.005, 'restore_quiet_s': 0.5,
    'lower_deg': -1.0, 't_pred_s': 0.2,
    'x_max_m': 0.003, 'y_max_m': 0.001, 'rot_max_deg': 0.5,
    's_end_m': 0.10, 's_stow_m': 0.095, 'retreat_max_s': 15.0, 'stow_lo_deg': -3.0, 'stow_hi_deg': 2.5,
    'nonfinger_m': 0.005, 'self_m': 0.005, 'base_m': 0.020, 'base_dev_m': 0.020, 'base_dev_deg': 2.0,
    'wrist_max_n': 30.0, 'unexpected_max_n': 0.1,
}
A_H = np.array([0.0, -1.0, 0.0])                       # 接近軸（把手座標）；退離為 +s
# 單調門角帶（規格 §3；上界 Δ 度，s_meas 向下取格點）
BAND = [(0.0, 0.1), (0.0025, 0.3), (0.005, 0.4), (0.0075, 0.5), (0.03, 0.75), (0.0325, 1.0), (0.0375, 1.25), (0.04, 1.5),
        (0.0475, 2.0), (0.06, 3.0), (0.08, 4.0), (0.09, 5.0)]


def band_upper_deg(s):
    """s_meas（m）→ 上界 Δ（度）。s < 0（未退）以 s＝0 計；非有限 ⇒ None。"""
    if not _fin(s):
        return None
    s = max(0.0, float(s))
    up = BAND[0][1]
    for s0, u in BAND:
        if s + 1e-12 >= s0:
            up = u
    return up


def z_max_m(s):
    return 0.004 if s < 0.01 else 0.006 if s < 0.02 else 0.007


def _fin(*xs):
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    return True


def _vec(v, n):
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return a if len(a) == n and all(math.isfinite(x) for x in a) else None


def tool_offsets(E, E_rel, s_cmd):
    """實測工具 E 相對參考 T(s_cmd)＝Trans(s_cmd·a_H)·E_rel（把手座標）：回傳 (s_meas, dx, dy, dz, rot_deg)。"""
    E, E_rel = np.asarray(E, float), np.asarray(E_rel, float)
    ref_p = E_rel[:3, 3] + s_cmd * A_H
    d = E[:3, 3] - ref_p
    R = E[:3, :3] @ E_rel[:3, :3].T
    rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    s_meas = float((E[:3, 3] - E_rel[:3, 3]) @ A_H)
    return s_meas, float(d[0]), float(d[1]), float(d[2]), rot


class ReleaseMonitor:
    def __init__(self, reg=None):
        self.r = dict(REG, **(reg or {}))
        self.state = 'IDLE'
        self.why = None
        self.t = {}
        self.q_rel = self.E_rel = None
        self.last_step = None           # 最後看到的步號（有效或無效都算；缺步以此核對）
        self.ok_run_t0 = None            # 確認條件連續成立起點
        self.invalid_t0 = None           # 連續無效起點
        self.quiet_t0 = None             # 退出後配對 ≤ 0.1 N 連續起點
        self.restore_requested = None
        self.events = []
        self.n = {'steps': 0, 'invalid': 0}
        self.worst = {'band_margin_deg': math.inf, 'pair_d_min_m': math.inf, 'nonfinger_min_m': math.inf}

    # ---------------------------------------------------------------- 入口
    def open(self, t, step, q_rel, E_rel):
        """t_open：呼叫端在同一物理步切換監看 profile 與手指目標後呼叫。"""
        if self.state != 'IDLE':
            return self._fail('open_twice', t, step)
        if not _fin(t, q_rel) or type(step) is not int or _vec(np.asarray(E_rel, float).reshape(-1), 16) is None:
            return self._fail('open_args_invalid', t, step)
        self.q_rel, self.E_rel = float(q_rel), np.asarray(E_rel, float)
        self.t['open'], self.state, self.last_step = float(t), 'A_OPENING', step
        self.events.append({'t': t, 'step': step, 'to': 'A_OPENING', 'q_rel': q_rel})
        return self.state

    def _fail(self, why, t, step):
        if self.state != 'FAILED':
            self.state, self.why = 'FAILED', why
            self.t['failed'] = t
            self.events.append({'t': t, 'step': step, 'to': 'FAILED', 'why': why})
        return self.state

    def _go(self, st, t, step, **kw):
        self.state = st
        self.t[st] = t
        self.events.append(dict({'t': t, 'step': step, 'to': st}, **kw))

    # ---------------------------------------------------------------- 有效性
    def _valid(self, x):
        if not isinstance(x, dict) or type(x.get('step')) is not int or not _fin(x.get('t')):
            return False
        if self.last_step is not None and x['step'] != self.last_step + 1 and x['step'] != self.last_step:
            return False                                                      # 缺步
        if _vec(x.get('finger_q'), 2) is None or _vec(x.get('pair_force'), 2) is None:
            return False
        if not _fin(x.get('contacts_age')) or x['contacts_age'] > self.r['contacts_age_max_s']:
            return False
        if type(x.get('contacts_invalid')) is not int or x['contacts_invalid'] > 0:
            return False
        keys = ('door_q', 'door_qd', 's_cmd', 'nonfinger_d', 'nonfinger_d_pred', 'self_d', 'self_d_pred', 'base_d', 'base_d_pred',
                'base_dev_m', 'base_dev_yaw', 'wrist_n', 'unexpected_n')
        if not all(_fin(x.get(k)) for k in keys):
            return False
        if _vec(x.get('pair_d'), 2) is None or _vec(x.get('pair_d_pred'), 2) is None:
            return False
        try:
            E = np.asarray(x.get('E'), float)
        except (TypeError, ValueError):
            return False
        return E.shape == (4, 4) and bool(np.all(np.isfinite(E)))

    # ---------------------------------------------------------------- 每步
    def step(self, x):
        if self.state in ('IDLE', 'FAILED', 'STOW_READY'):
            return self.state
        r = self.r
        t = x.get('t') if isinstance(x, dict) else None
        stp = x.get('step') if isinstance(x, dict) else None
        self.n['steps'] += 1
        ok_x = self._valid(x)
        if type(stp) is int:
            self.last_step = stp
        if not ok_x:
            self.n['invalid'] += 1
            self.ok_run_t0 = None                                              # 無效步不計入任何持續窗
            self.quiet_t0 = None
            if _fin(t):
                self.invalid_t0 = self.invalid_t0 if self.invalid_t0 is not None else t
                if t - self.invalid_t0 > r['invalid_max_s'] + 1e-9:
                    return self._fail('data_invalid', t, stp)
                return self._check_deadlines(t, stp)
            return self.state
        self.invalid_t0 = None
        # 一般保護（所有段都適用；底盤 20 mm、非手指與自碰 5 mm，含預測）
        if x['wrist_n'] > r['wrist_max_n']:
            return self._fail('wrist_force', t, stp)
        if x['unexpected_n'] > r['unexpected_max_n']:
            return self._fail('unexpected_contact', t, stp)
        if min(x['nonfinger_d'], x['nonfinger_d_pred']) < r['nonfinger_m']:
            return self._fail('nonfinger_clearance', t, stp)
        if min(x['self_d'], x['self_d_pred']) < r['self_m']:
            return self._fail('self_clearance', t, stp)
        if min(x['base_d'], x['base_d_pred']) < r['base_m']:
            return self._fail('base_clearance', t, stp)
        if x['base_dev_m'] > r['base_dev_m'] or abs(math.degrees(x['base_dev_yaw'])) > r['base_dev_deg']:
            return self._fail('base_anchor', t, stp)
        self.worst['nonfinger_min_m'] = min(self.worst['nonfinger_min_m'], x['nonfinger_d'])
        # 工具偏移與門角帶（含預測扣除）
        s_meas, dx, dy, dz, rot = tool_offsets(x['E'], self.E_rel, x['s_cmd'])
        if abs(dx) > r['x_max_m'] or abs(dy) > r['y_max_m'] or abs(dz) > z_max_m(max(s_meas, 0.0)) or rot > r['rot_max_deg']:
            return self._fail('tool_offset', t, stp)
        dq = math.degrees(x['door_q'] - self.q_rel)
        dq_pred = dq + math.degrees(max(x['door_qd'], 0.0)) * r['t_pred_s']
        up = band_upper_deg(s_meas)
        if dq < r['lower_deg'] or dq_pred > up:
            return self._fail('door_band', t, stp)
        self.worst['band_margin_deg'] = min(self.worst['band_margin_deg'], up - dq_pred)
        fq = x['finger_q']
        pf = x['pair_force']
        opened = min(fq) >= r['open_frac'] * r['finger_open_m']
        small = max(pf) <= r['f_small_n']
        # 段別語意
        if self.state == 'A_OPENING':
            if max(pf) > r['f_design_max_n']:
                return self._fail('pair_force_design_bound', t, stp)
            if opened and small:
                self.ok_run_t0 = self.ok_run_t0 if self.ok_run_t0 is not None else t
                if t - self.ok_run_t0 >= r['confirm_hold_s'] - 1e-9:
                    self._go('B_CONFIRMED', t, stp, q_conf=x['door_q'], qd_conf=x['door_qd'])
            else:
                self.ok_run_t0 = None
        elif self.state in ('B_CONFIRMED', 'C_CLEAR', 'C_RESTORED', 'RETREAT_DONE'):
            if not small:
                return self._fail('pair_force_after_confirm', t, stp)
            if not opened:
                return self._fail('finger_closed_after_confirm', t, stp)
            pd = min(min(x['pair_d']), min(x['pair_d_pred']))
            self.worst['pair_d_min_m'] = min(self.worst['pair_d_min_m'], min(x['pair_d']))
            if self.state == 'B_CONFIRMED':
                if pd < 0.0:
                    return self._fail('pair_penetration_near', t, stp)
                if s_meas >= r['near_s_m'] and pd >= r['pair_exit_m']:
                    self._go('C_CLEAR', t, stp, s_meas=s_meas, pair_d=pd)
                    self.quiet_t0 = t
            else:
                if pd < r['pair_exit_m']:
                    return self._fail('pair_clearance_after_exit', t, stp)
                if self.quiet_t0 is None:
                    self.quiet_t0 = t
                if self.state == 'C_RESTORED' and s_meas >= r['s_stow_m'] - 1e-9:
                    self._go('RETREAT_DONE', t, stp, s_meas=s_meas)
                if self.state == 'RETREAT_DONE':
                    if r['stow_lo_deg'] <= dq and dq_pred <= r['stow_hi_deg']:
                        self._go('STOW_READY', t, stp, dq=dq, dq_pred=dq_pred)
                    else:
                        return self._fail('stow_precondition_door', t, stp)
        if x.get('effort_readback_ok') is False:
            return self._fail('effort_readback', t, stp)
        if x.get('effort_readback_ok') is True and self.state in ('A_OPENING', 'B_CONFIRMED'):
            return self._fail('effort_restored_early', t, stp)                 # 確認解除且退出近區前不得恢復 5 N
        if x.get('effort_readback_ok') is True and self.state == 'C_CLEAR':
            if self.restore_requested is None:
                return self._fail('effort_restored_without_request', t, stp)
            self._go('C_RESTORED', t, stp)
        return self._check_deadlines(t, stp)

    def restore_allowed(self, t):
        """5 N 恢復的准許（呼叫端提出恢復請求前核）：C_CLEAR 且配對 ≤ 0.1 N 已連續 ≥ 0.5 s。"""
        ok = self.state == 'C_CLEAR' and self.quiet_t0 is not None and t - self.quiet_t0 >= self.r['restore_quiet_s'] - 1e-9
        if ok and self.restore_requested is None:
            self.restore_requested = t
            self.events.append({'t': t, 'to': 'restore_requested'})
        return ok

    def _check_deadlines(self, t, stp):
        r = self.r
        if self.state == 'A_OPENING' and t - self.t['open'] > r['t_conf_max_s'] + 1e-9:
            return self._fail('confirm_timeout', t, stp)
        if self.state == 'B_CONFIRMED' and t - self.t['B_CONFIRMED'] > r['near_exit_max_s'] + 1e-9:
            return self._fail('near_exit_timeout', t, stp)
        if self.state in ('B_CONFIRMED', 'C_CLEAR', 'C_RESTORED') and t - self.t['open'] > r['retreat_max_s'] + r['t_conf_max_s'] + 1e-9:
            return self._fail('retreat_timeout', t, stp)
        return self.state

    def report(self):
        return {'version': VERSION, 'state': self.state, 'why': self.why, 't': dict(self.t), 'q_rel': self.q_rel,
                'events': self.events[:200], 'n': dict(self.n), 'worst': dict(self.worst), 'reg': dict(self.r),
                'restore_requested': self.restore_requested}
