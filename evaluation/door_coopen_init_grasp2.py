#!/usr/bin/env python3
"""指定抓取初態的隔離驗證相位 r2（door_coopen_init_grasp.py 的版本化修正；Codex r16 §2、§3；r1 保留不改）。
r1 缺口（Codex 重現）：INIT_CHECK 只累計「位置誤差合格」的時間——沒有姿態准入、重複／倒退來源仍累積、資料空白不重置確認窗。
r2 的閉爪確認（尺度全部沿用既有值，不新增門檻）：
  - 本輪有效 valid＝T_tcp、H 有限，state_age 有限且 ≤ cfg.stale_s（0.2 s），m.t 有限且嚴格大於上一筆有效樣本時刻（重複／倒退 ⇒ 無效）。
  - 合格 good＝valid 且 位置誤差 < cfg.grip_tol_m（2 mm）且 姿態誤差 ≤ gcfg.rot_resume_rad（0.02 rad，登錄 approach_gate 既有值）。
  - 無效 ⇒ 確認窗清空、不推進；相鄰有效樣本時刻差 > cfg.stale_s ⇒ 確認窗清空（本筆可重新起算）。
  - good 連續 ≥ cfg.grip_settle_s（0.5 s）⇒ 凍結把手位姿、記閉爪起點、轉入 ENGAGE（之後同 K2）。
  - 每輪 act['init']＝{valid, why, err_p_m, err_r_rad, good, win_t0}（任務節點寫入 trace，供判定器核對確認窗，不只相信事件）。
其餘同 r1：(I1) UNFOLD 完成 ⇒ INIT_CHECK（不經 ALIGN／APPROACH）；(I3) 只在 INIT_CHECK 呼叫 monitor 期間把 door_k3a_policy.ACTIVE_MONITOR
暫時擴為「原集合＋INIT_CHECK」，try/finally 還原（Codex r16 §3 接受）。監看先於確認；不放寬門檻、不固定附著、不偽造接觸證據。
需要策略已有 gcfg（gated2 的 gate_init 先呼叫）；缺 ⇒ 第一次進入 INIT_CHECK 時中止（init_rot_tol_missing）。
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

import door_k3a_policy as KAP

PHASE = 'INIT_CHECK'


@contextlib.contextmanager
def _monitored():
    orig = KAP.ACTIVE_MONITOR
    KAP.ACTIVE_MONITOR = tuple(orig) + (PHASE,)
    try:
        yield
    finally:
        KAP.ACTIVE_MONITOR = orig


def _fin(*xs):
    return all(x is not None and np.isfinite(np.asarray(x, float)).all() for x in xs)


def init_grasp2(Base):
    class InitGrasp2(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if PHASE not in self.c.timeouts:
                raise ValueError('init_grasp2：cfg.timeouts 缺 INIT_CHECK')
            self.init_win = {'t0': None, 'last_t': None}
            self.init_stats = {'n_cycles': 0, 'n_invalid': 0, 'n_gap_reset': 0, 'n_window_reset': 0, 'n_rot_block': 0,
                               't_enter': None, 't_engage': None, 'err_enter_m': None, 'err_engage_m': None, 'rot_engage_rad': None}

        def monitor(self, m):
            if self.phase != PHASE:
                return super().monitor(m)
            with _monitored():
                return super().monitor(m)

        def _base_act(self):
            return {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}

        def _errs(self, m, T):
            ep = float(np.linalg.norm(m.T_tcp[:3, 3] - T[:3, 3]))
            R = T[:3, :3].T @ m.T_tcp[:3, :3]
            er = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
            return ep, er

        def _reset(self, key):
            if self.init_win['t0'] is not None:
                self.init_stats['n_window_reset'] += 1
            self.init_win['t0'] = None
            self.settle_t0 = None
            if key:
                self.init_stats[key] += 1

        def step(self, m, dt):
            if self.t_phase is None:
                self.t_phase = m.t
            if self.phase == 'UNFOLD' and m.unfold_done and _fin(m.T_tcp, m.H):                  # (I1)
                self.monitor(m)
                if self.phase == 'ABORTED':
                    return super().step(m, dt)
                self._go(PHASE, m.t)
                e = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, 0.0)[:3, 3]))
                self.init_stats.update(t_enter=m.t, err_enter_m=e)
                self.events.append({'t': m.t, 'init_check': 'enter', 'err_m': e})
                self.settle_t0 = None
                act = self._base_act()
                act['target'], act['solver_start'] = self.target(m.H, 0.0), True
                act['init'] = {'valid': None, 'why': 'enter', 'err_p_m': e, 'err_r_rad': None, 'good': False, 'win_t0': None}
                return act
            if self.phase == PHASE:
                self.monitor(m)
                if self.phase == 'ABORTED':
                    return super().step(m, dt)
                c, w = self.c, self.init_win
                self.init_stats['n_cycles'] += 1
                act = self._base_act()
                rec = {'valid': False, 'why': None, 'err_p_m': None, 'err_r_rad': None, 'good': False, 'win_t0': None}
                gc = getattr(self, 'gcfg', None)
                if gc is None:
                    self._abort('init_rot_tol_missing', m.t)
                    return super().step(m, dt)
                valid = (_fin(m.T_tcp, m.H) and m.state_age is not None and math.isfinite(m.state_age) and m.state_age <= c.stale_s
                         and m.t is not None and math.isfinite(m.t) and (w['last_t'] is None or m.t > w['last_t']))
                if not valid:
                    rec['why'] = 'invalid'
                    self._reset('n_invalid')
                    if _fin(m.H):
                        act['target'], act['solver_start'] = self.target(m.H, 0.0), True
                    act['init'] = rec
                    return act
                if w['last_t'] is not None and m.t - w['last_t'] > c.stale_s:
                    self._reset('n_gap_reset')
                    rec['why'] = 'gap_reset'
                w['last_t'] = m.t
                T = self.target(m.H, 0.0)
                act['target'], act['solver_start'] = T, True
                ep, er = self._errs(m, T)
                rec.update(valid=True, err_p_m=ep, err_r_rad=er)
                good = ep < c.grip_tol_m and er <= gc.rot_resume_rad
                if ep < c.grip_tol_m and not er <= gc.rot_resume_rad:
                    self.init_stats['n_rot_block'] += 1
                if good:
                    w['t0'] = m.t if w['t0'] is None else w['t0']
                    rec.update(good=True, win_t0=w['t0'])
                    if m.t - w['t0'] >= c.grip_settle_s:
                        self.H_frozen = np.asarray(m.H, float).copy()                         # 閉爪開始即凍結把手位姿（同 K2）
                        self.close_t0 = m.t
                        self.init_stats.update(t_engage=m.t, err_engage_m=ep, rot_engage_rad=er)
                        self.events.append({'t': m.t, 'init_check': 'pass', 'err_m': ep, 'rot_rad': er, 'win_t0': w['t0']})
                        self._go('ENGAGE', m.t, err=ep)
                else:
                    rec['why'] = rec['why'] or 'not_good'
                    self._reset(None)
                act['init'] = rec
                return act
            return super().step(m, dt)
    InitGrasp2.__name__ = f'InitGrasp2_{Base.__name__}'
    return InitGrasp2
