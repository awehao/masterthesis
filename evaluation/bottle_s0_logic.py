#!/usr/bin/env python3
"""瓶子 S0a 判定與停止（純邏輯、無 Isaac；規格：第七次進度報告素材/規格/瓶子S0_物理抓持設計.md）。
每物理步 update()：量測缺失以 None／NaN 表示，**一律當「無資料」，不當零力**。判定各項分開記錄，全部成立才算成功；任一停止條件成立即閂鎖第一阻點。
門檻皆為候選（設計 §4），事前指定、不事後調整：
  抓持：雙指各 ≥ 0.5 N 逐物理步連續 ≥ 1.0 s；期限 5 s 自**閉爪請求**起算，不因重送刷新；資料失效期間連續計數歸零。
  離桌：瓶位姿新鮮、瓶底最低點 − 桌面 ≥ 40 mm、瓶↔桌接觸資料有效且法向力 ≤ 0.02 N，三者同時連續 ≥ 1 s。判據矛盾記未通過。
  持瓶：抓持成立時固定瓶對 TCP 基準（不重設）；抬離後 3 s 內相對位移 ≤ 5 mm、轉角 ≤ 3°。
  未推瓶：閉爪請求前瓶位移 ≤ 5 mm、傾角 ≤ 3°。
  禁止接觸：力 ≥ 0.05 N 連續 ≥ 3 物理步 或 穿透 ≥ 1 mm ⇒ 停止；未達門檻的短脈衝照錄（只稱未觸發判定，不稱未碰撞）。
  失抓（證據成立後）：任一指 < 0.2 N 連續 0.2 s（較門段兩指判準保守）。資料失效：接觸或瓶位姿過期 > 0.1 s。
  抬離期限：抬升開始後 4 s 內離桌判據未成立 ⇒ 停止（候選；瓶子卡住或判據矛盾時記未通過，不調門檻）。
  必要接觸對（required）：逐對追蹤最後有效時刻；缺鍵、None、NaN 都算無效，過期 > stale_s ⇒ 停止，且不得判 no_forbidden 成立（Codex 20261031_185332 (1)）。
  閉爪請求前手指碰瓶：手指對瓶力視為禁止接觸對 finger_bottle_preclose（同 0.05 N／3 步），CLOSE 後才是指定接觸（同回覆 (2)）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

REG = dict(contact_n=0.5, evidence_s=1.0, deadline_s=5.0, lift_m=0.040, lift_hold_s=1.0, table_force_tol_n=0.02, pose_fresh_s=0.02,
           hold_s=3.0, slip_m=0.005, slip_deg=3.0, push_m=0.005, push_deg=3.0, forbid_n=0.05, forbid_steps=3, forbid_pen_m=0.001,
           loss_n=0.2, loss_s=0.2, stale_s=0.1, dt=0.01, lift_deadline_s=4.0)


def finite(x):
    return x is not None and all(math.isfinite(float(v)) for v in np.ravel(x))


def rel_pose(Ta, Tb):
    return np.linalg.inv(Ta) @ Tb


def pose_delta(T0, T1):
    """(位移 m, 轉角 deg)。"""
    R = T0[:3, :3].T @ T1[:3, :3]
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), math.degrees(math.acos(c))


@dataclass
class Meas:
    """一個物理步的量測。缺資料用 None／NaN；t_* 為該資料的量測時刻。"""
    t: float
    f_fingers: tuple | None = None            # (f1, f2) 每指對瓶力 N
    t_contact: float | None = None
    T_bottle: np.ndarray | None = None        # 瓶座標（底心、z 軸＝瓶軸）世界 4×4
    t_bottle: float | None = None
    f_table: float | None = None              # 瓶↔桌法向力 N（有效資料才給）
    forbidden: dict = field(default_factory=dict)   # {對名: 力 N}；NaN＝該對無資料
    penetration: float | None = None          # 禁止對最大穿透 m
    T_tcp: np.ndarray | None = None


class S0Judge:
    def __init__(self, table_top_z, bottle_radius, reg=None, required=()):
        self.r = dict(REG, **(reg or {}))
        self.required = tuple(required)
        self.t_valid = {k: None for k in self.required}
        self.t_first = None
        self.top_z, self.br = float(table_top_z), float(bottle_radius)
        self.t_close = None
        self.T_b0 = None                      # 初始瓶位姿（推瓶基準）
        self.n_ev = 0
        self.t_grasp = None
        self.T_rel = None                     # 抓持成立時固定的瓶對 TCP
        self.n_lift = 0
        self.t_lift_ok = None
        self.t_lift_start = None
        self.n_loss = [0, 0]
        self.forbid_cnt = {}
        self.pulses = {}                      # 對名 → 峰值、總步數（未觸發也記）
        self.pen_max = 0.0
        self.stop = None                      # 第一阻點 dict
        self.res = {'grasp_evidence': None, 'lifted': None, 'hold_ok': None, 'no_push': None, 'no_forbidden': True}
        self.slip_max = (0.0, 0.0)
        self.log = []

    # ---- 事件（由控制器呼叫；重送不刷新期限）
    def on_close_request(self, t, m: Meas):
        if self.t_close is None:
            self.t_close = float(t)
            ok, why = self._push_ok(m)
            self.res['no_push'] = ok
            if not ok:
                self._stop(t, why)

    def on_lift_start(self, t):
        if self.t_lift_start is None:
            self.t_lift_start = float(t)

    def _stop(self, t, why, **kw):
        if self.stop is None:
            self.stop = dict(t=float(t), why=why, **kw)
            self.log.append(dict(self.stop))

    def _fresh(self, t, t_data, lim):
        return t_data is not None and finite(t_data) and 0.0 <= t - t_data <= lim

    def _push_ok(self, m):
        if not finite(m.T_bottle) or self.T_b0 is None:
            return False, 'push_check_no_data'
        d, a = pose_delta(self.T_b0, m.T_bottle)
        return (d <= self.r['push_m'] and a <= self.r['push_deg']), f'bottle_pushed d={d * 1e3:.2f} mm a={a:.2f} deg'

    def bottle_bottom_gap(self, T_b):
        """瓶底最低點相對桌面（傾斜時圓盤最低點）。"""
        ax = T_b[:3, 2]
        s = math.sqrt(max(1.0 - float(ax[2]) ** 2, 0.0))
        return float(T_b[2, 3]) - self.br * s - self.top_z

    def update(self, m: Meas):
        r, t = self.r, float(m.t)
        pose_ok = finite(m.T_bottle) and self._fresh(t, m.t_bottle, r['stale_s'])
        contact_ok = finite(m.f_fingers) and self._fresh(t, m.t_contact, r['stale_s'])
        if self.T_b0 is None and pose_ok:
            self.T_b0 = np.array(m.T_bottle, float)
        # 資料失效（瓶位姿、手指對瓶自第一步起都須有效；閉爪前手指力也是禁止接觸判據）
        if self.t_first is None:
            self.t_first = t
        # 有效時刻取**資料自身的量測時刻**（時間戳停住即會過期）
        if finite(m.t_bottle) and finite(m.T_bottle):
            self.t_valid['__pose'] = max(float(m.t_bottle), self.t_valid.get('__pose') or float('-inf'))
        if finite(m.t_contact) and finite(m.f_fingers):
            self.t_valid['__contact'] = max(float(m.t_contact), self.t_valid.get('__contact') or float('-inf'))
        for k, why in (('__pose', 'data_stale_bottle_pose'), ('__contact', 'data_stale_contact')):
            t0_ = self.t_valid.get(k)
            if t - (self.t_first if t0_ is None else t0_) > r['stale_s']:
                self._stop(t, why)
        # 必要接觸對：缺鍵／None／NaN 都算無效
        for k in self.required:
            v = (m.forbidden or {}).get(k)
            if v is not None and math.isfinite(float(v)):
                self.t_valid[k] = t
            t0_ = self.t_valid.get(k)
            if t - (self.t_first if t0_ is None else t0_) > r['stale_s']:
                self.res['no_forbidden'] = None if self.res['no_forbidden'] is True else self.res['no_forbidden']   # 資料不足：不得判成立
                self._stop(t, f'data_stale_forbidden:{k}')
        # 閉爪請求前手指碰瓶＝禁止接觸
        if self.t_close is None and contact_ok:
            fm = max(float(x) for x in m.f_fingers)
            p = self.pulses.setdefault('finger_bottle_preclose', dict(peak=0.0, steps=0, n_nodata=0))
            p['peak'] = max(p['peak'], fm)
            if fm > 0.0:
                p['steps'] += 1
            self.forbid_cnt['finger_bottle_preclose'] = self.forbid_cnt.get('finger_bottle_preclose', 0) + 1 if fm >= r['forbid_n'] else 0
            if self.forbid_cnt['finger_bottle_preclose'] >= r['forbid_steps']:
                self.res['no_forbidden'] = False
                self._stop(t, 'forbidden_contact:finger_bottle_preclose', force=fm)
        # 禁止接觸（含脈衝紀錄；NaN＝無資料，不當零也不當碰撞，另記）
        for k, f in (m.forbidden or {}).items():
            p = self.pulses.setdefault(k, dict(peak=0.0, steps=0, n_nodata=0))
            if f is None or not math.isfinite(float(f)):
                p['n_nodata'] += 1
                self.forbid_cnt[k] = 0
                continue
            f = abs(float(f))
            p['peak'] = max(p['peak'], f)
            if f > 0.0:
                p['steps'] += 1
            self.forbid_cnt[k] = self.forbid_cnt.get(k, 0) + 1 if f >= r['forbid_n'] else 0
            if self.forbid_cnt[k] >= r['forbid_steps']:
                self.res['no_forbidden'] = False
                self._stop(t, f'forbidden_contact:{k}', force=f)
        if m.penetration is not None and math.isfinite(float(m.penetration)):
            self.pen_max = max(self.pen_max, float(m.penetration))
            if float(m.penetration) >= r['forbid_pen_m']:
                self.res['no_forbidden'] = False
                self._stop(t, 'forbidden_penetration', pen=float(m.penetration))
        # 抓持證據（閉爪請求後）
        if self.t_close is not None and self.t_grasp is None:
            good = contact_ok and pose_ok and min(m.f_fingers) >= r['contact_n']
            self.n_ev = self.n_ev + 1 if good else 0
            if self.n_ev * r['dt'] >= r['evidence_s'] - 1e-9:
                self.t_grasp = t
                self.res['grasp_evidence'] = True
                if finite(m.T_tcp):
                    self.T_rel = rel_pose(m.T_tcp, m.T_bottle)
            elif t - self.t_close > r['deadline_s']:
                self.res['grasp_evidence'] = False
                self._stop(t, 'grasp_evidence_deadline')
        # 失抓（證據成立後）
        if self.t_grasp is not None and contact_ok:
            for i in (0, 1):
                self.n_loss[i] = self.n_loss[i] + 1 if m.f_fingers[i] < r['loss_n'] else 0
                if self.n_loss[i] * r['dt'] >= r['loss_s'] - 1e-9:
                    self._stop(t, f'grasp_loss_finger{i + 1}')
        # 滑移（抓持基準固定）
        if self.T_rel is not None and pose_ok and finite(m.T_tcp):
            d, a = pose_delta(self.T_rel, rel_pose(m.T_tcp, m.T_bottle))
            self.slip_max = (max(self.slip_max[0], d), max(self.slip_max[1], a))
            if self.t_lift_start is not None and (d > r['slip_m'] or a > r['slip_deg']):
                self.res['hold_ok'] = False
                self._stop(t, f'slip d={d * 1e3:.2f} mm a={a:.2f} deg')
        # 離桌：三條件同時連續（缺資料不累計）
        if self.t_lift_start is not None:
            tbl_ok = m.f_table is not None and math.isfinite(float(m.f_table)) and abs(float(m.f_table)) <= r['table_force_tol_n']
            geo_ok = pose_ok and self._fresh(t, m.t_bottle, r['pose_fresh_s']) and self.bottle_bottom_gap(m.T_bottle) >= r['lift_m']
            self.n_lift = self.n_lift + 1 if (tbl_ok and geo_ok) else 0
            if self.t_lift_ok is None and t - self.t_lift_start > r['lift_deadline_s']:
                self.res['lifted'] = False
                self._stop(t, 'lift_deadline')
            if self.t_lift_ok is None and self.n_lift * r['dt'] >= r['lift_hold_s'] - 1e-9:
                self.t_lift_ok = t
                self.res['lifted'] = True
            if self.t_lift_ok is not None and self.res['hold_ok'] is None and t - self.t_lift_ok >= r['hold_s'] and self.stop is None:
                self.res['hold_ok'] = True
        return self.stop

    def success(self):
        return self.stop is None and all(self.res[k] is True for k in ('grasp_evidence', 'lifted', 'hold_ok', 'no_push', 'no_forbidden'))

    def report(self):
        return dict(result=dict(self.res), success=self.success(), stop=self.stop, t_close=self.t_close, t_grasp=self.t_grasp, t_lift_start=self.t_lift_start,
                    t_lift_ok=self.t_lift_ok, slip_max_mm_deg=[self.slip_max[0] * 1e3, self.slip_max[1]], pen_max_mm=self.pen_max * 1e3,
                    forbidden_pulses=self.pulses, reg=self.r)
