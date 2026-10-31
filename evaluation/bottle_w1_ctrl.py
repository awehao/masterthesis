#!/usr/bin/env python3
"""W1：底盤相機遠距定位＋手腕相機近距修正，同一個 GMPC（Codex 20261101_072908、20261101_074314）。
W1PlaceController(V2PlaceController)：V2 的底盤凍結值是初始抓取目標；接近時收手腕估計（wrist_intake），形成穩定量測後以限速連續修正 task.T_grasp（只修 xy，z 仍用桌面先驗），
未來參考依同一限速逐拍預測（_T_seq），不是只改當拍目標。不新增控制器、不交棒。
手腕接收：有效、擷取戳遞增、資料年齡 ≤ 0.3 s、估計距離 0.10–0.30 m（bottle_w1c_2 近距有效格 0.110–0.289 m 向外取到公分）、軸傾角 ≤ 3°、底盤凍結已存在、鎖定前。
穩定量測：最近 1.0 s 內 ≥ 3 筆、xy 兩兩差 ≤ 3 mm ⇒ 量測目標＝中位數。修正：施加目標以 5 mm/s 三維平移向量範數限速朝量測目標移動（本版 z 不動）；
相對底盤凍結值的總修正量（向量範數）> 5 mm ⇒ wrist_inconsistent。鎖定：INSERT 開始後 0.9 s，之後不再更新。
閉爪閘：必須已形成至少一組穩定量測，且施加目標已到達鎖定目標；未到達就讓 TCP 停在抓取位姿等待（暫時讓 INSERT→CLOSE 的位姿容差無法成立，不改 INSERT 時間剖面）。
期限：更新期間（第一筆手腕估計之後、鎖定前）最近有效手腕影格的擷取時間距當下 > 0.5 s ⇒ wrist_stale；鎖定時從未形成穩定量測 ⇒ wrist_no_fix；
鎖定後最近有效影格到閉爪請求 > 2.5 s ⇒ wrist_fix_expired（靜態瓶任務的候選期限，不代表能偵測瓶子被移動）。任何情況都不退回底盤目標閉爪，也不接回真值。
"""
import dataclasses
import math

import numpy as np

import bottle_ref as BR
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_v2_ctrl as V2

W1REG = dict(dist=(0.10, 0.30), tilt_deg=3.0, max_age=0.3, win_s=1.0, n_need=3, spread=0.003, rate=0.005, max_corr=0.005, lock_after_insert_s=0.9,
             stale_s=0.5, expire_after_lock_s=2.5)


class W1PlaceController(V2.V2PlaceController):
    def __init__(self, *a, w1=None, **kw):
        super().__init__(*a, **kw)
        self.w1 = dict(W1REG, **(w1 or {}))
        self.wbuf, self.wlog, self.wrej, self.w_last_cap, self.w_last_valid = [], [], {}, None, None
        self.w_goal, self.w_cur, self.w_t, self.w_locked, self.w_groups, self.w_hist = None, None, None, False, [], []
        self._tol_ins0 = tuple(self.task.tol_ins)

    def _wrej(self, why):
        self.wrej[why] = self.wrej.get(why, 0) + 1

    def wrist_intake(self, t_cap, t_recv, e):
        """e：dict(valid, cx, cy, ax(3), dist)；時間都是任務時間。"""
        R = self.w1
        self.wlog.append((float(t_cap), float(t_recv), bool(e.get('valid')), e.get('cx'), e.get('cy'), e.get('dist')))
        if self.judge.stop is not None:
            return
        if not (np.isfinite(t_cap) and np.isfinite(t_recv)):
            return self._wrej('time_nonfinite')
        if t_cap > t_recv + 1e-9:
            return self._wrej('time_future')
        if self.w_last_cap is not None and t_cap <= self.w_last_cap + 1e-9:
            return self._wrej('time_not_increasing')
        self.w_last_cap = float(t_cap)
        if t_recv - t_cap > R['max_age']:
            return self._wrej('stale')
        if not e.get('valid') or not all(np.isfinite([e['cx'], e['cy'], e['dist'], *e['ax']])):
            return self._wrej('invalid')
        ax = np.asarray(e['ax'], float)
        if float(np.linalg.norm(ax)) < 1e-6:
            return self._wrej('axis_zero')
        ax = ax / np.linalg.norm(ax)
        if not (R['dist'][0] <= e['dist'] <= R['dist'][1]):
            return self._wrej('distance')
        if math.degrees(math.acos(min(1.0, abs(float(ax[2]))))) > R['tilt_deg']:
            return self._wrej('tilt')
        if self.vfix is None:
            return self._wrej('no_base_fix')
        if self.w_locked:
            return self._wrej('locked')
        self.w_last_valid = float(t_cap)
        xy = np.array([e['cx'], e['cy']], float)
        self.wbuf.append((float(t_cap), xy))
        W = [b for b in self.wbuf if t_cap - R['win_s'] - 1e-9 <= b[0] <= t_cap + 1e-9]
        if len(W) >= R['n_need']:
            P = np.array([b[1] for b in W])
            spread = max(float(np.linalg.norm(P[i] - P[j])) for i in range(len(P)) for j in range(i + 1, len(P)))
            if spread <= R['spread']:
                g = np.median(P, axis=0)
                corr = float(np.linalg.norm(g - self.vfix['xy']))
                if corr > R['max_corr']:
                    self.judge._stop(self.t, 'wrist_inconsistent', corr=corr, goal=g.tolist(), base=self.vfix['xy'].tolist())
                    return
                if self.w_goal is None:
                    self._ev('wrist_fix', xy=[round(float(v), 5) for v in g], corr=corr, n=len(P), spread=spread)
                self.w_goal = g
                self.w_groups.append(dict(t=self.t, xy=g.tolist(), n=len(P), spread=spread, corr=corr))

    # ---- 限速施加
    def _Tg(self, xy):
        return C0.grasp_target(float(xy[0]), float(xy[1]), self.top_z + 0.10)

    def _w_pred(self, tau):
        """施加目標在 tau 秒後的預測（同一限速、朝目前量測目標）。"""
        if self.w_goal is None or self.w_cur is None:
            return self.w_cur
        d = self.w_goal - self.w_cur
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return self.w_cur.copy()
        return self.w_cur + d / n * min(n, self.w1['rate'] * max(tau, 0.0))

    def _w_step(self):
        if self.vfix is None:
            return
        if self.w_cur is None:
            self.w_cur, self.w_t = np.array(self.vfix['xy'], float), self.t
        if self.w_goal is not None:
            self.w_cur = self._w_pred(self.t - self.w_t)
            self.task.T_grasp = self._Tg(self.w_cur)
            self.w_hist.append((self.t, float(self.w_cur[0]), float(self.w_cur[1])))
        self.w_t = self.t

    def _w_reached(self):
        return self.w_goal is not None and self.w_cur is not None and float(np.linalg.norm(self.w_goal - self.w_cur)) < 1e-6

    def _w1_checks(self):
        R, st = self.w1, self.st
        if self.judge.stop is not None or self.vfix is None:
            return
        if not self.w_locked and st.phase == 'INSERT' and self.t - st.t_phase >= R['lock_after_insert_s'] - 1e-9:
            self.w_locked = True
            self._ev('wrist_lock', goal=None if self.w_goal is None else [round(float(v), 5) for v in self.w_goal])
            if self.w_goal is None:
                self.judge._stop(self.t, 'wrist_no_fix', n_buf=len(self.wbuf), rej=dict(self.wrej))
                return
        if not self.w_locked and self.w_last_valid is not None and self.t - self.w_last_valid > R['stale_s']:
            self.judge._stop(self.t, 'wrist_stale', age=self.t - self.w_last_valid)
            return
        if self.w_locked and self.t_close_req is None and self.t - self.w_last_valid > R['expire_after_lock_s']:
            self.judge._stop(self.t, 'wrist_fix_expired', age=self.t - self.w_last_valid)

    def _advance_inner(self, T_tcp, close_done, q):
        self._w_step()
        self._w1_checks()
        ok_close = self.w_locked and self._w_reached()
        self.task.tol_ins = self._tol_ins0 if ok_close else (-1.0, -1.0)   # 閉爪閘：未鎖定或未到達鎖定目標 ⇒ INSERT→CLOSE 不成立，TCP 停在抓取位姿
        super()._advance_inner(T_tcp, close_done, q)

    def _T_seq(self, q, t_eff):
        T = super()._T_seq(q, t_eff)
        if self.w_goal is not None and not self._w_reached() and not self._rel_task() and self.st.phase in ('CRUISE', 'PREGRASP', 'INSERT'):
            plan = self._plan_at(t_eff)
            T = np.stack([BR.target_at(dataclasses.replace(self.task, T_grasp=self._Tg(self._w_pred(t_eff + k * B.DT - self.w_t))), self.st, plan, t_eff + k * B.DT) for k in range(B.N + 1)])
            if self.no_tcp_preview:
                T = np.stack([T[0]] * (B.N + 1))
        return T

    def report(self):
        r = super().report()
        r['w1'] = dict(goal=None if self.w_goal is None else self.w_goal.tolist(), cur=None if self.w_cur is None else self.w_cur.tolist(), locked=self.w_locked,
                       corr_mm=None if self.w_goal is None else float(np.linalg.norm(self.w_goal - self.vfix['xy'])) * 1e3, n_groups=len(self.w_groups), groups=self.w_groups[-5:],
                       n_intake=len(self.wlog), rejected=self.wrej, last_valid=self.w_last_valid, reg=self.w1, hist_n=len(self.w_hist),
                       note='底盤相機遠距凍結＋手腕近距修正（只修 xy，限速 5 mm/s，鎖定於 INSERT＋0.9 s）；5 mm 淨空與禁止接觸用模擬真值（安全網）')
        return r
