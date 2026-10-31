#!/usr/bin/env python3
"""V2：視覺提供抓取目標、真值輔助安全監看的模擬驗證（設計：瓶子第二桌放置_最小設計.md「V2 最小設計」；Codex 20261101_052133）。
V2PlaceController(PlaceController)：接近時收 V1 估計，取得穩定的瓶位後凍結成世界座標目標，重算抓取／預抓／抬升位姿與手臂備妥分支，之後不再改，也不接回真值。
只適用靜態瓶。抓持成功仍由接觸證據驗收；5 mm 淨空拒絕線與禁止接觸用模擬真值的瓶位姿（安全網、評估），不是控制目標。
收估計（vision_intake）：有效、資料年齡 ≤ max_age、估計距離 0.5–2.0 m（估計值＋相機位姿算）、軸傾角 ≤ 3°、擷取戳不重複；
凍結：最近 1.0 s 內 ≥ 5 筆、xy 兩兩最大差 ≤ 3 mm ⇒ 取中位數 xy（z＝桌面，地圖先驗）。
凍結期限（draft-2）：開始朝瓶漸變（x ≥ x_blend）前；展臂在 x ≥ −1.55 開始時瓶多半還在 2 m 外（離線名目 28 筆有 26 筆距離 > 2.0 m），所以不要求展臂前凍結。
凍結只換抓取目標（T_grasp，預抓與抬升由它推出）；手臂備妥姿態沿用名目分支，避免展臂途中目標跳動；新目標仍以 window_branch 重算確認可行。
停止（受控，不退回真值）：漸變開始時仍未凍結 vision_no_fix；凍結後、PREGRASP 前有效估計偏離凍結值 > 5 mm vision_inconsistent；
重算後分支不成立 vision_target_unreachable；凍結到閉爪超過 15 s vision_fix_expired（只是本次靜態任務的使用期限，不保證瓶未移動）。
"""
import math

import numpy as np

import bottle_branch as BB
import bottle_place_ctrl as PC
import bottle_ref as BRF
import bottle_s0_ctrl as C0

V2REG = dict(dist=(0.5, 2.0), tilt_deg=3.0, max_age=0.3, win_s=1.0, n_need=5, spread=0.003, inconsistent=0.005, expire_s=15.0)


class V2PlaceController(PC.PlaceController):
    def __init__(self, *a, v2=None, top_z=0.60, win=(-0.35, 0.15), **kw):
        super().__init__(*a, **kw)
        self.v2, self.top_z, self.win = dict(V2REG, **(v2 or {})), float(top_z), tuple(win)
        self.vbuf, self.vfix, self.vlog, self.vrej, self.v_last = [], None, [], {}, None

    def _vrej(self, why):
        self.vrej[why] = self.vrej.get(why, 0) + 1

    def vision_intake(self, t_cap, t_recv, e):
        """e：dict(valid, cx, cy, ax(3), dist)；時間都是任務時間。"""
        R = self.v2
        self.vlog.append((float(t_cap), float(t_recv), bool(e.get('valid')), e.get('cx'), e.get('cy'), e.get('dist')))
        if self.judge.stop is not None:
            return
        if not (np.isfinite(t_cap) and np.isfinite(t_recv)):
            return self._vrej('time_nonfinite')
        if t_cap > t_recv + 1e-9:
            return self._vrej('time_future')
        if self.v_last is not None and t_cap <= self.v_last + 1e-9:          # 重複或倒序影格一律拒收
            return self._vrej('time_not_increasing')
        self.v_last = float(t_cap)
        if t_recv - t_cap > R['max_age']:
            return self._vrej('stale')
        if not e.get('valid') or not all(np.isfinite([e['cx'], e['cy'], e['dist'], *e['ax']])):
            return self._vrej('invalid')
        ax = np.asarray(e['ax'], float)
        if float(np.linalg.norm(ax)) < 1e-6:
            return self._vrej('axis_zero')
        ax = ax / np.linalg.norm(ax)
        if not (R['dist'][0] <= e['dist'] <= R['dist'][1]):
            return self._vrej('distance')
        if math.degrees(math.acos(min(1.0, abs(float(ax[2]))))) > R['tilt_deg']:
            return self._vrej('tilt')
        xy = np.array([e['cx'], e['cy']], float)
        if self.vfix is None:
            self.vbuf.append((float(t_cap), xy))
            W = [b for b in self.vbuf if t_cap - R['win_s'] - 1e-9 <= b[0] <= t_cap + 1e-9]   # 來源時間窗口兩端都限定
            if len(W) >= R['n_need']:
                P = np.array([b[1] for b in W[-R['n_need']:]])
                spread = max(float(np.linalg.norm(P[i] - P[j])) for i in range(len(P)) for j in range(i + 1, len(P)))
                if spread <= R['spread']:
                    self._freeze(np.median(P, axis=0), spread, len(P))
        elif self.st.phase == 'CRUISE' and not self.blending and self.t_close_req is None:
            d = float(np.linalg.norm(xy - self.vfix['xy']))
            if d > R['inconsistent']:
                self.judge._stop(self.t, 'vision_inconsistent', d=d, est=xy.tolist(), fix=self.vfix['xy'].tolist())

    def _freeze(self, xy, spread, n):
        T_g = C0.grasp_target(float(xy[0]), float(xy[1]), self.top_z + 0.10)
        T_p = T_g.copy()
        T_p[:3, 3] -= self.task.pre_m * T_g[:3, 2]
        br = BB.window_branch(T_p, T_g, BRF.lift_pose(T_g, self.task.lift_m), self.win[0], self.win[1])
        if br is None or not np.isfinite(br.get('margin', np.nan)) or br['margin'] <= 0.0:
            self.judge._stop(self.t, 'vision_target_unreachable', margin=None if br is None else br.get('margin'))
            return
        self.task.T_grasp = T_g                                      # 備妥姿態（q_arm_ready、T_ready_rel）沿用名目分支
        self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=spread, n=n, margin=float(br['margin']))
        self._ev('vision_freeze', xy=[round(float(v), 5) for v in xy], spread=spread, n=n, margin=float(br['margin']))

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None:
            if self.vfix is None and not self.blending and q[0] >= self.x_blend:
                self.judge._stop(self.t, 'vision_no_fix', n_buf=len(self.vbuf), rej=dict(self.vrej))
                return
            if self.vfix is not None and self.t_close_req is None and self.t - self.vfix['t'] > self.v2['expire_s']:
                self.judge._stop(self.t, 'vision_fix_expired', age=self.t - self.vfix['t'])
                return
        super()._advance_inner(T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['v2'] = dict(fix=None if self.vfix is None else dict(self.vfix, xy=self.vfix['xy'].tolist()), n_intake=len(self.vlog), rejected=self.vrej, reg=self.v2,
                       note='視覺提供抓取目標、真值輔助安全監看（5 mm 淨空、禁止接觸用模擬真值瓶位姿）；底盤定位為真值；第二桌放置點為已知位置')
        return r
