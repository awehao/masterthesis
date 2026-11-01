#!/usr/bin/env python3
"""多任務 v3（跨桌高抓放）：轉回高度依放置桌產生（Codex 20261102_064116／070114；版本化子類，基線＝bottle_w1_vhigh_relwin_slipwarn_ctrl〔v2w〕，其餘全部相同）。
v2w 的轉回（緊湊 → 升高 → 外伸）目標＝抓取端的側持姿態 T_side（＝抬升終點，高度＝抓取點＋抬升量），所以只參照第一桌：
  第一桌 0.55 m（B1）時側持 TCP 0.70 m、瓶底約 0.584 m，低於第二桌面＋Z_PAD，外伸淨空閘不可通過；第一桌 0.65 m（B2）時側持 0.80 m，外伸未到位就下放（落點 11.3 mm）。
v3 只改兩處：
1 轉回目標 T_back：收近完成（stage 進 compact）時，以 T_side 的姿態與 xy、高度改為
    z_back＝第二桌面頂（放置先驗 place['top']）＋Z_PAD＋H_CLEAR − 持瓶／夾爪包絡在側持姿態下相對 TCP 的最低點，
  包絡＝瓶身取樣點（判定器 T_rel＝v2w 的當下量測持瓶位姿）＋夾爪包圍盒格點（同 compactcarry2 的 _pts_tcp 定義）。
  H_CLEAR＝0.029 m：持瓶包絡最低點在 TCP 下約 0.101 m，名目 0.60 m 時 z_back ≈ 0.751 m（與原側持高度差約 1 mm；初版 0.014 誤以 0.116 估，名目會降到 0.735）。
  升高（T_UP）與外伸（T_OUT）的 min-jerk 分段、外伸淨空閘、
  停止軌跡核對都改用 T_back；收近（側持 → 緊湊）起點仍是原 T_side（不跳）。放置桌高度是任務先驗，不用抓取端量測回填。
2 下放前 TCP 收斂：外伸完成（side）後，TCP 在底盤座標對 T_back 的位置誤差 ≤ XH_TCP_TOL（2 mm）才開始累計停車靜止（PARKED 前提）；
  未收斂前照 compactcarry3 的作法把 n_still_p 歸零，不縮短、不放寬；一直不收斂由既有 PLACE_APPROACH 期限受控停止。
3 轉回時長重核（bottle_xh_timing.py → results/bottle/xh_timing.json）：沿本批各案實際收近起點做 IK 延續，所需最大 收近 4.81 s、升高 2.00 s、外伸 4.82 s
  （原 3.8／2.0／3.8 s 是對參考姿態求的，不涵蓋各案側持位置；B2 外伸需要 4.82 s 卻只給 3.8 s）⇒ 取 ×1.05 進位：T_TO 5.1、T_UP 2.1、T_OUT 5.1 s。
  實作同 bottle_w1_vhigh_outslow_ctrl：只在 _cc_T／_advance_inner 執行期間把 compactcarry2.T_TO 與 compactcarry3.T_UP／T_OUT 換成上值，結束即還原。
停車位（_park_solve）不變：xy 用 CARRY 開始時的實際手臂姿態，與 T_back 同 xy。其餘（滑移警告、相對排程、閘、Shield、停止）同 v2w。
"""
import numpy as np

import bottle_s1_geom as G
import bottle_w1_compactcarry2_ctrl as C2m
import bottle_w1_compactcarry3_ctrl as C3m
import bottle_w1_vhigh_relwin_slipwarn_ctrl as SW

C3, C2 = SW.C3, SW.C2
H_CLEAR = 0.029
XH_TCP_TOL = 0.002
T_TO_XH, T_UP_XH, T_OUT_XH = 5.1, 2.1, 5.1


class W1PlaceController(SW.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_XH'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.xh = None
        self.xh_conv = None

    def _xh_back(self, q):
        f = self.cc
        T_side = np.array(f['T_side'], float)
        Tr = np.asarray(self.judge.T_rel, float)
        BP = G.bottle_points(n_ang=12, n_z=5)
        Pt = np.vstack([(Tr[:3, :3] @ BP.T).T + Tr[:3, 3], C2m.grip_grid_tcp(self.q_arm_ready)])
        zmin = float((T_side[:3, :3] @ Pt.T)[2].min())
        top2 = float(self.place['top'])
        z_back = top2 + C2m.Z_PAD + H_CLEAR - zmin
        T_back = T_side.copy()
        T_back[2, 3] = z_back
        self.xh = dict(t=float(self.t), z_side_lift=float(T_side[2, 3]), z_back=z_back, dz=z_back - float(T_side[2, 3]), zmin_rel=zmin, top2=top2, z_pad=C2m.Z_PAD, h_clear=H_CLEAR,
                       bottom_at_back=z_back + zmin, xy=[float(T_side[0, 3]), float(T_side[1, 3])])
        f['T_side_lift'], f['T_side'] = T_side, T_back
        self._ev('xh_back', z_side_lift=round(float(T_side[2, 3]), 4), z_back=round(z_back, 4), zmin_rel=round(zmin, 4), top2=top2)

    def _dur(self, fn, *a):
        old = (C2m.T_TO, C3m.T_UP, C3m.T_OUT)
        C2m.T_TO, C3m.T_UP, C3m.T_OUT = T_TO_XH, T_UP_XH, T_OUT_XH
        try:
            return fn(*a)
        finally:
            C2m.T_TO, C3m.T_UP, C3m.T_OUT = old

    def _cc_T(self, t):
        return self._dur(super()._cc_T, t)

    def _advance_inner(self, T_tcp, close_done, q):
        self._dur(super()._advance_inner, T_tcp, close_done, q)
        if self.judge.stop is not None or self.cc is None:
            return
        f = self.cc
        if self.xh is None and f['stage'] in ('compact', 'raise', 'raised', 'out', 'side'):
            self._xh_back(q)
        if f['stage'] == 'side' and self.pph == 'PLACE_APPROACH' and self.xh is not None:
            Tt = self.K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(q, float)[3:]], 'link_tcp')
            e = float(np.linalg.norm(Tt[:3, 3] - np.asarray(f['T_side'], float)[:3, 3]))
            if e > XH_TCP_TOL:
                self.n_still_p = 0                                          # 未收斂：不累計停車靜止（同外伸完成前的作法）
            elif self.xh_conv is None:
                self.xh_conv = dict(t=float(self.t), err=e)
                self._ev('xh_tcp_converged', err_mm=round(e * 1e3, 2))

    def report(self):
        r = super().report()
        r['xh'] = dict(back=self.xh, converged=self.xh_conv, tol=XH_TCP_TOL, durations=dict(T_TO=T_TO_XH, T_UP=T_UP_XH, T_OUT=T_OUT_XH),
                       note='轉回高度依第二桌面＋Z_PAD＋H_CLEAR 與持瓶／夾爪包絡最低點產生；下放前 TCP 收斂到 T_back')
        return r
