#!/usr/bin/env python3
"""多任務 v2：量測目標相對排程（Codex 20261102_050244；版本化子類，基線＝bottle_w1_segtarget_vhigh_ctrl〔v1 凍結〕，其餘全部相同）。
v1 的抓取窗與接近排程是世界絕對 x（名目瓶 x＝0）。v2 在**首次合格量測**採用目標時（_seg_freeze 前）一次性平移整套排程：
  Δx＝量測瓶 x − 名目瓶 x（X_NOM＝0，任務先驗，資產 center_xy[0]）；
  抓取窗 win、x_ws／x_we（窗寬不變：仍是相對瓶心 [−0.35, +0.15]）、預抓起點 x_pre、漸變起點 x_blend、慢速點 x_slow、V_HIGH 回備妥觸發 x_dep 全部 +Δx；
  放行管的角落排除（S0b X_CORNER）改為相對量（X_CORNER＋Δx）。其後的全窗分支（window_branch）用平移後的窗。
窗口在首次採用時凍結：之後的限速 xy 修正不再移動排程。若採用當下底盤已越過平移後的漸變起點 ⇒ segtarget_window_shift_late 受控拒收（不追趕、不壓縮時長）。
不重選備妥姿態（沿用名目分支）、不移車道、不改抓取姿態；V_HIGH 已開始的過渡不重置。Δx＝0 時與 v1 完全相同（零位移退化）。
"""
import math

import bottle_s0b_ctrl as B
import bottle_w1_segtarget_vhigh_ctrl as VH

C3, C2 = VH.C3, VH.C2
X_NOM = 0.0
SHIFT_KEYS = ('x_ws', 'x_we', 'x_pre', 'x_blend', 'x_slow', 'x_dep')


class W1PlaceController(VH.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN'

    def __init__(self, *a, x_nom=X_NOM, **kw):
        super().__init__(*a, **kw)
        self.x_nom, self.relwin = float(x_nom), None

    def _seg_freeze(self, xy, zs, seq):
        if self.relwin is None:
            dx = float(xy[0]) - self.x_nom
            p = getattr(self, 'q_prev2', None)
            x_now = float(p[0]) if p is not None else -math.inf
            old = {k: float(getattr(self, k)) for k in SHIFT_KEYS}
            new = {k: v + dx for k, v in old.items()}
            self.relwin = dict(dx=dx, t=float(self.t), x_now=x_now, old=dict(old, win=list(self.win)), new=dict(new, win=[self.win[0] + dx, self.win[1] + dx]), seq=seq)
            if not self.blending and x_now >= new['x_blend']:
                self.judge._stop(self.t, 'segtarget_window_shift_late', x_now=x_now, x_blend_new=new['x_blend'], dx=dx)
                self.seg_z = None
                self.task.T_grasp = self.task.T_grasp * float('nan')
                return False
            for k, v in new.items():
                setattr(self, k, v)
            self.win = (self.win[0] + dx, self.win[1] + dx)
            self._ev('relwin_shift', dx=round(dx, 5), x_now=round(x_now, 4), win=[round(v, 4) for v in self.win], x_blend=round(self.x_blend, 4), x_dep=round(self.x_dep, 4))
        return super()._seg_freeze(xy, zs, seq)

    def _in_tube(self, q):
        dx = 0.0 if self.relwin is None else self.relwin['dx']
        if q[1] < 0.0 and q[2] < 0.0 and q[0] > B.X_CORNER + dx:          # 角落排除改相對瓶心（同 S0b 的判式，只平移 x）
            return False
        return self.x_ws <= q[0] <= self.x_we and B.Y_TUBE[0] <= q[1] <= B.Y_TUBE[1] and abs(q[2]) <= B.YAW_TUBE

    def report(self):
        r = super().report()
        r['relwin'] = dict(self.relwin or {}, x_nom=self.x_nom, keys=list(SHIFT_KEYS) + ['win', 'X_CORNER'],
                           note='首次合格量測時以量測瓶 x 平移整套接近排程（窗寬不變、凍結）；不重選備妥、不移車道')
        return r
