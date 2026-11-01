#!/usr/bin/env python3
"""懸持中受控側向擾動的短程觀測實驗（Codex 20261102_035741 第 4、5 項；版本化子類，基線＝bottle_w1_vhigh_hold_ctrl〔V_HIGH 移動抓取＋靜態懸持〕）。
只加一項：13 s 靜態懸持的中央 4 s（懸持開始後 4.5–8.5 s）讓持瓶目標沿「懸持開始時實測的手指閉合軸水平投影」往返：
  0→+A（0.5 s）、+A↔−A 六個半週期（各 0.5 s，三週期）、+A→0（0.5 s），各段端點速度與加速度為零的 min-jerk；A＝20 mm；名目峰速 0.15 m/s、峰加速度 0.924 m/s²。
  閉合軸＝手臂實測構型下 TCP 的 y 軸（底盤座標）取水平分量後正規化（側持時約為世界 x，不是世界 y）。
  **由底盤執行**：擾動窗內底盤參考（_ref_window）沿閉合軸（換到世界）加上位移剖面，並給對應參考速度（本體座標）；手臂維持 T_carry_rel（底盤座標），所以 TCP 隨底盤移動。
  （先試過由手臂執行：持瓶段 TCP 追蹤權重低〔W_FAR〕，假世界只實現 14.5／40 mm 峰峰值、追蹤誤差 15 mm；底盤位置追蹤強，且較接近實際運送時的擾動來源。）
  不改 T_carry_rel、滑移基準與監看。
懸持 13 s 結束 ⇒ 主動受控停止 hold_exp_done（主觀察完成，不接運送放置）。CBF、Shield、滑移判準、淨空與禁止接觸監看全部保留；安全層若改寫擾動，報「未實現預定激勵」。
不加夾持力、不改 Kp、不改摩擦。
"""
import math

import numpy as np

import bottle_ref as BR
from ammr_wholebody_mpc.se2 import from_xytheta
import bottle_s0b_ctrl as B
import bottle_w1_vhigh_hold_ctrl as HC

C3, C2 = HC.C3, HC.C2
A_SHAKE, T_SEG, T_PRE, T_SHAKE = 0.020, 0.5, 4.5, 4.0
SEGS = [(0.0, 1.0)] + [((1.0, -1.0) if k % 2 == 0 else (-1.0, 1.0)) for k in range(6)] + [(1.0, 0.0)]   # 以 A 為單位的端點


def disp(tau):
    """擾動位移（m）對擾動開始後時間 tau（s）；範圍外為 0。"""
    if tau <= 0.0 or tau >= T_SHAKE:
        return 0.0
    k = min(int(tau / T_SEG), len(SEGS) - 1)
    a, b = SEGS[k]
    s = BR.min_jerk(min(max((tau - k * T_SEG) / T_SEG, 0.0), 1.0))
    return A_SHAKE * (a + (b - a) * s)


class W1PlaceController(HC.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_HOLD_SHAKE'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.shake = dict(axis_base=None, t0=None, log=[], amp=A_SHAKE, t_seg=T_SEG, t_pre=T_PRE, t_shake=T_SHAKE)

    def _shake_T(self, tau_abs):
        T = np.array(self.st.T_carry_rel, float).copy()
        sh = self.shake
        if sh['t0'] is not None and sh['axis_base'] is not None:
            T[:3, 3] = T[:3, 3] + disp(tau_abs - sh['t0']) * sh['axis_base']
        return T

    def _ref_window(self, xs_eff):
        sh = self.shake
        t_eff = float(self.t) + B.C.D_STEPS * B.C.PHYS
        if not (self.st.phase == 'CARRY' and self.pph is None and self.hold['state'] == 'HOLD' and sh['t0'] is not None and sh['t0'] - 0.5 <= t_eff <= sh['t0'] + T_SHAKE + 1.0):
            return super()._ref_window(xs_eff)
        c_, s_ = math.cos(self.th_ref), math.sin(self.th_ref)
        axb = sh['axis_base']
        axw = np.array([c_ * axb[0] - s_ * axb[1], s_ * axb[0] + c_ * axb[1]])
        X_ref, xi_ref = [], []
        for k in range(B.N + 1):
            tk = t_eff + k * B.DT - sh['t0']
            d = disp(tk)
            dv = (disp(tk + 0.005) - disp(tk - 0.005)) / 0.01
            X_ref.append(from_xytheta(xs_eff + d * axw[0], self.y_ref + d * axw[1], self.th_ref))
            xi_ref.append([dv * axb[0], dv * axb[1], 0.0])                                    # 本體座標速度（偏航固定）
        return np.array(X_ref), np.array(xi_ref)

    def _carry_pending(self, q):
        h, t, sh = self.hold, float(self.t), self.shake
        if h['state'] == 'HOLD' and sh['axis_base'] is None:
            Tb = self.K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(q, float)[3:]], 'link_tcp')        # 實測手臂構型下 TCP 在底盤座標
            ax = np.array([Tb[0, 1], Tb[1, 1], 0.0])
            n = float(np.linalg.norm(ax))
            if n < 0.3 and self.judge.stop is None:
                self.judge._stop(self.t, 'shake_axis_not_horizontal', norm=n)
                return True
            sh.update(axis_base=ax / n, t0=h['t_stable'] + T_PRE, axis_norm_before=n)
            self._ev('shake_planned', axis=[round(float(v), 4) for v in ax / n], t0=round(sh['t0'], 3))
        if h['state'] == 'HOLD' and sh['t0'] is not None and sh['t0'] <= t <= sh['t0'] + T_SHAKE + 0.5 and self.n_step % 2 == 0:
            Tt = self.K.fk(np.asarray(q, float), 'link_tcp')                                   # 世界座標 TCP（含底盤）
            sh['log'].append((round(t, 3), round(disp(t - sh['t0']) * 1e3, 3), [round(float(v), 5) for v in Tt[:3, 3]], [round(float(v), 5) for v in np.asarray(q, float)[:3]]))
        if h['state'] == 'HOLD' and t - h['t_stable'] >= self.t_hold - 1e-9:
            h.update(state='END', t_end=t)
            self._ev('hold_end', hold_s=round(t - h['t_stable'], 3))
            if self.judge.stop is None:
                self.judge._stop(self.t, 'hold_exp_done', hold_s=round(t - h['t_stable'], 3))   # 主觀察完成：受控停止，不接運送
            return True
        if h['state'] == 'END':
            return True
        return super()._carry_pending(q)

    def report(self):
        r = super().report()
        sh = self.shake
        r['shake_exp'] = dict(amp_m=sh['amp'], t_seg=sh['t_seg'], t_pre=sh['t_pre'], t_shake=sh['t_shake'], axis_base=None if sh['axis_base'] is None else sh['axis_base'].tolist(),
                              t0=sh['t0'], n_log=len(sh['log']), log=sh['log'], segs=SEGS,
                              note='懸持中央 4 s 沿實測閉合軸水平投影 ±20 mm 三週期 min-jerk（底盤參考執行、手臂維持 T_carry_rel）；懸持結束主動停止 hold_exp_done')
        return r
