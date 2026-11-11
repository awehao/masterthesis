#!/usr/bin/env python3
"""偏差補償速度一致性原型（版本化子類，基線＝bottle_w1_lateslow_ctrl；Codex 20261111 第 11 輪核可離線＋一次假世界短跑，**未經 Isaac**）。
問題（lateslow_fake_late_swdiag_r1）：TPAS 的手臂相對參考以「底盤相對計畫位姿的橫向／偏航偏差 D」做補償，但預覽窗與切換檢查都把 D 當成當拍常數。
底盤還在橫向修正時（vy 0.039 m/s），常數 D 的模型把底盤橫移直接算成 TCP 世界速度 ⇒ 切換檢查的速度差 0.037 超門檻而拒絕；
同一個常數 D 也讓預覽窗裡的手臂參考沒有預期到下一拍 D 會變。
本版：D 隨時間前推 D(Δ)＝(dy＋ḋy·Δ, dθ＋ḋθ·Δ)，同一模型用在三處——
  1 實際送出的手臂相對參考 _rel_target（當拍與預覽每一步）；
  2 MPC 時域預覽（同上，_T_seq 經 _rel_target）；
  3 切換檢查（Tw1 用 D(dt)）。
ḋy＝vx·sin dθ＋vy·cos dθ（底盤命令在計畫座標的橫向分量）；ḋθ＝ωz−計畫偏航率（v_沿路徑·dθ_plan/dσ）。命令取 u_last（與切換檢查原本用的同一命令）。
有界：前推時間 ≤ D_HORIZON；|ḋy|、|ḋθ| 超過適用上限 ⇒ 該拍退回常數 D（不夾限後使用）並計數。門檻 SW_*、DEV_* 不變。"""
import math

import numpy as np

import bottle_s0_ctrl as C0
import bottle_w1_lateslow_ctrl as LS
import bottle_w1_tpas_ctrl as TS

D_HORIZON = 1.2                                                  # 前推時間上限（s；MPC 預覽 N·DT＝1.0 s＋在途延遲）
DY_RATE_MAX, DTH_RATE_MAX = 0.10, 0.30                           # ḋy、ḋθ 適用上限（m/s、rad/s）：高於此視為模型不適用，該拍退回常數 D


def d_rate(u_body, dth, yaw_rate_plan):
    ub = np.asarray(u_body, float)
    dy_d = float(ub[0]) * math.sin(dth) + float(ub[1]) * math.cos(dth)
    return dy_d, float(ub[2]) - float(yaw_rate_plan)


def d_at(dy, dth, dy_d, dth_d, delta):
    """D(Δ)：線性前推，Δ 夾在 [0, D_HORIZON]。與切換檢查對底盤的前推（照 u_last 等速）同一假設——不在過零處截斷
    （r1 版把過零後的偏差定為 0：橫偏 −1 mm、修正 0.04 m/s 時預測速度由 0.04 突降為 0，反而製造不連續）。回 (4×4, dy, dθ)。"""
    dl = min(max(float(delta), 0.0), D_HORIZON)
    y_, th_ = float(dy) + float(dy_d) * dl, float(dth) + float(dth_d) * dl
    return TS._tb(np.r_[0.0, y_, th_]), y_, th_


def switch_check(K, qp_base, u_base_body, P, tau_eff, rate, T_pre, D, rates, dt=0.02):
    """同 bottle_w1_tpas_ctrl.switch_check，但前推側的偏差用 D(dt)。位置／姿態差、分類規則、門檻相同。"""
    r0 = TS.switch_check(K, qp_base, u_base_body, P, tau_eff, rate, T_pre, dt=dt, D=D)
    qb = np.asarray(qp_base, float)[:3]
    dy, dth = float(D[1, 3]), float(math.atan2(D[1, 0], D[0, 0]))
    D1, _, _ = d_at(dy, dth, rates[0], rates[1], dt)
    Tw0 = TS._tb(qb) @ np.linalg.inv(D) @ TS.rel_ref(K, P, tau_eff)
    c_, s_ = math.cos(qb[2]), math.sin(qb[2])
    ub = np.asarray(u_base_body, float)
    qb1 = qb + dt * np.r_[c_ * ub[0] - s_ * ub[1], s_ * ub[0] + c_ * ub[1], ub[2]]
    Tw1 = TS._tb(qb1) @ np.linalg.inv(D1) @ TS.rel_ref(K, P, tau_eff + rate * dt)
    dv = float(np.linalg.norm(Tw1[:3, 3] - Tw0[:3, 3]) / dt)
    dp, dr = r0['dp'], r0['dr']
    if dp <= TS.SW_P and dr <= TS.SW_R and dv <= TS.SW_V:
        kind = 'direct'
    elif dp <= TS.SW_P_MAX and dr <= TS.SW_R_MAX and dv <= TS.SW_V:
        kind = 'rejoin'
    else:
        kind = 'reject'
    t_rj = max(TS.MJ_PEAK * dp / TS.V_RJ, TS.MJ_PEAK * dr / TS.W_RJ, TS.T_RJ_MIN) if kind == 'rejoin' else 0.0
    if kind == 'rejoin' and t_rj > TS.T_RJ_MAX:
        kind = 'reject'
    return dict(ok=kind != 'reject', kind=kind, dp=dp, dr=dr, dv=round(dv, 4), dv_const_D=r0['dv'], t_rj=round(t_rj, 3), T_from=r0['T_from'])


class W1PlaceController(LS.W1PlaceController):
    group = LS.W1PlaceController.group + '_DPRED'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dp_stat = dict(n=0, n_clip=0, dy_rate_max=0.0, dth_rate_max=0.0)

    def _d_rates(self, P):
        """當拍的 (ḋy, ḋθ)：u_last 與計畫偏航率；D 對應命令生效時刻（t_last＋在途延遲）。"""
        D = P.get('D')
        if D is None:
            return 0.0, 0.0
        dth = float(math.atan2(D[1, 0], D[0, 0]))
        ub = np.asarray(self.u_last, float)[:3]
        v_s = float(ub[0]) * math.cos(dth) - float(ub[1]) * math.sin(dth)
        wr = v_s * float(self.hd.dyaw(float(P.get('sig_raw', self.s_ref)))) if getattr(self, 'hd', None) is not None else 0.0
        dy_d, dth_d = d_rate(ub, dth, wr)
        st = self.dp_stat
        st['n'] += 1
        st['dy_rate_max'], st['dth_rate_max'] = max(st['dy_rate_max'], abs(dy_d)), max(st['dth_rate_max'], abs(dth_d))
        if abs(dy_d) > DY_RATE_MAX or abs(dth_d) > DTH_RATE_MAX:               # 超出模型適用域：不夾限後當作有效預測——該拍退回常數 D（既有行為；切換檢查因此照原模型判）
            st['n_clip'] += 1
            return 0.0, 0.0
        return float(dy_d), float(dth_d)

    def _rel_target(self, tau_t):
        P = self.tpa
        if P is None or P['switched'] or P.get('D') is None:
            return super()._rel_target(tau_t)
        D = P['D']
        r = P.get('Dd')
        if r is None or P.get('Dd_t') != P.get('t_last'):
            r = self._d_rates(P)
            P['Dd'], P['Dd_t'] = r, P.get('t_last')
        t_d = float(P.get('t_last', self.t)) + C0.D_STEPS * C0.PHYS          # D 所對應的時刻（命令生效）
        Dk, _, _ = d_at(float(D[1, 3]), float(math.atan2(D[1, 0], D[0, 0])), r[0], r[1], float(tau_t) - t_d)
        return np.linalg.inv(Dk) @ TS.rel_ref(self.K, P, max(self._tau_at(tau_t), 0.0))

    def _switch_check(self, qp_base, u_base, P, tau_eff, rate):
        r = self._d_rates(P)
        P['Dd'], P['Dd_t'] = r, P.get('t_last')
        return switch_check(self.K, qp_base, u_base, P, tau_eff, rate, P['T_pre'], P['D'], r)

    def report(self):
        r = super().report()
        r['dpred'] = dict(stat=self.dp_stat, horizon=D_HORIZON, rate_max=[DY_RATE_MAX, DTH_RATE_MAX],
                          note='偏差 D 以 u_last 與計畫偏航率線性前推；手臂相對參考、MPC 預覽、切換檢查共用；門檻不變')
        return r
