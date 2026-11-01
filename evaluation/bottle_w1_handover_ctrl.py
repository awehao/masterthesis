#!/usr/bin/env python3
"""W1 持瓶承接候選（版本化子類，2026-11-01；Codex 20261101_085238 (a)）：原 G（bottle_w1_ctrl.W1PlaceController）與 P 不動。
待驗證的主要機制：承接到 CARRY_NAV 時參考與執行狀態失配（bottle_w1_1：參考速度 0.10、實測 0.041 m/s，位置參考領先 3.8 mm），底盤在第一拍以加速度上限追趕。
本候選只改承接那一拍的參考錨定，對齊**命令生效時刻**（t＋D_STEPS·PHYS）：
  - 位姿：既有執行模型沿在途命令前推（_inflight），投影到新持瓶路徑得 σ_p。
  - 速度：生效時刻的底盤速度＝在途佇列最後一筆底盤命令（執行模型中底盤速度即命令），轉世界座標後投影到 σ_p 的路徑切線，負值取 0。
  - 參考：v_b ← 該投影速度；s_ref ← σ_p − v_b·D_STEPS·PHYS，使生效時刻的參考位置（s_ref＋v_b·D_STEPS·PHYS，同 _solve_place 的 xs_eff）正好在 σ_p，位置與速度用同一時間基準。
原承接的參考跳動（ref_jump）、速度跳動（v_ref_jump）與參考加速度剖面（ref_profile_check）全部用新錨定值重做，任一不過即同原處置受控停止；
不清空佇列、不事後濾命令；承接前與其他相位完全相同。無可調參數。
"""
import math

import numpy as np

import bottle_place_ctrl as PC
import bottle_s0_ctrl as C0
import bottle_w1_ctrl as W1


class W1PlaceController(W1.W1PlaceController):
    group = 'G_HO'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ho = None

    def _advance_inner(self, T_tcp, close_done, q):
        pre = self.pph
        snap = (float(self.s_ref), float(self.v_b), float(self.y_ref), float(self.th_ref))
        super()._advance_inner(T_tcp, close_done, q)
        if pre is None and self.pph == 'CARRY_NAV' and self.judge.stop is None:
            self._reanchor(q, snap)

    def _reanchor(self, q, snap):
        s0, vb0, y0, th0 = snap
        R, cp, tau = self.preg, self.cpath, C0.D_STEPS * C0.PHYS
        qp, _ = self._inflight(q)
        ub = np.asarray(self.pipe[-1][:3], float)
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        vw = np.array([c_ * ub[0] - s_ * ub[1], s_ * ub[0] + c_ * ub[1]])
        sp = float(cp.project(qp[:2], self.csig, back=0.0, ahead=0.3))          # 1 cm 格點投影
        tan = np.asarray(cp.tangent(sp)[0], float)
        sp += float((qp[:2] - np.asarray(cp.xy(sp)[0], float)) @ tan)         # 沿切線一階細化成連續值（直線段上精確），免得錨定本身帶入 ≤ 5 mm 的跳動
        tan = np.asarray(cp.tangent(sp)[0], float)
        v_t = max(float(vw @ tan), 0.0)
        s_new = sp - v_t * tau
        ref_old = np.array([s0 + vb0 * tau, y0])
        jump = float(np.linalg.norm(np.asarray(cp.xy(sp)[0], float) - ref_old))
        dv = float(np.linalg.norm(v_t * tan - vb0 * np.array([math.cos(th0), math.sin(th0)])))
        ok_p, prof = PC.ref_profile_check(cp, s_new, v_t, float(self.park[0]), self.vcap, self.v_carry, R)
        self.ho = dict(t=self.t, sigma_eff=sp, v_eff_world=vw.tolist(), v_t=v_t, s_ref_old=self.s_ref, v_b_old=self.v_b, s_ref_new=s_new, ref_jump=jump, v_ref_jump=dv, profile=prof,
                       lead_new=s_new + v_t * tau - sp)
        if jump > R['ref_jump']:
            self._pstop('carry_ref_discontinuous', jump=jump, where='reanchor')
            return
        if dv > R['v_ref_jump']:
            self._pstop('carry_ref_discontinuous', dv=dv, where='reanchor')
            return
        if not ok_p:
            self._pstop('carry_ref_accel', where='reanchor', **prof)
            return
        self.s_ref, self.v_b = s_new, v_t
        self._ev('handover_reanchor', **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.ho.items() if k not in ('profile', 't')})

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(name='G_W1_carry_handover_aligned', handover=self.ho,
                                    scope='同 G；只改承接 CARRY_NAV 那一拍的參考錨定（對齊命令生效時刻的執行狀態），檢查照原門檻重做；無可調參數')
        return r
