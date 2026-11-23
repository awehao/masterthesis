#!/usr/bin/env python3
"""接近末段排程提速候選（Codex 20261102_193444；版本化子類，基線＝bottle_w1_carrygov_ctrl：視向、抓取、持瓶、放置完全相同）。
限速來源（bottle_grasp_speed_limits）：漸變前固定慢速點 x_slow（＝x_dep 附近）起，底盤以 v_c 走 0.30 m（2.35–2.55 s），原因是 R 段（V_HIGH→備妥 2.35 s）
固定在 x_dep 才開始、且須在漸變起點前完成。有在線視向後抓取計畫多在 H 段之前就已承接，H 段高位保持已無取得用途。
1 R 段提早：抓取計畫已承接（vfix）且序列在 H（手臂已在 V_HIGH＋視向偏移，到位判定照舊）⇒ 立即開始 R（同 min-jerk、同時長、同 R→DONE 判定；
  起點＝H 的參考 V_HIGH，視向偏移照 gaze 規則在 R 段平順歸零）。未承接或已過 x_dep ⇒ 照原規則。x_dep 不改（抓取計畫最晚處置點不變）。
2 車速排程（CRUISE、未漸變、已承接、序列在 R 或 DONE）：沿候選自己的車速進度預測抵達漸變起點 x_blend 的時刻，取「抵達 ≥ R 完成時刻＋T_MARGIN」的最高巡航速度
  v_cr ∈ [v_c, v_nav]（二分），到離散煞停點再降到 v_c。預測與即時判定用同一個離散模型：參考車速每 2 控制拍更新、每次 ±A_SLEW·2·DT、
  已下達命令的在途距離 v·D_STEPS·PHYS、路徑速度換 x 速度用目前 dy。即使 v_cr＝v_c 也會早到 ⇒ 用 v_c（原慢速），既有「x_blend 時未展完就減速等手臂」只當後備。
  未承接或序列未到 R ⇒ 原規則（x_slow 前 v_nav、之後 v_c）。
不改：v_c、窗口、t_blend、t_insert、抓持證據、抬升、可達分支、淨空、停止條件、GMPC、持瓶段。
"""
import math

import numpy as np

import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_carrygov_ctrl as CG

C3, C2 = CG.C3, CG.C2
T_MARGIN = 0.10
N_BISECT = 12


class W1PlaceController(CG.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE_CARRYGOV_APPROACHSCHED'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.as_ev, self.as_log = [], []
        self.as_early_r = None
        self.as_wait = dict(arm_s=0.0, near_stop_s=0.0)

    # ---- 1 R 段提早
    def _vstep(self, q):
        sq = self.vh_seq
        if sq['ph'] == 'H' and self.vfix is not None and float(q[0]) < self.x_dep and self.as_early_r is None:
            self.as_early_r = dict(t=round(float(self.t), 3), x=round(float(q[0]), 4), x_dep=round(float(self.x_dep), 4), t_acc=round(float(self.vfix['t']), 3))
            self._vset('R', sq['qb'], self.q_arm_ready, float(q[0]))
            self._ev('early_R', **{k: v for k, v in self.as_early_r.items() if k != 't'})
        super()._vstep(q)

    # ---- 2 車速排程（離散模型；全部用參考的 x 速度：路徑速度 × 1／√(1＋dy²)，同 S1._v_target）
    @staticmethod
    def _brake_dist(v, v_lo):
        """x 速度 v 以每 2 控制拍 −A_SLEW·2·DT 降到 v_lo 所走的 x 距離（不含在途）。"""
        da, d = S1.A_SLEW * 2.0 * B.DT, 0.0
        while v > v_lo + 1e-12:
            d += v * 2.0 * B.DT
            v = max(v - da, v_lo)
        return d

    def _tgt(self, x, v, v_cr, v_lo):
        return v_lo if x + v * C0.D_STEPS * C0.PHYS >= self.x_blend - self._brake_dist(v, v_lo) else v_cr

    def _arrival(self, x0, v0, v_cr, v_lo):
        """自目前（x0、參考 x 速度 v0）照候選規則前進到 x_blend 所需時間：每 2 控制拍更新參考、斜率 ±A_SLEW·2·DT，先加已下達命令的在途距離。"""
        da, x, v, t, k = S1.A_SLEW * 2.0 * B.DT, float(x0), float(v0), 0.0, 0
        x += v * C0.D_STEPS * C0.PHYS
        while x < self.x_blend and k < 4000:
            if k % 2 == 0:
                v = float(np.clip(self._tgt(x, v, v_cr, v_lo), v - da, v + da))
            x += v * B.DT
            t += B.DT
            k += 1
        return t

    def _sched_active(self):
        return (self.st.phase == 'CRUISE' and not self.blending and self.vfix is not None and self.vh_seq['ph'] in ('R', 'DONE') and self.judge.stop is None)

    def _v_target(self, T_tcp, x):
        if not self._sched_active() or x >= self.x_blend:
            if self.st.phase == 'CRUISE' and not self.blending and x >= self.x_blend and not self._unfolded():
                self.as_wait['arm_s'] += 2.0 * B.DT                # 後備：抵達漸變起點仍未展完（每 2 控制拍呼叫一次）
            return super()._v_target(T_tcp, x)
        sq = self.vh_seq
        t_need = max((sq['t0'] + sq['T'] - float(self.t)) if sq['ph'] == 'R' else 0.0, 0.0) + T_MARGIN
        scale = 1.0 / math.sqrt(1.0 + self.path.dy(x) ** 2)
        v0, v_lo, v_hi = float(self.v_b), float(self.v_c) * scale, float(self.v_nav) * scale
        lo, hi = v_lo, v_hi
        if self._arrival(x, v0, hi, v_lo) >= t_need:
            v_cr, why = hi, 'nav'
        elif self._arrival(x, v0, lo, v_lo) < t_need:
            v_cr, why = lo, 'vc_early'
        else:
            for _ in range(N_BISECT):
                mid = 0.5 * (lo + hi)
                if self._arrival(x, v0, mid, v_lo) >= t_need:
                    lo = mid
                else:
                    hi = mid
            v_cr, why = lo, 'matched'
        v = self._tgt(x, v0, v_cr, v_lo)
        if len(self.as_log) < 2000:
            self.as_log.append((round(float(self.t), 3), round(float(x), 4), round(v0, 4), round(v, 4), round(v_cr, 4), why, round(t_need, 3), sq['ph']))
        return v

    def _advance_inner(self, T_tcp, close_done, q):
        if self.st.phase == 'CRUISE' and not self.blending and self.vfix is not None and float(self.v_b) < 0.02 and self.judge.stop is None:
            self.as_wait['near_stop_s'] += B.DT
        super()._advance_inner(T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['approach_sched'] = dict(t_margin=T_MARGIN, early_R=self.as_early_r, wait=dict({k: round(v, 3) for k, v in self.as_wait.items()}),
                                   log_cols=['t', 'x', 'v_b', 'v_tgt', 'v_cr', 'why', 't_need', 'vh_ph'], log=self.as_log,
                                   note='R 段承接後且在 H 即開始；R／DONE 期間依離散車速模型取抵達漸變起點不早於 R 完成＋T_MARGIN 的最高巡航速度；後備＝原等手臂規則')
        return r
