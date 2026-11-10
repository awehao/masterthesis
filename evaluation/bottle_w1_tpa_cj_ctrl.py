#!/usr/bin/env python3
"""TPA＋持瓶承接方向修正（版本化子類，基線＝bottle_w1_tpa_ctrl；其餘全部相同）。
問題（bottle_motm_tpa_nom_2，21.8 s carry_ref_accel 受控停止）：bottle_place_ctrl 進 CARRY 承接持瓶路徑時，接段 Bézier 的起點切線 dw 取
  「MPC 底盤計畫第一拍速度」的方向（|v| > 1e-3 即採用）。持瓶路徑回覆比 nom_1 晚一拍（規劃 329 ms vs 253 ms）：參考已停（v_b＝0）、實測超前參考 1.4 mm，
  MPC 計畫小幅倒退 ⇒ dw 朝後約 176° ⇒ 接段折返，參考剖面 a_x 1.27／a_y 1.07（> 1.0）。
  離線重建（同 build_carry／curvature_grid／ref_profile_check、區域座標）：nom_1 實錄值（0.800／0.512／0.365）由 dw 0° 重現（0.799／0.493／0.352）；
  nom_2 實錄值（1.266／1.073／v_cap 0.1397）落在 dw 175°（1.290／1.098／0.138）與 180° 之間；同一 nom_2 幾何以 dw 0° 為 0.799／0.497／0.355（通過）。
修正：承接那一拍，MPC 計畫第一拍速度沿車頭的分量 < V_DIR_MIN（含倒退、近乎靜止）⇒ dw 改用車頭方向（底盤座標 +x）；
  否則照原規則。只在 CARRY 承接（pph is None）那次呼叫期間替換 base_plan 第 0 列的方向，呼叫後還原；不改 GMPC、判準與門檻。
"""
import numpy as np

import bottle_w1_tpa_ctrl as TPA

V_DIR_MIN = 0.02            # m/s：計畫速度沿車頭分量低於此值不當作行進方向（接近靜止或倒退）


class W1PlaceController(TPA.W1PlaceController):
    group = TPA.W1PlaceController.group + '_CJ'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cj = []

    def _advance_inner(self, T_tcp, close_done, q):
        bp = getattr(self, 'base_plan', None)
        swap, bp2 = None, None
        if self.st.phase == 'CARRY' and getattr(self, 'pph', None) is None and bp is not None and len(bp) > 0:
            v0 = np.asarray(bp[0], float)
            if float(v0[0]) < V_DIR_MIN:
                swap = v0.copy()
                bp2 = np.array(bp, float, copy=True)
                bp2[0, 0], bp2[0, 1] = max(float(np.linalg.norm(v0[:2])), V_DIR_MIN), 0.0
                self.base_plan = bp2
        try:
            super()._advance_inner(T_tcp, close_done, q)
        finally:
            if swap is not None:
                if self.base_plan is bp2:
                    self.base_plan = bp                                    # 只在承接這次呼叫期間替換，呼叫後還原
                rec = dict(t=round(float(self.t), 3), v_plan_body=np.round(swap[:3], 4).tolist(), used='heading', pph_after=getattr(self, 'pph', None))
                self.cj.append(rec)
                if len(self.cj) == 1 or rec['pph_after'] is not None:
                    self._ev('carry_join_dir', **rec)

    def report(self):
        r = super().report()
        r['carry_join_dir'] = dict(v_dir_min=V_DIR_MIN, log=self.cj[-20:], n=len(self.cj),
                                   note='承接持瓶路徑那拍：MPC 計畫速度沿車頭分量 < V_DIR_MIN ⇒ 接段起點切線用車頭方向（原規則會在倒退時取到朝後方向）')
        return r
