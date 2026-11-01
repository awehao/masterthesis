#!/usr/bin/env python3
"""進 CARRY 時依實測重新規劃持瓶段（版本化子類，2026-11-01；Codex 20261101_125000 選 (a)）。以分段抬升候選（bottle_w1_liftb_ctrl）為底，凍結 G 不動。
動機（bottle_w1_dyn2l_g1）：持瓶段路徑只在 LIFT 開始請求一次，起點用「實測＋世界速度×t_lift」預測；第二案避讓使底盤在 LIFT 內多走約 0.2 m，
進 CARRY 時實測離規劃路徑 0.104 m > join_lat 0.05 m ⇒ carry_join_lateral。
本候選只驗證「避障改變實際狀態後，運送計畫能否從當下狀態承接」，不是一般途中重規劃：
  1 LIFT 開始不送請求（carry_req 標成延後）；進 CARRY 的第一個週期非同步請求一次：起點＝帶時間戳的實測狀態經在途命令前推（_inflight），
    目標＝以進 CARRY 時實際持瓶構型重解的停車位往回 entry_back（同原算法）。請求不阻塞時鐘、掃描、控制與安全監看。
  2 等待回覆：st.phase 已是 CARRY、放置相位未設，仍由同一 GMPC 求解（S0b CARRY 參考：錨定進 CARRY 時的實測位姿），持瓶目標＝底盤座標終點（連續承接）；
    進度參考速度目標改 0、經既有斜率限制降速（不直接下零速度、不凍結底盤）；CBF、Shield、滑移、接觸、淨空、掃描過期監看照常。
  3 回覆到達：照原採用核對（規劃有效、框架與起訖點、承接幾何 join_lat、接段連續 ref_jump／v_ref_jump、含直線段的完整路徑代價、代價圖新鮮度、參考加速度），
    全部以回覆當下的在途前推狀態計算；任一不過或規劃失敗 ⇒ 受控停止。等待超過 T_WAIT_MAX＝COSTMAP_MAX_AGE（代價圖新鮮度上限 2.0 s）⇒ carry_plan_timeout 受控停止。
    只請求一次，不重送。判準一律不放寬。
"""
import numpy as np

import bottle_n2_path as NP
import bottle_s0_ctrl as C0
import bottle_w1_liftb_ctrl as LB

T_WAIT_MAX = NP.COSTMAP_MAX_AGE


class W1PlaceController(LB.W1PlaceController):
    group = 'G_LIFTC'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.carry_req = dict(deferred='CARRY')             # 父類 LIFT 開始的請求條件是 carry_req is None：標成延後即不送
        self.cw = None

    def _carry_pending(self, q):
        if self.cw is None:
            qp, _ = self._inflight(q)
            self.park, _ = self._park_solve(q)
            goal = (float(self.park[0] - self.place['entry_back']), float(self.park[1]))
            self.carry_req = dict(t_req=self.t, at='CARRY', start=[float(v) for v in qp[:3]], q_meas=[float(v) for v in q[:3]],
                                  t_eff=self.t + C0.D_STEPS * C0.PHYS, goal=goal, park=self.park.tolist())
            self.carry_planner.request(tuple(qp[:3]), goal, self.t)
            self.cw = dict(t_req=float(self.t), v_b0=float(self.v_b), t_accept=None, n_wait=0)
            self._ev('carry_plan_request', **self.carry_req)
            return True
        self.cw['n_wait'] += 1
        if self.t - self.cw['t_req'] > T_WAIT_MAX:
            self._pstop('carry_plan_timeout', wait=float(self.t - self.cw['t_req']), limit=T_WAIT_MAX)
        return True

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY' and self.pph is None:
            return 0.0                                      # 等待持瓶段路徑：進度參考經斜率限制降到 0
        return super()._v_target(T_tcp, x)

    def _set(self, ph, **info):
        if ph == 'CARRY_NAV' and self.cw is not None:
            self.cw['t_accept'] = float(self.t)
            self.cw['wait_s'] = float(self.t - self.cw['t_req'])
            self.cw['v_b_accept'] = float(self.v_b)
        super()._set(ph, **info)

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_lift_transport_carry_replan', carry_wait=self.cw, t_wait_max=T_WAIT_MAX,
                                    scope=r['paired_strategy']['scope'] + '；持瓶段路徑改在進 CARRY 時依在途前推實測請求一次，等待時同一 GMPC 降速保持，回覆時照原核對承接')
        return r
