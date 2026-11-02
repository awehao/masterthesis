#!/usr/bin/env python3
"""在線 MotM 第五版：online3＋規劃路徑格心換算接口（Codex 20261102_085048；版本化子類，基線＝bottle_w1_online3_ctrl，控制與排程全部相同）。
控制器本身只多記錄：
1 持瓶承接當下的實際在途前推狀態：在 CARRY 且尚未進入放置子相位（承接嘗試期間）呼叫 _inflight 時，記錄任務時刻、在途前推位姿 qp（build_carry 使用的就是它）、實測 q；
  承接結果（CARRY_NAV 的 lateral 或停止原因）另由事件取得。請求時刻與起點見 carry_req，回覆在承接同一拍取用。
2 規劃座標換算設定與每個回覆的換算紀錄（由腳本設定 planner_conv／planner_ros；換算本身在 bottle_n2_ros_center.PlanRosCenter）。
"""
import numpy as np

import bottle_w1_online3_ctrl as ON3

C3, C2 = ON3.C3, ON3.C2


class W1PlaceController(ON3.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.join_log, self.planner_conv, self.planner_ros = [], None, None

    def _inflight(self, q):
        out = super()._inflight(q)
        if self.st.phase == 'CARRY' and self.pph is None:
            self.join_log.append(dict(t=float(self.t), qp=np.asarray(out[0], float)[:3].round(5).tolist(), q=np.asarray(q, float)[:3].round(5).tolist()))
        return out

    def report(self):
        r = super().report()
        pr = self.planner_ros
        r['carry_join_state'] = dict(log=self.join_log[-5:], note='CARRY 承接嘗試期間每次 _inflight 的在途前推位姿 qp（build_carry 用之）與實測 q；最後一筆＝承接當拍')
        r['planner_conv'] = dict(cfg=self.planner_conv, log=None if pr is None else [{k: v for k, v in x.items()} for x in getattr(pr, 'conv_log', [])])
        return r
