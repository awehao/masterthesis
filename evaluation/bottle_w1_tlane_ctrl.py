#!/usr/bin/env python3
"""目標專屬入口／車道的控制器（Codex 20261110_233102 第 2 項；版本化子類，基線＝bottle_w1_carrygov_ctrl，GMPC、抓取、安全規則全部相同）。
差別只在「入口與車道從哪裡來」：
  起跑前的規劃仍到任務定義的觀察導引入口（導引路徑，Codex 核可預先規劃）；**第一次送抓取計畫請求時**（＝取得第一個合格 known_type 快照），
  先以 bottle_tlane.decide 由本趟量測（obs_last）與當下狀態（σ_robot、煞停距離）生成目標專屬入口 x_entry、車道 y_lane 與重接路徑，
  經執行端注入的 lane_planner 對最新代價圖核對；通過 ⇒ 換上新路徑（s_ref、sig_robot 平移 Δσ0；執行端同時換路徑監看），
  之後抓取請求帶 y_lane（工作者沿該車道做分支核對）。不通過（無觀測、舊觀測、太晚、代價圖擋住或過期）⇒ 受控停止 target_lane_rejected，
  **不退回固定車道**。車道只在第一次決定；之後的重送（量測更新 > dev_tol）沿用已採用的車道（差異由手臂分支吸收，照記錄）。
"""
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_tlane as TL
import bottle_w1_carrygov_ctrl as CG
import bottle_w1_online_ctrl as ON


class W1PlaceController(CG.W1PlaceController):
    group = CG.W1PlaceController.group + '_TLANE'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lane_planner = None                     # 執行端注入：fn(obs, t, sig_robot, d_stop) → bottle_tlane.decide 的結果
        self.tlane, self.tlane_log = None, []
        sub = self.pc.submit

        def submit(kind, inp):
            if kind == 'grasp' and self.tlane is not None:
                inp = dict(inp, y_lane=float(self.tlane['y_lane']))
            return sub(kind, inp)
        self.pc.submit = submit

    def _gp_submit(self, xy, zs, seq):
        if self.tlane is None:
            if self.lane_planner is None:
                self.judge._stop(self.t, 'lane_planner_missing')
                return
            v = max(float(getattr(self, 'v_b', 0.0)), ON.V_STOP_FLOOR)
            d_stop = v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0])) + 0.02
            r = self.lane_planner(self.obs_last, float(self.t), float(self.sig_robot), d_stop)
            rec = {k: v_ for k, v_ in r.items() if k != 'path_new'}
            rec.update(t=round(float(self.t), 3), seq=int(seq), sig_robot=round(float(self.sig_robot), 4), d_stop=round(d_stop, 4))
            self.tlane_log.append(rec)
            if not r['ok']:
                self._ev('tlane_reject', why=r['why'])
                self.judge._stop(self.t, 'target_lane_rejected', why=r['why'])
                return
            ds = float(r['dsig0'])
            self.path = r['path_new']
            self.s_ref = float(self.s_ref) + ds
            self.sig_robot = float(self.sig_robot) + ds
            self.tlane = rec
            self._ev('tlane_accept', x_entry=r['x_entry'], y_lane=r['y_lane'], sig_join=r['sig_join'], dsig0=ds, cost_max=r.get('cost_max'))
        super()._gp_submit(xy, zs, seq)

    def report(self):
        r = super().report()
        r['tlane'] = dict(accepted=self.tlane, log=self.tlane_log, y_reach_nom=TL.Y_REACH_NOM, runup=TL.RUNUP,
                          note='目標專屬入口／車道：第一個合格量測後生成並核對；之後重送沿用；不通過受控停止、不退回固定車道')
        return r
