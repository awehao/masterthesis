#!/usr/bin/env python3
"""區域座標（Codex 20261102_090007 R-B1：抓取區域相對原場景平移、不旋轉）。
約定：上層任務與控制器全程在「區域座標」運作（＝原場景座標：第一桌在原點、抓取車道 y＝0、相對抓取窗、V_HIGH 觸發、第一桌閘、join_lane 都不變）；
世界端（Isaac 場景、/odom 與 TF、地圖、planner_server、掃描、真值安全網、相機）全程世界座標。兩者只差平移 O＝(ox, oy)：世界＝區域＋O。
邊界每處只轉一次：控制器輸入（底盤位姿、瓶位姿真值、障礙物、候選 xy、規劃路徑與代價格查詢、放置先驗）世界→區域；控制器對外（規劃請求起訖、掃描射線位姿）區域→世界。
底盤速度命令是車體座標，不受平移影響。代價圖本身不移動：RegionGrid 在查詢時把區域點加 O 後照世界格查。
"""
import time

import numpy as np

import bottle_n2_path as NP
import bottle_n2_ros as NR
import bottle_n2_ros_center as NRC
from bottle_s0_ctrl import PHYS


class RegionFrame:
    def __init__(self, ox, oy):
        self.O = np.array([float(ox), float(oy)])

    def to_world(self, xy):
        return np.asarray(xy, float)[..., :2] + self.O

    def to_region(self, xy):
        return np.asarray(xy, float)[..., :2] - self.O

    def q_to_region(self, q):
        q = np.array(q, float)
        q[:2] -= self.O
        return q

    def q_to_world(self, q):
        q = np.array(q, float)
        q[:2] += self.O
        return q

    def T_to_region(self, T):
        T = np.array(T, float)
        T[:2, 3] -= self.O
        return T

    def obs_to_region(self, obs):
        return [dict(o, x=float(o['x']) - self.O[0], y=float(o['y']) - self.O[1]) for o in obs]

    def place_to_region(self, place):
        p = dict(place)
        p['c'] = tuple(np.asarray(place['c'], float) - self.O)
        p['target'] = tuple(np.asarray(place['target'], float) - self.O)
        p['lane_y'] = float(place['lane_y']) - self.O[1]
        return p


class RegionGrid(NP.Grid):
    """世界代價格；查詢點為區域座標時加 O 再查（原點、解析度欄位維持世界值）。"""

    def __init__(self, data, res, ox, oy, w, h, stamp, O):
        super().__init__(data, res, ox, oy, w, h, stamp)
        self.O = np.asarray(O, float)

    def cost(self, P):
        return super().cost(np.atleast_2d(np.asarray(P, float)) + self.O)


class PlanRosRegion(NRC.PlanRosCenter):
    """請求起訖（區域）→ 世界；回覆：先 PlanRosCenter 的格心換算（世界），再轉區域（各一次）；代價格以 RegionGrid 提供。"""

    def __init__(self, node, conv, R):
        self.R = R
        super().__init__(node, conv)

    def _grid_cb(self, m):
        st = float(m.header.stamp.sec) + 1e-9 * float(m.header.stamp.nanosec)
        if m.header.frame_id != 'map':
            return
        i = m.info
        self.grid = RegionGrid(m.data, i.resolution, i.origin.position.x, i.origin.position.y, i.width, i.height, st, self.R.O)
        self.n_grid += 1
        if self.monitor is not None:
            self.monitor.on_grid(self.grid, self.sig_fn())

    def request(self, start_xyyaw, goal_xy, ros_t):
        s = list(start_xyyaw)
        sw = self.R.to_world(s[:2])
        gw = self.R.to_world(goal_xy)
        return super().request((float(sw[0]), float(sw[1]), float(s[2])), (float(gw[0]), float(gw[1])), ros_t)

    def poll(self):
        super().poll()
        r = self.result
        if r is None or r.get('region') is not None:
            return
        P = r.get('P')
        if P is not None and len(P):
            r['P_world'] = np.asarray(P, float).round(5).tolist()
            r['P'] = self.R.to_region(np.asarray(P, float))
        r['region'] = dict(O=self.R.O.tolist(), applied=P is not None and len(P) > 0)


def plan_before_start_region(node, chain, pr, hold_step, get_state, scan_ranges, goal_xy, x_lane, t0, R, wall_limit_s=60.0):
    """bottle_n2_ros.plan_before_start 的區域版：發布狀態用世界位姿（/odom、TF），規劃請求、核對與接直線段用區域位姿（PlanRosRegion 自行轉世界）。"""
    t, k, w0 = float(t0), 0, time.perf_counter()
    t_grid_need = t0 + 1.0
    sent = None
    while True:
        p, qt, q_arm = get_state()
        node.publish_state(t, p, qt, q_arm)
        chain.set_now(t)
        node.drain()
        if k % int(round(0.1 / PHYS)) == 0:
            chain.pub_scan_at(t, scan_ranges())
        pr.poll()
        p_reg = R.to_region(np.asarray(p[:2], float))
        if sent is None and pr.grid is not None and pr.grid.stamp >= t_grid_need and pr.ac.server_is_ready():
            yaw = 2.0 * np.arctan2(qt[3], qt[0])
            sent = pr.request((float(p_reg[0]), float(p_reg[1]), float(yaw)), goal_xy, t)
        if pr.result is not None:
            res = pr.result
            info = {k_: v for k_, v in res.items() if k_ != 'P'}
            if not res['ok']:
                return None, dict(why='no_valid_path', **info)
            age = t - pr.grid.stamp
            if age > NP.COSTMAP_MAX_AGE:
                return None, dict(why='plan_rejected', detail=f'採用時代價圖已 {age:.2f} s 未更新', **info)
            ok, why = NP.validate_plan(res['frame'], res['P'], p_reg, np.array(goal_xy, float), pr.grid)
            if not ok:
                return None, dict(why='plan_rejected', detail=why, **info)
            J = NP.join_lane(res['P'], x_lane)
            c = pr.grid.cost(J)
            if NP.blocked(c).any():
                return None, dict(why='plan_rejected', detail=f'接直線段後 {int(NP.blocked(c).sum())} 點在內切以上', **info)
            info.update(t_ready=t, n_raw=len(res['P']), cost_max=int(c.max()), grid_stamp=pr.grid.stamp, grid_age=round(age, 3), start=[float(v) for v in p_reg], start_world=[float(v) for v in p[:2]],
                        goal=list(goal_xy), goal_world=R.to_world(goal_xy).tolist(), raw=res['P'].round(5).tolist(), joined=J.round(5).tolist(), params=NR.planner_params(), log=pr.log, frame_note='raw／joined 為區域座標')
            return NP.PlannedPath(J, x_lane), info
        if time.perf_counter() - w0 > wall_limit_s:
            return None, dict(why='plan_timeout', sent=sent, n_grid=pr.n_grid, server=pr.ac.server_is_ready())
        hold_step()
        t += PHYS
        k += 1
