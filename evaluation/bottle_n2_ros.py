#!/usr/bin/env python3
"""瓶子 N2 的 ROS 端：向 planner_server 要路徑（ComputePathToPose，非阻塞）、訂代價圖餵 PathMonitor。
起跑前規劃一次：等代價圖已融入起跑後的掃描，送請求，期間機器人保持、/clock 照走；只採用最新請求的回覆，核對失敗或沒有路徑 ⇒ 不起跑（不退回工程繞行路線）。
"""
import time

import numpy as np

import bottle_n2_path as NP
from bottle_s0_ctrl import PHYS


class PlanRos:
    def __init__(self, node):
        from nav2_msgs.action import ComputePathToPose
        from nav_msgs.msg import OccupancyGrid
        from rclpy.action import ActionClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        self.node, self.Goal = node, ComputePathToPose.Goal
        self.ac = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
        self.grid, self.n_grid, self.monitor, self.sig_fn = None, 0, None, None
        self.req_id, self.pending, self.result, self.log = 0, None, None, []
        node.create_subscription(OccupancyGrid, '/global_costmap/costmap', node._w(self._grid_cb),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _grid_cb(self, m):
        st = float(m.header.stamp.sec) + 1e-9 * float(m.header.stamp.nanosec)
        if m.header.frame_id != 'map':
            return
        i = m.info
        self.grid = NP.Grid(m.data, i.resolution, i.origin.position.x, i.origin.position.y, i.width, i.height, st)
        self.n_grid += 1
        if self.monitor is not None:
            self.monitor.on_grid(self.grid, self.sig_fn())

    def request(self, start_xyyaw, goal_xy, ros_t):
        from geometry_msgs.msg import PoseStamped

        def pose(x, y, yaw):
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.header.stamp = self.node.stamp(ros_t)
            p.pose.position.x, p.pose.position.y = float(x), float(y)
            p.pose.orientation.z, p.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
            return p
        g = self.Goal()
        g.goal, g.start, g.use_start, g.planner_id = pose(goal_xy[0], goal_xy[1], 0.0), pose(*start_xyyaw), True, 'GridBased'
        self.req_id += 1
        rid = self.req_id
        self.pending = dict(id=rid, t=ros_t, wall=time.perf_counter(), stage='send', fut=self.ac.send_goal_async(g))
        self.log.append(dict(id=rid, t=ros_t, start=[round(float(v), 4) for v in start_xyyaw], goal=list(goal_xy)))
        return rid

    def poll(self):
        """推進未完成的請求；舊請求的回覆一律丟棄。"""
        p = self.pending
        if p is None or not p['fut'].done():
            return
        if p['stage'] == 'send':
            gh = p['fut'].result()
            if gh is None or not gh.accepted:
                self.result, self.pending = dict(id=p['id'], ok=False, why='goal_rejected'), None
                return
            p['stage'], p['fut'] = 'result', gh.get_result_async()
            return
        r = p['fut'].result()
        self.pending = None
        if p['id'] != self.req_id:
            self.log.append(dict(id=p['id'], dropped='stale_reply'))
            return
        path = r.result.path
        P = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in path.poses], float).reshape(-1, 2)
        self.result = dict(id=p['id'], ok=r.status == 4 and len(P) >= 2, status=int(r.status), error_code=int(getattr(r.result, 'error_code', 0)),
                           error_msg=str(getattr(r.result, 'error_msg', '')), frame=path.header.frame_id, P=P,
                           planning_time=float(r.result.planning_time.sec + 1e-9 * r.result.planning_time.nanosec), wall_ms=(time.perf_counter() - p['wall']) * 1e3)


def plan_before_start(node, chain, pr, hold_step, get_state, scan_ranges, goal_xy, x_lane, t0, wall_limit_s=60.0):
    """回傳 (PlannedPath, 資訊) 或 (None, 原因)。t0 是握手結束時的 ROS 時間；之後每步 hold_step 推進物理、/clock 前進。"""
    t, k, w0 = float(t0), 0, time.perf_counter()
    t_grid_need = t0 + 1.0                                 # 代價圖要融入握手後至少 1 s 的掃描
    sent = None
    while True:
        p, qt, q_arm = get_state()
        node.publish_state(t, p, qt, q_arm)
        chain.set_now(t)
        node.drain()
        if k % int(round(0.1 / PHYS)) == 0:
            chain.pub_scan_at(t, scan_ranges())
        pr.poll()
        if sent is None and pr.grid is not None and pr.grid.stamp >= t_grid_need and pr.ac.server_is_ready():
            yaw = 2.0 * np.arctan2(qt[3], qt[0])
            sent = pr.request((float(p[0]), float(p[1]), float(yaw)), goal_xy, t)
        if pr.result is not None:
            res = pr.result
            info = {k_: v for k_, v in res.items() if k_ != 'P'}
            if not res['ok']:
                return None, dict(why='no_valid_path', **info)
            ok, why = NP.validate_plan(res['frame'], res['P'], np.array(p[:2], float), np.array(goal_xy, float), pr.grid)
            if not ok:
                return None, dict(why='plan_rejected', detail=why, **info)
            J = NP.join_lane(res['P'], x_lane)
            c = pr.grid.cost(J)
            if NP.blocked(c).any():
                return None, dict(why='plan_rejected', detail=f'接直線段後 {int(NP.blocked(c).sum())} 點在內切以上', **info)
            info.update(t_ready=t, n_raw=len(res['P']), cost_max=int(c.max()), raw=res['P'][::5].round(4).tolist())
            return NP.PlannedPath(J, x_lane), info
        if time.perf_counter() - w0 > wall_limit_s:
            return None, dict(why='plan_timeout', sent=sent, n_grid=pr.n_grid, server=pr.ac.server_is_ready())
        hold_step()
        t += PHYS
        k += 1
