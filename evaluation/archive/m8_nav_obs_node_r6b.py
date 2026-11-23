#!/usr/bin/env python3
"""r6b（2026-11-10；Codex 20261110_220329）：r6 的版本化複本，只修離開到達契約：起步前以參數服務讀 GMPC 實際的 goal_tolerance_xy
  （讀不到不起步），傳給 WindowLayer3 當離開到位半徑；離開到位時切主動保持（避免 GMPC arrived 後不解 QP），之後由時窗層以實測位置重評。
r6（2026-11-10；Codex 20261110_215340 A）：r5 的版本化複本，只改決策收斂與參考接管：
  (1) 時窗層換 m8_window3.WindowLayer3（EXIT／COMMIT 固定並零餘裕持續驗證；預測參數同 r4／r5，B 未改）。
  (2) 保持解除與參考接管成對：在保持中先發新計畫，收到 /gmpc/plan_ack 同一時間戳才解除保持；記計畫版本、發布與接管時刻。
  (3) 少重規劃：自 EXIT／停止回巡航時，原路剩餘段只在「接回段 ≤ 1.2 m、轉角 ≤ 45°、接回段與剩餘段在最新代價圖都可通行」時沿用；
      否則重規劃，同時只留一個請求（已有未回覆請求就不再送）。重規劃結果在保持中發布，收到 ack 才解除。
r5（2026-11-10；Codex 20261110_200117 D）：r4 的版本化複本，只把「發空計畫停車」全部改成 gmpc_node_m8 的主動保持（/gmpc/hold）：
  STOP_BLOCKED、WAIT 的 stop_now、HOLD_STALE（只有上層航跡決策資料過期；底層障礙過期由 GMPC 自己的 stale 保護失效關閉）、WAIT_TIMEOUT，
  以及 WAIT 走到停點 0.25 m 內（否則 GMPC 判 arrived 直接歸零、不解 QP）。離開這些模式時先解除保持再發計畫。時窗層參數同 r4（B 未改）。
  另：起步須感知就緒（收到過 /gmpc/tracks_debug 且接收間隔 ≤ 0.5 s），起跑核對記錄航跡接收間隔與數量。
r4（2026-11-10；Codex 20261110_194328 核可 A）：r3b 的版本化複本，只把時窗層換成 m8_window2.WindowLayer2（不穩移動物用速度上界 0.8 m/s、
  機器人通過時間上下界、逐點時空衝突與整段 COMMIT 核對）。其餘同 r3b。
r3b（2026-11-10）：r3 的版本化複本，只改起步與規劃請求的強健性（m8fake_r3_T25p83：代價圖在地圖到達前以預設 5×5 m 啟動、
  機器人在圖外，第一個請求沒有回覆而卡在 PLAN）：送請求前須 action server 就緒、代價圖涵蓋機器人位置（該格非未知）；
  請求 T_PLAN_RETRY 秒無回覆即重送，最多 N_PLAN_TRY 次，仍無回覆 ⇒ result＝plan_no_reply。決策邏輯（m8_window）不變。
r3（2026-11-10；Codex 20261110_190030 選 C）：r2 的版本化複本，衝突層換成穿越時窗層 m8_window.WindowLayer；搭配 gmpc_node_m8 heading_source＝plan_yaw
  （計畫點朝向即航向參考：巡航路徑朝向＝路徑切線、離開線朝向＝當下車頭＝純側移）。模式：CRUISE／COMMIT 照原路徑；WAIT 截到停點
  （stop_now 發空計畫）；EXIT 側移線；STOP_BLOCKED／HOLD_STALE 空計畫；WAIT 回 CRUISE 重發原路徑剩餘段，EXIT／停止回 CRUISE 重新規劃；
  WAIT_TIMEOUT ⇒ result＝wait_timeout（未完成，不強行穿越）。
r2（2026-11-10；Codex 20261110_182724 候選）：r1 的版本化複本，只加兩件事——
  (1) 預測通行衝突層（m8_conflict.ConflictLayer；只用線上確認航跡 /gmpc/tracks_debug、剩餘路徑、位姿）：
      WAIT＝把截到停點的路徑交 GMPC（停點＝原地時發空計畫，GMPC 輸出零）；EXIT＝把直線離開路徑交 GMPC（各點朝向＝目前車頭）；
      回 CRUISE＝從當下位姿向 Smac 重規劃到門前觀察區。WAIT／EXIT 期間選擇器照樣轉送平滑器輸出（GMPC＋CBF＋Shield 仍在下層）。
  (2) 碰撞失敗：收到 /m8/collision（執行端 ev9r9c 鎖存）⇒ result＝collision、輸出零（實驗保護；不是避障成功或有效停止的證據）。
  到達（done）與安全結果分開記：rep['safety']。
r1 說明：第八次第一包：掃描驅動導航到門前觀察區（Codex 20261110_180004）。door_nav1_mission_node_r9 的命令鏈位置不變，換掉「固定折線計畫」：
  gmpc_node（封存；GMPC＋horizon CBF）→ /cmd_vel_nav → velocity_smoother → /cmd_vel_pre_mux
  → 本節點選擇器〔CRUISE：轉送平滑器輸出｜ALIGN：低速 P 伺服｜ZERO：精確零〕→ /cmd_vel_pre_shield → scan_safety_shield → /cmd_vel → 執行端
目標＝門前觀察區（evaluation/m8_scene_check.py 由門的先驗幾何＋底盤半徑算出，讀 第八次進度報告素材/資料/m8_scene_check.json）；不讀舊 D1 停位。
路徑＝向 planner_server（SmacPlanner2D；全域代價圖＝靜態地圖＋/scan 障礙層＋膨脹，evaluation/bottle_n2_planner.yaml）請求，
  bottle_n2_path.validate_plan 核對（frame、起訖、取樣間距、代價）後重取樣 0.05 m、航向取切線（末點＝目標航向），發 /plan（odom 框；map＝odom 單位變換）。
  未知箱體與行人不在靜態地圖，只能經 /scan 進代價圖（規劃）與追蹤器（CBF）；本節點不讀任何真值。
監看：每次代價圖更新核剩餘路徑；被擋（內切以上或未知）**連續 ≥ T_BLOCK** 才重規劃（行人走過造成的短暫佔據不觸發），最多 N_REPLAN 次；
  重規劃期間照舊轉送（GMPC＋CBF＋Shield 仍在）。代價圖過期 > COSTMAP_MAX_AGE ⇒ ZERO 並記 costmap_stale（fail-closed）。
到達：距目標 ≤ align_r 進 ALIGN（|v| ≤ v_align、|w| ≤ w_align、逐軸加速度限制）；位置 ≤ tol_pos 且偏航 ≤ tol_yaw ⇒ ZERO＋DONE。
    python3 evaluation/m8_nav_obs_node.py --out RUN_DIR/m8nav.json
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bottle_n2_path as NP  # noqa: E402
import m8_window3 as WL  # noqa: E402
import bottle_n2_ros as NR  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import Float32MultiArray, String  # noqa: E402

VERSION = 'm8_nav_obs_node/r6b'
REUSE_MAX, REUSE_TURN_DEG, REUSE_AHEAD = 1.2, 45.0, 10
T_PLAN_RETRY, N_PLAN_TRY = 5.0, 3
DS = 0.05
T_BLOCK, N_REPLAN = 2.0, 6
HEAD_WIN = 3


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class M8Nav(Node):
    def __init__(self, a):
        super().__init__('m8_nav_obs')
        self.a = a
        chk = json.load(open(a.goal_file))
        z = chk['obs_zone']
        self.goal = (float(z['x']), float(z['y']), float(z['yaw']))
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, '/plan', lat)
        self.out_pub = self.create_publisher(Twist, '/cmd_vel_pre_shield', 10)
        self.state_pub = self.create_publisher(String, '/m8nav/state', 10)
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Twist, '/cmd_vel_pre_mux', self._cin, 10)
        self.create_subscription(Twist, '/cmd_vel', self._shield_out, 10)
        self.create_subscription(LaserScan, '/scan', self._scan, qos_profile_sensor_data)
        self.pr = NR.PlanRos(self)
        self.create_subscription(Float32MultiArray, '/gmpc/tracks_debug', self._tracks, 10)
        self.create_subscription(String, '/m8/collision', self._collision, lat)
        self.create_subscription(String, '/gmpc/plan_ack', self._plan_ack, 10)
        self.takeover = None
        from rcl_interfaces.srv import GetParameters
        self._gp_cli = self.create_client(GetParameters, '/gmpc_controller/get_parameters')
        self._gp_fut, self.gmpc_goal_tol = None, None
        self.tracks, self.tracks_t = [], None
        self.cf = WL.WindowLayer3(blocked=self._blocked_static, exit_reached=float('nan'))   # 到位半徑待讀 GMPC 實際容差
        self.cf_mode, self.collision, self.cf_rev = 'CRUISE', None, None
        self.hold_pub = self.create_publisher(String, '/gmpc/hold', 10)
        self.holding, self.hold_t, self.hold_log = False, None, []
        self.vworld = (0.0, 0.0)
        self.pose = self.t = None
        self._vp, self.vmeas = None, None
        self.cin, self.cin_t = (0.0, 0.0, 0.0), None
        self.out_last = (0.0, 0.0, 0.0)
        self.seg = 'WAIT'
        self.P, self.plan_msg, self.t_plan, self.k_proj = None, None, None, 0
        self.t_block0, self.block_info, self.n_replan, self.req_kind = None, None, 0, None
        self.t_start, self.t_align = None, None
        self.u_align = np.zeros(3)
        self.rep_extra = dict(plan_versions=[], conflict=[], safety=dict(collision=None, note='到達（result＝done）與安全分開；碰撞＝執行端 ev9r9c 鎖存'))
        self.rep = dict(version=VERSION, args=vars(a), goal=list(self.goal), goal_source=dict(file=os.path.relpath(a.goal_file, WS), obs_zone=z),
                        planner=NR.planner_params(), segments=[], plans=[], replans=[], n_out=dict(CRUISE=0, ALIGN=0, ZERO=0),
                        shield=dict(n_out=0, n_modified=0, max_abs_diff=0.0), scan=dict(n=0, frame_id=None, last_wall=None), result=None,
                        note='目標由門的先驗幾何算；路徑只經 /scan＋靜態地圖的代價圖；不讀真值')
        self.create_timer(1.0 / a.rate, self._tick)
        self.create_timer(0.2, self._pub_state)
        self.create_timer(1.0, self.save)

    # PlanRos 需要的兩個介面
    def _w(self, fn):
        return fn

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(math.floor(t))
        s.nanosec = int((t - math.floor(t)) * 1e9)
        return s

    # ---- 訂閱
    def _odom(self, m):
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        cur = (self.t,) + self.pose
        if self._vp is None:
            self._vp = cur
        elif cur[0] - self._vp[0] >= 0.1:
            dt = cur[0] - self._vp[0]
            self.vmeas = (math.hypot(cur[1] - self._vp[1], cur[2] - self._vp[2]) / dt, abs(wrap(cur[3] - self._vp[3])) / dt)
            self.vworld = ((cur[1] - self._vp[1]) / dt, (cur[2] - self._vp[2]) / dt)
            self._vp = cur

    def _cin(self, m):
        self.cin, self.cin_t = (m.linear.x, m.linear.y, m.angular.z), self.t

    def _shield_out(self, m):
        s = self.rep['shield']
        s['n_out'] += 1
        d = max(abs(m.linear.x - self.out_last[0]), abs(m.linear.y - self.out_last[1]), abs(m.angular.z - self.out_last[2]))
        if d > 1e-9:
            s['n_modified'] += 1
            s['max_abs_diff'] = max(s['max_abs_diff'], d)

    def _tracks(self, m):
        d = np.asarray(m.data, float)
        d = d.reshape(-1, 12) if len(d) else np.zeros((0, 12))
        self.tracks = [dict(tid=int(r[0]), x=r[1], y=r[2], vx=r[3], vy=r[4], confirmed=bool(r[10] > 0.5)) for r in d]
        self.tracks_t = self.t

    def _collision(self, m):
        if self.collision is None:
            try:
                self.collision = json.loads(m.data)
            except ValueError:
                self.collision = dict(raw=m.data)
            self.rep_extra['safety']['collision'] = dict(self.collision, t_rx=self.t, seg=self.seg)

    def _blocked_static(self, p):
        g = self.pr.grid
        if g is None or self.t is None or self.t - g.stamp > NP.COSTMAP_MAX_AGE:      # 代價圖缺失／過期 ⇒ 擋住（失效關閉）
            return True
        return bool(NP.blocked(g.cost(np.atleast_2d(p)))[0])

    def _plan_ack(self, m):
        tk = self.takeover
        if tk is not None and m.data == tk['stamp']:
            tk['t_ack'] = self.t
            if self.holding:
                self._set_hold(False, 'takeover_ack:' + tk['kind'])
            tk['t_release'] = self.t
            self.takeover = None

    def _version(self, msg, kind, mode):
        st = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        v = dict(kind=kind, mode=mode, stamp=st, t_pub=self.t, held=self.holding, t_ack=None, t_release=None)
        self.rep_extra['plan_versions'].append(v)
        if self.holding:
            self.takeover = v                                             # 保持中：等 ack 才解除（先接管再放手）
        return v

    def _try_reuse(self):
        """原路剩餘段可否從當下位姿接回；回傳 (Q, j_join) 或 (None, 原因)。"""
        if self.P is None:
            return None, 'no_path'
        x, y, _ = self.pose
        rp = np.array([x, y])
        lo = max(self.k_proj - 20, 0)
        d = np.linalg.norm(self.P[lo:] - rp, axis=1)
        j = lo + int(np.argmin(d))
        jj = min(j + REUSE_AHEAD, len(self.P) - 1)
        seg = self.P[jj] - rp
        L = float(np.linalg.norm(seg))
        if L > REUSE_MAX:
            return None, f'join_far {L:.2f}'
        if jj + 1 < len(self.P) and L > 1e-3:
            tng = self.P[min(jj + 3, len(self.P) - 1)] - self.P[jj]
            ang = math.degrees(abs(math.atan2(seg[0] * tng[1] - seg[1] * tng[0], seg @ tng)))
            if ang > REUSE_TURN_DEG:
                return None, f'turn {ang:.0f}'
        pts = [rp + (self.P[jj] - rp) * a_ for a_ in np.linspace(0.0, 1.0, max(2, int(L / 0.025) + 1))] + list(self.P[jj:])
        if any(self._blocked_static(p_) for p_ in pts):
            return None, 'blocked'
        return np.vstack([rp[None, :], self.P[jj:]]), jj

    def _publish_path(self, P, yaws):
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        for (vx, vy), yw in zip(P, yaws):
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = float(vx), float(vy)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
            p.poses.append(ps)
        self.plan_pub.publish(p)
        return p

    def _apply_conflict(self, res):
        mode = res['mode']
        if mode == self.cf_mode and not res.get('changed'):
            return
        h = res.get('hold') or {}
        rec = dict(t=self.t, mode=mode, frm=self.cf_mode, pose=[round(v, 4) for v in self.pose], hold=h, n_tracks=len(self.tracks), tracks_age=None if self.tracks_t is None else round(self.t - self.tracks_t, 3))
        x, y, th = self.pose
        if mode == 'WAIT':
            if h.get('i_stop', -1) >= 0:
                Q = self.P[self.k_proj:self.k_proj + int(h['i_stop']) + 1]
                Q = np.vstack([np.array([[x, y]]), Q]) if len(Q) else np.array([[x, y], h['stop']])
                yaws = [math.atan2(Q[min(i + 1, len(Q) - 1)][1] - Q[i][1], Q[min(i + 1, len(Q) - 1)][0] - Q[i][0]) if i < len(Q) - 1 else th for i in range(len(Q))]
                v_ = self._version(self._publish_path(Q, yaws), 'truncated', mode)
                rec['plan'] = dict(kind='truncated', n=len(Q), end=[round(float(v), 3) for v in Q[-1]], stamp=v_['stamp'], held=v_['held'])
            else:
                self._set_hold(True, 'stop_now')
                rec['plan'] = dict(kind='active_hold_stop_here')
        elif mode == 'EXIT':
            g = np.array(h['goal'], float)
            n = max(2, int(math.ceil(math.hypot(g[0] - x, g[1] - y) / DS)) + 1)
            Q = np.c_[np.linspace(x, g[0], n), np.linspace(y, g[1], n)]
            v_ = self._version(self._publish_path(Q, [th] * len(Q)), 'exit_line', mode)
            rec['plan'] = dict(kind='exit_line', n=len(Q), end=[round(float(v), 3) for v in g], stamp=v_['stamp'], held=v_['held'])
        elif mode in ('STOP_BLOCKED', 'HOLD_STALE'):
            self._set_hold(True, mode.lower())
            rec['plan'] = dict(kind='active_hold_' + mode.lower(), why=res.get('why') or res.get('reason'), note='主動保持（GMPC＋CBF 仍求解）；停止不保證行人不撞上來')
        elif mode in ('CRUISE', 'COMMIT'):
            if self.cf_mode in ('WAIT', 'EXIT', 'STOP_BLOCKED', 'HOLD_STALE', 'WAIT_TIMEOUT'):
                Q, info = self._try_reuse()
                if Q is not None:
                    self.k_proj = int(info)
                    yaws = [math.atan2(Q[min(i + HEAD_WIN, len(Q) - 1)][1] - Q[i][1], Q[min(i + HEAD_WIN, len(Q) - 1)][0] - Q[i][0]) if i < len(Q) - 1 else self.goal[2] for i in range(len(Q))]
                    v_ = self._version(self._publish_path(Q, yaws), 'reuse_reconnect', mode)
                    rec['plan'] = dict(kind='reuse_reconnect', n=len(Q), stamp=v_['stamp'], held=v_['held'])
                elif self.pr.pending is None:
                    self._request('resume')
                    rec['plan'] = dict(kind='replan_requested', reuse_rejected=info, held=self.holding)
                else:
                    rec['plan'] = dict(kind='replan_already_pending', reuse_rejected=info)
            else:
                rec['plan'] = dict(kind='unchanged')
        elif mode == 'WAIT_TIMEOUT':
            self._set_hold(True, 'wait_timeout')
            rec['plan'] = dict(kind='active_hold_wait_timeout')
        self.rep_extra['conflict'].append(rec)
        print(f"[m8nav] 衝突層 {self.cf_mode} → {mode} @ sim {self.t:.2f} {rec.get('plan')}", flush=True)
        self.cf_mode = mode

    def _set_hold(self, act, why=''):
        if act != self.holding:
            self.holding = act
            self.hold_log.append(dict(t=self.t, active=act, why=why, pose=[round(v, 4) for v in self.pose] if self.pose else None))
            print(f"[m8nav] 主動保持 {'開' if act else '關'} @ sim {self.t:.2f}（{why}）", flush=True)
        self.hold_pub.publish(String(data=json.dumps(dict(active=bool(act), id=len(self.hold_log)))))
        self.hold_t = self.t

    def _hdr(self):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = 'odom'
        h.stamp = self.get_clock().now().to_msg()
        return h

    def _scan(self, m):
        s = self.rep['scan']
        s['n'] += 1
        s['frame_id'] = m.header.frame_id
        s['last_wall'] = time.monotonic()

    # ---- 計畫
    def _request(self, kind):
        x, y, th = self.pose
        rid = self.pr.request((x, y, th), (self.goal[0], self.goal[1]), self.t)
        self.req_kind = kind
        self.rep['plans'].append(dict(id=rid, kind=kind, t=self.t, start=[round(x, 4), round(y, 4), round(th, 4)], grid_n=self.pr.n_grid,
                                      grid_age=None if self.pr.grid is None else round(self.t - self.pr.grid.stamp, 3)))

    def _accept(self, res):
        rec = self.rep['plans'][-1]
        rec.update(ok=bool(res.get('ok')), status=res.get('status'), wall_ms=round(res.get('wall_ms', float('nan')), 1), n_raw=int(len(res.get('P', []))))
        if not res.get('ok'):
            rec['why'] = res.get('why') or res.get('error_msg')
            return False
        ok, why = NP.validate_plan(res['frame'], res['P'], np.asarray(self.pose[:2], float), np.asarray(self.goal[:2], float), self.pr.grid)
        rec['valid'], rec['why'] = bool(ok), why or None
        if not ok:
            return False
        P = NP.resample(res['P'], DS)
        P[-1] = self.goal[:2]
        self.P, self.k_proj = P, 0
        rec.update(L=round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 3), n=len(P), cost_max=int(self.pr.grid.cost(P).max()), pts=P[::5].round(3).tolist())
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        for i, (vx, vy) in enumerate(P):
            if i < len(P) - 1:
                j = min(i + HEAD_WIN, len(P) - 1)
                yw = math.atan2(P[j][1] - vy, P[j][0] - vx)
            else:
                yw = self.goal[2]
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = float(vx), float(vy)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
            p.poses.append(ps)
        self.plan_msg, self.t_plan = p, self.t
        self.plan_pub.publish(p)
        self._version(p, 'smac_' + str(self.req_kind), self.cf_mode)
        self.t_block0, self.block_info = None, None
        return True

    def _proj(self):
        if self.P is None:
            return
        d = np.linalg.norm(self.P[self.k_proj:min(self.k_proj + 40, len(self.P))] - np.asarray(self.pose[:2]), axis=1)
        if len(d):
            self.k_proj += int(np.argmin(d))

    def _blocked_now(self):
        """剩餘路徑（投影點之後）被擋的第一點；代價圖沒更新就沿用上次判斷。"""
        if self.P is None or self.pr.grid is None:
            return None
        d = np.linalg.norm(self.P[self.k_proj:min(self.k_proj + 40, len(self.P))] - np.asarray(self.pose[:2]), axis=1)
        self.k_proj += int(np.argmin(d)) if len(d) else 0
        R = self.P[self.k_proj:]
        if len(R) == 0:
            return None
        c = self.pr.grid.cost(R)
        b = NP.blocked(c)
        if not b.any():
            return None
        k = int(np.argmax(b))
        return dict(at=R[k].round(3).tolist(), cost=int(c[k]), ahead_m=round(k * DS, 2))

    def _go(self, seg, **kw):
        self.seg = seg
        self.rep['segments'].append(dict(seg=seg, t=self.t, pose=[round(v, 4) for v in self.pose] if self.pose else None, **kw))
        print(f'[m8nav] → {seg} @ sim {self.t:.2f} pose {None if self.pose is None else [round(v, 3) for v in self.pose]} {kw if kw else ""}', flush=True)

    def _finish(self, result, seg='ZERO', **kw):
        self.rep['result'] = result
        self._go(seg, result=result, **kw)

    # ---- 主迴圈
    def _tick(self):
        a = self.a
        if self.pose is None or self.t is None:
            return
        self.pr.poll()
        gx, gy, gyaw = self.goal
        x, y, th = self.pose
        dist, eyaw = math.hypot(gx - x, gy - y), wrap(gyaw - th)
        if self.collision is not None and self.seg not in ('ZERO',) and self.rep['result'] is None:
            return self._finish('collision', collision=self.collision)
        if self.seg == 'WAIT':
            sc = self.rep['scan']
            if sc['last_wall'] is None or time.monotonic() - sc['last_wall'] > 1.0 or not sc['frame_id'] or self.pr.grid is None:
                return
            if self.pr.n_grid < 2:                                              # 至少一張起跑後更新的代價圖（含掃描）
                return
            if not self.pr.ac.server_is_ready():                                # r3b：規劃 action server 未就緒不送
                return
            if int(self.pr.grid.cost(np.array([[x, y]]))[0]) < 0:              # r3b：代價圖未涵蓋機器人（地圖未到、預設小圖）
                return
            if self.gmpc_goal_tol is None:                                       # r6b：先讀 GMPC 實際到達容差（讀不到不起步）
                if self._gp_fut is None and self._gp_cli.service_is_ready():
                    from rcl_interfaces.srv import GetParameters
                    rq = GetParameters.Request()
                    rq.names = ['goal_tolerance_xy']
                    self._gp_fut = self._gp_cli.call_async(rq)
                if self._gp_fut is not None and self._gp_fut.done():
                    res_ = self._gp_fut.result()
                    vals = res_.values if res_ is not None else []
                    if vals and vals[0].type == 3:                                   # PARAMETER_DOUBLE
                        self.gmpc_goal_tol = float(vals[0].double_value)
                        self.cf.exit_reached = self.gmpc_goal_tol
                        self.rep['gmpc_goal_tol'] = dict(value=self.gmpc_goal_tol, source='/gmpc_controller get_parameters')
                    else:
                        self._gp_fut = None
                return
            if self.tracks_t is None or self.t - self.tracks_t > WL.TRACK_MAX_AGE:   # r5：感知就緒（收到過航跡且接收間隔內）才起步（m8fake_r4S_T44p0 起步即 HOLD_STALE）
                return
            self.t_start = self.t
            self.rep['start_check'] = dict(t=self.t, pose=list(self.pose), grid_n=self.pr.n_grid, grid_age=round(self.t - self.pr.grid.stamp, 3),
                                           tracks_recv_age=round(self.t - self.tracks_t, 3), n_tracks=len(self.tracks), n_confirmed=sum(1 for x_ in self.tracks if x_.get('confirmed')),
                                           note='tracks_recv_age 是接收間隔（/gmpc/tracks_debug 無來源時間戳），不是來源資料年齡')
            self._request('initial')
            self._go('PLAN')
        if self.seg == 'PLAN' and self.pr.result is None and self.rep['plans'] and self.t - self.rep['plans'][-1]['t'] > T_PLAN_RETRY:   # r3b：無回覆重送
            if len(self.rep['plans']) >= N_PLAN_TRY:
                return self._finish('plan_no_reply', n=len(self.rep['plans']))
            self.pr.pending = None
            self._request('initial_retry')
        if self.seg in ('PLAN', 'CRUISE') and self.pr.result is not None:
            res, self.pr.result = self.pr.result, None
            ok = self._accept(res)
            if self.seg == 'PLAN':
                if ok:
                    self._go('CRUISE', L=self.rep['plans'][-1].get('L'))
                else:
                    self._finish('initial_plan_failed', why=self.rep['plans'][-1].get('why'))
            elif self.req_kind == 'resume':
                self.rep_extra['conflict'].append(dict(t=self.t, resume_plan_ok=bool(ok), L=self.rep['plans'][-1].get('L'), why=self.rep['plans'][-1].get('why')))
                if not ok:
                    return self._finish('resume_plan_failed', why=self.rep['plans'][-1].get('why'))
            elif not ok:
                self.rep['replans'][-1]['accepted'] = False
            else:
                self.rep['replans'][-1].update(accepted=True, t_accept=self.t, L=self.rep['plans'][-1].get('L'))
        if self.seg == 'CRUISE':
            if self.t - self.t_start > a.nav_timeout_s:
                return self._finish('nav_timeout', dist=dist)
            if self.pr.grid is not None and self.t - self.pr.grid.stamp > NP.COSTMAP_MAX_AGE:
                return self._finish('costmap_stale', age=round(self.t - self.pr.grid.stamp, 3))
            vm = self.vmeas[0] if self.vmeas else 0.0
            if self.P is not None and self.cf_mode in ('CRUISE', 'COMMIT', 'WAIT', 'EXIT', 'STOP_BLOCKED', 'HOLD_STALE'):
                self._proj()                                                     # 各模式都更新投影索引（WAIT／EXIT 時機器人仍在動）
                age = None if self.tracks_t is None else self.t - self.tracks_t
                res = self.cf.update(self.t, self.pose, self.vworld, self.P[self.k_proj:], self.tracks, age)
                self._apply_conflict(res)
                hh = self.cf.hold or {}
                if self.cf_mode == 'EXIT' and hh.get('goal') is not None and not self.holding and math.hypot(hh['goal'][0] - self.pose[0], hh['goal'][1] - self.pose[1]) <= self.gmpc_goal_tol:
                    self._set_hold(True, 'exit_executed')                          # r6b：離開線執行到位 ⇒ 主動保持，交時窗層以實測位置重評
                if self.cf_mode == 'WAIT' and hh.get('kind') == 'wait' and not self.holding and math.hypot(hh['stop'][0] - self.pose[0], hh['stop'][1] - self.pose[1]) < 0.25:
                    self._set_hold(True, 'wait_at_stop')
                if self.holding and self.hold_t is not None and self.t - self.hold_t >= 1.0:
                    self._set_hold(True, 'refresh')
                if self.cf_mode == 'WAIT_TIMEOUT':
                    return self._finish('wait_timeout', waited=WL.T_WAIT_MAX)
            if self.cf_mode not in ('CRUISE', 'COMMIT'):
                b = None
                self.t_block0 = None
            else:
                b = self._blocked_now()
            if self.cf_mode != 'CRUISE':
                pass
            elif b is None:
                self.t_block0, self.block_info = None, None
            elif self.t_block0 is None:
                self.t_block0, self.block_info = self.t, b
            elif self.t - self.t_block0 >= T_BLOCK and self.pr.pending is None:
                if self.n_replan >= N_REPLAN:
                    return self._finish('replan_exceeded', n=self.n_replan, block=b)
                self.n_replan += 1
                self.rep['replans'].append(dict(n=self.n_replan, t=self.t, blocked_since=self.t_block0, first=self.block_info, now=b))
                self._request('replan')
                self.t_block0 = None
            if dist <= a.align_r and self.cf_mode in ('CRUISE', 'COMMIT'):
                self.t_align = self.t
                self.u_align = np.array(self.out_last, float)               # r2：從上一筆輸出接續（r1 歸零會造成命令跳變）
                self._go('ALIGN', dist=round(dist, 4), yaw_err_deg=round(math.degrees(eyaw), 2), vmeas=self.vmeas)
        if self.seg == 'ALIGN':
            if dist <= a.tol_pos and abs(eyaw) <= math.radians(a.tol_yaw_deg):
                self.rep['arrive'] = dict(t=self.t, dist=dist, yaw_err_deg=math.degrees(eyaw), t_nav=self.t - self.t_start, n_replan=self.n_replan)
                return self._finish('done', dist=round(dist, 4), yaw_err_deg=round(math.degrees(eyaw), 2))
            if self.t - self.t_align > a.align_timeout_s:
                return self._finish('align_timeout', dist=dist, yaw_err_deg=math.degrees(eyaw))
        if self.seg == 'CRUISE':
            u = self.cin if (self.cin_t is not None and self.t - self.cin_t <= 0.5) else (0.0, 0.0, 0.0)
        elif self.seg == 'ALIGN':
            c_, s_ = math.cos(th), math.sin(th)
            bx, by = c_ * (gx - x) + s_ * (gy - y), -s_ * (gx - x) + c_ * (gy - y)
            vb = np.array([bx, by]) * a.k_align
            n = float(np.linalg.norm(vb))
            if n > a.v_align:
                vb *= a.v_align / n
            w = float(np.clip(a.k_align * eyaw, -a.w_align, a.w_align))
            tgt = np.array([vb[0], vb[1], w])
            dmax = np.array([a.a_lin, a.a_lin, a.a_ang]) / a.rate
            self.u_align = self.u_align + np.clip(tgt - self.u_align, -dmax, dmax)
            u = tuple(float(v) for v in self.u_align)
        else:
            u = (0.0, 0.0, 0.0)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = u
        self.out_pub.publish(tw)
        self.out_last = u
        key = self.seg if self.seg in self.rep['n_out'] else 'ZERO'
        self.rep['n_out'][key] += 1

    def _pub_state(self):
        if self.pose is None:
            return
        d = dict(seg=self.seg, t=self.t, result=self.rep['result'], n_replan=self.n_replan, dist=None if self.pose is None else math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]))
        self.state_pub.publish(String(data=json.dumps(d)))

    def save(self):
        self.rep['planner_log'] = self.pr.log[-50:]
        self.rep.update(self.rep_extra, conflict_layer_log=self.cf.log, hold_log=self.hold_log)
        json.dump(self.rep, open(self.a.out, 'w'), ensure_ascii=False, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-file', default=os.path.join(WS, '第八次進度報告素材', '資料', 'm8_scene_check.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--nav-timeout-s', type=float, default=240.0)
    ap.add_argument('--align-r', type=float, default=0.25)
    ap.add_argument('--tol-pos', type=float, default=0.10)
    ap.add_argument('--tol-yaw-deg', type=float, default=5.0)
    ap.add_argument('--align-timeout-s', type=float, default=30.0)
    ap.add_argument('--k-align', type=float, default=0.8)
    ap.add_argument('--v-align', type=float, default=0.06)
    ap.add_argument('--w-align', type=float, default=0.2)
    ap.add_argument('--a-lin', type=float, default=0.5)
    ap.add_argument('--a-ang', type=float, default=2.0)
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = M8Nav(a)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    try:
        while rclpy.ok() and not stop['v'] and nd.seg != 'EXIT':
            rclpy.spin_once(nd, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        nd.save()
        nd.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
