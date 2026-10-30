#!/usr/bin/env python3
"""返程門操作任務節點（規格 results/specs/door_return_phase3_r1.md；Codex m6e）。單一節點、順向序號；--seg A＝推開穿越（bar_in）、B＝推關（bar_out）。
不用 W2 模式表、不倒放模式機：抓放與推動的准入在執行端 door_return_gates，本節點只請求並等確認；同時跑 GeometryMonitor（登錄 ds_err），stop ⇒ 通知執行端中止。
流程：等滾動進場（owner wholebody、底盤與手臂到位）→ solver_start → 由實測 TCP 橋接到首列 → 操作段逐列發 TCP 目標＋協同 → 交回導航（to nav，執行端 R2 包絡准入）→ 結束。
門角差：δ＝首列名目門角 − 當趟實測，繞鉸鏈剛體轉 δ，隨操作段推進衰減到 0。任一中止不重試、不補跑。
    python3 -u evaluation/door_return_task_node.py --ref ... --seg A --op 1 --out ...
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
import motm_coord as MC                                # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402


WD_S = 1.0                       # 狀態／時鐘看門狗（牆鐘，只認前進）
HEAD_TOL = 0.35                  # 弦方向與本段計畫首段方向容差（rad）
NAV_READY_V = 0.01               # 導航備妥的最小命令速度（m/s）
LAG_S = 0.2                      # 狀態相對 /clock 落後或超前的界線（模擬時間，同 GeomMonitorConfig.age_max_s）


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def log_so3(R):
    th = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot(x, d):
    c, s = math.cos(d), math.sin(d)
    y = np.array(x, float)
    y[0], y[1], y[2] = c * x[0] - s * x[1], s * x[0] + c * x[1], x[2] + d
    return y


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def seg_dist(p, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = float(np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * (b - a))))


def nav_clear(xy, q_deg):
    """底盤中心到門板（鉸鏈原點、寬 1.0、向 −y 開）與門洞兩側牆的最小距離（m）。"""
    q = math.radians(q_deg)
    return min(seg_dist(xy, (0.0, 0.0), (math.cos(q), -math.sin(q))), seg_dist(xy, (1.0, 0.0), (6.0, 0.0)), seg_dist(xy, (-6.0, 0.0), (0.0, 0.0)))


def plan_msg(xy, wps):
    """同 nav_goal_plan_node：點距 ≤5 cm 的折線計畫，起點為目前位置。"""
    w = [tuple(float(v) for v in xy)] + [tuple(float(v) for v in p.split(',')) for p in wps.split(';')]
    p = Path()
    p.header.frame_id = 'odom'
    for (x0, y0), (x1, y1) in zip(w[:-1], w[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.05)))
        h = math.atan2(y1 - y0, x1 - x0)
        for k in range(n + 1):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x, ps.pose.position.y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(h / 2), math.cos(h / 2)
            p.poses.append(ps)
    return p


class Nd(Node):
    def __init__(self):
        super().__init__('door_return_task')
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.req_pub = self.create_publisher(String, '/return/req', 50)
        self.tgt_pub = self.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10)
        self.coord_pub = self.create_publisher(Float64MultiArray, '/wgmpc/coord', 10)
        self.start_pub = self.create_publisher(Bool, '/drawer/solver_start', lat)
        self.stop_pub = self.create_publisher(String, '/wgmpc/stop', 10)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        self.hold_pub = self.create_publisher(Bool, '/return/offset_hold', lat)
        self.plan_pub = self.create_publisher(Path, '/plan', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.plan_h, self.n_adopt, self.n_cmd = None, 0, 0
        self.create_subscription(Twist, '/cmd_vel', self._nav_cmd, 10)
        self.create_subscription(Vector3Stamped, '/gmpc/heading', self._nav_head, 10)
        self.hold_ack = None
        self.create_subscription(String, '/return/offset_hold_ack', lambda m: setattr(self, 'hold_ack', json.loads(m.data)), 10)
        self.sync, self.state, self.hs, self.n_ws = None, None, None, 0
        self.sync_wall, self.clk_t, self.clk_wall = None, None, None
        self.create_subscription(Clock, '/clock', self._clk, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Float64MultiArray, '/return/sync', self._sync, 10)
        self.create_subscription(String, '/return/state', lambda m: setattr(self, 'state', json.loads(m.data)), 10)
        self.create_subscription(String, '/handover/state', lambda m: setattr(self, 'hs', json.loads(m.data)), 10)
        self.create_subscription(Float64MultiArray, '/wgmpc/status', self._ws, 10)

    def _sync(self, m):
        d = np.asarray(m.data, float)
        if self.sync is not None and not (d[1] > self.sync[1] and d[0] > self.sync[0]):
            return                                                          # 重複舊狀態不刷新
        self.sync, self.sync_wall = d, time.monotonic()

    def _nav_head(self, m):                                                     # GMPC 弦方向對準本段計畫＝已採用；只在發計畫後計
        if self.plan_h is not None and abs(wrap(m.vector.x - self.plan_h)) <= HEAD_TOL:
            self.n_adopt += 1

    def _nav_cmd(self, m):                                                      # 採用確認後才逐筆計新收到的非零命令（不重複計快取）
        if self.n_adopt >= 3:
            self.n_cmd = self.n_cmd + 1 if math.hypot(m.linear.x, m.linear.y) >= NAV_READY_V else 0

    def _clk(self, m):
        t = float(m.clock.sec) + 1e-9 * float(m.clock.nanosec)
        if self.clk_t is None or t > self.clk_t:
            self.clk_t, self.clk_wall = t, time.monotonic()

    def _ws(self, m):
        self.n_ws += 1

    def req(self, **kw):
        self.req_pub.publish(String(data=json.dumps(kw, default=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--seg', required=True, choices=['A', 'B'])
    ap.add_argument('--op', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bridge-s', type=float, default=4.0)
    ap.add_argument('--end-row', type=int, default=-1)                   # 末段只走到此列（列號相對該段；rt_21 基準 S8 第 800 列起距牆／門板 ≥0.69 m）
    ap.add_argument('--end-skip', type=int, default=1)                   # 末段每拍進幾列（無接觸縮時）
    ap.add_argument('--nav-wp', default='')                              # 交權前先發本段導航計畫，確認導航有非零命令才請求交棒
    ap.add_argument('--nav-ready-s', type=float, default=15.0)
    ap.add_argument('--nav-entry-clear', type=float, default=0.69)       # 實測入口：底盤中心距門板／牆（GMPC 動態 margin）
    ap.add_argument('--nav-arm-tol', type=float, default=0.02)
    ap.add_argument('--pre-nav-decel-s', type=float, default=1.5)        # 末列後沿參考有界減速，停穩才發計畫
    ap.add_argument('--tcp-backoff-m', type=float, default=0.0)
    ap.add_argument('--rel-backoff-v', type=float, default=0.008)        # 放手段 TCP 沿工具軸主動退（放手後門漂移朝掌面：rt_12、rt_16）
    ap.add_argument('--rel-backoff-max', type=float, default=0.012)
    ap.add_argument('--rel-backoff-delay-s', type=float, default=0.2)
    ap.add_argument('--bridge-speed', type=float, default=0.015)         # 橋接時長下限＝入口位置差／此值
    ap.add_argument('--bridge-k', type=float, default=0.5)               # 橋接底盤 P 增益（rt_6 B：參考速度 0 時底盤不走、手臂硬伸）
    ap.add_argument('--bridge-v-max', type=float, default=0.03)
    ap.add_argument('--bridge-w-max', type=float, default=0.10)
    ap.add_argument('--bridge-base-tol', type=float, default=0.02)
    ap.add_argument('--bridge-yaw-tol', type=float, default=0.03)
    ap.add_argument('--bridge-w-vref', type=float, default=1.0)
    ap.add_argument('--entry-timeout-s', type=float, default=240.0)
    ap.add_argument('--park-tol', type=float, default=0.20)
    ap.add_argument('--arm-tol', type=float, default=0.03)
    ap.add_argument('--door-track-deg', type=float, default=2.0)
    ap.add_argument('--gate-timeout-s', type=float, default=15.0)
    ap.add_argument('--w-vref', type=float, default=0.03)                # MotM 定版
    ap.add_argument('--w-vref-pre', type=float, default=0.3)             # 橋接（軟啟動）
    ap.add_argument('--w-qn', type=float, default=0.8)                   # C0 定版
    ap.add_argument('--w-a', type=float, default=0.01)
    ap.add_argument('--w-p', type=float, default=200.0)
    ap.add_argument('--w-vref-contact', type=float, default=1.0)         # 抓持接觸段：去程移動中拉關實測權重（mission_rt_3 用 MotM 權重時推門 5 Hz 極限環）
    ap.add_argument('--w-a-contact', type=float, default=0.05)
    ap.add_argument('--w-p-contact', type=float, default=50.0)
    ap.add_argument('--k-base', type=float, default=0.3)                 # 接觸段底盤位置修正（前饋之外）
    ap.add_argument('--base-fix-max', type=float, default=0.005)
    ap.add_argument('--yaw-fix-max', type=float, default=0.02)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    P = json.load(open(a.ref))
    rows, ent = P[a.seg], P['entry'][a.seg]
    bar, grp = ent['bar'], ('S', 'C')[a.seg == 'B']
    name = {'A': ('S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'), 'B': ('C2', 'C3', 'C4', 'C5', 'C6', 'C7')}[a.seg]
    X = np.array([r['x'] for r in rows], float)
    Qd = np.array([r['q_deg'] for r in rows], float)
    st = [r['stage'][:2] for r in rows]
    i_of = {n: [i for i, s in enumerate(st) if s == n] for n in name}
    if a.end_row >= 0:
        i_of[name[-1]] = i_of[name[-1]][:a.end_row + 1]
    i_end = i_of[name[-1]][-1]                                                 # 本趟實際末列（TO_NAV 目標不得追被截掉的舊終點）
    move_ = grp + '4'
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    rclpy.init()
    nd = Nd()
    rep = {'args': vars(a), 'events': [], 'result': None, 'monitor': {'n': 0, 'n_stop': 0, 'n_warn': 0, 'pred_min': {}}}

    def ev(t, **kw):
        rep['events'].append(dict(kw, t=t))
        print(f'[return] {kw} @ sim {t:.2f}', flush=True)

    def finish(result, why=None):
        rep['result'] = result
        rep['why'] = why
        nd.stop_pub.publish(String(data=f'door_return {a.seg} {result}'))
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
        print(f'[return] 結束 seg {a.seg}：{result} {why or ""}', flush=True)
        rclpy.spin_once(nd, timeout_sec=0.1)
        return 0 if result == 'DONE' else 1

    def fail(t, why):
        nd.req(kind='abort', why=why)
        return finish('ABORTED', why)

    w0 = time.monotonic()
    wait = {'t': None}
    armed = False
    phase, idx, t_tick, delta = 'ENTRY', 0, None, None
    T_b0, t_b0, bridge_s = None, None, a.bridge_s
    tgt_row = 0
    t_wait = None
    last_hb = 0.0
    fresh, n_fresh = False, [0, -1.0]
    last_arm, t_arm0, ready = 0.0, None, False
    t_rel, t_s5 = None, None
    hold, hold_noted = False, False                                            # 求解端 d̂ 保持：閉爪請求起、退出抓持止（rt_5：接觸中更新致側向力累積）
    if True:
        while rclpy.ok():
            for _ in range(8):                                                 # 先清佇列，年齡才不被未處理訊息墊高
                rclpy.spin_once(nd, timeout_sec=0.0)
            if nd.sync is None or nd.clk_t is None:
                if time.monotonic() - w0 > 60.0:
                    return finish('ABORTED', 'no_sync')
                time.sleep(0.005)
                continue
            _wa, _wc = time.monotonic() - nd.sync_wall, time.monotonic() - nd.clk_wall
            if _wa > WD_S or _wc > WD_S:                                       # 看門狗：不再送心跳 ⇒ 執行端閘門逾時
                nd.req(kind='abort', why=f'state_stale:{_wa:.2f}/{_wc:.2f}')
                return finish('ABORTED', f'state_stale:{_wa:.2f}/{_wc:.2f}')
            _lag = float(nd.clk_t) - float(nd.sync[0])                         # 斷流時 sync 不更新也要在此攔下，先於心跳
            if not fresh:                                                      # 起動：佇列舊狀態排空、連續 5 筆新狀態追上時鐘才武裝與開始監看（rt_4 首輪年齡超限）
                if abs(_lag) <= LAG_S / 4 and nd.sync[1] > n_fresh[1]:
                    n_fresh = [n_fresh[0] + 1, float(nd.sync[1])]
                elif abs(_lag) > LAG_S / 4:
                    n_fresh = [0, float(nd.sync[1])]
                fresh = n_fresh[0] >= 5
                if not fresh and time.monotonic() - w0 > 30.0:
                    return finish('ABORTED', f'state_not_fresh:{_lag:.3f}')
                if not fresh:
                    time.sleep(0.002)
                    continue
                rep['monitor']['fresh_lag_s'] = _lag
            if abs(_lag) > LAG_S:
                return fail(float(nd.sync[0]), f'state_lag:{_lag:.3f}')
            if time.monotonic() - last_hb >= 0.05:
                nd.req(kind='hb')
                last_hb = time.monotonic()
            s = nd.sync
            t, q_d = float(s[0]), math.degrees(float(s[2]))
            x_m = np.r_[s[4:7], s[7:13]]
            if nd.state and nd.state.get('abort'):
                return finish('ABORTED', f"gate:{nd.state['abort'].get('why')}")
            if t_tick is not None and t < t_tick + 0.05 - 1e-4:
                time.sleep(0.002)
                continue
            # 節拍鎖 20 Hz（rt_3 實測每列 0.06 s）：落後時跳過過期槽，同一筆狀態只處理一次、參考只進一列
            t_tick = t if t_tick is None else t_tick + 0.05 * max(1, int((t - t_tick + 1e-4) / 0.05))
            if not armed:                                                      # 重送到閘門回報本 op（rt_10：單發遺失）
                if time.monotonic() - last_arm >= 0.2:
                    nd.req(kind='arm', op=a.op, bar=bar)
                    last_arm = time.monotonic()
                if (nd.state or {}).get('op') == a.op:
                    armed = True
                    ev(t, armed=bar, q_door=q_d)
                elif t_arm0 is None:
                    t_arm0 = time.monotonic()
                elif time.monotonic() - t_arm0 > 10.0:
                    return fail(t, 'arm_not_acked')
            r = rows[tgt_row]
            _age = max(0.0, float(nd.clk_t) - t)                               # 模擬時間年齡（/clock − 狀態時戳）
            dec, det = mon.step(t, x_m, float(s[2]), float(s[15]), _age, r['target'] if phase != 'ENTRY' else None)
            rep['monitor']['n'] += 1
            rep['monitor']['age_max'] = max(rep['monitor'].get('age_max', 0.0), _age)
            rep['monitor']['n_warn'] += dec == 'warn'
            for c_, v_ in (det.get('classes') or {}).items():
                rep['monitor']['pred_min'][c_] = min(rep['monitor']['pred_min'].get(c_, 9.0), float(v_['d_pred']))
            if dec == 'stop':
                rep['monitor']['n_stop'] += 1
                rep['monitor']['stop_detail'] = det
                return fail(t, f"geometry:{det.get('why')}")
            if armed and not ready:
                ready = True
                print('[return] READY 已武裝、狀態新鮮、幾何監看通過', flush=True)

            nd.hold_pub.publish(Bool(data=hold))

            def xref(i):
                f = float(np.clip(np.interp(i, [i_of[move_][0], i_of[move_][-1]], [0.0, 1.0]), 0, 1))
                return rot(X[i], (delta or 0.0) * (1.0 - f))

            def tcp_ref(i):
                T = K.fk(xref(i), 'link_tcp')
                _xb = 0.0 if t_rel is None else float(np.clip((t - t_rel) * a.rel_backoff_v, 0.0, a.rel_backoff_max))
                _f7 = float(np.clip(np.interp(i, [i_of[grp + '7'][0], i_of[grp + '7'][-1]], [1.0, 0.0]), 0.0, 1.0))  # 收臂段歸零，終點回參考收攏姿態（rt_17 B 交棒碼 51）
                T[:3, 3] -= (a.tcp_backoff_m + _xb) * _f7 * T[:3, 2]           # 沿工具軸退（rt_9：實際比參考深約 1.5 mm，掌面間隙翻過監看）
                return T

            def pub_target(T):
                nd.tgt_pub.publish(Float64MultiArray(data=[float(v) for v in np.asarray(T, float).reshape(-1)]))

            def pub_coord(i, soft=False, contact=False):
                x0, x1 = xref(max(i - 1, 0)), xref(i)
                dv = (x1 - x0) / 0.05 * (a.end_skip if st[i] == name[-1] else 1)
                dv[2] = math.atan2(math.sin(x1[2] - x0[2]), math.cos(x1[2] - x0[2])) / 0.05 * (a.end_skip if st[i] == name[-1] else 1)
                if contact:
                    dv[:2] += np.clip(a.k_base * (x1[:2] - x_m[:2]), -a.base_fix_max, a.base_fix_max)
                    dv[2] += float(np.clip(a.k_base * math.atan2(math.sin(x1[2] - x_m[2]), math.cos(x1[2] - x_m[2])), -a.yaw_fix_max, a.yaw_fix_max))
                vx, vy = MC.world_to_body(float(dv[0]), float(dv[1]), float(x_m[2]))
                if contact:
                    nd.coord_pub.publish(Float64MultiArray(data=MC.coord_msg(w_vref=a.w_vref_contact, vref=[vx, vy, float(dv[2])], w_qn=0.0, w_a=a.w_a_contact, w_p=a.w_p_contact)))
                    return
                nd.coord_pub.publish(Float64MultiArray(data=MC.coord_msg(w_vref=a.w_vref_pre if soft else a.w_vref, vref=[vx, vy, float(dv[2])], w_qn=a.w_qn, q_nom=[float(v) for v in x1[3:]],
                                                                         w_a=None if soft else a.w_a, w_p=None if soft else a.w_p)))
            if phase == 'ENTRY':
                delta = math.radians(float(Qd[0]) - q_d)
                pub_target(tcp_ref(0))
                own = str((nd.hs or {}).get('owner', '')).lower()
                bpe = float(np.hypot(*(x_m[:2] - xref(0)[:2])))
                ape = float(np.max(np.abs(x_m[3:] - X[0][3:])))
                if own.startswith('wholebody') and bpe <= a.park_tol and ape <= a.arm_tol:
                    nd.start_pub.publish(Bool(data=True))
                    if nd.n_ws > 0:
                        T_b0, t_b0 = K.fk(x_m, 'link_tcp'), t
                        bridge_s = max(a.bridge_s, bpe / a.bridge_speed)
                        phase = 'BRIDGE'
                        ev(t, to='BRIDGE', bpe=bpe, ape=ape, delta_deg=math.degrees(delta))
                if time.monotonic() - w0 > a.entry_timeout_s:
                    return fail(t, 'entry_timeout')
            elif phase == 'BRIDGE':
                Tr = tcp_ref(0)
                f = min(1.0, (t - t_b0) / bridge_s)
                f = 10 * f ** 3 - 15 * f ** 4 + 6 * f ** 5
                T = Tr.copy()
                T[:3, 3] = T_b0[:3, 3] + (Tr[:3, 3] - T_b0[:3, 3]) * f
                T[:3, :3] = T_b0[:3, :3] @ rodrigues(log_so3(T_b0[:3, :3].T @ Tr[:3, :3]) * f)    # 姿態同步內插，不直接跳到終點
                pub_target(T)
                x0 = xref(0)
                e_b, e_y = x0[:2] - x_m[:2], math.atan2(math.sin(x0[2] - x_m[2]), math.cos(x0[2] - x_m[2]))
                v_b = a.bridge_k * e_b
                n_b = float(np.linalg.norm(v_b))
                v_b = v_b * min(1.0, a.bridge_v_max / max(n_b, 1e-9))
                vx, vy = MC.world_to_body(float(v_b[0]), float(v_b[1]), float(x_m[2]))
                nd.coord_pub.publish(Float64MultiArray(data=MC.coord_msg(w_vref=a.bridge_w_vref, vref=[vx, vy, float(np.clip(a.bridge_k * e_y, -a.bridge_w_max, a.bridge_w_max))],
                                                                         w_qn=a.w_qn, q_nom=[float(v) for v in x0[3:]])))
                if t - t_b0 >= bridge_s:
                    if float(np.linalg.norm(K.fk(x_m, 'link_tcp')[:3, 3] - Tr[:3, 3])) <= 0.005 and float(np.linalg.norm(e_b)) <= a.bridge_base_tol and abs(e_y) <= a.bridge_yaw_tol:
                        phase, idx = 'RUN', 0
                        ev(t, to='RUN', bpe=float(np.linalg.norm(e_b)), yaw_e=e_y)
                    elif t - t_b0 > bridge_s + a.gate_timeout_s:
                        return fail(t, 'bridge_timeout')
            elif phase == 'RUN':
                sn = st[idx]
                last = idx == i_of[sn][-1]
                if sn == grp + '5':                                            # 放手退離：最早 delay 後，且新鮮完整接觸證據兩指皆 <0.1 N 才啟動；啟動後再受力即中止
                    t_s5 = t if t_s5 is None else t_s5
                    _st = nd.state or {}
                    _ev = abs(t - float(_st.get('sim_t', -1e9))) <= 0.1 and _st.get('ev_ok') is True
                    _fm = max(_st.get('fb') or [9.0])
                    if t_rel is None and t - t_s5 >= a.rel_backoff_delay_s and _ev and _fm < 0.1:
                        t_rel = t
                        ev(t, rel_backoff_start=True, q=q_d)
                    elif t_rel is not None and _ev and _fm >= 0.1:
                        return fail(t, f'contact_during_rel_backoff:{_fm:.3f}')
                pub_target(tcp_ref(idx))
                pub_coord(idx, contact=sn in (grp + '3', move_, grp + '5'))
                tgt_row = idx
                gs = (nd.state or {}).get('state')

                def go_next(stage_now):
                    nonlocal idx, t_wait
                    k = name.index(stage_now) + 1
                    idx = i_of[name[k]][0]
                    t_wait = None

                def waited():
                    nonlocal t_wait
                    t_wait = t if t_wait is None else t_wait
                    return t - t_wait > a.gate_timeout_s
                if sn == grp + '2':
                    if not last:
                        idx += 1
                    else:
                        hold = True
                        nd.hold_pub.publish(Bool(data=True))                   # 求解端確認凍結後才送 close
                        if not (nd.hold_ack or {}).get('hold'):
                            if waited():
                                return fail(t, 'offset_hold_not_acked')
                        else:
                            if not hold_noted:
                                hold_noted = True
                                ev(t, offset_hold=nd.hold_ack)
                            nd.req(kind='close', T=[float(v) for v in tcp_ref(idx).reshape(-1)])
                            if gs == 'CLOSING':
                                ev(t, gate=gs)
                                go_next(sn)
                            elif waited():
                                return fail(t, f'close_not_admitted:{gs}')
                elif sn == grp + '3':
                    if gs == 'GRASPED':
                        nd.req(kind='move')
                    elif gs == 'MOVING':
                        ev(t, gate=gs, q=q_d)
                        go_next(sn)
                elif sn == move_:
                    _f = float(np.clip(np.interp(idx, [i_of[move_][0], i_of[move_][-1]], [0.0, 1.0]), 0, 1))
                    _qr = float(Qd[idx]) - math.degrees((delta or 0.0) * (1.0 - _f))
                    if abs(q_d - _qr) > a.door_track_deg:
                        return fail(t, f'door_track:{q_d:.2f} vs {_qr:.2f}')
                    if not last:
                        idx += 1
                    else:
                        nd.req(kind='open')
                        if gs == 'OPENING':
                            ev(t, gate=gs, q=q_d)
                            go_next(sn)
                        elif waited():
                            return fail(t, 'door_not_settled')
                elif sn == grp + '5':
                    if gs == 'RELEASED':
                        nd.req(kind='retreat')
                    elif gs == 'RETREAT':
                        ev(t, gate=gs)
                        hold = False
                        go_next(sn)
                    elif waited():
                        return fail(t, 'release_not_confirmed')
                elif not last:
                    idx = min(idx + (a.end_skip if sn == name[-1] else 1), i_of[sn][-1])
                elif sn != name[-1]:
                    go_next(sn)
                elif a.nav_wp:
                    phase = 'PRE_NAV'
                    wait['t'], wait['ns'], wait['r'], wait['p'] = t, -1, float(i_end), None
                    ev(t, to='PRE_NAV', x=[round(float(v), 3) for v in x_m[:3]])
                else:
                    nd.req(kind='done')
                    phase = 'TO_NAV'
                    wait['t'] = t
                    ev(t, to='TO_NAV', q=q_d, x=[round(float(v), 3) for v in x_m[:3]])
            elif phase == 'PRE_NAV':                                           # 沿參考有界減速→全身保持→本段計畫送達、GMPC 採用、確有新命令、實測入口成立，才請求交棒
                _dec = max(0.0, 1.0 - (t - wait['t']) / a.pre_nav_decel_s)
                if _dec > 0.0:
                    wait['r'] = min(wait['r'] + _dec, float(len(rows) - 1))
                    i_end = int(wait['r'])
                pub_target(tcp_ref(i_end))
                x0, x1 = xref(max(i_end - 1, 0)), xref(i_end)
                _dv = (x1[:2] - x0[:2]) / 0.05 * _dec
                _vx, _vy = MC.world_to_body(float(_dv[0]), float(_dv[1]), float(x_m[2]))
                nd.coord_pub.publish(Float64MultiArray(data=MC.coord_msg(w_vref=a.w_vref, vref=[_vx, _vy, wrap(x1[2] - x0[2]) / 0.05 * _dec], w_qn=a.w_qn, q_nom=[float(v) for v in x1[3:]], w_a=a.w_a, w_p=a.w_p)))
                if _dec <= 0.0:
                    if wait['p'] is None:
                        wait['p'] = plan_msg(x_m[:2], a.nav_wp)
                        _w0 = [float(v) for v in a.nav_wp.split(';')[0].split(',')]
                        nd.plan_h = math.atan2(_w0[1] - float(x_m[1]), _w0[0] - float(x_m[0]))
                        nd.n_adopt, nd.n_cmd = 0, 0
                        ev(t, plan_sent=True, x=[round(float(v), 3) for v in x_m[:3]], h=nd.plan_h)
                    _ns = nd.plan_pub.get_subscription_count()
                    if _ns > wait['ns']:
                        nd.plan_pub.publish(wait['p'])
                        wait['ns'] = _ns
                    _cl = nav_clear(x_m[:2], q_d)
                    _ae = float(np.max(np.abs(x_m[3:] - X[i_end][3:])))
                    if nd.n_cmd >= 5 and _cl >= a.nav_entry_clear and _ae <= a.nav_arm_tol:
                        nd.req(kind='done')
                        phase = 'TO_NAV'
                        wait['t'] = t
                        ev(t, to='TO_NAV', q=q_d, x=[round(float(v), 3) for v in x_m[:3]], n_adopt=nd.n_adopt, n_cmd=nd.n_cmd, clear=_cl, arm_err=_ae)
                    elif t - wait['t'] > a.pre_nav_decel_s + a.nav_ready_s:
                        return fail(t, f'nav_entry_not_ready:adopt{nd.n_adopt}/cmd{nd.n_cmd}/clear{_cl:.3f}/arm{_ae:.3f}')
            elif phase == 'TO_NAV':
                pub_target(tcp_ref(i_end))
                own = str((nd.hs or {}).get('owner', '')).lower()
                if own == 'nav' and not (nd.hs or {}).get('pending'):
                    ev(t, owner='nav', step=(nd.hs or {}).get('step'))
                    return finish('DONE')
                if not (nd.hs or {}).get('pending') and (t - wait['t']) % 3.0 < 0.06:
                    nd.ho_pub.publish(String(data=json.dumps({'to': 'nav', 'at_step': int(s[1]) + 5, 'window_steps': 300, 'why': f'return {a.seg} done'})))
                if t - wait['t'] > 45.0:
                    return fail(t, 'to_nav_timeout')
    return finish('ABORTED', 'loop_exit')


if __name__ == '__main__':
    raise SystemExit(main())
