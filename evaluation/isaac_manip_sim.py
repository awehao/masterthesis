"""Isaac：固定底盤的手臂位置命令短測試。

本檔**不做導航**。底盤由初始化程序直接放在操作案例的精確停放位姿，並在每一步
把速度歸零保持不動；guard 以零命令持續發布 /cmd_vel，維持它是唯一發布者 ——
這是把交接後的拓樸先跑通，不是導航停放驗收。

與導航跑批的關係：**完全不動 isaac_bigarena_sim.py**。導航實驗維持凍結。

手臂命令入口（本輪新增的唯一控制介面）
    /arm/joint_position_cmd   Float64MultiArray, 6   joint1..joint6 的位置目標

位置介面的界線：這裡只把目標交給 Isaac 的關節位置驅動，**不在模擬器內積分、
不做軌跡生成**。平滑與速度上限由上游的軌跡發布端負責，且那條軌跡必須先經
check_arm_path.py 逐點檢查過。

停止條件在跑之前就設定好（不是事後判讀）：
    模擬時間上限 / 牆鐘上限
    底盤位移超過 base_drift_max（底盤必須固定）
    關節逼近安全限位
    追蹤誤差超過 track_err_max
    CPU 溫度
"""
import argparse, json, math, os, sys, time
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--headless', default='true')
ap.add_argument('--sim-limit', type=float, default=60.0, help='模擬時間上限 s')
ap.add_argument('--wall-limit', type=float, default=600.0)
ap.add_argument('--base-drift-max', type=float, default=0.010, help='底盤容許位移 m')
ap.add_argument('--base-yaw-drift-max-deg', type=float, default=1.0)
ap.add_argument('--track-err-max', type=float, default=0.20, help='關節追蹤誤差上限 rad')
ap.add_argument('--limit-margin', type=float, default=0.02, help='距關節限位的最小餘裕 rad')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--cpu-limit', type=float, default=92.0)
ap.add_argument('--cpu-threads', type=int, default=8)
# **本輪唯一的變數**：物理主迴圈的牆鐘節流。
#   0   不節流，迴圈盡快跑（先前所有手臂測試都是這樣）
#   >0  以該倍率對齊牆鐘；time.sleep 同時讓出 GIL，背景 executor 才有機會
#       處理回呼。這是診斷「接收／套用節奏」的單變數，**不預先宣稱 executor
#       就是根因**——若沒改善，就依新的時序去定位下一層。
ap.add_argument('--rtf', type=float, default=0.0,
                help='>0 時以此倍率對齊牆鐘（1.0 = 即時）')
a = ap.parse_args()

CASES = yaml.safe_load(open(a.cases))
CASE_NAME = a.case or CASES['default_case']
CASE = CASES['cases'][CASE_NAME]
POSES = yaml.safe_load(open(a.poses))
ARM = [f'joint{i}' for i in range(1, 7)]
Q_START = np.array([float(POSES[CASE['pregrasp']['start_config']][j]) for j in ARM])
Q_GOAL = np.array([float(POSES[CASE['pregrasp']['arm_config']][j]) for j in ARM])
PARK = (CASE['parking']['x'], CASE['parking']['y'],
        math.radians(CASE['parking']['yaw_deg']))
TOL = CASE['tolerance']

import hashlib
URDF_SHA = hashlib.sha256(open(a.urdf, 'rb').read()).hexdigest()[:16]
print(f'[manip] 案例 {CASE_NAME}')
print(f'[manip] URDF {a.urdf}  sha {URDF_SHA}')
print(f'[manip] 停放 ({PARK[0]:.3f}, {PARK[1]:.3f}) yaw {CASE["parking"]["yaw_deg"]:.1f}°')
print(f'[manip] 手臂 {CASE["pregrasp"]["start_config"]} -> {CASE["pregrasp"]["arm_config"]}')

from isaacsim import SimulationApp                                   # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
if a.cpu_threads > 0:
    _cfg['limit_cpu_threads'] = a.cpu_threads
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane       # noqa: E402
from isaacsim.core.api.objects import FixedCuboid                    # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402
from pxr import UsdGeom, Gf                                          # noqa: E402
sys.path.insert(0, HERE)
from isaac_common import import_urdf                                 # noqa: E402

import rclpy                                                          # noqa: E402
from rclpy.node import Node                                           # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy    # noqa: E402
from rosgraph_msgs.msg import Clock                                   # noqa: E402
from sensor_msgs.msg import JointState                                # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist                      # noqa: E402
from std_msgs.msg import Float64MultiArray, String                    # noqa: E402


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def cpu_temp_c():
    try:
        import subprocess, re
        o = subprocess.run(['sensors'], capture_output=True, text=True, timeout=3).stdout
        v = [float(x) for x in re.findall(r'\+(\d+\.\d)°C', o)]
        return max(v) if v else None
    except Exception:
        return None


class ManipNode(Node):
    def __init__(self):
        super().__init__('isaac_manip_sim')
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tcp_pub = self.create_publisher(PoseStamped, '/manip/tcp_pose', be)
        self.st_pub = self.create_publisher(String, '/manip/status', 10)
        self.cmd = None              # 尚未收到任何關節命令
        self.cmd_n = 0               # **收到**幾則
        self.cmd_t = None
        self.cmd_first_t = None
        self.applied_n = 0           # **實際套用**幾次（與收到分開記）
        self.applied_last = None
        # 回呼入口計數（診斷用）：抵達回呼就 +1，與內容是否合格無關
        self.arm_cb_n = 0
        self.arm_cb_first_t = None
        self.arm_cb_last_t = None
        self.arm_cb_rejected = 0
        self.base_cb_n = 0
        self.base_cb_first_t = None
        self.base_cb_last_t = None
        self.cmd_seq = None
        self.snap = None             # (seq, q) —— 一次指派，讀取端一次取走
        self.rx = []                 # [(seq, kind, t_sched, recv_sim_t)]
        self.sim_t = 0.0
        self.create_subscription(Float64MultiArray, '/arm/joint_position_cmd',
                                 self._cmd, 10)
        # /cmd_vel 只訂閱不使用：本測試底盤固定，訂閱是為了讓 guard 的
        # 「唯一發布者」拓樸成立且可被 endpoint_check 看到。
        self.base_cmd = [0.0, 0.0, 0.0]
        self.create_subscription(Twist, '/cmd_vel', self._base, 10)
        self.base_cmd_nonzero = 0

    def _cmd(self, m):
        # 計數在回呼**入口**，先於任何內容檢查 —— 否則長度不符的訊息會被
        # 算成「沒收到」，把「收到但被拒絕」和「訊息沒抵達」混為一談。
        self.arm_cb_n += 1
        if self.arm_cb_first_t is None:
            self.arm_cb_first_t = self.sim_t
        self.arm_cb_last_t = self.sim_t
        # [seq, kind, t_sched, q1..q6]
        if len(m.data) != 9:
            self.arm_cb_rejected += 1
            return
        seq = int(m.data[0]); kind = int(m.data[1]); ts = float(m.data[2])
        q = np.array([float(v) for v in m.data[3:9]])
        # 序號與關節目標必須是**同一份快照**。先前分別存成 self.cmd 與
        # self.cmd_seq，而物理迴圈是「先讀 cmd → apply_action → 再讀 cmd_seq」，
        # 背景回呼可能在這兩次讀取之間更新，於是記下的序號未必對應真正套用的
        # 關節值，誤差就會被歸到錯的命令上。改成一次指派一個 tuple，
        # 讀取端一次取走，兩者必然一致。
        self.snap = (seq, q)
        self.cmd = q                 # 保留供既有欄位使用
        self.cmd_seq = seq
        self.cmd_n += 1
        self.cmd_t = self.sim_t
        if self.cmd_first_t is None:
            self.cmd_first_t = self.sim_t
        # 逐則記錄：序號、類別、預定模擬時間、實際進回呼的模擬時間
        self.rx.append((seq, kind, ts, self.sim_t))

    def _base(self, m):
        # 同樣計在入口。先前只計「非零」則數，而 guard 送的是零 ——
        # 0 與「沒收到」因此分不開，這正是要消除的歧義。
        self.base_cb_n += 1
        if self.base_cb_first_t is None:
            self.base_cb_first_t = self.sim_t
        self.base_cb_last_t = self.sim_t
        self.base_cmd = [m.linear.x, m.linear.y, m.angular.z]
        if max(abs(v) for v in self.base_cmd) > 1e-9:
            self.base_cmd_nonzero += 1

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time(); s.sec = int(t); s.nanosec = int(round((t - int(t)) * 1e9))
        if s.nanosec >= 1_000_000_000:
            s.sec += 1; s.nanosec -= 1_000_000_000
        return s

    def pub_clock(self, t):
        m = Clock(); m.clock = self.stamp(t); self.clock_pub.publish(m)

    def pub_joints(self, t, names, pos):
        m = JointState(); m.header.stamp = self.stamp(t)
        m.name = list(names); m.position = [float(v) for v in pos]
        self.js_pub.publish(m)

    def pub_tcp(self, t, p, q):
        m = PoseStamped(); m.header.stamp = self.stamp(t); m.header.frame_id = 'map'
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
        m.pose.orientation.w, m.pose.orientation.x = float(q[0]), float(q[1])
        m.pose.orientation.y, m.pose.orientation.z = float(q[2]), float(q[3])
        self.tcp_pub.publish(m)

    def pub_status(self, d):
        s = String(); s.data = json.dumps(d, ensure_ascii=False); self.st_pub.publish(s)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    if abs(float(world.get_rendering_dt()) - float(a.physics_dt)) > 1e-12:
        print('[manip] rendering_dt != physics_dt，中止'); return 4
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    # 靜態方塊：只放 known_obs（本測試沒有動態障礙）
    import re
    ws = re.sub(r'<!--.*?-->', '', open(a.world).read(), flags=re.S)
    nbox = 0
    for m in re.finditer(r'<model name="(known_obs_\d+)">(.*?)</model>', ws, re.S):
        n, b = m.group(1), m.group(2)
        p = [float(v) for v in re.search(r'<pose>([^<]+)</pose>', b).group(1).split()]
        sz = [float(v) for v in re.search(r'<box>\s*<size>([^<]+)</size>', b).group(1).split()]
        FixedCuboid(prim_path=f'/World/{n}', name=n,
                    position=np.array(p[:3]), scale=np.array(sz))
        nbox += 1
    print(f'[manip] 靜態方塊 {nbox} 個')

    import_urdf(a.urdf, '/World/omni_bot')
    world.reset()
    robot = SingleArticulation(prim_path='/World/omni_bot', name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: i for i, n in enumerate(names)}
    print(f'[manip] articulation DOF {len(names)}: {names}')
    missing = [j for j in ARM if j not in idx]
    if missing:
        print(f'[manip] URDF 缺關節 {missing}，中止'); return 5

    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for k, j in enumerate(ARM):
        q[idx[j]] = Q_START[k]; kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    for j in ('finger_joint1', 'finger_joint2'):
        if j in idx:
            q[idx[j]] = 0.0; kp[idx[j]] = 1.0e4; kd[idx[j]] = 1.0e3
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    robot.set_world_pose(np.array([PARK[0], PARK[1], 0.0]), np.array(q_yaw(PARK[2])))
    for _ in range(20):
        world.step(render=False)

    p0, qq0 = robot.get_world_pose()
    print(f'[manip] 初始化後底盤 ({float(p0[0]):.4f}, {float(p0[1]):.4f}) '
          f'yaw {math.degrees(yaw_of(qq0)):.3f}°')

    stage = world.stage
    tcp_prim = None
    for pr in stage.Traverse():
        if pr.GetName() == 'link_tcp':
            tcp_prim = pr; break
    print(f'[manip] link_tcp prim: {tcp_prim.GetPath() if tcp_prim else "**找不到**"}')
    if tcp_prim is None:
        return 6

    def tcp_world():
        M = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(0)
        t = M.ExtractTranslation()
        r = M.ExtractRotationQuat()
        i = r.GetImaginary()
        return (np.array([t[0], t[1], t[2]]),
                np.array([r.GetReal(), i[0], i[1], i[2]]))

    # 節點在背景執行緒用 SingleThreadedExecutor 轉 —— 與 isaac_bigarena_sim.py
    # 相同的寫法。第一版把 spin_once(timeout_sec=0.0) 放在物理迴圈裡，結果整個
    # 參與者在同 domain 的 `ros2 topic list` 裡完全看不到（連 /clock 都沒有），
    # 播放端因此永遠等不到 /arm/joint_position_cmd 的訂閱者。
    import threading
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = ManipNode()
    _ex = SingleThreadedExecutor()
    _ex.add_node(node)
    threading.Thread(target=_ex.spin, daemon=True).start()
    log, stop_reason = [], 'sim_limit'
    t = 0.0
    t_wall0 = time.monotonic()
    lim_lo = np.full(6, -np.inf); lim_hi = np.full(6, np.inf)
    try:
        dl = robot.dof_properties
        for k, j in enumerate(ARM):
            lim_lo[k] = float(dl['lower'][idx[j]]); lim_hi[k] = float(dl['upper'][idx[j]])
    except Exception:
        pass

    applied_seqs = set()
    nxt = time.monotonic()
    print(f'[manip] 牆鐘節流 rtf={a.rtf}'
          + ('（不節流）' if a.rtf <= 0 else f'（每步睡到 {a.physics_dt/a.rtf*1000:.1f} ms）'))
    print('[manip] 進入主迴圈；等待 /arm/joint_position_cmd', flush=True)
    while True:
        applied_seq = None
        node.sim_t = t
        tgt = q.copy()
        snap = node.snap             # 一次取走：序號與關節目標同源
        if snap is not None:
            applied_seq, q_cmd = snap
            for k, j in enumerate(ARM):
                tgt[idx[j]] = float(q_cmd[k])
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))
            node.applied_n += 1
            node.applied_last = [float(tgt[idx[j]]) for j in ARM]
            applied_seqs.add(applied_seq)
        # 底盤固定：每步歸零，並檢查它真的沒動
        robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        world.step(render=False)
        t = float(world.current_time)
        node.pub_clock(t)

        jp = robot.get_joint_positions()
        qa = np.array([float(jp[idx[j]]) for j in ARM])
        pb, qb = robot.get_world_pose()
        drift = math.dist((float(pb[0]), float(pb[1])), (PARK[0], PARK[1]))
        dyaw = abs(math.degrees((yaw_of(qb) - PARK[2] + math.pi) % (2 * math.pi) - math.pi))
        tp, tq = tcp_world()
        node.pub_joints(t, names, jp)
        node.pub_tcp(t, tp, tq)

        cmd_v = snap[1] if snap is not None else qa
        err = float(np.max(np.abs(qa - cmd_v)))
        log.append(dict(t=t, q=[float(v) for v in qa],
                        cmd=[float(v) for v in cmd_v], track_err=err,
                        applied_seq=applied_seq,
                        tcp=[float(v) for v in tp],
                        tcp_quat=[float(v) for v in tq],
                        base_drift=drift, base_yaw_drift_deg=dyaw))
        if len(log) % 50 == 0:
            node.pub_status(dict(t=t, cmd_recv=node.cmd_n, cmd_applied=node.applied_n,
                                 track_err=err, base_drift=drift,
                                 tcp=[float(v) for v in tp]))

        if drift > a.base_drift_max or dyaw > a.base_yaw_drift_max_deg:
            stop_reason = 'base_moved'; break
        if node.cmd is not None and err > a.track_err_max:
            stop_reason = 'track_error'; break
        near = float(np.min(np.minimum(qa - lim_lo, lim_hi - qa)))
        if np.isfinite(near) and near < a.limit_margin:
            stop_reason = 'joint_limit'; break
        if t >= a.sim_limit:
            stop_reason = 'sim_limit'; break
        if time.monotonic() - t_wall0 > a.wall_limit:
            stop_reason = 'wall_limit'; break
        if len(log) % 200 == 0 and a.cpu_limit > 0:
            c = cpu_temp_c()
            if c is not None and c >= a.cpu_limit:
                stop_reason = 'thermal_abort'; break
        if a.rtf > 0:
            nxt += a.physics_dt / a.rtf
            sl = nxt - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.monotonic()

    tp, tq = tcp_world()
    pb, qb = robot.get_world_pose()
    jp = robot.get_joint_positions()
    qa = np.array([float(jp[idx[j]]) for j in ARM])
    res = dict(
        schema='manip_run/1', case=CASE_NAME, urdf=a.urdf, urdf_sha=URDF_SHA,
        stop_reason=stop_reason, sim_time=t, wall_time=time.monotonic() - t_wall0,
        physics_dt=a.physics_dt, rendering_dt=float(world.get_rendering_dt()),
        rtf_throttle=a.rtf,
        parking_target=dict(x=PARK[0], y=PARK[1], yaw_deg=CASE['parking']['yaw_deg']),
        base_final=dict(x=float(pb[0]), y=float(pb[1]),
                        yaw_deg=math.degrees(yaw_of(qb)),
                        drift_m=math.dist((float(pb[0]), float(pb[1])), (PARK[0], PARK[1]))),
        arm_start=[float(v) for v in Q_START], arm_goal=[float(v) for v in Q_GOAL],
        arm_final=[float(v) for v in qa],
        arm_final_err=[float(v) for v in (qa - Q_GOAL)],
        arm_final_err_max=float(np.max(np.abs(qa - Q_GOAL))),
        tcp_final_world=[float(v) for v in tp],
        tcp_final_quat_wxyz=[float(v) for v in tq],
        # 命令交付分三層記錄，任何一層都不由另一層推論
        delivery=dict(received_msgs=node.cmd_n,
                      applied_actions=node.applied_n,
                      first_recv_sim_t=node.cmd_first_t,
                      last_recv_sim_t=node.cmd_t,
                      last_applied=node.applied_last),
        # 回呼入口計數：兩個受測 topic 分開記，先於內容檢查
        # 收到 vs 實際被套用：兩個物理步之間收到多則時，較早的會被覆寫、
        # 從未套用。這兩個集合的差就是「收到但沒執行」的序號。
        seq_trace=dict(received=[list(r) for r in node.rx],
                       applied_seqs=sorted(applied_seqs),
                       received_n=len(node.rx),
                       applied_unique_n=len(applied_seqs),
                       received_not_applied=sorted(
                           {r[0] for r in node.rx} - applied_seqs)),
        callbacks=dict(
            arm=dict(topic='/arm/joint_position_cmd', entered=node.arm_cb_n,
                     rejected_bad_len=node.arm_cb_rejected,
                     first_sim_t=node.arm_cb_first_t, last_sim_t=node.arm_cb_last_t),
            base=dict(topic='/cmd_vel', entered=node.base_cb_n,
                      nonzero=node.base_cmd_nonzero,
                      first_sim_t=node.base_cb_first_t, last_sim_t=node.base_cb_last_t)),
        cmd_msgs=node.cmd_n,
        base_cmd_nonzero=node.base_cmd_nonzero,
        dof_names=names, samples=len(log), log=log)
    json.dump(res, open(a.out, 'w'), ensure_ascii=False)
    print(f'[manip] 結束 stop_reason={stop_reason} sim={t:.2f}s '
          f'關節最終誤差 max {res["arm_final_err_max"]*1000:.3f} mrad '
          f'底盤位移 {res["base_final"]["drift_m"]*1000:.3f} mm')
    print(f'[manip] TCP 世界座標 ({tp[0]:.4f}, {tp[1]:.4f}, {tp[2]:.4f})')
    print(f'[manip] 命令交付：收到 {node.cmd_n} 則，實際套用 {node.applied_n} 次'
          f'（首則 sim {node.cmd_first_t}）')
    print(f'[manip] 序號：收到 {len(node.rx)} 則，實際套用過的相異序號 '
          f'{len(applied_seqs)} 個，收到但從未套用 '
          f'{len({r[0] for r in node.rx} - applied_seqs)} 個')
    print(f'[manip] 回呼入口計數：/arm/joint_position_cmd {node.arm_cb_n} 則'
          f'（拒絕 {node.arm_cb_rejected}）；/cmd_vel {node.base_cb_n} 則'
          f'（其中非零 {node.base_cmd_nonzero}）')
    print(f'[manip] -> {a.out}')
    _ex.shutdown(); node.destroy_node(); rclpy.shutdown(); sim_app.close()
    return 0


sys.exit(main())
