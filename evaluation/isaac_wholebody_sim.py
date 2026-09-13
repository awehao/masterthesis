"""Isaac 全身執行端：底盤與手臂由**同一筆** 9 維命令驅動。

定位
----
把已在 **Gazebo** 驗收過的全身同動能力（`wholebody_pregrasp_20260908.md`）
接到 **Isaac 執行端**。**不重新研究控制器**，上游沿用既有的
`wholebody_pregrasp.py → wholebody_safety_node → arm_vel_adapter`。

既有三支**一律不動**：`isaac_bigarena_sim.py`（導航，凍結）、
`isaac_manip_sim.py`（預抓取）、`isaac_drawer_sim.py`（抽屜）。

介面
----
    /wb_vel_cmd   Float64MultiArray, 9   [vx_body, vy_body, wz, dq1..dq6]

**只訂閱這一個 topic**。底盤與手臂分量來自同一次全身求解，
分開訂閱兩個 topic 取最新值**不算同步**，本檔不提供那條路徑。

不啟動 `arm_vel_gate` 與 `wheel_limit_guard` 兩個節點，
但**它們的必要檢查沒有省略** —— 全部由 `wb_cmd_chain.CmdChain` 承接，
並已由 `evaluation/test_wb_cmd_chain.py` 不開模擬器逐項測試。

手臂致動轉接層
--------------
Isaac 手臂是位置驅動，上游給的是關節速度，因此需要積分。
**這不是原生速度控制**：積分用 `world.current_time` 的相鄰物理步差、
初始設定點由實測關節位置建立、過期則停止積分並凍結設定點。
**凍結設定點只代表停止積分，不代表手臂實際速度瞬間為零** ——
停止行為由本檔逐步記錄的實測關節速度另行判定。

規格：`evaluation/results/specs/isaac_wholebody_port_v2.md`

用法（介面測試）：
    $ISAAC_PY evaluation/isaac_wholebody_sim.py --out <dir> --mode base
"""
import argparse, json, math, os, sys, threading, time
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--headless', default='true')
ap.add_argument('--sim-limit', type=float, default=30.0)
ap.add_argument('--rtf', type=float, default=1.0)
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
# --- 命令鏈 ---
ap.add_argument('--max-cmd-age-s', type=float, default=0.2,
                help='接收端**獨立**的逾時判定（模擬時間）')
ap.add_argument('--arm-rate-max', type=float, default=1.0,
                help='關節速度上限（rad/s）；超過即整筆失效')
# --- 輪級：正常輸出限制尚未實作，先用明確低速界限 + 越界中止 ---
ap.add_argument('--base-lin-max', type=float, default=0.05,
                help='**低速介面界限**（m/s）。這不是輪級限制功能，'
                     '只是介面測試用的明確界限；越界即整筆中止，不縮命令')
ap.add_argument('--base-ang-max', type=float, default=0.20,
                help='**低速介面界限**（rad/s），同上')
ap.add_argument('--mode', default='base',
                choices=['base', 'arm', 'sync', 'pregrasp'],
                help='驗收順序：base → arm → sync → pregrasp')
ap.add_argument('--solver-label', default='dls',
                help='僅供記錄：本趟上游用的求解模式。介面煙霧測試標示 dls；'
                     '**dls 不是 B 基線**')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# 輪級**正常輸出限制**目前尚未實作。程式裡沒有這個功能，
# 就**無條件禁止** pregrasp —— 不提供任何讓使用者自行宣告已實作的旗標。
# 實作完成後，把下面這個常數改為 True 並附上實作與測試，才可解鎖。
WHEEL_LIMIT_IMPLEMENTED = False

if a.mode == 'pregrasp' and not WHEEL_LIMIT_IMPLEMENTED:
    print('[wb] **輪級正常輸出限制尚未實作，無條件禁止 pregrasp**'
          '（規格 §2.1c）。目前只有低速介面界限，沒有輪級限制功能。')
    sys.exit(2)

import yaml                                                    # noqa: E402
from wb_cmd_chain import CmdChain                              # noqa: E402
from cpu_temp import read as cpu_temp_read                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE           # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
ROBOT = '/World/omni_bot'

from isaacsim import SimulationApp                             # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                            # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleArticulation             # noqa: E402
from isaacsim.core.utils.types import ArticulationAction       # noqa: E402
from pxr import UsdGeom, Gf, Usd                               # noqa: E402
from isaac_common import import_urdf                           # noqa: E402

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rclpy.executors import SingleThreadedExecutor             # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402
from sensor_msgs.msg import JointState                         # noqa: E402
from std_msgs.msg import Float64MultiArray, String             # noqa: E402
from nav_msgs.msg import Odometry                              # noqa: E402


class WBNode(Node):
    """只訂閱 /wb_vel_cmd 這一個 topic。"""

    def __init__(self, chain):
        super().__init__('isaac_wholebody_sim')
        self.chain = chain
        self.sim_t = 0.0
        self.n_cb = 0
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._cmd, 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.status_pub = self.create_publisher(String, '/wb_sim/status', 10)

    def _cmd(self, msg):
        self.n_cb += 1
        # **接收時間**用模擬時間，命名為 recv_sim_t；不冒稱來源發布時間
        self.chain.receive(list(msg.data), self.sim_t)

    def publish_feedback(self, t, names, q, dq, bp, byaw, bv, bw, status):
        """回授：上游控制器與安全鏈需要這些才能閉迴路。"""
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(t); stamp.nanosec = min(int(round((t - int(t)) * 1e9)),
                                                999999999)
        js = JointState(); js.header.stamp = stamp
        js.name = list(names)
        js.position = [float(x) for x in q]
        js.velocity = [float(x) for x in dq]
        self.js_pub.publish(js)

        od = Odometry(); od.header.stamp = stamp
        od.header.frame_id = 'odom'; od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(bp[0])
        od.pose.pose.position.y = float(bp[1])
        od.pose.pose.position.z = float(bp[2])
        od.pose.pose.orientation.w = math.cos(byaw / 2.0)
        od.pose.pose.orientation.z = math.sin(byaw / 2.0)
        od.twist.twist.linear.x = float(bv[0])
        od.twist.twist.linear.y = float(bv[1])
        od.twist.twist.angular.z = float(bw[2])
        self.odom_pub.publish(od)

        self.status_pub.publish(String(data=status))


def q_yaw(t):
    return [math.cos(t / 2.0), 0.0, 0.0, math.sin(t / 2.0)]


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    if abs(float(world.get_rendering_dt()) - float(a.physics_dt)) > 1e-12:
        print('[wb] rendering_dt != physics_dt，中止'); return 4
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    # **底盤不固定**：全身同動需要底盤能動，所以不用 fix_base
    import_urdf(a.urdf, ROBOT, fix_base=False)
    stage = world.stage
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    _rq = q_yaw(0.0)
    xf.AddOrientOp().Set(Gf.Quatf(_rq[0], _rq[1], _rq[2], _rq[3]))

    world.reset()
    robot = SingleArticulation(prim_path=ROBOT, name='omni_bot')
    robot.initialize()
    idx = {n: k for k, n in enumerate(robot.dof_names)}
    missing = [j for j in ARM if j not in idx]
    if missing:
        print(f'[wb] URDF 缺關節 {missing}，中止'); return 5
    print(f'[wb] articulation DOF {robot.num_dof}', flush=True)

    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM:
        kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q))

    # --- **低速介面界限**（不是輪級限制功能）：越界即整筆中止，不縮命令 ---
    def low_speed_bound(vx, vy, wz):
        lin = math.hypot(vx, vy)
        if lin > a.base_lin_max:
            return (False, f'線速度 {lin:.4f} > {a.base_lin_max} m/s')
        if abs(wz) > a.base_ang_max:
            return (False, f'角速度 {abs(wz):.4f} > {a.base_ang_max} rad/s')
        return (True, None)

    chain = CmdChain(max_cmd_age_s=a.max_cmd_age_s,
                     arm_rate_max=a.arm_rate_max, wheel_ok=low_speed_bound,
                     joint_lower=tuple(LITE6_SAFE.lower),
                     joint_upper=tuple(LITE6_SAFE.upper), mode=a.mode)
    print(f'[wb] 模式 {a.mode}：允許分量 {chain.mode}；'
          f'不符模式的命令整筆拒收', flush=True)

    rclpy.init()
    node = WBNode(chain)
    ex = SingleThreadedExecutor(); ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True); th.start()
    print('[wb] 進入主迴圈；等待 /wb_vel_cmd', flush=True)
    return loop(world, robot, idx, chain, node, ex, th)


LOG_COLS = ['t', 'recv_seq', 'cmd_age', 'vx_cmd', 'vy_cmd', 'wz_cmd',
            'base_x', 'base_y', 'base_yaw', 'base_lin_meas', 'base_ang_meas',
            'integrating'] + [f'{j}_sp' for j in ARM] \
    + [f'{j}_act' for j in ARM] + [f'{j}_rate_meas' for j in ARM]


def loop(world, robot, idx, chain, node, ex, th):
    log, stop = [], 'sim_limit'
    STOP_HOLD_STEPS = 100        # 失效後續量 1.0 s，證明停止行為而非直接關掉
    fail_steps = 0
    t_prev = None
    q_prev = None
    tc0, tsrc = cpu_temp_read()
    temp_max = tc0 or -1.0
    w0 = time.monotonic()
    nxt = time.monotonic()
    print(f'[wb] 起始 CPU {tc0} °C（{tsrc}）', flush=True)
    while True:
        world.step(render=False)
        t = float(world.current_time)
        if t_prev is not None and t <= t_prev:
            chain.note_time_reset(t)
            stop = 'time_not_monotonic'
            break
        dt = a.physics_dt if t_prev is None else (t - t_prev)
        node.sim_t = t
        sec = int(t); nsec = int(round((t - sec) * 1e9))
        c = Clock(); c.clock.sec = sec; c.clock.nanosec = min(nsec, 999999999)
        node.clock_pub.publish(c)

        qm = robot.get_joint_positions()
        qa = np.array([float(qm[idx[j]]) for j in ARM])
        # 實測關節速度：用來判定「停止」，**不用設定點凍結代替**
        rate = (np.zeros(6) if q_prev is None or dt <= 0
                else (qa - q_prev) / dt)
        bp, bq = robot.get_world_poses()
        bp = np.asarray(bp[0], float); byaw = yaw_of(np.asarray(bq[0], float))
        bv = robot.get_linear_velocities()[0]
        bw = robot.get_angular_velocities()[0]

        # 失效後**不關迴圈**：底盤停止、手臂保持設定點，並繼續量測，
        # 直到停止條件由實測資料判定。關閉模擬器不等於驗證停止。
        node.publish_feedback(
            t, ARM, qa, rate, np.array([bp[0], bp[1], bp[2]]), byaw, bv, bw,
            json.dumps({'mode': a.mode, 'fail': chain.fail,
                        'integrating': chain.integrating,
                        'recv': chain.n_recv, 'rejected': chain.n_rejected},
                       ensure_ascii=False))
        out = chain.step(t, dt, qa) if chain.fail is None else chain.stop_command()
        s = chain.applied
        if out is not None and out[1] is None:
            base_cmd, sp = out[0], None
        elif out is not None:
            base_cmd, sp = out
            if sp is not None:
                tgt = robot.get_joint_positions()
                for k, j in enumerate(ARM):
                    tgt[idx[j]] = sp[k]
                robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=tgt))
            # 底盤：本體速度 → 世界速度
            cy, sy = math.cos(byaw), math.sin(byaw)
            vwx = base_cmd[0] * cy - base_cmd[1] * sy
            vwy = base_cmd[0] * sy + base_cmd[1] * cy
            robot.set_linear_velocities(np.array([[vwx, vwy, 0.0]]))
            robot.set_angular_velocities(np.array([[0.0, 0.0, base_cmd[2]]]))
        else:
            base_cmd, sp = (float('nan'),) * 3, (float('nan'),) * 6
        if sp is None:
            sp = (float('nan'),) * 6

        log.append([round(t, 4),
                    s.recv_seq if s is not None else -1,
                    round(t - s.recv_sim_t, 4) if s is not None else float('nan'),
                    *[round(float(v), 6) for v in base_cmd],
                    round(float(bp[0]), 6), round(float(bp[1]), 6),
                    round(float(byaw), 6),
                    round(float(math.hypot(bv[0], bv[1])), 6),
                    round(float(bw[2]), 6),
                    1 if chain.integrating else 0]
                   + [round(float(v), 6) for v in sp]
                   + [round(float(v), 6) for v in qa]
                   + [round(float(v), 6) for v in rate])
        if len(log) == 1 and len(log[0]) != len(LOG_COLS):
            raise RuntimeError(f'log 欄名 {len(LOG_COLS)} vs 資料列 {len(log[0])}')

        if len(log) % 25 == 0:
            tc, src = cpu_temp_read()
            if tc is None:
                # **讀不到溫度不等於安全。** 沿用已確立的監看失效處置。
                chain.note_monitor_failure('cpu_temp', f'讀不到（來源 {src}）', t)
            else:
                temp_max = max(temp_max, tc)
                if tc >= 92.0:
                    chain.note_monitor_failure('cpu_temp',
                                               f'{tc:.2f} °C ≥ 92.0', t)
        # 失效後仍執行 stop_command 並量測；連續 `stop_hold_steps` 步後才收尾
        if chain.fail is not None:
            fail_steps += 1
            if fail_steps >= STOP_HOLD_STEPS:
                stop = ('monitor_failure' if '監看失效' in chain.fail
                        else 'cmd_chain_fail')
                break
        if t >= a.sim_limit:
            stop = 'sim_limit'; break
        if time.monotonic() - w0 > 900.0:
            stop = 'wall_limit'; break
        t_prev, q_prev = t, qa
        if a.rtf > 0:
            nxt += a.physics_dt / a.rtf
            sl = nxt - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.monotonic()

    print(f'[wb] 停止：{stop} @ sim {float(world.current_time):.3f}', flush=True)
    out = {
        'schema': 'wb_sim/1', 'mode': a.mode,
        'spec': 'evaluation/results/specs/isaac_wholebody_port_v2.md',
        'solver_label': a.solver_label,
        'solver_note': ('僅供記錄的上游求解模式標示。**dls 不是 B 基線**；'
                        'B 凍結於 baseline_B_frozen_20260909.md（--solver qp、'
                        'OSQP eps 1e-6、μ=0.03、下游濾波器保留），'
                        '移植前須逐項核對現行程式'),
        'command_interface': {
            'topic': '/wb_vel_cmd', 'dof': 9,
            'note': ('只訂閱單一 topic；底盤與手臂來自同一次全身求解。'
                     '分開訂閱兩個 topic 取最新值不算同步，本檔不提供該路徑')},
        'guards_not_started': ['arm_vel_gate', 'wheel_limit_guard'],
        'guards_note': ('節點未啟動，但其必要檢查未省略 —— 由 wb_cmd_chain 承接，'
                        '並已由 evaluation/test_wb_cmd_chain.py 不開模擬器測試'),
        'low_speed_interface_bound': {
            'lin_mps': a.base_lin_max, 'ang_rps': a.base_ang_max,
            'note': ('**低速介面界限**，不是輪級限制功能；越界即整筆中止，'
                     '不縮命令')},
        'wheel_level_limiting_implemented': WHEEL_LIMIT_IMPLEMENTED,
        'wheel_level_note': ('輪級正常輸出限制尚未實作；pregrasp 由程式常數'
                             '無條件禁止，不提供旗標讓使用者自行宣告'),
        'arm_adapter': {
            'kind': 'velocity_integrated_to_position_setpoint',
            'dt_source': 'world.current_time 相鄰物理步差',
            'setpoint_init': '第一筆有效命令時由實測關節位置建立',
            'on_expire': ('停止積分、設定點凍結；**不代表手臂實際速度瞬間為零** —— '
                          '停止行為由 log 的 *_rate_meas 另行判定'),
            'not_native_velocity_control': True},
        'cmd_chain': chain.summary(),
        'fail_hold_steps': fail_steps,
        'stop_flow_note': ('失效後每步送 stop_command（底盤停止、手臂保持設定點）'
                           f'並繼續量測 {STOP_HOLD_STEPS} 步（{STOP_HOLD_STEPS*a.physics_dt:.1f} s）'
                           '才收尾；停止行為由 log 的 base_lin_meas 與 '
                           '*_rate_meas 判定，不以關閉模擬器代替'),
        'feedback_published': ['/clock', '/joint_states', '/odom', '/wb_sim/status'],
        'callbacks': node.n_cb,
        'stop_reason': stop, 'sim_time_s': float(world.current_time),
        'wall_s': time.monotonic() - w0,
        'cpu_temp_start_c': tc0, 'cpu_temp_max_c': temp_max,
        'cpu_temp_source': tsrc, 'cpu_limit_c': 92.0,
        'log_cols': LOG_COLS, 'log': log,
    }
    json.dump(out, open(os.path.join(a.out, 'wb_run.json'), 'w'),
              ensure_ascii=False)
    print(f'[wb] -> {os.path.join(a.out, "wb_run.json")}', flush=True)
    ex.shutdown(); rclpy.try_shutdown()
    return 0


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    tb = traceback.format_exc(); print(tb, flush=True)
    open(os.path.join(a.out, 'traceback.txt'), 'w').write(tb)
    rc = 9
finally:
    sim_app.close()
sys.exit(rc)
