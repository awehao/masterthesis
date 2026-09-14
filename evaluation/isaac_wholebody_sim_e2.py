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
# 與 isaac_drawer_sim.py 相同：套件在 src/ 下，Isaac 的直譯器沒有 workspace 的
# install/ 在路徑上，所以直接加進來。
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

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
ap.add_argument('--no-frictionless', action='store_true',
                help='診斷用：不綁零摩擦材質（保留 PhysX 預設），用來對照。'
                     'URDF 的 <gazebo><mu1> importer 不讀，不綁就是預設約 0.5')
# ---- E2 輪級限制參數（**預設值不得為了觸發限制而放寬**）----
ap.add_argument('--wheel-radius', type=float, default=0.05)
ap.add_argument('--wheel-base-L', type=float, default=0.245,
                help='取自 URDF base_link_rim_*_joint 的 ±0.245')
ap.add_argument('--wheel-w-max', type=float, default=5.55, help='rad/s')
ap.add_argument('--wheel-a-max', type=float, default=125.0, help='rad/s²')
ap.add_argument('--keep-limit-rows', type=int, default=4000,
                help='存進 wb_run.json 的逐步限制記錄筆數上限')
# solver_freespace：允許底盤與手臂，另有場景條件把關；**不解除 pregrasp 禁止**
ap.add_argument('--mode', default='base',
                choices=['base', 'arm', 'sync', 'solver_freespace',
                         'pregrasp'],
                help='驗收順序：base → arm → sync → solver_freespace；'
                     'pregrasp 另由 PREGRASP_PRECONDITIONS_MET 禁止')
# ---- 執行時錄影（模擬器內相機；**不加任何文字、不改場景**）----
ap.add_argument('--record-frames', default='', help='輸出 PNG 的目錄；空=不錄')
ap.add_argument('--record-res', default='1920x1080')
ap.add_argument('--record-fps', type=float, default=30.0)
ap.add_argument('--record-eye', default='')
ap.add_argument('--record-at', default='')
ap.add_argument('--record-focal', type=float, default=24.0)
ap.add_argument('--record-from', type=float, default=0.0, help='起始模擬時間')
ap.add_argument('--record-to', type=float, default=1e9, help='結束模擬時間')
ap.add_argument('--record-warmup', type=int, default=12,
                help='RTX 註解器暖機次數；單次算繪會拿到過期影像')
ap.add_argument('--run-label', default='',
                help='趟次名稱標示，寫進輸出以免日後被誤讀')
ap.add_argument('--solver-label', default='dls',
                help='僅供記錄：本趟上游用的求解模式。介面煙霧測試標示 dls；'
                     '**dls 不是 B 基線**')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ===== 執行版本 E2 =====
# E1（isaac_wholebody_sim.py）**保持位元不變**；本檔是另立的版本。
# 與 E1 的唯一功能差異：命令鏈改用 CmdChainE2，加上輪級限制
#（策略 evaluation/results/specs/wb_wheel_limit_policy_v2.md）。
WHEEL_LIMIT_IMPLEMENTED = True

# **輪級限制已實作，不等於 pregrasp 可以解鎖。**
# 這裡刻意用**另一個獨立條件**，不是同一個旗標 —— 還缺三項：
#   1. 命令被修改後，上游全身安全性的重新論證（策略 §2 明說目前沒有）
#   2. 停止掃掠範圍的避碰保證（見停止預算撤回書，目前沒有）
#   3. 求解器驅動的受限測試規格（低速、自由空間）尚未訂定
PREGRASP_PRECONDITIONS_MET = False

# ===== 自由空間求解器測試模式的**條件**（不是換個名稱繞過原檢查）=====
# 這個模式允許的分量與 sync 相同，但多一道**可檢查的場景條件**。
#
# **語意檢查，不比對名稱或型別**：掃描機器人與地面**以外**是否存在
# 碰撞體（UsdPhysics.CollisionAPI）。有碰撞體才可能構成接觸，
# 材質、Looks、燈光都沒有碰撞體。
#
# 先前兩次都因為用名稱／型別判斷而誤判並中止：
#   wb_solver_iso_093451：子字串 'box' 比對到相機自身的網格
#   wb_solver_iso_093853：白名單把匯入產生的 Physics_Materials / Looks 當成物件
# 兩趟都標記為 ABORTED，不計為任務結果。
#
# pregrasp 的禁止**完全不受本模式影響**。
FREESPACE_OWN_SUBTREES = ('/World/omni_bot', '/World/ground')

if a.mode == 'pregrasp' and not PREGRASP_PRECONDITIONS_MET:
    print('[wb] **pregrasp 仍禁止**：輪級限制雖已實作，'
          '但「命令被修改後的上游安全性」「停止掃掠避碰保證」'
          '「求解器受限測試規格」三項前提未滿足。'
          '這不是同一個旗標，不能由 WHEEL_LIMIT_IMPLEMENTED 解鎖。')
    sys.exit(2)

import yaml                                                    # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                          # noqa: E402
from wb_wheel_limit import WheelLimitConfig                     # noqa: E402
from cpu_temp import read as cpu_temp_read                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE           # noqa: E402

# **單一命令順序**：`joints` 參數、六個速度分量的積分、以及對 articulation
# 的實際 DOF 索引與套用，全部用這一份。**不另外宣告一份讓檢查通過。**
CMD_JOINT_ORDER = [f'joint{i}' for i in range(1, 7)]
ARM = CMD_JOINT_ORDER          # 沿用既有名稱，指向同一個物件
ROBOT = '/World/omni_bot'

from isaacsim import SimulationApp                             # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                            # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleArticulation             # noqa: E402
from isaacsim.core.utils.types import ArticulationAction       # noqa: E402
from pxr import UsdGeom, Gf, Usd                               # noqa: E402
from isaac_common import (import_urdf, physics_parts,           # noqa: E402
                          bind_frictionless, collision_prims)

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rclpy.executors import SingleThreadedExecutor             # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402
from sensor_msgs.msg import JointState                         # noqa: E402
from std_msgs.msg import Float64MultiArray, String             # noqa: E402
from nav_msgs.msg import Odometry                              # noqa: E402
from geometry_msgs.msg import TransformStamped                 # noqa: E402
import tf2_ros                                                 # noqa: E402


class WBNode(Node):
    """只訂閱 /wb_vel_cmd 這一個 topic。"""

    def __init__(self, chain, joint_order):
        super().__init__('isaac_wholebody_sim')
        # 回報給 adapter 核對的 joints，**就是本端實際使用的命令順序**。
        self.declare_parameter('joints', list(joint_order))
        self.chain = chain
        self.sim_t = 0.0
        self.n_cb = 0
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._cmd, 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.status_pub = self.create_publisher(String, '/wb_sim/status', 10)
        # **Isaac 只發布到模型根部 base_footprint**。
        # base_footprint → base_link（URDF 固定 +0.05 m）由 robot_state_publisher
        # 發在 /tf_static；安全層查 odom → base_link 時由 TF 鏈自動組合。
        # 若這裡另發 odom → base_link，base_link 就有兩個父節點。
        self.tfb = tf2_ros.TransformBroadcaster(self)

    def _cmd(self, msg):
        self.n_cb += 1
        # **接收時間**用模擬時間，命名為 recv_sim_t；不冒稱來源發布時間
        self.chain.receive(list(msg.data), self.sim_t)

    def publish_feedback(self, t, names, q, dq, p_fp, quat_fp, R_fp,
                         v_world, w_world, status):
        """回授：**同一份完整位姿與同一個時間戳**。

        * 位姿取自 `base_footprint` prim 的實際世界變換，**保留完整姿態**
          （不只用 yaw 重建 —— 那會把底盤的 roll/pitch 藏掉）。
        * `/odom.twist` 以 **child_frame_id 表達**（本體座標），
          由世界座標速度旋轉而來。yaw = 0 時碰巧相同，不代表定義正確。
        """
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(t); stamp.nanosec = min(int(round((t - int(t)) * 1e9)),
                                                999999999)
        js = JointState(); js.header.stamp = stamp
        js.name = list(names)
        js.position = [float(x) for x in q]
        js.velocity = [float(x) for x in dq]
        self.js_pub.publish(js)

        # 世界 → 本體（child_frame）：twist 的定義要求如此
        v_b = R_fp.T @ np.asarray(v_world, float)
        w_b = R_fp.T @ np.asarray(w_world, float)

        od = Odometry(); od.header.stamp = stamp
        od.header.frame_id = 'odom'
        od.child_frame_id = 'base_footprint'
        od.pose.pose.position.x = float(p_fp[0])
        od.pose.pose.position.y = float(p_fp[1])
        od.pose.pose.position.z = float(p_fp[2])
        od.pose.pose.orientation.w = float(quat_fp[0])
        od.pose.pose.orientation.x = float(quat_fp[1])
        od.pose.pose.orientation.y = float(quat_fp[2])
        od.pose.pose.orientation.z = float(quat_fp[3])
        od.twist.twist.linear.x = float(v_b[0])
        od.twist.twist.linear.y = float(v_b[1])
        od.twist.twist.linear.z = float(v_b[2])
        od.twist.twist.angular.x = float(w_b[0])
        od.twist.twist.angular.y = float(w_b[1])
        od.twist.twist.angular.z = float(w_b[2])
        self.odom_pub.publish(od)

        tr = TransformStamped()
        tr.header.stamp = stamp                  # **與 /odom 同一時間戳**
        tr.header.frame_id = 'odom'
        tr.child_frame_id = 'base_footprint'     # **單一父節點**
        tr.transform.translation.x = float(p_fp[0])
        tr.transform.translation.y = float(p_fp[1])
        tr.transform.translation.z = float(p_fp[2])
        tr.transform.rotation.w = float(quat_fp[0])
        tr.transform.rotation.x = float(quat_fp[1])
        tr.transform.rotation.y = float(quat_fp[2])
        tr.transform.rotation.z = float(quat_fp[3])
        self.tfb.sendTransform(tr)

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
    prim = import_urdf(a.urdf, ROBOT, fix_base=False)
    stage = world.stage
    # 與導航版一致：**搜尋 articulation root**，不直接寫死 prim 路徑。
    if a.mode == 'solver_freespace':
        from pxr import UsdPhysics
        own = tuple(FREESPACE_OWN_SUBTREES)
        foreign = [str(pr.GetPath()) for pr in Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
            if pr.HasAPI(UsdPhysics.CollisionAPI)
            and not str(pr.GetPath()).startswith(own)]
        n_own = sum(len(collision_prims(stage, u)) for u in own)
        if foreign:
            print(f'[wb] **solver_freespace 場景條件未滿足**：'
                  f'機器人與地面以外有 {len(foreign)} 個碰撞體，'
                  f'例如 {foreign[:3]}。本模式僅適用於自由空間，中止。')
            return 8
        print(f'[wb] solver_freespace 場景條件已核對（語意檢查，非名稱比對）：'
              f'機器人與地面共 {n_own} 個碰撞體，其他位置 0 個', flush=True)
        globals()['FREESPACE_SCENE'] = {
            'check': 'UsdPhysics.CollisionAPI 於機器人與地面以外',
            'own_subtrees': list(own), 'own_colliders': n_own,
            'foreign_colliders': 0}

    bodies, arts = physics_parts(stage, prim)
    if not arts:
        print('[wb] **找不到 articulation root**，中止'); return 7
    ART_ROOT = arts[0]
    print(f'[wb] articulation root {ART_ROOT}（剛體 {len(bodies)}）', flush=True)
    globals()['ART_ROOT'] = ART_ROOT
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    _rq = q_yaw(0.0)
    xf.AddOrientOp().Set(Gf.Quatf(_rq[0], _rq[1], _rq[2], _rq[3]))

    # 零摩擦材質：URDF 的 <gazebo><mu1> importer 不讀，產生的 USD 沒有材質，
    # PhysX 退回預設（約 0.5）。導航版量到的效果是速度追蹤只有 37 %。
    # **接觸雙方都要綁**：預設混合規則取平均，只歸零機器人側仍留一半地面摩擦。
    fric_rb = None
    if not a.no_frictionless:
        sph = collision_prims(stage, prim, lambda p: 'support_' in p)
        gr = collision_prims(stage, '/World/ground')
        n_bound = bind_frictionless(stage, sph + gr)
        fric_rb = {'support_spheres': len(sph), 'ground': len(gr),
                   'bound': n_bound,
                   'note': ('URDF 的 <gazebo><mu1> 不會被 importer 讀入；'
                            '雙方都綁才有效（預設混合規則取平均）')}
        print(f'[wb] 零摩擦材質：支撐球 {len(sph)} + 地面 {len(gr)} → '
              f'綁定 {n_bound}', flush=True)
        if n_bound == 0:
            print('[wb] **零摩擦綁定數為 0，中止**'); return 8

    world.reset()
    robot = SingleArticulation(prim_path=ART_ROOT, name='omni_bot')
    robot.initialize()
    idx = {n: k for k, n in enumerate(robot.dof_names)}
    missing = [j for j in CMD_JOINT_ORDER if j not in idx]
    if missing:
        print(f'[wb] URDF 缺關節 {missing}，中止'); return 5
    if len(set(CMD_JOINT_ORDER)) != len(CMD_JOINT_ORDER):
        print('[wb] **命令順序有重複關節**，中止'); return 5
    dof_ids = [idx[j] for j in CMD_JOINT_ORDER]
    if len(set(dof_ids)) != len(dof_ids):
        print('[wb] **命令順序對應到重複的 DOF 索引**，中止'); return 5
    cmd_map = [{'cmd_field': 3 + k, 'joint': j, 'dof_index': idx[j]}
               for k, j in enumerate(CMD_JOINT_ORDER)]
    print('[wb] 命令欄位 → 關節名稱 → articulation 索引：', flush=True)
    for e in cmd_map:
        print(f"    v[{e['cmd_field']}]  {e['joint']:<8}  DOF {e['dof_index']}",
              flush=True)
    print(f'[wb] 無缺漏、無重複；執行中不改順序', flush=True)
    globals()['CMD_MAP'] = cmd_map
    print(f'[wb] articulation DOF {robot.num_dof}', flush=True)

    # **確認實際位姿對應的 prim**：不把 articulation 根的位姿直接當成某個連桿。
    tcp_prim = next((pr for pr in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        if pr.GetName() == 'link_tcp'
        and str(pr.GetPath()).startswith(ROBOT)), None)
    if tcp_prim is None:
        print('[wb] **找不到 link_tcp prim** —— 中止（判準需要它做獨立驗算）')
        return 9
    print(f'[wb] link_tcp prim {tcp_prim.GetPath()}', flush=True)
    globals()['TCP_PRIM'] = tcp_prim

    fp = next((pr for pr in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        if pr.GetName() == 'base_footprint'
        and str(pr.GetPath()).startswith(ROBOT)), None)
    if fp is None:
        print('[wb] 找不到 base_footprint prim，中止'); return 6
    print(f'[wb] base_footprint prim {fp.GetPath()}', flush=True)
    xc = UsdGeom.XformCache()
    m_fp = xc.GetLocalToWorldTransform(fp)
    t_fp = m_fp.ExtractTranslation()
    rp, _ = robot.get_world_pose()
    d_root = float(np.linalg.norm(np.array([t_fp[0], t_fp[1], t_fp[2]])
                                  - np.asarray(rp, float)))
    print(f'[wb] articulation 根 vs base_footprint 位置差 {d_root*1000:.4f} mm'
          f'（核對，非假設）', flush=True)
    globals()['ROOT_FP_OFFSET_MM'] = d_root * 1000.0

    # --- 執行時錄影：**這一趟真正在跑的畫面**，不是姿態重演 ---
    # 只加相機與燈光（無碰撞體，不影響 solver_freespace 的場景條件），
    # **不加任何文字、標記或覆疊**；算繪只讀場景、不寫回狀態。
    rec_dir = a.record_frames
    rec_index, rec_cam, rec_every = [], None, 1
    if rec_dir:
        import imageio.v2 as _imageio                          # noqa: E402
        from isaacsim.sensors.camera import Camera             # noqa: E402
        from pxr import UsdLux                                 # noqa: E402
        os.makedirs(rec_dir, exist_ok=True)
        _k = UsdLux.DistantLight.Define(stage, '/World/rec_key')
        _k.CreateIntensityAttr(3000.0)
        UsdGeom.Xformable(_k).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
        UsdLux.DomeLight.Define(stage, '/World/rec_dome').CreateIntensityAttr(450.0)
        _at = (np.array([float(v) for v in a.record_at.split(',')])
               if a.record_at else np.array([0.06, 0.0, 0.34]))
        _eye = (np.array([float(v) for v in a.record_eye.split(',')])
                if a.record_eye else _at + np.array([1.55, -2.30, 1.05]))
        _w, _h = (int(v) for v in a.record_res.split('x'))
        rec_cam = Camera(prim_path='/World/rec_cam', resolution=(_w, _h))
        _cx = UsdGeom.Xformable(stage.GetPrimAtPath('/World/rec_cam'))
        _cx.ClearXformOpOrder()
        _cx.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*[float(v) for v in _eye]), Gf.Vec3d(*[float(v) for v in _at]),
            Gf.Vec3d(0, 0, 1)).GetInverse())
        _cg = UsdGeom.Camera(stage.GetPrimAtPath('/World/rec_cam'))
        _cg.GetFocalLengthAttr().Set(float(a.record_focal))
        _cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 200.0))
        if a.record_target:
            _t = np.array([float(v) for v in a.record_target.split(',')])
            _sc = float(a.record_target_scale)
            # 目標姿態與判準同一組定義：工具 +z 指向世界 +x、工具 +x 指向世界 +z
            _zc = np.array([1.0, 0.0, 0.0])
            _xc = np.array([0.0, 0.0, 1.0])
            _xc = _xc - float(_xc @ _zc) * _zc
            _xc /= np.linalg.norm(_xc)
            _Rm = np.column_stack([_xc, np.cross(_zc, _xc), _zc])
            _M = np.eye(4)
            _M[:3, :3] = _Rm.T          # USD 為列向量慣例
            _M[3, :3] = _t
            UsdGeom.Xform.Define(stage, '/World/rec_target')
            UsdGeom.Xformable(stage.GetPrimAtPath('/World/rec_target')) \
                .AddTransformOp().Set(Gf.Matrix4d(*_M.flatten().tolist()))

            def _vis(prim, rgb, opacity):
                g = UsdGeom.Gprim(prim)
                g.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
                g.CreateDisplayOpacityAttr().Set([float(opacity)])

            _sp = UsdGeom.Sphere.Define(stage, '/World/rec_target/point')
            _sp.CreateRadiusAttr(0.020 * _sc)
            _vis(_sp.GetPrim(), (0.95, 0.45, 0.15), 0.45)
            for _ax, _rgb, _rot in (
                    ('x', (0.90, 0.25, 0.25), (0.0, 90.0, 0.0)),
                    ('y', (0.25, 0.75, 0.35), (-90.0, 0.0, 0.0)),
                    ('z', (0.25, 0.45, 0.95), (0.0, 0.0, 0.0))):
                _c = UsdGeom.Cylinder.Define(stage, f'/World/rec_target/ax_{_ax}')
                _c.CreateRadiusAttr(0.0045 * _sc)
                _c.CreateHeightAttr(0.080 * _sc)
                # **順序要緊**：USD 依 xformOpOrder 逐一套用（列向量慣例）。
                # 先旋轉再平移的話，位移會落在**父座標**而不是旋轉後的軸向 ——
                # x 軸圓柱會被推到父 +z。先平移、再旋轉才是沿各自軸向。
                _x = UsdGeom.Xformable(_c.GetPrim())
                _x.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.040 * _sc))
                _x.AddRotateXYZOp().Set(Gf.Vec3f(*_rot))
                _vis(_c.GetPrim(), _rgb, 0.85)
            # **自我查核**：標記底下不得有碰撞體或剛體，否則中止 ——
            # 有的話它就成了場景裡的物件，solver_freespace 的條件也會失真
            from pxr import UsdPhysics                            # noqa: E402
            _bad = [str(pr.GetPath()) for pr in Usd.PrimRange(
                stage.GetPrimAtPath('/World/rec_target'))
                if pr.HasAPI(UsdPhysics.CollisionAPI)
                or pr.HasAPI(UsdPhysics.RigidBodyAPI)]
            if _bad:
                print(f'[wb] **目標標記帶有物理 API {_bad}，中止**')
                return 13
            print(f'[wb] 目標標記（**純視覺**，無碰撞體／剛體）'
                  f' @ {np.round(_t, 3).tolist()}；'
                  f'距離節點的障礙物來自其 obstacles 參數，不掃 stage',
                  flush=True)
            globals()['REC_TARGET'] = {'xyz': [float(v) for v in _t],
                                       'visual_only': True,
                                       'has_collision': False,
                                       'has_rigid_body': False}

        rec_every = max(1, int(round(1.0 / (a.record_fps * a.physics_dt))))
        globals()['REC'] = (rec_cam, rec_dir, rec_every, rec_index, _imageio)
        globals()['REC_INDEX'] = rec_index
        # **RTX 註解器有延遲**：不 initialize、不暖機的話 get_rgba() 取不到影像
        #（先前錄影趟寫出 0 幀即為此）。取不到就中止，不讓趟次靜默錄成空的。
        rec_cam.initialize()
        for _ in range(max(1, a.record_warmup)):
            world.render()
            _img0 = rec_cam.get_rgba()
            if _img0 is not None and len(_img0):
                break
        else:
            print('[wb] **相機暖機後仍取不到影像，中止**'); return 12
        for _ in range(max(1, a.record_warmup)):
            world.render()
        print('[wb] 相機暖機完成', flush=True)

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

    wcfg = WheelLimitConfig(wheel_radius=a.wheel_radius,
                            wheel_base_L=a.wheel_base_L,
                            wheel_w_max=a.wheel_w_max,
                            wheel_a_max=a.wheel_a_max,
                            arm_rate_max=a.arm_rate_max,
                            dt_max=max(5.0 * a.physics_dt, 0.05))
    chain = CmdChainE2(max_cmd_age_s=a.max_cmd_age_s,
                       arm_rate_max=a.arm_rate_max, wheel_ok=low_speed_bound,
                       joint_lower=tuple(LITE6_SAFE.lower),
                       joint_upper=tuple(LITE6_SAFE.upper), mode=a.mode,
                       wheel_cfg=wcfg, keep_limit_rows=a.keep_limit_rows)
    print(f'[wb] **E2 輪級限制**：r={wcfg.wheel_radius} L={wcfg.wheel_base_L} '
          f'w_lim={wcfg.w_lim:.4f} m/s a_lim(dt={a.physics_dt})={wcfg.a_lim(a.physics_dt):.4f} m/s'
          f'；底盤分量為**本體座標**', flush=True)
    print(f'[wb] 模式 {a.mode}：允許分量 {chain.mode}；'
          f'不符模式的命令整筆拒收', flush=True)

    rclpy.init()
    node = WBNode(chain, CMD_JOINT_ORDER)
    ex = SingleThreadedExecutor(); ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True); th.start()
    print('[wb] 進入主迴圈；等待 /wb_vel_cmd', flush=True)
    return loop(world, robot, idx, chain, node, ex, th, fp)


# 三路同步（診斷用）：
#   sent_*    上一步**送進 set_linear_velocity/set_angular_velocity 的世界速度**
#   phys_*    **本步**（= 送出後的下一個物理步）由 PhysX 回報的位置與速度
#   usd_*     **同步**讀到的 USD base_footprint 位姿
# 程式順序是「物理步進 → 讀量測 → 套用新命令」，所以本列的量測是**上一列命令**
# 的執行結果；比較時要這樣對齊，不能拿同列的新命令比。
LOG_COLS = ['t', 'recv_seq', 'cmd_age', 'vx_cmd', 'vy_cmd', 'wz_cmd',
            # sent_prev_*：**產生本列量測的那一筆**世界速度命令
            # （不是本列剛送出的那筆）
            'sent_prev_vwx', 'sent_prev_vwy', 'sent_prev_wz',
            'phys_x', 'phys_y', 'phys_vx', 'phys_vy',
            'usd_x', 'usd_y',
            'base_x', 'base_y', 'base_yaw', 'base_lin_meas', 'base_ang_meas',
            'integrating',
            # E2 新增：輪級限制的逐步記錄
            'lam', 'modified', 'limit_mode', 'wheel_speed_max',
            'wheel_accel_max',
            # E2.1 新增：Isaac 的 link_tcp 世界位姿（供獨立讀回驗算）
            'tcp_x', 'tcp_y', 'tcp_z',
            'tcp_r00', 'tcp_r01', 'tcp_r02',
            'tcp_r10', 'tcp_r11', 'tcp_r12',
            'tcp_r20', 'tcp_r21', 'tcp_r22'] + [f'{j}_sp' for j in ARM] \
    + [f'{j}_act' for j in ARM] + [f'{j}_rate_meas' for j in ARM]


def loop(world, robot, idx, chain, node, ex, th, fp):
    log, stop = [], 'sim_limit'
    STOP_HOLD_STEPS = 100        # 失效後續量 1.0 s，證明停止行為而非直接關掉
    fail_steps = 0
    n_step = 0
    sent_prev = (float('nan'),) * 3     # 上一步送進速度 API 的世界速度
    t_prev = None
    q_prev = None
    tc0, tsrc = cpu_temp_read()
    temp_max = tc0 or -1.0
    w0 = time.monotonic()
    nxt = time.monotonic()
    print(f'[wb] 起始 CPU {tc0} °C（{tsrc}）', flush=True)
    while True:
        world.step(render=False)
        _R = globals().get('REC')
        _tn = float(world.current_time)
        if (_R is not None and (n_step % _R[2]) == 0
                and a.record_from <= _tn <= a.record_to):
            world.render()
            _im = _R[0].get_rgba()
            if _im is not None and len(_im):
                _R[4].imwrite(
                    os.path.join(_R[1], f'f{len(_R[3]):06d}.png'),
                    np.asarray(_im)[:, :, :3].astype(np.uint8))
                _R[3].append([len(_R[3]), round(float(world.current_time), 4)])
        n_step += 1
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
        # **物理端**位姿與速度（PhysX 回報，經 articulation 包裝）
        p_phys, _q_phys = robot.get_world_pose()
        p_phys = np.asarray(p_phys, float)
        v_phys = np.asarray(robot.get_linear_velocity(), float)
        # 位姿取自 **base_footprint prim 的實際世界變換**，保留完整姿態。
        xc = UsdGeom.XformCache()
        Mt = xc.GetLocalToWorldTransform(TCP_PRIM)
        tt_t = Mt.ExtractTranslation()
        R3t = np.array([[Mt[r][c] for c in range(3)] for r in range(3)])
        sct = np.linalg.norm(R3t, axis=1)
        R_tcp = (R3t / sct[:, None]).T          # USD 列向量慣例
        tcp_row = [round(float(tt_t[0]), 6), round(float(tt_t[1]), 6),
                   round(float(tt_t[2]), 6)] + \
                  [round(float(R_tcp[r][c]), 6)
                   for r in range(3) for c in range(3)]
        M = xc.GetLocalToWorldTransform(fp)
        tt = M.ExtractTranslation()
        bp = np.array([tt[0], tt[1], tt[2]], float)
        R3 = np.array([[M[r][c] for c in range(3)] for r in range(3)])
        sc = np.linalg.norm(R3, axis=1)
        R_fp = (R3 / sc[:, None]).T        # USD 是列向量慣例，取轉置
        rq = M.ExtractRotationQuat()
        ri = rq.GetImaginary()
        bquat = np.array([rq.GetReal(), ri[0], ri[1], ri[2]], float)
        byaw = yaw_of(bquat)
        # 速度：articulation 根剛體的速度（PhysX 回報，參考點為該剛體框架原點）。
        # **只旋轉表示，不移動參考點** —— 參考點差異未在此補正，列為限制。
        bv = np.asarray(robot.get_linear_velocity(), float)
        bw = np.asarray(robot.get_angular_velocity(), float)

        # **時序對齊**：本列的量測是**上一列送出命令**的執行結果
        #（迴圈順序是「物理步進 → 讀量測 → 套用新命令」）。
        # 先把那一筆存下來再套用新命令，log 記的才是對得上的那一筆。
        sent_effective = sent_prev
        # 失效後**不關迴圈**：底盤停止、手臂保持設定點，並繼續量測，
        # 直到停止條件由實測資料判定。關閉模擬器不等於驗證停止。
        # **夾爪手指也要發布**：robot_state_publisher 少了 finger_joint1/2
        # 就算不出 uflite_finger1/2 的 TF，逐連桿 TF 檢查會少兩個
        #（趟次 wb_solver_iso_094406 即為 TF 9/11）。
        # 本測試不動夾爪，但它們是模型的真實自由度，狀態照實發布。
        fj = [n for n in ('finger_joint1', 'finger_joint2') if n in idx]
        names_all = list(ARM) + fj
        qa_all = np.concatenate([qa, [float(qm[idx[n]]) for n in fj]])
        rate_all = np.concatenate([rate, np.zeros(len(fj))])
        node.publish_feedback(
            t, names_all, qa_all, rate_all, bp, bquat, R_fp, bv, bw,
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
            robot.set_linear_velocity(np.array([vwx, vwy, 0.0]))
            robot.set_angular_velocity(np.array([0.0, 0.0, base_cmd[2]]))
            sent_prev = (vwx, vwy, float(base_cmd[2]))
        else:
            base_cmd, sp = (float('nan'),) * 3, (float('nan'),) * 6
            sent_prev = (float('nan'),) * 3
        if sp is None:
            sp = (float('nan'),) * 6

        log.append([round(t, 4),
                    s.recv_seq if s is not None else -1,
                    round(t - s.recv_sim_t, 4) if s is not None else float('nan'),
                    *[round(float(v), 6) for v in base_cmd],
                    *[round(float(v), 6) for v in sent_effective],
                    round(float(p_phys[0]), 6), round(float(p_phys[1]), 6),
                    round(float(v_phys[0]), 6), round(float(v_phys[1]), 6),
                    round(float(bp[0]), 6), round(float(bp[1]), 6),
                    round(float(bp[0]), 6), round(float(bp[1]), 6),
                    round(float(byaw), 6),
                    round(float(math.hypot(bv[0], bv[1])), 6),
                    round(float(bw[2]), 6),
                    1 if chain.integrating else 0,
                    (float('nan') if not chain.last_limit
                     or chain.last_limit['lam'] is None
                     else chain.last_limit['lam']),
                    (float('nan') if not chain.last_limit
                     else (1 if chain.last_limit['modified'] else 0)),
                    (float('nan') if not chain.last_limit else
                     {'normal': 0, 'timeout': 1,
                      'stop_unverified': 2}.get(chain.last_limit['mode'], -1)),
                    (float('nan') if not chain.last_limit
                     else chain.last_limit['wheel_speed_max']),
                    (float('nan') if not chain.last_limit
                     else chain.last_limit['wheel_accel_max'])]
                   + tcp_row
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
        'run_label': a.run_label,
        'record_frames': (None if not a.record_frames else
                          {'dir': a.record_frames,
                           'n': len(globals().get('REC_INDEX', [])),
                           'window_sim_s': [a.record_from, a.record_to],
                           'index': globals().get('REC_INDEX', []),
                           'note': ('模擬器內相機，執行時錄影；'
                                    '無文字、無覆疊、場景未加碰撞體')}),
        'solver_label': a.solver_label,
        'solver_note': ('僅供記錄的上游求解模式標示。**dls 不是 B 基線**；'
                        'B 凍結於 baseline_B_frozen_20260909.md（--solver qp、'
                        'OSQP eps 1e-6、μ=0.03、下游濾波器保留），'
                        '移植前須逐項核對現行程式'),
        'command_interface': {
            'topic': '/wb_vel_cmd', 'dof': 9,
            'joint_order': list(CMD_JOINT_ORDER),
            'command_map': globals().get('CMD_MAP'),
            'joint_order_note': ('`joints` 參數、速度積分與 articulation 套用'
                                 '**共用同一份順序**；啟動時已檢查無缺漏、無重複，'
                                 '執行中不改'),
            'note': ('只訂閱單一 topic；底盤與手臂來自同一次全身求解。'
                     '分開訂閱兩個 topic 取最新值不算同步，本檔不提供該路徑')},
        'guards_not_started': ['arm_vel_gate', 'wheel_limit_guard'],
        'guards_note': ('節點未啟動，但其必要檢查未省略 —— 由 wb_cmd_chain 承接，'
                        '並已由 evaluation/test_wb_cmd_chain.py 不開模擬器測試'),
        'low_speed_interface_bound': {
            'lin_mps': a.base_lin_max, 'ang_rps': a.base_ang_max,
            'note': ('**低速介面界限**，不是輪級限制功能；越界即整筆中止，'
                     '不縮命令')},
        'execution_version': 'E2.1',
        'e2_1_change': ('相對 §23／§25 所用的 E2.0，**只新增 link_tcp 位姿記錄**，'
                        '命令路徑與限制邏輯未動'),
        'wheel_level_limiting_implemented': WHEEL_LIMIT_IMPLEMENTED,
        'pregrasp_preconditions_met': PREGRASP_PRECONDITIONS_MET,
        'freespace_scene_children': globals().get('FREESPACE_SCENE'),
        'pregrasp_note': ('pregrasp 由**另一個獨立條件**禁止，'
                          '不由 wheel_level_limiting_implemented 解鎖'),
        'limit_rows': chain.limit_rows,
        'limit_mode_codes': {'0': 'normal', '1': 'timeout',
                             '2': 'stop_unverified'},
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
        'feedback_published': ['/clock', '/joint_states', '/odom',
                               '/wb_sim/status', 'TF odom→base_footprint'],
        'frame_wiring': {
            'isaac_publishes_tf': 'odom → base_footprint（模型根部，單一父節點）',
            'rsp_publishes_static': 'base_footprint → base_link（URDF 固定 +0.05 m）',
            'safety_query': 'odom → base_link 由 TF 鏈自動組合',
            'odom_child_frame_id': 'base_footprint',
            'pose_source': 'base_footprint prim 的實際世界變換（完整姿態，非僅 yaw）',
            'twist_frame': 'child_frame（本體座標），由世界速度旋轉而來',
            'twist_reference_point_caveat': (
                '速度為 articulation 根剛體的 PhysX 回報值，參考點是該剛體框架'
                '原點；本檔只旋轉表示、**未移動參考點**，此差異列為限制'),
            'root_vs_footprint_offset_mm': globals().get('ROOT_FP_OFFSET_MM')},
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
