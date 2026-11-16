#!/usr/bin/env python3
"""遠距整合執行器（Codex 20261116 A1／A2(b)／A3）：isaac_m8_house_door_motm.py 的複本，前面接遠距前段。
  遠距前段：本行程只提供感測與受控對象（ROS：/clock、/odom_raw、/scan_raw、/joint_states、/far/truth、/far/handle_obs；收 /cmd_vel、/far/arm_cmd），原導航鏈與監督節點（merge_node.py fullho）在外。
  交接：監督節點交接判定通過發 /far/handover（鎖定參考與通過線）；本行程在下一個物理步邊界建立 MovingGraspCtrl r14（鎖定任務參考模式），設定點／真管線／上一命令／偏置原樣承接。
  交接後：MovingGraspCtrl → BindGate → 理想綁定 → DoorOpenExec（同門旁）；底盤提案發 /cmd_vel_grasp 經 選擇器→平滑器→Shield→GUARD，放行值照控制器生效時刻套用（far_chain_align）。
  自載入起 no_passive_arm_friction（後端讀回）。
  以下為門旁原檔說明——其中「理想感知（把手 prim 真值 30 Hz）」「底盤瞬移到通過線」兩段不適用本檔：本檔把手來源是腕部相機實際影像＋鎖定任務參考，起點是遠距起點。
住宅門 MotM 名目展示（Howard 20261113 13:49；Codex 20261113_141451）。單一 Isaac 程序；控制在主迴圈同步計算（非即時：控制計算期間物理不步進）。
**理想感知＋理想抓持**：把手估計＝把手 prim 真值以 30 Hz 取樣（無偏差、無延遲；腕部相機在抓持距離看不到把手中心，見 Codex 裁定 (a)）；抓持成立後以 FixedJoint 理想綁定。
起始條件（記錄於結果）：底盤瞬移到通過線（full_2 綁定底盤位姿的 y、偏航）上、可抓窗口起點前 L_LEAD＝0.01 m，手臂在預抓構型，靜置後**從靜止起步**；
  前導接近 0.2 s（前導長度受此通過線上 x≈0.80 m 的腕部奇異限制）（移動抓持控制器 HOLD 段，未接觸）→ 進窗口 → 插入 → 閉爪 → 確認 → READY；起跑後不瞬移、不回填時間、不暫停等規劃。
P2 首段：前導第一拍以當時估計與「預測 READY 狀態」送背景規劃程序（啟動模擬器前 fork 的獨立程序、完整快照）；可用時刻＝模擬迴圈實際收到結果的模擬時間；
  閉爪閘門：首段可用且可行才閉爪。
交接（READY 那拍首）：綁定條件（BindGate：TCP 對把手抓取位姿、雙指開度、雙指—橫桿接觸力、相對靜止 0.3 s、無禁止接觸、門角未動）＋
  開門執行層 r14（首段參考以本趟實測 T_HE 重投影手臂、移動承接）複本試跑第一拍，全過才 accept_handover → FixedJoint。
開門：P2 續接（每 0.4 s 請求；σ_s＝σ＋1.2；暫存＋首次影響那拍試跑；末段模式），規劃快照固定用本趟實測 T_HE。期限 50 ms（計算池 6、預熱）。
保護：40 N 工程中止（手腕原始傳遞力）、禁止接觸鎖存、閉鏈／淨空監看、期限失效處置。範圍：開到 85° 停穩為止，不解綁、不退出、不穿門。
--smoke：靜置＋前導＋插入前段到 1.2 s（未接觸；送首段並收結果；讀回與時序）後受控停止。不閉爪、不綁、不開門。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_motm.py --out <dir> [--smoke]"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--arm-friction-model', default='legacy', choices=['legacy', 'urdf_effort', 'no_passive_arm_friction'])   # no_passive_arm_friction（Codex Q2）：僅六個手臂關節的 legacy／逐軸摩擦歸零（研究假設變更，不是摩擦已修好）   # urdf_effort（Codex 20261115_1421xx）：手臂六關節逐軸摩擦 static＝dynamic＝1.0 N·m、viscous＝0；模型假設，不是實機辨識
ap.add_argument('--arm-fric-value', type=float, default=1.0)        # urdf_effort 模型的 static＝dynamic（N·m）；門任務＝1.0；探針用 0／0.2／1.0
ap.add_argument('--bind-model', default='fixed', choices=['fixed', 'd6_compliant'])   # d6_compliant（B3）：理想綁定改有限剛度六自由度連接（工程假設；外部 regular joint）
ap.add_argument('--preload-release', action='store_true')           # B2：綁定迴路約束方向預載釋放（door_preload_release；低頻有界設定點偏置）
ap.add_argument('--probe', action='store_true')                     # 手臂摩擦探針（Codex 核可 O1）：靜置後 joint4 限速限加速度小幅正反向動作，無門任務；不做抓持／綁定／開門
ap.add_argument('--diag', action='store_true')                      # 取證短診斷（Codex 20261115_14xxxx）：不送 P2、開門 0.5 s 後強制走閉鏈一致停止流程、每物理步補記根連桿三個時點等
ap.add_argument('--start', default='1.5,-2.8,1.5708')
ap.add_argument('--arm', default='-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018')
ap.add_argument('--far-limit-s', type=float, default=150.0)
ap.add_argument('--cam-hz', type=float, default=5.0)
a = ap.parse_args()
a.arm_friction_model = 'no_passive_arm_friction'                    # 整合趟自載入起統一零被動手臂摩擦（Codex 20261116）；設定並後端讀回，不在交接時切模型
if a.smoke or a.probe or a.diag:
    raise SystemExit('遠距整合執行器不支援 --smoke／--probe／--diag')

# ---- 啟動模擬器前：只匯入輕量的計算池／規劃程序模組並 fork（子程序自己匯入專案模組）；專案模組在 SimulationApp 之後才匯入
#      （專案模組會放入離線用的 rclpy 替身，若早於 Kit 載入會使 Kit 擴充套件匯入失敗——冒煙 1 實測崩潰 139）
import door_cpu_layout as CPUL                                   # noqa: E402
CPU_LAYOUT = os.environ.get('CPU_LAYOUT') == '1'                     # T2：排程與 BLAS 執行緒配置（只改排程；預設關＝舊行為）
if CPU_LAYOUT:
    CPUL.blas_single()                                           # 須在匯入 NumPy 前
import numpy as np                                               # noqa: E402
os.environ.setdefault('P2_TERM', '1')
os.environ.setdefault('P2_FINAL', '1')
import door_sweep_pool as DSP                                    # noqa: E402
import door_p2_planner_server as PS                              # noqa: E402
import door_first_seg as FSG                                     # noqa: E402
import door_cstop as DCS                                         # noqa: E402
POOL = DSP.SweepPool(8)
PLANNER = PS.PlannerServer()
PLANNER_FS = PS.PlannerServer()                                      # A1：首段滾動規劃用獨立程序（低優先權；交接時若仍在算即終止——可控取消，不停車等、不佔開門拍的核心）
try:
    os.setpriority(os.PRIO_PROCESS, PLANNER_FS.proc.pid, 10)
except Exception:                                                # noqa: BLE001
    pass
CSW = DCS.CStopWorker()                                              # 閉鏈一致備援停法的向前模擬程序（Codex 20261115_005224）
CPU_PIN = None
if CPU_LAYOUT:
    CPU_PIN = dict(workers=CPUL.pin_workers(POOL, CSW, (PLANNER, PLANNER_FS)), pre_kit=CPUL.pre_kit())

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
if CPU_LAYOUT:
    CPU_PIN['post_kit'] = CPUL.post_kit()                         # 只移主執行緒到 CPU 0（Kit 既有執行緒留在 Kit 集合）
    CPU_PIN['blas_after_kit'] = CPUL.blas_runtime(1)              # 模擬器啟動後把主程序的 OpenBLAS 直接設成單執行緒並讀回（環境變數會被模擬器改回）
import rclpy                                                     # noqa: E402  先匯入真正的 rclpy；專案模組（ros_stub）之後會把它換成替身，匯入完再還原
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy              # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String         # noqa: E402
from tf2_msgs.msg import TFMessage                               # noqa: E402
from builtin_interfaces.msg import Time as RosTime               # noqa: E402
_RCLPY_REAL = rclpy
_ROS_PREFIX = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces')
_ROS_SNAP = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in _ROS_PREFIX}
import door_wbopt as WB                                          # noqa: E402
import door_clear_door_fast as _F                                # noqa: E402
WB.clear_door = _F.clear_door_fast
import door_motm_grasp_ctrl as G                                 # noqa: E402
import door_open_exec_ctrl as XO                                 # noqa: E402
import door_motm_force_offline as FO                             # noqa: E402（須在 SimulationApp 之後：它匯入專案控制模組）
import door_body_model as BM                                     # noqa: E402
import door_bind_logic as BL                                     # noqa: E402
import door_full_logic as FL                                     # noqa: E402
import door_motm_p2 as P2                                        # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics   # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
import door_far_link as FARL                                     # noqa: E402
import far_chain_align as FCA                                    # noqa: E402
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in _ROS_PREFIX and k_ not in _ROS_SNAP]:
    sys.modules.pop(k_, None)
sys.modules.update(_ROS_SNAP)
if sys.modules.get('rclpy') is not _RCLPY_REAL or not hasattr(rclpy, 'init'):
    raise SystemExit('rclpy 還原失敗 ⇒ 拒跑')
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SCAN_N, SCAN_MIN, SCAN_MAX, RANGE_MIN, RANGE_MAX = 360, -3.14159, 3.14159, 0.12, 10.0
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
LIDAR_Z = float(json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))['lidar_z'])
GRASP_END_ABS = -0.285                                               # 抓持段絕對終點（通過線沿線座標；同監督節點）：抓取位可行下界 −0.30 留停止行程 0.015
EST_BOUND_M, EST_BOUND_DEG, SYNC_M = 0.00285, 0.885, 0.05 * 0.011    # 鎖定預算項（同監督節點 TERM；far_lock_error_envelope.json 的已觀測包絡，工程預算）
CMD_TIMEOUT_S, ARM_CMD_TIMEOUT_S = 0.5, 0.2


def T3(p):
    c_, s_ = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c_, -s_], [s_, c_]]; T[:2, 3] = p[:2]
    return T


class Bridge(Node):
    """遠距前段與安全鏈的 ROS 介面（同 isaac_m8_house_far_capture.py）＋交接後底盤提案 /cmd_vel_grasp。"""

    def __init__(self):
        super().__init__('far_door_sim')
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.p_state = self.create_publisher(String, '/far/exec_state', 10)
        self.p_grasp = self.create_publisher(Twist, '/cmd_vel_grasp', 10)
        self.p_hoack = self.create_publisher(String, '/far/handover_ack', 10)
        self.rx_steps, self.ho_tried = [], False
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 20)
        self.create_subscription(Twist, '/cmd_vel_nav', self._nav_echo, 20)   # 選擇器輸出（平滑器輸入）：確認提案已送到平滑器前
        self.nav_echo, self.echo_wait = None, []
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(Bool, '/far/done', self._done, 10)
        self.create_subscription(String, '/far/abort', self._abort, 10)
        self.create_subscription(Float64MultiArray, '/far/handover', self._ho, 10)
        self.t, self.n = 0.0, 0
        self.cmd, self.cmd_t, self.acmd, self.acmd_t, self.done, self.abort, self.ho = np.zeros(3), -1.0, np.zeros(6), -1.0, False, None, None
        self.ca = None                                               # 交接後：ChainAlign

    def _cmd(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z], float)
        if np.isfinite(v).all():
            self.cmd, self.cmd_t = v, self.t
            self.rx_steps.append(self.n)
            if self.ca is not None:
                self.ca.on_release(self.n, v)

    def _nav_echo(self, m):
        self.nav_echo = np.array([m.linear.x, m.linear.y, m.angular.z], float)

    def _acmd(self, m):
        if len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t

    def _done(self, m):
        self.done = self.done or bool(m.data)

    def _abort(self, m):
        if self.abort is None:
            self.abort = str(m.data)

    def _ho(self, m):
        if self.ho is None and len(m.data) == 27 and np.isfinite(m.data).all():
            self.ho = np.array(m.data, float)

    def stamp(self):
        s_ = RosTime()
        s_.sec = int(self.t)
        s_.nanosec = int(round((self.t - int(self.t)) * 1e9))
        if s_.nanosec >= 1000000000:
            s_.sec, s_.nanosec = s_.sec + 1, s_.nanosec - 1000000000
        return s_

K, PG = G.K, WB.PG
ROBOT, DOOR = '/World/omni_bot', '/World/door'
JOINT = '/World/door/ideal_grasp_attach'
ARM = [f'joint{i}' for i in range(1, 7)]
DT = 0.01
NPH = G.NPH
XO_D = G.D_STEPS * G.PHYS                                            # 管線延遲（s）
V, H_SEG, L_LAT = 0.05, 2.0, 1.2
L_LEAD, T_LEAD = 0.01, 0.2                                         # 冒煙 2：前導 0.15 m 起步會穿過 x≈0.80 的腕部奇異（關節 5 過零）⇒ 前導縮到 1 cm（σ 全程 ≥0.031）
GOAL = math.radians(85.0)
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
GRASP_LIMIT_S, OPEN_LIMIT_S = 12.0, 40.0
HOLD_DR_MAX = 0.3                                                    # 相對零摩擦基線的保持力矩殘差差值上限（N·m）
HOLD_R_MAX, HOLD_V_MAX = 1.3, 0.01                                   # P2：靜置保持力矩殘差上限（N·m）、低速門檻（rad/s）
# 首段適用窗與滾動參數在 door_first_seg（LEAD 0.10／TOL 0.05／AHEAD 1.0／GAP 0.4 s）
_UNUSED_PLAN = (0.10, 0.05, 1.0)                  # 首段：名目交接在預測 READY 後 0.10 s（第 3、4 趟實況）；適用窗 [0.10, 0.10＋0.05]；過期後重算的預測提前量 1.0 s（規劃在模擬時間約 0.8 s）
# B3 有限剛度綁定（事前選定的一組固定值；SI）：平移 1.0e4 N/m、200 N·s/m；轉動 100 N·m/rad、2.0 N·m·s/rad。USD 轉動驅動以「每度」為單位 ⇒ 寫入值乘 π/180。
BIND_KT, BIND_CT, BIND_KR, BIND_CR = 1.0e4, 200.0, 100.0, 2.0
BIND_DEF_POS_M, BIND_DEF_ROT_DEG = 0.003, 1.5                        # 變形預算（相對綁定瞬間的夾爪—門扇相對位姿）：超過即停止（執行器轉送）
DIAG_STOP_T = 0.5                                                   # --diag：開門執行層時間到此即強制本拍「晚到」（走與第四趟相同的閉鏈一致停止流程）
PH = ['L', 'I', 'C', 'F', 'R', 'O', 'S']                            # 前導(HOLD)／插入／閉爪／確認／READY／開門／停止
PH_ZH = dict(L='前導接近（預抓保持、移動）', I='插入（移動中）', C='閉爪（移動中）', F='確認（移動中）', R='READY（移動中）', O='開門（P2 續接）', S='停止')


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


def ik(b, T, seed):
    q = np.asarray(seed, float).copy()
    for _ in range(60):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = T[:3, :3] @ Tc[:3, :3].T
        e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-11:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


def main():
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.reg))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    al = np.asarray(ident['alpha'], float)
    Rg, T_HG, a_H = PG.baseline_reference()
    T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
    Lf2 = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'log.npz'))['rows']
    E4 = Lf2[Lf2[:, 1] == 4][-1]
    q_pass = np.r_[E4[5:8]]                                          # 通過線：full_2 綁定底盤位姿（y、偏航；x 為窗口原點 −0.03）
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
        env_paths.append(f"/World/unknown/{o['name']}")
    b_start = np.array([float(x) for x in a.start.split(',')], float)   # 遠距起點（同 far_capture 三趟）
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(b_start[0]), float(b_start[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(b_start[2])))
    apply_split(stage, ROBOT)
    if bind_frictionless(stage, collision_prims(stage, ROBOT, lambda p: 'support_' in p) + collision_prims(stage, '/World/ground')) == 0:
        print('[motm] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    nm = lambda b: str(b).rstrip('/').split('/')[-1]              # noqa: E731
    grip = [b for b in bodies if nm(b) == 'uflite_gripper_link']
    fingers = sorted(b for b in bodies if nm(b) in ('uflite_finger1', 'uflite_finger2'))
    others = [b for b in bodies if nm(b) in tuple(f'link{i}' for i in range(1, 7)) + ('link_eef',)]
    base_b = [b for b in bodies if nm(b) == 'base_link']
    if len(grip) != 1 or len(fingers) != 2 or len(others) != 7 or len(base_b) != 1:
        print('[motm] 剛體不符 ⇒ 拒跑', flush=True)
        return 16
    leaf = dauth['leaf_prim']
    v_f = [RigidPrim(prim_paths_expr=str(f), name=f'v_{nm(f)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf]) for f in fingers]
    v_o = [RigidPrim(prim_paths_expr=str(b), name=f'v_{nm(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths) for b in others + grip]
    v_base = RigidPrim(prim_paths_expr=str(base_b[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths)
    v_grip = RigidPrim(prim_paths_expr=str(grip[0]), name='v_grip_pose')
    v_leaf = RigidPrim(prim_paths_expr=leaf, name='v_leaf_pose')
    fric_usd = {}
    FRIC_ON = a.arm_friction_model in ('urdf_effort', 'no_passive_arm_friction')
    FRIC_VAL = 0.0 if a.arm_friction_model == 'no_passive_arm_friction' else float(a.arm_fric_value)
    if FRIC_ON:                                                      # world.reset 之前：逐軸 API（該軸優先於舊的無因次 jointFriction）；顯式保留原速度上限與 armature
        for pr in stage.Traverse():
            if pr.GetName() in ARM and pr.IsA(UsdPhysics.RevoluteJoint):
                old = PhysxSchema.PhysxJointAPI(pr)
                mv = old.GetMaxJointVelocityAttr().Get() if old.GetMaxJointVelocityAttr() else None
                am = old.GetArmatureAttr().Get() if old.GetArmatureAttr() and old.GetArmatureAttr().HasAuthoredValue() else 0.0
                jf = old.GetJointFrictionAttr().Get() if old.GetJointFrictionAttr() else None
                if mv is None or not math.isfinite(float(mv)):
                    print('[motm] 手臂關節原速度上限讀不到 ⇒ 拒跑', flush=True)
                    return 31
                pr.ApplyAPI('PhysxJointAxisAPI', 'angular')
                old.CreateJointFrictionAttr().Set(0.0)                # P1（Codex 20261115）：舊的無因次係數＝單位錯置的那一項，明設 0；摩擦由逐軸 1.0 N·m 表示
                for nm_, v_ in (('staticFrictionEffort', FRIC_VAL), ('dynamicFrictionEffort', FRIC_VAL), ('viscousFrictionCoefficient', 0.0), ('maxJointVelocity', float(mv)), ('armature', float(am))):
                    pr.CreateAttribute('physxJointAxis:angular:' + nm_, Sdf.ValueTypeNames.Float).Set(float(v_))
                dr = UsdPhysics.DriveAPI(pr, 'angular')
                fric_usd[pr.GetName()] = dict(legacy_jointFriction_unitless_imported=None if jf is None else float(jf), legacy_jointFriction_unitless_set=0.0, axis_static_Nm=FRIC_VAL, axis_dynamic_Nm=FRIC_VAL, axis_viscous=0.0, maxJointVelocity_deg_s=float(mv), armature=float(am),
                                              drive_maxForce_Nm=None if not dr.GetMaxForceAttr() else float(dr.GetMaxForceAttr().Get()))
        if sorted(fric_usd) != sorted(ARM):
            print('[motm] 手臂關節逐軸摩擦設定不完整 ⇒ 拒跑', sorted(fric_usd), flush=True)
            return 32
    world.reset()
    for v in v_f + v_o + [v_base, v_grip, v_leaf]:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    hprim = stage.GetPrimAtPath(f'{DOOR}/leaf/handle_bar')

    def truth_handle():                                            # 把手 prim 真值：BindGate 實際抓持條件與事後評估用（本檔不作把手估計來源）
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprim)
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    hinge = DA3.readback(stage, DOOR)
    if abs(hinge['drive_stiffness']) > 1e-12 or abs(hinge['drive_damping']) > 1e-12 or not hinge['is_rigid_body']:
        print('[motm] 鉸鏈不是被動 ⇒ 拒跑', flush=True)
        return 64
    try:                                                           # 門扇剛體角阻尼讀回（資產設定有 2.0；「鉸鏈零驅動」≠ 門無阻尼）
        leaf_damp = dict(angular=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(dauth['leaf_prim'])).GetAngularDampingAttr().Get(),
                         linear=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(dauth['leaf_prim'])).GetLinearDampingAttr().Get())
    except Exception as e_:                                        # noqa: BLE001
        leaf_damp = dict(error=repr(e_))
    bar_axis = 'XYZ'.index(str(hinge['handle_bar']['axis']))
    bar_r, bar_hl = float(hinge['handle_bar']['radius']), 0.5 * float(hinge['handle_bar']['length'])
    T_true0 = truth_handle()
    qa = np.array([float(x) for x in a.arm.split(',')], float)          # 遠距起始手臂構型（online_1 far_live 構型）
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(qa[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = XO.FINGER_OPEN, XO.FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    kps_rb, _ = robot.get_articulation_controller().get_gains()
    # ---- 後端（PhysX tensor view）有效屬性讀回：摩擦三項、速度／力矩上限、armature、增益
    arm_backend = dict(model=a.arm_friction_model)
    try:
        pv = robot._articulation_view._physics_view
        fp = np.asarray(pv.get_dof_friction_properties(), float)[0]
        arm_backend.update(dof=[names[i] for i in idx], friction_static_dynamic_viscous=[[float(x) for x in fp[i]] for i in idx],
                           max_velocity=[float(np.asarray(pv.get_dof_max_velocities(), float)[0][i]) for i in idx], max_force=[float(np.asarray(pv.get_dof_max_forces(), float)[0][i]) for i in idx],
                           armature=[float(np.asarray(pv.get_dof_armatures(), float)[0][i]) for i in idx],
                           kp=[float(np.asarray(pv.get_dof_stiffnesses(), float)[0][i]) for i in idx], kd=[float(np.asarray(pv.get_dof_dampings(), float)[0][i]) for i in idx],
                           legacy_friction_coefficients=[float(np.asarray(pv.get_dof_friction_coefficients(), float)[0][i]) for i in idx] if hasattr(pv, 'get_dof_friction_coefficients') else None,
                           units='摩擦：static／dynamic＝N·m，viscous＝後端單位（tensor API；角度制或弧度制依後端，值為 0）；max_velocity＝後端單位（rad/s）；max_force＝N·m', usd=fric_usd)
    except Exception as e:                                            # noqa: BLE001
        arm_backend['error'] = repr(e)[:300]
    print('[motm] arm_backend', json.dumps(arm_backend, ensure_ascii=False)[:1500], flush=True)
    if FRIC_ON:
        fpv = np.array(arm_backend.get('friction_static_dynamic_viscous', [[np.nan] * 3]), float)
        mvb, mfb = np.array(arm_backend.get('max_velocity', [np.nan]), float), np.array(arm_backend.get('max_force', [np.nan]), float)
        ok_f = ('error' not in arm_backend and fpv.shape == (6, 3) and np.isfinite(fpv).all() and np.allclose(fpv[:, 0], FRIC_VAL, atol=1e-6) and np.allclose(fpv[:, 1], FRIC_VAL, atol=1e-6) and np.allclose(fpv[:, 2], 0.0, atol=1e-9)
                and np.isfinite(mvb).all() and np.allclose(mvb, [math.radians(fric_usd[names[i]]['maxJointVelocity_deg_s']) for i in idx], rtol=1e-3)
                and np.isfinite(mfb).all() and np.allclose(mfb, [fric_usd[names[i]]['drive_maxForce_Nm'] for i in idx], rtol=1e-3)
                and np.allclose(arm_backend['kp'], a.kp, rtol=1e-3) and np.allclose(arm_backend['kd'], a.kd, rtol=1e-3)
                and arm_backend.get('legacy_friction_coefficients') is not None and np.allclose(arm_backend['legacy_friction_coefficients'], 0.0, atol=1e-9))
        if not ok_f:
            print('[motm] 手臂關節後端有效屬性讀回不符（摩擦／速度上限／力矩上限／增益）⇒ 拒跑', flush=True)
            json.dump(dict(refused='arm_backend_readback', arm_backend=arm_backend), open(os.path.join(a.out, 'refused.json'), 'w'), ensure_ascii=False, indent=1)
            return 33
    if not all(abs(float(np.asarray(kps_rb)[i]) - a.kp) <= 1e-3 * a.kp for i in idx):
        print('[motm] 增益讀回不符 ⇒ 拒跑', flush=True)
        return 66
    bodies_arm = BM.arm_bodies(K)
    q9a = np.r_[b_start, qa]
    s0 = qa - BM.arm_gravity(K, q9a, bodies_arm) / a.kp               # 靜置設定點：q−s＝G/kp
    tgt = q0.copy()
    for i, j in zip(idx, range(6)):
        tgt[i] = float(s0[j])
    tgt[f1] = XO.FINGER_OPEN

    def apply():
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32)[act], joint_indices=act))

    def base_cmd(u, yw):
        robot.set_linear_velocity(np.array([u[0] * math.cos(yw) - u[1] * math.sin(yw), u[0] * math.sin(yw) + u[1] * math.cos(yw), 0.0], np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, u[2]], np.float32))

    def q9():
        p, qt = robot.get_world_pose()
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), jp[idx]], jp

    def door_q():
        _, o = v_leaf.get_world_poses()
        w, x, y, z = (float(v) for v in np.asarray(o)[0])
        return -math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    meta = robot._articulation_view._metadata
    j6 = int(meta.joint_indices['joint6']) + 1

    def wrist_F6():
        return np.asarray(robot.get_measured_joint_forces(), float)[j6][:6].copy()

    # ---- B2：外界對夾爪的旋量（世界座標、作用點 TCP）＝−(joint6 量測轉世界＋下游重力−自由狀態基線)
    wb = dict(F=[], M=[], bF=None, bM=None)

    def wrench_parts(q9_, F6_):
        Fg_, Mg_, T6_ = FO.down_gravity(q9_)
        R6_ = T6_[:3, :3]
        return R6_ @ F6_[:3] + Fg_, R6_ @ F6_[3:] + Mg_, T6_

    def ext_wrench(q9_, F6_):
        if wb['bF'] is None:
            return None
        rF_, rM_, T6_ = wrench_parts(q9_, F6_)
        Fe_ = -(rF_ - wb['bF'])
        Mo_ = -(rM_ - T6_[:3, :3] @ wb['bM'])
        pt_ = K.fk(q9_, 'link_tcp')[:3, 3]
        w_ = (Fe_, Mo_ - np.cross(pt_ - T6_[:3, 3], Fe_))
        return w_ if (np.isfinite(w_[0]).all() and np.isfinite(w_[1]).all()) else None

    # ---- --diag：每物理步三個時點（覆寫前 A／覆寫後 B／物理步後 C）的根連桿位姿與六維速度＋其他取證訊號
    dlog, dstate, derr = [], {}, []
    v_l6 = [v for v, b in zip(v_o, others + grip) if nm(b) == 'link6'][0]

    def _root():
        p_, q_ = robot.get_world_pose()
        return np.r_[np.asarray(p_, float), np.asarray(q_, float), np.asarray(robot.get_linear_velocity(), float), np.asarray(robot.get_angular_velocity(), float)]

    def _pq(view):
        p_, q_ = view.get_world_poses()
        return np.r_[np.asarray(p_, float)[0], np.asarray(q_, float)[0]]

    def diag_A():
        if a.diag:
            try:
                dstate['A'] = _root()
            except Exception as e:                                     # noqa: BLE001
                derr.append('A:' + repr(e)[:120])

    def diag_B():
        if a.diag:
            try:
                dstate['B'] = _root()
            except Exception as e:                                     # noqa: BLE001
                derr.append('B:' + repr(e)[:120])

    def diag_C(tt, mode_i, twist, pipe_, u_last_):
        if not a.diag:
            return
        try:
            C_ = _root()
            jp_ = np.asarray(robot.get_joint_positions(), float)
            jv_ = np.asarray(robot.get_joint_velocities(), float)
            try:
                me_ = np.asarray(robot.get_measured_joint_efforts(), float)
            except Exception:                                          # noqa: BLE001
                me_ = np.full(len(jp_), np.nan)
            try:
                ae_ = np.asarray(robot.get_applied_joint_efforts(), float)
            except Exception:                                          # noqa: BLE001
                ae_ = np.full(len(jp_), np.nan)
            F_ = np.asarray(robot.get_measured_joint_forces(), float)
            dlog.append(np.r_[tt, mode_i, dstate.get('A', np.full(13, np.nan)), dstate.get('B', np.full(13, np.nan)), C_, np.asarray(twist, float), jp_, jv_, me_, ae_, np.asarray(tgt, float), F_[j6][:6], F_[j6 - 1][:6],
                              _pq(v_l6), _pq(v_grip), _pq(v_leaf), np.concatenate([np.asarray(u_, float) for u_ in pipe_]), np.asarray(u_last_, float), door_q()])
        except Exception as e:                                         # noqa: BLE001
            derr.append('C:' + repr(e)[:160])

    def rel_T():
        pg, qg = v_grip.get_world_poses()
        pl, ql = v_leaf.get_world_poses()
        Tg_, Tl_ = BL.pose_from_pq(pg[0], qg[0]), BL.pose_from_pq(pl[0], ql[0])
        if Tg_ is None or Tl_ is None:
            return None, Tg_, Tl_
        return np.linalg.inv(Tl_) @ Tg_, Tg_, Tl_

    def finger_contacts():
        Th = truth_handle()
        out_ = []
        for v in v_f:
            d_ = v.get_contact_force_data(dt=DT)
            fn, pts, cnt, st = np.asarray(d_[0], float).reshape(-1), np.asarray(d_[1], float).reshape(-1, 3), np.asarray(d_[4]).reshape(-1), np.asarray(d_[5]).reshape(-1)
            sel = np.concatenate([np.arange(int(s_), int(s_) + int(c_)) for s_, c_ in zip(st, cnt)]) if int(cnt.sum()) else np.zeros(0, int)
            fb, fo, _ = BL.classify_points(pts[sel], fn[sel], Th[:3, 3], Th[:3, bar_axis] / np.linalg.norm(Th[:3, bar_axis]), bar_hl, bar_r)
            out_.append((fb, fo))
        return out_

    def other_contacts():
        mx, who = 0.0, None
        for v, b in zip(v_o + [v_base], others + grip + base_b):
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, 3)
            n_ = np.linalg.norm(M, axis=1)
            if len(n_) and float(n_.max()) > mx:
                mx, who = float(n_.max()), nm(b)
        return mx, who
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([2.4, -2.2, 2.2]), target=np.array([float(b_start[0]) - 0.3, float(b_start[1]) + 0.3, 0.8]))
    except Exception:                                              # noqa: BLE001
        pass
    res = dict(run=os.path.basename(a.out.rstrip('/')), mode='smoke' if a.smoke else 'nominal',
               scope='遠距整合：把手來源＝腕部相機實際影像（凍結偵測＋幾何估計）；通過線＝途中第一格有效且通過關聯的估計；抓持與開門用鎖定任務參考（原始時間戳，不更新）；BindGate 的實際抓持條件以把手 prim 真值核（同門旁各趟）；理想抓持（FixedJoint）；理想里程；非即時；範圍：開到 85° 停穩為止（不解綁、不退出、不穿門）',
               init=dict(b_start=b_start.tolist(), q_arm_start=qa.tolist(), source='遠距起點（同 far_capture 三趟）；手臂＝online_1 far_live 構型；靜置後從靜止起步，起跑後不瞬移', v_pass=V,
                         note='通過線不是場景初始條件：由本趟途中影像估計在線決定（見 ros_merge.json anchor）'),
               gains=dict(kp=a.kp, kd=a.kd), arm_backend=arm_backend, door_readback=dict(hinge=hinge, leaf_rigid_body_damping=leaf_damp),
               finger=dict(command='CLOSE 段由張開斜降到 0（夾緊，同停車版）', geometry='兩指各自實測開度；CLOSE 段掃掠另含預測閉合開度'))
    # ---- 靜置
    quat0 = np.array([math.cos(b_start[2] / 2), 0.0, 0.0, math.sin(b_start[2] / 2)], np.float32)
    jv_hist, ct_hist, ct_err = [], [], []
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([b_start[0], b_start[1], 0.0], np.float32), quat0)
        apply()
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
        jv_hist.append(np.asarray(robot.get_joint_velocities(), float)[idx])
        try:
            fc_ = finger_contacts()
            ch_ = [float(fc_[0][0]), float(fc_[0][1]), float(fc_[1][0]), float(fc_[1][1])]
            for v_, b_ in zip(v_o + [v_base], others + grip + base_b):   # 各連桿原始通道逐一取值（先核有限，再彙整）
                ch_ += [float(x_) for x_ in np.linalg.norm(np.asarray(v_.get_contact_force_matrix(dt=DT), float).reshape(-1, 3), axis=1)]
            ct_hist.append(float(max(ch_)) if (len(ch_) > 4 and np.isfinite(ch_).all()) else float('nan'))
        except Exception as e:                                         # noqa: BLE001
            ct_hist.append(float('nan'))
            ct_err.append(repr(e)[:160])
    qs, jps = q9()
    # P2（Codex 20261115）：任務運動前的工程拒跑檢查——未綁定、無接觸、實測低速持續 0.3 s 後，r＝kp(s−q)＋kd(0−q̇)＋G(q)（速度目標＝0）
    jv_tail = np.abs(np.array(jv_hist[-int(round(0.3 / DT)):]))
    r_hold = a.kp * (s0 - qs[3:]) + a.kd * (0.0 - jv_hist[-1]) + BM.arm_gravity(K, qs, bodies_arm)
    hold_chk = dict(low_speed_0p3s=bool(jv_tail.max() <= HOLD_V_MAX), joint_rate_abs_max_0p3s=round(float(jv_tail.max()), 6), r_Nm=[round(float(x), 3) for x in r_hold], limit_Nm=HOLD_R_MAX, v_limit=HOLD_V_MAX,
                    note='r 含重力模型與驅動近似誤差：超過不證明新舊摩擦疊加，通過也不證明摩擦已校準；1.3＝1.0＋事前工程容差 0.3')
    n_need = int(round(0.3 / DT))
    ct_tail = np.array(ct_hist[-n_need:], float)
    hold_chk.update(contact_read_errors=ct_err[:3], n_contact_read_errors=len(ct_err), n_samples=int(min(len(jv_tail), len(ct_tail))), n_need=n_need, contact_force_max_n=None if not np.isfinite(ct_tail).all() else round(float(ct_tail.max()), 6),
                    no_contact=bool(len(ct_tail) == n_need and np.isfinite(ct_tail).all() and float(ct_tail.max()) <= 1e-6), finite=bool(np.isfinite(jv_tail).all() and np.isfinite(r_hold).all()))
    if a.arm_friction_model == 'no_passive_arm_friction':            # 相對凍結的零摩擦基線 r0（同起始構型；不得用本趟讀數重新歸零）
        B0 = json.load(open(os.path.join(HERE, 'experiments', '20261115_arm_hold_baseline_zero_friction.json')))
        dr = r_hold - np.asarray(B0['r0_Nm'], float)
        hold_chk.update(baseline=dict(file='experiments/20261115_arm_hold_baseline_zero_friction.json', source_run=B0['source_run'], source_sha256=B0['source_result_sha256'], r0_Nm=B0['r0_Nm']),
                        r_minus_r0_Nm=[round(float(x), 3) for x in dr], limit_rel_Nm=HOLD_DR_MAX, criterion='|r−r0|≤0.3 N·m（工程一致性檢查）')
        hold_chk['baseline_note'] = '凍結基線 r0 是門旁起始構型的值；本趟起始構型不同 ⇒ r−r0 只記錄，判定改用絕對殘差 |r|≤HOLD_R_MAX（原 legacy 分支的工程判準）'
        ok_r = bool(np.isfinite(r_hold).all() and float(np.abs(r_hold).max()) <= HOLD_R_MAX)
    else:
        ok_r = bool(float(np.abs(r_hold).max()) <= HOLD_R_MAX)
    hold_chk['ok'] = bool(hold_chk['low_speed_0p3s'] and hold_chk['finite'] and len(jv_tail) == n_need and hold_chk['no_contact'] and ok_r)
    print('[motm] hold_check', json.dumps(hold_chk, ensure_ascii=False), flush=True)
    if FRIC_ON and not hold_chk['ok'] and not a.probe:
        print('[motm] 靜置保持力矩殘差檢查不過 ⇒ 拒跑', flush=True)
        json.dump(dict(refused='hold_residual', hold_check=hold_chk, arm_backend=arm_backend), open(os.path.join(a.out, 'refused.json'), 'w'), ensure_ascii=False, indent=1)
        return 34
    off_static = qs[3:] - s0                                          # 靜置實測靜態偏置（含未建模）
    door0 = door_q()
    res['settle'] = dict(hold_check=hold_chk, base=qs[:3].tolist(), arm_err_vs_pregrasp=[round(float(x), 5) for x in (qs[3:] - qa)], off_static=off_static.tolist(), door_deg=round(math.degrees(door0), 5))
    if a.probe:
        # ---- 手臂摩擦探針：joint4 設定點以梯形速度（0.1 rad/s、加速度 0.5 rad/s²）走 +0.10 rad → 停 PROBE_HOLD → 回 −0.10 rad → 停；其餘關節設定點不動；底盤零命令
        PV, PA, PD_, PHOLD = 0.1, 0.5, 0.10, 3.0
        def prof(tt):                                                  # 梯形：回 (位移, 速度)
            ta = PV / PA
            tc = (PD_ - PV * ta) / PV
            if tt < ta:
                return 0.5 * PA * tt * tt, PA * tt
            if tt < ta + tc:
                return 0.5 * PA * ta * ta + PV * (tt - ta), PV
            if tt < 2 * ta + tc:
                td = 2 * ta + tc - tt
                return PD_ - 0.5 * PA * td * td, PA * td
            return PD_, 0.0
        Tm = 2 * PV / PA + (PD_ - PV * PV / PA) / PV
        j4 = 3
        s_base = np.array([tgt[i] for i in idx], float)
        plog, perr = [], []
        tt_ = 0.0
        while tt_ < 2 * (Tm + PHOLD) - 1e-9:
            if tt_ < Tm + PHOLD:
                d_, v_ = prof(tt_)
            else:
                d2, v2 = prof(tt_ - Tm - PHOLD)
                d_, v_ = PD_ - d2, -v2
            tgt[idx[j4]] = float(s_base[j4] + d_)
            apply()
            base_cmd(np.zeros(3), 0.0)
            world.step(render=True)
            tt_ += DT
            try:
                jp_ = np.asarray(robot.get_joint_positions(), float)[idx]
                jv_ = np.asarray(robot.get_joint_velocities(), float)[idx]
                me_ = np.asarray(robot.get_measured_joint_efforts(), float)[idx]
                ch_ = []
                for vv_ in v_o + [v_base] + v_f:
                    ch_ += [float(x_) for x_ in np.linalg.norm(np.asarray(vv_.get_contact_force_matrix(dt=DT), float).reshape(-1, 3), axis=1)]
                qq_, _ = q9()
                plog.append(np.r_[tt_, d_, v_, [tgt[i] for i in idx], jp_, jv_, me_, wrist_F6(), max(ch_) if ch_ else np.nan, BM.arm_gravity(K, qq_, bodies_arm), qq_[:3]])
            except Exception as e:                                     # noqa: BLE001
                perr.append(repr(e)[:160])
        P_ = np.array(plog)
        cols_ = ['t', 's4_disp', 's4_vel_ref'] + [f'target_{k_}' for k_ in range(1, 7)] + [f'q_{k_}' for k_ in range(1, 7)] + [f'qd_{k_}' for k_ in range(1, 7)] + [f'meas_effort_{k_}' for k_ in range(1, 7)] + [f'j6_wrench_{k_}' for k_ in range(6)] + ['contact_max_n'] + [f'G_model_{k_}' for k_ in range(1, 7)] + ['bx', 'by', 'byaw']
        np.savez_compressed(os.path.join(a.out, 'probe_steps.npz'), P=P_, cols=np.array(cols_))
        def seg(t0_, t1_):
            S_ = P_[(P_[:, 0] > t0_) & (P_[:, 0] <= t1_)]
            still = bool(np.abs(S_[:, 15:21]).max() < XO.STOP_STILL['arm_rps'])          # 既有停穩判準（手臂 2e-3 rad/s），該段最後 0.3 s 全程
            L_ = S_[-1]
            gi_ = cols_.index('G_model_1')                              # 更正：重力欄依欄名取（原寫 28:34 取到旋量尾段與接觸欄）
            r_ = a.kp * (L_[3:9] - L_[9:15]) - a.kd * L_[15:21] + L_[gi_:gi_ + 6]
            return dict(t=[round(t0_, 2), round(t1_, 2)], settled_per_STOP_STILL=still, qd_abs_max=round(float(np.abs(S_[:, 15:21]).max()), 6), qd4_mean=round(float(S_[:, 18].mean()), 6),
                        target_minus_q_mrad=[round(float(x_) * 1e3, 3) for x_ in (L_[3:9] - L_[9:15])], r_nominal_pd_plus_G_Nm=[round(float(x_), 3) for x_ in r_],
                        meas_effort_Nm=[round(float(x_), 3) for x_ in S_[:, 21:27].mean(axis=0)], contact_max_n=float(np.nanmax(S_[:, 33])))
        mv = P_[(P_[:, 0] > PV / PA) & (P_[:, 0] <= Tm - PV / PA)]
        res['probe'] = dict(fric_value_Nm=float(a.arm_fric_value), model=a.arm_friction_model, motion=dict(joint='joint4', disp_rad=PD_, v_rad_s=PV, a_rad_s2=PA, hold_s=PHOLD, velocity_target='0（位置驅動；未送速度目標）'),
                            after_fwd_last_0p3s=seg(Tm + PHOLD - 0.3, Tm + PHOLD), after_back_last_0p3s=seg(2 * (Tm + PHOLD) - 0.3, 2 * (Tm + PHOLD)),
                            fwd_cruise=dict(lag_target_minus_q4_mrad_mean=round(float((mv[:, 6] - mv[:, 12]).mean()) * 1e3, 3), qd4_mean=round(float(mv[:, 18].mean()), 5), meas_effort4_mean=round(float(mv[:, 24].mean()), 3)),
                            n_rows=int(len(P_)), n_err=len(perr), errors=perr[:3], contact_max_n=float(np.nanmax(P_[:, 33])),
                            note='r＝名目 PD＋重力模型殘差（不是量到的摩擦力矩）；settled 才可稱靜止殘差')
        res['result'] = 'probe_done'
        res['aborted'] = None
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print('[motm] probe', json.dumps(clean(res['probe']), ensure_ascii=False)[:1800], flush=True)
        return 0
    # ================= 遠距前段（ROS：原導航鏈＋監督節點；本行程只提供感測與受控對象）=================
    eef = [b for b in bodies if nm(b) == 'link_eef'][0]
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    cam = Camera(prim_path=f'{eef}/wrist_cam', resolution=CAM['res'], frequency=CAM['hz'])
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.set_horizontal_aperture(CAM['aperture'])
    cam.set_focal_length(CAM['aperture'] / (2.0 * math.tan(math.radians(CAM['hfov_deg']) / 2.0)))
    cam.set_clipping_range(*CAM['clip'])
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    for _ in range(5):
        apply(); base_cmd(np.zeros(3), 0.0); world.step(render=True)
    Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    def scan(q_):
        org, out_ = [float(q_[0]), float(q_[1]), LIDAR_Z], []
        for i_ in range(SCAN_N):
            th = float(q_[2]) + SCAN_MIN + i_ * SCAN_INC
            best = [float('inf')]

            def rep_(hit, best=best):
                coll = str(getattr(hit, 'collision', '') or '')
                body = str(getattr(hit, 'rigid_body', '') or getattr(hit, 'rigidBody', '') or '')
                if coll.startswith(ROBOT) or body.startswith(ROBOT):
                    return True
                d_ = float(getattr(hit, 'distance', 0.0))
                if RANGE_MIN <= d_ < best[0]:
                    best[0] = d_
                return True
            query.raycast_all(org, [math.cos(th), math.sin(th), 0.0], RANGE_MAX, rep_)
            out_.append(best[0])
        return out_
    rclpy.init()
    node = Bridge()
    obs = FARL.HandleObs()
    T0 = float(world.current_time)
    fdir = os.path.join(a.out, 'frames'); os.makedirs(fdir, exist_ok=True)
    hist, fmeta, far_rows, far_ev = {}, [], [], []
    last_rf, t_next_cam, seq = [None], 0.0, 0
    s_far = s0.copy()
    pipe_far = [np.zeros(9) for _ in range(G.D_STEPS)]
    scan_min = float('nan')
    grav_far = lambda q9_: BM.arm_gravity(K, q9_, bodies_arm)      # noqa: E731
    G_rest = grav_far(qs)

    def ros_step(q_, jp_, s_now, pipe_now, state):
        """每物理步：/clock；每 2 步：里程、關節、/far/truth、狀態；每 10 步：掃描。之後處理 ROS 回呼。"""
        ck = Clock(); ck.clock = node.stamp(); node.p_clock.publish(ck)
        if node.n % 2 == 0:
            m_ = Odometry(); m_.header.stamp = node.stamp(); m_.header.frame_id, m_.child_frame_id = 'odom', 'base_footprint'
            m_.pose.pose.position.x, m_.pose.pose.position.y = float(q_[0]), float(q_[1])
            m_.pose.pose.orientation.w, m_.pose.pose.orientation.z = math.cos(q_[2] / 2), math.sin(q_[2] / 2)
            node.p_odom.publish(m_)
            js = JointState(); js.header.stamp = node.stamp(); js.name = ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in q_[3:]] + [float(jp_[f1]), float(jp_[f2])]
            node.p_js.publish(js)
            s_end = s_now + DT * sum(p_[3:] for p_ in pipe_now)
            node.p_truth.publish(Float64MultiArray(data=[node.t, *[float(x) for x in q_], *[float(x) for x in s_now], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(pipe_now)]]))
            node.p_state.publish(String(data=state))
        if node.n % 10 == 0:
            r_ = scan(q_)
            m_ = LaserScan(); m_.header.stamp = node.stamp(); m_.header.frame_id = 'lidar_link'
            m_.angle_min, m_.angle_max, m_.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
            m_.range_min, m_.range_max = RANGE_MIN, RANGE_MAX
            m_.ranges = [float(x) if math.isfinite(x) else float('inf') for x in r_]
            node.p_scan.publish(m_)
        for _ in range(4):
            rclpy.spin_once(node, timeout_sec=0.0)

    def wait_forwarded(u_):
        """提案送出後、推進模擬時鐘前：等選擇器把它轉送到平滑器輸入（/cmd_vel_nav 回聲值相符），最多 0.2 s 牆鐘（非即時：等待期間物理不步進）。
        far_door_4：選擇器轉送延遲使平滑器在提案到達前取樣，第 7 筆被跳過。回 (ok, 等待 ms)。"""
        w0_ = time.perf_counter()
        while time.perf_counter() - w0_ < 0.2:
            rclpy.spin_once(node, timeout_sec=0.002)
            if node.nav_echo is not None and float(np.abs(node.nav_echo - np.asarray(u_, float)).max()) <= 1e-6:
                node.echo_wait.append((time.perf_counter() - w0_) * 1e3)
                return True, (time.perf_counter() - w0_) * 1e3
        node.echo_wait.append(float('nan'))
        return False, 200.0

    def far_event(what, **kw):
        far_ev.append(dict(kw, t=round(node.t, 3), what=what))
        print(f'[motm] far t={node.t:.2f} {what} {json.dumps(clean(kw), ensure_ascii=False)[:300]}', flush=True)
    far_out = None
    HOX = {}

    def est_fn_lock(t_):
        return HOX['T_lock'], 1, HOX['lock']['t_obs_rel']                 # 鎖定任務參考（原始時間戳；不更新）

    def lock_budget(t_, q9_, ep_, er_):
        lk_ = HOX['lock']
        pos = max(EST_BOUND_M, lk_['spread']) + SYNC_M + float(ep_)      # 估計包絡（工程預算）＋同步＋手臂 TCP 實測；理想里程分支里程項為零
        ang = EST_BOUND_DEG + lk_['dang'] + float(er_)
        return dict(pos_m=round(pos, 5), ang_deg=round(ang, 3), ok=bool(pos <= FARL.LOCK_BUDGET['pos_m'] and ang <= FARL.LOCK_BUDGET['ang_deg']), terms=dict(est_m=max(EST_BOUND_M, lk_['spread']), sync_m=SYNC_M, odom_m=0.0, arm_tcp_m=round(float(ep_), 5)))
    w_far0 = time.perf_counter()
    while node.t < a.far_limit_s:
        t_sim = float(world.current_time)
        node.t = t_sim - T0
        q, jp = q9()
        pos_r, q_r = cam.get_world_pose(camera_axes='ros')
        Tr = np.eye(4); Tr[:3, :3] = W0.wxyz_to_R(np.asarray(q_r, float)); Tr[:3, 3] = np.asarray(pos_r, float)
        hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), T_render=Tr)
        if len(hist) > 300:
            for kk in sorted(hist)[:len(hist) - 300]:
                hist.pop(kk, None)
        oc_, ow_ = other_contacts()
        fc_ = finger_contacts()
        cmax = max(oc_, fc_[0][0], fc_[0][1], fc_[1][0], fc_[1][1])
        ddeg = abs(math.degrees(door_q() - door0))
        if not (np.isfinite(q).all() and math.isfinite(cmax)) or cmax > 0.05 or ddeg > 0.05:   # 遠距前段保護：任何接觸／門角變化 ⇒ 模擬工程中止（不再步進）
            far_out = dict(result='far_engineering_abort', why='contact_or_door_or_nonfinite', contact_n=cmax, who=ow_, door_deg=ddeg)
            far_event('latched_sim_engineering_abort', **far_out)
            node.p_state.publish(String(data='latched:' + far_out['why']))
            break
        ros_step(q, jp, s_far, pipe_far, 'run')
        if node.abort is not None:
            far_out = dict(result='far_engineering_abort', why='supervisor_abort', detail=node.abort[:200])
            far_event('latched_sim_engineering_abort', **far_out)
            break
        if node.ho is not None and not node.ho_tried:
            ph_ = FCA.best_phase(node.rx_steps)
            if ph_ is None or node.n % NPH == ph_:                     # 在安全鏈節拍的合適相位接手（名目放行延遲約 6 步，落在 8 步管線內）
                node.ho_tried = True
                HO = node.ho
                HOX.update(T_lock=HO[1:17].reshape(4, 4).copy(), t_far_ho=node.t,
                           lock=dict(t_obs=float(HO[17]), age_at_lock_s=float(HO[18]), dang=float(HO[19]), spread=float(HO[20]), p0=HO[21:23].copy(), d=HO[23:25].copy(), yaw=float(HO[25]), ver=int(HO[26]), t_obs_rel=float(HO[17]) - node.t))
                off_ho = off_static + (grav_far(q) - G_rest) / a.kp    # 靜態偏置：靜置實測錨＋重力隨構型（同 K2 模型）
                snap = dict(s=s_far.copy(), pipe=[u_.copy() for u_ in pipe_far], u_last=pipe_far[-1].copy(), u_prev=pipe_far[-1 - NPH].copy(), off_static=off_ho.copy())
                w_try = time.perf_counter()
                # 以當下快照建立獨立的抓持控制器實例並試算首拍：不發任何命令、不動真管線（全部是複本）。通過 ⇒ 這個實例與它的首拍輸出就是接手後的第一拍（同一狀態、同一步）；不過 ⇒ 丟棄、回 nack、前段所有權不變
                a_to = float(-((q[:2] - HOX['lock']['p0']) @ HOX['lock']['d']))   # 接手當下的沿線位置（通過線真實座標）
                HOX.update(a_to=a_to, win=(L_LEAD, a_to - GRASP_END_ABS))      # 控制器窗口以接手位置為原點 ⇒ 換算成相對量，絕對終點固定在 GRASP_END_ABS
                gc_try = G.MovingGraspCtrl(ident, est_fn_lock, HOX['lock']['d'].copy(), V, HOX['win'], snap['s'].copy(), [u_.copy() for u_ in snap['pipe']], snap['u_last'].copy(), snap['u_prev'].copy(), snap['off_static'].copy(),
                                           sched=dict(hold_s=0.3 + T_LEAD), pool=POOL, fastkin=True, deadline_ms=50.0, lock_budget_fn=lock_budget)
                o_try = gc_try.step(0.0, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))
                ok_try = (a_to - GRASP_END_ABS >= 0.115) and gc_try.stop is None and gc_try.abort is None and not o_try.get('abort') and np.isfinite(np.asarray(o_try['base_twist'], float)).all() and np.isfinite(np.asarray(o_try['arm_target'], float)).all()
                trial = dict(along_at_takeover=round(a_to, 4), window_rel=[float(x_) for x_ in HOX['win']], ok=bool(ok_try), stop=gc_try.stop, abort=gc_try.abort, wall_ms=round((time.perf_counter() - w_try) * 1e3, 1), phase=ph_, n=node.n, rx_mod=[x_ % NPH for x_ in node.rx_steps[-8:]], budget=gc_try.lock_budget,
                             first_cmd=[float(x_) for x_ in gc_try.u_last], prev_cmd=[float(x_) for x_ in snap['u_last']], first_cmd_step_max=float(np.abs(gc_try.u_last - snap['u_last']).max()))
                HOX['trial'] = trial
                if ok_try:
                    node.p_hoack.publish(String(data='ack:%d' % HOX['lock']['ver']))
                    HOX.update(gc=gc_try, o_first=o_try, snap=snap, off_ho=off_ho)
                    far_event('handover_trial_passed_takeover', **{k_: trial[k_] for k_ in ('wall_ms', 'phase', 'first_cmd_step_max', 'budget')})
                    break
                node.p_hoack.publish(String(data='nack:%d:%s' % (HOX['lock']['ver'], str((gc_try.stop or gc_try.abort or {}).get('why'))[:60])))
                far_event('handover_trial_failed_nack', **trial)
        if node.done:
            far_out = dict(result='far_ended_without_handover', why='supervisor_done')
            far_event('supervisor_done_without_handover')
            break
        u_in = node.cmd if (node.t - node.cmd_t <= CMD_TIMEOUT_S and node.cmd_t >= 0) else np.zeros(3)
        ua_in = node.acmd if (node.t - node.acmd_t <= ARM_CMD_TIMEOUT_S and node.acmd_t >= 0) else np.zeros(6)
        pipe_far.append(np.r_[u_in, ua_in])
        ue = pipe_far.pop(0)
        s_far = s_far + DT * ue[3:]
        for i, j in zip(idx, range(6)):
            tgt[i] = float(s_far[j])
        tgt[f1] = XO.FINGER_OPEN
        apply()
        base_cmd(ue[:3], float(q[2]))
        fr = cam.get_current_frame()                                  # 取像與偵測（同 far_capture）
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is not None and rf != last_rf[0] and rf != (0, 0) and col is not None and fr.get('distance_to_image_plane') is not None:
            last_rf[0] = rf
            t_r = fr.get('rendering_time')
            hp = None if t_r is None else hist.get(int(round(float(t_r) / DT)))
            if hp is not None and abs(hp['t_sim'] - float(t_r)) <= 0.25 * DT and float(t_r) - T0 >= t_next_cam - 1e-9:
                dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
                dep = dep[:, :, 0] if dep.ndim == 3 else dep
                t_obs = float(t_r) - T0
                t_next_cam = t_obs + 1.0 / a.cam_hz
                seq += 1
                T_fk = K.fk(hp['q'], W0.OPT)
                T_est_, info = obs.intake(dict(dep=dep, K=Kr, T_cam=T_fk, t_obs=t_obs, q=hp['q'], t_state=t_obs, pose_source='fk_at_render_time', frame_id='map', fk_vs_render_m=float(np.linalg.norm(T_fk[:3, 3] - hp['T_render'][:3, 3]))), node.t)
                T_bh = None if T_est_ is None else np.linalg.inv(T3(hp['q'][:3])) @ np.asarray(T_est_, float)
                np.savez_compressed(os.path.join(fdir, f'f{seq:04d}_depth_m.npz'), dep=dep)
                try:
                    import imageio.v2 as imageio
                    imageio.imwrite(os.path.join(fdir, f'f{seq:04d}_rgb.png'), np.asarray(col)[:, :, :3].astype(np.uint8))
                except Exception as e_:                               # noqa: BLE001
                    print(f'[motm] 彩色影像保存失敗 {e_!r}', flush=True)
                Th_ = truth_handle()
                fmeta.append(dict(seq=seq, stem=f'f{seq:04d}', t_obs=t_obs, t_processed=node.t, cam_T_fk=T_fk.tolist(), q=[float(v) for v in hp['q']], odom_base_at_capture=[float(v) for v in hp['q'][:3]], valid=T_est_ is not None, info=info,
                                  T_base_handle=None if T_bh is None else T_bh.tolist(), eval_only=dict(err_pos_mm=None if T_est_ is None else round(float(np.linalg.norm(np.asarray(T_est_)[:3, 3] - Th_[:3, 3])) * 1e3, 2), range_m=round(float(np.linalg.norm(T_fk[:3, 3] - Th_[:3, 3])), 3))))
                node.p_obs.publish(Float64MultiArray(data=[float(seq), t_obs, 1.0 if T_est_ is not None else 0.0, *[float(v) for v in hp['q'][:3]], *([float(v) for v in T_bh.reshape(-1)] if T_bh is not None else [float('nan')] * 16), float(node.t)]))
        world.step(render=True)
        node.n += 1
        far_rows.append([node.t, *q, *s_far, *ue, cmax, ddeg])
    np.savez_compressed(os.path.join(a.out, 'far_log.npz'), L=np.array(far_rows, float), cols=np.array(['t'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['ux', 'uy', 'uw'] + [f'ua{i}' for i in range(6)] + ['contact_n', 'door_deg']))
    json.dump(clean(dict(K=Kr, cam=CAM, T_mount=T_mount, frames=fmeta)), open(os.path.join(a.out, 'frames_meta.json'), 'w'), ensure_ascii=False, indent=1)
    res['far'] = dict(sim_s=round(node.t, 2), wall_s=round(time.perf_counter() - w_far0, 1), events=far_ev, n_frames=len(fmeta), n_valid=sum(1 for f_ in fmeta if f_['valid']), base_travel_m=round(float(np.abs(np.diff(np.array(far_rows)[:, 1:3], axis=0)).sum()), 3) if len(far_rows) > 1 else 0.0,
                      sources=dict(odom='物理位姿直接當里程（理想）', scan='PhysX 射線，略過機器人自身（忽略手臂自遮擋）', delay='8 步在途管線（軟體模型）', arm_friction='no_passive_arm_friction（自載入起）'))
    if HOX.get('gc') is None:
        if far_out is None and HOX.get('trial') is not None:
            far_out = dict(result='far_ended_handover_rejected', why='trial_failed', trial=HOX['trial'])
        if far_out is None:
            far_out = dict(result='far_ended_without_handover', why='far_limit')
        res.update(result=far_out['result'], far_end=far_out, aborted=far_out if 'abort' in far_out['result'] else None)
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print(f"[motm] 遠距前段結束（未交接）：{far_out}", flush=True)
        return 70 if 'abort' in far_out['result'] else 6
    # ---- 已接手：控制器實例＝試跑通過的那一個（設定點、真管線、上一／前一拍命令、偏置皆為接手當下的快照）
    T_lock, lock, t_far_ho = HOX['T_lock'], HOX['lock'], HOX['t_far_ho']
    node.ca = FCA.ChainAlign(G.D_STEPS, NPH, pre_release=node.cmd.copy())   # 接手前最後一筆放行值（前擁有者的命令在最初 8 步內仍在鏈內）
    chain = dict(stop=None)
    res['handover'] = dict(window_rel=HOX['win'], along_at_takeover=HOX['a_to'], grasp_end_abs=GRASP_END_ABS, t_far=round(t_far_ho, 3), base=q[:3].tolist(), lock={k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in lock.items()}, T_lock=T_lock.tolist(), s=HOX['snap']['s'].tolist(), off_static=HOX['off_ho'].tolist(),
                           u_last=HOX['snap']['u_last'].tolist(), u_prev=HOX['snap']['u_prev'].tolist(), pipe=[u_.tolist() for u_ in HOX['snap']['pipe']], trial=HOX['trial'], eval_only=dict(lock_err_mm=round(float(np.linalg.norm(T_lock[:3, 3] - truth_handle()[:3, 3])) * 1e3, 2)))
    # ---- 把手估計來源＝鎖定任務參考（原始時間戳；不更新）
    frames = []

    def capture(t):
        return None

    def est_fn(t):
        return est_fn_lock(t)
    gate = BL.BindGate(COND, DT)
    t, t_wall0 = 0.0, time.perf_counter()
    capture(0.0)
    gc = HOX['gc']                                                    # 試跑通過並已接手的控制器實例（首拍已算，輸出在 HOX['o_first']）
    fs = FSG.FirstSeg([PLANNER_FS], G.DT)                               # 首段計畫管理（執行器與離線核對共用；提前滾動更新）
    first = dict(ok=False, res=None, sel=None)
    gc.close_gate = lambda cc: fs.gate_ok()                          # 照本排程 READY 時刻算的首段已可用且可行
    ev = []
    rows, ts = [], []
    STATE.update(res=res, rows=rows, ts=ts, ev=ev)                    # 例外時存部分紀錄（名目趟 1 例外未存檔）
    box = dict(o_first=HOX['o_first'], mode='grasp', ex=None, aborted=None, t_bind=None, T_WH_b=None, T_HE=None, phi_b=None, helper=None, pend=None, t_next=None, k=1, decisions=[], reqs=[])

    def log_event(what, **kw):
        ev.append(dict(t=round(t, 3), wall=round(time.perf_counter() - t_wall0, 3), what=what, **kw))
        print(f"[motm] t={t:.2f} {what} {json.dumps(clean(kw), ensure_ascii=False)[:300]}", flush=True)
    phase_i = lambda: PH.index(dict(HOLD='L', INSERT='I', CLOSE='C', CONFIRM='F', READY='R')[gc.phase]) if box['mode'] == 'grasp' else 5   # noqa: E731
    t_end_lim = 1.2 if a.smoke else GRASP_LIMIT_S
    prev_b = qs[:3].copy()
    # ================= 移動抓持 =================
    while t < t_end_lim and box['mode'] == 'grasp':
        q, jp = q9()
        first_it = box.get('o_first') is not None
        tick = first_it or gc.n_step % NPH == 0
        # P2 首段：前導第一拍送出（預測 READY 狀態）；過期後依最新狀態重算（Codex 20261115：首段計畫過期修正）
        def make_req(t_now, target):
            T_plan = gc.handle_ref().copy() if gc.lock is not None else gc.T_est.copy()
            b_pred = q[:3] + np.r_[np.array([-V, 0.0]) * max(target - t_now, 0.0), 0.0]      # 通過線等速（抓持控制器的底盤命令；在途管線同為等速）
            qa_pred = ik(b_pred, T_at(T_plan, 0.0), q[3:])
            phi_now = door_q()
            x_pred = np.r_[b_pred, qa_pred, phi_now]
            T_HE_pred = np.linalg.inv(T_plan) @ T_at(T_plan, 0.0)
            WB.set_model(phi0=phi_now, finger_q=G.FINGER_CLOSED)
            hlp = WB.DoorWBOpt(T_plan, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE_pred, project_start=False)
            v_pred = np.r_[-V, 0.0, 0.0, hlp.closure_velocity(x_pred, np.r_[-V, 0.0, 0.0], 0.0), 0.0]
            return (dict(T_WH0=T_plan, T_HE=T_HE_pred, x0=x_pred, v0=v_pred, H=H_SEG, goal=GOAL, fq=G.FINGER_CLOSED, phi_model=phi_now, try_final=True),
                    dict(T_plan=T_plan, T_HE_pred=T_HE_pred, x_pred=x_pred, phi_req=phi_now, wall_req=time.perf_counter() - t_wall0))

        if tick and gc.t0 is not None and gc.T_est is not None and gc.stop is None:
            sched_t = gc.t0 + gc.sched['hold_s'] + gc.sched['ins_s'] + gc.sched['close_s'] + gc.sched['confirm_s'] - G.DT
            for e_ in fs.step(t, sched_t, make_req, rolling=not a.smoke):
                if e_['what'] == 'request':
                    log_event('first_segment_request', k=e_['rid'], why=e_['why'], t_ready_pred=e_['target'])
                elif e_['what'] == 'result':
                    pl_ = [p_ for p_ in fs.plans if p_['rid'] == e_['rid']]
                    r__ = pl_[0]['res'] if pl_ else None
                    log_event('first_segment_available', k=e_['rid'], t_ready_pred=e_['target'], feasible=e_['ok'], late=e_['late'], plan_wall_s=e_['plan_wall_s'], failed=None if r__ is None else r__[3], final=None if r__ is None else r__[4], info=None if r__ is None else r__[6])
                    if e_['rid'] == 1 and r__ is not None:
                        first.update(ok=e_['ok'], res=r__, t_avail=t, wall_avail=time.perf_counter() - t_wall0, t_req=pl_[0]['t_req'], t_ready_pred=e_['target'], **{k_: pl_[0]['meta'][k_] for k_ in ('wall_req', 'phi_req')})
                else:
                    log_event('first_segment_desync', **{k_: v_ for k_, v_ in e_.items() if k_ != 'what'})
            if fs.desync is not None and gc.stop is None:              # 規劃通訊失同步：拒交接、走抓持控制器既有停止流程
                gc._stop('planner_desync', detail=fs.desync)
        pl_sel = fs.applicable(t) if tick else None
        plan_fit = pl_sel is not None
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None and not a.smoke and not plan_fit:
            log_event('handover_wait_plan', next_window=fs.next_window(t), ready_ver=gc.ready_ver)
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None and not a.smoke and plan_fit:
            w_ho = time.perf_counter()                                   # 交接計算（綁定條件、建執行層、預熱、試跑）的牆鐘：在本體 50 ms 計時之外，另記
            Th = truth_handle()
            Ttcp = K.fk(q, 'link_tcp')
            Tg_des = T_at(Th, 0.0)
            fc = finger_contacts()
            c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), finger_q=[float(jp[f1]), float(jp[f2])], finger_bar_n=[float(fc[0][0]), float(fc[1][0])],
                     door_pre_deg=math.degrees(abs(door_q() - door0)))
            okb, okd = gate.decide(c['pos_m'], c['rot_deg'], c['finger_q'], c['finger_bar_n'], c['door_pre_deg'])
            if not okb:                                                  # 先核實際抓持條件；不過就不做昂貴的預熱與試跑（Codex 20261113_144548 §4）
                log_event('handover_bind_not_ready', bind=okd, bind_values=c, ready_ver=gc.ready_ver, compute_ms=round((time.perf_counter() - w_ho) * 1e3, 1))
            else:
                phi_b = door_q()
                T_WH_b = gc.handle_ref(phi_b)
                T_HE_act = np.linalg.inv(T_WH_b) @ Ttcp
                fq_b = 0.5 * (float(jp[f1]) + float(jp[f2]))
                Xd0 = pl_sel['res'][1]
                first['sel'] = dict(rid=pl_sel['rid'], t_ready_pred=round(pl_sel['target'], 3), t_req=round(pl_sel['t_req'], 3), t_avail=round(pl_sel['t_avail'], 3))
                WB.set_model(phi0=phi_b, finger_q=fq_b)
                hlp2 = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[q, phi_b], GOAL, N=2, dt=0.2, T_HE=T_HE_act, project_start=True)
                Xd0p = np.array([hlp2.project(x, iters=30) for x in Xd0])     # 首段參考以本趟實測 T_HE 重投影手臂（底盤、門角不變）；以試跑重核
                ho = dict(s=gc.s.copy(), pipe=[u_.copy() for u_ in gc.pipe], u_last=gc.u_last.copy(), u_prev=gc.u_prev.copy(), bias=np.asarray(gc.cfg.arm_model.bias, float).copy(), moving=True, off_static=gc.off(q))
                exn = XO.DoorOpenExec(ident, Xd0p, 0.05, T_WH_b, phi_b, T_HE_act, GOAL, ho, fq_b, arm_mode='closure', joint_kp=a.kp,
                                      g2=dict(gravity2=True, gravity3=True, ik_fc=True, pool=POOL, prep_fc=True, deadline_ms=50.0, cstop=CSW, preload_release=bool(a.preload_release)))
                exn.p2_open = True
                res.setdefault('cpu_layout', {}).update(enabled=CPU_LAYOUT, pin=CPU_PIN, at_handover=CPUL.readback(POOL, CSW, (PLANNER, PLANNER_FS)))
                if a.diag:
                    exn.deadline_inject = lambda cc: cc.t >= DIAG_STOP_T - 1e-9   # 取證：0.5 s 後強制本拍晚到 ⇒ 與第四趟相同的閉鏈一致停止流程
                wu = exn.warmup(q, phi_b)                                     # 交接前預熱（複本空跑，本體不動）；試跑前做，避免試跑拍含冷啟成本
                w_dry = time.perf_counter()
                tmp = exn._light_copy()
                tmp.step(0.0, q.copy(), phi_b, float(np.linalg.norm(wrist_F6()[:3])))
                dry_ms = (time.perf_counter() - w_dry) * 1e3
                dry = dict(stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail, fb_ok=bool(tmp._fb_ok),
                           stop_closure_pos_mm=round(tmp.mins['stop_closure_pos'] * 1e3, 3), reproj_arm_max_rad=round(float(np.abs(Xd0p[:, 3:9] - Xd0[:, 3:9]).max()), 5),
                           start_base_vs_meas_mm=round(float(np.linalg.norm(Xd0p[0, :2] - q[:2])) * 1e3, 3))
                valid = tmp.stop is None and tmp.abort is None and tmp._cl_fail is None and tmp.mins['stop_closure_pos'] <= XO.CL_TOL['pos_m'] and tmp._fb_ok
                log_event('handover_attempt', bind_ok=okb, bind=okd, bind_values=c, dry_ok=valid, dry=dry, ready_ver=gc.ready_ver, warmup_ms=round(wu, 1), dry_run_ms=round(dry_ms, 1),
                          handover_compute_ms=round((time.perf_counter() - w_ho) * 1e3, 1), note='交接計算在本體控制拍計時之外；計算期間物理不步進')
                blas_ok = True
                if CPU_LAYOUT:                                           # 預熱／試跑之後、綁定之前：各份已載入 OpenBLAS 設單執行緒並以 getter 讀回；不是全部為 1（或讀不到）⇒ 不交接
                    bl_ = CPUL.blas_runtime(1)
                    blas_ok = CPUL.blas_all_single(bl_)
                    res.setdefault('cpu_layout', {})['blas_before_bind'] = bl_
                    if not blas_ok and gc.stop is None:
                        log_event('blas_threads_not_single', readback=bl_)
                        gc._stop('blas_threads_not_single')              # 不跳出迴圈：之後照抓持控制器既有停止流程走到停穩或逾時中止
                if okb and valid and blas_ok and gc.accept_handover(gc.ready_ver):
                    if fs.pend[0] is not None:                         # A1：在途的首段滾動規劃已無用 ⇒ 終止該程序（之後不再送首段請求）
                        PLANNER_FS.proc.kill()
                        log_event('first_segment_cancelled', rid=fs.pend[0]['rid'], target=round(fs.pend[0]['target'], 3))
                        fs.pend[0] = None
                    Rr0, Tg_b, _ = rel_T()
                    if a.bind_model == 'd6_compliant':
                        j = UsdPhysics.Joint.Define(stage, JOINT)         # 通用六自由度 joint（無限位）＋六軸彈簧阻尼驅動，目標 0＝綁定瞬間相對位姿；不重錨、不加門角驅動
                        j.CreateExcludeFromArticulationAttr().Set(True)
                        bind_rb = {}
                        for ax_, k_, c_ in (('transX', BIND_KT, BIND_CT), ('transY', BIND_KT, BIND_CT), ('transZ', BIND_KT, BIND_CT),
                                            ('rotX', BIND_KR * math.pi / 180.0, BIND_CR * math.pi / 180.0), ('rotY', BIND_KR * math.pi / 180.0, BIND_CR * math.pi / 180.0), ('rotZ', BIND_KR * math.pi / 180.0, BIND_CR * math.pi / 180.0)):
                            dr_ = UsdPhysics.DriveAPI.Apply(j.GetPrim(), ax_)
                            dr_.CreateTypeAttr().Set('force')
                            dr_.CreateStiffnessAttr().Set(float(k_))
                            dr_.CreateDampingAttr().Set(float(c_))
                            dr_.CreateTargetPositionAttr().Set(0.0)
                            dr_.CreateTargetVelocityAttr().Set(0.0)
                            bind_rb[ax_] = dict(stiffness_usd=float(dr_.GetStiffnessAttr().Get()), damping_usd=float(dr_.GetDampingAttr().Get()), type=str(dr_.GetTypeAttr().Get()), max_force=None if not dr_.GetMaxForceAttr() or dr_.GetMaxForceAttr().Get() is None else float(dr_.GetMaxForceAttr().Get()))
                        res['bind_model'] = dict(model='d6_compliant', si=dict(k_trans_N_per_m=BIND_KT, c_trans_Ns_per_m=BIND_CT, k_rot_Nm_per_rad=BIND_KR, c_rot_Nms_per_rad=BIND_CR), usd_readback=bind_rb,
                                                 usd_units='平移：N/m、N·s/m；轉動：N·m/度、N·m·s/度（寫入值＝SI×π/180）', exclude_from_articulation=bool(j.GetExcludeFromArticulationAttr().Get()),
                                                 budget=dict(pos_m=BIND_DEF_POS_M, rot_deg=BIND_DEF_ROT_DEG), note='工程假設，不稱更符合真實夾持；平衡位姿＝綁定瞬間相對位姿')
                    else:
                        j = UsdPhysics.FixedJoint.Define(stage, JOINT)
                        res['bind_model'] = dict(model='fixed')
                    box['Rr0'] = Rr0.copy()
                    j.CreateBody0Rel().SetTargets([Sdf.Path(str(grip[0]))])
                    j.CreateBody1Rel().SetTargets([Sdf.Path(leaf)])
                    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    qr = W0.R_to_wxyz(Rr0[:3, :3])
                    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rr0[:3, 3]]))
                    j.CreateLocalRot1Attr().Set(Gf.Quatf(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])))
                    j.CreateJointEnabledAttr().Set(True)
                    j.CreateBreakForceAttr().Set(float('inf'))
                    j.CreateBreakTorqueAttr().Set(float('inf'))
                    box.update(mode='open', ex=exn, t_bind=t, T_WH_b=T_WH_b, T_HE=T_HE_act, phi_b=phi_b, fq=fq_b, helper=hlp2, t_next=t + P2.T_RE)
                    np.savez(os.path.join(a.out, 'first_segment.npz'), Xd0=Xd0, Xd0p=Xd0p, T_plan=pl_sel['meta']['T_plan'], T_HE_pred=pl_sel['meta']['T_HE_pred'], T_WH_b=T_WH_b, T_HE_act=T_HE_act, phi_b=phi_b)
                    log_event('bind_and_handover', base_speed_cmd=round(float(np.linalg.norm(gc.u_last[:2])), 4), joint_enabled=bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get()), warmup_ms=round(wu, 1))
                    break
        o = box.pop('o_first') if first_it else gc.step(t, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))   # 首拍＝接手試跑的輸出（同一狀態、同一步）
        node.t = float(world.current_time) - T0
        if tick:                                                     # A2(b)：底盤提案走 選擇器→平滑器→Shield→GUARD（GUARD 唯一出口）
            node.ca.propose(node.n, gc.u_last[:3])
            mt_ = Twist(); mt_.linear.x, mt_.linear.y, mt_.angular.z = (float(x_) for x_ in gc.u_last[:3]); node.p_grasp.publish(mt_)
            okw_, _ = wait_forwarded(gc.u_last[:3])
            if not okw_:
                log_event('grasp_proposal_not_forwarded_within_200ms', n=node.n)
        ros_step(q, jp, gc.s, gc.pipe, 'grasp')
        if node.abort is not None:                                   # 全域中止貫穿交接後：收完回呼、下一次物理步之前檢查；觸發即保存資料、不再步進
            box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（監督節點 /far/abort；不再步進，不是受控停止）', detail=node.abort[:200])
            log_event('supervisor_abort_engineering_abort', detail=node.abort[:200])
            break
        ckc = node.ca.check(node.n)                                  # 本步有提案生效：放行須與控制器核過的命令一致；缺失／不一致 ⇒ 在它生效前拒絕＝模擬工程中止（控制器的停止證書以它的管線為前提，管線已不成立）
        if not ckc['ok']:
            chain['stop'] = dict(t=round(t, 3), phase='grasp', **ckc)
            box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（安全鏈放行與已核命令不一致；不再步進，不是受控停止）', chain=ckc)
            log_event('safety_chain_mismatch_engineering_abort', **ckc)
            node.p_state.publish(String(data='latched:safety_chain_' + ckc['why']))
            break
        if o.get('abort'):
            box['aborted'] = dict(t=round(t, 3), controller=gc.abort)
            log_event('grasp_abort', abort=gc.abort)
            break
        for i, j_ in zip(idx, range(6)):
            tgt[i] = float(o['arm_target'][j_])
        tgt[f1] = float(o['finger'])
        diag_A()
        apply()
        base_cmd(o['base_twist'], float(q[2]))
        diag_B()
        world.step(render=True)
        node.n += 1
        t += DT
        diag_C(t, 0, o['base_twist'], gc.pipe, gc.u_last)
        capture(t)
        q2, jp2 = q9()
        Rr, _, _ = rel_T()
        fc = finger_contacts()
        oc, ow = other_contacts()
        gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
        F6 = wrist_F6()
        if a.preload_release and gc.phase == 'HOLD' and 0.1 <= t <= 0.4 and gc.stop is None:   # 自由狀態基線（未接觸、前導等速）：量測轉世界＋下游重力的殘差
            rF_, rM_, T6_ = wrench_parts(q2, F6)
            wb['F'].append(rF_)
            wb['M'].append(T6_[:3, :3].T @ rM_)
        elif a.preload_release and wb['bF'] is None and gc.phase != 'HOLD' and len(wb['F']) >= 20 and np.isfinite(wb['F']).all() and np.isfinite(wb['M']).all():
            wb['bF'], wb['bM'] = np.mean(wb['F'], axis=0), np.mean(wb['M'], axis=0)
            wb['tried'] = True
            log_event('wrench_baseline', n=len(wb['F']), bias_F_world=[round(float(x_), 3) for x_ in wb['bF']], bias_M_link6=[round(float(x_), 4) for x_ in wb['bM']], F_std=[round(float(x_), 3) for x_ in np.std(wb['F'], axis=0)])
        if a.preload_release and wb['bF'] is None and gc.phase != 'HOLD' and gc.stop is None:   # 指定 B2 開而基線無效（樣本不足或非有限）⇒ 不默默關掉 B2：走抓持控制器既有停止流程、本趟不進任務
            log_event('wrench_baseline_invalid', n=len(wb['F']))
            gc._stop('wrench_baseline_invalid', n=len(wb['F']))
            res['refused'] = 'wrench_baseline_invalid'
        if not np.isfinite(F6).all() and gc.stop is None:              # 力訊號失效：40 N 監看無法成立 ⇒ 停止
            log_event('force_signal_invalid', phase='grasp')
            gc._stop('force_signal_invalid')
        rows.append([t, phase_i() if gc.stop is None else 6, door_q(), float(jp2[f1]), float(jp2[f2]), *q2, *F6, time.perf_counter() - t_wall0])
        if gc.n_step % NPH == 0:
            bsp = float(np.linalg.norm(q2[:2] - prev_b[:2])) / G.DT
            prev_b = q2[:3].copy()
            Tt = T_at(truth_handle(), 0.0)
            ts.append([t, time.perf_counter() - t_wall0, 0, phase_i(), bsp, door_q(), float(np.linalg.norm(K.fk(q2, 'link_tcp')[:3, 3] - Tt[:3, 3])), np.nan, np.nan])
        if gate.latched is not None and gc.stop is None:
            log_event('forbidden_contact_latched', latch=gate.latched, who=ow)
            gc._stop('exec_forbidden_contact', contact_info=dict(gate.latched, who=ow))
        if gc.stop is not None and gc.stop_ok is not None:
            log_event('grasp_stopped', stop=gc.stop, stop_ok=gc.stop_ok)
            break
        if a.smoke and t >= 1.2 - 1e-9 and gc.stop is None:              # 冒煙：INSERT 中途（未接觸、未閉爪）即停
            gc._stop('smoke_end')
            log_event('smoke_stop_requested')
            t_end_lim = t + 2.0
    rep_g = gc.report()
    res['grasp'] = {k_: rep_g.get(k_) for k_ in ('phases', 'stop', 'abort', 'stop_effect', 'ready_t_rel', 'accepted', 'accept_t_rel', 'n_ready_revoked', 'min_clear', 'limits', 'contact_lock', 'lock_inconsistency_max', 'contact_stop', 'fast_slip')}
    res['grasp']['n_deadline_miss'] = gc.n_deadline_miss
    res['grasp']['tick_log'] = gc.tick_log
    res['first_segment'] = {k_: first.get(k_) for k_ in ('t_req', 'wall_req', 't_avail', 'wall_avail', 'ok', 't_ready_pred', 'phi_req', 'sel')}
    res['first_segment'].update(n_req=fs.n_req, n_dropped=fs.n_stale_drop, log=fs.log, window=dict(lead_s=fs.lead, tol_s=fs.tol, ahead_s=fs.ahead, gap_s=fs.gap))
    if first['res'] is not None:
        res['first_segment'].update(feasible=first['res'][2], failed=first['res'][3], final=first['res'][4], plan_wall_s=round(first['res'][5], 3), info=first['res'][6])
    # ================= 開門（P2 續接）=================
    if box['mode'] == 'open':
        ex = box['ex']
        olog, deflog, wlog = [], [], []
        while t - box['t_bind'] < OPEN_LIMIT_S:
            q, jp = q9()
            tr = t - box['t_bind']
            phm = door_q()
            tick = ex.n_step % NPH == 0
            if tick and ex.staged is not None and ex.stop is None:
                dec = ex.staged_step(tr, q.copy(), phm)
                if isinstance(dec, dict):
                    box['decisions'].append(dec)
                    log_event('p2_decision', ok=dec.get('ok'), ver=dec.get('ver'), why=dec.get('why'), dry=dec.get('dry'))
            r_ = PLANNER.poll() if box['pend'] is not None else None
            if r_ is not None:
                pend = box['pend']; box['pend'] = None
                rec = pend['rec']
                rec.update(t_avail=round(t, 3), wall_avail=round(time.perf_counter() - t_wall0, 3), plan_wall_s=round(r_[5], 3), feasible=r_[2], final=r_[4], failed=r_[3], warm_used=(r_[6] or {}).get('warm_used'), why=(r_[6] or {}).get('why'), iters=(r_[6] or {}).get('iters'))
                xr = ex.ref(ex.sigma)                                  # σ 已以管線延遲起算 ⇒ 與實測同一時刻
                ey, ed = abs(math.degrees(q[2] - xr[2])), abs(math.degrees(phm - xr[9]))
                rec['track_err'] = dict(base_mm=round(float(np.linalg.norm(q[:2] - xr[:2])) * 1e3, 3), yaw_deg=round(ey, 3), door_deg=round(ed, 3))
                if not r_[2] or r_[1] is None:
                    rec['result'] = 'infeasible'
                elif ex.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
                    rec['result'] = 'missed_deadline'
                # 偏航／門角追蹤誤差只記錄（rec['track_err']）；接管由接縫核對＋真狀態試跑決定（Codex 20261115_004634 §3）
                elif ex.staged is not None or ex.stop is not None:
                    rec['result'] = 'staged_busy_or_stopped'
                else:
                    okS, info = ex.stage_reference(r_[1], pend['sigma_s'], final=r_[4], ver=rec['k'])
                    rec['result'] = 'staged' if okS else 'stage_rejected'
                    if not okS:
                        rec['why'] = info.get('why')
                log_event('p2_plan_result', k=rec['k'], result=rec['result'], plan_wall_s=rec['plan_wall_s'], final=rec['final'])
            if box['pend'] is None and ex.stop is None and ex.p2_open and not a.diag and t >= box['t_next'] - 1e-9:
                sig_s = math.ceil((ex.sigma + L_LAT) / ex.h - 1e-9) * ex.h
                rest = sig_s >= ex.sig_end - 1e-9
                still_ok = ex.rate['arm'] <= XO.END['arm_rps'] and ex.rate['base'] <= XO.END['base_mps'] and ex.rate['door'] <= XO.END['door_rps']
                if rest and not still_ok:
                    box['t_next'] = t + 0.05
                else:
                    if rest:
                        sig_s = ex.sig_end
                    i_ = min(int(round(sig_s / ex.h)), len(ex.Xd) - 1)
                    xs = ex.Xd[i_].copy()
                    if rest or i_ >= len(ex.Xd) - 1:
                        vs = np.zeros(10)
                    else:
                        vv = (ex.Xd[min(i_ + 1, len(ex.Xd) - 1)] - ex.Xd[max(i_ - 1, 0)]) / (2 * ex.h)
                        WB.set_model(phi0=box['phi_b'], finger_q=box['fq'])
                        vs = np.r_[vv[:3], box['helper'].closure_velocity(xs, vv[:3], vv[9]), vv[9]]
                    box['k'] += 1
                    rec = dict(k=box['k'], t_req=round(t, 3), wall_req=round(time.perf_counter() - t_wall0, 3), sigma=round(ex.sigma, 3), sigma_s=round(sig_s, 3), rest=rest)
                    box['reqs'].append(rec)
                    PLANNER.submit(box['k'], dict(T_WH0=box['T_WH_b'], T_HE=box['T_HE'], x0=xs, v0=vs, H=H_SEG, goal=GOAL, fq=box['fq'], phi_model=box['phi_b'], try_final=True, warm_ref=None if rest else (ex.Xd.copy(), ex.h, sig_s)))
                    box['pend'] = dict(sigma_s=sig_s, rest=rest, rec=rec)
                    box['t_next'] = t + P2.T_RE
            F6 = wrist_F6()
            fn_ = float(np.linalg.norm(F6[:3]))
            w_wall0 = time.perf_counter()
            w_ext_ = ext_wrench(q, F6) if a.preload_release else None
            wlog.append(np.r_[tr, F6, np.full(6, np.nan) if w_ext_ is None else np.r_[w_ext_[0], w_ext_[1]], (time.perf_counter() - w_wall0) * 1e3])
            if ex.abort is None and (not np.isfinite(F6).all() or (a.preload_release and w_ext_ is None)):   # 力訊號或外力旋量失效 ⇒ 中止（不得讓 40 N 監看失效後繼續）
                why_ = 'force_signal_invalid' if not np.isfinite(F6).all() else 'wrench_invalid'
                log_event(why_, phase='open')
                ex._stop(why_)
            out = ex.step(tr, q, phm, fn_, w_ext=w_ext_)
            node.t = float(world.current_time) - T0
            if tick:
                node.ca.propose(node.n, ex.u_last[:3])
                mt_ = Twist(); mt_.linear.x, mt_.linear.y, mt_.angular.z = (float(x_) for x_ in ex.u_last[:3]); node.p_grasp.publish(mt_)
                okw_, _ = wait_forwarded(ex.u_last[:3])
                if not okw_:
                    log_event('open_proposal_not_forwarded_within_200ms', n=node.n)
            ros_step(q, jp, ex.s, ex.pipe, 'open')
            if node.abort is not None:                                   # 全域中止貫穿交接後：收完回呼、下一次物理步之前檢查；觸發即保存資料、不再步進
                box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（監督節點 /far/abort；不再步進，不是受控停止）', detail=node.abort[:200])
                log_event('supervisor_abort_engineering_abort', detail=node.abort[:200])
                break
            ckc = node.ca.check(node.n)
            if not ckc['ok']:
                chain['stop'] = dict(t=round(t, 3), phase='open', **ckc)
                box['aborted'] = dict(t=round(t, 3), controller=None, kind='模擬工程中止（安全鏈放行與已核命令不一致；不再步進，不是受控停止）', chain=ckc)
                log_event('safety_chain_mismatch_engineering_abort', **ckc)
                node.p_state.publish(String(data='latched:safety_chain_' + ckc['why']))
                break
            if out['abort']:
                box['aborted'] = dict(t=round(t, 3), controller=ex.abort)
                log_event('open_abort', abort=ex.abort)
                break
            for i, j_ in zip(idx, range(6)):
                tgt[i] = float(out['arm_target'][j_])
            diag_A()
            apply()
            base_cmd(out['base_twist'], float(q[2]))
            diag_B()
            world.step(render=True)
            node.n += 1
            t += DT
            diag_C(t, 1 if ex.stop is None else 2, out['base_twist'], ex.pipe, ex.u_last)
            q2, jp2 = q9()
            Rr, _, _ = rel_T()
            fc = finger_contacts()
            oc, ow = other_contacts()
            gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
            F6 = wrist_F6()
            rows.append([t, 5 if ex.stop is None else 6, door_q(), float(jp2[f1]), float(jp2[f2]), *q2, *F6, time.perf_counter() - t_wall0])
            olog.append(np.r_[tr, q2, door_q(), fn_, ex.sigma, ex.ref(ex.sigma)[9]])
            if box.get('Rr0') is not None:                              # 連接變形：綁定期間每物理步都查（含停止中）；超量、非有限或位姿讀取失敗 ⇒ 下一物理步前中止
                bad_def = None
                if Rr is None or not np.isfinite(Rr).all():
                    bad_def = dict(why='bind_pose_read_failed')
                else:
                    dT_ = np.linalg.inv(box['Rr0']) @ Rr
                    dpos_, drot_ = float(np.linalg.norm(dT_[:3, 3])), math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(dT_[:3, :3]) - 1.0)))))
                    pv_ = deflog[-1] if deflog else None
                    vpos_ = 0.0 if pv_ is None else float(np.linalg.norm(dT_[:3, 3] - np.array(pv_[3:6]))) / DT
                    vrot_ = 0.0 if pv_ is None else math.radians(abs(drot_ - pv_[2])) / DT
                    # 估算（非量測）：彈簧分量 k×變形、阻尼分量 c×相對速度（差分）；D6 驅動同時含兩項
                    deflog.append([tr, dpos_, drot_, *[float(x_) for x_ in dT_[:3, 3]], vpos_, vrot_, BIND_KT * dpos_, BIND_CT * vpos_, BIND_KR * math.radians(drot_), BIND_CR * vrot_])
                    if not (math.isfinite(dpos_) and math.isfinite(drot_)):
                        bad_def = dict(why='bind_deformation_nonfinite')
                    elif a.bind_model == 'd6_compliant' and (dpos_ > BIND_DEF_POS_M or drot_ > BIND_DEF_ROT_DEG):
                        bad_def = dict(why='bind_deformation_over_budget', pos_mm=round(dpos_ * 1e3, 3), rot_deg=round(drot_, 3))
                if bad_def is not None and ex.abort is None and (a.bind_model == 'd6_compliant' or bad_def['why'] != 'bind_deformation_over_budget'):
                    log_event(bad_def['why'], **{k_: v_ for k_, v_ in bad_def.items() if k_ != 'why'}, during_stop=ex.stop is not None)
                    ex._stop(bad_def['why'], **{k_: v_ for k_, v_ in bad_def.items() if k_ != 'why'})   # 不在 PATH_OK ⇒ 中止（停止中則升級為中止）
            if ex.n_step % NPH == 0:
                bsp = float(np.linalg.norm(q2[:2] - prev_b[:2])) / G.DT
                prev_b = q2[:3].copy()
                xr = ex.ref(ex.sigma)
                ts.append([t, time.perf_counter() - t_wall0, 1, 5 if ex.stop is None else 6, bsp, door_q(), np.nan, ex.sigma, float(xr[9])])
            if gate.latched is not None and ex.stop is None:
                log_event('forbidden_contact_latched', latch=gate.latched, who=ow)
                BL.forward_forbidden(gate.latched, ow, ex=ex)
            if ex.stop is not None and ex.stop_ok is not None and t - box['t_bind'] - ex.stop_ok['t'] >= 0.3:
                log_event('open_end', stop=ex.stop, done=ex.done, stop_ok=ex.stop_ok)
                break
        if box['aborted'] is None and ex.stop_ok is None:
            ex._stop('open_time_limit')
        rep_o = ex.report()
        res['open'] = {k_: rep_o.get(k_) for k_ in ('arm_mode', 'closure_anchor', 'closure_stats', 'stop', 'done', 'stop_effect', 'phi_end_deg', 't_end', 'closure_measured', 'min_clear_actual', 'force_max_n', 'timing')}
        res['open']['bind_deformation'] = None if not deflog else dict(pos_mm_max=round(max(x_[1] for x_ in deflog) * 1e3, 3), rot_deg_max=round(max(x_[2] for x_ in deflog), 3), pos_mm_end=round(deflog[-1][1] * 1e3, 3), rot_deg_end=round(deflog[-1][2], 3),
                                                                     est_spring_force_max_n=round(max(x_[8] for x_ in deflog), 2), est_damper_force_max_n=round(max(x_[9] for x_ in deflog), 2), est_spring_torque_max_nm=round(max(x_[10] for x_ in deflog), 3), est_damper_torque_max_nm=round(max(x_[11] for x_ in deflog), 3),
                                                                     note='力／力矩為估算（k×變形、c×差分相對速度；fixed 模型下僅供對照，非實際連接力）', cols=['t', 'pos_m', 'rot_deg', 'dx', 'dy', 'dz', 'vpos', 'vrot', 'F_spring', 'F_damper', 'T_spring', 'T_damper'])
        QL_ = np.array(ex.qp_log, float).reshape(-1, 8)
        def qsum(m_):
            S_ = QL_[m_]
            if not len(S_):
                return None
            med = lambda c_: (None if not np.isfinite(S_[:, c_]).any() else round(float(np.nanmedian(S_[:, c_])), 2))   # noqa: E731
            return dict(n=int(len(S_)), pre_ms=med(2), mpc_solve_ms=med(3), mpc_solve_ms_max=round(float(S_[:, 3].max()), 2), asm_ms=med(4), gi_ms=med(5), osqp_ms=med(6), n_osqp_fallback=int((S_[:, 7] > 0.5).sum()))
        res['open']['qp_timing'] = dict(normal_ticks=qsum(QL_[:, 1] < 0.5), path_stop_ticks=qsum(QL_[:, 1] > 0.5), note='正常拍與停止中求解分列；cstop 停止拍不求解')
        res['cpu_layout_end'] = CPUL.readback(POOL, CSW, (PLANNER,)) if CPU_LAYOUT else None
        res['open'].update(n_deadline_miss=ex.n_deadline_miss, p2_log=ex.p2_log, reqs=box['reqs'], fb_log=ex.fb_log, preload_release=dict(enabled=bool(ex.g2['preload_release']), n_committed=int(sum(1 for x_ in ex.pr_ts if x_[1] == 1)), n_rejected=int(sum(1 for x_ in ex.pr_ts if x_[1] == -1)), wrench_compute_ms_max=(None if not wlog else round(float(max(x_[-1] for x_ in wlog)), 3)), wrench_compute_note='外力旋量換算在控制器 50 ms 計時之外，另列', baseline=None if wb['bF'] is None else dict(F_world=[float(x_) for x_ in wb['bF']], M_link6=[float(x_) for x_ in wb['bM']], n=len(wb['F'])), n=len(ex.pr_log), delta_abs_max=max((x_[1] for x_ in ex.pr_log), default=None), delta_end=[round(float(x_), 5) for x_ in ex._pr_delta], Fc_first=(ex.pr_log[0][2] if ex.pr_log else None), Fc_max=max((x_[2] for x_ in ex.pr_log), default=None), Fc_last=(ex.pr_log[-1][2] if ex.pr_log else None), Mc_last=(ex.pr_log[-1][3] if ex.pr_log else None), lam_last=(ex.pr_log[-1][4] if ex.pr_log else None), sat_any=any(x_[5] for x_ in ex.pr_log), params=dict(KI=XO.PR.KI, RATE_MAX=XO.PR.RATE_MAX, DELTA_MAX=XO.PR.DELTA_MAX), log_cols=['t', 'delta_abs_max', 'Fc_norm', 'Mc_norm', 'lambda', 'sat', 'updated']), base_hold_log_tail=ex.hold_log[-5:], base_hold_n=len(ex.hold_log), base_hold_cmd_max=[max((x[1] for x in ex.hold_log), default=None), max((x[2] for x in ex.hold_log), default=None)], cstop_ticks=len(ex.cs_log), stop_mode=ex.stop_mode, tick_ms_normal=[round(float(m), 1) for i, m in enumerate(ex.tick_ms) if ex.stop is None or i * 0.05 < float(ex.stop['t']) - 1e-9])
        np.savez_compressed(os.path.join(a.out, 'open_steps.npz'), qp_log=np.array(ex.qp_log, float), qp_log_cols=np.array(['t', 'stopped', 'pre_ms', 'mpc_solve_ms', 'asm_ms', 'gi_ms', 'osqp_ms', 'solver(0gi,1osqp)']), tick_ms=np.array(ex.tick_ms, float), preload_ts=np.array(ex.pr_ts, float), preload_ts_cols=np.array(['t', 'event(1提交,-1拒絕或晚到,0無候選)'] + [f'cand_{k_}' for k_ in range(6)] + [f'committed_{k_}' for k_ in range(6)] + [f'in_setpoint_{k_}' for k_ in range(6)] + ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']), wrench=np.array(wlog, float), wrench_cols=np.array(['t'] + [f'raw_j6_{k_}' for k_ in range(6)] + ['Fext_x', 'Fext_y', 'Fext_z', 'Mtcp_x', 'Mtcp_y', 'Mtcp_z', 'compute_ms']), preload=np.array(ex.pr_log, float), bind_deform=np.array(deflog), meas=np.array(olog), Xd_final=ex.Xd, h=ex.h, ctrl=np.array(ex.step_log))
    res['events'] = ev
    res['forbidden_contact'] = dict(latched=gate.latched)
    res['aborted'] = box['aborted']
    res['wall_total_s'] = round(time.perf_counter() - t_wall0, 1)
    res['sim_total_s'] = round(t, 2)
    res['rtf'] = round(t / max(res['wall_total_s'], 1e-9), 3)
    o_ = res.get('open') or {}
    if a.smoke:
        sw = ((res['grasp'] or {}).get('stop') or {}).get('why')
        res['result'] = 'smoke_done' if (res['first_segment'].get('t_avail') is not None and box['aborted'] is None and sw in (None, 'smoke_end')) else 'smoke_blocked：' + str(sw)
    elif box['aborted'] is not None:
        res['result'] = 'aborted：模擬中止（停止有效性未驗證；綁定中則未解綁）'
    elif o_.get('done') and (o_.get('stop_effect') or {}).get('ok'):
        res['result'] = 'open_done_85'
    elif box['mode'] == 'grasp':
        res['result'] = 'grasp_not_handed_over：' + str((res['grasp'] or {}).get('stop'))
    else:
        res['result'] = 'open_not_done：' + str(o_.get('stop'))
    if a.diag:
        try:
            nd = len(robot.dof_names)
            lay = [('t', 1), ('mode(0抓持,1開門,2停止)', 1), ('rootA_pos3_quat4_lin3_ang3(覆寫前)', 13), ('rootB(覆寫後)', 13), ('rootC(物理步後)', 13), ('base_twist_cmd(本體座標)', 3), ('joint_pos', nd), ('joint_vel', nd),
                   ('measured_joint_efforts', nd), ('applied_joint_efforts', nd), ('position_targets(tgt)', nd), ('measured_joint_forces[j6]', 6), ('measured_joint_forces[j6-1]', 6),
                   ('link6_pos3_quat4', 7), ('gripper_pos3_quat4', 7), ('leaf_pos3_quat4', 7), ('pipe(8×9)', 72), ('u_last', 9), ('door_q', 1)]
            jinfo = {}
            for pr in stage.Traverse():
                if pr.GetName() in ('joint5', 'joint6') or str(pr.GetPath()) == JOINT:
                    jj = UsdPhysics.Joint(pr)
                    g_ = lambda at: (None if not at or at.Get() is None else [float(x) for x in (list(at.Get()) if not hasattr(at.Get(), 'GetReal') else [at.Get().GetReal(), *at.Get().GetImaginary()])])   # noqa: E731
                    jinfo[str(pr.GetPath())] = dict(type=str(pr.GetTypeName()), body0=[str(x) for x in jj.GetBody0Rel().GetTargets()], body1=[str(x) for x in jj.GetBody1Rel().GetTargets()],
                                                    localPos0=g_(jj.GetLocalPos0Attr()), localRot0_wxyz=g_(jj.GetLocalRot0Attr()), localPos1=g_(jj.GetLocalPos1Attr()), localRot1_wxyz=g_(jj.GetLocalRot1Attr()))
            dmeta = dict(layout=lay, dof_names=list(robot.dof_names), joint_indices={str(k_): int(v_) for k_, v_ in dict(meta.joint_indices).items()}, j6_row=j6,
                         link_names=[str(x) for x in getattr(meta, 'link_names', [])], joint_names=[str(x) for x in getattr(meta, 'joint_names', [])],
                         api_doc=dict(get_measured_joint_forces=str(getattr(robot.get_measured_joint_forces, '__doc__', ''))[:1500], get_measured_joint_efforts=str(getattr(robot.get_measured_joint_efforts, '__doc__', ''))[:1500],
                                      get_applied_joint_efforts=str(getattr(robot.get_applied_joint_efforts, '__doc__', ''))[:1500]),
                         usd_joints=jinfo, errors=derr[:20], n_err=len(derr), diag_stop_t=DIAG_STOP_T, phys_dt=DT)
            json.dump(dmeta, open(os.path.join(a.out, 'diag_meta.json'), 'w'), ensure_ascii=False, indent=1)
            if dlog:
                nmax = max(len(x_) for x_ in dlog)
                np.savez_compressed(os.path.join(a.out, 'diag_steps.npz'), D=np.array([np.r_[x_, np.full(nmax - len(x_), np.nan)] for x_ in dlog]))
        except Exception as e:                                         # noqa: BLE001
            print('[motm] diag 存檔失敗', repr(e)[:300], flush=True)
        res['diag'] = dict(note='取證短診斷：不送 P2；開門執行層 0.5 s 後強制本拍晚到，走閉鏈一致停止流程（與第四趟相同）；不是開門成果', forced_stop_t=DIAG_STOP_T, n_rows=len(dlog), n_err=len(derr))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=np.array(rows), cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'wall']),
                        phases=np.array(PH), ts=np.array(ts), ts_cols=np.array(['t', 'wall', 'ctrl', 'phase', 'base_speed', 'door_q', 'tcp_err_grasp', 'sigma', 'phi_ref']))
    ew_ = np.array(node.echo_wait, float)
    res['safety_chain'] = dict(node.ca.report(), forward_wait_ms=None if not len(ew_) else dict(n=int(len(ew_)), n_timeout=int(np.isnan(ew_).sum()), p50=float(np.nanmedian(ew_)) if np.isfinite(ew_).any() else None, max=float(np.nanmax(ew_)) if np.isfinite(ew_).any() else None), stop=chain['stop'], note='抓持／開門段底盤提案經 選擇器→平滑器→Shield→GUARD；每筆提案在生效步核對放行與提案一致才執行；缺失／不一致 ⇒ 生效前模擬工程中止')
    res['lock_budget_max'] = gc.lock_budget_max
    try:
        np.savez_compressed(os.path.join(a.out, 'chain_stream.npz'), prop_n=np.array([p_['n'] for p_ in node.ca.props]), prop_u=np.array([p_['u'] for p_ in node.ca.props]), prop_conf=np.array([-1 if p_['n_conf'] is None else p_['n_conf'] for p_ in node.ca.props]), rel_n=np.array([r_[0] for r_ in node.ca.releases]), rel_v=np.array([r_[1] for r_ in node.ca.releases]))
    except Exception as e_:                                        # noqa: BLE001
        print('[motm] chain_stream 存檔失敗', repr(e_), flush=True)
    try:
        node.p_state.publish(String(data='done')); rclpy.spin_once(node, timeout_sec=0.0)
    except Exception:                                              # noqa: BLE001
        pass
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[motm] 完成 result={res['result']} sim={res['sim_total_s']} s wall={res['wall_total_s']} s RTF={res['rtf']}", flush=True)
    return 0 if res['result'] in ('open_done_85', 'smoke_done') else 5


STATE = {}


def save_partial(tb):
    """例外中斷時存部分紀錄：log_partial.npz（逐步列、時間序列）＋result_partial.json（事件、例外）。"""
    try:
        if STATE.get('rows'):
            np.savez_compressed(os.path.join(a.out, 'log_partial.npz'), rows=np.array(STATE['rows']), ts=np.array(STATE['ts']),
                                cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'wall']), phases=np.array(PH))
        r = dict(STATE.get('res') or {}, events=STATE.get('ev'), exception=tb, result='exception：程式例外中斷（非受控停止；停止有效性未驗證）')
        json.dump(clean(r), open(os.path.join(a.out, 'result_partial.json'), 'w'), ensure_ascii=False, indent=1)
    except Exception as e_:                                          # noqa: BLE001
        print('[motm] 部分紀錄存檔失敗', repr(e_), flush=True)


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print('[motm] 例外', tb, flush=True)
        save_partial(tb)
    try:
        POOL.close()
        PLANNER.close()
        PLANNER_FS.close()
        CSW.close()
    except Exception:                                              # noqa: BLE001
        pass
    try:
        with open(os.path.join(a.out, 'exit_code.txt'), 'w') as fh:
            fh.write(str(int(rc)))
            fh.flush(); os.fsync(fh.fileno())
    except Exception:                                              # noqa: BLE001
        pass
    print(f'[motm] exit_code {rc}', flush=True)
    sim_app.close()
    sys.exit(rc)
