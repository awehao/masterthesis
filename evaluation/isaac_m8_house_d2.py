#!/usr/bin/env python3
"""主線 B 無接觸取像執行器（Codex 20261117_034316：一條候選、一趟無接觸取像；不閉爪、不綁定、不開門）＝isaac_m8_house_far_capture.py 的複本，改兩處：
  (1) **物理依牆鐘固定 RTF 推進，不等任何計算**：每物理步目標牆鐘＝起點＋模擬時間/RTF，早到就睡、晚到就記落後（迴圈本身算圖／射線比目標慢時照實記錄實得 RTF）；
      監督節點與導航鏈在 ROS 模擬時鐘上跑，它們的計算期間物理照常前進。
  (2) **影像偵測＋估計在獨立程序**（啟動模擬器前 fork）：主迴圈只送影格、不等結果；結果回來時以「當下模擬時間」為處理完成時刻發布（年齡＝實際延遲）。
      偵測程序忙時新影格不送（記 detector_busy），不排隊。原始深度與彩色由偵測程序保存。
以下為原檔說明（遠距實際取像執行器，Codex 20261116）：

本程式只做「模擬器＋感測＋受控對象介面」，不含任何導航或任務決策：
  場景＝住宅 m8_house_r1（同 isaac_m8_house_door_approach.py 的建法：牆、家具、門、未知靜物、機器人、腕部相機）。
  ROS 介面（與乾跑世界節點 far_dry/world_node.py 同一組話題，導航鏈與監督節點不用改）：
    出：/clock、/odom_raw（物理位姿，frame odom；＝理想里程，照實標示）、/joint_states、/scan_raw（PhysX 射線 360 道 10 Hz，略過機器人自身形狀）、
        /far/truth（t、q9、設定點 6、在途末端設定點 6、8 步在途命令 8×9）、/far/handle_obs（每格處理過的影像的估計結果）、/far/exec_state
    入：/cmd_vel（GUARD 放行的底盤命令）、/far/arm_cmd（六關節速率命令）、/far/done（監督節點結束）、/far/abort（監督節點要求模擬工程中止）
  命令到受控對象經 8 步在途管線（同門任務控制器的延遲模型）；手臂＝速率命令積分成設定點、位置驅動追設定點。
取像：腕部相機每個新渲染影格都核對「渲染時刻的狀態紀錄」；依 --cam-hz 取用，逐格保存原始深度、彩色、內參、
  相機外參（渲染時刻關節／底盤讀值的 FK）、擷取同刻的里程位姿／關節／最新 map←odom（訂閱 /tf）、偵測與估計的成功或失敗原因。
  估計以「相對擷取同刻底盤」的位姿發布（相機＋FK 給得出的量，不含世界定位）。
保護：非預期接觸（>0.05 N）、門角變化（>0.05°）、手指張開量偏離模型 ⇒ 鎖存：底盤命令歸零、手臂設定點凍結，並在 /far/exec_state 發 latched（監督節點據此全域停止）。
真值（把手、門角）只寫進紀錄供事後評估，不進任何發布的量。"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json')))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--start', default='1.5,-2.8,1.5708')
ap.add_argument('--arm', default='-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018', help='起始手臂構型（預設＝online_1 far_live 影格的構型）')
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--sim-limit-s', type=float, default=150.0)
ap.add_argument('--cam-hz', type=float, default=5.0, help='取用影格的頻率（偵測＋保存）；相機本身 30 Hz')
ap.add_argument('--scan-hz', type=float, default=10.0)
ap.add_argument('--static-s', type=float, default=0.0, help='>0：靜態取像冒煙——不接受任何運動命令，取像 N 秒後結束')
ap.add_argument('--rtf', type=float, default=0.4, help='物理推進的目標即時比（牆鐘配速；不等計算）')
a = ap.parse_args()
import multiprocessing as _mp                                    # noqa: E402


def _det_worker(conn, fdir):
    """偵測＋估計程序（模擬器啟動前 fork）：收影格 → FL.HandleObs.intake → 回結果；並保存原始深度與彩色。"""
    try:                                                              # 父行程結束即跟著結束（unified_1–_4：本程序繼承 Isaac 互斥鎖描述子，父行程退出後仍等管道而握鎖）
        import ctypes as _ct, signal as _sg
        _ct.CDLL('libc.so.6', use_errno=True).prctl(1, _sg.SIGTERM)
    except Exception:                                                 # noqa: BLE001
        pass
    sys.path.insert(0, HERE)
    import numpy as np_
    import door_far_link as FL_
    obs_ = FL_.HandleObs()
    conn.send(('ready', None))
    while True:
        m_ = conn.recv()
        if m_ is None:
            break
        w0_ = time.perf_counter()
        try:
            T_est_, info_ = obs_.intake(dict(dep=m_['dep'], K=m_['K'], T_cam=m_['T_cam'], t_obs=m_['t_obs'], q=m_['q'], t_state=m_['t_obs'], pose_source='fk_at_render_time', frame_id='map', fk_vs_render_m=m_['fk_vs_render_m']), m_['t_obs'])
            err_ = None
        except Exception as e_:                                  # noqa: BLE001
            T_est_, info_, err_ = None, dict(cls='exception', why='exception'), repr(e_)[:200]
        cyl_ = None
        if m_.get('dbox') is not None:                                 # 抽屜內筆深度偵測（幾何法；範圍＝已知家具＋控制器開度估計）
            try:
                import marker_detect as MD_
                p_c, y_c, i_c = MD_.detect(m_['dep'], m_['K'], m_['T_cam'], m_['dbox'])
                cyl_ = dict(p=None if p_c is None else np_.asarray(p_c, float), yaw=y_c, info=i_c)
            except Exception as e_:                              # noqa: BLE001
                cyl_ = dict(p=None, yaw=None, info=dict(why='exception', err=repr(e_)[:200]))
        elif m_.get('table') is not None:                                # 茶几圓柱體深度偵測（同一張影格；幾何法、不讀真值）
            try:
                import cyl_detect as CD_
                p_c, i_c = CD_.detect(m_['dep'], m_['K'], m_['T_cam'], m_['table'])
                cyl_ = dict(p=None if p_c is None else np_.asarray(p_c, float), info=i_c)
            except Exception as e_:                              # noqa: BLE001
                cyl_ = dict(p=None, info=dict(why='exception', err=repr(e_)[:200]))
        leaf_ = None
        if m_.get('leafdet'):                                          # 門扇開度深度偵測（door_leaf_detect；門局部＝世界座標）
            try:
                import door_leaf_detect as DL_
                ph_, il_ = DL_.detect(m_['dep'], m_['K'], m_['T_cam'])
                leaf_ = dict(phi=ph_, info=il_)
            except Exception as e_:                              # noqa: BLE001
                leaf_ = dict(phi=None, info=dict(why='exception', err=repr(e_)[:200]))
        hdl_ = None
        if m_.get('hprior') is not None:                               # 抽屜把手深度偵測（handle_detect；登錄位姿只當搜尋先驗）
            try:
                import handle_detect as HD_
                import yaml as yaml_h_
                if not hasattr(HD_, '_SPEC'):
                    HD_._SPEC = yaml_h_.safe_load(open(m_['hspec']))
                r_h, i_h = HD_.detect(m_['dep'], m_['K'], m_['T_cam'], m_['hprior'], HD_._SPEC)
                hdl_ = dict(origin=None if r_h is None else np_.asarray(r_h['origin'], float), yaw=None if r_h is None else float(r_h['yaw']), info=i_h)
            except Exception as e_:                              # noqa: BLE001
                hdl_ = dict(origin=None, yaw=None, info=dict(why='exception', err=repr(e_)[:200]))
        ms_ = (time.perf_counter() - w0_) * 1e3
        try:
            np_.savez_compressed(os.path.join(fdir, m_['stem'] + '_depth_m.npz'), dep=m_['dep'])
            import imageio.v2 as imageio_
            imageio_.imwrite(os.path.join(fdir, m_['stem'] + '_rgb.png'), m_['rgb'])
            save_ = True
        except Exception:                                        # noqa: BLE001
            save_ = False
        conn.send(('det', dict(seq=m_['seq'], T_est=None if T_est_ is None else np_.asarray(T_est_, float), info=info_, ms=ms_, err=err_, saved=save_, cyl=cyl_, hdl=hdl_, leaf=leaf_)))


os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
_DET, _det_child = _mp.Pipe()
_DET_P = _mp.get_context('fork').Process(target=_det_worker, args=(_det_child, os.path.join(a.out, 'frames')), daemon=True)
_DET_P.start()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
import rclpy                                                     # noqa: E402  先匯入真正的 rclpy（部分專案模組會在缺 rclpy 時裝替身）
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy              # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String         # noqa: E402
from tf2_msgs.msg import TFMessage                               # noqa: E402
_RCLPY_REAL = rclpy
_ROS_PREFIX = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces')
_ROS_SNAP = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in _ROS_PREFIX}
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation, XFormPrim    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics     # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import door_far_link as FL                                       # noqa: E402
import door_motm_grasp_ctrl as G                                 # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_friction, bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCylinder   # noqa: E402
import walkers as WK                                             # noqa: E402

# 專案控制模組（ros_stub）會無條件把 sys.modules 裡的 rclpy 與訊息套件換成替身（它們離線用）。這裡在匯入後把真正的模組放回去：
# 專案模組已拿到的替身參照不受影響（它們不經 ROS 通訊），本程式的 ROS 節點用的是真正的 rclpy。
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in _ROS_PREFIX and k_ not in _ROS_SNAP]:
    sys.modules.pop(k_, None)
sys.modules.update(_ROS_SNAP)
if sys.modules.get('rclpy') is not _RCLPY_REAL or not hasattr(rclpy, 'init') or getattr(sys.modules.get('rclpy.node'), 'Node', None) is not Node:
    raise SystemExit('rclpy 還原失敗 ⇒ 拒跑')
K = G.K
ROBOT, DOOR = '/World/omni_bot', '/World/door'
ARM = [f'joint{i}' for i in range(1, 7)]
FINGER_OPEN = float(G.XO.FINGER_OPEN)
FINGER_TOL_M = 0.0005
DT, D_STEPS = 0.01, 8
STEP_PROF = os.environ.get('D2_STEP_PROF', '0') == '1'               # 主迴圈分項計時（即時化剖析；不改任何行為）
RENDER_EVERY = max(1, int(os.environ.get('D2_RENDER_EVERY', '1')))
RENDER_BURST = max(1, int(os.environ.get('D2_RENDER_BURST', '1')))   # 讀相機前連續繪圖步數（RTX 管線延遲）   # 繪圖頻率：每 N 個物理步繪一次（Howard 11/21 問 RTF 1：每步 100 Hz 繪圖＋讀相機）
_SP_T, _SP_ACC = [0.0], {}


def _sp_mark(name):
    now_ = time.perf_counter(); a_ = _SP_ACC.setdefault(name, [0.0, 0])
    a_[0] += now_ - _SP_T[0]; a_[1] += 1; _SP_T[0] = now_
CONTACT_N, DOOR_DEG = 0.05, 0.05
NO_CATCHUP = os.environ.get('D2_NO_CATCHUP', '1') == '1'          # RTF 上限配速（落後不補步）
R_REM = int(os.environ.get('D2_R_REM', '3'))                       # 放行收到後到套用的剩餘延遲（步）
GRIP_F_ABORT_N, DOOR_PREBIND_DEG = 40.0, 0.5                        # 交權後：夾爪合力上限（模擬工程中止值，同舊開門執行器 40 N）、綁定前門動上限（同 BindGate 0.5°）
JOINT = '/World/door/ideal_grasp_attach'
PICK = os.environ.get('D2_PICK', '0') == '1'                         # 茶几圓柱體取物（穿門後）
WALKERS = os.environ.get('D2_WALKERS', '0') == '1'                   # 行人（動態障礙物）
HANDLE_VISION = os.environ.get('D2_HANDLE_VISION', '0') == '1'      # 抽屜把手視覺偵測（Howard 11/19）
DOOR_CLOSE = os.environ.get('D2_DOOR_CLOSE', '0') == '1'             # 回程關門（Howard 11/19）：門扇開度深度偵測
DRAWER = os.environ.get('D2_DRAWER', '0') == '1'                     # 開抽屜→取筆（Howard 11/18）：建淺抽屜櫃＋筆代理；圓柱不建
OBJ = '/World/marker_proxy' if DRAWER else '/World/cyl'              # 理想連接的物件剛體
CHAIN = DRAWER and PICK and os.environ.get('D2_CHAIN', '0') == '1'   # Howard 11/19：同趟 茶几圓柱 → 車尾收納盒 → 抽屜筆 → 收納盒 → 返回（圓柱與筆都建）
BIN = os.environ.get('D2_BIN', '0') == '1'                           # 車尾收納盒（rear_bin_v1.yaml；碰撞方盒掛在 base_link 剛體上）
OBJS = (['/World/cyl'] if (PICK and (CHAIN or not DRAWER)) else []) + (['/World/marker_proxy'] if DRAWER else [])   # 依任務順序；理想連接／解除依序作用
CMD_TIMEOUT_S, ARM_CMD_TIMEOUT_S = 0.5, 0.2                       # 同乾跑世界節點
WALL_LIMIT_S = float(os.environ.get('D2_WALL_LIMIT_S', '1500'))   # 第二十四趟：筆電實際 RTF＜0.15，1500 s 牆鐘在模擬 217.8 s 切斷返回段 ⇒ 可設
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SCAN_N, SCAN_MIN, SCAN_MAX, RANGE_MIN, RANGE_MAX = 360, -3.14159, 3.14159, 0.12, 10.0   # 同乾跑世界節點
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
LIDAR_Z = float(json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))['lidar_z'])
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float), np.asarray(G.XO.LITE6_SAFE.upper, float)


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def T_of(pos, qwxyz):
    T = np.eye(4)
    T[:3, :3] = W0.wxyz_to_R(np.asarray(qwxyz, float))
    T[:3, 3] = np.asarray(pos, float)
    return T


def T3(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = p[:2]
    return T


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


def stop_still(R, cols, aborted):
    """整機停穩驗收：既有 STOP_STILL（door_open_exec_ctrl.py）——手臂各關節速率 <0.002 rad/s、底盤平移速率 <0.003 m/s、偏航速率 <0.003 rad/s；
    命令歸零且在途管線排清後連續成立 0.3 s（窗口完整）；驗收期限 4 s。平移、偏航分開算，用實際時間差。"""
    out = dict(rule='STOP_STILL：arm <0.002 rad/s、base 平移 <0.003 m/s、偏航 <0.003 rad/s；命令歸零＋管線排清後連續 0.3 s；期限 4 s', valid=False)
    if aborted:
        out['why'] = '工程中止或例外：物理未繼續步進，沒有停穩可驗'
        return out
    if len(R) < 40:
        out['why'] = 'record_too_short'
        return out
    iu = cols.index('uin_x')
    nz = np.where(np.abs(R[:, iu:iu + 9]).max(axis=1) > 1e-9)[0]
    k0 = int(nz[-1]) + 1 if len(nz) else 0
    dt = np.maximum(np.diff(R[:, 0]), 1e-6)
    v = np.linalg.norm(np.diff(R[:, 1:3], axis=0), axis=1) / dt
    w = np.abs(np.diff(R[:, 3])) / dt
    da = np.abs(np.diff(R[:, 4:10], axis=0)).max(axis=1) / dt
    ix = cols.index('ux')
    clear = np.abs(R[1:, ix:ix + 9]).max(axis=1) < 1e-12
    ok = (da < 0.002) & (v < 0.003) & (w < 0.003) & clear
    k_ok = None
    for k_ in range(k0, len(ok)):
        t_end = R[k_, 0] + 0.3
        j = int(np.searchsorted(R[:, 0], t_end))
        if j >= len(ok):
            break
        if ok[k_:j].all():
            k_ok = k_
            break
    t0 = float(R[max(k0 - 1, 0), 0])
    out.update(t_last_nonzero_input=round(t0, 3), settle_window_start_s=None if k_ok is None else round(float(R[k_ok, 0]) - t0, 3), settle_time_s=None if k_ok is None else round(float(R[k_ok, 0]) - t0 + 0.3, 3), record_after_s=round(float(R[-1, 0]) - t0, 2), deadline_s=4.0,
               base_speed_at_last_input=round(float(v[min(max(k0 - 1, 0), len(v) - 1)]), 4), arm_rate_at_last_input=round(float(da[min(max(k0 - 1, 0), len(da) - 1)]), 4),   # 模擬上限時仍在動 ⇒ k0 可能等於長度（第七十四趟剖析趟索引越界）
               residual_after_settle=None if k_ok is None else dict(base_mm_s=round(float(v[k_ok:].max()) * 1e3, 4), yaw_mrad_s=round(float(w[k_ok:].max()) * 1e3, 4), arm_mrad_s=round(float(da[k_ok:].max()) * 1e3, 4)))
    out['valid'] = bool(k_ok is not None and out['settle_time_s'] <= 4.0)
    if k_ok is None:
        out['why'] = 'not_settled_within_record'
    return out


class Bridge(Node):
    def __init__(self):
        super().__init__('d2_isaac_sim')
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.p_cyl = self.create_publisher(Float64MultiArray, '/cyl/obs', 10)
        self.p_hdl = self.create_publisher(Float64MultiArray, '/dr/handle_obs', 10)
        self.p_leaf = self.create_publisher(Float64MultiArray, '/door/leaf_obs', 10)
        self.p_state = self.create_publisher(String, '/far/exec_state', 10)
        self.p_ack = self.create_publisher(String, '/d2/ack', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(Bool, '/far/done', self._done, 10)
        self.create_subscription(String, '/far/abort', self._abort, 10)
        self.abort = None
        self.create_subscription(TFMessage, '/tf', self._tf, 50)
        self.t = 0.0
        self.cmd, self.cmd_t, self.acmd, self.acmd_t, self.done = np.zeros(3), -1.0, np.zeros(6), -1.0, False
        self.map_odom = None                                           # (stamp, x, y, yaw)：最新 map←odom（只記錄）
        self.n_cmd, self.n_acmd = 0, 0
        self.p_d2 = self.create_publisher(Float64MultiArray, '/d2/state', 20)
        self.create_subscription(Float64MultiArray, '/d2/out', self._d2out, 50)
        self.create_subscription(Float64MultiArray, '/d2/seg', self._d2seg, 50)
        self.d2_segs, self.n_now, self.bsched = [], 0, []            # 已核段（起始步、名目手臂、手指、尾段手臂）；本步序號；放行排程（生效步、值、前值）
        self.create_subscription(String, '/d2/ctl', self._d2ctl, 20)
        self.d2_out, self.d2_n0, self.d2_bind, self.d2_end, self.d2_abort, self.d2_ctl_log = {}, None, False, False, None, []

    def _d2out(self, m):
        d = np.asarray(m.data, float)
        if len(d) >= 11 and np.isfinite(d[:10]).all():
            self.d2_out[int(round(d[0]))] = d[1:].copy()
            if len(self.d2_out) > 400:
                for kk in sorted(self.d2_out)[:len(self.d2_out) - 400]:
                    self.d2_out.pop(kk, None)

    def _d2seg(self, m):
        d = np.asarray(m.data, float)
        try:
            n0, ns = int(round(d[0])), int(round(d[1])); i = 2
            arm = d[i:i + 6 * ns].reshape(ns, 6); i += 6 * ns
            fg = d[i:i + ns]; i += ns
            nt = int(round(d[i])); i += 1
            tail = d[i:i + 6 * nt].reshape(nt, 6)
        except Exception:                                             # noqa: BLE001
            return
        self.d2_segs.append((n0, arm.copy(), fg.copy(), tail.copy()))
        self.d2_segs = self.d2_segs[-6:]

    def base_at(self, n):
        e = None
        for e_ in reversed(self.bsched):
            if e_[0] <= n:
                e = e_; break
        if e is None:
            return None
        a = min(n - e[0] + 1, 5) / 5.0
        return e[2] + a * (e[1] - e[2])

    def _d2ctl(self, m):
        try:
            c = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            return
        self.d2_ctl_log.append(dict(c, t_rx=round(self.t, 3)))
        w = c.get('what')
        if w == 'takeover' and self.d2_n0 is None:
            self.d2_n0 = int(c['n0'])
        elif w == 'bind':
            self.d2_bind = True
        elif w == 'attach_cyl':
            self.d2_attach_cyl = True                                # 圓柱理想連接請求（夾住證據成立後）
        elif w == 'drawer_reset':
            self.dr_reset = True                                     # 關抽屜：第二次連接前重設
        elif w == 'drawer_pose':
            self.dr_pose_vis = c.get('pose')                         # 監督融合的視覺抽屜位姿 ⇒ 筆偵測內部範圍依此
        elif w == 'detach_obj':
            self.d2_detach_obj, self.d2_detach_req = True, c.get('req_n')   # 車尾收納盒：解除夾爪↔物體連接
        elif w == 'attach_drawer':
            self.d2_attach_dr = True                                 # 抽屜把手理想連接請求（夾住證據成立後）
        elif w == 'unbind_drawer':
            self.d2_unbind_dr, self.d2_unbind_dr_req, self.dr_q_open = True, c.get('req_n'), c.get('q_open')
        elif w == 'door_reset':
            self.door_reset = True                                   # 回程關門：門扇第二次綁定前重設
        elif w == 'unbind':
            self.d2_unbind = True                                    # 穿門段：放手 ⇒ 解除 FixedJoint
            self.d2_unbind_req = c.get('req_n')
        elif w == 'end':
            self.d2_end = True
        elif w == 'abort' and self.d2_abort is None:
            self.d2_abort = str(c.get('why'))

    def _cmd(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z], float)
        if np.isfinite(v).all():
            self.cmd, self.cmd_t = v, self.t
            self.n_cmd += 1
            if self.d2_n0 is not None:                                # 交權後：放行依收件順序在收件步＋R_REM 生效（拍內斜坡；同控制伺服器的因果排程）
                k0 = self.n_now + R_REM
                if self.bsched and self.bsched[-1][0] >= k0:
                    self.bsched[-1] = (self.bsched[-1][0], v.copy(), self.bsched[-1][2])
                else:
                    vp = self.base_at(k0 - 1)
                    self.bsched.append((k0, v.copy(), vp if vp is not None else v.copy()))
                    self.bsched = self.bsched[-50:]

    def _acmd(self, m):
        if len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t
            self.n_acmd += 1

    def _abort(self, m):
        if self.abort is None:
            self.abort = str(m.data)

    def _done(self, m):
        self.done = self.done or bool(m.data)

    def _tf(self, m):
        for tr in m.transforms:
            if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                o = tr.transform.rotation
                self.map_odom = (tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9, float(tr.transform.translation.x), float(tr.transform.translation.y),
                                 math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z)))

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(self.t)
        s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        if s.nanosec >= 1000000000:
            s.sec, s.nanosec = s.sec + 1, s.nanosec - 1000000000
        return s


def main():
    os.makedirs(a.out, exist_ok=True)
    fdir = os.path.join(a.out, 'frames')
    os.makedirs(fdir, exist_ok=True)
    reg = json.load(open(a.reg))
    start_b = np.array([float(x) for x in a.start.split(',')], float)
    start_q = np.array([float(x) for x in a.arm.split(',')], float)
    static = a.static_s > 0

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[fc] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:                                    # 未知靜物照場景登錄放；本趟有 LiDAR 與原導航避障
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
        env_paths.append(f"/World/unknown/{o['name']}")
    walk_list, walk_prims = [], []
    if WALKERS:                                                       # 行人（Howard 11/18：路線要影響任務）：有碰撞的圓柱，每步依 walkers.walker_pos 移動；雷射（PhysX 射線）看得到；碰到底盤 ⇒ 底盤接觸中止
        for w_ in ob.get('walkers', []):
            p_ = WK.walker_pos(w_, 0.0)
            FixedCylinder(prim_path=f"/World/walkers/{w_['name']}", name=w_['name'], position=np.array([p_[0], p_[1], w_['height'] / 2]), radius=float(w_['radius']), height=float(w_['height']), color=np.array([0.55, 0.25, 0.65]))
            env_paths.append(f"/World/walkers/{w_['name']}")
            walk_list.append(w_)
    cyl_info, dr_info = None, None
    if DRAWER:                                                        # 淺抽屜櫃（drawer_unit_shallow30，登錄位姿）＋筆代理（登錄中心；隨抽屜滑出）
        import drawer_asset as DRA_
        import house_assets as HA_
        import marker_proxy_asset as MK_
        import yaml as yaml_
        dspec_ = DRA_.load(os.path.join(WS, reg['drawer']['asset']))
        dr_pose_true = list(reg['drawer']['pose'])                    # 真實擺放（D2_DRAWER_TRUE：登錄＋本地偏移 dx, dy, dyaw°；Howard 11/19 要靠視覺，登錄只當搜尋先驗）
        if os.environ.get('D2_DRAWER_TRUE'):
            import drawer_scene as DRT_
            _dt = [float(x) for x in os.environ['D2_DRAWER_TRUE'].split(',')]
            _px, _py, _yw = DRT_.offset_pose(*_dt)
            dr_pose_true = [_px, _py, math.degrees(_yw)]
        drb_ = HA_.build_drawer_yaw(stage, DRA_, dspec_, dr_pose_true)
        mspec_ = yaml_.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
        _yr = math.radians(float(dr_pose_true[2])); _cl = dspec_['marker_placement']['center'] if isinstance(dspec_, dict) and 'marker_placement' in dspec_ else reg['marker']['center_drawer']
        mk_c_true = [float(dr_pose_true[0]) + math.cos(_yr) * _cl[0] - math.sin(_yr) * _cl[1], float(dr_pose_true[1]) + math.sin(_yr) * _cl[0] + math.cos(_yr) * _cl[1], float(_cl[2])]
        mobj_, mmat_ = HA_.build_marker_yaw(world, MK_, mspec_, mk_c_true, float(reg['marker']['axis_yaw_deg']) + float(dr_pose_true[2]) - float(reg['drawer']['pose'][2]))
        for n_, _, _ in dspec_['cabinet']:
            env_paths.append(f'/World/drawer_unit/cabinet/{n_}')
        dr_info = dict(asset=reg['drawer']['asset'], pose=reg['drawer']['pose'], pose_true=dr_pose_true, marker_true=mk_c_true, pull=reg['drawer']['pull_dir'], readback=drb_, marker0=reg['marker']['center_world'], marker_yaw_deg=reg['marker']['axis_yaw_deg'])
    if PICK and (CHAIN or not DRAWER):                                # 茶几圓柱體（Howard 11/17「隨機放」；規格＝瓶子代理物；位置 seed 登錄；真值只做驗收記錄；串接時與抽屜同建）
        tab_ = next(f_ for f_ in reg['furniture'] if f_['name'] == 'coffee_table')
        if os.environ.get('CYL_XY'):
            cxy = np.array([float(x) for x in os.environ['CYL_XY'].split(',')])
        else:
            rc_ = np.random.default_rng(int(os.environ.get('CYL_SEED', '0')))
            e_ = os.environ.get('CYL_SIDE') == 'east'                 # Howard 11/21「放桌子右邊」＋「不要寫死」：右側邊緣帶隨機（離東緣 0.05–0.18、離兩端 ≥0.25；更內側從走道搆不到）
            x_lo_ = tab_['c'][0] + tab_['s'][0] / 2 - 0.18 if e_ else tab_['c'][0] - tab_['s'][0] / 2 + 0.05
            y_m_ = 0.25 if e_ else 0.05
            cxy = np.array([rc_.uniform(x_lo_, tab_['c'][0] + tab_['s'][0] / 2 - 0.05), rc_.uniform(tab_['c'][1] - tab_['s'][1] / 2 + y_m_, tab_['c'][1] + tab_['s'][1] / 2 - y_m_)])
        DynamicCylinder(prim_path='/World/cyl', name='cyl', position=np.array([cxy[0], cxy[1], float(tab_['s'][2]) + 0.10]), radius=0.015, height=0.20, mass=0.20, color=np.array([0.2, 0.6, 0.9]))
        n_cf = bind_friction(stage, [stage.GetPrimAtPath('/World/cyl')], 1.0, 0.8, 0.0, mat_path='/World/PhysicsMaterials/cyl')
        n_tf = bind_friction(stage, [stage.GetPrimAtPath('/World/coffee_table')], 0.6, 0.5, 0.0, mat_path='/World/PhysicsMaterials/coffee_table')
        cyl_info = dict(xy0=cxy.tolist(), seed=int(os.environ.get('CYL_SEED', '0')), radius=0.015, height=0.20, mass=0.20, mu_cyl=[1.0, 0.8], mu_table=[0.6, 0.5], bound=[int(n_cf), int(n_tf)], table=dict(c=tab_['c'], s=tab_['s']))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(start_b[0]), float(start_b[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(start_b[2])))
    apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    if bind_frictionless(stage, sph + collision_prims(stage, '/World/ground')) == 0:
        print('[fc] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    name_of = lambda b: str(b).rstrip('/').split('/')[-1]            # noqa: E731
    tool_names = tuple(f'link{i}' for i in range(1, 7)) + ('link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
    tool_bodies = [b for b in bodies if name_of(b) in tool_names]
    b_base = [b for b in bodies if name_of(b) == 'base_link']
    if not arts or len(tool_bodies) != len(tool_names) or len(b_base) != 1:
        print(f'[fc] 剛體不符：{[name_of(b) for b in tool_bodies]}／base {b_base}／arts {arts} ⇒ 拒跑', flush=True)
        return 16
    v_tool = [RigidPrim(prim_paths_expr=str(b), name=f'v_{name_of(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True) for b in tool_bodies]
    base_filters = env_paths + [dauth['leaf_prim']] + (['/World/drawer_unit/drawer'] if DRAWER else [])
    v_base = RigidPrim(prim_paths_expr=str(b_base[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=base_filters)
    leaf_v = RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_state')
    grip_b = [b for b in bodies if name_of(b) == 'uflite_gripper_link']
    v_grip = RigidPrim(prim_paths_expr=str(grip_b[0]), name='v_grip_pose') if len(grip_b) == 1 else None
    bin_info = None
    if BIN:                                                           # 車尾收納盒：五片碰撞方盒（底＋四壁）掛在 base_link 剛體（隨底盤；不另加質量）
        import rear_bin as RBIN_
        bp_ = str(b_base[0]) + '/rear_bin'
        c_, h_, w_t, b_t = RBIN_.C_LOC.copy(), RBIN_.HS.copy(), float(RBIN_.SPEC['wall']), float(RBIN_.SPEC['bottom'])
        c_[2] -= 0.05                                                 # base_footprint → base_link（base_joint z 0.05）
        parts_ = dict(bottom=(c_ + np.array([0, 0, -h_[2] + b_t / 2]), np.array([h_[0], h_[1], b_t / 2])),
                      wall_xp=(c_ + np.array([h_[0] - w_t / 2, 0, 0]), np.array([w_t / 2, h_[1], h_[2]])), wall_xn=(c_ + np.array([-h_[0] + w_t / 2, 0, 0]), np.array([w_t / 2, h_[1], h_[2]])),
                      wall_yp=(c_ + np.array([0, h_[1] - w_t / 2, 0]), np.array([h_[0], w_t / 2, h_[2]])), wall_yn=(c_ + np.array([0, -h_[1] + w_t / 2, 0]), np.array([h_[0], w_t / 2, h_[2]])))
        UsdGeom.Xform.Define(stage, bp_)
        for nm_, (pc_, ph_) in parts_.items():
            cb_ = UsdGeom.Cube.Define(stage, f'{bp_}/{nm_}')
            cb_.CreateSizeAttr(1.0)
            xb_ = UsdGeom.Xformable(cb_)
            xb_.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pc_]))
            xb_.AddScaleOp().Set(Gf.Vec3f(*[float(2 * v) for v in ph_]))
            cb_.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.2)])
            UsdPhysics.CollisionAPI.Apply(cb_.GetPrim())
        bin_info = dict(spec=os.path.relpath(RBIN_.SPEC_PATH, WS), prim=bp_, parts={k_: dict(c_base_link=[round(float(v), 4) for v in pc_], half=[round(float(v), 4) for v in ph_]) for k_, (pc_, ph_) in parts_.items()}, binned=[])
    v_objs = [RigidPrim(prim_paths_expr=o_, name=f'v_obj{i_}_pose') for i_, o_ in enumerate(OBJS)] if PICK else []
    obj_i = 0
    v_cyl = v_objs[0] if v_objs else None
    v_dr = RigidPrim(prim_paths_expr='/World/drawer_unit/drawer', name='v_drawer_pose') if DRAWER else None
    walk_views = [XFormPrim(prim_paths_expr=f"/World/walkers/{w_['name']}", name=f"wv_{w_['name']}") for w_ in walk_list]
    world.reset()
    for v in v_tool + [v_base, leaf_v] + ([v_grip] if v_grip is not None else []) + ([v_dr] + v_objs if DRAWER else []):
        v.initialize()
    if v_grip is None:
        print('[fc] 夾爪剛體不唯一 ⇒ 拒跑', flush=True)
        return 16
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[fc] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(start_q[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = FINGER_OPEN, FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def apply(tgt_full):
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt_full, np.float32)[act], joint_indices=act))

    def base_cmd(u, yw):
        vx, vy, wz = (float(v) for v in u)
        robot.set_linear_velocity(np.array([vx * math.cos(yw) - vy * math.sin(yw), vx * math.sin(yw) + vy * math.cos(yw), 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))

    def q9():
        p, qt = robot.get_world_pose()
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), [float(jp[i]) for i in idx]], jp

    def door_q():
        _, o = leaf_v.get_world_poses()
        w, x, y, z = (float(v) for v in np.asarray(o)[0])
        return -math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    hprims = [pr for pr in collision_prims(stage, DOOR) if str(pr.GetPath()) == f'{DOOR}/leaf/handle_bar']
    if len(hprims) != 1:
        print(f'[fc] 把手 prim 數 {len(hprims)} ≠ 1 ⇒ 拒跑', flush=True)
        return 22

    def truth_handle():                                               # 只供事後評估
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprims[0])
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    eef = [b for b in bodies if name_of(b) == 'link_eef'][0]
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    cam = Camera(prim_path=f'{eef}/wrist_cam', resolution=CAM['res'], frequency=CAM['hz'])
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.set_horizontal_aperture(CAM['aperture'])
    cam.set_focal_length(CAM['aperture'] / (2.0 * math.tan(math.radians(CAM['hfov_deg']) / 2.0)))
    cam.set_clipping_range(*CAM['clip'])
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([3.4, -4.2, 3.2]), target=np.array([1.0, -0.8, 0.6]))
    except Exception as e:                                            # noqa: BLE001
        print(f'[fc] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    try:
        g_kp, g_kd = robot.get_articulation_controller().get_gains()
        gains_rb = dict(arm_kp=[float(np.asarray(g_kp).reshape(-1)[i]) for i in idx], arm_kd=[float(np.asarray(g_kd).reshape(-1)[i]) for i in idx], finger_kp=float(np.asarray(g_kp).reshape(-1)[f1]), finger_kd=float(np.asarray(g_kd).reshape(-1)[f1]), source='ArticulationController.get_gains()')
    except Exception as e:                                            # noqa: BLE001
        gains_rb = dict(error=repr(e))
    readback = dict(gains_api_readback=gains_rb, gains_requested=dict(kp=a.kp, kd=a.kd, finger_kp=a.finger_kp, finger_kd=a.finger_kd), physics_dt=float(world.get_physics_dt()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()), kp=a.kp, kd=a.kd, start_base=start_b.tolist(), start_q=start_q.tolist(),
                    finger_open=FINGER_OPEN, static_s=a.static_s, cam_hz=a.cam_hz, d_steps=D_STEPS)
    print(f'[fc] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - DT) > 1e-12 or not readback['ext_forces_every_iter']:
        print('[fc] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64
    # 射線查詢（LiDAR）：略過機器人自身的所有形狀（乾跑世界節點同樣不含機器人自身；下游 scan_relay 另有自體扇區遮罩）
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    def scan(q):
        org = [float(q[0]), float(q[1]), LIDAR_Z]
        out, who_min = [], (float('inf'), None)
        for i in range(SCAN_N):
            th = float(q[2]) + SCAN_MIN + i * SCAN_INC
            best = [float('inf'), None]

            def rep(hit, best=best):
                coll = str(getattr(hit, 'collision', '') or '')
                body = str(getattr(hit, 'rigid_body', '') or getattr(hit, 'rigidBody', '') or '')
                if coll.startswith(ROBOT) or body.startswith(ROBOT):
                    return True
                d = float(getattr(hit, 'distance', 0.0))
                if RANGE_MIN <= d < best[0]:
                    best[0], best[1] = d, coll or body
                return True
            query.raycast_all(org, [math.cos(th), math.sin(th), 0.0], RANGE_MAX, rep)
            out.append(best[0])
            if best[0] < who_min[0]:
                who_min = (best[0], best[1])
        return out, who_min

    tgt = np.array(robot.get_joint_positions(), float)
    quat0 = np.array([math.cos(start_b[2] / 2), 0.0, 0.0, math.sin(start_b[2] / 2)], np.float32)
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([start_b[0], start_b[1], 0.0], np.float32), quat0)
        apply(tgt)
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
    Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    q_s, jp_s = q9()
    if abs(float(jp_s[f1]) - FINGER_OPEN) > FINGER_TOL_M or abs(float(jp_s[f2]) - FINGER_OPEN) > FINGER_TOL_M:
        print(f'[fc] 手指張開量 {float(jp_s[f1])}/{float(jp_s[f2])} 與模型 {FINGER_OPEN} 不符 ⇒ 拒跑', flush=True)
        return 65
    door_q0 = door_q()
    r0, who0 = scan(q_s)
    truth = dict(note='只供事後評估；不進任何發布的量', T_handle_start=truth_handle().tolist(), door_q0_rad=door_q0)
    print(f'[fc] 起點讀回 底盤 {np.round(q_s[:3], 4).tolist()} 手臂靜態偏移 {np.round(q_s[3:] - start_q, 4).tolist()} 門角 {math.degrees(door_q0):.4f}° 掃描最近 {who0}', flush=True)
    if not np.isfinite(who0[0]):
        print('[fc] 掃描沒有任何回波 ⇒ 拒跑', flush=True)
        return 68

    rclpy.init()
    node = Bridge()
    tg_, _ = _DET.recv()
    if tg_ != 'ready':
        print('[fc] 偵測程序未就緒 ⇒ 拒跑', flush=True)
        return 69
    det_busy, det_pend = False, {}
    pace = dict(behind_steps=0, behind_max_s=0.0, slept_s=0.0)
    T0 = float(world.current_time)
    hist, frames, rows, events = {}, [], [], []
    cam_stat = dict(read=0, new=0, used=0, rejects={}, valid=0, fail={}, detect_ms=[])
    last_rf, t_next_cam, seq = [None], 0.0, 0
    s = q_s[3:].copy()                                                # 手臂設定點
    pipe = [np.zeros(9) for _ in range(D_STEPS)]
    latch, contact_max, clock_bad, step_wall = None, dict(n=0.0, who=None), 0, []
    w_all0 = time.perf_counter()
    t_done, scan_min_log = None, float('nan')
    n_steps = int(round((a.static_s if static else a.sim_limit_s) / DT))
    scan_every, pub_every = max(int(round(1.0 / (a.scan_hz * DT))), 1), 2

    def ev(what, **kw):
        """事件時間只在這裡定義一次（kw 內若帶 t 會被改名為 t_detail，不重複）；記錄失敗不得影響呼叫端。"""
        try:
            kw = {('t_detail' if k_ == 't' else k_): v_ for k_, v_ in kw.items()}
            events.append(dict(kw, t=round(node.t, 3), what=what))
            print(f'[fc] t={node.t:.2f} {what} {kw}', flush=True)
        except Exception as e_:                                       # noqa: BLE001
            events.append(dict(t=round(node.t, 3), what=what, record_error=repr(e_)))
            print(f'[fc] 事件記錄失敗 {what}: {e_!r}', flush=True)

    ffor = {'uflite_finger1': 0.0, 'uflite_finger2': 0.0}
    walk_log = []
    cyl_log = []
    obj_log = []
    det_ack_ = None
    closing2 = False

    def contacts(split=False):
        mx, who = 0.0, None
        for v, b in zip(v_tool, tool_bodies):
            try:
                n_ = float(np.linalg.norm(np.asarray(v.get_net_contact_forces(dt=DT), float).reshape(-1)[:3]))
            except Exception as e:                                    # noqa: BLE001
                return (None, f'read_fail:{e!r}', None, None) if split else (None, f'read_fail:{e!r}')
            if name_of(b) in ('uflite_finger1', 'uflite_finger2'):
                ffor[name_of(b)] = n_                                 # 兩指各自接觸力（取物證據）
            if n_ > mx:
                mx, who = n_, name_of(b)
        try:
            Mb = np.asarray(v_base.get_contact_force_matrix(dt=DT), float).reshape(-1, len(base_filters), 3)
            nb = np.linalg.norm(Mb[0], axis=1)
            mb, wb = float(nb.max()), 'base_link|' + base_filters[int(nb.argmax())]
        except Exception as e:                                        # noqa: BLE001
            return (None, f'read_fail:{e!r}', None, None) if split else (None, f'read_fail:{e!r}')
        if split:
            return mx, who, mb, wb
        if mb > mx:
            mx, who = mb, wb
        return mx, who

    def grab():
        fr = cam.get_current_frame()
        cam_stat['read'] += 1
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is None or rf == last_rf[0] or rf == (0, 0) or col is None or fr.get('distance_to_image_plane') is None:
            return None
        last_rf[0] = rf
        cam_stat['new'] += 1

        def rej(why):
            cam_stat['rejects'][why] = cam_stat['rejects'].get(why, 0) + 1
        t_r = fr.get('rendering_time')
        if t_r is None:
            return rej('no_rendering_time')
        hp = hist.get(int(round(float(t_r) / DT)))
        if hp is None or abs(hp['t_sim'] - float(t_r)) > 0.25 * DT:
            return rej('no_state_at_render_time')
        dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
        rgba = np.asarray(col)
        if dep.size == 0 or rgba.size == 0:
            return rej('empty_image')
        T_fk = K.fk(hp['q'], W0.OPT)
        return dict(t_obs=float(t_r) - T0, dep=dep[:, :, 0] if dep.ndim == 3 else dep, rgb=rgba[:, :, :3].astype(np.uint8), T_cam=T_fk, q=hp['q'], rf=rf,
                    fk_vs_render_m=float(np.linalg.norm(T_fk[:3, 3] - hp['T_render'][:3, 3])), map_odom=hp['map_odom'], s=hp['s'])

    exc = None
    scans = []
    d2_stat, d2_last, bound, bind_pending, fcmd, released, f_hold, cyl_attached = dict(late=0, on_time=0), None, False, False, FINGER_OPEN, False, None, False
    dr_bound, dr_released, dr_log = False, False, []
    if DRAWER:
        dr_p0 = np.asarray(v_dr.get_world_poses()[0], float)[0].copy()
        dr_pull = np.array([math.sin(math.radians(float(dr_pose_true[2]))), -math.cos(math.radians(float(dr_pose_true[2]))), 0.0])   # 真實拉出方向（開度真值記錄用）
        import drawer_scene as DRS_
    pace_off = 0.0
    try:
        for k in range(n_steps):
            w_step0 = time.perf_counter()
            if STEP_PROF: _SP_T[0] = w_step0
            t_sim = float(world.current_time)
            node.t = t = t_sim - T0
            q, jp = q9()
            pos_r, q_r = cam.get_world_pose(camera_axes='ros')
            hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), T_render=T_of(pos_r, q_r), map_odom=node.map_odom, s=s.copy())
            if len(hist) > 300:
                for kk in sorted(hist)[:len(hist) - 300]:
                    hist.pop(kk, None)
            # ---- 保護
            n_now = int(round(t / DT))
            post = node.d2_n0 is not None and n_now >= node.d2_n0
            phi_now = door_q()
            node.n_now = n_now
            kb_ = []
            for i_ in range(R_REM + 1):                               # 步 n…n＋R_REM 的實際底盤（已收到的放行決定；排程未生效前＝在途管線輸出）
                b_ = node.base_at(n_now + i_) if post else None
                if b_ is None:
                    b_ = np.asarray(pipe[i_][:3], float) if i_ < len(pipe) else np.full(3, np.nan)
                kb_ += [float(x) for x in b_]
            node.p_d2.publish(Float64MultiArray(data=[float(n_now), float(t), *[float(x) for x in q], float(phi_now), float(jp[f1]), float(jp[f2]), *[float(x) for x in s], *[float(x) for x in np.concatenate(pipe)], *kb_, float(ffor['uflite_finger1']), float(ffor['uflite_finger2'])]))   # 控制伺服器（world_d2_node）每步狀態（尾端＝兩指接觸力）
            cn_t, cwho, cn_b, cwho_b = contacts(split=True)
            cn = None if cn_t is None else max(cn_t, cn_b)
            dq_deg = abs(math.degrees(phi_now - door_q0))
            why = None
            if cn is None:
                why = ('contact_read_fail', dict(detail=cwho))
            else:
                if cn > contact_max['n']:
                    contact_max.update(n=cn, who=cwho if cn_t >= cn_b else cwho_b)
                if not post and cn > CONTACT_N:
                    why = ('unexpected_contact', dict(force_n=round(cn, 4), who=cwho if cn_t >= cn_b else cwho_b))
                elif post and cn_b > CONTACT_N:
                    why = ('base_contact', dict(force_n=round(cn_b, 4), who=cwho_b))
                elif post and cn_t > GRIP_F_ABORT_N:
                    why = ('gripper_force_over', dict(force_n=round(cn_t, 3), who=cwho, limit_n=GRIP_F_ABORT_N))
            if why is None and not post and dq_deg > DOOR_DEG:
                why = ('door_moved', dict(door_deg=round(dq_deg, 4)))
            if why is None and post and not bound and not released and not closing2 and dq_deg > DOOR_PREBIND_DEG:   # 回程關門重設後門本來就開著，不適用
                why = ('door_moved_before_bind', dict(door_deg=round(dq_deg, 4)))
            if why is None and not post and max(abs(float(jp[f1]) - FINGER_OPEN), abs(float(jp[f2]) - FINGER_OPEN)) > FINGER_TOL_M:
                why = ('finger_opening_out_of_model', dict(q1=float(jp[f1]), q2=float(jp[f2])))
            if why is None and not (np.isfinite(q).all() and np.isfinite(jp).all() and math.isfinite(dq_deg)):
                why = ('non_finite_state', {})
            if why is not None and latch is None:
                latch = dict(t=round(t, 3), why=why[0], kind='模擬工程中止（保留現場資料、不再步進物理；不是受控停止）', **why[1])
                rows.append([t, *q, *s, *np.zeros(9), *np.zeros(9), 0.0 if cn is None else cn, dq_deg, scan_min_log, 0.0, 1.0, float(post), float(bound), float(jp[f1]), float(jp[f2]), float(phi_now)])   # 中止當步的紀錄列先寫，再記事件
                ev('latched_sim_engineering_abort', **latch)
                node.p_state.publish(String(data='latched:' + latch['why']))
                break
            if walk_views:                                            # 行人移動（模擬時間；D2_WALKER_YIELD 時各自進度、遇機器人讓路）
                WK.crowd_step(t, q[:2], walk_list)
                for w_, v_ in zip(walk_list, walk_views):
                    p_ = WK.crowd_pos(w_, t)
                    v_.set_world_poses(positions=np.array([[p_[0], p_[1], w_['height'] / 2]], np.float32))
                if k % 10 == 0:
                    walk_log.append([float(t)] + [float(x) for w_ in walk_list for x in WK.crowd_pos(w_, t)])
            if STEP_PROF: _sp_mark('state')
            # ---- ROS 收命令
            for _ in range(4):
                rclpy.spin_once(node, timeout_sec=0.0)
            if node.d2_abort is not None and node.abort is None:
                node.abort = 'd2_world:' + node.d2_abort
            if node.abort is not None and latch is None:              # 監督節點要求模擬工程中止：沒有核過的停法 ⇒ 不再步進物理
                latch = dict(t=round(t, 3), why='supervisor_abort', detail=node.abort[:200], kind='模擬工程中止（保留現場資料、不再步進物理；不是受控停止）')
                rows.append([t, *q, *s, *np.zeros(9), *np.zeros(9), 0.0 if cn is None else cn, dq_deg, scan_min_log, 0.0, 1.0, float(post), float(bound), float(jp[f1]), float(jp[f2]), float(phi_now)])   # 中止當步的紀錄列先寫，再記事件
                ev('latched_sim_engineering_abort', **latch)
                break
            u_in = node.cmd if (t - node.cmd_t <= CMD_TIMEOUT_S and node.cmd_t >= 0) else np.zeros(3)
            ua_in = node.acmd if (t - node.acmd_t <= ARM_CMD_TIMEOUT_S and node.acmd_t >= 0) else np.zeros(6)
            if static:
                u_in, ua_in = np.zeros(3), np.zeros(6)
            if not post:
                pipe.append(np.r_[u_in, ua_in])
                ue = pipe.pop(0)
                s = np.clip(s + DT * ue[3:], LO, HI)
                fcmd = FINGER_OPEN
            else:                                                     # 交權後：本地播放控制伺服器下發的已核段／尾段（Codex 20261117_173459）；底盤＝放行排程
                if n_now % 5 == node.d2_n0 % 5:                       # 拍首：等新段到本步配速時限為止（物理照牆鐘，不多等）
                    tgt_w_ = w_all0 + pace_off + (t + DT) / a.rtf
                    while not any(sg[0] == n_now for sg in node.d2_segs) and time.perf_counter() < tgt_w_:
                        rclpy.spin_once(node, timeout_sec=0.001)
                ent, src_ = None, None
                for sg in reversed(node.d2_segs):
                    n0_, arm_, fg_, tl_ = sg
                    if n0_ > n_now:
                        continue                                      # 提前下發、尚未開始的段
                    if n0_ <= n_now < n0_ + len(arm_):
                        ent, src_ = (arm_[n_now - n0_], fg_[n_now - n0_]), 'seg'; break
                    if n0_ + len(arm_) <= n_now < n0_ + len(arm_) + len(tl_):
                        ent, src_ = (tl_[n_now - n0_ - len(arm_)], np.nan), 'tail'; break
                    if n_now >= n0_ + len(arm_) + len(tl_) and len(tl_):
                        ent, src_ = (tl_[-1], np.nan), 'tail_end'; break   # 已核停止尾段播完＝已停：保持其末端
                    break                                             # 只看最新一段（舊段不回放）
                if ent is None:
                    d2_stat['no_segment'] = d2_stat.get('no_segment', 0) + 1
                    if latch is None:
                        latch = dict(t=round(t, 3), why='d2_no_valid_segment', n=n_now, kind='模擬工程中止（無有效名目段或已核尾段；保留現場資料、不再步進物理）')
                        ev('latched_sim_engineering_abort', **latch)
                    break
                d2_stat[src_] = d2_stat.get(src_, 0) + 1
                s_prev = s.copy()
                s = np.clip(np.asarray(ent[0], float), LO, HI)
                pipe.append(np.zeros(9)); pe_ = pipe.pop(0)            # 在途管線排空（交權前已送出的命令仍照原時刻生效）
                b_ = node.base_at(n_now)
                ub_ = np.asarray(b_ if b_ is not None else pe_[:3], float)
                if src_ == 'tail_end' and float(np.abs(ub_).max()) > 1e-3:   # 本地尾段只含手臂：尾段播完、無新段而底盤命令未歸零 ⇒ 工程中止（Codex 20261117_175718；tail_end 不等於整機已停）
                    if latch is None:
                        latch = dict(t=round(t, 3), why='d2_tail_end_base_nonzero', n=n_now, base_cmd=[round(float(x), 4) for x in ub_], kind='模擬工程中止（本地尾段耗盡而底盤命令未歸零；不再步進物理）')
                        ev('latched_sim_engineering_abort', **latch)
                    break
                ue = np.r_[ub_, (s - s_prev) / DT]
                if np.isfinite(ent[1]):
                    fcmd = float(ent[1])
                if bind_pending and not bound and not released:        # 綁定：FixedJoint 以當下夾爪對門扇的相對位姿（理想抓持假設）；已解綁不再建立（第十趟：旗標未清 ⇒ 解綁後下一步重建）
                    Tg_ = T_of(*[np.asarray(x)[0] for x in v_grip.get_world_poses()])
                    Tl_ = T_of(*[np.asarray(x)[0] for x in leaf_v.get_world_poses()])
                    Rr0 = np.linalg.inv(Tl_) @ Tg_
                    j_ = UsdPhysics.FixedJoint.Define(stage, JOINT)
                    j_.CreateBody0Rel().SetTargets([Sdf.Path(str(grip_b[0]))])
                    j_.CreateBody1Rel().SetTargets([Sdf.Path(dauth['leaf_prim'])])
                    j_.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    j_.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    qr_ = W0.R_to_wxyz(Rr0[:3, :3])
                    j_.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rr0[:3, 3]]))
                    j_.CreateLocalRot1Attr().Set(Gf.Quatf(float(qr_[0]), float(qr_[1]), float(qr_[2]), float(qr_[3])))
                    j_.CreateJointEnabledAttr().Set(True)
                    j_.CreateBreakForceAttr().Set(float('inf')); j_.CreateBreakTorqueAttr().Set(float('inf'))
                    bound, bind_pending = True, False
                    f_hold = float(jp[f1])                             # 綁定實際生效時鎖存手指開度（Codex 第十趟：理想綁定下載荷由 FixedJoint 承擔，手指不再朝全閉持續施力）
                    ev('finger_hold_latched', finger1=round(float(jp[f1]), 6), finger2=round(float(jp[f2]), 6))
                    ev('bind_fixed_joint', door_deg=round(math.degrees(phi_now), 4), rel_pos=[round(float(v), 5) for v in Rr0[:3, 3]], note='理想抓持假設（FixedJoint）')
            if node.d2_bind and not bound and not released:
                bind_pending = True
            if getattr(node, 'd2_attach_cyl', False) and PICK and not cyl_attached and v_grip is not None:   # Howard 11/18「夾到就改成理想連接」：FixedJoint 以當下夾爪對圓柱的相對位姿（理想抓持假設）
                Tg_ = T_of(*[np.asarray(x)[0] for x in v_grip.get_world_poses()])
                Tc_ = T_of(*[np.asarray(x)[0] for x in v_cyl.get_world_poses()])
                Rc0 = np.linalg.inv(Tc_) @ Tg_
                jc_path_ = '/World/cyl_ideal_attach' if obj_i == 0 else f'/World/obj_ideal_attach_{obj_i}'
                jc_ = UsdPhysics.FixedJoint.Define(stage, jc_path_)
                jc_.CreateBody0Rel().SetTargets([Sdf.Path(str(grip_b[0]))])
                jc_.CreateBody1Rel().SetTargets([Sdf.Path(OBJS[obj_i])])
                jc_.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                jc_.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                qc_ = W0.R_to_wxyz(Rc0[:3, :3])
                jc_.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rc0[:3, 3]]))
                jc_.CreateLocalRot1Attr().Set(Gf.Quatf(float(qc_[0]), float(qc_[1]), float(qc_[2]), float(qc_[3])))
                jc_.CreateJointEnabledAttr().Set(True)
                jc_.CreateBreakForceAttr().Set(float('inf')); jc_.CreateBreakTorqueAttr().Set(float('inf'))
                cyl_attached = True
                node.d2_attach_cyl = False                              # 請求已執行（下一件需新請求）
                ev('cyl_attach_fixed_joint', obj=OBJS[obj_i], rel_pos=[round(float(v), 5) for v in Rc0[:3, 3]], cyl_z=round(float(Tc_[2, 3]), 4), note='理想抓持假設（FixedJoint）：夾住證據成立後')
            if BIN and getattr(node, 'd2_detach_obj', False):         # 車尾收納盒：解除夾爪↔物體理想連接（物體靠物理落入盒內）
                node.d2_detach_obj = False
                if cyl_attached:
                    jc_path_ = '/World/cyl_ideal_attach' if obj_i == 0 else f'/World/obj_ideal_attach_{obj_i}'
                    stage.GetPrimAtPath(jc_path_).GetAttribute('physics:jointEnabled').Set(False)
                    v_jc_ = stage.GetPrimAtPath(jc_path_).GetAttribute('physics:jointEnabled').Get()
                    cyl_attached = False
                    Tb_ = T_of(*[np.asarray(x)[0] for x in v_base.get_world_poses()])
                    po_ = np.asarray(v_objs[obj_i].get_world_poses()[0], float)[0]
                    bin_info['binned'].append(dict(obj=OBJS[obj_i], t=round(t, 3), n=int(n_now), req=getattr(node, 'd2_detach_req', None), joint_enabled_readback=str(v_jc_), obj_in_base_link_at_release=[round(float(v), 4) for v in (np.linalg.inv(Tb_) @ np.r_[po_, 1.0])[:3]]))
                    ev('obj_detach_fixed_joint', obj=OBJS[obj_i], n=int(n_now), readback=str(v_jc_))
                    if v_jc_ is not False and latch is None:
                        latch = dict(t=round(t, 3), why='obj_detach_not_confirmed', value=str(v_jc_), kind='模擬工程中止（物體連接解除未確認）')
                        ev('latched_sim_engineering_abort', **latch)
                        node.p_state.publish(String(data='latched:' + latch['why']))
                        break
                    det_ack_ = dict(what='obj_detached', n=int(n_now), req=getattr(node, 'd2_detach_req', None), obj=OBJS[obj_i], until=int(n_now) + 100)
                    if obj_i + 1 < len(OBJS):                           # 串接：下一件
                        obj_i += 1
                        v_cyl = v_objs[obj_i]
                else:
                    ev('obj_detach_without_attach', n=int(n_now))
            if BIN and det_ack_ is not None and k % 10 == 0 and int(n_now) <= det_ack_['until']:
                node.p_ack.publish(String(data=json.dumps({k_: v_ for k_, v_ in det_ack_.items() if k_ != 'until'})))
            if getattr(node, 'door_reset', False):                   # 回程關門：門扇綁定／解綁旗標重設
                node.door_reset, node.d2_bind, node.d2_unbind = False, False, False
                bound, released, bind_pending, closing2 = False, False, False, True
                ev('door_reset_for_close', n=int(n_now), door_deg=round(math.degrees(phi_now), 3))
            if bound and k % 10 == 0:                                 # 綁定生效回報（回程關門的控制器等它；開門段伺服器不採用）
                node.p_ack.publish(String(data=json.dumps(dict(what='door_bound', n=int(n_now)))))
            if DRAWER and getattr(node, 'dr_reset', False):         # 關抽屜：第二次連接（FixedJoint 同路徑重新定義、啟用）
                node.dr_reset, node.d2_attach_dr, node.d2_unbind_dr = False, False, False
                dr_bound, dr_released = False, False
                ev('drawer_reset_for_close', n=int(n_now))
            if DRAWER and getattr(node, 'd2_attach_dr', False) and not dr_bound and not dr_released:   # 抽屜把手理想連接：FixedJoint 夾爪↔抽屜剛體（當下相對位姿；同門把）
                Tg_ = T_of(*[np.asarray(x)[0] for x in v_grip.get_world_poses()])
                Td_ = T_of(*[np.asarray(x)[0] for x in v_dr.get_world_poses()])
                Rd0 = np.linalg.inv(Td_) @ Tg_
                jd_ = UsdPhysics.FixedJoint.Define(stage, '/World/drawer_ideal_attach')
                jd_.CreateBody0Rel().SetTargets([Sdf.Path(str(grip_b[0]))])
                jd_.CreateBody1Rel().SetTargets([Sdf.Path('/World/drawer_unit/drawer')])
                jd_.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                jd_.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                qd_ = W0.R_to_wxyz(Rd0[:3, :3])
                jd_.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rd0[:3, 3]]))
                jd_.CreateLocalRot1Attr().Set(Gf.Quatf(float(qd_[0]), float(qd_[1]), float(qd_[2]), float(qd_[3])))
                jd_.CreateJointEnabledAttr().Set(True)
                jd_.CreateBreakForceAttr().Set(float('inf')); jd_.CreateBreakTorqueAttr().Set(float('inf'))
                dr_bound = True
                ev('drawer_attach_fixed_joint', rel_pos=[round(float(v), 5) for v in Rd0[:3, 3]], opening_truth=round(float((Td_[:3, 3] - dr_p0) @ dr_pull), 5), note='理想抓持假設（FixedJoint）：夾住證據成立後')
            if DRAWER and getattr(node, 'd2_unbind_dr', False) and dr_bound and not dr_released:   # 拉開完成：停用夾爪↔抽屜連接
                stage.GetPrimAtPath('/World/drawer_ideal_attach').GetAttribute('physics:jointEnabled').Set(False)
                dr_bound, dr_released = False, True
                Td_ = T_of(*[np.asarray(x)[0] for x in v_dr.get_world_poses()])
                ev('drawer_unbind_fixed_joint', n=int(n_now), q_open_ctrl=getattr(node, 'dr_q_open', None), opening_truth=round(float((Td_[:3, 3] - dr_p0) @ dr_pull), 5))
            if DRAWER and dr_released:
                jp_ = stage.GetPrimAtPath('/World/drawer_ideal_attach')
                v_ = jp_.GetAttribute('physics:jointEnabled').Get() if (jp_ and jp_.IsValid()) else None
                if v_ is not False and latch is None:
                    latch = dict(t=round(t, 3), why='drawer_unbind_not_confirmed', value=str(v_), kind='模擬工程中止（抽屜連接解除未確認；保留現場資料、不再步進物理）')
                    ev('latched_sim_engineering_abort', **latch)
                    node.p_state.publish(String(data='latched:' + latch['why']))
                    break
                if k % 10 == 0:
                    node.p_ack.publish(String(data=json.dumps(dict(what='drawer_unbound', n=int(n_now), req=getattr(node, 'd2_unbind_dr_req', None)))))
            if getattr(node, 'd2_unbind', False) and bound and not released:   # 解除綁定：停用 FixedJoint（門回到被動鉸鏈）
                stage.GetPrimAtPath(JOINT).GetAttribute('physics:jointEnabled').Set(False)
                bound, released, bind_pending = False, True, False
                ev('unbind_fixed_joint', door_deg=round(math.degrees(phi_now), 4), n=int(n_now))
            unb_ok = False
            if released:                                               # 回報前確認：關節與屬性有效、讀回明確為 False；解綁後若讀回啟用或關節失效 ⇒ 工程中止
                jp_ = stage.GetPrimAtPath(JOINT)
                at_ = jp_.GetAttribute('physics:jointEnabled') if (jp_ and jp_.IsValid()) else None
                v_ = at_.Get() if (at_ is not None and at_.IsValid()) else None
                unb_ok = (v_ is False) and not bound
                if v_ is not False and latch is None:
                    latch = dict(t=round(t, 3), why='unbind_not_confirmed', value=str(v_), bound=bool(bound), kind='模擬工程中止（解綁後關節未確認停用；保留現場資料、不再步進物理）')
                    ev('latched_sim_engineering_abort', **latch)
                    node.p_state.publish(String(data='latched:' + latch['why']))
                    break
            if unb_ok and k % 10 == 0:                                 # 解綁生效回報（重送，直到收尾）
                node.p_ack.publish(String(data=json.dumps(dict(what='unbound', n=int(n_now), req=getattr(node, 'd2_unbind_req', None)))))
            for i, j in zip(idx, range(6)):
                tgt[i] = float(s[j])
            tgt[f1] = f_hold if (bound and not released and f_hold is not None) else fcmd   # 綁定中保持鎖存開度（名目段、停止尾段的閉爪命令都不覆寫）；解綁後照命令（張開）
            if STEP_PROF: _sp_mark('ctrl_wait')
            apply(tgt)
            base_cmd(ue[:3], float(q[2]))
            if STEP_PROF: _sp_mark('apply')
            # ---- 發布
            ck = Clock()
            ck.clock = node.stamp()
            node.p_clock.publish(ck)
            if DRAWER and k % 10 == 0:                                # 抽屜開度真值（只做驗收記錄）
                try:
                    dr_log.append([float(t), float((np.asarray(v_dr.get_world_poses()[0], float)[0] - dr_p0) @ dr_pull), float(dr_bound)])
                except Exception:                                     # noqa: BLE001
                    pass
            if BIN and v_objs and k % 10 == 0:                       # 各物體真值（只做驗收記錄：入盒判定、重播）
                try:
                    row_ = [float(t), *[float(x) for x in np.asarray(v_base.get_world_poses()[0], float)[0]]]
                    for v_ in v_objs:
                        pp_, qq_ = v_.get_world_poses()
                        row_ += [float(x) for x in np.asarray(pp_, float)[0]] + [float(x) for x in np.asarray(qq_, float)[0]]
                    obj_log.append(row_)
                except Exception:                                     # noqa: BLE001
                    pass
            if v_cyl is not None and k % 10 == 0:                     # 圓柱真值（只做驗收記錄：離桌高度、相對夾爪滑移）
                try:
                    pc_t = np.asarray(v_cyl.get_world_poses()[0], float)[0]
                    pg_t = np.asarray(v_grip.get_world_poses()[0], float)[0] if v_grip is not None else np.full(3, np.nan)
                    cyl_log.append([float(t), *[float(x) for x in pc_t], *[float(x) for x in pg_t], float(ffor['uflite_finger1']), float(ffor['uflite_finger2'])])
                except Exception:                                     # noqa: BLE001
                    pass
            if k % pub_every == 0:
                m = Odometry()
                m.header.stamp = node.stamp()
                m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
                m.pose.pose.position.x, m.pose.pose.position.y = float(q[0]), float(q[1])
                m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(q[2] / 2), math.sin(q[2] / 2)
                m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = float(ue[0]), float(ue[1]), float(ue[2])
                node.p_odom.publish(m)
                js = JointState()
                js.header.stamp = node.stamp()
                js.name = ARM + ['finger_joint1', 'finger_joint2']
                js.position = [float(x) for x in q[3:]] + [float(jp[f1]), float(jp[f2])]
                node.p_js.publish(js)
                s_end = s + DT * sum(p_[3:] for p_ in pipe)
                node.p_truth.publish(Float64MultiArray(data=[t, *[float(x) for x in q], *[float(x) for x in s], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(pipe)]]))
                node.p_state.publish(String(data='latched:' + latch['why'] if latch is not None else ('static' if static else 'run')))
            if STEP_PROF: _sp_mark('publish')
            if k % scan_every == 0:
                r, whom = scan(q)
                m = LaserScan()
                m.header.stamp = node.stamp()
                m.header.frame_id = 'lidar_link'
                m.angle_min, m.angle_max, m.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
                m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
                m.ranges = [float(x) if math.isfinite(x) else float('inf') for x in r]
                node.p_scan.publish(m)
                scan_min_log = float(whom[0])
                scans.append(np.r_[t, q[:3], r])
            if STEP_PROF: _sp_mark('scan')
            # ---- 相機：持續取像（送偵測程序，不等結果）
            if DRAWER and getattr(node, 'dr_pose_vis', None) is not None and not getattr(node, '_dr_pose_set', False):
                DRS_.set_pose(*[float(v) for v in node.dr_pose_vis]); node._dr_pose_set = True
                ev('drawer_pose_vision_applied', pose=[round(float(v), 4) for v in node.dr_pose_vis], pose_true=dr_pose_true)
            f = grab() if (k % RENDER_EVERY == 0) else None              # 只在連續繪圖段的最後一步讀相機（D2_RENDER_EVERY／D2_RENDER_BURST；相機 5 Hz）
            if f is not None and f['t_obs'] >= t_next_cam - 1e-9:
                if det_busy:
                    cam_stat['rejects']['detector_busy'] = cam_stat['rejects'].get('detector_busy', 0) + 1
                else:
                    t_next_cam = f['t_obs'] + 1.0 / a.cam_hz
                    seq += 1
                    cam_stat['used'] += 1
                    stem = f'f{seq:04d}'
                    det_pend[seq] = dict(f=dict((k_, v_) for k_, v_ in f.items() if k_ not in ('dep', 'rgb')), stem=stem, Th=truth_handle(), t_sent=t)
                    _DET.send(dict(seq=seq, stem=stem, dep=f['dep'], rgb=f['rgb'], K=Kr, T_cam=f['T_cam'], t_obs=f['t_obs'], q=f['q'], fk_vs_render_m=f['fk_vs_render_m'], table=None if cyl_info is None else cyl_info['table'],
                                   dbox=(DRS_.interior_box(float(node.dr_q_open)) if (DRAWER and dr_released and getattr(node, 'dr_q_open', None) is not None and not cyl_attached) else None),
                                   hspec=os.path.join(WS, reg['drawer']['asset']) if DRAWER else None,
                                   leafdet=bool(DOOR_CLOSE and released and not bound and float(np.hypot(f['q'][0], f['q'][1])) < 3.5),
                                   hprior=([float(v) for v in reg['drawer']['pose'][:2]] + [math.radians(float(reg['drawer']['pose'][2]))] if (DRAWER and HANDLE_VISION and not dr_bound and not dr_released
                                           and float(np.hypot(f['q'][0] - reg['drawer']['pose'][0], f['q'][1] - reg['drawer']['pose'][1])) < 3.0) else None)))
                    det_busy = True
            if det_busy and _DET.poll(0):
                _, rr = _DET.recv()
                det_busy = False
                pd = det_pend.pop(rr['seq'])
                f, stem, seq_r = pd['f'], pd['stem'], rr['seq']
                T_est, info, ms = rr['T_est'], dict(rr['info'], t_processed=round(t, 4), age_s=round(t - f['t_obs'], 4)), rr['ms']
                if rr['err']:
                    info['exception'] = rr['err']
                if not rr['saved']:
                    cam_stat['rejects']['frame_save_fail'] = cam_stat['rejects'].get('frame_save_fail', 0) + 1
                cam_stat['detect_ms'].append(ms)
                T_bh = None if T_est is None else np.linalg.inv(T3(f['q'][:3])) @ np.asarray(T_est, float)
                if T_est is None:
                    key = str(info.get('why') or info.get('cls') or 'unknown')
                    cam_stat['fail'][key] = cam_stat['fail'].get(key, 0) + 1
                else:
                    cam_stat['valid'] += 1
                Th = pd['Th']
                frames.append(dict(seq=seq_r, stem=stem, t_sent=pd['t_sent'], t_obs=f['t_obs'], t_processed=t, age_s=round(t - f['t_obs'], 4), cam_T_fk=f['T_cam'].tolist(), q=[float(v) for v in f['q']], arm_setpoint=[float(v) for v in f['s']],
                                   odom_base_at_capture=[float(v) for v in f['q'][:3]], map_odom_latest_at_capture=f['map_odom'], fk_vs_render_m=f['fk_vs_render_m'], detect_ms=round(ms, 1), valid=T_est is not None, info=info,
                                   T_base_handle=None if T_bh is None else T_bh.tolist(), rendering_frame=[int(v) if v is not None else None for v in f['rf']] if isinstance(f['rf'], tuple) else f['rf'],
                                   eval_only=dict(err_pos_mm=None if T_est is None else round(float(np.linalg.norm(np.asarray(T_est)[:3, 3] - Th[:3, 3])) * 1e3, 2), range_m=round(float(np.linalg.norm(f['T_cam'][:3, 3] - Th[:3, 3])), 3))))
                node.p_obs.publish(Float64MultiArray(data=[float(seq_r), float(f['t_obs']), 1.0 if T_est is not None else 0.0, *[float(v) for v in f['q'][:3]], *([float(v) for v in T_bh.reshape(-1)] if T_bh is not None else [float('nan')] * 16), float(t)]))
                if rr.get('leaf') is not None:                         # 門扇開度觀測（rad）
                    l_ = rr['leaf']
                    node.p_leaf.publish(Float64MultiArray(data=[float(seq_r), float(f['t_obs']), 1.0 if l_.get('phi') is not None else 0.0, *[float(v) for v in f['q'][:3]], float('nan') if l_.get('phi') is None else float(l_['phi'])]))
                    frames[-1]['leaf'] = dict(phi_deg=None if l_.get('phi') is None else round(math.degrees(float(l_['phi'])), 3), door_true_deg=round(math.degrees(float(phi_now)), 3), info=l_.get('info'))
                if rr.get('hdl') is not None:                          # 把手觀測：抽屜原點（世界＝里程）與偏航
                    h_ = rr['hdl']
                    node.p_hdl.publish(Float64MultiArray(data=[float(seq_r), float(f['t_obs']), 1.0 if h_.get('origin') is not None else 0.0, *[float(v) for v in f['q'][:3]],
                                                               *([float(h_['origin'][0]), float(h_['origin'][1]), float(h_['yaw'])] if h_.get('origin') is not None else [float('nan')] * 3)]))
                    frames[-1]['hdl'] = dict(origin=None if h_.get('origin') is None else [round(float(v), 4) for v in h_['origin']], yaw=h_.get('yaw'), info=h_.get('info'))
                if rr.get('cyl') is not None:                          # 圓柱觀測：底盤座標中心（擷取時刻底盤）
                    pc_ = rr['cyl']['p']
                    prel_ = None if pc_ is None else (np.linalg.inv(T3(f['q'][:3])) @ np.r_[pc_, 1.0])[:3]
                    yrel_ = [] if not DRAWER else [float('nan') if rr['cyl'].get('yaw') is None else float(rr['cyl']['yaw']) - float(f['q'][2])]   # 抽屜筆：第 11 欄＝筆軸偏航（底盤座標）
                    node.p_cyl.publish(Float64MultiArray(data=[float(seq_r), float(f['t_obs']), 1.0 if pc_ is not None else 0.0, *[float(v) for v in f['q'][:3]], *([float(v) for v in prel_] if prel_ is not None else [float('nan')] * 3), float(t), *yrel_]))
                    frames[-1]['cyl'] = dict(p=None if pc_ is None else [round(float(v), 4) for v in pc_], yaw=rr['cyl'].get('yaw'), info=rr['cyl']['info'])
                if seq_r <= 3 or seq_r % 25 == 0:
                    print(f"[fc] 影格 {seq_r} t_obs={f['t_obs']:.2f} valid={T_est is not None} {'' if T_est is not None else info.get('why') or info.get('cls')} 距離 {frames[-1]['eval_only']['range_m']} m 偵測 {ms:.0f} ms", flush=True)
            if STEP_PROF: _sp_mark('camera')
            tgt_w = w_all0 + pace_off + (t + DT) / a.rtf                       # RTF 上限配速：早到就睡；晚到不補步（重設基準，實得 RTF ≤ 目標）——物理照牆鐘推進、不等任何計算
            now_w = time.perf_counter()
            if now_w < tgt_w:
                time.sleep(tgt_w - now_w); pace['slept_s'] += tgt_w - now_w
            else:
                if now_w - tgt_w > 0.05:
                    pace['behind_steps'] += 1; pace['behind_max_s'] = max(pace['behind_max_s'], now_w - tgt_w)
                if NO_CATCHUP:                                         # 不連續補步（m8house_d2_unified_2：補步時每步 16 ms，控制側預算被壓縮 ⇒ not_ready）
                    pace_off += now_w - tgt_w; pace['reanchored_s'] = pace.get('reanchored_s', 0.0) + (now_w - tgt_w)
            if STEP_PROF: _sp_mark('sleep')
            st0 = float(world.current_time)
            world.step(render=((k + 1) % RENDER_EVERY == 0 or (k + 1) % RENDER_EVERY >= RENDER_EVERY - RENDER_BURST + 1 if RENDER_BURST > 1 else (k + 1) % RENDER_EVERY == 0) if RENDER_EVERY > 1 else True)   # 物理每步；繪圖在下一次讀相機前連續 RENDER_BURST 步（第七十七趟：每 10 步單繪 ⇒ 深度與彩色幀錯位、門把中心觀測不到 2758 次）
            if STEP_PROF: _sp_mark('step')
            if abs((float(world.current_time) - st0) - DT) > 1e-9:
                clock_bad += 1
            sw = time.perf_counter() - w_step0
            step_wall.append(sw)
            rows.append([t, *q, *s, *ue, *u_in, *ua_in, 0.0 if cn is None else cn, dq_deg, scan_min_log, sw, float(latch is not None), float(post), float(bound), float(jp[f1]), float(jp[f2]), float(phi_now)])
            if (node.done or node.d2_end) and t_done is None:
                t_done = t
                ev('supervisor_done' if node.done else 'd2_world_end')
            if t_done is not None and t - t_done >= 1.0:
                break
            if time.perf_counter() - w_all0 > WALL_LIMIT_S:
                ev('wall_limit')
                break
    except Exception as e_top:                                         # noqa: BLE001  例外也保留已取得的部分結果
        import traceback
        if type(e_top).__name__ == 'ExternalShutdownException' and latch is None:   # 外部 TERM（熱守護／跑法收尾）打斷 rclpy：記為外部終止（模擬工程中止類），不是程式例外
            latch = dict(t=round(node.t, 3), why='external_term', kind='外部終止（熱守護或跑法送 TERM；物理不再步進）')
            print('[fc] 外部終止（ExternalShutdownException）', flush=True)
        else:
            exc = traceback.format_exc()
            print('[fc] 例外：' + exc, flush=True)
    wall = time.perf_counter() - w_all0
    cols = ['t'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['ux', 'uy', 'uw'] + [f'ua{i}' for i in range(6)] + ['uin_x', 'uin_y', 'uin_w'] + [f'uain{i}' for i in range(6)] + ['contact_n', 'door_deg', 'scan_min', 'step_wall_s', 'latched', 'd2_post', 'bound', 'finger1', 'finger2', 'phi']
    R = np.array(rows, float)
    np.savez_compressed(os.path.join(a.out, 'log.npz'), L=R, cols=np.array(cols))
    np.savez_compressed(os.path.join(a.out, 'scans.npz'), S=np.array(scans, float), note=np.array('每列：t, 底盤 x y yaw, 360 道距離（inf＝無回波）；角度＝yaw＋SCAN_MIN＋i·SCAN_INC'))
    json.dump(clean(dict(K=Kr, cam=CAM, T_mount=T_mount, frames=frames)), open(os.path.join(a.out, 'frames_meta.json'), 'w'), ensure_ascii=False, indent=1)
    truth['T_handle_end'] = truth_handle().tolist()
    truth['door_q_end_rad'] = door_q()
    json.dump(clean(truth), open(os.path.join(a.out, 'truth.json'), 'w'), ensure_ascii=False, indent=1)
    dms = cam_stat.pop('detect_ms')
    res = dict(mode='static' if static else 'run', exception=exc, partial=exc is not None, start_scan=dict(nearest_m=None if not np.isfinite(who0[0]) else round(float(who0[0]), 4), nearest_prim=who0[1], n_echo=int(np.isfinite(np.asarray(r0, float)).sum()), n_beams=SCAN_N), n_scans=len(scans), readback=readback, sim_s=round(float(R[-1, 0]), 2) if len(R) else 0.0, wall_s=round(wall, 1), rtf=round(float(R[-1, 0]) / wall, 3) if len(R) and wall > 0 else None,
               latch=latch, events=events, contact_max=contact_max, door_deg_max=round(float(R[:, cols.index('door_deg')].max()), 4) if len(R) else None, clock_bad=clock_bad,
               camera=dict(cam_stat, detect_ms_p50=round(float(np.median(dms)), 1) if dms else None, detect_ms_max=round(float(np.max(dms)), 1) if dms else None),
               cmds_received=dict(base=node.n_cmd, arm=node.n_acmd), base_travel_m=round(float(np.abs(np.diff(R[:, 1:3], axis=0)).sum()), 3) if len(R) > 1 else 0.0,
               sources=dict(odom='物理位姿直接當里程（理想，無滑移）', scan='PhysX 射線 360 道，略過機器人自身形狀，無雜訊', pose_for_camera='渲染時刻的關節／底盤讀值 FK', delay='8 步在途管線（軟體模型）'),
               step_wall_ms_p50=round(float(np.median(step_wall)) * 1e3, 1) if step_wall else None, supervisor_done=t_done is not None,
               pacing=dict(pace, rtf_target=a.rtf, note='物理依牆鐘固定 RTF 推進、不等計算；behind＝迴圈本身（算圖、射線、發布）比目標慢的步數與最大落後'), detector=dict(process='獨立程序（模擬器前 fork）', pending_at_end=len(det_pend)))
    if STEP_PROF:
        res['step_prof'] = {k_: dict(total_s=round(v_[0], 2), per_step_ms=round(1e3 * v_[0] / max(v_[1], 1), 3), n=v_[1]) for k_, v_ in _SP_ACC.items()}
        res['render_every'] = RENDER_EVERY
    res['walkers'] = dict(names=[w_['name'] for w_ in walk_list], log=walk_log, note='每 10 步位置（模擬時間）') if walk_list else None
    res['drawer'] = None if dr_info is None else dict(dr_info, bound_end=dr_bound, released=dr_released, log_cols=['t', 'opening_truth', 'bound'], log=dr_log, note='開度真值只做驗收記錄；控制用開度＝控制器 TCP 位移估計')
    res['marker'] = None if not DRAWER else dict(attached=cyl_attached, log_cols=['t', 'x', 'y', 'z', 'grip_x', 'grip_y', 'grip_z', 'f1_n', 'f2_n'], log=cyl_log, note='筆真值只做驗收記錄，不進控制')
    res['cyl'] = None if cyl_info is None else dict(cyl_info, attached=cyl_attached, attach_note='夾住證據成立後 FixedJoint 理想連接（Howard 11/18）', log_cols=['t', 'x', 'y', 'z', 'grip_x', 'grip_y', 'grip_z', 'f1_n', 'f2_n'], log=cyl_log, note='真值只做驗收記錄，不進控制')
    if bin_info is not None:                                          # 收納盒驗收（真值）：結束時各物體在 base_link 座標、是否在盒內範圍（xy 在內壁內、z 在盒高內）
        try:
            Tb_ = T_of(*[np.asarray(x)[0] for x in v_base.get_world_poses()])
            inner_ = RBIN_.HS[:2] - float(RBIN_.SPEC['wall'])
            ends_ = []
            for o_, v_ in zip(OBJS, v_objs):
                pl_ = (np.linalg.inv(Tb_) @ np.r_[np.asarray(v_.get_world_poses()[0], float)[0], 1.0])[:3]
                cz_ = RBIN_.C_LOC[2] - 0.05
                ins_ = bool(np.all(np.abs(pl_[:2] - RBIN_.C_LOC[:2]) <= inner_) and (cz_ - RBIN_.HS[2]) <= pl_[2] <= (cz_ + RBIN_.HS[2] + 0.05))
                ends_.append(dict(obj=o_, p_base_link=[round(float(x), 4) for x in pl_], in_bin=ins_))
            bin_info['end'] = ends_
        except Exception as e_:                                       # noqa: BLE001
            bin_info['end_err'] = repr(e_)[:200]
        res['bin'] = dict(bin_info, objs=OBJS, log_cols=['t', 'base_x', 'base_y', 'base_z'] + [f'{o_.split("/")[-1]}_{c_}' for o_ in OBJS for c_ in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')], log=obj_log, note='真值只做驗收記錄；放入＝解除理想連接後物理落下')
    res['d2'] = dict(stat=d2_stat, bound=bound, released=released, n0=node.d2_n0, ctl=node.d2_ctl_log, note='交權後每步命令＝/d2/out；late＝本步配速時限內沒收到、保持上一筆')
    # 整機停穩驗收（物理紀錄；rc 與 supervisor_done 都不能代替）：最後一筆非零輸入命令之後，在途管線排清（8 步），
    # 底盤每步位移 <1e-6 m 且偏航 <1e-6 rad、手臂各關節速率 <1e-3 rad/s，連續 0.2 s。
    sv = dict(rule='最後非零輸入命令後：管線排清＋底盤每步位移 <1e-6 m／偏航 <1e-6 rad＋手臂各關節速率 <1e-3 rad/s，連續 0.2 s', valid=False)
    try:
        if latch is not None or exc is not None:
            sv['why'] = '工程中止或例外：物理未繼續步進，沒有停穩可驗'
        elif len(R) > 30:
            nz = np.where(np.abs(R[:, cols.index('uin_x'):cols.index('uin_x') + 9]).max(axis=1) > 1e-9)[0]
            k0 = int(nz[-1]) + 1 if len(nz) else 0
            db = np.abs(np.diff(R[:, 1:4], axis=0)).max(axis=1)
            da = np.abs(np.diff(R[:, 4:10], axis=0)).max(axis=1) / DT
            still = (db < 1e-6) & (da < 1e-3)
            pipe_clear = np.abs(R[1:, cols.index('ux'):cols.index('ux') + 9]).max(axis=1) < 1e-12
            okk = still & pipe_clear
            run_, k_ok = 0, None
            for k_ in range(k0, len(okk)):
                run_ = run_ + 1 if okk[k_] else 0
                if run_ >= 20:
                    k_ok = k_ - 19
                    break
            kb = next((k_ for k_ in range(k0, len(db)) if db[k_:k_ + 20].max() < 1e-6), None)
            sv.update(valid=k_ok is not None, t_last_nonzero_input=round(float(R[max(k0 - 1, 0), 0]), 3), pipe_drained_s=round(D_STEPS * DT, 2), base_stop_time_s=None if kb is None else round((kb - k0) * DT, 2),
                      whole_robot_settle_time_s=None if k_ok is None else round((k_ok - k0) * DT, 2), arm_rate_at_last_input=round(float(da[max(k0 - 1, 0)]), 4), base_speed_at_last_input=round(float(np.linalg.norm(R[k0, 1:3] - R[max(k0 - 1, 0), 1:3]) / DT), 4),
                      residual_arm_rate_end=round(float(da[-10:].max()), 6), record_after_s=round(float(R[-1, 0] - R[max(k0 - 1, 0), 0]), 2))
    except Exception as e_:                                           # noqa: BLE001
        sv['error'] = repr(e_)
    res['stop_validation_strict_extra'] = sv                          # 我另訂的較嚴條件（每步位移），只作補充
    res['stop_validation'] = stop_still(R, cols, latch is not None or exc is not None)
    sv = res['stop_validation']
    print(f'[fc] 整機停穩驗收 {sv}', flush=True)
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[fc] 結束 sim {res['sim_s']} s wall {res['wall_s']} s 影格 取用 {cam_stat['used']} 有效 {cam_stat['valid']} 失敗 {cam_stat['fail']} 鎖存 {latch}", flush=True)
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:                                                 # noqa: BLE001
        pass
    return 71 if exc is not None else (0 if latch is None else 70)


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    except Exception:                                                 # 第二十四趟：牆鐘上限路徑整理結果時例外未印出 ⇒ result.json 遺失
        import traceback
        print('[fc] main 例外：' + traceback.format_exc()[-3000:], flush=True)
    finally:
        # sim_app.close() 會直接結束行程（離開碼 0），後面的 sys.exit(rc) 走不到——m8house_far_capture_1 因此在 partial／工程中止下仍回 0。
        # 離開碼改寫進檔案，由跑法讀取；檔案不存在＝異常結束。
        try:
            with open(os.path.join(a.out, 'exit_code.txt'), 'w') as fh:
                fh.write(str(int(rc if rc is not None else 1)))
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as e_:                                       # noqa: BLE001
            print(f'[fc] 離開碼寫檔失敗：{e_!r}', flush=True)
        print(f'[fc] exit_code {rc}（0 正常；70 模擬工程中止／保護鎖存；71 例外 partial）', flush=True)
        print('[fc] 結果與離開碼已落盤 ⇒ 結束偵測子程序後直接結束行程（不呼叫會卡住的 sim_app.close）', flush=True)
        try:
            _DET_P.terminate(); _DET_P.join(3.0)
            if _DET_P.is_alive():
                _DET_P.kill()
        except Exception:                                             # noqa: BLE001
            pass
        os._exit(rc)
    sys.exit(rc)
