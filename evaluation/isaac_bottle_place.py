#!/usr/bin/env python3
"""瓶子第二桌放置（isaac_bottle_n2.py 的版本化複本，2026-11-01；設計 第七次進度報告素材/規格/瓶子第二桌放置_最小設計.md）：
場景加第二桌（同資產，位移 PLACE），控制器換 PlaceController；持瓶段在 LIFT 開始時向 planner_server 規劃（bottle_place_ros.RosCarryPlanner）。
新增量測：瓶↔第二桌面接觸力向量（支撐判定）、手指實際開度；新增禁止通道：手指／手臂↔第二桌（併入 finger_table／arm_table）、底盤↔第二桌腳、瓶↔第二桌腳（併入 bottle_env）、抬離後瓶↔第一桌。
瓶位姿、底盤位姿、第二桌位置用真值（第一版，不是視覺）。以下為 N2 說明。
瓶子 N2（isaac_bottle_ros.py 的版本化複本，2026-11-01）：導航段改用 planner_server（SmacPlanner2D）的路徑，進度用弧長（S1Planned）；
握手後先規劃一次（bottle_n2_ros.plan_before_start），沒有有效路徑就不起跑（離開碼 72），不退回 DetourPath；任務中路徑被新障礙佔到或代價圖過期 ⇒ 受控停止，不重規劃。
--plan-only：握手＋規劃後結束（冒煙用）。以下為 N1 說明。
瓶子 N1（isaac_bottle_s1.py 的版本化複本，2026-11-01；Codex 條件核可）：感測保護鏈改走真 ROS 話題，其餘同 S1。
本程序發 /clock、/odom、/tf、/joint_states、/scan_raw、/cmd_vel_pre_shield；scan_relay、追蹤器（bottle_tracker_stamped）、Shield、map_server、RSP 是獨立節點（run_bottle_ros.sh）。
控制器仍在本程序與物理同步，不是即時部署驗證；路線仍是 DetourPath，不是規劃器。命令時序與逾時處置見 bottle_ros_chain。--handshake-only：只做話題、TF、發布者握手後結束，不跑任務（冒煙用）。
以下為原版說明。
瓶子 S1 候選：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（核心 gmpc_wb；開 Isaac 視窗）。規格：第七次進度報告素材/規格/瓶子S1_候選規格.md。
控制邏輯全在 bottle_s1_ctrl.S1Controller（繼承 S0bController、S0aController、S0Judge，與 test_bottle_s1.py 負測同一份）；本檔只做 Isaac 轉接。
場景同 S0b 加一個固定方塊障礙。**導航資料來源：PhysX 射線掃描 → 既有 scan_obstacle_tracker → gmpc_wb；底盤命令一律經既有 scan_safety_shield 才寫入（bottle_nav_chain，原始節點程式、同程序同步）。** 地圖只有桌腳，方塊是未知物。障礙真值只留作評估，不進控制器。
底盤位姿與瓶位姿仍取真值（無定位、無視覺）；無 ROS 命令鏈與 DDS 延遲；參數差異見 bottle_nav_chain。不得稱完整感知導航整合。
參考路徑是工程給定的升餘弦繞行，不是規劃器輸出；命令延遲 8 物理步是注入設定；底盤速度為工程設定與啟發式調速。
量測有缺一律給 None／NaN；必要接觸對全程有效，起跑前形狀／有限值核對失敗即拒跑；僅完整任務（抓持五項＋實際進入出口區＋底盤停下）回 0。不自動補跑。
    $ISAAC_PY evaluation/isaac_bottle_s1.py --asset src/my_omnibot_description/config/bottle_s0.yaml --out <dir>
"""
import argparse
import json
import math
import os
import sys
import time

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--asset', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--x-start', type=float, default=-3.6, help='底盤起點 x（瓶軸在 x＝0）')
ap.add_argument('--no-tcp-preview', action='store_true', help='消融：時域 TCP 目標固定為 T_seq[0]，其餘不變')
ap.add_argument('--s2-box', default='', help='S2 三維障礙 cx,cy,cz,sx,sy,sz；模擬真值進控制器，LiDAR 看不到')
ap.add_argument('--wall-box', default='', help='側牆 cx,cy,cz,sx,sy,sz：進場景、掃描、地圖與接觸判定')
ap.add_argument('--s2-links', default='', help='納入約束的連桿，逗號分隔，例如 link3,link4,link5')
ap.add_argument('--s2-active', action='store_true', help='開啟預測距離約束（無此旗標時障礙在場但控制器不知道）')
ap.add_argument('--no-inflight', action='store_true', help='消融：求解初態不沿在途命令前推；實際延遲與安全核對不變')
ap.add_argument('--group', default='G', choices=['G', 'P', 'M'], help='對照組：G gmpc_wb（凍結）、P 停車操作、M MotM 式單步 QP')
ap.add_argument('--m-kp', type=float, default=4.0)
ap.add_argument('--m-mu', type=float, default=0.01)
ap.add_argument('--y-start', type=float, default=0.0, help='幾何擾動：底盤起點 y')
ap.add_argument('--yaw-start-deg', type=float, default=0.0, help='幾何擾動：底盤起始偏航')
ap.add_argument('--bottle-dx', type=float, default=0.0, help='幾何擾動：瓶位置 x（瓶位姿仍是真值進控制器）')
ap.add_argument('--bottle-dy', type=float, default=0.0)
ap.add_argument('--x-end', type=float, default=0.85, help='底盤參考終點 x（出口區內）')
ap.add_argument('--obst-x', type=float, default=-2.0)
ap.add_argument('--path', default='-3.4,-0.6,-0.81', help='升餘弦繞行 x1,x2,A')
ap.add_argument('--exit-c', default='1.0,0.0')
ap.add_argument('--exit-r', type=float, default=0.30)
ap.add_argument('--win-lo', type=float, default=-0.35)
ap.add_argument('--win-hi', type=float, default=0.15)
ap.add_argument('--v-c', type=float, default=0.10, help='接近／抓取參考速度上限 m/s（啟發式調速只減速不加速）')
ap.add_argument('--v-nav', type=float, default=0.35)
ap.add_argument('--v-carry', type=float, default=0.25)
ap.add_argument('--x-unfold', type=float, default=-1.55)
ap.add_argument('--t-unfold', type=float, default=3.4)
ap.add_argument('--t-blend', type=float, default=3.0)
ap.add_argument('--t-insert', type=float, default=1.5)
ap.add_argument('--x-slow', type=float, default=-0.90)
ap.add_argument('--q-tuck', default='0,0,0.035,0,-1.5708,0', help='起始與行進姿態：K2_READY（Howard 指定）；開始後不直接設定關節')
ap.add_argument('--sim-s', type=float, default=90.0)
ap.add_argument('--settle-s', type=float, default=1.0, help='開始前保持（量靜態偏移 d̂ 初值）')
ap.add_argument('--stop-tail-s', type=float, default=2.0, help='停止或完成後續跑（核停止有效性）')
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--headless', default='false')
ap.add_argument('--handshake-only', action='store_true')
ap.add_argument('--plan-only', action='store_true')
ap.add_argument('--goal', default='-0.9,0.0', help='規劃目標＝直線段入口（任務定義）')
ap.add_argument('--x-lane', type=float, default=-0.9)
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation        # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from isaac_common import bind_friction, bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from finger_collision import apply_split                             # noqa: E402
import bottle_branch as BB                                           # noqa: E402
import bottle_ros_chain as RC                                        # noqa: E402
NC, C2, GR, N2C, PCm = RC.import_with_real_ros('bottle_nav_chain', 'bottle_s1_ctrl', 'bottle_s1_groups', 'bottle_n2_ctrl', 'bottle_place_ctrl')   # 不讓 ros_stub 蓋掉真 rclpy
import bottle_n2_path as NP                                          # noqa: E402
import bottle_n2_ros as NR                                           # noqa: E402
import bottle_place_ros as PRS                                       # noqa: E402
import bottle_ref as BRF                                             # noqa: E402
import bottle_s0_ctrl as C                                           # noqa: E402
import bottle_s1_geom as G                                           # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]


def yaw_of(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ray_range(query, origin, direction, max_range, excluded):
    """同 ev9r9a：回傳 (距離, 命中路徑)，未命中為 inf。"""
    best = [float('inf'), None]

    def report(hit):
        coll = str(getattr(hit, 'collision', '') or '')
        body = str(getattr(hit, 'rigidBody', '') or '')
        d = float(getattr(hit, 'distance', 0.0))
        if any(coll == e or (body and e.startswith(body + '/')) for e in excluded):
            return True
        if 0.0 <= d < best[0]:
            best[0], best[1] = d, (coll or body)
        return True

    query.raycast_all(origin, direction, max_range, report)
    return best[0], best[1]


def read_pairs(view, n_filters, dt):
    """回傳 (n_sensor, n_filter, 3) 陣列；讀不到回 None（不當零）。"""
    try:
        M = np.asarray(view.get_contact_force_matrix(dt=dt), float)
        return M.reshape(-1, n_filters, 3)
    except Exception:
        return None


def main():
    S = yaml.safe_load(open(a.asset))
    T, B, PH = S['table'], S['bottle'], S['physics']
    dt = float(PH['dt'])
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[s1] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(bool(PH.get('ext_forces_every_iter')))
    # 桌與瓶（同 isaac_bottle_smoke）
    FixedCuboid(prim_path='/World/table/top', name='table_top', position=np.array([0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]),
                scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.55, 0.42, 0.30]))
    hz = T['top_z'] - T['thick']
    leg_paths = []
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            pth = f'/World/table/leg_{nm}_{"p" if sx > 0 else "m"}'
            FixedCuboid(prim_path=pth, name=f'leg_{nm}_{sx}', position=np.array([lx, ly, hz / 2]), scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.4, 0.3, 0.2]))
            leg_paths.append(pth)
    OB_C, OB_S = np.array([a.obst_x, 0.0, 0.25]), np.array([0.40, 0.40, 0.50])
    FixedCuboid(prim_path='/World/obstacle', name='obstacle', position=OB_C, scale=OB_S, color=np.array([0.7, 0.25, 0.2]))
    WALL = None
    if a.wall_box:
        v_ = [float(x) for x in a.wall_box.split(',')]
        WALL = (np.array(v_[:3]), np.array(v_[3:6]))
        FixedCuboid(prim_path='/World/wall', name='wall', position=WALL[0], scale=WALL[1], color=np.array([0.5, 0.5, 0.55]))
    S2 = None
    if a.s2_box:
        v_ = [float(x) for x in a.s2_box.split(',')]
        S2 = (np.array(v_[:3]), np.array(v_[3:6]))
        FixedCuboid(prim_path='/World/s2_box', name='s2_box', position=S2[0], scale=S2[1], color=np.array([0.2, 0.7, 0.3]))
    T2D = np.array([PCm.PLACE['target'][0] - float(B['center_xy'][0]), PCm.PLACE['lane_y'], 0.0])     # 第二桌＝同資產整體位移
    FixedCuboid(prim_path='/World/table2/top', name='table2_top', position=np.array([0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]) + T2D,
                scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.45, 0.50, 0.62]))
    leg2_paths = []
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            pth = f'/World/table2/leg_{nm}_{"p" if sx > 0 else "m"}'
            FixedCuboid(prim_path=pth, name=f'leg2_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + T2D, scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.35, 0.38, 0.45]))
            leg2_paths.append(pth)
    BX, BY = float(B['center_xy'][0]) + a.bottle_dx, float(B['center_xy'][1]) + a.bottle_dy
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([BX, BY, T['top_z'] + B['height'] / 2]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath('/World/table/top'), stage.GetPrimAtPath('/World/table2/top')] + [stage.GetPrimAtPath(p) for p in leg_paths + leg2_paths], T['material']['static_friction'],
                  T['material']['dynamic_friction'], T['material']['restitution'], mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'],
                  B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle')
    # 機器人（同 ev9r9a）
    base = np.array([a.x_start, a.y_start, math.radians(a.yaw_start_deg)])
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base[0]), float(base[1]), 0.0))
    xf.AddRotateZOp().Set(float(a.yaw_start_deg))
    fc = apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    gr = collision_prims(stage, '/World/ground')
    if bind_frictionless(stage, sph + gr) == 0:
        print('[s1] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    grip = [pr for pr in collision_prims(stage, ROBOT) if 'finger' in str(pr.GetPath()).lower()]
    if bind_friction(stage, grip, B['material']['static_friction'], B['material']['dynamic_friction']) == 0:
        print('[s1] 夾持面摩擦沒綁到 ⇒ 拒跑', flush=True)
        return 21
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[s1] 找不到 articulation root ⇒ 拒跑', flush=True)
        return 15
    fingers = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('uflite_finger1', 'uflite_finger2'))
    links = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in tuple(f'link{i}' for i in range(1, 7)))
    extra = [b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('link_eef', 'uflite_gripper_link')]
    if len(fingers) != 2 or len(links) != 6:
        print(f'[s1] 剛體不符：手指 {fingers} 連桿 {links} ⇒ 拒跑', flush=True)
        return 16
    print(f'[s1] 剛體 {len(bodies)}；手指 {fingers}；附加 {extra}；手指拆塊 {list(fc.get("fingers", {}))}', flush=True)
    table_paths = ['/World/table/top'] + leg_paths + ['/World/table2/top'] + leg2_paths + ['/World/obstacle'] + (['/World/s2_box'] if S2 else []) + (['/World/wall'] if WALL else [])   # 含障礙：禁止接觸對涵蓋 桌＋障礙
    # 接觸視圖（須在 world.reset 前建；用字元類別，不用路徑清單）
    v_finger = RigidPrim(prim_paths_expr=str(fingers[0])[:-1] + '[12]', name='v_finger', track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/bottle'] + table_paths)
    # 連桿路徑是巢狀的（link1/link2/…），字元類別只比對到 link1（bottle_s0a_1 拒跑）⇒ 每根連桿各一個視圖
    v_arm = [RigidPrim(prim_paths_expr=str(b), name=f'v_arm{i}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                       contact_filter_prim_paths_expr=['/World/bottle'] + table_paths) for i, b in enumerate(links)]
    v_extra = [RigidPrim(prim_paths_expr=str(b), name=f'v_extra{i}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                         contact_filter_prim_paths_expr=['/World/bottle'] + table_paths) for i, b in enumerate(extra)]
    b_base = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'base_link']
    if len(b_base) != 1:
        print(f'[s1] base_link 剛體不符 {b_base} ⇒ 拒跑', flush=True)
        return 16
    v_base = RigidPrim(prim_paths_expr=str(b_base[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                       contact_filter_prim_paths_expr=['/World/obstacle'] + leg_paths + leg2_paths + (['/World/wall'] if WALL else []))
    v_bt = RigidPrim(prim_paths_expr='/World/bottle', name='v_bt', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table/top'])
    b_env = (['/World/s2_box'] if S2 else []) + (['/World/wall'] if WALL else []) + leg2_paths   # 瓶↔S2 方塊、側牆、第二桌腳：缺資料不當零
    v_b2 = RigidPrim(prim_paths_expr='/World/bottle', name='v_b2', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table2/top'])   # 瓶↔第二桌面：指定支撐，力向量做支撐判定
    v_bs = RigidPrim(prim_paths_expr='/World/bottle', name='v_bs', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=b_env) if b_env else None
    world.reset()
    if v_bs is not None:
        v_bs.initialize()
    for v in [v_finger, v_bt, v_base, v_b2] + v_arm + v_extra:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[s1] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    T_g = C.grasp_target(BX, BY, float(T['top_z']) + 0.10)
    T_p = T_g.copy()
    T_p[:3, 3] -= 0.06 * T_g[:3, 2]
    br = BB.window_branch(T_p, T_g, BRF.lift_pose(T_g, 0.05), a.win_lo, a.win_hi)
    q_ready, T_ready_rel, _ = BB.ready_in_branch(np.array([a.win_lo + 0.05, 0.0, 0.0]), T_p, br['q_mid'], br['x_mid'])
    q_tuck = np.array([float(v) for v in a.q_tuck.split(',')])
    q0 = robot.get_joint_positions()
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_tuck[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = C.FINGER_OPEN, C.FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd                                     # 主從：只驅動 finger_joint1（同 --finger-leader-only）
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def apply(tgt_full):
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt_full, np.float32)[act], joint_indices=act))

    def base_zero():
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))

    def base_cmd(u, yw):
        """本體速度 → 世界速度（同 ev9r9a 寫法）。"""
        vx, vy, wz = (float(v) for v in u)
        robot.set_linear_velocity(np.array([vx * math.cos(yw) - vy * math.sin(yw), vx * math.sin(yw) + vy * math.cos(yw), 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))

    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([-0.6, -4.2, 2.6]), target=np.array([-1.0, -0.2, 0.4]))
    except Exception as e:
        print(f'[s1] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    readback = dict(solver_type=str(sapi.GetSolverTypeAttr().Get()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()),
                    physics_dt=float(world.get_physics_dt()), bottle_mass=float(bottle.get_mass()), kp=a.kp, kd=a.kd, base=[float(v) for v in base],
                    q_ready=[float(v) for v in q_ready], q_tuck=[float(v) for v in q_tuck], group=a.group, wall_box=a.wall_box, s2_box=a.s2_box, s2_active=a.s2_active, no_tcp_preview=a.no_tcp_preview, no_inflight=a.no_inflight, m=dict(kp=a.m_kp, mu=a.m_mu) if a.group == 'M' else None, perturb=dict(bottle_dx=a.bottle_dx, bottle_dy=a.bottle_dy, x_start=a.x_start, y_start=a.y_start, yaw_start_deg=a.yaw_start_deg), speeds=dict(v_nav=a.v_nav, v_c=a.v_c, v_carry=a.v_carry, x_slow=a.x_slow, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend, t_insert=a.t_insert), branch_margin=float(br['margin']), finger_split=bool(fc.get('fingers')))
    print(f'[s1] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - dt) > 1e-12 or readback['ext_forces_every_iter'] != bool(PH.get('ext_forces_every_iter')):
        print('[s1] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64

    def q9():
        p, qt = robot.get_world_pose()
        jp = robot.get_joint_positions()
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), [float(jp[i]) for i in idx]], jp

    # 靜態保持：量 d̂ 初值
    tgt = np.array(robot.get_joint_positions(), float)
    for _ in range(int(round(a.settle_s / dt))):
        apply(tgt)
        base_zero()
        world.step(render=True)
    q_s, _ = q9()
    d0 = q_s[3:] - q_tuck
    M_ = read_pairs(v_bt, 1, dt)                                              # 力方向約定：瓶靜置第一桌時桌面給瓶的力應向上
    fz0 = float('nan') if M_ is None else float(M_[0, 0, 2])
    if not np.isfinite(fz0) or abs(fz0) < 0.5 * float(B['mass']) * 9.81:
        print(f'[s1] 起始瓶↔第一桌力 {fz0} 無法定方向 ⇒ 拒跑', flush=True)
        return 68
    F_SIGN = 1.0 if fz0 > 0.0 else -1.0
    readback['f_sign'] = dict(fz0=fz0, sign=F_SIGN, mg=float(B['mass']) * 9.81)
    print(f'[s1] 接觸力方向約定：起始瓶↔第一桌 fz {fz0:.3f} N ⇒ 符號 {F_SIGN:+.0f}', flush=True)
    n_filters = 1 + len(table_paths)
    required = ['finger_table', 'arm_bottle', 'arm_table', 'base_env', 'bottle_table1', 'bottle_table2'] + (['bottle_env'] if b_env else []) + [f'extra{i}_{w}' for i in range(len(v_extra)) for w in ('bottle', 'table')]
    # 動作前：所有接觸視圖形狀、感測數、有限值都須有效，否則拒跑（不當成沒有接觸）
    for nm, v, ns, nf in [('finger', v_finger, 2, n_filters), ('bottle_table', v_bt, 1, 1), ('bottle_table2', v_b2, 1, 1), ('base', v_base, 1, 1 + len(leg_paths) + len(leg2_paths) + (1 if WALL else 0))] + ([('bottle_env', v_bs, 1, len(b_env))] if b_env else []) + [(f'arm{i}', v, 1, n_filters) for i, v in enumerate(v_arm)] \
            + [(f'extra{i}', v, 1, n_filters) for i, v in enumerate(v_extra)]:
        M = read_pairs(v, nf, dt)
        if M is None or M.shape != (ns, nf, 3) or not np.isfinite(M).all():
            print(f'[s1] 接觸視圖 {nm} 無效（形狀 {None if M is None else M.shape}，期望 {(ns, nf, 3)}）⇒ 拒跑', flush=True)
            return 68
    # 掃描鏈：障礙只從這裡進控制器；OB_C 真值只留作評估
    lidar_prim = next((pr for pr in stage.Traverse() if str(pr.GetPath()).startswith(ROBOT) and pr.GetName() == 'lidar_link'), None)
    if lidar_prim is None:
        print('[s1] 找不到 lidar_link ⇒ 拒跑', flush=True)
        return 18
    excl = [str(pr.GetPath()) for pr in stage.Traverse() if str(pr.GetPath()).startswith(ROBOT) and pr.HasAPI(UsdPhysics.CollisionAPI)
            and (str(pr.GetPath()).endswith('/base_link/cylinder') or '/lidar_link/' in str(pr.GetPath()))]
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    xfc = UsdGeom.XformCache()

    def scan_fn(t_, q_):
        xfc.Clear()
        o5 = xfc.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
        org = [float(o5[0]), float(o5[1]), float(o5[2])]
        ang = NC.scan_angles() + float(q_[2])
        return np.array([ray_range(query, org, [math.cos(v), math.sin(v), 0.0], NC.RANGE_MAX, excl)[0] for v in ang])

    leg_boxes = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top'] + ([WALL] if WALL else [])   # 地圖：桌腳＋側牆
    import rclpy                                                              # 地圖（leg_boxes 同一份）由 run_bottle_ros.sh 寫檔給 map_server
    rclpy.init()
    chain = RC.RosChain()
    node = RC.make_bridge(chain)
    chain.tx = node
    q_chk, _ = q9()
    r0 = scan_fn(0.0, q_chk)
    hits = [ray_range(query, [float(q_chk[0]), float(q_chk[1]), 0.2652], [math.cos(v + float(q_chk[2])), math.sin(v + float(q_chk[2])), 0.0], 1.0, excl) for v in NC.scan_angles()[::30]]
    print(f"[s1] 掃描核對：有效 {int(np.isfinite(r0).sum())}／{r0.size}、最近 {np.nanmin(np.where(np.isfinite(r0), r0, np.nan)) if np.isfinite(r0).any() else float('nan'):.3f} m；近距命中 {[h for h in hits if h[1]][:3]}；排除 {excl}", flush=True)
    if np.isfinite(r0).any() and float(np.min(r0[np.isfinite(r0)])) < 0.32:
        print('[s1] 掃描打到自身（<0.32 m）⇒ 拒跑', flush=True)
        return 69
    p_ = [float(v) for v in a.path.split(',')]
    if a.group != 'G':
        print('[s1] N2 只接 G', flush=True)
        return 2
    cls_ = PCm.PlaceController
    ctrl = cls_(C.S0_K(), ident, T_g, T_ready_rel, q_ready, float(T['top_z']), float(B['radius']), a.x_start, (a.win_lo, a.win_hi), C2.DetourPath(*p_),
                           tuple(float(v) for v in a.exit_c.split(',')), a.exit_r, chain.obstacles, chain=chain, scan_fn=scan_fn, x_exit_stop=a.x_end, v_c=a.v_c, v_nav=a.v_nav, v_carry=a.v_carry, x_slow=a.x_slow, q_tuck=q_tuck, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend, t_insert=a.t_insert, no_tcp_preview=a.no_tcp_preview, no_inflight=a.no_inflight, body_boxes=[S2] if (S2 and a.s2_active) else None, body_links=tuple(x for x in a.s2_links.split(',') if x), reg=None, required=required, **(dict(kp=a.m_kp) if a.group == 'M' else {}))
    if a.group == 'M':
        ctrl.p1.mu_post = a.m_mu
    ctrl.compact = []                                                         # CBF 證據；反事實重解在實跑後（bottle_s1_cbf_replay.py）
    ctrl.obs.d = np.clip(d0, -ctrl.obs.max, ctrl.obs.max)
    ctrl.cfg.arm_model.bias = ctrl.bias0 + np.asarray(ctrl.cfg.arm_model.alpha, float) * ctrl.obs.d
    print(f'[s1] 靜態偏移 d̂₀ {np.round(ctrl.obs.d, 4).tolist()}；底盤 {np.round(q_s[:3], 4).tolist()}', flush=True)
    chain.ctrl = ctrl

    def hs_state():
        p_, qt_ = robot.get_world_pose()
        jp_ = robot.get_joint_positions()
        return p_, qt_, [float(jp_[i]) for i in idx]

    def hs_hold():
        apply(tgt)
        base_zero()
        world.step(render=True)
    T0, why = RC.handshake(node, chain, hs_hold, hs_state, lambda: scan_fn(0.0, q9()[0]))
    if T0 is None:
        print(f'[s1] 握手未過 ⇒ 拒跑：{json.dumps(why, ensure_ascii=False, default=str)[:1500]}', flush=True)
        return 70
    print(f"[s1] 握手通過 T0 {T0:.2f}；發布者 {chain.graph['publishers']}；TF 對 {len(chain.graph['tf_pairs'])}", flush=True)
    if a.handshake_only:
        node.destroy_node()
        rclpy.shutdown()
        return 0
    pr = NR.PlanRos(node)
    goal = tuple(float(v) for v in a.goal.split(','))
    pp, plan_info = NR.plan_before_start(node, chain, pr, hs_hold, hs_state, lambda: scan_fn(0.0, q9()[0]), goal, a.x_lane, T0 + RC.PHYS)
    json.dump(dict(plan=plan_info), open(os.path.join(a.out, 'plan.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    if pp is None:
        print(f'[s1] 無有效路徑 ⇒ 不起跑：{json.dumps(plan_info, ensure_ascii=False, default=float)[:800]}', flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return 72
    print(f"[s1] 規劃 L {pp.L:.3f} m、σ0 {pp.sig0:.3f}、原始 {plan_info['n_raw']} 點、代價峰 {plan_info['cost_max']}、牆鐘 {plan_info['wall_ms']:.0f} ms；y 最低 {pp.P[:, 1].min():.3f}", flush=True)
    if a.plan_only:
        node.destroy_node()
        rclpy.shutdown()
        return 0
    T0 = plan_info['t_ready'] + RC.PHYS
    mon = NP.PathMonitor(pp)
    mon.on_grid(pr.grid, pp.sig0)                                             # 先用規劃時那張代價圖
    t_now = [0.0]
    cpl = PRS.RosCarryPlanner(pr, lambda: T0 + t_now[0], lambda: ctrl.csig)
    ctrl.carry_planner, ctrl.exit_c, ctrl.bottle_mass = cpl, np.array([99.0, 99.0]), float(readback['bottle_mass'])
    ctrl.path, ctrl.monitor = pp, mon
    ctrl.s_ref = ctrl.sig_robot = pp.sig0
    pr.monitor, pr.sig_fn = mon, (lambda: ctrl.sig_robot)
    chain.start(T0)
    BP_S = G.bottle_points()
    rows, step_ms = [], []                                                    # step_ms：物理＋渲染，分開計時
    t_end_tail = None
    n = int(round((a.sim_s + a.stop_tail_s + 1.5) / dt))                      # 總期限後仍走停止尾段
    penetration = 0.0
    for k in range(n):
        t = k * dt
        if t >= a.sim_s and not ctrl.mission_done and ctrl.judge.stop is None:
            ctrl._trigger_stop('sim_limit')
        q, jp = q9()
        p_, qt_ = robot.get_world_pose()
        node.publish_state(T0 + t, p_, qt_, q[3:])
        chain.set_now(T0 + t)
        node.drain()
        t_now[0] = t
        cpl.tick()
        if ctrl.pph is None:                                                  # 承接後代價圖只餵持瓶段監看器
            mon.check_age(T0 + t, ctrl.sig_robot)
        elif ctrl.cmon is not None:
            ctrl.cmon.check_age(T0 + t, ctrl.csig)
        # 瓶位姿（幾何中心 → 底心座標：z 軸＝瓶軸）
        pb, qb = bottle.get_world_pose()
        Rb = quat_R(qb)
        Tb = np.eye(4)
        Tb[:3, :3] = Rb
        Tb[:3, 3] = np.asarray(pb, float) - Rb[:, 2] * float(B['height']) / 2
        ok_b = bool(np.isfinite(Tb).all())
        # 接觸
        Mf = read_pairs(v_finger, n_filters, dt)
        Ma_l = [read_pairs(v, n_filters, dt) for v in v_arm]
        Ma = None if any(x is None for x in Ma_l) else np.concatenate(Ma_l, axis=0)   # 任一連桿讀不到 ⇒ 整對無效（不當零）
        Me = [read_pairs(v, n_filters, dt) for v in v_extra]
        Mt = read_pairs(v_bt, 1, dt)
        ff = None if Mf is None or not np.isfinite(Mf[:, 0]).all() else tuple(float(x) for x in np.linalg.norm(Mf[:, 0], axis=1))
        ft = None if Mt is None or not np.isfinite(Mt).all() else float(abs(Mt[0, 0, 2]))
        forb = {}
        if b_env:
            Ms = read_pairs(v_bs, len(b_env), dt)
            forb['bottle_env'] = float('nan') if Ms is None or not np.isfinite(Ms).all() else float(np.linalg.norm(Ms, axis=2).max())
        Mb = read_pairs(v_base, 1 + len(leg_paths) + len(leg2_paths) + (1 if WALL else 0), dt)
        M2 = read_pairs(v_b2, 1, dt)
        fp = None if M2 is None or not np.isfinite(M2).all() else np.asarray(M2[0, 0], float)
        if fp is not None:
            fp = F_SIGN * fp                                                  # 固定方向約定（起始靜置時瓶↔第一桌量得的符號），反向力不會被翻成支撐
        forb['bottle_table1'] = float('nan') if ft is None else (float(ft) if ctrl.judge.t_lift_ok is not None else 0.0)   # 抬離成立後瓶不得再碰第一桌
        forb['base_env'] = float('nan') if Mb is None else float(np.linalg.norm(Mb, axis=2).max())
        forb['finger_table'] = float('nan') if Mf is None else float(np.linalg.norm(Mf[:, 1:], axis=2).max())
        forb['arm_bottle'] = float('nan') if Ma is None else float(np.linalg.norm(Ma[:, 0], axis=1).max())
        forb['arm_table'] = float('nan') if Ma is None else float(np.linalg.norm(Ma[:, 1:], axis=2).max())
        for i, M in enumerate(Me):
            forb[f'extra{i}_bottle'] = float('nan') if M is None else float(np.linalg.norm(M[:, 0], axis=1).max())
            forb[f'extra{i}_table'] = float('nan') if M is None else float(np.linalg.norm(M[:, 1:], axis=2).max())
        # 穿透估計（每控制週期）：非手指連桿點雲對瓶軸與桌面
        if k % C.NPH == 0:
            W = G.link_world(q, ctrl.f_cmd)
            Pl = np.vstack([P for n_, P in W.items() if n_ not in G.C.FINGERS])
            Pb = (Tb[:3, :3].T @ (Pl - Tb[:3, 3]).T).T
            dmin = float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, float(B['height']), float(B['radius'])).min())
            tb = G.table_boxes()
            tb = tb + [(n_, c_ + T2D, sz_) for n_, c_, sz_ in G.table_boxes()]
            dmin = min(dmin, min(float(G.box_dist(Pl, c, sz).min()) for _, c, sz in tb), float(G.box_dist(Pl, OB_C, OB_S).min()), float(G.box_dist(Pl, S2[0], S2[1]).min()) if S2 else 9.9, float(G.box_dist(Pl, WALL[0], WALL[1]).min()) if WALL else 9.9)
            if ok_b:                                                          # 瓶身對第二桌面與桌腳的穿透（瓶底支撐的正常接觸遠小於 1 mm 判準）
                Pb_w2 = (Tb[:3, :3] @ BP_S.T).T + Tb[:3, 3]
                dmin = min(dmin, min(float(G.box_dist(Pb_w2, c_ + T2D, sz_).min()) for _, c_, sz_ in G.table_boxes()))
            if (S2 or WALL) and ok_b:                                         # 實測瓶位姿對 S2 方塊與側牆
                Pb_w = (Tb[:3, :3] @ BP_S.T).T + Tb[:3, 3]
                for box_ in [b_ for b_ in (S2, WALL) if b_ is not None]:
                    dmin = min(dmin, float(G.box_dist(Pb_w, box_[0], box_[1]).min()))
            elif S2 or WALL:
                dmin = float('nan')                                           # 瓶位姿無效：不當零，由資料失效判定處理
            penetration = max(0.0, -dmin) if np.isfinite(dmin) else float('nan')
        out = ctrl.step(t, q, f_fingers=ff, t_contact=(t if ff is not None else None), T_bottle=Tb if ok_b else None, t_bottle=t if ok_b else None,
                        f_table=ft, forbidden=forb, penetration=penetration, f_place=fp, q_fingers=(float(jp[f1]), float(jp[f2])))
        tgt = np.array(jp, float)
        for j, i in enumerate(idx):
            tgt[i] = float(out['arm_target'][j])
        tgt[f1] = out['finger_target']
        apply(tgt)
        base_cmd(out['base_twist'], float(q[2]))
        t_w = time.perf_counter()
        world.step(render=True)
        step_ms.append((time.perf_counter() - t_w) * 1e3)
        rows.append([t, *q, *out['arm_target'], *out['base_twist'], ctrl.v_b, ctrl.s_ref, float(jp[f1]), float(jp[f2]), out['finger_target'], *(ff if ff else (float('nan'),) * 2),
                     float('nan') if ft is None else ft, *Tb[:3, 3], *[float(v) for v in qb], BR_PH.index(out['phase']), float(out['stopped']),
                     *[forb[kk] for kk in sorted(forb)]])
        if (out['stopped'] or out['done']) and t_end_tail is None:
            t_end_tail = t + (a.stop_tail_s if out['done'] else max(a.stop_tail_s, 1.2))  # 停止尾段須涵蓋 1 s 停止有效性
        if t_end_tail is not None and t >= t_end_tail:
            break
    rep = ctrl.report()
    rep['timing']['world_step'] = dict(p50=round(float(np.percentile(step_ms, 50)), 2), p95=round(float(np.percentile(step_ms, 95)), 2), max=round(float(np.max(step_ms)), 2), n=len(step_ms))
    rep.update(plan=plan_info, readback=readback, asset=S, forb_keys=sorted(forb), d0=[float(v) for v in d0], n_steps=len(rows), obstacle=dict(c=OB_C.tolist(), sz=OB_S.tolist(), r_cbf=0.2 * math.sqrt(2.0), margin=0.47, truth_for_eval_only=True), exit=dict(c=a.exit_c, r=a.exit_r), path=p_, note='核心 gmpc_wb；PhysX 射線掃描經真 ROS 話題：/scan_raw→scan_relay→/scan→追蹤器與 Shield 獨立節點，障礙走 /gmpc/obstacles_stamped，底盤命令走 /cmd_vel_pre_shield→Shield→/cmd_vel；控制器與物理同步（非即時部署驗證）；底盤與瓶位姿真值；地圖只有桌腳與側牆；參考路徑：導航段起跑前 SmacPlanner2D 規劃到抓取直線入口；抓取直線為任務定義；持瓶段 LIFT 開始時再規劃到放置直線入口（平滑、承接、局部降速、參考加速度核對）；放置直線與停車放置為任務定義；不重規劃。第二桌位置與瓶位姿為真值；命令總延遲 8 物理步＝實測往返＋注入補足（分列於 nav_chain.ros）；停止處置為腳本自有', ros_chain=True)
    cols = ['t', 'base_x', 'base_y', 'base_yaw'] + [f'q{i}' for i in range(1, 7)] + [f's{i}' for i in range(1, 7)] + ['base_cmd_vx', 'base_cmd_vy', 'base_cmd_wz', 'v_b', 's_ref', 'finger1_q', 'finger2_q', 'finger_cmd', 'f_finger1_bottle', 'f_finger2_bottle',
            'f_bottle_table', 'bottle_base_x', 'bottle_base_y', 'bottle_base_z', 'bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz', 'phase_index', 'stopped'] + [f'forb_{kk}' for kk in sorted(forb)]
    np.savez_compressed(os.path.join(a.out, 's1_steps.npz'), rows=np.array(rows, float), cols=np.array(cols), phases=np.array(BR_PH))
    rep['cols'] = cols
    rep['reached_end'] = bool(ctrl.mission_done or ctrl.judge.stop is not None)
    if not RC.write_report(rep, a.out):                                       # bottle_n1_1 寫到一半崩潰；失敗另存診斷檔、不算完成
        node.destroy_node()
        rclpy.shutdown()
        return 71
    try:                                                                      # 主紀錄存好後才存 CBF 證據：這裡失敗不影響主紀錄
        C2.save_compact(ctrl, os.path.join(a.out, 's1_cbf.npz'))
    except Exception as e:
        print(f'[s1] CBF 紀錄存檔失敗：{e!r}', flush=True)
    print(f"[s1] 結果 success={rep['success']} 判定 {rep['result']} 第一阻點 {rep['stop']} 停止有效 {rep['stop_effect']} 出口 {rep['exit_reached']}@{rep['t_exit_in']} 任務 {rep['mission_done']} 瓶淨空 {rep['clear_bottle']} 滑移 {rep['slip_max_mm_deg']} 求解 {rep['solve_ms']} fallback {rep['n_fallback']} 求解器 {rep['qp_counts']} GI 未採用 {rep['gi_fail']} >50ms {rep['n_over_50ms']} 掃描鏈 { {k: rep['nav_chain'][k] for k in ('n_scan', 'n_dyn_max', 'n_static_max', 'shield')} } 分項 {rep['timing']}", flush=True)
    print(f"[s1] 事件 {rep['events']}", flush=True)
    print(f"[s1] ROS { {k: v for k, v in rep['nav_chain']['ros'].items() if k not in ('graph', 'note', 'shield_age_top_s')} }", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return C2.exit_code(rep)


BR_PH = ['CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY']

if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    except BaseException:
        import traceback
        print('[s1] 例外：' + traceback.format_exc().replace('\n', ' | '), flush=True)
    finally:
        # sim_app.close() 會直接結束程序、sys.exit 到不了（bottle_s0a_1 拒跑卻回 0）⇒ 先寫離開碼檔，由跑法讀回
        with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
            fh.write(str(rc))
        sim_app.close()
    sys.exit(rc)
