#!/usr/bin/env python3
"""瓶子 S1 候選：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（核心 gmpc_wb；開 Isaac 視窗）。規格：第七次進度報告素材/規格/瓶子S1_候選規格.md。
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

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--asset', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--x-start', type=float, default=-3.6, help='底盤起點 x（瓶軸在 x＝0）')
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
import bottle_nav_chain as NC                                        # noqa: E402
import bottle_ref as BRF                                             # noqa: E402
import bottle_s0_ctrl as C                                           # noqa: E402
import bottle_s1_ctrl as C2                                          # noqa: E402
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
    BX, BY = float(B['center_xy'][0]), float(B['center_xy'][1])
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([BX, BY, T['top_z'] + B['height'] / 2]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath('/World/table/top')] + [stage.GetPrimAtPath(p) for p in leg_paths], T['material']['static_friction'],
                  T['material']['dynamic_friction'], T['material']['restitution'], mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'],
                  B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle')
    # 機器人（同 ev9r9a）
    base = np.array([a.x_start, 0.0, 0.0])
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base[0]), float(base[1]), 0.0))
    xf.AddRotateZOp().Set(0.0)
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
    table_paths = ['/World/table/top'] + leg_paths + ['/World/obstacle']          # 含障礙：禁止接觸對涵蓋 桌＋障礙
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
                       contact_filter_prim_paths_expr=['/World/obstacle'] + leg_paths)
    v_bt = RigidPrim(prim_paths_expr='/World/bottle', name='v_bt', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table/top'])
    world.reset()
    for v in [v_finger, v_bt, v_base] + v_arm + v_extra:
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
                    q_ready=[float(v) for v in q_ready], q_tuck=[float(v) for v in q_tuck], speeds=dict(v_nav=a.v_nav, v_c=a.v_c, v_carry=a.v_carry, x_slow=a.x_slow, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend), branch_margin=float(br['margin']), finger_split=bool(fc.get('fingers')))
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
    n_filters = 1 + len(table_paths)
    required = ['finger_table', 'arm_bottle', 'arm_table', 'base_env'] + [f'extra{i}_{w}' for i in range(len(v_extra)) for w in ('bottle', 'table')]
    # 動作前：所有接觸視圖形狀、感測數、有限值都須有效，否則拒跑（不當成沒有接觸）
    for nm, v, ns, nf in [('finger', v_finger, 2, n_filters), ('bottle_table', v_bt, 1, 1), ('base', v_base, 1, 1 + len(leg_paths))] + [(f'arm{i}', v, 1, n_filters) for i, v in enumerate(v_arm)] \
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

    leg_boxes = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    chain = NC.NavChain(map_boxes=leg_boxes)
    q_chk, _ = q9()
    r0 = scan_fn(0.0, q_chk)
    hits = [ray_range(query, [float(q_chk[0]), float(q_chk[1]), 0.2652], [math.cos(v + float(q_chk[2])), math.sin(v + float(q_chk[2])), 0.0], 1.0, excl) for v in NC.scan_angles()[::30]]
    print(f"[s1] 掃描核對：有效 {int(np.isfinite(r0).sum())}／{r0.size}、最近 {np.nanmin(np.where(np.isfinite(r0), r0, np.nan)) if np.isfinite(r0).any() else float('nan'):.3f} m；近距命中 {[h for h in hits if h[1]][:3]}；排除 {excl}", flush=True)
    if np.isfinite(r0).any() and float(np.min(r0[np.isfinite(r0)])) < 0.32:
        print('[s1] 掃描打到自身（<0.32 m）⇒ 拒跑', flush=True)
        return 69
    p_ = [float(v) for v in a.path.split(',')]
    ctrl = C2.S1Controller(C.S0_K(), ident, T_g, T_ready_rel, q_ready, float(T['top_z']), float(B['radius']), a.x_start, (a.win_lo, a.win_hi), C2.DetourPath(*p_),
                           tuple(float(v) for v in a.exit_c.split(',')), a.exit_r, chain.obstacles, chain=chain, scan_fn=scan_fn, x_exit_stop=a.x_end, v_c=a.v_c, v_nav=a.v_nav, v_carry=a.v_carry, x_slow=a.x_slow, q_tuck=q_tuck, x_unfold=a.x_unfold, t_unfold=a.t_unfold, t_blend=a.t_blend, reg=None, required=required)
    ctrl.compact = []                                                         # CBF 證據；反事實重解在實跑後（bottle_s1_cbf_replay.py）
    ctrl.obs.d = np.clip(d0, -ctrl.obs.max, ctrl.obs.max)
    ctrl.cfg.arm_model.bias = ctrl.bias0 + np.asarray(ctrl.cfg.arm_model.alpha, float) * ctrl.obs.d
    print(f'[s1] 靜態偏移 d̂₀ {np.round(ctrl.obs.d, 4).tolist()}；底盤 {np.round(q_s[:3], 4).tolist()}', flush=True)
    rows = []
    t_end_tail = None
    n = int(round((a.sim_s + a.stop_tail_s + 1.5) / dt))                      # 總期限後仍走停止尾段
    penetration = 0.0
    for k in range(n):
        t = k * dt
        if t >= a.sim_s and not ctrl.mission_done and ctrl.judge.stop is None:
            ctrl._trigger_stop('sim_limit')
        q, jp = q9()
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
        Mb = read_pairs(v_base, 1 + len(leg_paths), dt)
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
            dmin = min(dmin, min(float(G.box_dist(Pl, c, sz).min()) for _, c, sz in tb), float(G.box_dist(Pl, OB_C, OB_S).min()))
            penetration = max(0.0, -dmin)
        out = ctrl.step(t, q, f_fingers=ff, t_contact=(t if ff is not None else None), T_bottle=Tb if ok_b else None, t_bottle=t if ok_b else None,
                        f_table=ft, forbidden=forb, penetration=penetration)
        tgt = np.array(jp, float)
        for j, i in enumerate(idx):
            tgt[i] = float(out['arm_target'][j])
        tgt[f1] = out['finger_target']
        apply(tgt)
        base_cmd(out['base_twist'], float(q[2]))
        world.step(render=True)
        rows.append([t, *q, *out['arm_target'], *out['base_twist'], ctrl.v_b, ctrl.s_ref, float(jp[f1]), float(jp[f2]), out['finger_target'], *(ff if ff else (float('nan'),) * 2),
                     float('nan') if ft is None else ft, *Tb[:3, 3], *[float(v) for v in qb], BR_PH.index(out['phase']), float(out['stopped']),
                     *[forb[kk] for kk in sorted(forb)]])
        if (out['stopped'] or out['done']) and t_end_tail is None:
            t_end_tail = t + (a.stop_tail_s if out['done'] else max(a.stop_tail_s, 1.2))  # 停止尾段須涵蓋 1 s 停止有效性
        if t_end_tail is not None and t >= t_end_tail:
            break
    rep = ctrl.report()
    rep.update(readback=readback, asset=S, forb_keys=sorted(forb), d0=[float(v) for v in d0], n_steps=len(rows), obstacle=dict(c=OB_C.tolist(), sz=OB_S.tolist(), r_cbf=0.2 * math.sqrt(2.0), margin=0.47, truth_for_eval_only=True), exit=dict(c=a.exit_c, r=a.exit_r), path=p_, note='核心 gmpc_wb；PhysX 射線掃描＋既有追蹤器與 Shield（同程序）；底盤與瓶位姿真值；地圖只有桌腳；無 ROS 命令鏈；參考路徑工程給定；命令延遲 8 物理步為注入設定；停止處置為腳本自有')
    cols = ['t', 'base_x', 'base_y', 'base_yaw'] + [f'q{i}' for i in range(1, 7)] + [f's{i}' for i in range(1, 7)] + ['base_cmd_vx', 'base_cmd_vy', 'base_cmd_wz', 'v_b', 's_ref', 'finger1_q', 'finger2_q', 'finger_cmd', 'f_finger1_bottle', 'f_finger2_bottle',
            'f_bottle_table', 'bottle_base_x', 'bottle_base_y', 'bottle_base_z', 'bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz', 'phase_index', 'stopped'] + [f'forb_{kk}' for kk in sorted(forb)]
    np.savez_compressed(os.path.join(a.out, 's1_steps.npz'), rows=np.array(rows, float), cols=np.array(cols), phases=np.array(BR_PH))
    rep['cols'] = cols
    rep['reached_end'] = bool(ctrl.mission_done or ctrl.judge.stop is not None)
    json.dump(rep, open(os.path.join(a.out, 's1_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    try:                                                                      # 主紀錄存好後才存 CBF 證據：這裡失敗不影響主紀錄
        C2.save_compact(ctrl, os.path.join(a.out, 's1_cbf.npz'))
    except Exception as e:
        print(f'[s1] CBF 紀錄存檔失敗：{e!r}', flush=True)
    print(f"[s1] 結果 success={rep['success']} 判定 {rep['result']} 第一阻點 {rep['stop']} 停止有效 {rep['stop_effect']} 出口 {rep['exit_reached']}@{rep['t_exit_in']} 任務 {rep['mission_done']} 瓶淨空 {rep['clear_bottle']} 滑移 {rep['slip_max_mm_deg']} 求解 {rep['solve_ms']} fallback {rep['n_fallback']} 求解器 {rep['qp_counts']} GI 未採用 {rep['gi_fail']} >50ms {rep['n_over_50ms']} 掃描鏈 { {k: rep['nav_chain'][k] for k in ('n_scan', 'n_dyn_max', 'n_static_max', 'shield')} }", flush=True)
    print(f"[s1] 事件 {rep['events']}", flush=True)
    return C2.exit_code(rep)


BR_PH = ['CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY']

if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        # sim_app.close() 會直接結束程序、sys.exit 到不了（bottle_s0a_1 拒跑卻回 0）⇒ 先寫離開碼檔，由跑法讀回
        with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
            fh.write(str(rc))
        sim_app.close()
    sys.exit(rc)
