#!/usr/bin/env python3
"""瓶子 S0a：停車抓取（底盤零速度保持；手臂 W-GMPC 局部未來參考核心；開 Isaac 視窗）。規格：第七次進度報告素材/規格/瓶子S0_物理抓持設計.md。
控制邏輯全在 bottle_s0_ctrl.S0aController／bottle_s0_logic（與 test_bottle_s0a.py 負測同一份）；本檔只做 Isaac 轉接：場景、量測、套用命令、紀錄。
機器人設定沿用實跑 ev9r9a：omni_bot_manip.urdf、手臂 kp 600／kd 60、手指 kp 1e4／kd 1e3、finger_leader_only、手指拆塊碰撞 split、支撐球零摩擦、物理 dt 0.01、TGS、
  ext_forces_every_iter；夾持面 μs／μd 同瓶子資產。與實跑的差異（須揭露）：**無 ROS 命令鏈、無幾何監看節點；命令延遲 8 物理步是注入設定**；底盤每步下零速度（同執行端 set velocity 寫法，非 PARK_HOLD 伺服）。
量測有缺一律給 None／NaN（不當零）。禁止接觸以接觸過濾對讀力；穿透以連桿點雲對瓶／桌幾何距離估計（每控制週期）。
不自動補跑；不延伸 S0b。
    $ISAAC_PY evaluation/isaac_bottle_s0a.py --asset src/my_omnibot_description/config/bottle_s0.yaml --out <dir>
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
ap.add_argument('--base-x', type=float, default=-0.10, help='底盤 x（瓶軸在 x＝0；可達窗內）')
ap.add_argument('--sim-s', type=float, default=30.0)
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
        print(f'[s0a] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
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
    BX, BY = float(B['center_xy'][0]), float(B['center_xy'][1])
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([BX, BY, T['top_z'] + B['height'] / 2]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath('/World/table/top')] + [stage.GetPrimAtPath(p) for p in leg_paths], T['material']['static_friction'],
                  T['material']['dynamic_friction'], T['material']['restitution'], mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'],
                  B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle')
    # 機器人（同 ev9r9a）
    base = np.array([a.base_x, 0.0, 0.0])
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base[0]), float(base[1]), 0.0))
    xf.AddRotateZOp().Set(0.0)
    fc = apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    gr = collision_prims(stage, '/World/ground')
    if bind_frictionless(stage, sph + gr) == 0:
        print('[s0a] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    grip = [pr for pr in collision_prims(stage, ROBOT) if 'finger' in str(pr.GetPath()).lower()]
    if bind_friction(stage, grip, B['material']['static_friction'], B['material']['dynamic_friction']) == 0:
        print('[s0a] 夾持面摩擦沒綁到 ⇒ 拒跑', flush=True)
        return 21
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[s0a] 找不到 articulation root ⇒ 拒跑', flush=True)
        return 15
    fingers = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('uflite_finger1', 'uflite_finger2'))
    links = sorted(b for b in bodies if str(b).rstrip('/').split('/')[-1] in tuple(f'link{i}' for i in range(1, 7)))
    extra = [b for b in bodies if str(b).rstrip('/').split('/')[-1] in ('link_eef', 'uflite_gripper_link')]
    if len(fingers) != 2 or len(links) != 6:
        print(f'[s0a] 剛體不符：手指 {fingers} 連桿 {links} ⇒ 拒跑', flush=True)
        return 16
    print(f'[s0a] 剛體 {len(bodies)}；手指 {fingers}；附加 {extra}；手指拆塊 {list(fc.get("fingers", {}))}', flush=True)
    table_paths = ['/World/table/top'] + leg_paths
    # 接觸視圖（須在 world.reset 前建；用字元類別，不用路徑清單）
    v_finger = RigidPrim(prim_paths_expr=str(fingers[0])[:-1] + '[12]', name='v_finger', track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/bottle'] + table_paths)
    v_arm = RigidPrim(prim_paths_expr=str(links[0])[:-1] + '[1-6]', name='v_arm', track_contact_forces=True, max_contact_count=128,
                      prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/bottle'] + table_paths)
    v_extra = [RigidPrim(prim_paths_expr=str(b), name=f'v_extra{i}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                         contact_filter_prim_paths_expr=['/World/bottle'] + table_paths) for i, b in enumerate(extra)]
    v_bt = RigidPrim(prim_paths_expr='/World/bottle', name='v_bt', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                     contact_filter_prim_paths_expr=['/World/table/top'])
    world.reset()
    for v in [v_finger, v_arm, v_bt] + v_extra:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[s0a] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    T_g = C.grasp_target(BX, BY, float(T['top_z']) + 0.10)
    T_p = T_g.copy()
    T_p[:3, 3] -= 0.06 * T_g[:3, 2]
    q_ready, T_ready_rel = C.ready_from_pre(base, T_p)
    q0 = robot.get_joint_positions()
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_ready[j])
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

    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([1.3, -0.9, 1.3]), target=np.array([0.0, 0.3, 0.7]))
    except Exception as e:
        print(f'[s0a] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    readback = dict(solver_type=str(sapi.GetSolverTypeAttr().Get()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()),
                    physics_dt=float(world.get_physics_dt()), bottle_mass=float(bottle.get_mass()), kp=a.kp, kd=a.kd, base=[float(v) for v in base],
                    q_ready=[float(v) for v in q_ready], finger_split=bool(fc.get('fingers')))
    print(f'[s0a] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - dt) > 1e-12 or readback['ext_forces_every_iter'] != bool(PH.get('ext_forces_every_iter')):
        print('[s0a] 物理設定讀回不符 ⇒ 拒跑', flush=True)
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
    d0 = q_s[3:] - np.asarray(q_ready, float)
    ctrl = C.S0aController(C.S0_K(), ident, T_g, T_ready_rel, q_ready, float(T['top_z']), float(B['radius']), reg=None)
    ctrl.obs.d = np.clip(d0, -ctrl.obs.max, ctrl.obs.max)
    ctrl.cfg.arm_model.bias = ctrl.bias0 + np.asarray(ctrl.cfg.arm_model.alpha, float) * ctrl.obs.d
    print(f'[s0a] 靜態偏移 d̂₀ {np.round(ctrl.obs.d, 4).tolist()}；底盤 {np.round(q_s[:3], 4).tolist()}', flush=True)
    n_filters = 1 + len(table_paths)
    rows = []
    t_end_tail = None
    n = int(round(a.sim_s / dt))
    penetration = 0.0
    for k in range(n):
        t = k * dt
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
        Ma = read_pairs(v_arm, n_filters, dt)
        Me = [read_pairs(v, n_filters, dt) for v in v_extra]
        Mt = read_pairs(v_bt, 1, dt)
        ff = None if Mf is None or not np.isfinite(Mf[:, 0]).all() else tuple(float(x) for x in np.linalg.norm(Mf[:, 0], axis=1))
        ft = None if Mt is None or not np.isfinite(Mt).all() else float(abs(Mt[0, 0, 2]))
        forb = {}
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
            dmin = min(dmin, min(float(G.box_dist(Pl, c, sz).min()) for _, c, sz in tb))
            penetration = max(0.0, -dmin)
        out = ctrl.step(t, q, f_fingers=ff, t_contact=(t if ff is not None else None), T_bottle=Tb if ok_b else None, t_bottle=t if ok_b else None,
                        f_table=ft, forbidden=forb, penetration=penetration)
        tgt = np.array(jp, float)
        for j, i in enumerate(idx):
            tgt[i] = float(out['arm_target'][j])
        tgt[f1] = out['finger_target']
        apply(tgt)
        base_zero()
        world.step(render=True)
        rows.append([t, *q, *out['arm_target'], float(jp[f1]), float(jp[f2]), out['finger_target'], *(ff if ff else (float('nan'),) * 2),
                     float('nan') if ft is None else ft, *Tb[:3, 3], float(Rb[2, 2]), BR_PH.index(out['phase']), float(out['stopped']),
                     *[forb[kk] for kk in sorted(forb)]])
        if (out['stopped'] or out['done']) and t_end_tail is None:
            t_end_tail = t + a.stop_tail_s
        if t_end_tail is not None and t >= t_end_tail:
            break
    rep = ctrl.report()
    rep.update(readback=readback, asset=S, forb_keys=sorted(forb), d0=[float(v) for v in d0], n_steps=len(rows), note='命令延遲 8 物理步為注入設定；無 ROS 命令鏈；停止處置為腳本自有')
    np.savez_compressed(os.path.join(a.out, 's0a_steps.npz'), rows=np.array(rows, float))
    json.dump(rep, open(os.path.join(a.out, 's0a_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"[s0a] 結果 success={rep['success']} 判定 {rep['result']} 第一阻點 {rep['stop']} 停止有效 {rep['stop_effect']} 滑移 {rep['slip_max_mm_deg']} 求解 {rep['solve_ms']}", flush=True)
    print(f"[s0a] 事件 {rep['events']}", flush=True)
    return 0


BR_PH = ['CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY']

if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        sim_app.close()
    sys.exit(rc)
