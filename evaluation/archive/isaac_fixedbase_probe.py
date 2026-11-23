"""S3 固定底座短測：用**真正的固定關係**取代每步覆寫底盤速度。

與舊做法的界線（另立版本，不是「物理條件不變只移除擾動」）：

    舊：正常物理迴圈每一步呼叫 set_linear_velocity(0) / set_angular_velocity(0)
    新：world → 根連桿的**固定關節**，正常迴圈**不做任何底盤位置／速度覆寫**

**這代表底盤被外部固定支撐**，不代表輪子靠地面摩擦能承受相同的操作負載；
移動底盤的操作要另外驗證。

只固定底盤根部，手臂與夾爪保持動態；**不疊加第二個 world 固定關節**。
不動導航版、不改手臂增益、軌跡速度、力門檻與抽屜被動性。

固定方式（--fix-mode）：
    importer  匯入器的 fix_base=True，再檢查它實際建立的關節；
              若錨點在世界原點，**就地修改該關節的 localPos0/localRot0**
              到停放位姿（維持單一固定關係，不另建一個）
    authored  匯入器 fix_base=False，自行建立 world→根 的固定關節

固定根部可能改變 articulation 索引，所以手腕力的列號**不沿用舊值**：
重新由 metadata 對應，並用**已知外力**核對。
"""
import argparse, csv, json, math, os, sys
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--traj', required=True)
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--fix-mode', default='importer', choices=('importer', 'authored'))
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--hold-s', type=float, default=3.0)
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--probe-force-n', type=float, default=20.0,
                help='核對手腕力列號用的已知外力')
ap.add_argument('--base-jump-max-m', type=float, default=0.001)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

import drawer_asset as DA                                          # noqa: E402
from cpu_temp import read as cpu_temp_read                         # noqa: E402

CASE = yaml.safe_load(open(a.cases))['cases'][a.case]
SPEC = DA.load(a.spec)
ARM = [f'joint{i}' for i in range(1, 7)]
PARK = (float(CASE['parking']['x']), float(CASE['parking']['y']),
        math.radians(float(CASE['parking']['yaw_deg'])))
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
os.makedirs(a.out, exist_ok=True)
ROWS = [r for r in csv.DictReader(open(a.traj))]
CUT = max(i for i, r in enumerate(ROWS) if r['phase'] in ('reach', 'approach', 'engage'))
PLAY = ROWS[:CUT + 1]
Q_HOLD = np.array([float(PLAY[-1][j]) for j in ARM])
print(f'[fb] fix-mode={a.fix_mode}  重放 {len(PLAY)} 點，**不連接抽屜**')

from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True, 'limit_cpu_threads': a.cpu_threads})
from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.prims import SingleArticulation, RigidPrim      # noqa: E402
from isaacsim.core.utils.types import ArticulationAction           # noqa: E402
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom, Gf, Sdf     # noqa: E402
from isaac_common import import_urdf                               # noqa: E402

ROBOT = '/World/omni_bot'


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def describe_joint(stage, path):
    pr = stage.GetPrimAtPath(path)
    j = UsdPhysics.Joint(pr)
    gv = lambda A: (list(A.Get()) if A and A.Get() is not None else None)
    b0 = [str(t) for t in (j.GetBody0Rel().GetTargets() or [])]
    b1 = [str(t) for t in (j.GetBody1Rel().GetTargets() or [])]
    lp0, lp1 = j.GetLocalPos0Attr(), j.GetLocalPos1Attr()
    lr0, lr1 = j.GetLocalRot0Attr(), j.GetLocalRot1Attr()
    qz = lambda A: ([float(A.Get().GetReal())] + [float(v) for v in A.Get().GetImaginary()]
                    if A and A.Get() is not None else None)
    return {'path': path, 'type': pr.GetTypeName(), 'body0': b0, 'body1': b1,
            'localPos0': [float(v) for v in (lp0.Get() or (0, 0, 0))],
            'localPos1': [float(v) for v in (lp1.Get() or (0, 0, 0))],
            'localRot0_wxyz': qz(lr0), 'localRot1_wxyz': qz(lr1),
            'enabled': (bool(j.GetJointEnabledAttr().Get())
                        if j.GetJointEnabledAttr().Get() is not None else None)}


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    DA.build_usd(world.stage, SPEC, POSE)
    import_urdf(a.urdf, ROBOT, fix_base=(a.fix_mode == 'importer'))
    stage = world.stage
    rep = {'schema': 'fixedbase_probe/1', 'fix_mode': a.fix_mode,
           'case': a.case, 'park': list(PARK), 'physics_dt': a.physics_dt,
           'note': ('底盤由外部固定支撐；**不代表輪子靠地面摩擦能承受相同負載**')}

    # ---- 找出匯入器建立的 world→根 固定關節 ----
    # 只認 world→根：body1 是 articulation 根（base_footprint），
    # body0 為空或就是機器人 Xform 本身。第一版只判斷 "World" in body0，
    # 結果把 URDF 裡全部 99 個底盤固定關節都收進來，就地修改因此被跳過。
    fixed, all_fixed = [], 0
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        if not str(p.GetPath()).startswith(ROBOT):
            continue
        if not p.IsA(UsdPhysics.FixedJoint):
            continue
        all_fixed += 1
        d = describe_joint(stage, str(p.GetPath()))
        b1 = d['body1'][0] if d['body1'] else ''
        b0 = d['body0'][0] if d['body0'] else ''
        if b1.endswith('base_footprint') and (b0 == '' or b0 == ROBOT):
            fixed.append(d)
    print(f'[fb] 模型內固定關節共 {all_fixed} 個（多數是底盤本身的結構）')
    print(f'[fb] 匯入器建立的 world→根 固定關節 {len(fixed)} 個')
    for d in fixed:
        print(f'     {d["path"]}  body0={d["body0"]} body1={d["body1"]}')
        print(f'       localPos0={np.round(d["localPos0"],6).tolist()} '
              f'localRot0={np.round(d["localRot0_wxyz"],6).tolist() if d["localRot0_wxyz"] else None}')
    rep['root_fixed_joints_before'] = fixed
    if a.fix_mode == 'importer' and len(fixed) != 1:
        print(f'[fb] **預期恰好 1 個 world→根 固定關節，實得 {len(fixed)}**')

    # ---- 把單一固定關節的錨點移到停放位姿（不另建第二個）----
    ROOTQ = q_yaw(PARK[2])
    if a.fix_mode == 'importer':
        if len(fixed) != 1:
            print(f'[fb] **預期恰好 1 個 world→根 固定關節，實得 {len(fixed)}**')
        # root_joint 的 body0 是 /World/omni_bot 這個 Xform，錨點 localPos0 = 0，
        # 所以把**那個 Xform** 移到停放位姿即可，錨點會跟著走。
        # 這樣維持匯入器建立的單一固定關係，不動 root_joint、也不另建第二個。
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(PARK[0], PARK[1], 0.0))
        xf.AddOrientOp().Set(Gf.Quatf(ROOTQ[0], ROOTQ[1], ROOTQ[2], ROOTQ[3]))
        print(f'[fb] 已把 {ROBOT} 這個 Xform 移到停放位姿 '
              f'({PARK[0]:.4f}, {PARK[1]:.4f}) yaw {math.degrees(PARK[2]):.1f}°；'
              f'root_joint 未更動')

    world.reset()
    robot = SingleArticulation(prim_path=ROBOT, name='b'); robot.initialize()
    names = list(robot.dof_names); idx = {n: i for i, n in enumerate(names)}
    ai = [idx[j] for j in ARM]
    print(f'[fb] articulation DOF {robot.num_dof}: {names}')
    rep['dof_names'] = names

    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    q0 = np.array([float(ROWS[0][j]) for j in ARM])
    for k, j in enumerate(ARM):
        q[idx[j]] = q0[k]; kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
    F_OPEN = float(SPEC['grasp_surface']['finger_joint_open'])
    for j in FJ:
        q[idx[j]] = F_OPEN; kp[idx[j]] = a.finger_kp; kd[idx[j]] = a.finger_kd
    robot.set_joint_positions(q)
    ctl = robot.get_articulation_controller()
    ctl.set_gains(kps=kp, kds=kd)
    ctl.apply_action(ArticulationAction(joint_positions=q))

    p_before, q_before = robot.get_world_pose()
    # 固定底座下**不呼叫 set_world_pose** —— 位姿由 Xform 與 root_joint 決定，
    # 再去覆寫會與固定關節打架，那就等於又回到覆寫的做法。
    p_set, q_set = p_before, q_before
    print(f'[fb] reset 後根位姿 ({float(p_before[0]):.4f}, {float(p_before[1]):.4f}, '
          f'{float(p_before[2]):.4f}) yaw {math.degrees(yaw_of(q_before)):.3f}°')
    print(f'[fb] 設定後       ({float(p_set[0]):.4f}, {float(p_set[1]):.4f}, '
          f'{float(p_set[2]):.4f}) yaw {math.degrees(yaw_of(q_set)):.3f}°')

    # ---- 暖機：**正常迴圈不做任何底盤位置／速度覆寫** ----
    trace = []
    for k in range(60):
        world.step(render=False)
        p, qq = robot.get_world_pose()
        trace.append([float(p[0]), float(p[1]), float(p[2]),
                      math.degrees(yaw_of(qq))])
    p_warm = np.array(trace[-1][:3])
    jump = float(np.linalg.norm(p_warm - np.array([PARK[0], PARK[1], 0.0])))
    dyaw = abs(trace[-1][3] - math.degrees(PARK[2]))
    print(f'[fb] 暖機 60 步後根位姿 ({p_warm[0]:.5f}, {p_warm[1]:.5f}, {p_warm[2]:.5f}) '
          f'yaw {trace[-1][3]:.4f}°  離停放點 {jump*1000:.4f} mm / {dyaw:.4f}°')
    rep['base_pose_after_warmup'] = {'pos': p_warm.tolist(), 'yaw_deg': trace[-1][3],
                                     'offset_from_park_m': jump, 'dyaw_deg': dyaw,
                                     'trace_first5': trace[:5], 'trace_last5': trace[-5:]}
    ok_base = jump <= a.base_jump_max_m and dyaw <= 0.5
    print(f'[fb] 初始跳變檢查：{"通過" if ok_base else "**未通過**"}'
          f'（門檻 {a.base_jump_max_m*1000:.1f} mm / 0.5°）')

    # ---- 手腕力列號：重新由 metadata 對應，並用已知外力核對 ----
    md = robot._articulation_view._metadata
    ji = dict(md.joint_indices)
    row_meta = int(ji['joint6']) + 1 if 'joint6' in ji else None
    F0 = np.array(robot.get_measured_joint_forces())
    print(f'[fb] measured_joint_forces 形狀 {F0.shape}；'
          f'metadata 推得 joint6 列 {row_meta}（舊版為 100，**不沿用**）')
    grip = [str(p.GetPath()) for p in stage.Traverse()
            if p.GetName() == 'uflite_gripper_link'][0]
    gv = RigidPrim(prim_paths_expr=grip, name='gv'); gv.initialize()
    for _ in range(200):
        ctl.apply_action(ArticulationAction(joint_positions=q)); world.step(render=False)
    n0 = np.linalg.norm(np.array(robot.get_measured_joint_forces())[:, :3], axis=1)
    Fv = np.array([[0.0, 0.0, -a.probe_force_n]], dtype=np.float32)
    for _ in range(200):
        gv.apply_forces(Fv, is_global=True)
        ctl.apply_action(ArticulationAction(joint_positions=q)); world.step(render=False)
    n1 = np.linalg.norm(np.array(robot.get_measured_joint_forces())[:, :3], axis=1)
    d = n1 - n0
    cand = int(np.argmin(np.abs(np.abs(d) - a.probe_force_n)))
    print(f'[fb] 已知外力核對：施 {a.probe_force_n:.1f} N 於夾爪')
    print(f'     metadata 列 {row_meta} 的 Δ|F| = {d[row_meta]:.4f} N')
    print(f'     最接近施力值的列 = {cand}（Δ|F| {d[cand]:.4f} N）')
    rep['wrist_row'] = {'metadata_row': row_meta, 'delta_at_metadata_row': float(d[row_meta]),
                        'best_match_row': cand, 'delta_at_best': float(d[cand]),
                        'applied_N': a.probe_force_n, 'n_rows': int(F0.shape[0]),
                        'old_row_do_not_reuse': 100,
                        'agree': bool(abs(abs(d[row_meta]) - a.probe_force_n)
                                      <= 0.05 * a.probe_force_n)}
    print(f'     判定：{"metadata 列通過已知外力核對" if rep["wrist_row"]["agree"] else "**metadata 列未通過**"}')
    for _ in range(100):
        ctl.apply_action(ArticulationAction(joint_positions=q)); world.step(render=False)

    # ---- 重放並逐步記錄保持段（無任何底盤覆寫）----
    for r in PLAY:
        tgt = robot.get_joint_positions()
        for k, j in enumerate(ARM):
            tgt[idx[j]] = float(r[j])
        for j in FJ:
            tgt[idx[j]] = float(r['finger'])
        ctl.apply_action(ArticulationAction(joint_positions=tgt)); world.step(render=False)
    tgt = robot.get_joint_positions()
    for k, j in enumerate(ARM):
        tgt[idx[j]] = Q_HOLD[k]
    act = ArticulationAction(joint_positions=tgt)
    csv_p = os.path.join(a.out, 'hold_steps.csv')
    f = open(csv_p, 'w', newline=''); w = csv.writer(f)
    w.writerow(['step', 't', 'base_x', 'base_y', 'base_z', 'base_yaw_deg']
               + [f'cmd_{j}' for j in ARM] + [f'q_{j}' for j in ARM]
               + [f'qd_{j}' for j in ARM] + [f'meff_{j}' for j in ARM])
    n = int(a.hold_s / a.physics_dt)
    for k in range(n):
        ctl.apply_action(act); world.step(render=False)     # **不覆寫底盤**
        p, qq = robot.get_world_pose()
        qm = np.array(robot.get_joint_positions())
        qv = np.array(robot.get_joint_velocities())
        me = np.array(robot.get_measured_joint_efforts())
        w.writerow([k, f'{float(world.current_time):.5f}',
                    f'{float(p[0]):.8f}', f'{float(p[1]):.8f}', f'{float(p[2]):.8f}',
                    f'{math.degrees(yaw_of(qq)):.7f}']
                   + [f'{Q_HOLD[i]:.9f}' for i in range(6)]
                   + [f'{qm[i]:.9f}' for i in ai] + [f'{qv[i]:.9f}' for i in ai]
                   + [f'{me[i]:.6f}' for i in ai])
    f.close()
    tc, tsrc = cpu_temp_read()
    rep['base_override_calls_in_loop'] = 0
    rep['hold_csv'] = os.path.basename(csv_p)
    rep['cpu_temp_c'] = tc; rep['cpu_temp_source'] = tsrc
    rep['pass_base_jump'] = bool(ok_base)
    json.dump(rep, open(os.path.join(a.out, 'fixedbase_probe.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'\nCPU {tc} °C（{tsrc}）  -> {csv_p}')
    return 0 if ok_base else 1


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
