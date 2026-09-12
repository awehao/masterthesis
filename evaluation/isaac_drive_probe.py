"""定姿保持診斷：連接前的保持誤差是哪裡來的。

**不建立固定連接、不調增益、不補重力、不改速度。** 只把手臂帶到 engage 前的
同一姿態、維持同一組命令，然後把驅動端的東西全部讀回來對帳。

回答四件事：

    A 驅動設定讀回   drive 型別（force / acceleration）、Kp、Kd、maxForce、
                     target position / velocity，以及**單位**（USD 角度驅動的
                     target 與 stiffness 是以「度」為單位，與 rad 差 57.3 倍）
    B 同一物理時刻   已套用命令、實際 q 與 q̇、驅動 effort ——
                     0.0043 rad 是不是靜止誤差？有沒有輸出飽和？
    C 接觸狀態       還沒 engage 不代表還沒接觸；夾爪／手指與抽屜、櫃體的接觸力
    D 量測來源       手腕反作用力**不是** joint2 的驅動力矩，兩者分開列

最後有一段明確標示的**擾動探測**：把單一關節的命令改動已知的微小量，量
Δeffort / Δ誤差，直接得到**實際生效的剛度**。這是量測，不是調參 —— 沒有連接，
沒有負載風險，結束後命令歸位。

用法：
    $ISAAC_PY evaluation/isaac_drive_probe.py --traj <traj.csv> --out <dir>
"""
import argparse, csv, json, math, os, sys
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

ap = argparse.ArgumentParser()
ap.add_argument('--traj', required=True, help='對準版或未對準版的 traj.csv')
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--hold-s', type=float, default=4.0, help='到位後保持多久')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--perturb-rad', type=float, default=0.001,
                help='擾動探測的命令改動量（0 = 不做）')
ap.add_argument('--perturb-joint', default='joint2')
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
# 只重放到 engage 相位結束為止（**不發 engage 事件、不建立連接**）
CUT = max(i for i, r in enumerate(ROWS) if r['phase'] in ('reach', 'approach', 'engage'))
PLAY = ROWS[:CUT + 1]
Q_HOLD = np.array([float(PLAY[-1][j]) for j in ARM])
print(f'[drive] 重放 {len(PLAY)} 點（到 engage 相位結束），**不建立固定連接**')
print(f'[drive] 保持命令 {np.round(Q_HOLD,6).tolist()}')

from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True, 'limit_cpu_threads': a.cpu_threads})

from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.prims import SingleArticulation, RigidPrim      # noqa: E402
from isaacsim.core.utils.types import ArticulationAction           # noqa: E402
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom              # noqa: E402
from isaac_common import import_urdf                               # noqa: E402

ROBOT = '/World/omni_bot'
DRAWER = '/World/drawer_unit/drawer'


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    DA.build_usd(world.stage, SPEC, POSE)
    import_urdf(a.urdf, ROBOT)
    stage = world.stage
    fingers = [str(p.GetPath()) for p in stage.Traverse()
               if p.GetName() in ('uflite_finger1', 'uflite_finger2')]
    grip = [str(p.GetPath()) for p in stage.Traverse()
            if p.GetName() == 'uflite_gripper_link'][0]

    # 接觸 view 必須在 reset 之前建立
    drawer_v = RigidPrim(prim_paths_expr=DRAWER, name='dv', track_contact_forces=True,
                         max_contact_count=128, prepare_contact_sensors=True,
                         contact_filter_prim_paths_expr=[grip] + fingers)
    grip_v = RigidPrim(prim_paths_expr=grip, name='gv', track_contact_forces=True,
                       max_contact_count=128, prepare_contact_sensors=True)
    fing_v = RigidPrim(prim_paths_expr=fingers[0][:-1] + '[12]', name='fv',
                       track_contact_forces=True, max_contact_count=128,
                       prepare_contact_sensors=True)

    world.reset()
    robot = SingleArticulation(prim_path=ROBOT, name='b'); robot.initialize()
    names = list(robot.dof_names); idx = {n: i for i, n in enumerate(names)}
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
    robot.set_world_pose(np.array([PARK[0], PARK[1], 0.0]), np.array(q_yaw(PARK[2])))
    drawer_v.initialize(); grip_v.initialize(); fing_v.initialize()
    for _ in range(20):
        world.step(render=False)

    rep = {'schema': 'drive_probe/1', 'case': a.case, 'traj': os.path.abspath(a.traj),
           'physics_dt': a.physics_dt, 'set_kp': a.kp, 'set_kd': a.kd,
           'q_hold_cmd': Q_HOLD.tolist(), 'attachment': 'none（本診斷不建立固定連接）'}

    # ---------------------------------------------------------- A 驅動設定讀回
    print('\n[A] 驅動設定讀回（USD 屬性 ＋ articulation 控制器）')
    jprims = {}
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        n = p.GetName()
        if n in ARM and (p.HasAPI(UsdPhysics.DriveAPI)
                         or UsdPhysics.RevoluteJoint(p)):
            jprims.setdefault(n, str(p.GetPath()))
    drv = {}
    for j in ARM:
        path = jprims.get(j)
        if path is None:
            drv[j] = {'error': '找不到關節 prim'}; continue
        pr = stage.GetPrimAtPath(path)
        d = {'prim': path}
        for token in ('angular', 'linear'):
            api = UsdPhysics.DriveAPI.Get(pr, token)
            if not api:
                continue
            gv = lambda A: (float(A.Get()) if A and A.Get() is not None else None)
            ty = api.GetTypeAttr()
            d.update({
                'drive_token': token,
                'type': (str(ty.Get()) if ty and ty.Get() is not None else None),
                'stiffness': gv(api.GetStiffnessAttr()),
                'damping': gv(api.GetDampingAttr()),
                'maxForce': gv(api.GetMaxForceAttr()),
                'targetPosition': gv(api.GetTargetPositionAttr()),
                'targetVelocity': gv(api.GetTargetVelocityAttr())})
            break
        rj = UsdPhysics.RevoluteJoint(pr)
        if rj:
            lo, hi = rj.GetLowerLimitAttr(), rj.GetUpperLimitAttr()
            d['limit_deg'] = [float(lo.Get()) if lo.Get() is not None else None,
                              float(hi.Get()) if hi.Get() is not None else None]
        drv[j] = d
        print(f'  {j}: type={d.get("type")} stiffness={d.get("stiffness")} '
              f'damping={d.get("damping")} maxForce={d.get("maxForce")} '
              f'target={d.get("targetPosition")} limit_deg={d.get("limit_deg")}')
    try:
        g_kp, g_kd = ctl.get_gains()
        rep['controller_gains'] = {'kps': [float(v) for v in np.array(g_kp).ravel()],
                                   'kds': [float(v) for v in np.array(g_kd).ravel()]}
        print(f'  控制器 get_gains() kps[joint2]='
              f'{np.array(g_kp).ravel()[idx["joint2"]]:.6g}  kds[joint2]='
              f'{np.array(g_kd).ravel()[idx["joint2"]]:.6g}')
    except Exception as e:
        rep['controller_gains'] = {'error': repr(e)}
        print(f'  get_gains 失敗 {e!r}')
    # 單位線索：USD 角度驅動的 target 與 limit 以「度」為單位
    j2 = drv.get('joint2', {})
    if j2.get('targetPosition') is not None:
        print(f'  單位線索：joint2 的 USD targetPosition = {j2["targetPosition"]:.6g}；'
              f'命令角 {q0[1]:.6g} rad = {math.degrees(q0[1]):.6g} deg')
    rep['drive_readback'] = drv

    # ---------------------------------------------------------- 重放到保持姿態
    log = []
    def sample(tag, cmd):
        qm = np.array(robot.get_joint_positions())
        qd = np.array(robot.get_joint_velocities())
        try:
            eff = np.array(robot.get_measured_joint_efforts())
        except Exception as e:
            eff = np.full(robot.num_dof, np.nan)
        ai = [idx[j] for j in ARM]
        d = {'tag': tag, 't': float(world.current_time),
             'cmd': [float(v) for v in cmd],
             'q': [float(qm[i]) for i in ai],
             'qd': [float(qd[i]) for i in ai],
             'err': [float(cmd[k] - qm[ai[k]]) for k in range(6)],
             'effort': [float(eff[i]) for i in ai]}
        return d

    for r in PLAY:
        cmd = np.array([float(r[j]) for j in ARM])
        tgt = robot.get_joint_positions()
        for k, j in enumerate(ARM):
            tgt[idx[j]] = cmd[k]
        for j in FJ:
            tgt[idx[j]] = float(r['finger'])
        ctl.apply_action(ArticulationAction(joint_positions=tgt))
        world.step(render=False)

    print(f'\n[B] 保持 {a.hold_s:.1f} s（命令固定為 engage 相位最後一點）')
    tgt = robot.get_joint_positions()
    for k, j in enumerate(ARM):
        tgt[idx[j]] = Q_HOLD[k]
    n_hold = int(a.hold_s / a.physics_dt)
    for i in range(n_hold):
        ctl.apply_action(ArticulationAction(joint_positions=tgt))
        world.step(render=False)
        if i % 25 == 0 or i == n_hold - 1:
            log.append(sample('hold', Q_HOLD))
    last = log[-1]
    print(f'  最後一筆（t={last["t"]:.3f}）')
    print(f'    命令   {np.round(last["cmd"],6).tolist()}')
    print(f'    實際 q {np.round(last["q"],6).tolist()}')
    print(f'    誤差   {np.round(last["err"],6).tolist()}')
    print(f'    q̇      {np.round(last["qd"],8).tolist()}')
    print(f'    effort {np.round(last["effort"],4).tolist()}  (N·m)')
    e2 = last['err'][1]; f2 = last['effort'][1]
    print(f'  joint2：誤差 {e2:.6f} rad，量測 effort {f2:.4f} N·m')
    if abs(e2) > 1e-9:
        print(f'    → 由這兩個推得的等效剛度 {f2/e2:.6g} N·m/rad'
              f'（設定值 {a.kp:.6g}；比值 {abs(f2/e2)/a.kp:.6g}）')
    lim = {'joint1': 50., 'joint2': 50., 'joint3': 32., 'joint4': 32.,
           'joint5': 32., 'joint6': 20.}
    sat = [(j, last['effort'][k], lim[j]) for k, j in enumerate(ARM)
           if abs(last['effort'][k]) >= 0.98 * lim[j]]
    print(f'  飽和檢查（對 URDF effort 上限）：'
          f'{"**有關節達 98 % 以上** " + str(sat) if sat else "沒有關節接近上限"}')
    rep['hold'] = log
    rep['effort_limits_urdf'] = lim

    # ---------------------------------------------------------- C 接觸狀態
    print('\n[C] 接觸狀態（尚未 engage，但不代表沒有接觸）')
    ct = {}
    for nm, v in (('drawer', drawer_v), ('gripper', grip_v), ('fingers', fing_v)):
        f = np.array(v.get_net_contact_forces(dt=a.physics_dt))
        ct[nm + '_net'] = f.tolist()
        print(f'  {nm:8s} 淨接觸力 {np.round(f,5).tolist()}')
    try:
        M = np.array(drawer_v.get_contact_force_matrix(dt=a.physics_dt))
        ct['drawer_vs_[grip,f1,f2]'] = M.tolist()
        print(f'  抽屜 對 [夾爪, 指1, 指2] 的接觸力矩陣 {np.round(M,5).tolist()}')
    except Exception as e:
        ct['matrix_error'] = repr(e)
    rep['contact'] = ct

    # ---------------------------------------------------------- D 量測來源
    print('\n[D] 量測來源（手腕反作用力**不是** joint2 的驅動力矩，分開列）')
    F = np.array(robot.get_measured_joint_forces())
    md = robot._articulation_view._metadata
    r6 = int(md.joint_indices['joint6']) + 1
    r2 = int(md.joint_indices['joint2']) + 1
    rep['measured_joint_forces'] = {
        'shape': [int(v) for v in F.shape],
        'row_joint2': r2, 'raw_row_joint2': [float(v) for v in F[r2]],
        'row_joint6': r6, 'raw_row_joint6': [float(v) for v in F[r6]],
        'frame': '各列在該連桿座標系；力 [0:3]、力矩 [3:6]',
        'note': 'get_measured_joint_efforts 才是 DOF 方向上的驅動力矩'}
    print(f'  joint2 反作用力列 {r2}: 力 {np.round(F[r2][:3],4).tolist()} '
          f'力矩 {np.round(F[r2][3:],4).tolist()}')
    print(f'  joint6 反作用力列 {r6}: 力 {np.round(F[r6][:3],4).tolist()} '
          f'力矩 {np.round(F[r6][3:],4).tolist()}')
    print(f'  joint2 的 measured_joint_effort = {last["effort"][1]:.4f} N·m'
          f'（這才是驅動在 DOF 方向的輸出）')

    # ---------------------------------------------------------- 擾動探測
    if a.perturb_rad > 0:
        pj = a.perturb_joint
        k = ARM.index(pj)
        print(f'\n[E] 擾動探測（**量測用，不是調參**）：{pj} 命令 '
              f'{a.perturb_rad:+.6f} rad，之後歸位')
        pert = []
        for delta in (0.0, a.perturb_rad, -a.perturb_rad, 0.0):
            t2 = robot.get_joint_positions()
            for kk, j in enumerate(ARM):
                t2[idx[j]] = Q_HOLD[kk] + (delta if kk == k else 0.0)
            for _ in range(int(1.5 / a.physics_dt)):
                ctl.apply_action(ArticulationAction(joint_positions=t2))
                world.step(render=False)
            s = sample(f'perturb{delta:+.4f}', Q_HOLD + np.eye(6)[k] * delta)
            pert.append(s)
            print(f'   Δcmd {delta:+.6f}: 誤差 {s["err"][k]:+.6f} rad，'
                  f'effort {s["effort"][k]:+.4f} N·m，q̇ {s["qd"][k]:+.2e}')
        de = pert[1]['err'][k] - pert[0]['err'][k]
        df = pert[1]['effort'][k] - pert[0]['effort'][k]
        if abs(de) > 1e-12:
            print(f'   → Δeffort/Δ誤差 = {df/de:.6g} N·m/rad'
                  f'（設定 kp {a.kp:.6g}，比值 {abs(df/de)/a.kp:.6g}；'
                  f'1/57.3 = {1/57.29578:.6g}）')
            rep['perturb_stiffness_Nm_per_rad'] = float(df / de)
        rep['perturb'] = pert

    tc, tsrc = cpu_temp_read()
    rep['cpu_temp_c'] = tc; rep['cpu_temp_source'] = tsrc
    json.dump(rep, open(os.path.join(a.out, 'drive_probe.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'\nCPU {tc} °C（{tsrc}）')
    print(f'-> {os.path.join(a.out, "drive_probe.json")}')
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
