"""成果展示影片：用**模擬器內的相機感測器**回放已錄下的狀態。

界線（寫在程式裡，讓它可被查核）：
  * **不修改任何已驗證的執行程式**；本檔只讀取已完成趟次的 log。
  * 這是**已錄狀態的回放**，不是重新模擬：每一幀把關節角與抽屜開度
    直接設回場景，再讀回核對，偏差寫進 render_meta.json。
    影片中的姿態因此代表「該趟實際量到的狀態」，不是另一次模擬的結果。
  * **不擷取桌面畫面**：影像只來自 isaacsim.sensors.camera 的相機感測器。
  * 保護條件照舊：CPU 溫度達 92 °C 即中止（中止線不調整）。
"""
import argparse, json, math, os, re, sys, csv
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, 'evaluation'))

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=('drawer', 'pregrasp'))
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf'))
ap.add_argument('--t0', type=float, required=True)
ap.add_argument('--t1', type=float, required=True)
ap.add_argument('--fps', type=float, default=30.0)
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=720)
ap.add_argument('--focal', type=float, default=17.0)
ap.add_argument('--eye', default='')
ap.add_argument('--at', default='')
ap.add_argument('--limit-frames', type=int, default=0)
ap.add_argument('--no-color', action='store_true')
ap.add_argument('--diag', action='store_true',
                help='第一幀印出各連桿的世界位置，用來核對回放姿態')
ap.add_argument('--settle-steps', type=int, default=4,
                help='每幀在取像前重複算繪的次數：RTX 標註器有延遲，只算一次會取到前幾幀的畫面')
ap.add_argument('--temp-limit', type=float, default=92.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import yaml                                                        # noqa: E402
import drawer_asset as DA                                          # noqa: E402
from cpu_temp import read as cpu_temp_c                        # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
ROBOT = '/World/omni_bot'
DRAWER = '/World/drawer_unit/drawer'

# ---------- 讀回已錄狀態（唯讀，不重新模擬） ----------
if a.kind == 'drawer':
    J = json.load(open(os.path.join(a.run, 'sim', 'drawer_run.json')))
    ci = {c: k for k, c in enumerate(J['log_cols'])}
    LOG = J['log']
    T = np.array([r[ci['t']] for r in LOG], dtype=float)
    Q = np.array([[r[ci[x]] for x in ARM] for r in LOG], dtype=float)
    OP = np.array([r[ci['opening']] for r in LOG], dtype=float)
    PH = [r[ci['phase']] for r in LOG]
    SEQ = np.array([r[ci['applied_seq']] for r in LOG], dtype=float)
    PARK = [float(v) for v in J['park']]
    CASE_NAME = J['case']
    EVENTS = {e['event']: float(e['sim_t']) for e in J.get('events', [])}
    FING = {}
    tp = os.path.join(a.run, 'traj', 'traj.csv')
    if os.path.exists(tp):
        for row in csv.DictReader(open(tp)):
            FING[int(row['seq'])] = float(row['finger'])
else:
    J = json.load(open(os.path.join(a.run, 'manip_run.json')))
    LOG = J['log']
    T = np.array([r['t'] for r in LOG], dtype=float)
    Q = np.array([r['q'] for r in LOG], dtype=float)
    OP = np.zeros(len(T))
    PH = [''] * len(T)
    SEQ = np.array([r.get('applied_seq', -1) for r in LOG], dtype=float)
    pk = J['parking_target']
    PARK = [float(pk['x']), float(pk['y']), math.radians(float(pk['yaw_deg']))]
    CASE_NAME = J['case']
    EVENTS, FING = {}, {}

CASES = yaml.safe_load(open(a.cases))
CASE = CASES['cases'][CASE_NAME]
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
print(f'[render] {a.kind}：讀入 {len(T)} 個已錄狀態，t = {T.min():.2f} … {T.max():.2f} s')
print(f'[render] 案例 {CASE_NAME}；停放 ({PARK[0]:.4f}, {PARK[1]:.4f}, '
      f'{math.degrees(PARK[2]):.2f}°)；物件 {POSE}')

from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True})
from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.api.objects import FixedCuboid                  # noqa: E402
from isaacsim.core.prims import SingleArticulation, RigidPrim      # noqa: E402
from isaacsim.core.utils.types import ArticulationAction           # noqa: E402
from isaacsim.sensors.camera import Camera                         # noqa: E402
from pxr import UsdGeom, UsdLux, Gf                                # noqa: E402
from isaac_common import import_urdf                               # noqa: E402
import imageio.v2 as imageio                                       # noqa: E402


def q_yaw(t):
    return [math.cos(t / 2.0), 0.0, 0.0, math.sin(t / 2.0)]


# 只影響顯示顏色的著色：**純視覺辨識用**，不改任何幾何、質量或物理參數。
# 櫃體與抽屜在預設材質下同為灰色，20 mm 的行程在畫面上讀不出來。
TINT = {'cabinet': (0.40, 0.38, 0.36), 'drawer': (0.22, 0.41, 0.62),
        'handle': (0.95, 0.60, 0.12)}


def tint(stage, root):
    from pxr import Sdf, Vt
    n = 0
    for p in stage.Traverse():
        sp = str(p.GetPath())
        if not sp.startswith(root) or not p.IsA(UsdGeom.Gprim):
            continue
        name = p.GetName()
        key = ('handle' if name.startswith(('handle', 'post')) else
               'cabinet' if '/cabinet/' in sp else 'drawer')
        UsdGeom.Gprim(p).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*TINT[key])]))
        n += 1
    print(f'[render] 已著色 {n} 個幾何（僅顯示用，未改幾何與物理）')


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    key = UsdLux.DistantLight.Define(stage, '/World/key_light')
    key.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
    dome = UsdLux.DomeLight.Define(stage, '/World/dome_light')
    dome.CreateIntensityAttr(450.0)

    if a.kind == 'drawer':
        SPEC = DA.load(a.spec)
        DA.build_usd(stage, SPEC, POSE)
        if not a.no_color:
            tint(stage, '/World/drawer_unit')
        focus = np.array([POSE[0], POSE[1] - 0.38, 0.45])
        d_eye = np.array([1.70, -2.45, 0.92])
    else:
        ws = re.sub(r'<!--.*?-->', '', open(a.world).read(), flags=re.S)
        nb = 0
        for m in re.finditer(r'<model name="(known_obs_\d+)">(.*?)</model>', ws, re.S):
            n, b = m.group(1), m.group(2)
            p = [float(v) for v in re.search(r'<pose>([^<]+)</pose>', b).group(1).split()]
            sz = [float(v) for v in re.search(r'<box>\s*<size>([^<]+)</size>',
                                              b).group(1).split()]
            tgt = (n == CASE['object'].get('name'))
            FixedCuboid(prim_path=f'/World/{n}', name=n,
                        position=np.array(p[:3]), scale=np.array(sz),
                        color=(np.array(TINT['handle']) if (tgt and not a.no_color)
                               else np.array([0.78, 0.78, 0.80])))
            nb += 1
            if tgt:
                print(f'[render] 目標物件 {n} 已著色（僅顯示用）')
        print(f'[render] 靜態方塊 {nb} 個')
        focus = np.array([POSE[0], PARK[1] + 0.33, 0.45])
        d_eye = np.array([1.80, -2.40, 1.00])

    import_urdf(a.urdf, ROBOT, fix_base=True)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(PARK[0], PARK[1], 0.0))
    rq = q_yaw(PARK[2])
    xf.AddOrientOp().Set(Gf.Quatf(rq[0], rq[1], rq[2], rq[3]))

    eye = np.array([float(x) for x in a.eye.split(',')]) if a.eye else focus + d_eye
    at = np.array([float(x) for x in a.at.split(',')]) if a.at else focus
    cam = Camera(prim_path='/World/demo_cam', resolution=(a.width, a.height))
    cxf = UsdGeom.Xformable(stage.GetPrimAtPath('/World/demo_cam'))
    cxf.ClearXformOpOrder()
    cxf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*[float(v) for v in eye]), Gf.Vec3d(*[float(v) for v in at]),
        Gf.Vec3d(0, 0, 1)).GetInverse())
    cg = UsdGeom.Camera(stage.GetPrimAtPath('/World/demo_cam'))
    cg.GetFocalLengthAttr().Set(float(a.focal))
    cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 200.0))

    drawer_v = RigidPrim(prim_paths_expr=DRAWER, name='drawer_v') \
        if a.kind == 'drawer' else None

    world.reset()
    cam.initialize()
    robot = SingleArticulation(prim_path=ROBOT, name='omni_bot')
    robot.initialize()
    idx = {n: k for k, n in enumerate(robot.dof_names)}
    miss = [j for j in ARM if j not in idx]
    if miss:
        print(f'[render] URDF 缺關節 {miss}，中止'); return 5
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM + FJ:
        kp[idx[j]] = 8e4; kd[idx[j]] = 4e3
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    print(f'[render] articulation DOF {robot.num_dof}；相機 '
          f'eye={np.round(eye,3).tolist()} at={np.round(at,3).tolist()} '
          f'f={a.focal}mm', flush=True)

    # 相機標註器要先跑幾個算繪幀才會有資料（首次取像會回 None）。
    for _ in range(30):
        world.step(render=True)
        _w = cam.get_rgba()
        if _w is not None and len(_w):
            break
    else:
        print('[render] 相機暖機 30 幀後仍無影像，中止'); return 3
    print('[render] 相機暖機完成', flush=True)

    ts = np.arange(a.t0, a.t1, 1.0 / a.fps)
    if a.limit_frames:
        ts = ts[:a.limit_frames]
    sel = np.clip(np.searchsorted(T, ts), 0, len(T) - 1)

    dq_max = 0.0                      # 讀回 vs 已錄：關節最大差（rad）
    dop_max = 0.0                     # 讀回 vs 已錄：開度最大差（m）
    tmax = -1.0
    rows = []
    n = 0
    for k, tt in zip(sel, ts):
        q = robot.get_joint_positions()
        for j, nm in enumerate(ARM):
            q[idx[nm]] = Q[k, j]
        # idle 階段還沒有套用過任何命令，applied_seq 是 NaN
        _sq = SEQ[k]
        fv = (FING.get(int(_sq)) if (FING and np.isfinite(_sq)) else None)
        if fv is not None:
            for j in FJ:
                q[idx[j]] = fv
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=q))
        if drawer_v is not None:
            drawer_v.set_world_poses(
                positions=np.array([[POSE[0], POSE[1] - float(OP[k]), 0.0]]),
                orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
            drawer_v.set_velocities(np.zeros((1, 6), dtype=np.float32))

        # 狀態每次都重設成同一個值，所以重複步進不會改變姿態，
        # 只是讓算繪管線追上——RTX 標註器會落後數幀。
        for _ in range(max(1, a.settle_steps)):
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
            if drawer_v is not None:
                drawer_v.set_world_poses(
                    positions=np.array([[POSE[0], POSE[1] - float(OP[k]), 0.0]]),
                    orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
                drawer_v.set_velocities(np.zeros((1, 6), dtype=np.float32))
            world.step(render=True)

        qb = robot.get_joint_positions()
        dq = max(abs(float(qb[idx[nm]]) - Q[k, j]) for j, nm in enumerate(ARM))
        dq_max = max(dq_max, dq)
        dop = 0.0
        if drawer_v is not None:
            pb = drawer_v.get_world_poses()[0][0]
            dop = abs((POSE[1] - float(pb[1])) - float(OP[k]))
            dop_max = max(dop_max, dop)

        if n == 0 and a.diag:
            from pxr import UsdGeom as _UG
            _xc = _UG.XformCache()
            print('[diag] /World/omni_bot 之下的 Xformable：', flush=True)
            for _pp in stage.Traverse():
                _sp = str(_pp.GetPath())
                if not _sp.startswith(ROBOT) or not _pp.IsA(UsdGeom.Xformable):
                    continue
                if not any(w in _pp.GetName() for w in
                            ('link', 'finger', 'gripper', 'footprint')):
                    continue
                _t = _xc.GetLocalToWorldTransform(_pp).ExtractTranslation()
                print(f'   {_sp:<58s} ({_t[0]:.5f}, {_t[1]:.5f}, {_t[2]:.5f})',
                      flush=True)
            _rp = stage.GetPrimAtPath(ROBOT)
            _t = _xc.GetLocalToWorldTransform(_rp).ExtractTranslation()
            print(f'   {ROBOT} Xform ({_t[0]:.5f}, {_t[1]:.5f}, {_t[2]:.5f})')
            print(f'[diag] 本幀 log 關節 {np.round(Q[k], 5).tolist()}', flush=True)

        img = cam.get_rgba()
        for _ in range(5):
            if img is not None and len(img):
                break
            world.step(render=True)
            img = cam.get_rgba()
        if img is None or not len(img):
            print(f'[render] 第 {n} 幀取不到影像，中止'); return 3
        imageio.imwrite(os.path.join(a.out, f'f{n:05d}.png'),
                        np.asarray(img)[:, :, :3].astype(np.uint8))
        rows.append([round(float(tt), 4), round(float(T[k]), 4),
                     round(float(OP[k]) * 1000.0, 4), PH[k],
                     round(dq, 8), round(dop, 8)])
        if n % 30 == 0:
            tc, src = cpu_temp_c()
            if tc is not None:
                tmax = max(tmax, tc)
                if tc >= a.temp_limit:
                    print(f'[render] **CPU {tc:.1f} °C ≥ {a.temp_limit} °C，中止**')
                    return 7
            print(f'  {n}/{len(ts)}  t={tt:.2f}s  開度 {OP[k]*1000:6.2f} mm  '
                  f'{PH[k]:<10s} Δq {dq*1e3:.4f} mrad  Δ開度 {dop*1e3:.4f} mm  '
                  f'CPU {tc if tc is None else round(tc,1)} °C', flush=True)
        n += 1

    with open(os.path.join(a.out, 'replay_check.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_t', 'log_t', 'opening_mm', 'phase',
                    'readback_dq_rad', 'readback_dopening_m'])
        w.writerows(rows)
    meta = {
        'schema': 'demo_render/1', 'kind': a.kind, 'case': CASE_NAME,
        'run': os.path.basename(os.path.abspath(a.run)),
        'frames': n, 'fps': a.fps, 't0': a.t0, 't1': a.t1,
        'resolution': [a.width, a.height], 'focal_mm': a.focal,
        'eye': [float(v) for v in eye], 'at': [float(v) for v in at],
        'park': PARK, 'object_pose': list(POSE), 'events_sim_t': EVENTS,
        'readback_dq_max_rad': dq_max, 'readback_dopening_max_m': dop_max,
        'cpu_temp_max_c': None if tmax < 0 else tmax,
        'cpu_temp_limit_c': a.temp_limit,
        'image_source': ('isaacsim.sensors.camera 相機感測器；'
                         '非桌面擷取'),
        'nature': ('已錄狀態回放：逐幀設定關節角與抽屜開度後讀回核對，'
                   '不是重新模擬，不會改動原始趟次資料'),
        'display_tint': not a.no_color,
        'tint_note': '櫃體／抽屜／把手／目標方塊分色僅為視覺辨識，未改幾何、質量或物理參數',
    }
    json.dump(meta, open(os.path.join(a.out, 'render_meta.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'[render] 完成 {n} 幀 -> {a.out}')
    print(f'[render] 讀回核對：關節最大差 {dq_max*1e3:.5f} mrad、'
          f'開度最大差 {dop_max*1e3:.5f} mm；CPU 峰值 {tmax:.1f} °C')
    return 0


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    tb = traceback.format_exc()
    print(tb, flush=True)
    open(os.path.join(a.out, 'traceback.txt'), 'w').write(tb)
    rc = 9
finally:
    sim_app.close()
sys.exit(rc)
