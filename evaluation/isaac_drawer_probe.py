"""抽屜資產的**獨立診斷**：建構、參數讀回、被動性、開度正負號。

這支**不是**正式試驗，也不會被正式試驗載入。它會對抽屜施加外力去確認滑動方向
與限位 —— 正式拉抽屜試驗裡抽屜只能由手臂經抓取關係帶動，絕不施加外力。
兩件事分開在不同檔案，就不會有人不小心把診斷用的施力留在正式流程裡。

四項檢查，任何一項不過就中止（回傳非零）：

    A 幾何讀回   從 stage 重新讀出每個碰撞體的型別、局部位姿、尺寸，與規格逐項比對
    B 物理讀回   質量、阻尼、關節摩擦、限位，以及 drive 五項必須全為 0
    C 單位校正   已知質量的方塊靜置地面，量淨接觸力，核對 dt 的單位換算
    D 被動與方向 無外力時不自行開啟；施力後正開度確實沿拉出方向增加；兩端限位有效

用法：
    ISAAC_PY=~/venvs/isaacsim-6.0.1/bin/python
    $ISAAC_PY evaluation/isaac_drawer_probe.py --out evaluation/results/drawer_probe_<stamp>
"""
import argparse, json, math, os, sys, time
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')

ap = argparse.ArgumentParser()
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--headless', default='true')
ap.add_argument('--settle-s', type=float, default=1.0, help='建構後靜置多久再開始量')
ap.add_argument('--passive-s', type=float, default=5.0, help='被動性觀察時長')
ap.add_argument('--passive-max-m', type=float, default=0.001, help='無外力時容許的開度')
ap.add_argument('--probe-force-n', type=float, default=8.0, help='方向診斷施力')
ap.add_argument('--probe-s', type=float, default=3.0)
ap.add_argument('--cpu-limit', type=float, default=92.0)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

sys.path.insert(0, HERE)
import drawer_asset as DA                                          # noqa: E402
import yaml                                                        # noqa: E402

SPEC = DA.load(a.spec)
CASE = yaml.safe_load(open(a.cases))['cases'][a.case]
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
os.makedirs(a.out, exist_ok=True)

import hashlib                                                     # noqa: E402
SPEC_SHA = hashlib.sha256(open(a.spec, 'rb').read()).hexdigest()[:16]
print(f'[probe] 規格 {SPEC["name"]}  sha {SPEC_SHA}')
print(f'[probe] 櫃體世界位置 ({POSE[0]:.3f}, {POSE[1]:.3f})')
print(f'[probe] physics_dt {a.physics_dt}')

from isaacsim import SimulationApp                                 # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
if a.cpu_threads > 0:
    _cfg['limit_cpu_threads'] = a.cpu_threads
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid               # noqa: E402
from isaacsim.core.prims import RigidPrim                          # noqa: E402
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Usd          # noqa: E402

G = 9.81
FAIL = []


def bad(tag, msg):
    FAIL.append(f'{tag}: {msg}')
    print(f'  ✗ {tag}  {msg}')


def ok(tag, msg):
    print(f'  ✓ {tag}  {msg}')


from cpu_temp import read as cpu_temp_read                       # noqa: E402


def local_box(stage, path):
    """從 stage 讀回一個 Cube 的世界中心與世界尺寸（只支援平移＋縮放）。"""
    pr = stage.GetPrimAtPath(path)
    if not pr.IsValid():
        return None
    M = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = M.ExtractTranslation()
    # Cube size=2 搭配 scale=half-extent，所以世界尺寸 = 2 * scale
    R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
    sc = np.linalg.norm(R, axis=0)
    sz_attr = UsdGeom.Cube(pr).GetSizeAttr().Get()
    return np.array([t[0], t[1], t[2]]), sc * float(sz_attr)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    if abs(float(world.get_rendering_dt()) - float(a.physics_dt)) > 1e-12:
        print('[probe] rendering_dt != physics_dt，中止'); return 4
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    authored = DA.build_usd(world.stage, SPEC, POSE)
    root = authored['root']
    dpath = authored['drawer_prim']
    jpath = authored['joint_prim']

    # 單位校正用：已知質量的方塊，放在遠離抽屜的地方
    CAL_M = 3.0
    cal = DynamicCuboid(prim_path='/World/cal_box', name='cal_box',
                        position=np.array([POSE[0] + 3.0, POSE[1] + 3.0, 0.10]),
                        scale=np.array([0.2, 0.2, 0.2]), mass=CAL_M)

    # **view 必須在 world.reset() 之前建立**：prepare_contact_sensors 要在 PhysX
    # 場景建好之前把 PhysxContactReportAPI 套上去才有效。reset 之後才建 view，
    # API 雖然套得上、threshold 也讀得到 0，但接觸力**一律回傳 0**
    # （已用已知質量的方塊實測確認：view 先建 → 29.52 N，view 後建 → 0.0 N）。
    drawer = RigidPrim(prim_paths_expr=dpath, name='drawer_view',
                       track_contact_forces=True, max_contact_count=128,
                       prepare_contact_sensors=True)
    calv = RigidPrim(prim_paths_expr='/World/cal_box', name='cal_view',
                     track_contact_forces=True, max_contact_count=128,
                     prepare_contact_sensors=True)

    world.reset()
    stage = world.stage

    rep = {'schema': 'drawer_probe/1', 'spec': SPEC['name'], 'spec_sha256_16': SPEC_SHA,
           'case': a.case, 'pose': list(POSE), 'physics_dt': a.physics_dt,
           'authored': authored}

    # ---------------------------------------------------------- A 幾何讀回
    print('\n[A] 幾何讀回（從 stage 重新讀，不看 build_usd 的回傳值）')
    geo = []
    want = []
    for n, c, s in SPEC['cabinet']:
        want.append((f'{root}/cabinet/{n}', np.array([c[0] + POSE[0], c[1] + POSE[1], c[2]]),
                     np.array(s, float)))
    for n, c, s in SPEC['drawer']['body']:
        want.append((f'{dpath}/{n}', np.array([c[0] + POSE[0], c[1] + POSE[1], c[2]]),
                     np.array(s, float)))
    for n, c, s in SPEC['drawer']['handle']['posts']:
        want.append((f'{dpath}/{n}', np.array([c[0] + POSE[0], c[1] + POSE[1], c[2]]),
                     np.array(s, float)))
    worst_c, worst_s, worst_at, worst_s_at = 0.0, 0.0, None, None
    for path, wc, ws in want:
        got = local_box(stage, path)
        if got is None:
            bad('A', f'{path} 不存在'); continue
        gc, gs = got
        dc = float(np.abs(gc - wc).max()); dsz = float(np.abs(gs - ws).max())
        if dc > worst_c: worst_c, worst_at = dc, path
        if dsz > worst_s: worst_s, worst_s_at = dsz, path
        has_col = stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
        if not has_col:
            bad('A', f'{path} 沒有 CollisionAPI')
        geo.append({'path': path, 'center': gc.tolist(), 'size': gs.tolist(),
                    'center_err': dc, 'size_err': dsz, 'collision_api': bool(has_col)})
    # 圓柱把手
    bp = f'{dpath}/handle_bar'
    bpr = stage.GetPrimAtPath(bp)
    b = SPEC['drawer']['handle']['bar']
    if not bpr.IsValid():
        bad('A', f'{bp} 不存在')
    else:
        cyl = UsdGeom.Cylinder(bpr)
        r_got = float(cyl.GetRadiusAttr().Get()); h_got = float(cyl.GetHeightAttr().Get())
        ax_got = str(cyl.GetAxisAttr().Get())
        M = UsdGeom.Xformable(bpr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation()
        wc = np.array([b['center'][0] + POSE[0], b['center'][1] + POSE[1], b['center'][2]])
        dc = float(np.abs(np.array([t[0], t[1], t[2]]) - wc).max())
        # PhysX 對 Cylinder 通常以凸包近似（內接多邊形），有效半徑會略小於 5 mm。
        # 方向上是安全的（包覆餘裕變大），但摩擦版的接觸法線會改變，所以要記下來。
        approx = None
        if bpr.HasAPI(UsdPhysics.MeshCollisionAPI):
            v = UsdPhysics.MeshCollisionAPI(bpr).GetApproximationAttr().Get()
            approx = str(v) if v is not None else None
        rep['handle_bar'] = {'radius': r_got, 'height': h_got, 'axis': ax_got,
                             'center_err': dc, 'collision_api':
                             bool(bpr.HasAPI(UsdPhysics.CollisionAPI)),
                             'physx_approximation': approx}
        for nm, g_, w_ in (('radius', r_got, b['radius']), ('height', h_got, b['length'])):
            if abs(g_ - w_) > 1e-9:
                bad('A', f'橫桿 {nm} 讀回 {g_} != 規格 {w_}')
        if ax_got != b['axis'].upper():
            bad('A', f'橫桿軸向讀回 {ax_got} != 規格 {b["axis"].upper()}')
        if dc > 1e-6:
            bad('A', f'橫桿中心誤差 {dc*1000:.4f} mm')
        else:
            ok('A', f'橫桿 ⌀{r_got*2*1000:.1f} mm 長 {h_got*1000:.0f} mm 軸 {ax_got}，'
                    f'PhysX 近似 {approx or "（未標註，預設凸包）"}')
    if worst_at is not None:
        (ok if worst_c <= 1e-6 and worst_s <= 1e-6 else bad)(
            'A', f'{len(geo)} 個方塊：中心最大誤差 {worst_c*1e6:.3f} µm'
                 f'（{os.path.basename(worst_at)}）、'
                 f'尺寸最大誤差 {worst_s*1e6:.3f} µm'
                 f'（{os.path.basename(worst_s_at) if worst_s_at else "-"}）')
    rep['geometry'] = geo

    # ---------------------------------------------------------- B 物理讀回
    print('\n[B] 物理參數讀回')
    ph = SPEC['drawer']['physics']; jt = SPEC['drawer']['joint']
    dpr = stage.GetPrimAtPath(dpath); jpr = stage.GetPrimAtPath(jpath)
    m_got = float(UsdPhysics.MassAPI(dpr).GetMassAttr().Get())
    ld_got = float(PhysxSchema.PhysxRigidBodyAPI(dpr).GetLinearDampingAttr().Get())
    jf_got = float(PhysxSchema.PhysxJointAPI(jpr).GetJointFrictionAttr().Get())
    pj = UsdPhysics.PrismaticJoint(jpr)
    lo_got = float(pj.GetLowerLimitAttr().Get()); hi_got = float(pj.GetUpperLimitAttr().Get())
    ax_j = str(pj.GetAxisAttr().Get())
    drv = UsdPhysics.DriveAPI.Get(jpr, 'linear')
    drive = {k: (float(v.Get()) if v and v.Get() is not None else None) for k, v in {
        'stiffness': drv.GetStiffnessAttr(), 'damping': drv.GetDampingAttr(),
        'targetPosition': drv.GetTargetPositionAttr(),
        'targetVelocity': drv.GetTargetVelocityAttr(),
        'maxForce': drv.GetMaxForceAttr()}.items()}
    rep['physics_readback'] = {'mass_kg': m_got, 'linear_damping': ld_got,
                               'joint_friction': jf_got, 'axis': ax_j,
                               'limit_lower': lo_got, 'limit_upper': hi_got,
                               'drive': drive}
    # USD 的 limit / stiffness 等屬性是**單精度**，0.22 存進去讀回來是
    # 0.2199999988...，用 1e-9 比會假性失敗。容差取 float32 的解析度量級。
    F32 = 1e-6
    for nm, g_, w_ in (('質量', m_got, ph['mass_kg']),
                       ('線性阻尼', ld_got, ph['linear_damping']),
                       ('關節摩擦', jf_got, ph['joint_friction']),
                       ('限位下界', lo_got, jt['lower']),
                       ('限位上界', hi_got, jt['upper'])):
        d_ = abs(g_ - w_)
        (ok if d_ <= F32 else bad)(
            'B', f'{nm} 讀回 {g_:.9g}（規格 {w_:.9g}，差 {d_:.3g}）'
                 + ('' if d_ <= F32 else '  **超出 float32 容差**'))
    for k, v in drive.items():
        if v is None:
            bad('B', f'drive.{k} 讀不到（必須存在且為 0）')
        elif v != 0.0:
            bad('B', f'drive.{k} = {v!r}，**必須為 0**（抽屜不接受任何命令）；'
                     f'不以「數值很小」通融')
    if all(v == 0.0 for v in drive.values() if v is not None):
        ok('B', 'drive 五項讀回皆為 0：stiffness / damping / targetPosition / '
                'targetVelocity / maxForce')

    for _ in range(int(a.settle_s / a.physics_dt)):
        world.step(render=False)

    drawer.initialize()
    calv.initialize()
    for v in (drawer, calv):
        try:
            v.set_sleep_thresholds(np.zeros((1,), dtype=np.float32))
        except Exception as e:
            print(f'  （set_sleep_thresholds 失敗：{e}）')

    def opening():
        p, _ = drawer.get_world_poses()
        return float(Y0 - p[0][1])

    p0, _ = drawer.get_world_poses()
    Y0 = float(p0[0][1])
    rep['drawer_y0'] = Y0

    # ---------------------------------------------------------- C 單位校正
    print('\n[C] 接觸力單位校正（已知質量方塊靜置地面）')
    for _ in range(int(2.0 / a.physics_dt)):
        world.step(render=False)
    vel = calv.get_velocities()[0]
    print(f'      校正方塊速度 {np.round(vel[:3],6).tolist()} m/s（應已靜止）')
    f_dt = calv.get_net_contact_forces(dt=a.physics_dt)[0]
    f_default = calv.get_net_contact_forces()[0]
    w_exp = CAL_M * G
    got = float(abs(f_dt[2])); rel = abs(got - w_exp) / w_exp
    rep['contact_calibration'] = {
        'mass_kg': CAL_M, 'expected_N': w_exp,
        'with_dt_physics_dt_N': [float(v) for v in f_dt],
        'with_default_dt': [float(v) for v in f_default],
        'rel_err': rel}
    print(f'      dt=physics_dt  → {np.round(f_dt,4).tolist()}   期望 z ≈ {w_exp:.3f} N')
    print(f'      dt 預設 1.0    → {np.round(f_default,6).tolist()}   （文件稱此為衝量）')
    (ok if rel < 0.05 else bad)('C', f'量到 {got:.3f} N vs m·g {w_exp:.3f} N，相對誤差 {rel*100:.2f} %')
    if abs(float(f_default[2])) > 1e-12:
        ratio = got / abs(float(f_default[2]))
        print(f'      兩者比值 {ratio:.4f}（1/physics_dt = {1/a.physics_dt:.1f}）')
        rep['contact_calibration']['ratio'] = ratio

    # ---------------------------------------------------------- D 被動與方向
    print('\n[D] 被動性與開度正負號')
    drawer.set_velocities(np.zeros((1, 6), dtype=np.float32))
    mx = 0.0; trace = []
    n = int(a.passive_s / a.physics_dt)
    for i in range(n):
        world.step(render=False)
        o = opening(); mx = max(mx, abs(o))
        if i % 50 == 0: trace.append([round(i * a.physics_dt, 3), round(o, 6)])
    rep['passive'] = {'duration_s': a.passive_s, 'max_abs_opening_m': mx, 'trace': trace}
    (ok if mx <= a.passive_max_m else bad)(
        'D', f'無外力 {a.passive_s:.1f} s，開度最大絕對值 {mx*1000:.4f} mm '
             f'(門檻 {a.passive_max_m*1000:.1f} mm)')

    def push(fy, secs, label):
        F = np.array([[0.0, fy, 0.0]], dtype=np.float32)
        tr = []
        k = int(secs / a.physics_dt)
        for i in range(k):
            drawer.apply_forces(F, is_global=True)
            world.step(render=False)
            if i % 25 == 0: tr.append([round(i * a.physics_dt, 3), round(opening(), 6)])
        o = opening()
        print(f'      {label}: 施力 y = {fy:+.2f} N {secs:.1f} s → 開度 {o*1000:+8.3f} mm')
        return o, tr

    # 拉出方向 = 世界 −y（規格：抽屜沿本地 −y 拉出，且 yaw = 0）
    o_pull, tr1 = push(-a.probe_force_n, a.probe_s, '往拉出方向')
    (ok if o_pull > 0.005 else bad)(
        'D', f'沿 −y 施力後開度 {o_pull*1000:+.3f} mm > 0 ⇒ '
             f'**正開度確實沿拉出方向增加**' if o_pull > 0.005
             else f'沿 −y 施力後開度 {o_pull*1000:+.3f} mm，正負號或方向不符')
    o_max, tr2 = push(-40.0, 4.0, '壓到上限')
    hi = float(jt['upper'])
    (ok if abs(o_max - hi) < 0.003 else bad)(
        'D', f'大力持續後停在 {o_max*1000:.3f} mm，硬限位 {hi*1000:.1f} mm')
    o_back, tr3 = push(+40.0, 4.0, '推回下限')
    (ok if abs(o_back) < 0.003 else bad)(
        'D', f'反向大力後停在 {o_back*1000:.3f} mm，硬限位 {float(jt["lower"])*1000:.1f} mm')
    # 側向與垂直：滑動關節必須擋住
    for axis, vec in (('x', [30.0, 0.0, 0.0]), ('z', [0.0, 0.0, -30.0])):
        pa, _ = drawer.get_world_poses(); pa = pa[0].copy()
        F = np.array([vec], dtype=np.float32)
        for _ in range(int(2.0 / a.physics_dt)):
            drawer.apply_forces(F, is_global=True)
            world.step(render=False)
        pb, _ = drawer.get_world_poses()
        d = float(np.abs(np.array(pb[0]) - np.array(pa))[{'x': 0, 'z': 2}[axis]])
        (ok if d < 0.001 else bad)(
            'D', f'沿 {axis} 施 30 N，該軸位移 {d*1000:.4f} mm（滑動關節應擋住）')
    rep['direction'] = {'probe_force_n': a.probe_force_n,
                        'opening_after_pull_m': o_pull,
                        'opening_at_upper_limit_m': o_max,
                        'opening_at_lower_limit_m': o_back,
                        'trace_pull': tr1, 'trace_upper': tr2, 'trace_lower': tr3}

    t, tsrc = cpu_temp_read()
    rep['cpu_temp_c'] = t
    rep['cpu_temp_source'] = tsrc
    rep['fail'] = FAIL
    rep['pass'] = not FAIL
    json.dump(rep, open(os.path.join(a.out, 'probe.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'\nCPU {t if t is not None else "讀不到"} °C（來源 {tsrc}，'
          f'中止線 {a.cpu_limit:.0f} °C）')
    if t is not None and t >= a.cpu_limit:
        bad('溫度', f'{t:.1f} °C 已達中止線 {a.cpu_limit:.0f} °C')
    print(f'判定：{"通過" if not FAIL else "**未通過**"}')
    for f in FAIL:
        print(f'   ✗ {f}')
    print(f'輸出：{os.path.join(a.out, "probe.json")}')
    return 0 if not FAIL else 1


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
