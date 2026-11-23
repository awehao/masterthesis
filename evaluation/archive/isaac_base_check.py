"""Isaac Sim: ground plane + chassis only. Geometry first, then direct velocity.

Scope
-----
Gate 0  scene builds, chassis imports, and its DIMENSIONS, ORIENTATION, ORIGIN
        and GROUND CLEARANCE match the URDF the Gazebo work used.
Gate 1  direct velocity control: forward, lateral, rotation, and the zero /
        silence pair that Gazebo was measured on.

What this is not
----------------
It is not tuned to reproduce Gazebo's numbers. The same INPUT is applied and the
response is recorded; where the two differ, the difference is the finding. The
Gazebo reference (`evaluation/results/gz_base_stop_*.json`) holds the same
commands and the measured trajectory for exactly this comparison.

On "direct velocity": the API sets the articulation root's velocity state
immediately. Whether the following physics steps preserve it, and how contacts
and joint constraints act on it, is measured here rather than assumed -- the
same caution the Gazebo measurements turned out to need.

    ISAAC=~/venvs/isaacsim-6.0.1/bin/python
    $ISAAC evaluation/isaac_base_check.py --gate 0
    $ISAAC evaluation/isaac_base_check.py --gate 1 --apply hold --stop zero
    $ISAAC evaluation/isaac_base_check.py --gate 1 --apply hold --stop silence
    $ISAAC evaluation/isaac_base_check.py --gate 1 --apply once --stop silence
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--gate', type=int, default=0, choices=[0, 1])
ap.add_argument('--urdf', default='/tmp/omni_bot_base.urdf')
ap.add_argument('--headless', default='true')
ap.add_argument('--dt', type=float, default=1.0 / 1000.0,
                help='physics step; Gazebo used 1 ms')
ap.add_argument('--rate', type=float, default=50.0,
                help='command rate, Hz; Gazebo reference used 50')
# Two INDEPENDENT dimensions. They were conflated in the first draft, where
# the 'once' run still called the API a single time with zero during its stop
# segment -- that is an explicit zero, not silence, so silence was never tested.
ap.add_argument('--apply', default='hold', choices=['hold', 'once'],
                help="how the velocity STATE is applied during motion: "
                     "'hold' re-applies every control tick, 'once' applies it "
                     "a single time and then stops touching it")
ap.add_argument('--stop', default='zero', choices=['zero', 'silence'],
                help="how motion is ENDED: 'zero' commands zero explicitly "
                     "(at the same cadence --apply uses), 'silence' stops "
                     "calling the API altogether without ever sending zero")
ap.add_argument('--friction', default='zero', choices=['zero', 'default'],
                help="'zero' 套用 URDF 宣告的 mu=0（importer 不讀 <gazebo> 標籤）；"
                     "'default' 沿用 PhysX 預設，用來重現追蹤率不足的情形")
ap.add_argument('--vx', type=float, default=-0.10)
ap.add_argument('--drive-s', type=float, default=10.0)
ap.add_argument('--watch-s', type=float, default=4.0)
ap.add_argument('--out', default='evaluation/results/isaac_base')
a = ap.parse_args()

from isaacsim import SimulationApp                                # noqa: E402
sim_app = SimulationApp({"headless": a.headless.lower() == 'true'})

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from isaacsim.core.utils.extensions import enable_extension       # noqa: E402

enable_extension('isaacsim.asset.importer.urdf')
sim_app.update()
# 6.0 exposes URDFImporter / URDFImporterConfig; the old _urdf binding moved to
# an impl submodule and is not the supported entry point.
from isaacsim.asset.importer.urdf import (                        # noqa: E402
    URDFImporter, URDFImporterConfig)
from isaacsim.core.utils.stage import add_reference_to_stage      # noqa: E402


def import_chassis(path: str, prim_path: str = '/World/omni_bot') -> str:
    """URDF -> USD -> referenced onto the stage. Returns the prim path.

    fix_base=False: the chassis is a floating base, as in Gazebo, where it is
    held up by the four frictionless support spheres rather than pinned to the
    world. merge_fixed_joints stays False so the link names survive -- the port
    checklist compares link-by-link, and merging would silently rename them.
    """
    out_dir = os.path.join(os.path.dirname(path) or '.', 'isaac_usd')
    os.makedirs(out_dir, exist_ok=True)
    cfg = URDFImporterConfig(
        urdf_path=path, usd_path=out_dir,
        merge_fixed_joints=False, fix_base=False,
        allow_self_collision=False, collision_from_visuals=False)
    usd = URDFImporter(cfg).import_urdf()
    print(f'  URDF → USD：{usd}')
    add_reference_to_stage(usd_path=usd, prim_path=prim_path)
    # The importer puts all physics behind a variant set, and then authors a
    # selection that does not exist: variantSet "Physics" defines "none" and
    # "physics", while the root layer selects "physx". A dangling selection
    # composes nothing, so the stage carries geometry and no physics at all --
    # which surfaces later as "did not match any articulations" and reads like
    # the model has no physics. Select the variant that actually exists.
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    vset = stage.GetPrimAtPath(prim_path).GetVariantSets().GetVariantSet('Physics')
    names = vset.GetVariantNames()
    pick = 'physics' if 'physics' in names else (names[0] if names else '')
    if pick:
        vset.SetVariantSelection(pick)
    print(f'  Physics 變體：可選 {names}，匯入時選擇 '
          f'{vset.GetVariantSelection()!r}（原始選擇懸空需修正）')
    stage.Load(prim_path)
    return prim_path


def aabb(prim_path: str):
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    return compute_aabb(create_bbox_cache(), prim_path=prim_path,
                        include_children=True)


def apply_frictionless(prim_path: str, ground_path: str) -> int:
    """Bind mu=0 to the four support spheres and to the ground.

    The URDF states zero friction inside <gazebo><mu1>0</mu1></gazebo> tags,
    which the URDF importer does not read at all: the generated USD carries no
    PhysicsMaterialAPI and no material binding, so PhysX falls back to its
    default (~0.5) and the chassis is dragged to a stop between velocity
    writes. Both sides of the contact are set, because the default combine mode
    averages the two materials -- zeroing only the spheres would still leave
    half the ground's friction.
    """
    import omni.usd
    from pxr import UsdPhysics, UsdShade, Usd, Sdf
    stage = omni.usd.get_context().get_stage()
    mat_path = '/World/PhysicsMaterials/frictionless'
    stage.DefinePrim(Sdf.Path(mat_path).GetParentPath(), 'Scope')
    mp = stage.DefinePrim(mat_path, 'Material')
    m = UsdPhysics.MaterialAPI.Apply(mp)
    m.CreateStaticFrictionAttr().Set(0.0)
    m.CreateDynamicFrictionAttr().Set(0.0)
    m.CreateRestitutionAttr().Set(0.0)

    targets = []
    walk = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
        Usd.PrimDefaultPredicate))
    for pr in walk:
        sp = str(pr.GetPath())
        if not sp.startswith(prim_path) or not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if 'support_' in sp:
            targets.append(pr)
    gp = stage.GetPrimAtPath(ground_path)
    for pr in list(gp.GetChildren()) + [gp]:
        if pr.HasAPI(UsdPhysics.CollisionAPI):
            targets.append(pr)
    n = 0
    for pr in targets:
        # an instance proxy cannot be edited; bind on the instance's own prim
        if pr.IsInstanceProxy():
            continue
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(
            UsdShade.Material(mp), UsdShade.Tokens.weakerThanDescendants,
            'physics')
        n += 1
    print(f'  零摩擦材質已綁定 {n} 個碰撞體'
          f'（支撐球 {len([t for t in targets if "support_" in str(t.GetPath())])}'
          f' + 地面）')
    return n


def main() -> int:
    world = World(stage_units_in_meters=1.0,
                  physics_dt=a.dt, rendering_dt=a.dt * 20)
    GroundPlane(prim_path="/World/ground", name="ground",
                z_position=0.0)
    if not os.path.exists(a.urdf):
        print(f'  找不到 URDF：{a.urdf}\n'
              f'  先產生：xacro src/my_omnibot_description/urdf/omni_bot.urdf.xacro '
              f'use_arm:=false > {a.urdf}', file=sys.stderr)
        return 1
    prim = import_chassis(a.urdf)
    print(f'  匯入 prim：{prim}')
    if a.friction == 'zero':
        apply_frictionless(prim, '/World/ground')
    else:
        print('  摩擦：沿用 PhysX 預設（未套用 URDF 的 mu=0）')
    world.reset()
    for _ in range(50):
        world.step(render=False)

    # All 92 chassis URDF joints are fixed, but the importer still emits one
    # PhysicsArticulationRootAPI (on Geometry/base_footprint) over 84 rigid
    # bodies -- a single-link articulation, not a bare rigid body. The earlier
    # "no articulation" reading was wrong: the physics existed, the variant
    # selection was dangling. Which prim carries it is found and reported, not
    # assumed, because it decides what a velocity command acts on.
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    bodies = [str(pr.GetPath()) for pr in stage.Traverse()
              if pr.HasAPI(UsdPhysics.RigidBodyAPI)]
    arts = [str(pr.GetPath()) for pr in stage.Traverse()
            if pr.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f'    ArticulationRootAPI 的 prim：{arts if arts else "無"}')
    print(f'    RigidBodyAPI 的 prim（前 6）：{bodies[:6]}  共 {len(bodies)} 個')
    if not bodies and not arts:
        print('    !! 找不到剛體或 articulation，速度命令無處施加', file=sys.stderr)
        return 1
    if arts:
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(prim_path=arts[0], name='base')
        target = arts[0]
    else:
        from isaacsim.core.prims import SingleRigidPrim
        art = SingleRigidPrim(prim_path=bodies[0], name='base')
        target = bodies[0]
    art.initialize()
    p, q = art.get_world_pose()

    print(f'\n  ── Gate 0：幾何 ──')
    from pxr import Usd, UsdGeom

    def owner(pr):
        """Nearest ancestor rigid body -- the link this geometry belongs to."""
        x = pr
        while x and x.IsValid():
            if x.HasAPI(UsdPhysics.RigidBodyAPI):
                return x.GetName()
            x = x.GetParent()
        return '?'

    from isaacsim.core.utils.bounds import create_bbox_cache
    # Collision geometry is authored with purpose="guide", which a default
    # bbox cache excludes -- it returns an empty range, so an earlier pass
    # dropped all 7 collision prims before they were ever classified and
    # reported "no collision geometry". Both purposes need their own cache.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    gcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               ['default', 'render', 'proxy', 'guide'])
    vis, col = {}, {}
    # The importer marks link subtrees instanceable, and a plain Traverse()
    # stops at an instance without descending -- which is why an earlier pass
    # saw 5 visual links and zero collision prims. Instance proxies must be
    # visited explicitly to reach the geometry that actually defines the bounds.
    walk = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
        Usd.PrimDefaultPredicate))
    for pr in walk:
        if not str(pr.GetPath()).startswith(prim) or not pr.IsA(UsdGeom.Gprim):
            continue
        is_col = pr.HasAPI(UsdPhysics.CollisionAPI)
        rng = (gcache if is_col else cache).ComputeWorldBound(
            pr).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
        # prims carrying no geometry come back as FLT_MAX sentinels, not a box
        if not np.all(np.isfinite(mn)) or np.max(np.abs(mx)) > 1e30:
            continue
        d = col if is_col else vis
        n = owner(pr)
        if n in d:
            d[n] = (np.minimum(d[n][0], mn), np.maximum(d[n][1], mx))
        else:
            d[n] = (mn, mx)

    def envelope(d):
        lo = np.min([v[0] for v in d.values()], axis=0)
        hi = np.max([v[1] for v in d.values()], axis=0)
        return lo, hi

    vlo, vhi = envelope(vis)
    print(f'    【整機視覺 AABB】({vhi[0]-vlo[0]:.4f}, {vhi[1]-vlo[1]:.4f}, '
          f'{vhi[2]-vlo[2]:.4f}) m，z 範圍 [{vlo[2]:+.4f}, {vhi[2]:+.4f}]')
    print(f'      涵蓋 {len(vis)} 個 link 的視覺件（含輪、滾輪、lidar、相機）')
    if not col:
        print('    !! 未偵測到任何碰撞幾何', file=sys.stderr)
    if col:
        clo, chi = envelope(col)
        print(f'    【整機碰撞 AABB】({chi[0]-clo[0]:.4f}, {chi[1]-clo[1]:.4f}, '
              f'{chi[2]-clo[2]:.4f}) m，z 範圍 [{clo[2]:+.4f}, {chi[2]:+.4f}]')
        print(f'      只有 {len(col)} 個 link 有碰撞體（URDF 亦為 7 個）：')
        for n in sorted(col):
            a_, b_ = col[n]
            print(f'        {n:32s} x[{a_[0]:+.4f},{b_[0]:+.4f}] '
                  f'y[{a_[1]:+.4f},{b_[1]:+.4f}] z[{a_[2]:+.4f},{b_[2]:+.4f}]')
        print(f'      → 視覺與碰撞是不同範圍，輪子/感測器突出屬視覺件，不是尺寸錯誤')
    # base_link 的碰撞圓柱單獨對照 URDF，不與整機混談
    if 'base_link' in col:
        bl, bh = col['base_link']
        print(f'\n    【base_link 碰撞體單獨 AABB】'
              f'({bh[0]-bl[0]:.4f}, {bh[1]-bl[1]:.4f}, {bh[2]-bl[2]:.4f}) m')
        print(f'      z 範圍 [{bl[2]:+.4f}, {bh[2]:+.4f}]')
        print(f'      對照 URDF：圓柱半徑 0.300（直徑 0.600）、z 0.000–0.280')
        dx = abs((bh[0]-bl[0]) - 0.600); dz = abs((bh[2]-bl[2]) - 0.280)
        print(f'      直徑差 {dx*1000:+.2f} mm、高度差 {dz*1000:+.2f} mm')
    R = 0.300
    # AABB 角點距離對圓柱會高估到 0.424，這裡報 x/y 方向的半寬，才和半徑可比
    beyond = sorted(((n, max(abs(v[0][0]), abs(v[1][0]),
                             abs(v[0][1]), abs(v[1][1])))
                     for n, v in vis.items()), key=lambda o: -o[1])[:5]
    print(f'\n    視覺件 x/y 最大半寬前 5（底盤圓柱 R={R:.3f}）：'
          f'{[(n, round(r, 4)) for n, r in beyond]}')
    print(f'    最低視覺件 z = {vlo[2]*1000:+.1f} mm（接地面為 z=0）')

    print(f'    根部位姿 p = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})')
    print(f'    根部姿態 q (w,x,y,z) = ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})')
    print(f'    速度命令將施加於：{target}')

    print(f'\n  ── Gate 0：關節清單 ──')
    jt = {}
    for pr in stage.Traverse():
        if not str(pr.GetPath()).startswith(prim):
            continue
        if pr.IsA(UsdPhysics.Joint):
            jt.setdefault(pr.GetTypeName(), []).append(pr.GetName())
    total = sum(len(v) for v in jt.values())
    for k in sorted(jt):
        print(f'    {k}: {len(jt[k])} 個')
    print(f'    合計 {total} 個關節（URDF 92 個，全部 fixed）')
    print(f'    差 {92-total} 個：URDF 中 10 個無視覺也無碰撞的座標框，'
          f'除根 base_footprint 外皆被匯入器併除（剛體 84 vs URDF link 93 同理）')
    try:
        dof = art.num_dof
        print(f'    articulation 可動自由度 num_dof = {dof}'
              f'（全 fixed → 0，底盤僅有浮動基座 6 DOF）')
    except Exception as e:
        dof = None
        print(f'    num_dof 讀取失敗：{e}')
    print(f'\n    base_link 在 base_footprint 上方 0.050（URDF）；'
          f'接地由四顆半徑 0.05 的零摩擦球承擔，輪與滾輪為純視覺')

    rec = dict(gate=0,
               visual_aabb=[vlo.tolist(), vhi.tolist()],
               visual_size=(vhi - vlo).tolist(),
               collision_aabb=([clo.tolist(), chi.tolist()] if col else None),
               collision_size=((chi - clo).tolist() if col else None),
               collision_links={n: [col[n][0].tolist(), col[n][1].tolist()]
                                for n in sorted(col)},
               root_pos=[float(x) for x in p],
               root_quat=[float(x) for x in q],
               rigid_bodies=bodies, articulation_roots=arts,
               command_prim=target, joints={k: len(v) for k, v in jt.items()},
               num_dof=dof, links=sorted(vis))

    if a.gate == 1:
        rec['gate'] = 1
        rec.update(drive(world, art))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    path = (f'{a.out}_gate{a.gate}.json' if a.gate == 0 else
            f'{a.out}_gate1_{a.apply}_{a.stop}.json')
    json.dump(rec, open(path, 'w'), ensure_ascii=False)
    print(f'\n  已寫入 {path}')
    return 0


def drive(world, art) -> dict:
    """Forward, lateral, rotation, then the stop pair."""
    import numpy as np
    steps_per_cmd = max(1, int(round((1.0 / a.rate) / a.dt)))
    log = []

    def apply(v):
        art.set_linear_velocity(np.array([v[0], v[1], 0.0], dtype=np.float32))
        art.set_angular_velocity(np.array([0.0, 0.0, v[2]], dtype=np.float32))

    def run(v, secs, hold: bool, tag: str, touch: bool = True):
        """touch=False means the API is never called in this segment at all."""
        n = int(secs / a.dt)
        applied = False
        for k in range(n):
            if touch:
                if hold and k % steps_per_cmd == 0:
                    apply(v)
                elif (not hold) and not applied:
                    apply(v); applied = True
            world.step(render=False)
            if k % steps_per_cmd == 0:
                p, q = art.get_world_pose()
                lv = art.get_linear_velocity()
                av = art.get_angular_velocity()
                # yaw from the (w,x,y,z) quaternion: the rotation segment moves
                # almost nothing in x/y, so a translation-only metric would
                # report a stopped robot while it is still turning
                yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                       1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))
                log.append(dict(tag=tag, t=k * a.dt, x=float(p[0]),
                                y=float(p[1]), z=float(p[2]), yaw=yaw,
                                vx=float(lv[0]), vy=float(lv[1]),
                                wz=float(av[2]),
                                cmd=[float(c) for c in v]))

    print(f'\n  ── Gate 1：直接速度  施加={a.apply}  結束={a.stop}  '
          f'命令率 {a.rate:.0f} Hz  物理步 {a.dt*1000:.1f} ms ──')
    hold = a.apply == 'hold'
    for tag, v in (('forward', (a.vx, 0.0, 0.0)),
                   ('lateral', (0.0, a.vx, 0.0)),
                   ('yaw', (0.0, 0.0, 0.30))):
        run(v, a.drive_s, hold, tag)
        # stop segment: 'zero' commands zero at the same cadence the motion
        # used; 'silence' never calls the API again for the whole window.
        run((0.0, 0.0, 0.0), a.watch_s,
            hold if a.stop == 'zero' else False,
            f'{a.stop}_{tag}', touch=(a.stop == 'zero'))

    def unwrap(a_):
        return np.unwrap(np.asarray(a_))

    def report(tag, label):
        seg = [r for r in log if r['tag'] == tag]
        if len(seg) < 5:
            return None
        t = np.array([r['t'] for r in seg])
        rot = tag.endswith('yaw')
        if rot:
            th = unwrap([r['yaw'] for r in seg])
            d = np.abs(np.diff(th)) / np.diff(t)
            unit, k = 'rad/s', 1.0
            travel = abs(th[-1] - th[0])
            cmd = abs(seg[0]['cmd'][2])
        else:
            x = np.array([r['x'] for r in seg])
            y = np.array([r['y'] for r in seg])
            d = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
            unit, k = 'm/s', 1.0
            travel = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
            cmd = abs(seg[0]['cmd'][0] or seg[0]['cmd'][1])
        return dict(tag=tag, unit=unit, cmd=cmd, travel=travel,
                    med=float(np.median(d[len(d) // 2:])),
                    med_all=float(np.median(d)), mx=float(d.max()))

    res = {}
    for tag in ('forward', 'lateral', 'yaw'):
        r = report(tag, '驅動')
        if not r:
            continue
        res[tag] = r
        ratio = (r['med'] / r['cmd'] * 100.0) if r['cmd'] else float('nan')
        print(f"    {tag:8} 穩態 {r['med']:.4f} {r['unit']}"
              f"   命令 {r['cmd']:.4f}   追蹤率 {ratio:.1f}%")
    for tag in [f'{a.stop}_{t}' for t in ('forward', 'lateral', 'yaw')]:
        r = report(tag, '停止')
        if not r:
            continue
        res[tag] = r
        sc = 1000.0
        print(f"    {tag:16} 停止後 中位 {r['med_all']*sc:.3f}"
              f"   最大 {r['mx']*sc:.3f}  (m{r['unit'][0]}/s)"
              f"   期間變化 {r['travel']*sc:.2f}")

    return dict(apply=a.apply, stop=a.stop, rate=a.rate, dt=a.dt,
                summary=res, log=log)


try:
    code = main()
finally:
    sim_app.close()
raise SystemExit(code)
