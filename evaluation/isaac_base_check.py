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
    $ISAAC evaluation/isaac_base_check.py --gate 1 --mode hold
    $ISAAC evaluation/isaac_base_check.py --gate 1 --mode once
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
ap.add_argument('--mode', default='hold', choices=['hold', 'once'],
                help="'hold' re-applies the command every control tick, "
                     "'once' applies it a single time and then stops")
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
from isaacsim.asset.importer.urdf import _urdf                    # noqa: E402


def import_chassis(path: str) -> str:
    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = False
    cfg.fix_base = False
    cfg.make_default_prim = True
    cfg.self_collision = False
    cfg.distance_scale = 1.0
    cfg.density = 0.0                 # keep the URDF's own inertials
    iface = _urdf.acquire_urdf_interface()
    model = iface.parse_urdf(os.path.dirname(path), os.path.basename(path), cfg)
    prim = iface.import_robot(os.path.dirname(path), os.path.basename(path),
                              model, cfg, "")
    return prim


def aabb(prim_path: str):
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
    return compute_aabb(create_bbox_cache(), prim_path=prim_path,
                        include_children=True)


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
    world.reset()
    for _ in range(50):
        world.step(render=False)

    lo, hi = np.array(aabb(prim)[:3]), np.array(aabb(prim)[3:])
    size = hi - lo
    from isaacsim.core.prims import SingleArticulation
    art = SingleArticulation(prim_path=prim, name='base')
    art.initialize()
    p, q = art.get_world_pose()

    print(f'\n  ── Gate 0：幾何 ──')
    print(f'    AABB 尺寸 (x,y,z) = ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}) m')
    print(f'    AABB z 範圍 = [{lo[2]:+.4f}, {hi[2]:+.4f}]  → 離地 {lo[2]*1000:+.1f} mm')
    print(f'    根部位姿 p = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})')
    print(f'    根部姿態 q (w,x,y,z) = ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})')
    print(f'    關節數 {art.num_dof}；關節名 {list(art.dof_names)[:8]}')
    print(f'\n    對照 URDF（Gazebo 端實測）：底盤 collision 圓柱半徑 0.300、'
          f'z 0.000–0.280（base_link）；base_link 在 base_footprint 上方 0.050')

    rec = dict(gate=0, aabb_lo=lo.tolist(), aabb_hi=hi.tolist(),
               size=size.tolist(), root_pos=[float(x) for x in p],
               root_quat=[float(x) for x in q], num_dof=int(art.num_dof),
               dof_names=list(art.dof_names))

    if a.gate == 1:
        rec['gate'] = 1
        rec.update(drive(world, art))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    path = f'{a.out}_gate{a.gate}.json'
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

    def run(v, secs, hold: bool, tag: str):
        n = int(secs / a.dt)
        applied = False
        for k in range(n):
            if hold and k % steps_per_cmd == 0:
                apply(v)
            elif (not hold) and not applied:
                apply(v); applied = True
            world.step(render=False)
            if k % steps_per_cmd == 0:
                p, _ = art.get_world_pose()
                lv = art.get_linear_velocity()
                log.append(dict(tag=tag, t=k * a.dt, x=float(p[0]),
                                y=float(p[1]), z=float(p[2]),
                                vx=float(lv[0]), vy=float(lv[1]),
                                cmd=[float(c) for c in v]))

    print(f'\n  ── Gate 1：直接速度（{a.mode}，命令率 {a.rate:.0f} Hz，'
          f'物理步 {a.dt*1000:.1f} ms）──')
    hold = a.mode == 'hold'
    run((a.vx, 0.0, 0.0), a.drive_s, hold, 'forward')
    run((0.0, 0.0, 0.0), a.watch_s, hold, 'zero' if hold else 'silence')
    run((0.0, a.vx, 0.0), a.drive_s, hold, 'lateral')
    run((0.0, 0.0, 0.0), a.watch_s, hold, 'zero2' if hold else 'silence2')
    run((0.0, 0.0, 0.30), a.drive_s, hold, 'yaw')
    run((0.0, 0.0, 0.0), a.watch_s, hold, 'zero3' if hold else 'silence3')

    for tag in ('forward', 'lateral', 'yaw'):
        seg = [r for r in log if r['tag'] == tag]
        if len(seg) < 5:
            continue
        t = np.array([r['t'] for r in seg])
        x = np.array([r['x'] for r in seg]); y = np.array([r['y'] for r in seg])
        d = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
        half = d[len(d) // 2:]
        print(f'    {tag:8} 穩態速度 {np.median(half):.4f} m/s'
              f'   命令 {abs(seg[0]["cmd"][0] or seg[0]["cmd"][1]):.4f}')
    for tag in ('zero', 'zero2', 'zero3', 'silence', 'silence2', 'silence3'):
        seg = [r for r in log if r['tag'] == tag]
        if len(seg) < 5:
            continue
        t = np.array([r['t'] for r in seg])
        x = np.array([r['x'] for r in seg]); y = np.array([r['y'] for r in seg])
        d = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
        print(f'    {tag:8} 停止後速度 中位 {np.median(d)*1000:.3f} mm/s'
              f'   最大 {d.max()*1000:.3f}   期間位移 '
              f'{np.hypot(x[-1]-x[0], y[-1]-y[0])*1000:.2f} mm')
    return dict(mode=a.mode, rate=a.rate, dt=a.dt, log=log)


try:
    code = main()
finally:
    sim_app.close()
raise SystemExit(code)
