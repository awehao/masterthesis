#!/usr/bin/env python3
"""瓶子 S0 前置冒煙：**只放桌子與動態瓶子**（不載入機器人、不啟動控制器）。Codex 20261031 核可一次。
依 Howard：同一程序開 Isaac 視窗並持續更新 viewport（不錄影、不逐步寫圖、不另開世界）；物理 dt 0.01、TGS 讀回、每次迭代施加外力（同既有實跑）。
量測（每物理步）：瓶位置、傾角、線速度、瓶↔桌面接觸法向力（接觸過濾對）、瓶底對桌面的幾何穿透／分離、資料有效性；另讀回實際物理設定。
支撐力不是噪聲；本步不能驗證機器人禁止接觸門檻或手指抓持管線。失敗即報，不自動補跑。
    $ISAAC_PY evaluation/isaac_bottle_smoke.py --asset src/my_omnibot_description/config/bottle_s0.yaml --out <dir> [--sim-s 2.0] [--hold-s 3.0]
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--asset', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--sim-s', type=float, default=2.0)
ap.add_argument('--hold-s', type=float, default=3.0, help='量測結束後視窗保留秒數（牆鐘；只渲染、不推進物理）')
ap.add_argument('--headless', default='false')
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isaac_common import bind_friction                               # noqa: E402


def main():
    S = yaml.safe_load(open(a.asset))
    T, B, PH = S['table'], S['bottle'], S['physics']
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=PH['dt'], rendering_dt=PH['dt'])
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from pxr import PhysxSchema, UsdPhysics
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[bottle] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    if PH.get('ext_forces_every_iter'):
        sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    # 桌子（固定）
    top_c = [0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]
    top = FixedCuboid(prim_path='/World/table/top', name='table_top', position=np.array(top_c),
                      scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.55, 0.42, 0.30]))
    hz = T['top_z'] - T['thick']
    legs = []
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            legs.append(FixedCuboid(prim_path=f'/World/table/leg_{nm}_{"p" if sx > 0 else "m"}', name=f'leg_{nm}_{sx}',
                                    position=np.array([lx, ly, hz / 2]), scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.4, 0.3, 0.2])))
    # 瓶子（動態，立於桌面）
    bz = T['top_z'] + B['height'] / 2
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([B['center_xy'][0], B['center_xy'][1], bz]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    n_t = bind_friction(stage, [stage.GetPrimAtPath('/World/table/top')] + [stage.GetPrimAtPath(l.prim_path) for l in legs],
                        T['material']['static_friction'], T['material']['dynamic_friction'], T['material']['restitution'],
                        mat_path='/World/PhysicsMaterials/table')
    n_b = bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'],
                        B['material']['restitution'], mat_path='/World/PhysicsMaterials/bottle')
    pair = RigidPrim(prim_paths_expr='/World/bottle', name='bottle_pair', track_contact_forces=True, max_contact_count=64,
                     prepare_contact_sensors=True, contact_filter_prim_paths_expr=['/World/table/top'])
    world.reset()
    # 讀回實際物理設定
    mapi = UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/bottle'))
    rb = {'mass_readback': float(bottle.get_mass()), 'mass_attr': float(mapi.GetMassAttr().Get()) if mapi else None}
    readback = dict(solver_type=str(sapi.GetSolverTypeAttr().Get()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()),
                    physics_dt=float(world.get_physics_dt()), table_bound=n_t, bottle_bound=n_b, **rb,
                    bottle_radius=float(bottle.get_radius()), bottle_height=float(bottle.get_height()))
    print(f'[bottle] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - PH['dt']) > 1e-12 or (PH.get('ext_forces_every_iter') and not readback['ext_forces_every_iter']):
        print('[bottle] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64
    rows = []
    n = int(round(a.sim_s / PH['dt']))
    p0 = None
    for k in range(n):
        world.step(render=True)
        t = (k + 1) * PH['dt']
        pos, quat = bottle.get_world_pose()
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)        # quat (w, x, y, z)
        if p0 is None:
            p0 = pos.copy()
        w, x, y, z = quat
        # 瓶軸（本體 z）在世界的方向 ⇒ 傾角
        axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, axis[2] / max(np.linalg.norm(axis), 1e-12)))))
        v = np.asarray(bottle.get_linear_velocity(), float)
        try:
            M = np.asarray(pair.get_contact_force_matrix(dt=PH['dt']), float).reshape(-1, 3)
            f = M.sum(axis=0)
            valid = bool(np.isfinite(f).all())
        except Exception as e:                                             # 讀不到 ⇒ 記無效，不當成零力
            f, valid = np.full(3, np.nan), False
            if k == 0:
                print(f'[bottle] 接觸力讀取失敗：{e!r}', flush=True)
        bottom = pos[2] - (B['height'] / 2) * axis[2] / max(np.linalg.norm(axis), 1e-12) - B['radius'] * math.sin(math.radians(tilt))
        gap = bottom - T['top_z']                                         # 正＝分離、負＝穿透（最低點近似）
        rows.append(dict(t=round(t, 4), x=pos[0], y=pos[1], z=pos[2], tilt_deg=tilt, v=float(np.linalg.norm(v)), fz=float(f[2]) if valid else None,
                         f=[float(c) for c in f] if valid else None, valid=valid, gap_mm=gap * 1e3))
    W = B['mass'] * 9.81
    P = np.array([[r['x'], r['y'], r['z']] for r in rows])
    fz = np.array([abs(r['fz']) for r in rows if r['valid'] and r['fz'] is not None])
    last = rows[int(len(rows) / 2):]
    summ = dict(n_steps=len(rows), n_valid=int(sum(r['valid'] for r in rows)),
                drift_xy_mm=float(np.max(np.hypot(P[:, 0] - p0[0], P[:, 1] - p0[1])) * 1e3), drift_z_mm=float((P[-1, 2] - p0[2]) * 1e3),
                tilt_max_deg=float(max(r['tilt_deg'] for r in rows)), v_max=float(max(r['v'] for r in rows)),
                gap_min_mm=float(min(r['gap_mm'] for r in rows)), gap_max_mm=float(max(r['gap_mm'] for r in rows)),
                support_fz_mean_second_half=float(np.mean([abs(r['fz']) for r in last if r['valid'] and r['fz'] is not None])) if fz.size else None,
                support_fz_std_second_half=float(np.std([abs(r['fz']) for r in last if r['valid'] and r['fz'] is not None])) if fz.size else None,
                weight_N=W, first_valid_step=next((i for i, r in enumerate(rows) if r['valid'] and r['fz'] is not None and abs(r['fz']) > 0), None))
    json.dump(dict(asset=S, readback=readback, summary=summ, rows=rows), open(os.path.join(a.out, 'bottle_smoke.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'[bottle] 摘要 {json.dumps(summ, ensure_ascii=False, default=float)}', flush=True)
    import time
    t_end = time.monotonic() + a.hold_s                                    # 視窗保留（只渲染）
    while time.monotonic() < t_end and sim_app.is_running():
        world.render()
    return 0


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        sim_app.close()
    sys.exit(rc)
