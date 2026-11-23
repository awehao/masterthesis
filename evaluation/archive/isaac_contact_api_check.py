#!/usr/bin/env python3
"""接觸／摩擦資料介面核對（Codex 20261102_041534 第 2 項：先驗證讀值再量測；最小場景，不跑任務、不載機器人）。
場景：地面＋桌面（FixedCuboid，頂 0.60 m）＋瓶（DynamicCylinder，資產尺寸與質量、材質 1.0／0.8，桌面同實跑綁定）；物理步長 0.01 s。
瓶上建 RigidPrim（track_contact_forces、max_contact_count 64、filter＝桌面）。兩段：
  A 靜置 1.5 s；B 每步對瓶施加世界 +x 0.5 N（小於 μmg≈1.96 N，應保持靜止）1.5 s。
每步讀 get_contact_force_matrix(dt=0.01)、get_contact_force_data(dt=0.01)、get_friction_data(dt=0.01)，另讀一次 dt 預設（=1）比較倍數。核：
  1 法向資料各點力×法向的和 vs 接觸矩陣（文件：和＝矩陣）；2 作用於瓶的總接觸力鉛直分量 ≈ m·g；3 B 段摩擦合力 ≈ −0.5 N（x）、A 段 ≈ 0；
  4 dt 預設值讀數 ≈ dt=0.01 讀數 × 0.01（即衝量）；5 數量、起始索引、有限值、容量。輸出 <out>/contact_api_check.json。
由 run_bottle_replay.sh 啟動（接受 --src／--out／--headless；--src 不使用）。
"""
import argparse
import json
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--src', default='')
ap.add_argument('--out', required=True)
ap.add_argument('--headless', default='false')
ap.add_argument('--asset', default=os.path.join(WS, 'src/my_omnibot_description/config/bottle_s0.yaml'))
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim                            # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
from isaac_common import bind_friction                               # noqa: E402

DT = 0.01


def agg_normal(d, k):
    """第 k 個（瓶）× filter 0 的法向資料：回傳 (Σ f_i·n_i, 點數, 有限)。"""
    f, p, n, dist, cnt, st = (np.asarray(x) for x in d)
    c, s = int(np.asarray(cnt).reshape(-1)[k] if np.asarray(cnt).ndim else cnt), int(np.asarray(st).reshape(-1)[k] if np.asarray(st).ndim else st)
    if c == 0:
        return np.zeros(3), 0, True
    F = (f[s:s + c].reshape(-1, 1) * n[s:s + c].reshape(-1, 3)).sum(axis=0)
    return F, c, bool(np.isfinite(F).all())


def agg_fric(d, k):
    t, p, cnt, st = (np.asarray(x) for x in d)
    c, s = int(np.asarray(cnt).reshape(-1)[k]), int(np.asarray(st).reshape(-1)[k])
    if c == 0:
        return np.zeros(3), 0, True
    F = t[s:s + c].reshape(-1, 3).sum(axis=0)
    return F, c, bool(np.isfinite(F).all())


def main():
    S = yaml.safe_load(open(a.asset))
    T, B = S['table'], S['bottle']
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    FixedCuboid(prim_path='/World/table/top', name='table_top', position=np.array([0.0, 0.0, T['top_z'] - T['thick'] / 2]), scale=np.array([0.8, 0.5, T['thick']]),
                color=np.array([0.55, 0.42, 0.30]))
    DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([0.0, 0.0, T['top_z'] + B['height'] / 2]), radius=B['radius'], height=B['height'], mass=B['mass'],
                    color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath('/World/table/top')], T['material']['static_friction'], T['material']['dynamic_friction'], T['material']['restitution'],
                  mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'], B['material']['restitution'],
                  mat_path='/World/PhysicsMaterials/bottle')
    v = RigidPrim(prim_paths_expr='/World/bottle', name='v_bottle', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True,
                  contact_filter_prim_paths_expr=['/World/table/top'])
    world.reset()
    v.initialize()
    mg = float(B['mass']) * 9.81
    rows, dt_ratio = [], []
    for k in range(300):
        seg = 'A' if k < 150 else 'B'
        if seg == 'B':
            v.apply_forces(np.array([[0.5, 0.0, 0.0]], np.float32), is_global=True)
        world.step(render=False)
        M = np.asarray(v.get_contact_force_matrix(dt=DT)).reshape(-1, 3)[0]
        dN = v.get_contact_force_data(dt=DT)
        dF = v.get_friction_data(dt=DT)
        Fn, cn, okn = agg_normal(dN, 0)
        Ff, cf, okf = agg_fric(dF, 0)
        if k in (100, 250):
            M1 = np.asarray(v.get_contact_force_matrix()).reshape(-1, 3)[0]
            Fn1, _, _ = agg_normal(v.get_contact_force_data(), 0)
            Ff1, _, _ = agg_fric(v.get_friction_data(), 0)
            dt_ratio.append(dict(step=k, matrix_ratio=float(np.linalg.norm(M1) / max(np.linalg.norm(M), 1e-12)), normal_ratio=float(np.linalg.norm(Fn1) / max(np.linalg.norm(Fn), 1e-12)),
                                 friction_ratio=None if np.linalg.norm(Ff) < 1e-9 else float(np.linalg.norm(Ff1) / np.linalg.norm(Ff))))
        p, _ = v.get_world_poses()
        rows.append(dict(k=k, seg=seg, matrix=M.tolist(), normal_sum=Fn.tolist(), n_normal=cn, friction_sum=Ff.tolist(), n_fric=cf, finite=bool(okn and okf and np.isfinite(M).all()),
                         pos=np.asarray(p).reshape(-1).tolist()))
    R = {s: [x for x in rows if x['seg'] == s and x['k'] % 150 >= 50] for s in 'AB'}               # 各段去掉前 0.5 s 收斂
    def med(key, s, i):
        return float(np.median([x[key][i] for x in R[s]]))
    out = dict(mg=mg, dt=DT,
               A=dict(matrix_z=med('matrix', 'A', 2), normal_z=med('normal_sum', 'A', 2), normal_vs_matrix=float(np.median([np.linalg.norm(np.array(x['normal_sum']) - np.array(x['matrix'])) for x in R['A']])),
                      friction=[med('friction_sum', 'A', i) for i in range(3)], n_normal=int(np.median([x['n_normal'] for x in R['A']])), n_fric=int(np.median([x['n_fric'] for x in R['A']]))),
               B=dict(matrix=[med('matrix', 'B', i) for i in range(3)], normal_z=med('normal_sum', 'B', 2), friction=[med('friction_sum', 'B', i) for i in range(3)],
                      drift_x_mm=float((rows[-1]['pos'][0] - rows[150]['pos'][0]) * 1e3), n_fric=int(np.median([x['n_fric'] for x in R['B']]))),
               dt_default_ratio=dt_ratio, all_finite=all(x['finite'] for x in rows), max_count=max(max(x['n_normal'], x['n_fric']) for x in rows),
               note='作用方向以讀值核對；摩擦錨點與法向接觸點是不同索引，不逐點配對')
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(os.path.join(a.out, 'contact_api_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('[ck] ' + json.dumps(out, ensure_ascii=False, default=float), flush=True)
    with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
        fh.write('0')
    return 0


rc = 99
try:
    rc = main()
except BaseException:
    import traceback
    print('[ck] 例外：' + traceback.format_exc().replace('\n', ' | '), flush=True)
finally:
    sim_app.close()
sys.exit(rc)
