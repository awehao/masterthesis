#!/usr/bin/env python3
"""K1：局部通行門的被動物理讀回與短測（**只有門＋地面，不放機器人**；外力由腳本施加，不是機器人開門成果）。

事前登錄（Codex 20261021 裁定；數值不依結果調整）：
  單位：公尺、公斤、秒；角度內部用 rad、紀錄另附度。physics_dt 0.01 s；重力 −9.81 m/s²。
  施力：沿開門切向 t̂(q) = â×(p_H−c)/‖·‖（â = 開門軸 (0,0,−1)），作用點＝把手桿心（隨實測門姿態每步更新）；
        以等效力旋量施加：力施於門扇質心、另加力矩 τ = (p_H − p_COM) × F（不依賴 apply_*_at_pos 的位置座標語意）。
        每步記錄實際施加的 F、τ 與開門力矩 â·[(p_H−c)×F]（須 > 0 才是拉開；反推時 < 0）。
  睡眠：門扇 view 設睡眠門檻 0（記錄成功與否）；若失敗，速度歸零段落不得解讀為阻尼衰減。
  各短測**獨立初始化**（world.reset()；S5 另以初始化設定非零門角，測試過程中不重設門角），各有最長施力時間、總觀察時間與牆鐘上限：
    S1 靜置：q0 = 0，無外力 10 s。
    S2 方向：q0 = 0，切向 10 N × 0.5 s 後撤力，再觀察 5 s。
    S3 反推：q0 = 0，反切向 10 N × 1.0 s，再觀察 1 s；記門扇對兩側牆的接觸力（區分限位與門框）。
    S4 上限：q0 = 0，切向 20 N 至多 6.0 s（到達上限即停止施力），再觀察 2 s。
    S5 非零角被動：初始化 q0 = 45°（只在開始前設定姿態並歸零速度），無外力 10 s。
  牆鐘上限：每項 120 s、全部 600 s；超過即中止並保存已得資料。
判定不在本檔（door_k1_check.py 讀本檔輸出另判）。

    RUN_ID=door_k1_1 ROS_DOMAIN_ID=96 RUNNER=evaluation/run_door_k1.sh bash evaluation/run_guarded_v2.sh
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
import door_asset as DA                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--spec', default=os.path.join(WS, 'src/my_omnibot_description/config/door_local_v1.yaml'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

PLAN = {'S1': {'q0_deg': 0.0, 'force_n': 0.0, 'force_s': 0.0, 'observe_s': 10.0},
        'S2': {'q0_deg': 0.0, 'force_n': 10.0, 'force_s': 0.5, 'observe_s': 5.0},
        'S3': {'q0_deg': 0.0, 'force_n': -10.0, 'force_s': 1.0, 'observe_s': 1.0},
        'S4': {'q0_deg': 0.0, 'force_n': 20.0, 'force_s': 6.0, 'observe_s': 2.0, 'stop_force_at_deg': None},
        'S5': {'q0_deg': 45.0, 'force_n': 0.0, 'force_s': 0.0, 'observe_s': 10.0}}
WALL_TEST_S, WALL_TOTAL_S = 120.0, 600.0
A_OPEN = np.array([0.0, 0.0, -1.0])

os.makedirs(a.out, exist_ok=True)
OUT = os.path.join(a.out, 'door_k1.json')
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} 已存在，不覆寫')
D = DA.load(a.spec)
PLAN['S4']['stop_force_at_deg'] = float(D['leaf']['limit_q_deg'][1]) - 0.5
rep = {'schema': 'door_k1/1', 'spec': os.path.relpath(a.spec, WS), 'spec_sha256': hashlib.sha256(open(a.spec, 'rb').read()).hexdigest(),
       'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'physics_dt': a.physics_dt, 'units': {'length': 'm', 'mass': 'kg'},
       'plan': PLAN, 'wall_limits_s': {'per_test': WALL_TEST_S, 'total': WALL_TOTAL_S}, 'tests': {}, 'complete': False}


def save():
    tmp = OUT + '.tmp'
    json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)


from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True, 'limit_cpu_threads': a.cpu_threads})
from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.prims import RigidPrim                          # noqa: E402
from pxr import UsdPhysics                                         # noqa: E402


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    t_wall0 = time.monotonic()
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt, rendering_dt=a.physics_dt)
    UsdPhysics.SetStageKilogramsPerUnit(world.stage, 1.0)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    authored = DA.build_usd(world.stage, D)
    rep['readback_usd'] = authored
    leaf_path = authored['leaf_prim']
    walls = ['/World/door/frame/wall_hinge_side', '/World/door/frame/wall_latch_side']
    leaf = RigidPrim(prim_paths_expr=leaf_path, name='leaf_view', track_contact_forces=True, max_contact_count=256,
                     prepare_contact_sensors=True, contact_filter_prim_paths_expr=walls)
    world.reset()
    leaf.initialize()
    try:
        leaf.set_sleep_thresholds(np.zeros((1,), dtype=np.float32))
        rep['sleep_threshold_zero'] = True
    except Exception as e:                               # noqa: BLE001
        rep['sleep_threshold_zero'] = f'failed: {type(e).__name__}: {e}'
    # PhysX 端讀回（view 解析值）
    try:
        rep['readback_physx'] = {'masses': leaf.get_masses().tolist(), 'inertias': leaf.get_inertias().tolist(),
                                 'coms': [x.tolist() for x in leaf.get_coms()]}
    except Exception as e:                               # noqa: BLE001
        rep['readback_physx'] = {'error': f'{type(e).__name__}: {e}'}
    hinge = np.array([D['placement']['pose']['x'], D['placement']['pose']['y'], 0.0])
    Hd = D['handle']
    m, com_local, _ = DA.leaf_inertia(D)

    def state():
        p, o = leaf.get_world_poses()
        R = quat_wxyz_to_R(o[0])
        q = -math.atan2(R[1, 0], R[0, 0])               # 門扇繞 +z 轉 −q（placement yaw 0）
        v = leaf.get_velocities()[0]
        pH = np.array(p[0]) + R @ np.array([Hd['from_hinge'], -Hd['standoff'], Hd['height']])
        pC = np.array(p[0]) + R @ np.array(com_local)
        return q, -float(v[5]), pH, pC, R

    def set_q0(qd):
        q = math.radians(qd)
        w, z = math.cos(-q / 2), math.sin(-q / 2)
        leaf.set_world_poses(positions=np.array([hinge]), orientations=np.array([[w, 0.0, 0.0, z]]))
        leaf.set_velocities(np.zeros((1, 6), dtype=np.float32))

    for name, P in PLAN.items():
        if time.monotonic() - t_wall0 > WALL_TOTAL_S:
            rep['aborted'] = f'總牆鐘超過 {WALL_TOTAL_S} s（{name} 前）'
            break
        world.reset()
        leaf.initialize()
        try:
            leaf.set_sleep_thresholds(np.zeros((1,), dtype=np.float32))
        except Exception:                                # noqa: BLE001
            pass
        if P['q0_deg'] != 0.0:
            set_q0(P['q0_deg'])
        t0w = time.monotonic()
        rows, n_force = [], int(round(P['force_s'] / a.physics_dt))
        n_total = n_force + int(round(P['observe_s'] / a.physics_dt))
        force_stopped_at = None
        q_init, _, _, _, _ = state()
        for k in range(n_total):
            if time.monotonic() - t0w > WALL_TEST_S:
                rep.setdefault('wall_timeouts', []).append(name)
                break
            q, qd, pH, pC, R = state()
            F = np.zeros(3)
            tau = np.zeros(3)
            applying = k < n_force and P['force_n'] != 0.0 and force_stopped_at is None
            if applying and P.get('stop_force_at_deg') is not None and math.degrees(q) >= P['stop_force_at_deg']:
                force_stopped_at = k * a.physics_dt
                applying = False
            if applying:
                rr = pH - hinge
                t_hat = np.cross(A_OPEN, rr)
                t_hat /= np.linalg.norm(t_hat)
                F = P['force_n'] * t_hat
                tau = np.cross(pH - pC, F)
                leaf.apply_forces_and_torques_at_pos(forces=F[None, :].astype(np.float32), torques=tau[None, :].astype(np.float32),
                                                     positions=None, is_global=True)
            world.step(render=False)
            cm = leaf.get_contact_force_matrix(dt=a.physics_dt)
            cw = [float(np.linalg.norm(cm[0][i])) for i in range(len(walls))] if cm is not None else None
            rows.append({'t': round(k * a.physics_dt, 4), 'q_rad': q, 'q_deg': math.degrees(q), 'qdot_rad_s': qd, 'F': F.tolist(), 'tau': tau.tolist(),
                         'open_torque_nm': float(A_OPEN @ np.cross(pH - hinge, F)), 'applying': bool(applying), 'contact_walls_n': cw})
        qf, qdf, _, _, _ = state()
        rep['tests'][name] = {'plan': P, 'q_init_deg': math.degrees(q_init), 'q_final_deg': math.degrees(qf), 'force_stopped_at_s': force_stopped_at,
                              'n_steps': len(rows), 'wall_s': round(time.monotonic() - t0w, 2), 'rows': rows}
        save()
        print(f'[k1] {name}: q {math.degrees(q_init):.3f}° → {math.degrees(qf):.3f}°，{len(rows)} 步，{time.monotonic()-t0w:.1f} s', flush=True)
    rep['complete'] = all(n in rep['tests'] for n in PLAN) and not rep.get('wall_timeouts') and not rep.get('aborted')
    save()


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except Exception as e:                               # noqa: BLE001
        rep['fatal'] = f'{type(e).__name__}: {e}'
        rep['traceback'] = traceback.format_exc()
        save()
        rc = 1
    finally:
        sim_app.close()
    sys.exit(rc if rep.get('complete') else (rc or 2))
