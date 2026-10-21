#!/usr/bin/env python3
"""K1：局部通行門的被動物理讀回與短測（**只有門＋地面，不放機器人**；外力由腳本施加，不是機器人開門成果）。

事前登錄（Codex 20261021 裁定；數值不依結果調整；登錄計畫 PLAN 與判定器共用 door_k1_plan.py）：
  單位：公尺、公斤、秒；角度內部 rad、紀錄另附度。physics_dt 0.01 s；重力 −9.81 m/s²。
  施力（door_k1_force.py）：世界座標施力點＝把手桿心 p_H（每步由實測門扇姿態算出），只施力 F 沿開門切向，不另加力矩；
        每列記錄實際送入 API 的施力點、F、額外力矩（0），以及由它們重算的鉸鏈合力矩 τ_open。
  紀錄列：t_pre 的狀態與施力 → world.step() → t_post 的狀態與門扇對兩側牆接觸力（contact force matrix）。
  每項 world.reset() 獨立初始化；S5 只在開始前設 45° 並歸零速度；重設後核對實際 q0 與速度；睡眠門檻逐項設 0 並逐項記錄成敗。
  牆鐘上限：每項 120 s、全部 600 s。每 100 步存檔；TERM／INT 時保存目前短測的部分紀錄並標 interrupted。
判定在 door_k1_check.py。

    RUN_ID=door_k1_1 ROS_DOMAIN_ID=96 RUNNER=evaluation/run_door_k1.sh bash evaluation/run_guarded_v2.sh
"""
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
import door_asset as DA                                          # noqa: E402
import door_k1_force as KF                                       # noqa: E402
import door_k1_plan as KP                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--spec', default=os.path.join(WS, KP.SPEC_REL))
ap.add_argument('--out', required=True)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
OUT = os.path.join(a.out, 'door_k1.json')
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} 已存在，不覆寫')
D = DA.load(a.spec)
PLAN = KP.plan(D)
STOP = {'sig': None}
rep = {'schema': 'door_k1/2', 'spec': os.path.relpath(a.spec, WS), 'spec_sha256': hashlib.sha256(open(a.spec, 'rb').read()).hexdigest(),
       'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'physics_dt': KP.DT, 'units': {'length': 'm', 'mass': 'kg'},
       'plan': PLAN, 'force_semantics': 'world positions = p_H, forces only, torques None', 'wall_limits_s': {'per_test': KP.WALL_TEST_S, 'total': KP.WALL_TOTAL_S},
       'tests': {}, 'complete': False}


def save():
    tmp = OUT + '.tmp'
    json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, OUT)


def _on_sig(s, _f):
    STOP['sig'] = signal.Signals(s).name


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)

from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True, 'limit_cpu_threads': a.cpu_threads})
signal.signal(signal.SIGTERM, _on_sig)                             # Kit 啟動後再掛一次，避免被覆蓋
signal.signal(signal.SIGINT, _on_sig)
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
    world = World(stage_units_in_meters=1.0, physics_dt=KP.DT, rendering_dt=KP.DT)
    UsdPhysics.SetStageKilogramsPerUnit(world.stage, 1.0)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    rep['readback_usd'] = DA.build_usd(world.stage, D)
    leaf_path = rep['readback_usd']['leaf_prim']
    leaf = RigidPrim(prim_paths_expr=leaf_path, name='leaf_view', track_contact_forces=True, max_contact_count=256,
                     prepare_contact_sensors=True, contact_filter_prim_paths_expr=list(KP.WALLS))
    world.reset()
    leaf.initialize()
    rb = {}
    for k, fn in (('masses', leaf.get_masses), ('inertias', leaf.get_inertias), ('coms', leaf.get_coms)):
        try:
            v = fn()
            rb[k] = [np.asarray(x).tolist() for x in v] if isinstance(v, tuple) else np.asarray(v).tolist()
        except Exception as e:                           # noqa: BLE001
            rb[k] = {'error': f'{type(e).__name__}: {e}'}
    rep['readback_physx'] = rb
    hinge = np.array([D['placement']['pose']['x'], D['placement']['pose']['y'], 0.0])
    Hd = D['handle']

    def state():
        p, o = leaf.get_world_poses()
        R = quat_wxyz_to_R(o[0])
        q = -math.atan2(R[1, 0], R[0, 0])
        v = leaf.get_velocities()[0]
        pH = np.array(p[0]) + R @ np.array([Hd['from_hinge'], -Hd['standoff'], Hd['height']])
        return q, -float(v[5]), pH

    def contacts():
        cm = leaf.get_contact_force_matrix(dt=KP.DT)
        return None if cm is None else [float(np.linalg.norm(np.asarray(cm)[0][i])) for i in range(len(KP.WALLS))]

    for name, P in PLAN.items():
        if STOP['sig']:
            break
        if time.monotonic() - t_wall0 > KP.WALL_TOTAL_S:
            rep['aborted'] = f'總牆鐘超過 {KP.WALL_TOTAL_S} s（{name} 前）'
            break
        world.reset()
        leaf.initialize()
        T = {'plan': P, 'status': 'running', 'rows': []}
        try:
            leaf.set_sleep_thresholds(np.zeros((1,), dtype=np.float32))
            T['sleep_threshold_zero'] = True
        except Exception as e:                           # noqa: BLE001
            T['sleep_threshold_zero'] = f'failed: {type(e).__name__}: {e}'
        if P['q0_deg'] != 0.0:
            q0 = math.radians(P['q0_deg'])
            leaf.set_world_poses(positions=np.array([hinge]), orientations=np.array([[math.cos(-q0 / 2), 0.0, 0.0, math.sin(-q0 / 2)]]))
            leaf.set_velocities(np.zeros((1, 6), dtype=np.float32))
        q_a, qd_a, _ = state()
        T['q0_actual_deg'], T['qdot0_actual'] = math.degrees(q_a), qd_a
        rep['tests'][name] = T
        t0w = time.monotonic()
        n_force, n_total = KP.n_steps(P)
        stopped = None
        for k in range(n_total):
            if STOP['sig']:
                T['status'] = f'interrupted:{STOP["sig"]}'
                break
            if time.monotonic() - t0w > KP.WALL_TEST_S:
                T['status'] = 'wall_timeout'
                break
            q, qd, pH = state()
            applying = k < n_force and P['force_n'] != 0.0 and stopped is None
            if applying and P.get('stop_force_at_deg') is not None and math.degrees(q) >= P['stop_force_at_deg']:
                stopped, applying = k * KP.DT, False
            if applying:
                cmd = KF.handle_force(pH, hinge, KP.A_OPEN, P['force_n'])
                KF.apply(leaf, cmd)
            else:
                cmd = {'p_apply': pH, 'F': np.zeros(3), 'tau_extra': np.zeros(3), 'tau_open': 0.0}
            world.step(render=False)
            q2, qd2, _ = state()
            T['rows'].append({'k': k, 't_pre': round(k * KP.DT, 6), 'q_pre_deg': math.degrees(q), 'qdot_pre': qd, 'applying': bool(applying),
                              'p_apply': cmd['p_apply'].tolist(), 'F': cmd['F'].tolist(), 'tau_extra': cmd['tau_extra'].tolist(),
                              'tau_open_nm': cmd['tau_open'], 't_post': round((k + 1) * KP.DT, 6), 'q_post_deg': math.degrees(q2),
                              'qdot_post': qd2, 'contact_walls_n': contacts()})
            if k % 100 == 99:
                save()
        else:
            T['status'] = 'done'
        T['force_stopped_at_s'] = stopped
        T['wall_s'] = round(time.monotonic() - t0w, 2)
        save()
        print(f'[k1] {name}: {T["status"]}，q0 {T["q0_actual_deg"]:.3f}°，{len(T["rows"])} 步', flush=True)
        if T['status'] != 'done':
            break
    rep['complete'] = (all(n in rep['tests'] and rep['tests'][n]['status'] == 'done' for n in PLAN)
                       and not rep.get('aborted') and not STOP['sig'])
    if STOP['sig']:
        rep['interrupted'] = STOP['sig']
    save()


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except Exception as e:                               # noqa: BLE001
        rep['fatal'] = f'{type(e).__name__}: {e}'
        rep['traceback'] = traceback.format_exc()
        rep['complete'] = False
        save()
        rc = 1
    finally:
        try:
            sim_app.close()
        except Exception:                                # noqa: BLE001
            pass
    sys.exit(0 if rep.get('complete') else (rc or 2))
