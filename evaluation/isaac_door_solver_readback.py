#!/usr/bin/env python3
"""門場景物理求解設定讀回（Codex pb1_4 回覆第 1 步）：與執行端同一建法（World physics_dt 0.01＋door_asset_v3），
讀 PhysicsScene 的 PhysxSceneAPI（solverType、min／max position／velocity iteration、stabilization）與門扇 PhysxRigidBodyAPI
（solverPositionIterationCount／solverVelocityIterationCount；authored 或 fallback）。只讀，不步進控制。

    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_door_solver_readback.py [--door-vel-iters N]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--door-asset', default=os.path.join(WS, 'src/my_omnibot_description/config/door_local_v3.yaml'))
ap.add_argument('--door-vel-iters', type=int, default=0)
a = ap.parse_args()
from isaacsim import SimulationApp                               # noqa: E402
app = SimulationApp({'headless': True})
from isaacsim.core.api import World                              # noqa: E402
from pxr import PhysxSchema, UsdPhysics                          # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402

w = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
st = w.stage
DA3.build_usd(st, DA3.load(a.door_asset), root='/World/door')
leaf = st.GetPrimAtPath('/World/door/leaf')
if a.door_vel_iters:
    PhysxSchema.PhysxRigidBodyAPI(leaf).CreateSolverVelocityIterationCountAttr().Set(int(a.door_vel_iters))
w.reset()
for _ in range(3):
    w.step(render=False)


def attr(api, name):
    at = getattr(api, f'Get{name}Attr')()
    return {'value': at.Get(), 'authored': at.HasAuthoredValue()}


out = {'scenes': [], 'leaf': {}}
for p in st.Traverse():
    if p.IsA(UsdPhysics.Scene):
        s = PhysxSchema.PhysxSceneAPI(p) if p.HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(p)
        out['scenes'].append({'path': str(p.GetPath()), **{n: attr(s, n) for n in ('SolverType', 'MinPositionIterationCount', 'MaxPositionIterationCount',
                                                                                  'MinVelocityIterationCount', 'MaxVelocityIterationCount', 'EnableStabilization')}})
rb = PhysxSchema.PhysxRigidBodyAPI(leaf)
out['leaf'] = {n: attr(rb, n) for n in ('SolverPositionIterationCount', 'SolverVelocityIterationCount', 'AngularDamping')}
out['leaf']['has_api'] = leaf.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
print('READBACK ' + json.dumps(out, default=str), flush=True)
app.close()
