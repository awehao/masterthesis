#!/usr/bin/env python3
"""機器人 articulation 求解迭代讀回（Codex 20261029 leader_2 回覆第 1 步；只讀）：與執行端同一匯入（import_urdf、fix_base False），world.reset 後讀
SingleArticulation 的 solver position／velocity iteration count 與根 prim 的 PhysxArticulationAPI（是否套用、authored）。

    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_robot_solver_readback.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from isaacsim import SimulationApp                               # noqa: E402
app = SimulationApp({'headless': True})
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from pxr import PhysxSchema, UsdPhysics                          # noqa: E402
from isaac_common import import_urdf                             # noqa: E402

w = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
st = w.stage
import_urdf(os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'), '/World/omni_bot', fix_base=False)
roots = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI) and str(p.GetPath()).startswith('/World/omni_bot')]
w.reset()
out = {'roots': []}
for r in roots:
    has = r.HasAPI(PhysxSchema.PhysxArticulationAPI)
    api = PhysxSchema.PhysxArticulationAPI(r)
    pa, va = api.GetSolverPositionIterationCountAttr(), api.GetSolverVelocityIterationCountAttr()
    out['roots'].append({'path': str(r.GetPath()), 'physx_articulation_api': has,
                         'pos_iter': {'value': pa.Get() if pa else None, 'authored': bool(pa and pa.HasAuthoredValue())},
                         'vel_iter': {'value': va.Get() if va else None, 'authored': bool(va and va.HasAuthoredValue())}})
art = SingleArticulation(prim_path=str(roots[0].GetPath()) if roots else '/World/omni_bot', name='rb')
art.initialize()
for nm in ('get_solver_position_iteration_count', 'get_solver_velocity_iteration_count'):
    try:
        out[nm] = int(getattr(art, nm)())
    except Exception as e:
        out[nm] = f'err {type(e).__name__}: {e}'
print('READBACK ' + json.dumps(out), flush=True)
app.close()
