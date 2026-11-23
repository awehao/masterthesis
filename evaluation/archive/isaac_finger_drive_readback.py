#!/usr/bin/env python3
"""夾爪兩指有效驅動設定讀回（Codex 20261102_035124 第 1 項；只讀，不跑任務）。
照實跑腳本（isaac_bottle_w1dynsr18）的方式匯入機器人（import_urdf、apply_split）、world.reset、articulation 初始化，並套同樣增益
（手臂 kp 600／kd 60；只驅動 finger_joint1：kp 1e4／kd 1e3；finger_joint2 kp 0／kd 0），讀回：
  USD（套增益前與後）：finger_joint1／2 的關節型別、軸、限位、套用的 schema 清單、所有 drive／mimic／physx 關節相關屬性（含 drive 類型 force／acceleration、maxForce、stiffness、damping）；
  執行期 articulation view：dof 名稱、增益（stiffness／damping）、最大力（max efforts）、最大速度、dof 類型（可取得者）。
不修改任何設定以外的屬性；不推進任務。輸出 <out>/drive_readback.json。
（由 run_bottle_replay.sh 啟動：接受 --src／--out／--headless，--src 只為相容不使用。）
"""
import argparse
import json
import math
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--src', default='')
ap.add_argument('--out', required=True)
ap.add_argument('--headless', default='false')
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import GroundPlane                    # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
from isaac_common import import_urdf, physics_parts, walk            # noqa: E402
from finger_collision import apply_split                             # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]


def jprim(stage, name):
    return [pr for pr in walk(stage, ROBOT) if pr.GetName() == name]


def read_joint(stage, name):
    out = []
    for pr in jprim(stage, name):
        attrs = {}
        for at in pr.GetAttributes():
            n = at.GetName()
            if any(k in n.lower() for k in ('drive', 'mimic', 'physxjoint', 'physics:', 'limit', 'gearing', 'reference', 'offset', 'tendon', 'armature', 'friction', 'maxjoint')):
                try:
                    v = at.Get()
                    attrs[n] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                except Exception as e:                               # noqa: BLE001
                    attrs[n] = f'<err {e!r}>'
        rels = {}
        for rl in pr.GetRelationships():
            try:
                rels[rl.GetName()] = [str(x) for x in rl.GetTargets()]
            except Exception:                                        # noqa: BLE001
                pass
        out.append(dict(path=str(pr.GetPath()), type=pr.GetTypeName(), schemas=[str(x) for x in pr.GetAppliedSchemas()], attrs=attrs, rels=rels, instance_proxy=bool(pr.IsInstanceProxy())))
    return out


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    import_urdf(a.urdf, ROBOT, fix_base=False)
    fc = apply_split(stage, ROBOT)
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[rb] 找不到 articulation root', flush=True)
        return 15
    usd_import = {j: read_joint(stage, j) for j in ('finger_joint1', 'finger_joint2')}
    world.reset()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    v = robot._articulation_view
    rt = {}

    def grab(tag):
        d = {}
        for nm, fn in (('gains', lambda: v.get_gains()), ('max_efforts', lambda: v.get_max_efforts()), ('max_joint_velocities', lambda: v.get_max_joint_velocities()),
                       ('dof_types', lambda: v.get_dof_types()), ('armatures', lambda: v.get_armatures()), ('friction_coefficients', lambda: v.get_friction_coefficients())):
            try:
                x = fn()
                if isinstance(x, tuple):
                    d[nm] = [np.asarray(xx).reshape(-1).tolist() if hasattr(xx, '__len__') else xx for xx in x]
                else:
                    d[nm] = np.asarray(x).reshape(-1).tolist() if hasattr(x, '__len__') else (x.tolist() if hasattr(x, 'tolist') else str(x))
            except Exception as e:                                   # noqa: BLE001
                d[nm] = f'<不可取得 {type(e).__name__}: {e}>'
        rt[tag] = d
    grab('after_reset')
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i in idx:
        kp[i], kd[i] = a.kp, a.kd
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    grab('after_set_gains')
    usd_after = {j: read_joint(stage, j) for j in ('finger_joint1', 'finger_joint2')}
    out = dict(dof_names=names, finger_idx=dict(finger_joint1=f1, finger_joint2=f2), runtime=rt, usd_after_import=usd_import, usd_after_set_gains=usd_after,
               set_gains=dict(arm=[a.kp, a.kd], finger_joint1=[a.finger_kp, a.finger_kd], finger_joint2=[0.0, 0.0]), split=str(fc)[:500],
               note='只讀：匯入與增益同實跑腳本；runtime 為 articulation view 讀回（單位依 Isaac：prismatic 剛度 N/m、阻尼 N·s/m、最大力 N，若 drive 為 acceleration 型則不是牛頓）')
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'drive_readback.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    for j in ('finger_joint1', 'finger_joint2'):
        for p in usd_after[j]:
            print(f"[rb] {j} {p['path']} {p['type']} schemas={p['schemas']} attrs={ {k: v_ for k, v_ in p['attrs'].items() if 'drive' in k.lower() or 'mimic' in k.lower() or 'gearing' in k.lower()} }", flush=True)
    print(f"[rb] runtime after_set_gains finger1/2: " + json.dumps({k: (val[f1] if isinstance(val, list) and len(val) == len(names) else val) for k, val in rt['after_set_gains'].items()}, default=str)[:1200], flush=True)
    with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
        fh.write('0')
    return 0


rc = 99
try:
    rc = main()
except BaseException:
    import traceback
    print('[rb] 例外：' + traceback.format_exc().replace('\n', ' | '), flush=True)
finally:
    sim_app.close()
sys.exit(rc)
