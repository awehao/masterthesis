#!/usr/bin/env python3
"""B0 無門底盤表觀速度殘差短測：Isaac 端（規格 B0_base_residual_probe_spec.md；Codex 20261025_020934）。

一個程序＝一個條件（獨立初始化）。場景與執行鏈取自 isaac_door_k2_sim_ev4.py，**不建門**：
  World(physics_dt 0.01)、GroundPlane、同 URDF、同停位、手指碰撞 split、支撐球＋地面零摩擦材質、手指夾持面摩擦 μs 1.0／μd 0.8、
  手臂 kp 600／kd 60、手指 kp 1e4／kd 1e3、手指開度固定 0.0089（全條件同，夾持姿態下也張開——無橫桿，揭露）。
逐步（與 ev4 同順序）：讀位姿 → 命令（core.CommandSource）→ set_linear_velocity(R(yaw)·cmd)／set_angular_velocity → 讀回 →
  apply_action（手臂固定姿態、手指固定開度）→ world.step → 讀步進後位姿與根速度。輪框／滾子關節速度可讀就記，讀不到記 None。
不做偏差補償、不調伺服、不改任何門檻。輸出 <out>/probe.json。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--condition', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--headless', default='true')
a = ap.parse_args()

sys.path.insert(0, HERE)
import base_residual_probe_core as C                             # noqa: E402

POSE_NAME, MODE = C.parse_condition(a.condition)

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import GroundPlane                # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, UsdGeom                                      # noqa: E402

from isaac_common import (bind_friction, bind_frictionless,     # noqa: E402
                          collision_prims, import_urdf,
                          physics_parts, walk)

ARM = [f'joint{i}' for i in range(1, 7)]
ROBOT = '/World/omni_bot'
KP, KD, FKP, FKD, FOPEN = 600.0, 60.0, 1.0e4, 1.0e3, 0.0089      # ev4 預設值
MU_S, MU_D = 1.0, 0.8


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _l(v):
    return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=C.PHYSICS_DT, rendering_dt=C.PHYSICS_DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    rx, ry, ryaw = C.STAND
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, 0.0))
    xf.AddRotateZOp().Set(math.degrees(ryaw))
    from finger_collision import apply_split
    fcol = apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    gr = collision_prims(stage, '/World/ground')
    nb = bind_frictionless(stage, sph + gr)
    print(f'[probe] 零摩擦材質：支撐球 {len(sph)} + 地面 {len(gr)} → 綁定 {nb}', flush=True)
    if nb == 0:
        print('[probe] **零摩擦一個都沒綁到** ⇒ 中止', flush=True)
        return 19
    grip = [pr for pr in collision_prims(stage, ROBOT) if 'finger' in str(pr.GetPath()).lower()]
    n_grip = bind_friction(stage, grip, MU_S, MU_D)
    from friction_audit import audit as _fric_audit
    fr_ok, fr_rep = _fric_audit(stage, walk, robot_root=ROBOT, drawer_root='/World/__no_door__')
    json.dump(fr_rep, open(os.path.join(a.out, 'friction_audit.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    if not fr_ok:
        print('[probe] **摩擦核對未通過** ⇒ 中止', flush=True)
        return 20
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[probe] **找不到 articulation root** ⇒ 中止', flush=True)
        return 15
    world.reset()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: names.index(n) for n in ARM}
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in names]
    rim_idx = [i for i, n in enumerate(names) if n.startswith('base_link_rim_') and n.endswith('_joint')]
    roller_idx = [i for i, n in enumerate(names) if n.startswith('roller_')]
    qarm = C.POSES[POSE_NAME]['q']
    q0 = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for k, j in enumerate(ARM):
        q0[idx[j]] = qarm[k]
        kp[idx[j]], kd[idx[j]] = KP, KD
    for j in FJ:
        q0[names.index(j)] = FOPEN
        kp[names.index(j)], kd[names.index(j)] = FKP, FKD
    robot.set_joint_positions(q0)                                 # 獨立初始化：直接置於條件姿態（不經展開過程）；順序同 ev4
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q0))
    tgt = np.array(q0, dtype=np.float32)
    src = C.CommandSource(MODE)
    n_steps = int(round((C.WARMUP_S + C.OBSERVE_S) / C.PHYSICS_DT))
    rows = []
    print(f'[probe] 條件 {a.condition}：姿態 {POSE_NAME}（{C.POSES[POSE_NAME]["source"]}）模式 {MODE}；'
          f'輪框關節 {len(rim_idx)}、滾子關節 {len(roller_idx)}；進入主迴圈', flush=True)
    for step in range(n_steps):
        t = step * C.PHYSICS_DT
        p, q = robot.get_world_pose()
        yaw = yaw_of(q)
        pose = [float(p[0]), float(p[1]), yaw]
        v_before = _l(robot.get_linear_velocity())
        cmd = src.cmd(step, pose)
        wx, wy = C.world_of(cmd, yaw)
        robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, cmd[2]], dtype=np.float32))
        v_rb, w_rb = _l(robot.get_linear_velocity()), _l(robot.get_angular_velocity())
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=tgt))
        qj, qd = robot.get_joint_positions(), robot.get_joint_velocities()
        world.step(render=False)
        p2, q2 = robot.get_world_pose()
        qd_np = None if qd is None else np.asarray(qd, float)
        rows.append([step, t, pose, [float(p2[0]), float(p2[1]), yaw_of(q2)], list(cmd), [wx, wy, 0.0], float(cmd[2]),
                     v_rb, w_rb, v_before, _l(robot.get_linear_velocity()), _l(robot.get_angular_velocity()),
                     [float(qj[idx[j]]) for j in ARM], None if qd_np is None else [float(qd_np[idx[j]]) for j in ARM],
                     [float(qj[names.index(j)]) for j in FJ],
                     None if (qd_np is None or not rim_idx) else [float(qd_np[i]) for i in rim_idx],
                     None if (qd_np is None or not roller_idx) else float(np.abs(qd_np[roller_idx]).max()),
                     'warmup' if t < C.WARMUP_S - 1e-9 else 'observe'])
    summ = C.summarize(rows)
    if MODE == 'park_hold':
        summ['park_steady'] = C.park_steady(rows, src.anchor)
    rep = {'schema': 'b0_base_residual_probe/1', 'condition': a.condition, 'pose': POSE_NAME, 'mode': MODE,
           'pose_source': C.POSES[POSE_NAME]['source'], 'q_arm_target': qarm, 'anchor': src.anchor, 'servo_updates': src.n_updates,
           'config': {'physics_dt': C.PHYSICS_DT, 'warmup_s': C.WARMUP_S, 'observe_s': C.OBSERVE_S, 'stand': C.STAND,
                      'kp': KP, 'kd': KD, 'finger_kp': FKP, 'finger_kd': FKD, 'finger_open': FOPEN, 'grip_mu': [MU_S, MU_D],
                      'frictionless_bound': nb, 'grip_bound': n_grip, 'finger_collision': 'split',
                      'servo_period_steps': C.SERVO_PERIOD_STEPS, 'urdf': a.urdf},
           'dof': {'rim': [names[i] for i in rim_idx], 'n_roller': len(roller_idx)},
           'finger_collision': fcol, 'summary': summ, 'steps_cols': C.STEP_COLS, 'steps': rows}
    json.dump(rep, open(os.path.join(a.out, 'probe.json'), 'w'), ensure_ascii=False, default=float)
    print('[probe] 彙整 ' + json.dumps(summ, ensure_ascii=False, default=float), flush=True)
    return 0


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        sim_app.close()
    sys.exit(rc)
