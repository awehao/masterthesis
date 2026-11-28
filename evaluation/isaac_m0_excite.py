#!/usr/bin/env python3
"""M0 手臂模型資料蒐集（11/27 方法導向規劃：固定名目配置、三個構型區域、各四段獨立激勵紀錄）。

一次執行＝一段紀錄（一個區域、一個種子）。命令路徑與住宅任務 Isaac（isaac_m8_house_d2.py）逐項相同：
  * 同一機器人資產（omni_bot_manip.urdf）、同一 apply_split／零摩擦支撐、同一手臂增益（kp 600、kd 60、手指 1e4／1e3）、physics_dt 0.01；
  * 手臂命令是六關節速率，20 Hz 更新（每 5 個物理步換一次值）；
  * 經 D_STEPS＝8 個物理步的延遲管線後，每物理步積分成位置設定點 s ← clip(s + dt·u, LO, HI)，再寫入位置目標；
  * 底盤每步給零速度（同住宅任務的 base_cmd(0)），手指張開。
每步順序同住宅任務：讀實測 q_i → 收命令 → 管線 → 更新 s_i → 寫入目標 → world.step。
因此紀錄第 i 列的 q_i 是寫入前的量測、s_i 是支配 i→i+1 這段物理的設定點（辨識式 q_{i+1} = q_i + α(s_i − q_i) + b）。

激勵：設定點參考 s_ref(t) = 中心 + Σ_m a_m sin(2π f_m t + φ_m)（每軸 6 個頻率 0.07–1.3 Hz、隨機相位，種子決定），
速率命令＝ds_ref/dt 經 ±v_max 限幅（再按 20 Hz 取樣保持）；開頭 2 s、結尾 3 s 與中間 3 段 1 s 保持（命令 0）量穩定段。
輸出：<out>/m0.npz（t、u_cmd、u_eff、s、q、dq、base、hold）＋ meta.json（區域、中心、振幅、種子、增益、讀回）。
用法：isaac_m0_excite.py --out DIR --region folded --center "q1,..,q6" --seed 1000 --split train [--dur 60]
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--region', required=True, choices=['folded', 'middle', 'extended'])
ap.add_argument('--center', required=True, help='區域中心六關節角（rad）')
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
ap.add_argument('--dur', type=float, default=60.0, help='紀錄長度（模擬秒，含頭尾保持）')
ap.add_argument('--amp', type=float, default=0.20, help='各軸設定點參考相對中心的最大偏移（rad）')
ap.add_argument('--vmax', type=float, default=0.6, help='手臂速率命令上限（rad/s；同 PASS_STOW_QD）')
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--render-every', type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 960, 'height': 540})
import numpy as np                                                   # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import GroundPlane                    # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics         # noqa: E402

import door_motm_grasp_ctrl as G                                     # noqa: E402  同住宅任務：手指張開量與安全限位
from finger_collision import apply_split                             # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]
DT, D_STEPS, CTRL_EVERY = 0.01, 8, 5
FINGER_OPEN = float(G.XO.FINGER_OPEN)
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float), np.asarray(G.XO.LITE6_SAFE.upper, float)


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def excitation(center, seed, dur):
    """回傳每物理步的速率命令（20 Hz 取樣保持）與保持旗標。"""
    rng = np.random.default_rng(seed)
    n = int(round(dur / DT)); t = np.arange(n) * DT
    F = np.array([0.07, 0.13, 0.23, 0.41, 0.73, 1.3])
    u = np.zeros((n, 6))
    for j in range(6):
        w = rng.uniform(0.5, 1.0, len(F)); w = w / w.sum() * a.amp          # 振幅總和＝a.amp ⇒ |s_ref − 中心| ≤ a.amp
        ph = rng.uniform(0, 2 * math.pi, len(F))
        u[:, j] = np.sum([w[m] * 2 * math.pi * F[m] * np.cos(2 * math.pi * F[m] * t + ph[m]) for m in range(len(F))], axis=0)
    hold = (t < 2.0) | (t >= dur - 3.0)
    for c in rng.uniform(8.0, dur - 8.0, 3):                            # 中間 3 段 1 s 保持
        hold |= (t >= c) & (t < c + 1.0)
    u[hold] = 0.0
    u = np.clip(u, -a.vmax, a.vmax)
    u_zoh = u[(np.arange(n) // CTRL_EVERY) * CTRL_EVERY]                # 20 Hz 取樣保持（同控制拍）
    return t, u_zoh, hold


def main():
    center = np.array([float(x) for x in a.center.split(',')], float)
    if center.shape != (6,) or np.any(center - a.amp - 0.05 < LO) or np.any(center + a.amp + 0.05 > HI):
        print(f'[m0] 中心 {center} ± {a.amp} 超出安全限位 ⇒ 拒跑', flush=True)
        return 2
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[m0] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT)); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0)); xf.AddRotateZOp().Set(0.0)
    apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    if bind_frictionless(stage, sph + collision_prims(stage, '/World/ground')) == 0:
        print('[m0] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(800.0)
    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[m0] 找不到 articulation ⇒ 拒跑', flush=True)
        return 16
    world.reset()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[m0] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(center[j]); kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = FINGER_OPEN, FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    try:
        g_rb = robot.get_articulation_controller().get_gains()
        gains_rb = dict(kp=[float(np.asarray(g_rb[0])[i]) for i in idx], kd=[float(np.asarray(g_rb[1])[i]) for i in idx])
    except Exception as e_:                                           # noqa: BLE001
        gains_rb = dict(err=repr(e_)[:120])

    def apply(s6):
        tgt = np.array(robot.get_joint_positions(), float)
        for i, j in zip(idx, range(6)):
            tgt[i] = float(s6[j])
        tgt[f1] = FINGER_OPEN
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32)[act], joint_indices=act))

    def base_zero():
        robot.set_linear_velocity(np.zeros(3, np.float32)); robot.set_angular_velocity(np.zeros(3, np.float32))

    for k in range(int(round(a.settle_s / DT))):                      # 靜置（同住宅任務 settle：寫中心目標、底盤零速度）
        apply(center); base_zero(); world.step(render=(k % a.render_every == 0))
    t, u_cmd, hold = excitation(center, a.seed, a.dur)
    n = len(t)
    jp = np.asarray(robot.get_joint_positions(), float)
    s = np.array([jp[i] for i in idx])                                # 設定點從實測起（同住宅任務 s = q_s[3:]）
    pipe = [np.zeros(6) for _ in range(D_STEPS)]
    Q, DQ, S, UE = (np.zeros((n, 6)) for _ in range(4))
    BASE = np.zeros((n, 3))
    w0 = time.perf_counter()
    for i in range(n):
        jp = np.asarray(robot.get_joint_positions(), float); jv = np.asarray(robot.get_joint_velocities(), float)
        p, qt = robot.get_world_pose()
        Q[i] = [jp[k] for k in idx]; DQ[i] = [jv[k] for k in idx]; BASE[i] = [float(p[0]), float(p[1]), yaw_of(qt)]
        pipe.append(u_cmd[i].copy()); ue = pipe.pop(0)
        s = np.clip(s + DT * ue, LO, HI)
        S[i] = s; UE[i] = ue
        apply(s); base_zero()
        world.step(render=(i % a.render_every == 0))
        if not np.all(np.isfinite(Q[i])):
            print(f'[m0] 第 {i} 步讀回非有限值 ⇒ 中止', flush=True)
            n = i; break
    wall = time.perf_counter() - w0
    np.savez_compressed(os.path.join(a.out, 'm0.npz'), t=t[:n], u_cmd=u_cmd[:n], u_eff=UE[:n], s=S[:n], q=Q[:n], dq=DQ[:n], base=BASE[:n], hold=hold[:n])
    err = S[:n] - Q[:n]
    meta = dict(region=a.region, split=a.split, seed=a.seed, center=center.tolist(), amp=a.amp, vmax=a.vmax, dur=a.dur, n_steps=int(n),
                physics_dt=float(world.get_physics_dt()), d_steps=D_STEPS, ctrl_every=CTRL_EVERY, gains_requested=dict(kp=a.kp, kd=a.kd, finger_kp=a.finger_kp, finger_kd=a.finger_kd),
                gains_readback=gains_rb, urdf=os.path.relpath(a.urdf, WS), wall_s=round(wall, 2),
                base_drift_m=float(np.hypot(*(BASE[n - 1, :2] - BASE[0, :2]))) if n else None,
                q_range=[[float(Q[:n, j].min()), float(Q[:n, j].max())] for j in range(6)],
                track_err_rms_mrad=[float(1e3 * np.sqrt(np.mean(err[:, j] ** 2))) for j in range(6)],
                order='每步：讀 q_i → 管線 → s_i ← clip(s + dt·u_eff) → 寫入 s_i → world.step（同 isaac_m8_house_d2.py）')
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[m0] 完成 {a.region}/{a.split} seed {a.seed}：{n} 步、牆鐘 {wall:.1f} s、底盤漂移 {meta['base_drift_m']:.4f} m、設定點−實測 RMS {np.round(meta['track_err_rms_mrad'], 1).tolist()} mrad", flush=True)
    return 0


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        sim_app.close()
    sys.exit(rc)
