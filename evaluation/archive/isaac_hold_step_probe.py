"""逐**物理步**的定姿保持短測：補上前次診斷缺少的時間解析度。

前次每 0.25 s 取樣一次（17 筆），位置樣本接近**不代表**兩筆之間沒有往返運動，
所以那時把速度讀值稱為「求解殘差、不是真的在動」是沒有根據的。本檔每一個
物理步都記錄，並用相鄰位置差分與速度積分交叉核對。

**不建立固定連接、不調增益、不補重力、不改速度。** 只改取樣密度，另外加兩件事：

  1. 兩段保持做單一變數對照 ——
     A 段：完全不下任何位置／速度覆寫
     B 段：每步把底盤線速度／角速度歸零（**正式試驗就是這樣做的**）
     這樣才能知道那個覆寫本身是不是擾動來源。
  2. 加錄**獨立**於 measured effort 的量：
     get_generalized_gravity_forces()      重力廣義力（不由 effort 導出）
     get_coriolis_and_centrifugal_forces() 科氏／離心項
     get_max_efforts() / get_effort_modes() 執行期的上限與驅動模式

界線（依上一輪的裁決保留）：
  * measured effort 是「關節力在自由度運動方向上的投影」，**不保證只含位置驅動輸出**。
  * get_applied_joint_efforts() 回報的是使用者經 effort 介面設定的值，
    **不是內部位置 drive 的輸出**；照錄但明確標示，不拿它冒充。
  * 找不到可直接讀回「內部位置 drive 實際輸出」的 API，此項標為**無法取得**。
"""
import argparse, csv, json, math, os, sys
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--traj', required=True)
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--hold-a-s', type=float, default=3.0, help='A 段：無任何覆寫')
ap.add_argument('--hold-b-s', type=float, default=3.0, help='B 段：每步歸零底盤速度')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

import drawer_asset as DA                                          # noqa: E402
from cpu_temp import read as cpu_temp_read                         # noqa: E402

CASE = yaml.safe_load(open(a.cases))['cases'][a.case]
SPEC = DA.load(a.spec)
ARM = [f'joint{i}' for i in range(1, 7)]
PARK = (float(CASE['parking']['x']), float(CASE['parking']['y']),
        math.radians(float(CASE['parking']['yaw_deg'])))
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
os.makedirs(a.out, exist_ok=True)
ROWS = [r for r in csv.DictReader(open(a.traj))]
CUT = max(i for i, r in enumerate(ROWS) if r['phase'] in ('reach', 'approach', 'engage'))
PLAY = ROWS[:CUT + 1]
Q_HOLD = np.array([float(PLAY[-1][j]) for j in ARM])
print(f'[hold] 重放 {len(PLAY)} 點到 engage 相位結束，**不建立固定連接**')

from isaacsim import SimulationApp                                 # noqa: E402
sim_app = SimulationApp({'headless': True, 'limit_cpu_threads': a.cpu_threads})
from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from isaacsim.core.prims import SingleArticulation                 # noqa: E402
from isaacsim.core.utils.types import ArticulationAction           # noqa: E402
from isaac_common import import_urdf                               # noqa: E402

ROBOT = '/World/omni_bot'


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def arr(x):
    return None if x is None else np.array(x).ravel()


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    DA.build_usd(world.stage, SPEC, POSE)
    import_urdf(a.urdf, ROBOT)
    world.reset()
    robot = SingleArticulation(prim_path=ROBOT, name='b'); robot.initialize()
    names = list(robot.dof_names); idx = {n: i for i, n in enumerate(names)}
    ai = [idx[j] for j in ARM]
    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    q0 = np.array([float(ROWS[0][j]) for j in ARM])
    for k, j in enumerate(ARM):
        q[idx[j]] = q0[k]; kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
    F_OPEN = float(SPEC['grasp_surface']['finger_joint_open'])
    for j in FJ:
        q[idx[j]] = F_OPEN; kp[idx[j]] = a.finger_kp; kd[idx[j]] = a.finger_kd
    robot.set_joint_positions(q)
    ctl = robot.get_articulation_controller()
    ctl.set_gains(kps=kp, kds=kd)
    ctl.apply_action(ArticulationAction(joint_positions=q))
    robot.set_world_pose(np.array([PARK[0], PARK[1], 0.0]), np.array(q_yaw(PARK[2])))
    for _ in range(20):
        world.step(render=False)

    av = robot._articulation_view
    rt = {}
    for nm, fn in (('gains', lambda: ctl.get_gains()),
                   ('max_efforts', lambda: av.get_max_efforts()),
                   ('effort_modes', lambda: av.get_effort_modes())):
        try:
            v = fn()
            if nm == 'gains':
                rt['kp'] = [float(x) for x in np.array(v[0]).ravel()[ai]]
                rt['kd'] = [float(x) for x in np.array(v[1]).ravel()[ai]]
            elif nm == 'effort_modes':
                vv = np.array(v).ravel()
                rt[nm] = [str(vv[i]) for i in ai]
            else:
                rt[nm] = [float(x) for x in np.array(v).ravel()[ai]]
        except Exception as e:
            rt[nm] = {'error': repr(e)}
    print(f'[hold] 執行期讀回：kp {rt.get("kp")}')
    print(f'[hold]              kd {rt.get("kd")}')
    print(f'[hold]     max_efforts {rt.get("max_efforts")}')
    print(f'[hold]    effort_modes {rt.get("effort_modes")}')

    have = {}
    for nm, fn in (('gravity', lambda: av.get_generalized_gravity_forces()),
                   ('coriolis', lambda: av.get_coriolis_and_centrifugal_forces()),
                   ('applied_effort', lambda: robot.get_applied_joint_efforts()),
                   ('measured_effort', lambda: robot.get_measured_joint_efforts())):
        try:
            arr(fn()); have[nm] = True
        except Exception as e:
            have[nm] = False
            print(f'[hold] {nm} 讀不到：{e!r}')
    have['internal_drive_output'] = False   # 找不到可讀回內部位置 drive 實際輸出的 API
    print(f'[hold] 可取得的量 {have}')

    for r in PLAY:
        tgt = robot.get_joint_positions()
        for k, j in enumerate(ARM):
            tgt[idx[j]] = float(r[j])
        for j in FJ:
            tgt[idx[j]] = float(r['finger'])
        ctl.apply_action(ArticulationAction(joint_positions=tgt))
        world.step(render=False)

    tgt = robot.get_joint_positions()
    for k, j in enumerate(ARM):
        tgt[idx[j]] = Q_HOLD[k]
    act = ArticulationAction(joint_positions=tgt)

    csv_p = os.path.join(a.out, 'hold_steps.csv')
    fcsv = open(csv_p, 'w', newline=''); w = csv.writer(fcsv)
    cols = (['seg', 'step', 't']
            + [f'cmd_{j}' for j in ARM] + [f'q_{j}' for j in ARM]
            + [f'qd_{j}' for j in ARM] + [f'meff_{j}' for j in ARM]
            + [f'aeff_{j}' for j in ARM] + [f'grav_{j}' for j in ARM]
            + [f'cor_{j}' for j in ARM]
            + ['base_vx', 'base_vy', 'base_vz', 'base_wx', 'base_wy', 'base_wz'])
    w.writerow(cols)
    override_count = {'A': 0, 'B': 0}

    def run_seg(tag, secs, zero_base):
        n = int(secs / a.physics_dt)
        for k in range(n):
            ctl.apply_action(act)
            if zero_base:
                # 正式試驗每步都做這件事（底盤固定）。它是**根剛體的速度覆寫**，
                # 不是關節覆寫；是否對關節造成擾動就是 A/B 對照要回答的。
                robot.set_linear_velocity(np.zeros(3))
                robot.set_angular_velocity(np.zeros(3))
                override_count[tag] += 1
            world.step(render=False)
            qm = np.array(robot.get_joint_positions())
            qv = np.array(robot.get_joint_velocities())
            me = arr(robot.get_measured_joint_efforts()) if have['measured_effort'] \
                else np.full(robot.num_dof, np.nan)
            ae = arr(robot.get_applied_joint_efforts()) if have['applied_effort'] \
                else np.full(robot.num_dof, np.nan)
            gv = arr(av.get_generalized_gravity_forces()) if have['gravity'] \
                else np.full(robot.num_dof, np.nan)
            cv = arr(av.get_coriolis_and_centrifugal_forces()) if have['coriolis'] \
                else np.full(robot.num_dof, np.nan)
            bl = np.array(robot.get_linear_velocity()).ravel()
            bw = np.array(robot.get_angular_velocity()).ravel()
            w.writerow([tag, k, f'{float(world.current_time):.5f}']
                       + [f'{Q_HOLD[i]:.9f}' for i in range(6)]
                       + [f'{qm[i]:.9f}' for i in ai] + [f'{qv[i]:.9f}' for i in ai]
                       + [f'{me[i]:.6f}' for i in ai] + [f'{ae[i]:.6f}' for i in ai]
                       + [f'{gv[i]:.6f}' for i in ai] + [f'{cv[i]:.6f}' for i in ai]
                       + [f'{v:.8f}' for v in bl] + [f'{v:.8f}' for v in bw])

    print(f'\n[hold] A 段 {a.hold_a_s:.1f} s：**不下任何位置／速度覆寫**')
    run_seg('A', a.hold_a_s, False)
    print(f'[hold] B 段 {a.hold_b_s:.1f} s：每步歸零底盤速度（與正式試驗相同）')
    run_seg('B', a.hold_b_s, True)
    fcsv.close()

    tc, tsrc = cpu_temp_read()
    json.dump({'schema': 'hold_step_probe/1', 'case': a.case,
               'traj': os.path.abspath(a.traj), 'physics_dt': a.physics_dt,
               'attachment': 'none', 'q_hold_cmd': Q_HOLD.tolist(),
               'runtime': rt, 'available': have,
               'override_counts': override_count,
               'notes': {
                   'measured_effort': ('關節力在自由度運動方向上的投影；'
                                       '不保證只含位置驅動輸出'),
                   'applied_effort': ('使用者經 effort 介面設定的值；'
                                      '**不是內部位置 drive 的輸出**'),
                   'internal_drive_output': '找不到可直接讀回的 API，標為無法取得',
                   'gravity': 'get_generalized_gravity_forces，獨立於 measured effort',
               },
               'csv': os.path.basename(csv_p),
               'cpu_temp_c': tc, 'cpu_temp_source': tsrc},
              open(os.path.join(a.out, 'hold_step_probe.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'\nCPU {tc} °C（{tsrc}）  -> {csv_p}')
    return 0


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    tb = traceback.format_exc(); print(tb, flush=True)
    open(os.path.join(a.out, 'traceback.txt'), 'w').write(tb)
    rc = 9
finally:
    sim_app.close()
sys.exit(rc)
