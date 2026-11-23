#!/usr/bin/env python3
"""Isaac 時間基準驗收：實際物理推進 vs 主迴圈記帳 vs /clock 會發布的值。

無接觸場景（單一剛體浮在空中，重力關閉），每步以 set_linear_velocity /
set_angular_velocity 施加固定命令。因為沒有接觸、沒有阻尼、速度每步被覆寫，
位移與轉角在解析上就等於 命令 × 實際物理時間，所以「位移 ÷ 命令」直接給出
實際物理時間，不需要相信任何 API 自報的數字。

掃描 render 設定 × rendering_dt 設定，並單獨測「命令在哪一個物理步生效」。
"""
import argparse, json, math, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--vx', type=float, default=0.20)
ap.add_argument('--wz', type=float, default=0.50)
ap.add_argument('--render-hz', default='0,5,12,30',
                help='0 代表整趟都不 render')
ap.add_argument('--rendering-mult', default='4,1',
                help='rendering_dt = physics_dt * 這個倍數')
ap.add_argument('--only', default='',
                help='"render_hz,mult" 只跑單一設定，一個設定一個行程，'
                     '避免 clear_instance 的殘留影響')
ap.add_argument('--skip-onset', action='store_true')
ap.add_argument('--out', default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results/isaac_time_audit.json'))
a = ap.parse_args()

from isaacsim import SimulationApp                                # noqa: E402
sim_app = SimulationApp({'headless': True})

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.simulation_context import SimulationContext  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid              # noqa: E402
from pxr import UsdPhysics                                        # noqa: E402
import omni.usd                                                   # noqa: E402


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def sim_time_attrs(world):
    """能取得的所有『實際模擬時間』來源，一併回報而不預設哪個可信。"""
    out = {}
    for name in ('current_time', 'current_time_step_index'):
        try:
            out[name] = getattr(world, name)
        except Exception as e:
            out[name] = f'不可用: {e}'
    return out


def fresh_world(dt, mult):
    """A World whose constructor arguments actually take effect.

    Without clear_instance() the second and later World(...) calls in one
    process log 'already initialized, constructor parameters are ignored' and
    silently keep the FIRST configuration. An earlier version of this audit
    swept rendering_dt without clearing, so every case ran at the first
    setting and the sweep proved nothing. The readback below is what makes the
    configuration verified rather than assumed.
    """
    try:
        World.clear_instance()
    except Exception:
        pass
    try:
        SimulationContext.clear_instance()
    except Exception:
        pass
    w = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt * mult)
    back = {}
    for name in ('get_physics_dt', 'get_rendering_dt'):
        try:
            back[name] = getattr(w, name)()
        except Exception as e:
            back[name] = f'不可用: {e}'
    return w, back


def run_case(render_hz, mult, steps, dt, vx, wz):
    world, back = fresh_world(dt, mult)
    stage = omni.usd.get_context().get_stage()
    body = DynamicCuboid(prim_path='/World/probe', name='probe',
                         position=np.array([0.0, 0.0, 1.0]),
                         scale=np.array([0.2, 0.2, 0.2]), mass=1.0)
    # No contact and no gravity: the body's motion is exactly the commanded
    # velocity integrated over whatever time physics actually advanced.
    UsdPhysics.RigidBodyAPI(body.prim).CreateKinematicEnabledAttr(False)
    try:
        world.get_physics_context().set_gravity(0.0)
    except Exception:
        pass
    world.reset()

    re_ = 0 if render_hz <= 0 else max(1, int(round((1.0 / render_hz) / dt)))
    p0, q0 = body.get_world_pose()
    t0_attrs = sim_time_attrs(world)
    renders = 0
    for k in range(steps):
        body.set_linear_velocity(np.array([vx, 0.0, 0.0], dtype=np.float32))
        body.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))
        do_render = (re_ > 0 and k % re_ == 0)
        renders += int(do_render)
        world.step(render=do_render)
    p1, q1 = body.get_world_pose()
    t1_attrs = sim_time_attrs(world)

    book = steps * dt                       # 主迴圈記帳（現行 /clock 的來源）
    dx = float(p1[0]) - float(p0[0])
    dyaw = (yaw_of(q1) - yaw_of(q0) + math.pi) % (2 * math.pi) - math.pi
    # 旋轉可能超過一圈，用位移為主、轉角作交叉檢查
    t_from_x = dx / vx if abs(vx) > 1e-9 else float('nan')
    cur0, cur1 = t0_attrs.get('current_time'), t1_attrs.get('current_time')
    t_api = (cur1 - cur0) if isinstance(cur0, float) and isinstance(cur1, float) \
        else float('nan')
    idx0 = t0_attrs.get('current_time_step_index')
    idx1 = t1_attrs.get('current_time_step_index')
    nsub = (idx1 - idx0) if isinstance(idx0, int) and isinstance(idx1, int) \
        else None
    pred = float('nan') if re_ <= 0 else (re_ - 1 + mult) / re_
    res = dict(render_hz=render_hz, rendering_mult=mult, steps=steps, dt=dt,
               re_=re_, renders=renders,
               bookkeeping_s=book, dx=dx, dyaw_deg=math.degrees(dyaw),
               phys_s_from_dx=t_from_x,
               ratio_measured=t_from_x / book if book else float('nan'),
               ratio_predicted=pred if re_ > 0 else 1.0,
               phys_s_from_api=t_api,
               ratio_from_api=t_api / book if book else float('nan'),
               physics_substeps=nsub, readback=back,
               substeps_per_loop=(nsub / steps) if nsub else None,
               api=dict(t0=t0_attrs, t1=t1_attrs))
    world.stop()
    world.clear()
    return res


def run_onset(mult, dt, vx, steps=12, cmd_at=5, render_every=0):
    """命令在哪一個物理步生效，以及帶 render 的步吃掉多少命令。

    render_every>0 時讓 cmd_at 那一步剛好 render，用來分辨「命令套用一次但被
    積分了多個子步」與「命令只作用一個子步」。
    """
    world, back = fresh_world(dt, mult)
    body = DynamicCuboid(prim_path='/World/onset', name='onset',
                         position=np.array([0.0, 0.0, 1.0]),
                         scale=np.array([0.2, 0.2, 0.2]), mass=1.0)
    try:
        world.get_physics_context().set_gravity(0.0)
    except Exception:
        pass
    world.reset()
    rows = []
    for k in range(steps):
        v = vx if k == cmd_at else 0.0
        body.set_linear_velocity(np.array([v, 0.0, 0.0], dtype=np.float32))
        body.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        before = float(body.get_world_pose()[0][0])
        did = (render_every > 0 and k % render_every == 0)
        world.step(render=did)
        after = float(body.get_world_pose()[0][0])
        rows.append(dict(k=k, cmd=v, rendered=did, x_before=before,
                         x_after=after, moved=after - before))
    world.stop(); world.clear()
    return dict(cmd_at=cmd_at, vx=vx, dt=dt, rendering_mult=mult,
                render_every=render_every, readback=back, rows=rows)


def main():
    if a.only:
        _rh, _m = a.only.split(',')
        rhs, mults = [float(_rh)], [int(_m)]
    else:
        rhs = [float(x) for x in a.render_hz.split(',')]
        mults = [int(x) for x in a.rendering_mult.split(',')]
    out = dict(physics_dt=a.physics_dt, steps=a.steps, vx=a.vx, wz=a.wz,
               cases=[], onset=[])
    print(f'{"render_hz":>9} {"mult":>4} {"re_":>4} {"renders":>7} '
          f'{"記帳s":>7} {"物理s(位移)":>11} {"倍率(量測)":>10} '
          f'{"倍率(預測)":>10} {"物理s(API)":>10} {"子步/迴圈":>9}')
    for mult in mults:
        for rh in rhs:
            r = run_case(rh, mult, a.steps, a.physics_dt, a.vx, a.wz)
            out['cases'].append(r)
            sp = r['substeps_per_loop']
            print(f'    回讀: {r["readback"]}')
            print(f'{rh:9.1f} {mult:4d} {r["re_"]:4d} {r["renders"]:7d} '
                  f'{r["bookkeeping_s"]:7.3f} {r["phys_s_from_dx"]:11.4f} '
                  f'{r["ratio_measured"]:10.4f} {r["ratio_predicted"]:10.4f} '
                  f'{r["phys_s_from_api"]:10.4f} '
                  f'{("%.3f" % sp) if sp else "n/a":>9}', flush=True)
    if a.skip_onset:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')
        return
    print('\n-- 命令生效步 --')
    for mult in mults:
        for rev in (0, 5):
            o = run_onset(mult, a.physics_dt, a.vx, render_every=rev)
            out['onset'].append(o)
            moved = [r['k'] for r in o['rows'] if abs(r['moved']) > 1e-9]
            lbl = '不 render' if rev == 0 else f'每 {rev} 步 render（命令那步剛好 render）'
            print(f'  rendering_mult={mult}，{lbl}：命令下在 k={o["cmd_at"]}，'
                  f'移動發生在 k={moved}  回讀 {o["readback"]}')
            for r in o['rows']:
                if abs(r['moved']) > 1e-12 or r['cmd']:
                    print(f'    k={r["k"]:2d} cmd={r["cmd"]:.2f} '
                          f'render={int(r["rendered"])} '
                          f'移動 {r["moved"]*1000:+.4f} mm '
                          f'(= {r["moved"]/max(a.vx,1e-9)/a.physics_dt:.2f} 個物理步)')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'\n-> {a.out}')


main()
sim_app.close()
