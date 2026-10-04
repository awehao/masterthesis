#!/usr/bin/env python3
"""接觸感測的**正向對照**：三段（接觸前／已知接觸／分離後）。

為什麼需要
----------
階段 A 四趟的抽屜淨接觸力都是 0 N。那只能說「**可用證據未顯示接觸**」——
**不能**說「完全沒有接觸」，因為沒有任何證據顯示這個感測器在**有**接觸時
會回報非零。本診斷就是補那一塊。

**獨立證據**怎麼來
------------------
用一個**位姿由程式直接設定**的探針剛體（不是機器人），沿 +y 推向把手橫桿。
接觸雙方與時刻由**位姿幾何重疊**判定 —— 探針球心到橫桿表面的距離，
由兩者的位姿與已知幾何算出，**與力訊號完全無關**。
力訊號只是被核對的對象，不參與判定何時接觸。

**本診斷能證明什麼**
  可以：感測器對**這一組受測接觸**（探針 ↔ 抽屜）有反應
  不能：
    * 不能證明其他情形不會漏 —— 淨合力互相抵銷、極短暫的接觸、
      以及**未覆蓋的接觸對**都仍可能漏掉
    * 不能回頭替階段 A 既有四趟補出當時沒有記錄的接觸證據

輸出 contact_pc.json：三段的背景／響應／恢復讀值與判定。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--drawer-asset', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--drawer-pose', default='0.0,1.45')
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--probe-radius', type=float, default=0.02)
# 三段的模擬時間
ap.add_argument('--t-before', type=float, default=2.0)
ap.add_argument('--t-press', type=float, default=3.0)
ap.add_argument('--t-after', type=float, default=2.0)
# 推進：從剛好不接觸推進到**幾何重疊** overlap_m
ap.add_argument('--approach-gap', type=float, default=0.02,
                help='接觸前段探針表面離橫桿的間隙（m）')
ap.add_argument('--overlap', type=float, default=0.002,
                help='已知接觸段的幾何重疊深度（m）')
ap.add_argument('--contact-range', type=float, default=0.001,
                help='判定「兩體相觸」的實測間隙上限（m）')
ap.add_argument('--headless', action='store_true', default=True)
a = ap.parse_args()

from isaacsim import SimulationApp                             # noqa: E402
app = SimulationApp({'headless': True})

from isaacsim.core.api import World                            # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane  # noqa: E402
from isaacsim.core.prims import RigidPrim                      # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics           # noqa: E402
import drawer_asset as DA                                      # noqa: E402

DRAWER_ROOT = '/World/drawer_unit'
DRAWER_PRIM = DRAWER_ROOT + '/drawer'
PROBE = '/World/contact_probe'


def main() -> int:
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    stage = world.stage
    spec = DA.load(a.drawer_asset)
    pose = tuple(float(v) for v in a.drawer_pose.split(','))
    auth = DA.build_usd(stage, spec, pose, root=DRAWER_ROOT)
    for k in ('drive_stiffness', 'drive_damping', 'drive_max_force'):
        if float(auth[k]) != 0.0:
            print(f'[pc] **抽屜 {k} 讀回 {auth[k]} 不為 0，中止**')
            return 14

    # ---- 探針：運動學剛體（位姿由程式設定，不受力） ----
    bar = spec['drawer']['handle']['bar']
    bar_c = np.array([bar['center'][0] + pose[0],
                      bar['center'][1] + pose[1],
                      bar['center'][2]], float)
    bar_r = float(bar['radius'])
    sph = UsdGeom.Sphere.Define(stage, PROBE)
    sph.GetRadiusAttr().Set(a.probe_radius)
    xf = UsdGeom.Xformable(sph.GetPrim())
    op = xf.AddTranslateOp()
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sph.GetPrim())
    # **運動學**：位姿由程式驅動，不被接觸力推開 ⇒ 幾何重疊可控
    UsdPhysics.RigidBodyAPI(sph.GetPrim()).CreateKinematicEnabledAttr().Set(True)

    def probe_pos(gap):
        """探針球心位置：表面離橫桿表面 gap（gap<0 表示重疊）。"""
        return np.array([bar_c[0], bar_c[1] - (bar_r + a.probe_radius + gap),
                         bar_c[2]], float)

    op.Set(Gf.Vec3d(*probe_pos(a.approach_gap)))

    # 抽屜與探針的接觸力視圖：**view 要在 reset 之前建**
    drawer_v = RigidPrim(prim_paths_expr=DRAWER_PRIM, name='drawer_v',
                         track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True,
                         contact_filter_prim_paths_expr=[PROBE])
    probe_v = RigidPrim(prim_paths_expr=PROBE, name='probe_v',
                        track_contact_forces=True, max_contact_count=128,
                        prepare_contact_sensors=True,
                        contact_filter_prim_paths_expr=[DRAWER_PRIM])
    world.reset()
    drawer_v.initialize()
    probe_v.initialize()
    dp0, _ = drawer_v.get_world_poses()
    DY0 = float(dp0[0][1])
    print(f'[pc] 抽屜已建於 {DRAWER_ROOT} @ {pose}；探針半徑 {a.probe_radius}；'
          f'橫桿中心 {np.round(bar_c,4).tolist()} 半徑 {bar_r}；DY0={DY0:.6f}',
          flush=True)

    phases = [('before', a.t_before, a.approach_gap),
              ('press', a.t_press, -abs(a.overlap)),
              ('after', a.t_after, a.approach_gap)]
    rec = {'phases': {}, 'drawer_authored': auth,
           'probe': {'prim': PROBE, 'radius_m': a.probe_radius,
                     'kinematic': True},
           'bar': {'center_world': bar_c.tolist(), 'radius_m': bar_r},
           'independent_evidence': (
               '接觸雙方與時刻由**位姿幾何重疊**判定（探針球心到橫桿表面的'
               '距離，由兩者位姿與已知幾何算出），**與力訊號無關**'),
           'api': {'net': 'RigidPrim.get_net_contact_forces(dt)',
                   'matrix': 'RigidPrim.get_contact_force_matrix(dt)'
                             '（contact_filter_prim_paths_expr 已給）'},
           'scope': {
               'can_show': '感測器對**這一組受測接觸**（探針↔抽屜）有反應',
               'cannot_show': [
                   '不能證明其他情形不會漏：淨合力互相抵銷、極短暫的接觸、'
                   '以及**未覆蓋的接觸對**仍可能漏掉',
                   '不能回頭替階段 A 既有四趟補出當時沒有記錄的接觸證據'],
           }}
    for name, dur, gap in phases:
        op.Set(Gf.Vec3d(*probe_pos(gap)))
        n = int(round(dur / a.physics_dt))
        rows = []
        for _ in range(n):
            world.step(render=False)
            pp, _ = probe_v.get_world_poses()
            dpp, _ = drawer_v.get_world_poses()
            # **獨立幾何距離**：探針表面到橫桿表面（負值 = 重疊）
            pc = np.asarray(pp[0], float)
            # 橫桿隨抽屜移動：抽屜沿 y 的位移要補上
            dy = float(dpp[0][1]) - DY0
            bc = bar_c + np.array([0.0, dy, 0.0])
            radial = math.hypot(pc[1] - bc[1], pc[2] - bc[2])
            gap_geom = radial - bar_r - a.probe_radius
            try:
                net = np.asarray(drawer_v.get_net_contact_forces(
                    dt=a.physics_dt), float).reshape(-1, 3).sum(axis=0)
                net_mag = float(np.linalg.norm(net))
            except Exception as e:
                net_mag = float('nan'); rec.setdefault('net_err', repr(e))
            try:
                M = np.asarray(drawer_v.get_contact_force_matrix(
                    dt=a.physics_dt), float)
                pair_mag = float(np.linalg.norm(M.reshape(-1, 3).sum(axis=0)))
            except Exception as e:
                pair_mag = float('nan')
                rec.setdefault('matrix_err', repr(e))
            rows.append((float(world.current_time), gap_geom, net_mag,
                         pair_mag, DY0 - float(dpp[0][1])))
        A = np.array(rows, float)
        g, nm, pm, opn = A[:, 1], A[:, 2], A[:, 3], A[:, 4]
        gf, nf, pf = np.isfinite(g), np.isfinite(nm), np.isfinite(pm)
        rec['phases'][name] = {
            'commanded_gap_m': gap,
            'n_steps': len(A),
            't_span': [float(A[0, 0]), float(A[-1, 0])],
            'geom_gap_min_m': (float(g[gf].min()) if gf.any() else None),
            'geom_gap_max_m': (float(g[gf].max()) if gf.any() else None),
            'geom_overlapping_frac': (float((g[gf] < 0).mean())
                                      if gf.any() else None),
            # **判定用這一項**：實測間隙是否塌進接觸範圍。
            # 持續「幾何重疊」不是正確判準 —— 探針是運動學剛體，
            # PhysX 會把重疊推給另一個物體化解（實測：press 段間隙由
            # −2.000 mm 回到 +0.654 mm），所以重疊比例必然接近 0。
            'geom_in_contact_frac': (
                float((g[gf] <= a.contact_range).mean()) if gf.any() else None),
            'net_mag_max_n': (float(nm[nf].max()) if nf.any() else None),
            'net_mag_p50_n': (float(np.median(nm[nf])) if nf.any() else None),
            'net_finite_frac': float(nf.mean()),
            'pair_mag_max_n': (float(pm[pf].max()) if pf.any() else None),
            'pair_finite_frac': float(pf.mean()),
            'drawer_opening_max_abs_m': float(np.abs(opn).max()),
        }
        print(f'[pc] {name}: 接觸範圍內 '
              f'{rec["phases"][name]["geom_in_contact_frac"]*100:.1f}%；幾何間隙 '
              f'{rec["phases"][name]["geom_gap_min_m"]:+.6f}…'
              f'{rec["phases"][name]["geom_gap_max_m"]:+.6f} m'
              f'（重疊比例 {rec["phases"][name]["geom_overlapping_frac"]*100:.0f}%）'
              f'；淨合力 max {rec["phases"][name]["net_mag_max_n"]}'
              f'；逐對 max {rec["phases"][name]["pair_mag_max_n"]}',
              flush=True)

    # ---- 判定 ----
    b, p, c = (rec['phases'][k] for k in ('before', 'press', 'after'))
    bad = []
    # **分段有效性用「實測間隙是否在接觸範圍內」判定**（見上方註解）
    if (b['geom_in_contact_frac'] or 0) > 0.0:
        bad.append(f'接觸前段就有步落在接觸範圍內 '
                   f'（{(b["geom_in_contact_frac"] or 0)*100:.1f}%）⇒ 分段無效')
    if (p['geom_in_contact_frac'] or 0) < 0.9:
        bad.append(f'已知接觸段只有 '
                   f'{(p["geom_in_contact_frac"] or 0)*100:.1f}% 的步落在接觸'
                   f'範圍內 ⇒ 接觸未建立')
    if (c['geom_in_contact_frac'] or 0) > 0.0:
        bad.append(f'分離後段仍有步落在接觸範圍內 '
                   f'（{(c["geom_in_contact_frac"] or 0)*100:.1f}%）⇒ 未分離')
    responded = ((p['net_mag_max_n'] or 0.0) > (b['net_mag_max_n'] or 0.0)
                 and (p['net_mag_max_n'] or 0.0) > 0.0)
    recovered = ((c['net_mag_max_n'] or 0.0) < (p['net_mag_max_n'] or 0.0))
    # **第三條獨立證據**：抽屜自己被推動的量（與力訊號、與探針位姿都無關）
    rec['drawer_motion_evidence'] = {
        'before_opening_max_abs_m': b['drawer_opening_max_abs_m'],
        'press_opening_max_abs_m': p['drawer_opening_max_abs_m'],
        'after_opening_max_abs_m': c['drawer_opening_max_abs_m'],
        'note': ('press 段抽屜自身的位移是獨立於力訊號的第三條證據；'
                 'after 段的殘量是被推動後未回彈的部分'),
    }
    rec['force_magnitude_caveat'] = (
        '力的**量級不可當成真實接觸力**：探針是運動學剛體、被強制壓入 '
        f'{abs(a.overlap)*1e3:.1f} mm，而抽屜受滑動限位約束，'
        '所以數值反映的是求解器化解穿透的反作用，不是物理上合理的接觸力。'
        '本對照只問**訊號是否響應**，不問量級是否真實。')
    rec['verdict'] = {
        'phases_valid': not bad,
        'phase_problems': bad,
        'force_responded_to_known_contact': bool(responded),
        'force_recovered_after_separation': bool(recovered),
        'background_net_max_n': b['net_mag_max_n'],
        'contact_net_max_n': p['net_mag_max_n'],
        'after_net_max_n': c['net_mag_max_n'],
        'pass': bool((not bad) and responded and recovered),
    }
    print(f'[pc] 判定 {rec["verdict"]}', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'[pc] -> {a.out}', flush=True)
    return 0 if rec['verdict']['pass'] else 75


try:
    rc = main()
finally:
    app.close()
raise SystemExit(rc)
