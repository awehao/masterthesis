#!/usr/bin/env python3
"""D2 單一控制器趟（m8house_d2_unified_*）的整機視角 Isaac 重播：isaac_m8_house_far_door_replay.py 的複本。只擺位＋算圖，不是物理重跑、不跑控制。
狀態來源＝同趟 Isaac 每物理步紀錄 log.npz（L：q0..q8 底盤＋手臂、finger1／2、door_deg、bound）。
視角：總覽（房間斜上方，看得到整台底盤、手臂與門）＋側拍（臥室內，看通過線上的底盤與手臂）。腕部視角用同趟實際影格，不在這裡算。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_d2_replay.py --run m8house_d2_unified_6 --out <dir>
門扇依 log.npz 的 door_deg（Isaac 實測門角）擺位；手指依 finger1／2 實測開度。"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--every', type=int, default=10)
ap.add_argument('--t-from', type=float, default=-1.0, help='<0：從底盤開始移動前 1 s 起')
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json')))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCylinder   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from isaac_common import import_urdf, physics_parts             # noqa: E402

ROBOT, AP = '/World/omni_bot', 0.020
FINGER_OPEN = 0.0089


def look_R(eye, tgt):
    z = (np.asarray(tgt, float) - np.asarray(eye, float))
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0, 0, 1.0]))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def main():
    from PIL import Image
    reg = json.load(open(a.reg))
    RD = os.path.join(HERE, 'runs', a.run)
    Dz = np.load(os.path.join(RD, 'log.npz')); Dc = [str(x) for x in Dz['cols']]; DL = Dz['L']
    g = lambda n: DL[:, Dc.index(n)]                                  # noqa: E731
    # 統一列格式：[t, seg(0 綁定前／1 綁定後), x, y, yaw, q1..q6, f1, f2, door_deg]；全部取自 Isaac 每物理步紀錄（實測）
    R = np.c_[g('t'), g('bound'), g('q0'), g('q1'), g('q2'), *[g(f'q{k}') for k in range(3, 9)], g('finger1'), g('finger2'), g('door_deg')]
    mv = np.where(np.linalg.norm(np.diff(R[:, 2:4], axis=0), axis=1) > 1e-5)[0]
    t_from = a.t_from if a.t_from >= 0 else max(0.0, (float(R[mv[0], 0]) - 1.0 if len(mv) else 0.0))
    R = R[R[:, 0] >= t_from]
    # Howard 11/25「跟車旋轉太厲害，在後面看、不要跟著車轉」：跟車鏡頭方向＝車頭角度零相位高斯平滑（σ＝REPLAY_FOLLOW_SIGMA 秒）；俯視預設固定地圖方向（REPLAY_TOP_YAW＝fixed，+y 朝上）
    FOLLOW_SIG = float(os.environ.get('REPLAY_FOLLOW_SIGMA', '4.0'))
    TOP_YAW = os.environ.get('REPLAY_TOP_YAW', 'fixed')
    _yu = np.unwrap(R[:, 4]); _dt = float(np.median(np.diff(R[:, 0]))) if len(R) > 1 else 0.01
    if FOLLOW_SIG > 0 and len(R) > 3:
        from scipy.ndimage import gaussian_filter1d
        CAM_YAW = gaussian_filter1d(_yu, sigma=max(FOLLOW_SIG / max(_dt, 1e-3), 1.0), mode='nearest')
    else:
        CAM_YAW = _yu
    def cam_yaw(t_):
        return float(np.interp(t_, R[:, 0], CAM_YAW))
    os.makedirs(a.out, exist_ok=False)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root='/World/door')
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(R[0, 2]), float(R[0, 3]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(R[0, 4])))
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef'][0]
    views = dict(overview=dict(eye=[3.4, -3.9, 3.0], tgt=[1.0, -1.1, 0.5], hfov=math.radians(60.0), res=(1280, 720)),
                 side=dict(eye=[0.85, -2.05, 1.45], tgt=[0.85, -0.40, 0.75], hfov=math.radians(62.0), res=(960, 540)),
                 living=dict(eye=[1.9, 2.3, 2.3], tgt=[0.3, -0.4, 0.5], hfov=math.radians(62.0), res=(960, 540)),   # 客廳側：看穿門
                 table=dict(eye=[-0.4, 1.2, 1.7], tgt=[1.1, 2.0, 0.5], hfov=math.radians(62.0), res=(960, 540)))   # 茶几：看夾取圓柱
    if 'r2' in os.environ.get('M8_SCENE_REG', ''):                    # r2 大房子視角
        views = dict(overview=dict(eye=[4.6, -7.0, 4.2], tgt=[1.0, -1.5, 0.4], hfov=math.radians(62.0), res=(1280, 720)),
                     door=dict(eye=[0.85, -2.6, 1.6], tgt=[0.6, -0.3, 0.75], hfov=math.radians(62.0), res=(960, 540)),
                     living=dict(eye=[4.6, 7.6, 4.0], tgt=[1.0, 1.5, 0.4], hfov=math.radians(62.0), res=(1280, 720)),
                     table=dict(eye=[0.2, 2.6, 1.8], tgt=[2.0, 4.0, 0.5], hfov=math.radians(62.0), res=(960, 540)))
    if os.environ.get('REPLAY_VIEWS'):                                 # 只做指定視角（如 follow,drawer）
        keep = os.environ['REPLAY_VIEWS'].split(',')
        if 'follow' in keep:
            views['follow'] = dict(eye=[0.0, -6.0, 2.0], tgt=[0.0, -4.0, 0.5], hfov=math.radians(70.0), res=(1280, 720))   # 跟車：每格依底盤位姿重設
        if 'top' in keep:                                              # 俯視（Howard 11/19「成功趟都要給我影片 俯視跟跟車」）：跟著底盤的近正上方 5 m，車頭朝畫面上方
            views['top'] = dict(eye=[0.0, -4.6, 5.0], tgt=[0.0, -4.4, 0.0], hfov=math.radians(62.0), res=(1280, 720))
        if 'drawer' in keep and reg.get('drawer'):
            dp_ = reg['drawer']['pose']
            views['drawer'] = dict(eye=[dp_[0] + 1.6, dp_[1] - 1.3, 1.5], tgt=[dp_[0] + 0.35, dp_[1], 0.5], hfov=math.radians(62.0), res=(1280, 720))
        views = {k: v for k, v in views.items() if k in keep}
    cams = {}
    for nm, f in views.items():
        c = Camera(prim_path=f'/World/cams/{nm}', resolution=f['res'], frequency=30)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(f['hfov'] / 2.0)))
        c.set_clipping_range(0.05, 60.0)
        c.set_world_pose(position=np.array(f['eye'], float), orientation=W0.R_to_wxyz(look_R(f['eye'], f['tgt'])), camera_axes='ros')
        cams[nm] = c
    world.reset()
    from isaacsim.core.prims import RigidPrim
    leaf = RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_replay'); leaf.initialize()
    CYL = None
    res_c = json.load(open(os.path.join(RD, 'result.json'))).get('cyl')
    if res_c and res_c.get('log'):                                    # 圓柱：依同趟真值紀錄擺位（每 10 物理步一筆，線性內插）
        CYL = np.array(res_c['log'], float)
        cyl_v = VisualCylinder(prim_path='/World/cyl_replay', name='cyl_replay', position=CYL[0, 1:4], radius=float(res_c.get('radius', 0.015)), height=float(res_c.get('height', 0.20)), color=np.array([0.2, 0.6, 0.9]))
    res_all = json.load(open(os.path.join(RD, 'result.json')))
    DRL, MKL, WKL, mk_v, wk_v = None, None, None, None, []
    dr_cubes = []
    if res_all.get('drawer') and res_all['drawer'].get('log'):       # 抽屜：純顯示方盒（drawer_scene 幾何），依同趟開度真值擺位
        import drawer_scene as DRS_
        from isaacsim.core.api.objects import VisualCuboid as _VC
        DRL = np.array(res_all['drawer']['log'], float)
        for j_, (nm_, c_, R_, h_) in enumerate(DRS_.shapes(0.0)):
            yq = math.atan2(R_[1, 0], R_[0, 0]) / 2
            col = np.array([0.80, 0.66, 0.45]) if nm_.startswith('drawer_cab') else np.array([0.70, 0.50, 0.30])
            dr_cubes.append((nm_, _VC(prim_path=f'/World/drawer_vis/c{j_}', name=f'dv{j_}', position=c_, scale=2 * h_, orientation=np.array([math.cos(yq), 0, 0, math.sin(yq)]), color=col)))
        hb = DRS_.SPEC['drawer']['handle']['bar']
        hc = DRS_.to_world(hb['center'], 0.0, True)
        dr_cubes.append(('drawer_bar', VisualCylinder(prim_path='/World/drawer_vis/bar', name='dvbar', position=hc, radius=float(hb['radius']), height=float(hb['length']),
                                                      orientation=np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]), color=np.array([0.6, 0.6, 0.65]))))
    if res_all.get('marker') and res_all['marker'].get('log'):
        MKL = np.array(res_all['marker']['log'], float)
        from isaacsim.core.api.objects import VisualCuboid
        y2 = math.radians(float(reg['marker']['axis_yaw_deg'])) / 2
        mk_v = VisualCuboid(prim_path='/World/marker_replay', name='marker_replay', position=MKL[0, 1:4], scale=np.array([0.14, 0.03, 0.026]),
                            orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.85, 0.15, 0.15]))
    BNL, bin_v, obj_v = None, [], []
    if res_all.get('bin') and res_all['bin'].get('log'):            # 車尾收納盒：盒片（隨底盤）＋各物體依同趟真值位置與姿態（取代上面的單一物體擺位）
        from isaacsim.core.api.objects import VisualCuboid as _VC2
        BNL = np.array(res_all['bin']['log'], float)
        for nm_, pr_ in res_all['bin']['parts'].items():
            c_fp = np.array(pr_['c_base_link'], float) + np.array([0.0, 0.0, 0.05])
            bin_v.append((c_fp, _VC2(prim_path=f'/World/bin_replay/{nm_}', name=f'bin_{nm_}', position=c_fp, scale=2 * np.array(pr_['half'], float), color=np.array([0.85, 0.55, 0.2]))))
        for j_, o_ in enumerate(res_all['bin']['objs']):
            if o_.endswith('cyl'):
                v_ = cyl_v if CYL is not None else VisualCylinder(prim_path='/World/cyl_replay', name='cyl_replay', position=BNL[0, 4 + 7 * j_:7 + 7 * j_], radius=0.015, height=0.20, color=np.array([0.2, 0.6, 0.9]))
            else:
                from isaacsim.core.api.objects import VisualCuboid as _VC3
                v_ = mk_v if mk_v is not None else _VC3(prim_path='/World/marker_replay', name='marker_replay', position=BNL[0, 4 + 7 * j_:7 + 7 * j_], scale=np.array([0.14, 0.03, 0.026]), color=np.array([0.85, 0.15, 0.15]))
            obj_v.append(v_)
        CYL, MKL = None, None
    if res_all.get('walkers') and res_all['walkers'].get('log'):
        WKL = np.array(res_all['walkers']['log'], float)
        for j_, nm_ in enumerate(res_all['walkers']['names']):
            w_ = next(w for w in ob['walkers'] if w['name'] == nm_)
            wk_v.append(VisualCylinder(prim_path=f'/World/walkers_replay/{nm_}', name=f'wr_{nm_}', position=np.array([WKL[0, 1 + 2 * j_], WKL[0, 2 + 2 * j_], w_['height'] / 2]),
                                       radius=float(w_['radius']), height=float(w_['height']), color=np.array([0.55, 0.25, 0.65])))
    lp0, lq0 = [np.asarray(x, float)[0] for x in leaf.get_world_poses()]
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(f'joint{i}') for i in range(1, 7)]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q_full = np.array(robot.get_joint_positions(), dtype=np.float32)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array(views['overview']['eye']), target=np.array(views['overview']['tgt']))
    except Exception as e:                                            # noqa: BLE001
        print(f'[replay] 視角設定失敗：{e!r}', flush=True)

    def pose(row):
        q = q_full.copy()
        for k, i in enumerate(idx):
            q[i] = row[5 + k]
        q[f1], q[f2] = float(row[11]), float(row[12])
        yw = float(row[4])
        robot.set_world_pose(np.array([row[2], row[3], 0.0], dtype=np.float32), np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)], dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        ph = -math.radians(float(row[13]))                            # 門扇繞鉸鏈（原點、z 軸）轉 −φ（同執行器 door_q 的定義）
        c_, s_ = math.cos(ph), math.sin(ph)
        pp = np.array([c_ * lp0[0] - s_ * lp0[1], s_ * lp0[0] + c_ * lp0[1], lp0[2]])
        w0_, x0_, y0_, z0_ = lq0; cw, sw = math.cos(ph / 2), math.sin(ph / 2)
        qq = np.array([cw * w0_ - sw * z0_, cw * x0_ - sw * y0_, cw * y0_ + sw * x0_, cw * z0_ + sw * w0_])
        leaf.set_world_poses(positions=pp[None, :].astype(np.float32), orientations=qq[None, :].astype(np.float32))
        leaf.set_velocities(np.zeros((1, 6), dtype=np.float32))
        if CYL is not None:
            pc = np.array([np.interp(row[0], CYL[:, 0], CYL[:, k]) for k in (1, 2, 3)])
            cyl_v.set_world_pose(position=pc)
        if DRL is not None:
            op_ = float(np.interp(row[0], DRL[:, 0], DRL[:, 1]))
            sh_ = {n_: c_ for n_, c_, _, _ in DRS_.shapes(op_)}
            for nm_, v_ in dr_cubes:
                if nm_ == 'drawer_bar':
                    v_.set_world_pose(position=DRS_.to_world(hb['center'], op_, True))
                elif not nm_.startswith('drawer_cab'):
                    v_.set_world_pose(position=sh_[nm_])
        if MKL is not None:
            mk_v.set_world_pose(position=np.array([np.interp(row[0], MKL[:, 0], MKL[:, k]) for k in (1, 2, 3)]))
        if BNL is not None:
            i_ = int(np.clip(np.searchsorted(BNL[:, 0], row[0]), 0, len(BNL) - 1))
            for j_, v_ in enumerate(obj_v):
                pq_ = BNL[i_, 4 + 7 * j_:11 + 7 * j_]
                v_.set_world_pose(position=np.array([np.interp(row[0], BNL[:, 0], BNL[:, 4 + 7 * j_ + k_]) for k_ in range(3)]), orientation=pq_[3:7])
            yb = float(row[4]); cb, sb = math.cos(yb), math.sin(yb)
            for c_fp, v_ in bin_v:
                v_.set_world_pose(position=np.array([row[2] + cb * c_fp[0] - sb * c_fp[1], row[3] + sb * c_fp[0] + cb * c_fp[1], c_fp[2]]), orientation=np.array([math.cos(yb / 2), 0.0, 0.0, math.sin(yb / 2)]))
        for j_, v_ in enumerate(wk_v):
            p_ = np.array([np.interp(row[0], WKL[:, 0], WKL[:, 1 + 2 * j_]), np.interp(row[0], WKL[:, 0], WKL[:, 2 + 2 * j_])])
            v_.set_world_pose(position=np.array([p_[0], p_[1], 0.85]))
        if 'top' in cams:                                            # 俯視：底盤後 0.25 m、高 5 m，看底盤前 0.2 m 的地面（近正上方，避開正上方視線退化）
            yw = math.pi / 2 if TOP_YAW == 'fixed' else (cam_yaw(row[0]) if TOP_YAW == 'smooth' else float(row[4])); hx, hy = math.cos(yw), math.sin(yw)
            eye = np.array([row[2] - 0.25 * hx, row[3] - 0.25 * hy, 5.0]); tgt = np.array([row[2] + 0.2 * hx, row[3] + 0.2 * hy, 0.0])
            cams['top'].set_world_pose(position=eye, orientation=W0.R_to_wxyz(look_R(eye, tgt)), camera_axes='ros')
        if 'follow' in cams:                                         # 跟車：底盤斜後上方 2.4 m、看底盤前 0.8 m（方向用平滑後的車頭角度）
            yw = cam_yaw(row[0]); hx, hy = math.cos(yw), math.sin(yw)
            eye = np.array([row[2] - 2.2 * hx + 0.9 * hy, row[3] - 2.2 * hy - 0.9 * hx, 1.9]); tgt = np.array([row[2] + 0.8 * hx, row[3] + 0.8 * hy, 0.55])
            cams['follow'].set_world_pose(position=eye, orientation=W0.R_to_wxyz(look_R(eye, tgt)), camera_axes='ros')
    for _ in range(40):
        pose(R[0])
        world.step(render=True)
    meta = dict(run=a.run, every=a.every, fps=round(1.0 / (0.01 * a.every), 3), frames=[], note='依實錄逐格擺位＋算圖（非物理重跑）；模擬時間 1×；實錄為非即時計算')
    for n, k in enumerate(range(0, len(R), a.every)):
        for _ in range(3):                                            # 擺位後多算兩次圖，避免讀到上一格
            pose(R[k])
            world.step(render=True)
        for nm, c in cams.items():
            Image.fromarray(np.asarray(c.get_rgba())[:, :, :3].astype(np.uint8)).save(os.path.join(a.out, f'{nm}_{n:05d}.png'))
        meta['frames'].append(dict(n=n, k=int(k), t=float(R[k, 0]), seg='before_bind' if R[k, 1] < 0.5 else 'bound', door_deg=float(R[k, 13])))
        if n % 50 == 0:
            print(f'[replay] 第 {n} 格 t={R[k, 0]:.2f}', flush=True)
    json.dump(meta, open(os.path.join(a.out, 'replay_meta.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[replay] 完成 {len(meta['frames'])} 格", flush=True)
    return 0


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[replay] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
