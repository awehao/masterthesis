#!/usr/bin/env python3
"""區域偏移版（isaac_bottle_replay_w1z.py 的版本化複本，2026-11-02；R-B1）：第一桌（桌面、桌腳）放在 原場景位置＋來源趟 s1_report.json 的 region.offset（沒有則 0,0），
第二桌、原障礙、相機照舊；機器人與瓶照逐步紀錄（世界座標）擺放。
以下為 w1z 說明。
第一桌高度版（isaac_bottle_replay_w1.py 的版本化複本，2026-11-02）：第一桌桌面頂讀來源趟 s1_report.json 的 table1_z（沒有則＝資產），第二桌維持資產高度。
以下為原說明。
瓶子 W1 視覺抓放「該趟實錄重播」（isaac_bottle_replay.py 的版本化複本，2026-11-01；Howard：要錄影並且要有手腕攝影機畫面）：
加第二桌（同資產整體位移，位移讀來源 readback 的瓶位與 PLACE）、手腕相機（link_eef 下，外參＝URDF link_eef→camera_color_optical_frame，hfov 69°、640×480、裁切 0.05–10 m，同實跑 W1），
全景改為涵蓋兩桌的固定視角；不加出口區標記（放置任務終點是第二桌）。四路影格：m 跟拍、h 手腕、w 全景、t 俯視。其餘同原版。
以下為原版說明。
瓶子任務「該趟實錄重播」：只渲染、不跑控制、不起 ROS。場景與 isaac_bottle_s1.py 同（桌、障礙、側牆、懸空方塊、瓶、機器人、燈），參數一律讀來源趟 s1_report.json。
逐幀把底盤、關節、手指與瓶擺到 s1_steps.npz 的量測位姿，不推進物理（update_transformations 後 world.render），擷取前回讀實際姿態對來源的偏差寫入 frames.csv。
三台相機：main 斜後方跟拍（底盤軌跡零相位平滑後等距跟隨）、wide 固定全景（與實跑 viewport 同視角）、top 俯視跟拍。出口區圓盤是重播才加的視覺標記，無碰撞。
起始 1 s 靜態沉降只用來取底盤 z（紀錄沒有 z），之後物理不再推進。
    $ISAAC_PY evaluation/isaac_bottle_replay.py --src bottle_s3_1 --out evaluation/runs/bottle_s3_1_replay
"""
import argparse
import json
import math
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='來源 RUN_ID')
ap.add_argument('--out', required=True)
ap.add_argument('--asset', default=os.path.join(WS, 'src/my_omnibot_description/config/bottle_s0.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--hz', type=float, default=25.0)
ap.add_argument('--t0', type=float, default=0.0)
ap.add_argument('--t1', type=float, default=1e9)
ap.add_argument('--renders', type=int, default=4, help='每幀 render 次數：讓影像與本幀姿態對齊')
ap.add_argument('--res', default='1280x720')
ap.add_argument('--res-wide', default='640x360')
ap.add_argument('--eye-ofs', default='-1.7,-1.3,1.7', help='跟拍相機對注視點的偏移')
ap.add_argument('--at-ofs', default='0.1,0.2,0.55', help='注視點對平滑底盤的偏移（z 為絕對高度）')
ap.add_argument('--foc', type=float, default=26.0)
ap.add_argument('--headless', default='false')
ap.add_argument('--res-hand', default='640x480')
ap.add_argument('--res-wide2', default='640x240')
ap.add_argument('--wide-eye', default='-0.4,-5.6,3.4')
ap.add_argument('--wide-at', default='-0.4,-0.3,0.3')
ap.add_argument('--wide-foc', type=float, default=14.0)
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402
from isaacsim.sensors.camera import Camera                           # noqa: E402
from PIL import Image                                                # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from isaac_common import import_urdf, physics_parts                  # noqa: E402
from finger_collision import apply_split                             # noqa: E402
import door_replay_capture as REPC                                   # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]
TOL = dict(base_mm=1.0, yaw_deg=0.1, joint_mrad=1.0, bottle_mm=1.0)
WIDE_EYE, WIDE_AT, WIDE_FOC = (-0.6, -4.2, 2.6), (-1.0, -0.2, 0.4), 18.15
TOP_H, TOP_FOC, TOP_YOFS = 5.0, 30.0, 0.15


def box_of(s):
    v = [float(x) for x in s.split(',')]
    return np.array(v[:3]), np.array(v[3:])


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def smooth(v, dt, sigma=0.7):
    n = int(round(3 * sigma / dt))
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * dt / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, n, mode='edge'), k, mode='valid')


def main():
    D = os.path.join(HERE, 'runs', a.src)
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    rb = rep['readback']
    S = yaml.safe_load(open(a.asset))
    T, B, PH = S['table'], S['bottle'], S['physics']
    T2 = dict(T)                                                             # 第二桌：資產高度
    T = dict(T, top_z=float(rep.get('table1_z', T['top_z'])))                 # 第一桌：來源趟的 table1_z
    RO = np.array(list((rep.get('region') or {}).get('offset', [0.0, 0.0])) + [0.0], float)   # 第一桌世界平移（R-B1）
    print(f"[rp] 區域偏移 {RO[:2].tolist()}", flush=True)
    print(f"[rp] 第一桌 {T['top_z']:.3f} m、第二桌 {T2['top_z']:.3f} m", flush=True)
    dt = float(PH['dt'])
    os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
    if os.listdir(os.path.join(a.out, 'frames')):
        print('[rp] frames 已有影格 ⇒ 拒絕覆寫', flush=True)
        return 32
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from omni.physx import get_physx_interface
    from pxr import Gf, UsdGeom, UsdLux
    # 場景（同 isaac_bottle_s1.py）
    FixedCuboid(prim_path='/World/table/top', name='table_top', position=np.array([0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]) + RO,
                scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.55, 0.42, 0.30]))
    hz = T['top_z'] - T['thick']
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            FixedCuboid(prim_path=f'/World/table/leg_{nm}_{"p" if sx > 0 else "m"}', name=f'leg_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + RO,
                        scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.4, 0.3, 0.2]))
    ob = rep['obstacle']
    FixedCuboid(prim_path='/World/obstacle', name='obstacle', position=np.array(ob['c']), scale=np.array(ob['sz']), color=np.array([0.7, 0.25, 0.2]))
    if rb.get('wall_box'):
        w_ = box_of(rb['wall_box'])
        FixedCuboid(prim_path='/World/wall', name='wall', position=w_[0], scale=w_[1], color=np.array([0.5, 0.5, 0.55]))
    if rb.get('s2_box'):
        s_ = box_of(rb['s2_box'])
        FixedCuboid(prim_path='/World/s2_box', name='s2_box', position=s_[0], scale=s_[1], color=np.array([0.2, 0.7, 0.3]))
    pt = rb.get('perturb') or dict(bottle_dx=0.0, bottle_dy=0.0, x_start=rb['base'][0], y_start=rb['base'][1], yaw_start_deg=math.degrees(rb['base'][2]))   # 舊版紀錄無此欄＝名目
    bx0, by0 = float(B['center_xy'][0]) + pt['bottle_dx'] + RO[0], float(B['center_xy'][1]) + pt['bottle_dy'] + RO[1]
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([bx0, by0, T['top_z'] + B['height'] / 2]),
                             radius=B['radius'], height=B['height'], mass=B['mass'], color=np.array([0.2, 0.6, 0.9]))
    import bottle_place_ctrl as PCm                                      # 第二桌：同 isaac_bottle_place（位移＝放置點 x − 名目瓶 x、lane_y）
    T2D = np.array([PCm.PLACE['target'][0] - float(B['center_xy'][0]), PCm.PLACE['lane_y'], 0.0])
    T = T2                                                                   # 以下第二桌用資產高度
    hz = T['top_z'] - T['thick']
    FixedCuboid(prim_path='/World/table2/top', name='table2_top', position=np.array([0.0, T['near_y'] + T['size_y'] / 2, T['top_z'] - T['thick'] / 2]) + T2D,
                scale=np.array([T['size_x'], T['size_y'], T['thick']]), color=np.array([0.45, 0.50, 0.62]))
    for sx in (-1, 1):
        lx = sx * (T['size_x'] / 2 - T['leg_inset_x'] - T['leg'] / 2)
        for nm, ly in (('near', T['near_y'] + T['leg_inset_near'] + T['leg'] / 2), ('far', T['near_y'] + T['size_y'] - T['leg_inset_far'] - T['leg'] / 2)):
            FixedCuboid(prim_path=f'/World/table2/leg_{nm}_{"p" if sx > 0 else "m"}', name=f'leg2_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + T2D,
                        scale=np.array([T['leg'], T['leg'], hz]), color=np.array([0.35, 0.38, 0.45]))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pt['x_start']), float(pt['y_start']), 0.0))
    xf.AddRotateZOp().Set(float(pt['yaw_start_deg']))
    apply_split(stage, ROBOT)
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    bodies, arts = physics_parts(stage, ROBOT)
    b_eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef']
    if len(b_eef) != 1:
        print(f'[rp] link_eef 剛體不符 {b_eef} ⇒ 拒跑', flush=True)
        return 16
    if not arts:
        print('[rp] 找不到 articulation root ⇒ 拒跑', flush=True)
        return 15

    def res(s):
        return tuple(int(v) for v in s.split('x'))

    cam_m = Camera(prim_path='/World/cam_main', resolution=res(a.res))
    cam_t = Camera(prim_path='/World/cam_top', resolution=res(a.res_wide))
    cam_w = Camera(prim_path='/World/cam_wide2', resolution=res(a.res_wide2))
    import wrist_v0_capture as W0
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', 'camera_color_optical_frame')
    cam_h = Camera(prim_path=f'{b_eef[0]}/w1_cam', resolution=res(a.res_hand))
    world.reset()
    cam_m.initialize()
    cam_w.initialize()
    cam_t.initialize()
    cam_h.initialize()
    cam_h.set_horizontal_aperture(0.020)                                  # 同 bottle_w1_cam.CAMS['w1']
    cam_h.set_focal_length(0.020 / (2.0 * math.tan(math.radians(69.0) / 2.0)))
    cam_h.set_clipping_range(0.05, 10.0)
    cam_h.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')

    def aim(path, eye, at, foc=None, up=(0, 0, 1)):
        if foc is not None:
            UsdGeom.Camera(stage.GetPrimAtPath(path)).GetFocalLengthAttr().Set(float(foc))
        x_ = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        x_.ClearXformOpOrder()
        x_.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*[float(v) for v in eye]), Gf.Vec3d(*[float(v) for v in at]), Gf.Vec3d(*up)).GetInverse())

    aim('/World/cam_wide2', [float(v) for v in a.wide_eye.split(',')], [float(v) for v in a.wide_at.split(',')], a.wide_foc)
    UsdGeom.Camera(stage.GetPrimAtPath('/World/cam_top')).GetFocalLengthAttr().Set(TOP_FOC)
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q0 = robot.get_joint_positions()
    for i, v in zip(idx, rb['q_tuck']):
        q0[i] = float(v)
    q0[f1], q0[f2] = float(c['finger1_q'][0]), float(c['finger2_q'][0])
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    act = np.array(sorted(idx + [f1, f2]), np.int32)
    kp[act], kd[act] = 600.0, 60.0
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    px = get_physx_interface()
    px.update_transformations(True, True)
    world.render()                                                        # 視窗立刻顯示 K2_READY，不停在模型預設全零
    for _ in range(100):                                                  # 只為取靜置後的底盤 z
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(q0, np.float32)[act], joint_indices=act))
        world.step(render=False)
    z0 = float(robot.get_world_pose()[0][2])
    t = c['t']
    every = max(1, int(round(1.0 / a.hz / dt)))
    pick = [i for i in range(0, len(t), every) if a.t0 <= t[i] <= a.t1]
    sx, sy = smooth(c['base_x'], dt), smooth(c['base_y'], dt)
    eo, ao = [float(v) for v in a.eye_ofs.split(',')], [float(v) for v in a.at_ofs.split(',')]
    print(f'[rp] 來源 {a.src}：{len(t)} 步 ⇒ {len(pick)} 幀（每 {every} 步）；底盤 z {z0:.4f}；相機 eye {eo} at {ao} foc {a.foc}', flush=True)
    nj = robot.num_dof
    aim('/World/cam_main', (sx[pick[0]] + eo[0], sy[pick[0]] + eo[1], eo[2]), (sx[pick[0]] + ao[0], sy[pick[0]] + ao[1], ao[2]), a.foc)
    csv = open(os.path.join(a.out, 'frames.csv'), 'w')
    csv.write('frame,sim_t,step,img_ok,pose_ok,dev_base_mm,dev_yaw_deg,dev_joint_mrad,dev_bottle_mm,retry\n')
    n_bad_img = n_bad_pose = 0
    worst = dict(base_mm=0.0, yaw_deg=0.0, joint_mrad=0.0, bottle_mm=0.0)
    for n, i in enumerate(pick, start=1):
        x, y, yw = float(c['base_x'][i]), float(c['base_y'][i]), float(c['base_yaw'][i])
        robot.set_world_pose(np.array([x, y, z0], np.float32), np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)], np.float32))
        qq = np.array(q0, np.float32)
        src_j = [float(c[f'q{j}'][i]) for j in range(1, 7)] + [float(c['finger1_q'][i]), float(c['finger2_q'][i])]
        for j, v in zip(idx + [f1, f2], src_j):
            qq[j] = v
        robot.set_joint_positions(qq)
        robot.set_joint_velocities(np.zeros(nj, np.float32))
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=qq[act], joint_indices=act))
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))
        qb = np.array([c['bottle_qw'][i], c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i]], float)
        pb = np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]], float) + quat_R(qb)[:, 2] * float(B['height']) / 2     # 底心 → 幾何中心
        bottle.set_world_pose(pb.astype(np.float32), qb.astype(np.float32))
        bottle.set_linear_velocity(np.zeros(3, np.float32))
        bottle.set_angular_velocity(np.zeros(3, np.float32))
        aim('/World/cam_main', (sx[i] + eo[0], sy[i] + eo[1], eo[2]), (sx[i] + ao[0], sy[i] + ao[1], ao[2]))
        aim('/World/cam_top', (sx[i], sy[i] + TOP_YOFS, TOP_H), (sx[i], sy[i] + TOP_YOFS, 0.0), up=(0, 1, 0))
        px.update_transformations(True, True)
        for _ in range(a.renders):
            world.render()

        def read_pose():
            p, o = robot.get_world_pose()
            ja = np.asarray(robot.get_joint_positions(), float)
            pb2, _ = bottle.get_world_pose()
            dy = yaw_of(o) - yw
            return dict(base_mm=math.hypot(float(p[0]) - x, float(p[1]) - y) * 1e3, yaw_deg=abs(math.degrees(math.atan2(math.sin(dy), math.cos(dy)))),
                        joint_mrad=max(abs(float(ja[j]) - v) for j, v in zip(idx + [f1, f2], src_j)) * 1e3, bottle_mm=float(np.linalg.norm(np.asarray(pb2, float) - pb)) * 1e3)
        img, retry, dev = REPC.capture(cam_m.get_rgba, world.render, read_pose, max_retry=3)
        img2, img3, img4 = cam_w.get_rgba(), cam_t.get_rgba(), cam_h.get_rgba()
        ok_i = REPC.has_image(img) and REPC.has_image(img2) and REPC.has_image(img3) and REPC.has_image(img4)
        ok_p = all(dev[k] <= TOL[k] for k in TOL)
        n_bad_img += not ok_i
        n_bad_pose += not ok_p
        for k in worst:
            worst[k] = max(worst[k], dev[k])
        if ok_i:
            Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(a.out, 'frames', f'm_{n:05d}.jpg'), quality=95)
            Image.fromarray(np.asarray(img2)[..., :3]).save(os.path.join(a.out, 'frames', f'w_{n:05d}.jpg'), quality=92)
            Image.fromarray(np.asarray(img3)[..., :3]).save(os.path.join(a.out, 'frames', f't_{n:05d}.jpg'), quality=92)
            Image.fromarray(np.asarray(img4)[..., :3]).save(os.path.join(a.out, 'frames', f'h_{n:05d}.jpg'), quality=94)
        csv.write(f"{n},{t[i]:.3f},{i},{int(ok_i)},{int(ok_p)},{dev['base_mm']:.3f},{dev['yaw_deg']:.4f},{dev['joint_mrad']:.3f},{dev['bottle_mm']:.3f},{retry}\n")
        if n % 100 == 0:
            csv.flush()
            print(f'[rp] {n}/{len(pick)}', flush=True)
    csv.close()
    res_ = dict(src=a.src, n_frames=len(pick), hz=1.0 / (every * dt), n_bad_img=n_bad_img, n_bad_pose=n_bad_pose, worst=worst, tol=TOL, z0=z0, mode='nostep', eye_ofs=eo, at_ofs=ao, foc=a.foc, hand=dict(T_mount=T_mount.tolist(), hfov_deg=69.0, clip=[0.05, 10.0], res=a.res_hand), wide=dict(eye=a.wide_eye, at=a.wide_at, foc=a.wide_foc))
    json.dump(res_, open(os.path.join(a.out, 'replay_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'[rp] 結束 {res_}', flush=True)
    with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as f:
        f.write('0' if n_bad_img == 0 and n_bad_pose == 0 else '5')
    return 0


rc = 99
try:
    rc = main()
except BaseException:
    import traceback
    print('[rp] 例外：' + traceback.format_exc().replace('\n', ' | '), flush=True)
finally:
    sim_app.close()
sys.exit(rc)
