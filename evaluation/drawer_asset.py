"""被動抽屜櫃資產：離線幾何 ＋ Isaac USD 建構，同一份規格。

一份規格兩個用途，避免檢查用的幾何與模擬裡跑的幾何各寫一次而悄悄分岔
（這個專案已經為了「YAML 外接圓半徑」與「SDF 真實幾何」不一致付出過代價）。

    shapes_world(spec, pose, q_d)   離線檢查用的世界座標形狀清單
    min_distance(P, shapes)         點雲到這些形狀的最短距離（內部為負）
    build_usd(stage, spec, pose)    在 Isaac stage 上建出櫃體、抽屜與滑動關節

**抽屜是被動的**：build_usd 明確把 drive stiffness 與 damping 寫成 0，並在回傳
值裡帶出讀回的實際數值，呼叫端必須自己檢查。本模組不發任何開度命令。
"""
from __future__ import annotations

import math
import numpy as np
import yaml

SCHEMA = 'drawer_unit/1'


def load(path: str) -> dict:
    d = yaml.safe_load(open(path))
    if d.get('schema') != SCHEMA:
        raise ValueError(f'schema {d.get("schema")!r} != {SCHEMA!r}')
    return d


# --------------------------------------------------------------- 幾何
def shapes_world(spec: dict, pose, q_d: float = 0.0):
    """回傳 [(名稱, 種類, 參數...)]，世界座標。

    pose = (x, y)：櫃體底面中心的世界位置。本地 yaw 固定 0，所以本地軸 = 世界軸
    （規格檔已載明）；不支援旋轉擺放，需要時請改規格並同步改這裡，不要默默容忍。
    q_d：開度 m，抽屜沿本地 −y 平移這麼多。
    """
    ox, oy = float(pose[0]), float(pose[1])
    out = []
    for n, c, s in spec['cabinet']:
        out.append((f'cabinet/{n}', 'box',
                    np.array([c[0] + ox, c[1] + oy, c[2]]), np.array(s)))
    dy = -float(q_d)
    for n, c, s in spec['drawer']['body']:
        out.append((f'drawer/{n}', 'box',
                    np.array([c[0] + ox, c[1] + dy + oy, c[2]]), np.array(s)))
    h = spec['drawer']['handle']
    for n, c, s in h['posts']:
        out.append((f'handle/{n}', 'box',
                    np.array([c[0] + ox, c[1] + dy + oy, c[2]]), np.array(s)))
    b = h['bar']
    if b['shape'] != 'cylinder' or b['axis'] != 'x':
        raise ValueError('本模組只支援沿 x 的圓柱把手')
    out.append(('handle/bar', 'cyl',
                np.array([b['center'][0] + ox,
                          b['center'][1] + dy + oy, b['center'][2]]),
                float(b['radius']), float(b['length'])))
    return out


def _box_dist(P, c, sz):
    d = np.abs(P - c) - sz / 2.0
    return (np.linalg.norm(np.maximum(d, 0.0), axis=1)
            + np.minimum(d.max(axis=1), 0.0))


def _cyl_x_dist(P, c, r, L):
    """沿 x 的圓柱：軸向與徑向兩個帶號距離合成，與方塊同一套寫法。"""
    da = np.abs(P[:, 0] - c[0]) - L / 2.0
    dr = np.hypot(P[:, 1] - c[1], P[:, 2] - c[2]) - r
    out = np.hypot(np.maximum(da, 0.0), np.maximum(dr, 0.0))
    return out + np.minimum(np.maximum(da, dr), 0.0)


def min_distance(P, shapes):
    """(最短距離, 造成它的形狀名稱)。P 為 (N,3) 點雲。"""
    best, who = np.inf, None
    for item in shapes:
        if item[1] == 'box':
            d = _box_dist(P, item[2], item[3]).min()
        else:
            d = _cyl_x_dist(P, item[2], item[3], item[4]).min()
        if d < best:
            best, who = float(d), item[0]
    return best, who


def grasp_tcp_world(spec: dict, pose, q_d: float = 0.0):
    """夾持點的 TCP 世界位置。

    工具 z（接近軸）指向世界 +y，所以 TCP 比橫桿中心深 tcp_offset_along_tool_z。
    """
    b = spec['drawer']['handle']['bar']
    off = float(spec['grasp_surface']['tcp_offset_along_tool_z'])
    return np.array([b['center'][0] + float(pose[0]),
                     b['center'][1] - float(q_d) + float(pose[1]) + off,
                     b['center'][2]])


# --------------------------------------------------------------- Isaac
def build_usd(stage, spec: dict, pose, root: str = '/World/drawer_unit'):
    """建出櫃體（靜態）、抽屜（剛體）與滑動關節，回傳讀回的參數。

    關節框架繞 Z 轉 180°，使關節 +Y = 世界 −Y，這樣**關節座標值直接就是開度**，
    不必在讀數上補負號（補負號是日後判錯方向最常見的來源）。
    """
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

    ox, oy = float(pose[0]), float(pose[1])
    UsdGeom.Xform.Define(stage, root)

    def cube(path, c, s):
        p = UsdGeom.Cube.Define(stage, path)
        p.GetSizeAttr().Set(2.0)
        x = UsdGeom.Xformable(p)
        # 平移用 double、縮放用 double3：Gf.Vec3f 是 float32，0.45 之類的值會在
        # 讀回時差幾十 µm，看起來像幾何不一致，其實只是精度。
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*[float(v) / 2.0 for v in s]))
        UsdPhysics.CollisionAPI.Apply(p.GetPrim())
        return p

    # 兩個子 Xform 都放在 (ox, oy, 0)，底下的幾何用**規格裡的本地座標**。
    # 第一版把子幾何寫成世界絕對座標、Xform 本身不位移，結果滑動關節的
    # body1 框架在原點、body0 錨點在 (ox, oy, 0)，PhysX 報
    # "joint with disjointed body transforms" 並把抽屜瞬移過去 —— 限位的零點
    # 因此整個偏掉。關節兩端的框架必須真的重合。
    cpath = f'{root}/cabinet'
    cx = UsdGeom.Xform.Define(stage, cpath)
    UsdGeom.Xformable(cx).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.0))
    for n, c, s in spec['cabinet']:
        cube(f'{cpath}/{n}', c, s)

    dpath = f'{root}/drawer'
    dx = UsdGeom.Xform.Define(stage, dpath)
    UsdGeom.Xformable(dx).AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(dx.GetPrim())
    ph = spec['drawer']['physics']
    m = UsdPhysics.MassAPI.Apply(dx.GetPrim())
    m.GetMassAttr().Set(float(ph['mass_kg']))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(dx.GetPrim())
    rb.CreateLinearDampingAttr().Set(float(ph['linear_damping']))

    for n, c, s in spec['drawer']['body']:
        cube(f'{dpath}/{n}', c, s)
    h = spec['drawer']['handle']
    for n, c, s in h['posts']:
        cube(f'{dpath}/{n}', c, s)
    b = h['bar']
    cyl = UsdGeom.Cylinder.Define(stage, f'{dpath}/handle_bar')
    cyl.GetAxisAttr().Set('X')
    cyl.GetRadiusAttr().Set(float(b['radius']))
    cyl.GetHeightAttr().Set(float(b['length']))
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*b['center']))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    j = UsdPhysics.PrismaticJoint.Define(stage, f'{root}/drawer_slide')
    j.CreateBody1Rel().SetTargets([Sdf.Path(dpath)])      # body0 空 = 錨在世界
    j.CreateAxisAttr().Set('Y')
    # 兩端框架都繞 Z 轉 180°：關節 +Y 對到世界 −Y（抽屜拉出方向）
    rot = Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))
    # body0 空 ⇒ 錨在世界；錨點與抽屜 Xform 的原點同為 (ox, oy, 0)，兩端框架重合
    anchor = Gf.Vec3f(ox, oy, 0.0)
    j.CreateLocalPos0Attr().Set(anchor)
    j.CreateLocalRot0Attr().Set(rot)
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(rot)
    jt = spec['drawer']['joint']
    j.CreateLowerLimitAttr().Set(float(jt['lower']))
    j.CreateUpperLimitAttr().Set(float(jt['upper']))

    pj = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
    pj.CreateJointFrictionAttr().Set(float(ph['joint_friction']))

    # **明確不裝驅動**：抽屜不接受任何位置／速度命令。
    # 不是「沒寫就沒有」——寫成 0 並讀回，讓它成為可查核的事實。
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
    drv.CreateStiffnessAttr().Set(0.0)
    drv.CreateDampingAttr().Set(0.0)
    drv.CreateTargetPositionAttr().Set(0.0)
    drv.CreateTargetVelocityAttr().Set(0.0)
    drv.CreateMaxForceAttr().Set(0.0)

    return {
        'root': root,
        'cabinet_prim': cpath,
        'drawer_prim': dpath,
        'joint_prim': f'{root}/drawer_slide',
        'mass_kg': float(m.GetMassAttr().Get()),
        'linear_damping': float(rb.GetLinearDampingAttr().Get()),
        'joint_friction': float(pj.GetJointFrictionAttr().Get()),
        'drive_stiffness': float(drv.GetStiffnessAttr().Get()),
        'drive_damping': float(drv.GetDampingAttr().Get()),
        'drive_max_force': float(drv.GetMaxForceAttr().Get()),
        'limit_lower': float(j.GetLowerLimitAttr().Get()),
        'limit_upper': float(j.GetUpperLimitAttr().Get()),
    }
