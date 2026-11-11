#!/usr/bin/env python3
"""住宅格局場景的可動資產放置（m8_house_r1）：抽屜櫃加偏航、代理物依偏航放置。既有 drawer_asset.build_usd／marker_proxy_asset 不改（偏航 0 的既有趟次前提不動）。
抽屜偏航作法：櫃體與抽屜兩個 Xform 原本都只平移到 (ox, oy, 0)；在平移之後各加一個 RotateZ（繞自身原點＝櫃心），滑動關節的世界側框架 localRot0 同乘 Rz(yaw)。
**偏航版的滑動、限位尚未做拉動驗證**（只做過靜態預覽）。"""
import math

import numpy as np


def build_drawer_yaw(stage, DRA, spec, pose, root='/World/drawer_unit'):
    from pxr import Gf, UsdGeom, UsdPhysics
    ox, oy, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    rb = DRA.build_usd(stage, spec, (ox, oy, 0.0), root=root)
    for sub in ('cabinet', 'drawer'):
        UsdGeom.Xformable(stage.GetPrimAtPath(f'{root}/{sub}')).AddRotateZOp().Set(yaw)
    j = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f'{root}/drawer_slide'))
    h = math.radians(yaw + 180.0) / 2.0                             # 原本 localRot0＝繞 Z 180°；再乘偏航
    j.GetLocalRot0Attr().Set(Gf.Quatf(float(math.cos(h)), Gf.Vec3f(0.0, 0.0, float(math.sin(h)))))
    return dict(rb, yaw_deg=yaw)


def drawer_display_offset(stage, pose, opening, root='/World/drawer_unit'):
    """顯示用：把抽屜剛體沿拉出方向平移 opening（不是物理拉開）。拉出方向＝本地 −y 轉偏航。"""
    from pxr import Gf, UsdGeom
    yaw = math.radians(float(pose[2]))
    d = np.array([math.sin(yaw), -math.cos(yaw)]) * float(opening)
    UsdGeom.Xformable(stage.GetPrimAtPath(f'{root}/drawer')).GetOrderedXformOps()[0].Set(Gf.Vec3d(float(pose[0]) + d[0], float(pose[1]) + d[1], 0.0))
    return d


def build_marker_yaw(world, MK, spec, center_world, yaw_deg, opening_vec=(0.0, 0.0)):
    """代理物（獨立動態剛體）：世界中心與桿軸偏航由登錄檔給。"""
    from pxr import Gf, UsdGeom
    obj, mat = MK.build(world, spec, (0.0, 0.0, 0.0))
    ops = UsdGeom.Xformable(obj.prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(float(center_world[0]) + float(opening_vec[0]), float(center_world[1]) + float(opening_vec[1]), float(center_world[2])))
    h = math.radians(float(yaw_deg)) / 2.0
    ops[1].Set(Gf.Quatd(math.cos(h), 0.0, 0.0, math.sin(h)))
    pos = np.array([float(center_world[0]) + float(opening_vec[0]), float(center_world[1]) + float(opening_vec[1]), float(center_world[2])])
    obj.set_default_state(position=pos, orientation=np.array([math.cos(h), 0.0, 0.0, math.sin(h)]))   # world.reset 會回到預設狀態：r1 預覽時代理物被放回建構時的原點附近
    return obj, mat
