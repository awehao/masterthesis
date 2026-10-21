#!/usr/bin/env python3
"""K0：局部通行門的 USD 建構與讀回（類比 drawer_asset_v2.build_usd）。只建舞台與讀回屬性；物理行為由 K1 短測驗證。

結構（root＝門根 Xform，放在 placement 位姿；其下用 door_local 規格的局部座標）：
    {root}/frame/wall_hinge_side, wall_latch_side    靜態碰撞方盒（門洞挖空）
    {root}/leaf                                      剛體 Xform，原點＝鉸鏈軸（門角 0 時與 root 重合）
        leaf_box、post0、post1、handle_bar           碰撞幾何（門扇局部 (u, v, z)）
    {root}/hinge                                     RevoluteJoint：body0 空＝錨在世界、body1＝leaf

關節約定：兩端框架都繞 X 轉 180°，使關節 +Z＝世界 −Z（door_local：門扇繞 +z 轉 −q 為開門）⇒ **關節角就是門角 q**（度），
正值＝朝機器人側拉開，不在讀數上補負號。限位 [0, limit_q_deg[1]]。
**明確不裝驅動**：寫入零剛度、零阻尼、零最大力的角驅動並讀回，讓「沒有隱藏位置保持」成為可查核的事實。
阻尼用剛體角阻尼（PhysxRigidBodyAPI.angularDamping）；質量、質心、主慣量明寫（門扇方盒；把手質量忽略並揭露）。
"""
from __future__ import annotations

import math

import yaml


def register_physx_plugins():
    """離線（不啟動 Kit）時註冊 PhysX schema 外掛；由 run_usd_offline.sh 設定 USD_OFFLINE_PHYSX_PLUGINS。在 Kit 內不需要。"""
    import glob
    import os
    root = os.environ.get('USD_OFFLINE_PHYSX_PLUGINS')
    if root:
        from pxr import Plug
        for p in glob.glob(os.path.join(root, '**', 'plugInfo.json'), recursive=True):
            Plug.Registry().RegisterPlugins(os.path.dirname(p))


def load(path: str) -> dict:
    d = yaml.safe_load(open(path))
    for k in ('wall', 'frame_clear_width', 'leaf', 'handle', 'physics', 'placement'):
        if k not in d:
            raise ValueError(f'門規格缺 {k}')
    if 'height' not in d['handle']:
        raise ValueError('門規格缺 handle.height（K0 須定一個把手高）')
    lo, hi = d['leaf']['limit_q_deg']
    if not (lo == 0.0 and 0.0 < hi <= 180.0):
        raise ValueError('limit_q_deg 須為 [0, (0, 180]]')
    if d['physics'].get('drive') != 'none':
        raise ValueError('K0 只允許 drive: none（被動門）')
    return d


def leaf_inertia(d):
    m = float(d['physics']['leaf_mass_kg'])
    L, t = d['leaf']['length'], d['leaf']['thickness']
    z0, z1 = d['leaf']['z']
    Hh = z1 - z0
    # 門扇局部 (u, v, z) 方盒對質心主慣量
    return m, (L / 2, t / 2, (z0 + z1) / 2), (m / 12 * (t * t + Hh * Hh), m / 12 * (L * L + Hh * Hh), m / 12 * (L * L + t * t))


def build_usd(stage, d: dict, root: str = '/World/door'):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    p = d['placement']['pose']
    ox, oy, yaw = float(p['x']), float(p['y']), math.radians(float(p['yaw_deg']))
    rx = UsdGeom.Xform.Define(stage, root)
    xf = UsdGeom.Xformable(rx)
    xf.AddTranslateOp().Set(Gf.Vec3d(ox, oy, 0.0))
    xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(math.degrees(yaw))

    def cube(path, c, s):
        g = UsdGeom.Cube.Define(stage, path)
        g.GetSizeAttr().Set(2.0)
        x = UsdGeom.Xformable(g)
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) / 2.0 for v in s]))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        return g

    W, T, Ht = d['frame_clear_width'], d['wall']['thickness'], d['wall']['height']
    x0, x1 = d['wall']['extent_x']
    fpath = f'{root}/frame'
    UsdGeom.Xform.Define(stage, fpath)
    cube(f'{fpath}/wall_hinge_side', (x0 / 2, T / 2, Ht / 2), (-x0, T, Ht))
    cube(f'{fpath}/wall_latch_side', ((W + x1) / 2, T / 2, Ht / 2), (x1 - W, T, Ht))

    lpath = f'{root}/leaf'
    lx = UsdGeom.Xform.Define(stage, lpath)
    UsdPhysics.RigidBodyAPI.Apply(lx.GetPrim())
    m, com, Idiag = leaf_inertia(d)
    ma = UsdPhysics.MassAPI.Apply(lx.GetPrim())
    ma.CreateMassAttr().Set(m)
    ma.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in com]))
    ma.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*[float(v) for v in Idiag]))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(lx.GetPrim())
    rb.CreateAngularDampingAttr().Set(float(d['physics']['angular_damping']))
    L, t = d['leaf']['length'], d['leaf']['thickness']
    z0, z1 = d['leaf']['z']
    cube(f'{lpath}/leaf_box', (L / 2, t / 2, (z0 + z1) / 2), (L, t, z1 - z0))
    Hd = d['handle']
    h = float(Hd['height'])
    su, sn, sz = Hd['posts']['size']
    for k, du in enumerate(Hd['posts']['along_u']):
        cube(f'{lpath}/post{k}', (Hd['from_hinge'] + du, -sn / 2, h), (su, sn, sz))
    bar = UsdGeom.Cylinder.Define(stage, f'{lpath}/handle_bar')
    bar.GetAxisAttr().Set('X')
    bar.GetRadiusAttr().Set(float(Hd['bar']['radius']))
    bar.GetHeightAttr().Set(float(Hd['bar']['length']))
    UsdGeom.Xformable(bar).AddTranslateOp().Set(Gf.Vec3d(float(Hd['from_hinge']), -float(Hd['standoff']), h))
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())

    j = UsdPhysics.RevoluteJoint.Define(stage, f'{root}/hinge')
    j.CreateBody1Rel().SetTargets([Sdf.Path(lpath)])          # body0 空＝錨在世界
    j.CreateAxisAttr().Set('Z')
    flip = Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))              # 繞 X 180°：關節 +Z＝世界 −Z
    rz = Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2)))
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(ox, oy, 0.0))          # 世界錨點＝鉸鏈軸（＝root 原點）
    j.CreateLocalRot0Attr().Set(rz * flip)
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr().Set(flip)
    j.CreateLowerLimitAttr().Set(float(d['leaf']['limit_q_deg'][0]))
    j.CreateUpperLimitAttr().Set(float(d['leaf']['limit_q_deg'][1]))
    pj = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
    pj.CreateJointFrictionAttr().Set(float(d['physics']['joint_friction']))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'angular')
    for a, v in (('Stiffness', 0.0), ('Damping', 0.0), ('TargetPosition', 0.0), ('TargetVelocity', 0.0), ('MaxForce', 0.0)):
        getattr(drv, f'Create{a}Attr')().Set(v)
    bind_handle_material(stage, d, root)
    return readback(stage, root)


def bind_handle_material(stage, d, root='/World/door', mat_path='/World/PhysicsMaterials/door_handle'):
    """把手橫桿的摩擦**明確指定**（與 isaac_common.bind_friction 同機制：physics 用途、weakerThanDescendants）。
    模擬端若另以夾持材質同時綁手指＋橫桿，會覆寫本綁定——K2 讀回時以實際綁定為準並核對數值等於規格。"""
    from pxr import Sdf, UsdPhysics, UsdShade
    m = d['physics']['handle_material']
    stage.DefinePrim(Sdf.Path(mat_path).GetParentPath(), 'Scope')
    mp = stage.DefinePrim(mat_path, 'Material')
    api = UsdPhysics.MaterialAPI.Apply(mp)
    api.CreateStaticFrictionAttr().Set(float(m['static_friction']))
    api.CreateDynamicFrictionAttr().Set(float(m['dynamic_friction']))
    api.CreateRestitutionAttr().Set(0.0)
    bar = stage.GetPrimAtPath(f'{root}/leaf/handle_bar')
    UsdShade.MaterialBindingAPI.Apply(bar).Bind(UsdShade.Material(mp), UsdShade.Tokens.weakerThanDescendants, 'physics')


def handle_material_readback(stage, root='/World/door'):
    """讀回把手橫桿**實際生效**的 physics 材質（沿祖先找直接綁定）與其摩擦值。"""
    from pxr import UsdPhysics, UsdShade
    pr = stage.GetPrimAtPath(f'{root}/leaf/handle_bar')
    cur = pr
    while cur and cur.IsValid():
        rel = UsdShade.MaterialBindingAPI(cur).GetDirectBindingRel('physics')
        if rel and rel.GetTargets():
            mp = stage.GetPrimAtPath(rel.GetTargets()[0])
            api = UsdPhysics.MaterialAPI(mp)
            return {'bound_at': str(cur.GetPath()), 'material': str(mp.GetPath()),
                    'static_friction': float(api.GetStaticFrictionAttr().Get()),
                    'dynamic_friction': float(api.GetDynamicFrictionAttr().Get()),
                    'restitution': float(api.GetRestitutionAttr().Get())}
        cur = cur.GetParent()
    return {'bound_at': None}


def readback(stage, root='/World/door'):
    """從舞台讀回（不依賴建構時的變數）。"""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics
    lp = stage.GetPrimAtPath(f'{root}/leaf')
    jp = stage.GetPrimAtPath(f'{root}/hinge')
    j = UsdPhysics.RevoluteJoint(jp)
    ma = UsdPhysics.MassAPI(lp)
    drv = UsdPhysics.DriveAPI(jp, 'angular')
    rq = lambda q: [float(q.GetReal()), *[float(x) for x in q.GetImaginary()]]   # noqa: E731
    bar = UsdGeom.Cylinder(stage.GetPrimAtPath(f'{root}/leaf/handle_bar'))
    bar_t = UsdGeom.Xformable(bar).GetOrderedXformOps()[0].Get()
    return {
        'root': root, 'leaf_prim': f'{root}/leaf', 'joint_prim': f'{root}/hinge', 'handle_bar_prim': f'{root}/leaf/handle_bar',
        'is_rigid_body': lp.HasAPI(UsdPhysics.RigidBodyAPI),
        'mass_kg': float(ma.GetMassAttr().Get()), 'center_of_mass': [float(x) for x in ma.GetCenterOfMassAttr().Get()],
        'diagonal_inertia': [float(x) for x in ma.GetDiagonalInertiaAttr().Get()],
        'angular_damping': float(PhysxSchema.PhysxRigidBodyAPI(lp).GetAngularDampingAttr().Get()),
        'joint_type': jp.GetTypeName(), 'joint_axis': j.GetAxisAttr().Get(),
        'joint_body0': [str(x) for x in j.GetBody0Rel().GetTargets()], 'joint_body1': [str(x) for x in j.GetBody1Rel().GetTargets()],
        'joint_local_pos0': [float(x) for x in j.GetLocalPos0Attr().Get()], 'joint_local_rot0_wxyz': rq(j.GetLocalRot0Attr().Get()),
        'joint_local_pos1': [float(x) for x in j.GetLocalPos1Attr().Get()], 'joint_local_rot1_wxyz': rq(j.GetLocalRot1Attr().Get()),
        'limit_lower_deg': float(j.GetLowerLimitAttr().Get()), 'limit_upper_deg': float(j.GetUpperLimitAttr().Get()),
        'joint_friction': float(PhysxSchema.PhysxJointAPI(jp).GetJointFrictionAttr().Get()),
        'drive_stiffness': float(drv.GetStiffnessAttr().Get()), 'drive_damping': float(drv.GetDampingAttr().Get()),
        'drive_max_force': float(drv.GetMaxForceAttr().Get()),
        'handle_bar': {'axis': bar.GetAxisAttr().Get(), 'radius': float(bar.GetRadiusAttr().Get()), 'length': float(bar.GetHeightAttr().Get()),
                       'center_leaf_local': [float(x) for x in bar_t]},
        'handle_material': handle_material_readback(stage, root),
        'collision_prims': sorted(str(p.GetPath()) for p in stage.Traverse()
                                  if str(p.GetPath()).startswith(root) and p.HasAPI(UsdPhysics.CollisionAPI)),
    }
