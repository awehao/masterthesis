#!/usr/bin/env python3
"""粗馬克筆代理物（marker_proxy_m8_hex.yaml，正六角柱）的 Isaac 建構：**獨立動態剛體**＋凸包碰撞＋物理材料，放在抽屜底板上（世界座標＝抽屜位姿＋抽屜座標配置）。
不掛在抽屜剛體底下、不加任何關節、不加阻尼——只靠接觸停在抽屜裡（Codex 20261111_180841 第 2 項、183534）。外觀與碰撞是同一個六角網格。
圓柱版（schema object_proxy/1）靜置核對 r1／r2 未過，本檔不再支援。"""
import math

import numpy as np
import yaml


def load(path):
    return yaml.safe_load(open(path, encoding='utf-8'))


def world_center(spec, drawer_pose, opening=0.0):
    c = spec['placement']['center']
    return np.array([drawer_pose[0] + c[0], drawer_pose[1] + c[1] - opening, c[2]], float)   # 抽屜沿世界 −y 拉出


def hex_points(spec):
    """本體座標頂點：軸沿 x；頂點在 ±y（對角），平面朝 ±z（對邊）。"""
    R, h = float(spec['circumradius']), float(spec['length']) / 2
    ring = [(R * math.cos(math.radians(60 * k)), R * math.sin(math.radians(60 * k))) for k in range(6)]
    return [(-h, y, z) for y, z in ring] + [(h, y, z) for y, z in ring]


def build(world, spec, drawer_pose, prim_path='/World/marker_proxy', opening=0.0):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleRigidPrim
    assert spec['shape'] == 'hex_prism' and spec['axis'] == 'x'
    stage = world.stage
    m = spec['material']
    mat = PhysicsMaterial(prim_path=prim_path + '_material', static_friction=float(m['static_friction']), dynamic_friction=float(m['dynamic_friction']),
                          restitution=float(m['restitution']))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    pts = hex_points(spec)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    faces = [[k, (k + 1) % 6, 6 + (k + 1) % 6, 6 + k] for k in range(6)] + [[5, 4, 3, 2, 1, 0], [6, 7, 8, 9, 10, 11]]
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.55, 0.25)])
    a = np.asarray(pts)
    mesh.CreateExtentAttr([Gf.Vec3f(*a.min(0).tolist()), Gf.Vec3f(*a.max(0).tolist())])
    prim = mesh.GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*world_center(spec, drawer_pose, opening).tolist()))
    UsdGeom.Xformable(prim).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(spec['mass_kg']))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat.material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    obj = SingleRigidPrim(prim_path, name='marker_proxy')
    world.scene.add(obj)
    return obj, mat


def readback(stage, obj, mat, prim_path='/World/marker_proxy'):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    bound = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel('physics').GetTargets()
    um = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(prim_path + '_material'))
    ada = PhysxSchema.PhysxRigidBodyAPI(prim).GetAngularDampingAttr()
    return dict(parent=str(prim.GetParent().GetPath()), type=str(prim.GetTypeName()), rigid_body=prim.HasAPI(UsdPhysics.RigidBodyAPI),
                rigid_enabled=bool(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()),
                kinematic=bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() or False),
                collision=prim.HasAPI(UsdPhysics.CollisionAPI), collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                collision_approx=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()),
                angular_damping_authored=bool(ada and ada.HasAuthoredValue()),
                mass_usd=float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()), mass_sim=float(obj.get_mass()),
                n_points=int(len(pts)), size_xyz=[round(float(v), 6) for v in (pts.max(0) - pts.min(0))], scale=[float(v) for v in obj.get_local_scale()],
                material_bound=[str(t) for t in bound], static_friction=float(um.GetStaticFrictionAttr().Get()), dynamic_friction=float(um.GetDynamicFrictionAttr().Get()),
                restitution=float(um.GetRestitutionAttr().Get()))
