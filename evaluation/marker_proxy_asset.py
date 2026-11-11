#!/usr/bin/env python3
"""粗馬克筆代理物（marker_proxy_m8.yaml）的 Isaac 建構：**獨立動態剛體**＋碰撞＋物理材料，放在抽屜底板上（世界座標＝抽屜位姿＋抽屜座標配置）。
不掛在抽屜剛體底下、不加任何關節——只靠接觸停在抽屜裡，所以可以被真正取出（Codex 20261111_180841 第 2 項）。"""
import math

import numpy as np
import yaml


def load(path):
    return yaml.safe_load(open(path, encoding='utf-8'))


def world_center(spec, drawer_pose, opening=0.0):
    c = spec['placement']['center']
    return np.array([drawer_pose[0] + c[0], drawer_pose[1] + c[1] - opening, c[2]], float)   # 抽屜沿世界 −y 拉出


def build(world, spec, drawer_pose, prim_path='/World/marker_proxy', opening=0.0):
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCylinder
    m = spec['material']
    mat = PhysicsMaterial(prim_path=prim_path + '_material', static_friction=float(m['static_friction']), dynamic_friction=float(m['dynamic_friction']),
                          restitution=float(m['restitution']))
    assert spec['shape'] == 'cylinder' and spec['axis'] == 'x'
    obj = DynamicCylinder(prim_path=prim_path, name='marker_proxy', position=world_center(spec, drawer_pose, opening),
                          orientation=np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]),      # 圓柱預設軸 z → 繞 y 轉 90° 對到 x
                          radius=float(spec['radius']), height=float(spec['length']), mass=float(spec['mass_kg']),
                          color=np.array([0.05, 0.55, 0.25]), physics_material=mat)
    ad = float(spec.get('physics', {}).get('angular_damping', 0.0))
    if ad > 0.0:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim).CreateAngularDampingAttr().Set(ad)
    world.scene.add(obj)
    return obj, mat


def readback(stage, obj, mat, prim_path='/World/marker_proxy'):
    from pxr import UsdGeom, UsdPhysics, UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    g = UsdGeom.Cylinder(prim)
    bound = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel('physics').GetTargets()
    mp = stage.GetPrimAtPath(prim_path + '_material')
    um = UsdPhysics.MaterialAPI(mp)
    from pxr import PhysxSchema
    ada = PhysxSchema.PhysxRigidBodyAPI(prim).GetAngularDampingAttr()
    return dict(angular_damping=float(ada.Get()) if ada and ada.Get() is not None else None, parent=str(prim.GetParent().GetPath()), rigid_body=prim.HasAPI(UsdPhysics.RigidBodyAPI),
                rigid_enabled=bool(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None,
                kinematic=bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() or False),
                collision=prim.HasAPI(UsdPhysics.CollisionAPI), collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
                mass_usd=float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()), mass_sim=float(obj.get_mass()),
                radius=float(g.GetRadiusAttr().Get()), height=float(g.GetHeightAttr().Get()), scale=[float(v) for v in obj.get_local_scale()],
                material_bound=[str(t) for t in bound], static_friction=float(um.GetStaticFrictionAttr().Get()), dynamic_friction=float(um.GetDynamicFrictionAttr().Get()),
                restitution=float(um.GetRestitutionAttr().Get()))
