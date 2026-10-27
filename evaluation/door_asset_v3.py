#!/usr/bin/env python3
"""【v3，2026-10-27】雙面把手門資產（Codex r45 §4；規格 door_local_v3.yaml）。door_asset_v2.py 不改；本檔沿用其建構，**只新增**門扇 −n 面
（楔形／+y 側）的內側把手：
    {root}/leaf/post_in0、post_in1、handle_bar_inner   碰撞幾何（門扇局部 (u, v, z)；v＝門扇厚度方向，門扇占 v ∈ [0, thickness]）
  內側橫桿：沿 u（Cylinder axis X）、桿心 (from_hinge, thickness＋standoff, height)；立柱：中心 (from_hinge＋du, thickness＋sn/2, height)、尺寸同外側。
  與外側（handle_bar 桿心 (from_hinge, −standoff, height)、post0／post1 中心 v＝−sn/2）對門扇中面鏡像；與 door_geom_v3.handle_pose_inner／shapes 同一定義。
  碰撞偏移（contactOffset／restOffset）與摩擦材質（door_handle：μs 1.0、μd 0.8、restitution 0）與外側相同並各自讀回；外側 prim 名稱不變（相容既有模擬端）。
不改：門扇、鉸鏈（RevoluteJoint、零驅動、限位、摩擦）、牆、質量、質心、慣量、角／線阻尼——全部沿用 v2。
  把手（兩支橫桿與四根立柱）質量忽略，與 v2 相同並揭露；內側把手不改質心與慣量。
讀回：door_asset_v2.readback 全部欄位＋內側橫桿（軸、半徑、長度、門扇局部桿心）、內側立柱（中心、半尺寸）、兩支橫桿各自實際生效的 physics 材質、
  分別的 prim 路徑（handle_bar_prim／handle_bar_inner_prim、post_prims／post_inner_prims），供模擬端把接觸分到「外側橫桿／內側橫桿／立柱／門扇」而不混稱。
界線：只建舞台與讀回；不啟動物理演化、不含接觸行為驗證。
"""
from __future__ import annotations

import door_asset_v2 as A2

register_physx_plugins = A2.register_physx_plugins
leaf_inertia = A2.leaf_inertia


def load(path: str) -> dict:
    d = A2.load(path)
    hi = d.get('handle_inner')
    if not hi or hi.get('mirror_of') != 'handle':
        raise ValueError('v3 規格須有 handle_inner（mirror_of: handle）')
    for k in ('from_hinge', 'standoff', 'height'):
        if k not in hi:
            raise ValueError(f'handle_inner 缺 {k}')
    return d


def _offsets(stage, prim, d):
    from pxr import PhysxSchema
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateContactOffsetAttr().Set(float(d['collision']['contact_offset']))
    pc.CreateRestOffsetAttr().Set(float(d['collision']['rest_offset']))


def build_usd(stage, d: dict, root: str = '/World/door'):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    A2.build_usd(stage, d, root)
    lpath = f'{root}/leaf'
    H, Hi, t = d['handle'], d['handle_inner'], float(d['leaf']['thickness'])
    h = float(Hi['height'])
    su, sn, sz = H['posts']['size']
    for k, du in enumerate(H['posts']['along_u']):
        g = UsdGeom.Cube.Define(stage, f'{lpath}/post_in{k}')
        g.GetSizeAttr().Set(2.0)
        x = UsdGeom.Xformable(g)
        x.AddTranslateOp().Set(Gf.Vec3d(float(Hi['from_hinge'] + du), t + sn / 2, h))
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(su / 2, sn / 2, sz / 2))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        _offsets(stage, g.GetPrim(), d)
    bar = UsdGeom.Cylinder.Define(stage, f'{lpath}/handle_bar_inner')
    bar.GetAxisAttr().Set('X')
    bar.GetRadiusAttr().Set(float(H['bar']['radius']))
    bar.GetHeightAttr().Set(float(H['bar']['length']))
    UsdGeom.Xformable(bar).AddTranslateOp().Set(Gf.Vec3d(float(Hi['from_hinge']), t + float(Hi['standoff']), h))
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())
    _offsets(stage, bar.GetPrim(), d)
    mp = stage.GetPrimAtPath('/World/PhysicsMaterials/door_handle')
    UsdShade.MaterialBindingAPI.Apply(bar.GetPrim()).Bind(UsdShade.Material(mp), UsdShade.Tokens.weakerThanDescendants, 'physics')
    return readback(stage, root)


def _material_of(stage, path):
    from pxr import UsdPhysics, UsdShade
    cur = stage.GetPrimAtPath(path)
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


def _cube(stage, path):
    from pxr import UsdGeom
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()
    return {'center_leaf_local': [float(v) for v in ops[0].Get()], 'half_size': [float(v) for v in ops[1].Get()]}


def readback(stage, root='/World/door'):
    from pxr import UsdGeom
    out = A2.readback(stage, root)
    lpath = f'{root}/leaf'
    bar = UsdGeom.Cylinder(stage.GetPrimAtPath(f'{lpath}/handle_bar_inner'))
    out['handle_bar_inner_prim'] = f'{lpath}/handle_bar_inner'
    out['handle_bar_inner'] = {'axis': bar.GetAxisAttr().Get(), 'radius': float(bar.GetRadiusAttr().Get()), 'length': float(bar.GetHeightAttr().Get()),
                               'center_leaf_local': [float(v) for v in UsdGeom.Xformable(bar).GetOrderedXformOps()[0].Get()]}
    out['post_prims'] = [f'{lpath}/post0', f'{lpath}/post1']
    out['post_inner_prims'] = [f'{lpath}/post_in0', f'{lpath}/post_in1']
    out['posts'] = {p.split('/')[-1]: _cube(stage, p) for p in out['post_prims'] + out['post_inner_prims']}
    out['leaf_box'] = _cube(stage, f'{lpath}/leaf_box')
    out['handle_material_by_bar'] = {'handle_bar': _material_of(stage, f'{lpath}/handle_bar'),
                                     'handle_bar_inner': _material_of(stage, f'{lpath}/handle_bar_inner')}
    return out
