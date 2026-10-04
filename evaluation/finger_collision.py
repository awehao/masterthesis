"""手指碰撞形狀：把 L 形手指網格拆成**兩個凸塊**（根部、指片）。

為什麼
------
URDF 匯入後手指碰撞是單一 convexHull。手指網格是 L 形：根部（手指座標
z ≤ 8 mm）內面在 y≈0，指片（z 8–28 mm）內面在 y = ±11.2 mm。整顆包成凸包會
把 L 的內角填成約 29° 的楔面 —— 夾持行為因此改變（10 mm 桿在凸包下卡得住、
在真實形狀下夾不到；見 results/wgmpc_drawer_open_close.yaml 的
gripper_geometry_check）。拆成兩塊各自取凸包，指片內面保持平的 11.2 mm。

只用網格自己的頂點，不手畫盒子：兩塊的凸包都是原頂點子集的凸包，
所以每一塊都**包含在**原網格的凸包內、且覆蓋原網格在該 z 段的所有頂點。
"""
from __future__ import annotations

import numpy as np

Z_SPLIT_M = 0.008      # 根部／指片分界（手指連桿座標 z）


def split_hulls(pts: np.ndarray, z_split: float = Z_SPLIT_M):
    """回傳 {'base': (V, F), 'blade': (V, F)}，V 頂點、F 三角形索引。

    分界上的頂點兩塊都收，兩塊在分界處接合，不留縫。
    """
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    out = {}
    eps = 1e-6
    for name, m in (('base', pts[:, 2] <= z_split + eps),
                    ('blade', pts[:, 2] >= z_split - eps)):
        sub = pts[m]
        if len(sub) < 4:
            raise ValueError(f'{name} 頂點不足（{len(sub)}）')
        h = ConvexHull(sub)
        used = np.unique(h.simplices)
        remap = {int(o): i for i, o in enumerate(used)}
        F = np.array([[remap[int(v)] for v in tri] for tri in h.simplices])
        # 外法向一致：scipy 的 simplices 不保證繞向，依 equations 校正
        V = sub[used]
        c = V.mean(axis=0)
        for k, tri in enumerate(F):
            n = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
            if n @ (V[tri[0]] - c) < 0:
                F[k] = tri[[0, 2, 1]]
        out[name] = (V, F)
    return out


def inner_face_y(V: np.ndarray, z: float, sign: float = 1.0) -> float:
    """凸塊在高度 z 的內面 y（finger1 取最小 y，finger2 取最大 y）。

    以凸包的半空間表示求 x=0 截線上的端點；z 不在凸塊範圍內回 nan。
    """
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    ys = np.linspace(V[:, 1].min() - 1e-3, V[:, 1].max() + 1e-3, 4001)
    P = np.stack([np.zeros_like(ys), ys, np.full_like(ys, z)], 1)
    inside = (P @ h.equations[:, :3].T + h.equations[:, 3]).max(axis=1) <= 1e-9
    if not inside.any():
        return float('nan')
    return float(ys[inside].min() if sign > 0 else ys[inside].max())


def apply_split(stage, robot_root: str, z_split: float = Z_SPLIT_M):
    """把兩指的單一凸包碰撞換成兩個凸塊。回傳報告 dict。

    原碰撞 prim **保留但停用**（collisionEnabled = False），方便核對；
    新凸塊建在手指連桿下，路徑含 'finger'，所以夾持摩擦綁定會照樣套上。
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Usd
    from isaac_common import walk
    rep = {'z_split_m': z_split, 'fingers': {}}
    cache = UsdGeom.XformCache()
    for link_name in ('uflite_finger1', 'uflite_finger2'):
        cols = [pr for pr in walk(stage, robot_root)
                if pr.HasAPI(UsdPhysics.CollisionAPI)
                and f'/{link_name}/' in str(pr.GetPath())
                and pr.IsA(UsdGeom.Mesh)]
        if len(cols) != 1:
            raise RuntimeError(f'{link_name} 的碰撞網格不是恰好一個：'
                               f'{[str(c.GetPath()) for c in cols]}')
        col = cols[0]
        # 實例化的子樹不能編輯：把它的實例根改成非實例
        p = col
        while p and p.IsValid() and p.IsInstanceProxy():
            p = p.GetParent()
        if p and p.IsInstance():
            p.SetInstanceable(False)
        col = stage.GetPrimAtPath(col.GetPath())
        link = col.GetParent()
        while link and link.GetName() != link_name:
            link = link.GetParent()
        Lw = np.array(cache.GetLocalToWorldTransform(link)).T
        Mw = np.array(cache.GetLocalToWorldTransform(col)).T
        R = np.linalg.inv(Lw) @ Mw
        pts = np.array(UsdGeom.Mesh(col).GetPointsAttr().Get(), float)
        P = (R[:3, :3] @ pts.T).T + R[:3, 3]
        UsdPhysics.CollisionAPI(col).CreateCollisionEnabledAttr().Set(False)
        parts = split_hulls(P, z_split)
        made = {}
        for name, (V, F) in parts.items():
            path = link.GetPath().AppendChild(f'grip_col_{name}')
            m = UsdGeom.Mesh.Define(stage, path)
            m.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in V])
            m.CreateFaceVertexCountsAttr([3] * len(F))
            m.CreateFaceVertexIndicesAttr([int(i) for i in F.reshape(-1)])
            m.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
            mc.CreateApproximationAttr().Set('convexHull')
            made[name] = {'path': str(path), 'n_vertices': int(len(V)),
                          'y_range_mm': [round(float(V[:, 1].min()) * 1e3, 2),
                                         round(float(V[:, 1].max()) * 1e3, 2)],
                          'z_range_mm': [round(float(V[:, 2].min()) * 1e3, 2),
                                         round(float(V[:, 2].max()) * 1e3, 2)]}
        rep['fingers'][link_name] = {'disabled': str(col.GetPath()),
                                     'parts': made}
    return rep
