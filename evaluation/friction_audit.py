#!/usr/bin/env python3
"""接觸面的**實際材質與摩擦值**核對（需要已載入的 USD stage）。

為什麼不能只看綁定數
--------------------
`bind_frictionless` 回傳的是「綁了幾個」，**不是**「該綁的都綁了」。
漏掉的面不會出現在那個數字裡，而且 URDF 的 `<gazebo><mu1>` importer 不讀，
沒綁到的面會靜靜退回 PhysX 預設（約 0.5）。本檔逐面讀回**實際**材質與
摩擦係數，並對每一類面核它**應該**是什麼。

兩類面，要求相反
----------------
    低摩擦面  支撐球、地面 —— 底盤被直接設速度，摩擦會抵抗橫向運動並產生
              淨力矩（實測不綁時 5 s 偏航 21.1°）。這些要 mu = 0。
    設計接觸面  手指、把手橫桿 —— **需要摩擦才夾得住**。把它們一起歸零等於
              讓夾持失效。這些**不得**為零。

其餘面（櫃體、抽屜本體、牆、傢俱）不在機器人的驅動路徑上，保持預設即可，
但仍然列出來，讓「有沒有誰被誤綁」看得見。
"""
from __future__ import annotations


def _mat_of(stage, prim):
    """回傳 (材質路徑, 靜摩擦, 動摩擦, 綁在哪個 prim)。

    **要往上找**：綁在祖先上的 weakerThanDescendants binding 會向下傳遞，
    碰撞 prim 自己沒有直接綁定不代表它沒有材質。instance proxy 不可編輯，
    材質只能綁在可編輯的祖先上，所以直接看自己會誤判成「沒綁」。
    """
    from pxr import UsdPhysics, UsdShade
    cur, rel = prim, None
    while cur and cur.IsValid():
        r = UsdShade.MaterialBindingAPI(cur).GetDirectBindingRel('physics')
        if r and r.GetTargets():
            rel = r
            break
        cur = cur.GetParent()
    if rel is None:
        return (None, None, None, None)
    mp = stage.GetPrimAtPath(rel.GetTargets()[0])
    if not mp or not mp.IsValid():
        return (str(rel.GetTargets()[0]), None, None, str(cur.GetPath()))
    m = UsdPhysics.MaterialAPI(mp)
    sa = m.GetStaticFrictionAttr()
    da = m.GetDynamicFrictionAttr()
    return (str(mp.GetPath()),
            float(sa.Get()) if sa and sa.HasAuthoredValue() else None,
            float(da.Get()) if da and da.HasAuthoredValue() else None,
            str(cur.GetPath()))


def audit(stage, walk, *, robot_root: str, drawer_root: str,
          low_friction_name='support_', grip_names=('finger', 'handle_bar'),
          physx_default_mu=0.5, tol=1e-9):
    """逐面讀回實際材質與摩擦，回傳 (ok, 報告 dict)。"""
    from pxr import UsdPhysics

    def collisions(under):
        out = []
        for pr in walk(stage, under):
            if pr.HasAPI(UsdPhysics.CollisionAPI):
                out.append(pr)
        return out

    faces = []
    for under, owner in ((robot_root, 'robot'), (drawer_root, 'drawer'),
                         ('/World/ground', 'ground')):
        for pr in collisions(under):
            p = str(pr.GetPath())
            mat, sf, df, bound_at = _mat_of(stage, pr)
            low = (low_friction_name in p) or owner == 'ground'
            grip = any(g in p.lower() for g in grip_names)
            faces.append({'path': p, 'owner': owner, 'material': mat,
                          'static_friction': sf, 'dynamic_friction': df,
                          'bound_at': bound_at,
                          'expect': ('low' if low else
                                     ('grip' if grip else 'default')),
                          'instance_proxy': bool(pr.IsInstanceProxy())})

    bad = []
    for f in faces:
        eff_s = f['static_friction']
        eff_d = f['dynamic_friction']
        unbound = f['material'] is None
        if f['expect'] == 'low':
            if unbound:
                bad.append(f'{f["path"]}：應為低摩擦但**沒有綁任何材質** ⇒ '
                           f'退回 PhysX 預設（約 {physx_default_mu}）')
            elif eff_s is None or eff_d is None:
                bad.append(f'{f["path"]}：綁了材質但摩擦值**未寫入** '
                           f'（static={eff_s}, dynamic={eff_d}）')
            elif abs(eff_s) > tol or abs(eff_d) > tol:
                bad.append(f'{f["path"]}：應為零摩擦，實際 '
                           f'static={eff_s}, dynamic={eff_d}')
        elif f['expect'] == 'grip':
            # **夾持面不得被歸零** —— 歸零就夾不住
            if (eff_s is not None and abs(eff_s) <= tol) or \
                    (eff_d is not None and abs(eff_d) <= tol):
                bad.append(f'{f["path"]}：**夾持面被綁成零摩擦** ⇒ 夾不住 '
                           f'（static={eff_s}, dynamic={eff_d}）')
            # **沒綁材質也是問題。** 退回 PhysX 預設（約 0.5）不是「沒事」，
            # 而是「夾持靠一個沒有被指定過的值」。接觸操作的摩擦要像抽屜的
            # 質量與阻尼那樣**執行前選定並記錄**。
            elif unbound:
                bad.append(f'{f["path"]}：夾持面**沒有綁材質** ⇒ '
                           f'退回 PhysX 預設（約 {physx_default_mu}），'
                           f'那是未指定值，不得作為夾持的依據')
            elif eff_s is None or eff_d is None:
                bad.append(f'{f["path"]}：夾持面綁了材質但摩擦值**未寫入** '
                           f'（static={eff_s}, dynamic={eff_d}）')

    by = {}
    for f in faces:
        by[f['expect']] = by.get(f['expect'], 0) + 1
    rep = {
        'n_faces': len(faces),
        'by_expectation': by,
        'n_unbound': sum(1 for f in faces if f['material'] is None),
        'n_instance_proxy': sum(1 for f in faces if f['instance_proxy']),
        'violations': bad,
        'faces': faces,
        'note': ('低摩擦面要 mu=0（底盤直接設速度，摩擦會產生淨力矩）；'
                 '夾持面**不得**為零（否則夾不住）。'
                 '只核綁定數不為零會漏掉未綁的面。'),
    }
    return (not bad), rep


def summarise(rep) -> str:
    lines = [f'接觸面 {rep["n_faces"]} 個；未綁材質 {rep["n_unbound"]} 個；'
             f'instance proxy {rep["n_instance_proxy"]} 個',
             f'  分類 {rep["by_expectation"]}']
    for f in rep['faces']:
        if f['expect'] in ('low', 'grip'):
            _inh = ('' if f['bound_at'] == f['path']
                    else f'  ←綁在 {str(f["bound_at"] or "-")[-28:]}')
            lines.append(f'  [{f["expect"]:>4}] {f["path"][-50:]:<50} '
                         f'μs={f["static_friction"]} μd={f["dynamic_friction"]}'
                         f'{_inh}')
    for b in rep['violations']:
        lines.append('  **' + b + '**')
    return '\n'.join(lines)
