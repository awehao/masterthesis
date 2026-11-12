#!/usr/bin/env python3
"""全碰撞對淨空的等值快速版（判準、幾何、點雲、門檻與 m8_door_reaim_online_r3.cfg_clear_full(contact=False) 完全相同；只改計算順序）。
作法：每個連桿點雲的包圍球給出各配對距離的下界（方盒／橫桿的符號距離與點對點距離都是 1-Lipschitz），下界不小於目前最小值的配對不必細算 ⇒ 最小值不變。
自碰撞：把 y 連桿點雲轉到 x 連桿座標，查 x 連桿預先建好的 KD 樹（剛體變換不改距離）。
等值性由 door_clear_fast_check.py 對原函式逐構型比對（不是假設）。
張開手指（finger_q>0）：原模型手指固定在閉合 0；本版把兩指點雲沿滑動軸各移 finger_q。此幾何由 Isaac 起跑時讀回兩指連桿世界位姿核對（執行器，不符拒跑）。"""
import numpy as np
from scipy.spatial import cKDTree

import m8_door_reaim_online_r4 as R4

R3, R1, PG, DOC, K = R4.R3, R4.R1, R4.PG, R4.DOC, R4.K
_C = {n: P.mean(axis=0) for n, P in DOC.CL.items()}
_R = {n: float(np.linalg.norm(P - _C[n], axis=1).max()) for n, P in DOC.CL.items()}
_TREE, _STATIC, _SARR = {}, None, None
TOOL = ('uflite_gripper_link',) + tuple(DOC.FINGERS)


def _pairs():
    if DOC.SELF_PAIRS is None:
        DOC.self_clear(np.zeros(9))                                   # 用原函式建立配對表（同一份）
    return DOC.SELF_PAIRS


def clear_full(qf, T_WH, finger_q=0.0, exempt_fingers_bar=False):
    """finger_q＝兩指張開量（m；finger_joint1 沿夾爪 +y、finger_joint2 沿 −y，URDF 互為 mimic）。0＝原模型（閉合），與原函式等值。"""
    global _STATIC, _SARR
    qf = np.asarray(qf, float)
    pairs = _pairs()
    if _STATIC is None:
        _STATIC = PG.static_shapes()
        _SARR = (np.array([c for _, c, _, _ in _STATIC]), np.array([R for _, _, R, _ in _STATIC]), np.array([h for _, _, _, h in _STATIC]))
    names = sorted(set(DOC.ARM_LINKS) | {n for p in pairs for n in p})
    T = {n: K.fk(qf, n) for n in names}
    if finger_q:
        for n, sg in (('uflite_finger1', 1.0), ('uflite_finger2', -1.0)):
            if n in T:
                T[n] = T[n].copy()
                T[n][:3, 3] += sg * float(finger_q) * T[n][:3, 1]      # 手指座標與夾爪殼同向；滑動軸＝局部 y
    Cw = {n: T[n][:3, :3] @ _C[n] + T[n][:3, 3] for n in names}
    W = {}

    def pts(n):
        if n not in W:
            W[n] = (T[n][:3, :3] @ DOC.CL[n].T).T + T[n][:3, 3]
        return W[n]
    # 自碰撞
    lb = sorted((float(np.linalg.norm(Cw[x] - Cw[y])) - _R[x] - _R[y], x, y) for x, y in pairs)
    sc = np.inf
    for b, x, y in lb:
        if b >= sc:
            break
        if x not in _TREE:
            _TREE[x] = cKDTree(DOC.CL[x])
        Py = (pts(y) - T[x][:3, 3]) @ T[x][:3, :3]
        sc = min(sc, float(_TREE[x].query(Py, k=1, distance_upper_bound=sc)[0].min()))   # 超過目前最小值者回 inf，不影響最小值
    # 環境（靜態方盒＋立柱；橫桿只對非手指、非夾爪殼連桿）
    posts, bar = PG.handle_shapes(T_WH)
    cand = []
    Cs, Rs, Hs = _SARR
    P_all = np.array([Cw[n] for n in DOC.ARM_LINKS])                  # (L,3)
    Lc = np.einsum('lsj,sjk->lsk', P_all[:, None, :] - Cs[None], Rs)  # 與 box_dist 相同：(P−c)@R
    dd = np.abs(Lc) - Hs[None]
    lbs = np.linalg.norm(np.maximum(dd, 0), axis=2) + np.minimum(dd.max(axis=2), 0)
    for i, n in enumerate(DOC.ARM_LINKS):
        for j, (name, c, R, hs) in enumerate(_STATIC):
            cand.append((float(lbs[i, j]) - _R[n], n, name, (c, R, hs)))
        c1 = Cw[n][None, :]
        for name, c, R, hs in posts:
            cand.append((float(DOC.box_dist(c1, c, R, hs)[0]) - _R[n], n, name, (c, R, hs)))
        if n not in DOC.FINGERS and n != 'uflite_gripper_link':
            cand.append((float(PG.bar_dist(c1, bar)[0]) - _R[n], n, 'bar', None))
    cand.sort(key=lambda z: z[0])
    env, who = np.inf, None
    for b, n, name, g in cand:
        if b >= env:
            break
        d = float(PG.bar_dist(pts(n), bar).min()) if g is None else float(DOC.box_dist(pts(n), *g).min())
        if d < env:
            env, who = d, f'{n}|{name}'
    bc, bwho = min((float(DOC.box_dist(np.array([[qf[0], qf[1], c[2]]]), c, R, hs)[0]) - DOC.RB, name) for name, c, R, hs in _STATIC)
    # exempt_fingers_bar（預設 False＝不變）：插入／抓持段，兩指對橫桿是指定接觸，只核夾爪殼對橫桿（同 R3.cfg_clear_full(contact=True) 的規則）
    bt, btw = min((float(PG.bar_dist(pts(n), bar).min()), n) for n in (TOOL[:1] if exempt_fingers_bar else TOOL) if n in T)
    return dict(env=env, env_who=who, base=bc, base_who=bwho, self=sc, bar_tool=bt, bar_tool_who=btw,
                ok=bool(env >= R1.ENV_M and bc >= R1.BASE_M and sc >= R1.SELF_M and bt >= R1.ENV_M))
