#!/usr/bin/env python3
"""G1：door_wbopt 淨空列的解析梯度（MotM 開門第一包（5）時間帳：規劃 89% 在碰撞列有限差分）。
值與 door_clear_door_fast.clear_door_fast 完全同式（同計算順序；逐構型核對見 door_clear_grad_check.py）；另外記下取到最小值的那一對
（連桿點雲中的點、方盒／橫桿／另一連桿的點），在該對上取解析梯度：
  點到方盒有號距離  外部：∇＝R·(sign(L)·max(|L|−h,0))/‖·‖；內部：最近面的外法向（取 max 分量）。
  點到橫桿（膠囊）  ∇＝(P−最近軸點)/‖·‖。
  點隨連桿動        ∂P/∂x[:9]＝位置 Jacobian（K.jacobian 以該點在連桿座標的位置為 offset；手指含沿局部 y 的開度平移）。
  門角              門扇、立柱、橫桿繞世界 z（鉸鏈在原點）轉 −φ ⇒ ∂d/∂φ＝n·(ẑ×P)；其他靜態方盒 0。
  底盤圓盤          ∂/∂(x,y)＝方盒梯度的水平分量；門扇另有 φ 項；偏航 0。
  自碰撞            點對點：n＝(P_x−P_y)/‖·‖，∇＝nᵀ(J_x−J_y)。
限制（照實）：min 在最近對切換、點雲最近點跳點、方盒內部最近面切換處不可微；此處回傳的是「目前取到最小值那一段」的梯度（單側方向導數的上界，
可能偏樂觀）。這不是新的安全保證：SQP 每輪的步仍由非線性殘差、merit 與最終 verify 全非線性重核把關（與有限差分版相同）。"""
import numpy as np

import door_wbopt as WB
import door_clear_fast as FAST
import door_clear_door_fast as F

DOC, PG, K = WB.DOC, WB.PG, WB.K
TOOL = WB.TOOL
_C, _R = FAST._C, FAST._R
EZ = np.array([0.0, 0.0, 1.0])
ROT = ('leaf', 'bar')                                                # 隨門角轉的形狀（立柱名稱 post*）


def _rot(name):
    return name == 'leaf' or name == 'bar' or name.startswith('post')


def box_grad(p, c, R, hs):
    L = (p - c) @ R
    d = np.abs(L) - hs
    if (d > 0).any():
        e = np.maximum(d, 0.0) * np.sign(L)
        return R @ (e / np.linalg.norm(e))
    i = int(np.argmax(d))
    e = np.zeros(3)
    e[i] = 1.0 if L[i] >= 0 else -1.0
    return R @ e


def bar_grad(p, bar):
    c, u, hl, r = bar
    s = float(np.clip((p - c) @ u, -hl, hl))
    v = p - (c + s * u)
    return v / np.linalg.norm(v)


def _local(n, i, fq):
    off = DOC.CL[n][i].copy()
    if n in DOC.FINGERS and fq:
        off[1] += (1.0 if n.endswith('1') else -1.0) * fq
    return off


def margins_grad(x, T_WH0):
    """回 (m, G, c)：m＝四類餘裕（env, base, self, shell_bar 減門檻；同 door_wbopt.margins），G＝(4,10) 解析梯度，c＝clear_door_fast 同欄位 dict。"""
    qf, phi = np.asarray(x[:9], float), float(x[9])
    fq = WB.FINGER_Q
    sh, leaf = WB._static()
    Rp = WB.Rz4(-phi)[:3, :3]
    leaf_p = (leaf[0], Rp @ leaf[1], Rp, leaf[3])
    T_WH = WB.handle_pose(T_WH0, phi)
    posts, bar = PG.handle_shapes(T_WH)
    near = [s for s in sh if float(DOC.box_dist(np.array([[qf[0], qf[1], s[1][2]]]), s[1], s[2], s[3])[0]) < 1.6]
    pairs = FAST._pairs()
    names = sorted(set(DOC.ARM_LINKS) | {n for p in pairs for n in p})
    T = F.fk_all(qf, names)
    for n in DOC.FINGERS:
        if n in T and fq:
            T[n] = T[n].copy()
            T[n][:3, 3] += (1.0 if n.endswith('1') else -1.0) * fq * T[n][:3, 1]
    W_ = {}

    def pts(n):
        if n not in W_:
            W_[n] = (T[n][:3, :3] @ DOC.CL[n].T).T + T[n][:3, 3]
        return W_[n]
    Cw = {n: T[n][:3, :3] @ _C[n] + T[n][:3, 3] for n in DOC.ARM_LINKS}
    cand = []
    shapes = [leaf_p] + near + posts
    C = np.array([Cw[n] for n in DOC.ARM_LINKS])
    LB = F._lb_boxes(C, shapes)
    for i, n in enumerate(DOC.ARM_LINKS):
        for j, (name, c, R, hs) in enumerate(shapes):
            cand.append((float(LB[i, j]) - _R[n], n, name, (c, R, hs)))
        if n not in TOOL:
            cand.append((float(PG.bar_dist(C[i][None, :], bar)[0]) - _R[n], n, 'bar', None))
    cand.sort(key=lambda z: z[0])
    env, who, arg = np.inf, None, None
    for b, n, name, g in cand:
        if b >= env:
            break
        D = PG.bar_dist(pts(n), bar) if g is None else DOC.box_dist(pts(n), *g)
        i = int(np.argmin(D))
        d = float(D[i])
        if d < env:
            env, who, arg = d, f'{n}|{name}', (n, name, i, g)
    Ds = PG.bar_dist(pts('uflite_gripper_link'), bar)
    i_sh = int(np.argmin(Ds))
    shell = float(Ds[i_sh])
    bvals = [(float(DOC.box_dist(np.array([[qf[0], qf[1], c[2]]]), c, R, hs)[0]) - DOC.RB, name, (c, R, hs)) for name, c, R, hs in [leaf_p] + near]
    bc, bwho = min((v, nm) for v, nm, _ in bvals)
    bgeo = next(g for v, nm, g in bvals if nm == bwho and v == bc)
    # 自碰撞（同 door_clear_door_fast._self_clear，另記最近點對）
    Tn = {n: T[n] for n in names}
    Cs = {n: Tn[n][:3, :3] @ _C[n] + Tn[n][:3, 3] for n in Tn}
    lb = sorted((float(np.linalg.norm(Cs[a] - Cs[b])) - _R[a] - _R[b], a, b) for a, b in pairs)
    sc, sarg = np.inf, None
    for b_, a, b in lb:
        if b_ >= sc:
            break
        if a not in FAST._TREE:
            FAST._TREE[a] = FAST.cKDTree(DOC.CL[a])
        Py = (pts(b) - Tn[a][:3, 3]) @ Tn[a][:3, :3]
        dd, ii = FAST._TREE[a].query(Py, k=1, distance_upper_bound=sc)
        j = int(np.argmin(dd))
        if float(dd[j]) < sc:
            sc, sarg = float(dd[j]), (a, int(ii[j]), b, j)
    c_out = dict(env=env, base=bc, self=sc, shell_bar=shell, who=dict(env=who, base=bwho))
    m = np.array([c_out[k] - WB.THR[k] for k in ('env', 'base', 'self', 'shell_bar')])
    G = np.zeros((4, WB.NX))
    # env
    if arg is not None:
        n, name, i, g = arg
        p = pts(n)[i]
        nw = bar_grad(p, bar) if g is None else box_grad(p, *g)
        J = K.jacobian(qf, n, offset=_local(n, i, fq))[:3]
        G[0, :9] = nw @ J
        if _rot(name):
            G[0, 9] = float(nw @ np.cross(EZ, p))
    # base
    pb = np.array([qf[0], qf[1], bgeo[0][2]])
    nb = box_grad(pb, *bgeo)
    G[1, :2] = nb[:2]
    if _rot(bwho):
        G[1, 9] = float(nb @ np.cross(EZ, pb))
    # self
    if sarg is not None and sc > 0:
        a, ia, b, jb = sarg
        pa = Tn[a][:3, :3] @ DOC.CL[a][ia] + Tn[a][:3, 3]
        pb_ = pts(b)[jb]
        ns = (pa - pb_) / np.linalg.norm(pa - pb_)
        G[2, :9] = ns @ (K.jacobian(qf, a, offset=_local(a, ia, fq))[:3] - K.jacobian(qf, b, offset=_local(b, jb, fq))[:3])
    # shell_bar
    p = pts('uflite_gripper_link')[i_sh]
    nw = bar_grad(p, bar)
    G[3, :9] = nw @ K.jacobian(qf, 'uflite_gripper_link', offset=_local('uflite_gripper_link', i_sh, fq))[:3]
    G[3, 9] = float(nw @ np.cross(EZ, p))
    return m, G, c_out
