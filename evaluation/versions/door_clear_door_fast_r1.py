#!/usr/bin/env python3
"""door_wbopt.clear_door 的等值快速版（MotM 開門第一包（5）：開門規劃 89% 時間、控制拍停止掃掠都花在這個函式）。
判準、幾何、點雲、門扇轉角、手指張開量、豁免規則與輸出欄位完全相同；只改計算順序（同 door_clear_fast 的作法）：
  每個連桿點雲的包圍球給出各配對距離的下界（方盒符號距離、到橫桿距離都是 1-Lipschitz），依下界排序，下界不小於目前最小值的配對不必細算 ⇒ 最小值不變。
  自碰撞同 door_clear_fast（KD 樹＋包圍球下界）。底盤項與原式相同（便宜）。
等值性由 door_clear_door_fast_check.py 對原函式逐構型比對（不是假設）。"""
import numpy as np

import door_wbopt as WB
import door_clear_fast as FAST

DOC, PG, K = WB.DOC, WB.PG, WB.K
TOOL = WB.TOOL
_C, _R = FAST._C, FAST._R


def fk_all(qf, names):
    """所有指定連桿的世界變換：沿父連桿遞迴、每個關節只算一次（與 K.fk 的連乘順序相同）。"""
    memo = {}

    def go(link):
        if link in memo:
            return memo[link]
        jn = K.parent_of.get(link)
        if jn is None:
            T = np.eye(4)
        else:
            j = K.joints[jn]
            T = go(j.parent) @ j.transform(K._q_of(jn, qf))
        memo[link] = T
        return T
    return {n: go(n) for n in names}


def _lb_boxes(C, shapes):
    """(L,3) 包圍球中心對 S 個方盒的符號距離（與 box_dist 同式），一次算完。"""
    Cs = np.array([sh[1] for sh in shapes]); Rs = np.array([sh[2] for sh in shapes]); Hs = np.array([sh[3] for sh in shapes])
    Lc = np.einsum('lsj,sjk->lsk', C[:, None, :] - Cs[None], Rs)
    dd = np.abs(Lc) - Hs[None]
    return np.linalg.norm(np.maximum(dd, 0), axis=2) + np.minimum(dd.max(axis=2), 0)


def _self_clear(T, pts):
    pairs = FAST._pairs()
    Cw = {n: T[n][:3, :3] @ _C[n] + T[n][:3, 3] for n in T}
    lb = sorted((float(np.linalg.norm(Cw[x] - Cw[y])) - _R[x] - _R[y], x, y) for x, y in pairs)
    sc = np.inf
    for b, x, y in lb:
        if b >= sc:
            break
        if x not in FAST._TREE:
            FAST._TREE[x] = FAST.cKDTree(DOC.CL[x])
        Py = (pts(y) - T[x][:3, 3]) @ T[x][:3, :3]
        sc = min(sc, float(FAST._TREE[x].query(Py, k=1, distance_upper_bound=sc)[0].min()))
    return sc


def clear_door_fast(x, T_WH0):
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
    T = fk_all(qf, names)
    # 手指沿滑動軸平移（同原式：P＋(±1)·fq·T[:3,1]）——整個剛體平移，等價於改連桿原點
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
    LB = _lb_boxes(C, shapes)
    for i, n in enumerate(DOC.ARM_LINKS):
        for j, (name, c, R, hs) in enumerate(shapes):
            cand.append((float(LB[i, j]) - _R[n], n, name, (c, R, hs)))
        if n not in TOOL:
            cand.append((float(PG.bar_dist(C[i][None, :], bar)[0]) - _R[n], n, 'bar', None))
    cand.sort(key=lambda z: z[0])
    env, who = np.inf, None
    for b, n, name, g in cand:
        if b >= env:
            break
        d = float(PG.bar_dist(pts(n), bar).min()) if g is None else float(DOC.box_dist(pts(n), *g).min())
        if d < env:
            env, who = d, f'{n}|{name}'
    shell = float(PG.bar_dist(pts('uflite_gripper_link'), bar).min())
    bc, bwho = min((float(DOC.box_dist(np.array([[qf[0], qf[1], c[2]]]), c, R, hs)[0]) - DOC.RB, name) for name, c, R, hs in [leaf_p] + near)
    # 自碰撞：原式用 FAST.clear_full(qf, T_WH, finger_q=FINGER_Q)['self']——其手指同樣沿 y 平移 finger_q
    sc = _self_clear({n: T[n] for n in names}, pts)
    return dict(env=env, base=bc, self=sc, shell_bar=shell, who=dict(env=who, base=bwho))
