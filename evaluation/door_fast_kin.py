#!/usr/bin/env python3
"""G2：控制拍的等值加速元件（Codex 20261113 時間帳裁定：IK 熱啟動＋FastChain、只共用真正相同的掃掠前綴、以完整狀態為鍵的快取）。
  arm_gravity_fast  door_body_model.arm_gravity 的單趟版：所有連桿一次前向（fk_all 同 K.fk 的連乘順序），各關節軸與位置由父連桿＋關節原點得出，
                    G_i＝Σ_{b 在關節 i 下游} (a_i×(p_b−p_i))·(m_b·g)——與 K.jacobian(offset＝質心)ᵀ·m·g 同式。逐構型核對見 door_fast_kin_check.py。
  ik_closure_fc     開門執行層 _ik_closure 的 FastChain 版：同一牛頓迭代（同種子、同阻尼、同停止條件、同 30 次上限），每次迭代 FK＋Jacobian 一趟算完。
  ClearCache        clear_door 結果快取：鍵＝構型 10 維的位元組＋FINGER_Q＋PHI0＋T_WH0 位元組（淨空函式的全部輸入）；只有輸入完全相同才命中
                    ⇒ 結果與重算相同（不是近似）。每控制拍清空。"""
import math

import numpy as np

import door_wbopt as W
import door_body_model as BM
import door_clear_door_fast as F

K = W.K
FC = W.FC


class FastGravity:
    def __init__(self, K_, bodies):
        self.K = K_
        self.bodies = list(bodies)
        self.arm_j = list(K_.dof_names[3:9])
        self.inc = np.array([[jn in K_.chain(b) for jn in self.arm_j] for b in self.bodies], bool)
        self.mg = [BM.INERT[b][0] * BM.GRAV for b in self.bodies]
        self.com = [np.asarray(BM.INERT[b][1], float) for b in self.bodies]
        self.jpar = [K_.joints[jn].parent for jn in self.arm_j]
        self.names = sorted(set(self.bodies) | set(self.jpar))

    def __call__(self, q9):
        T = F.fk_all(np.asarray(q9, float), self.names)
        G = np.zeros(6)
        A, P = [], []
        for jn, par in zip(self.arm_j, self.jpar):
            j = self.K.joints[jn]
            Tp = T[par]
            Rj = Tp[:3, :3] @ j.origin_rot
            A.append(Rj @ j.axis)
            P.append(Tp[:3, :3] @ j.origin_xyz + Tp[:3, 3])
        for bi, b in enumerate(self.bodies):
            Tb = T[b]
            pb = Tb[:3, :3] @ self.com[bi] + Tb[:3, 3]
            for i in range(6):
                if self.inc[bi, i]:
                    G[i] += float(np.cross(A[i], pb - P[i]) @ self.mg[bi])
        return G


def ik_closure_fc(Tg, b, seed):
    """回 (q, 殘差, σ_min)；與 door_open_exec_ctrl._ik_closure 同一迭代。"""
    q = np.asarray(seed, float).copy()
    res, J = np.inf, None
    for _ in range(30):
        q9 = np.r_[b, q]
        Tc, Jf = FC.fk_jac(q9)
        J = Jf[:, 3:9]
        dR = Tg[:3, :3] @ Tc[:3, :3].T
        e = np.r_[Tg[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        res = float(np.linalg.norm(e))
        if res < 1e-10:
            break
        q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
        J = None
    if J is None:
        J = FC.fk_jac(np.r_[b, q])[1][:, 3:9]
    return q, res, float(np.linalg.svd(J, compute_uv=False).min())


class ClearCache:
    def __init__(self, fn):
        self.fn, self.d, self.hit, self.miss = fn, {}, 0, 0

    def clear(self):
        self.d = {}

    def __call__(self, x, T_WH0):
        x = np.asarray(x, float)
        key = (x.tobytes(), float(W.FINGER_Q), float(W.PHI0), np.asarray(T_WH0, float).tobytes())
        r = self.d.get(key)
        if r is None:
            self.miss += 1
            r = self.fn(x, T_WH0)
            self.d[key] = r
        else:
            self.hit += 1
        return dict(r, who=dict(r['who']))


class FastGravity2:
    """單趟串鏈版（G3 階段）：沿底盤→joint1…joint6 一次前向，記下每個手臂關節的軸、位置與關節後座標；各剛體掛在其鏈上最後一個手臂關節之後（固定偏移預先算好），
    手臂是串鏈 ⇒ 關節 i 的下游剛體＝i 之後全部 ⇒ G_i＝a_i·((S_i−M_i·p_i)×g)，S_i、M_i 為下游 Σm·p、Σm（後綴和）。
    與 arm_gravity 同一物理量，只有加總順序不同（捨入級差異；door_fast_kin_check 另核）。"""

    def __init__(self, K_, bodies):
        self.K = K_
        self.bodies = list(bodies)
        dof = list(K_.dof_names)
        self.arm_j = dof[3:9]
        last = 'link6' if 'link6' in K_.parent_of else K_.joints[self.arm_j[-1]].child
        self.chain = list(K_.chain(last))
        assert [jn for jn in self.chain if jn in dof][3:] == self.arm_j
        q0 = np.zeros(len(dof))
        self.att, self.cl, self.m = [], [], []
        for b in self.bodies:
            ch = K_.chain(b)
            jd = [jn for jn in ch if jn in self.arm_j][-1]
            Tt = np.eye(4)
            for jn in ch[ch.index(jd) + 1:]:
                Tt = Tt @ K_.joints[jn].transform(K_._q_of(jn, q0))
            m, com, _ = BM.INERT[b]
            self.att.append(self.arm_j.index(jd))
            self.cl.append(Tt[:3, :3] @ np.asarray(com, float) + Tt[:3, 3])
            self.m.append(float(m))
        self.att = np.array(self.att)
        self.cl = np.array(self.cl)
        self.m = np.array(self.m)
        self.Mk = np.array([self.m[self.att == k].sum() for k in range(6)])
        self.Msuf = np.cumsum(self.Mk[::-1])[::-1]

    def __call__(self, q9):
        q9 = np.asarray(q9, float)
        T = np.eye(4)
        A, P, Tk = [], [], []
        for jn in self.chain:
            j = self.K.joints[jn]
            Tj = T @ _iso(j)
            if jn in self.arm_j:
                A.append(Tj[:3, :3] @ j.axis)
                P.append(Tj[:3, 3].copy())
            T = T @ j.transform(self.K._q_of(jn, q9))
            if jn in self.arm_j:
                Tk.append(T.copy())
        Sk = np.zeros((6, 3))
        for bi in range(len(self.m)):
            k = self.att[bi]
            Sk[k] += self.m[bi] * (Tk[k][:3, :3] @ self.cl[bi] + Tk[k][:3, 3])
        Ssuf = np.cumsum(Sk[::-1], axis=0)[::-1]
        G = np.zeros(6)
        for i in range(6):
            G[i] = float(A[i] @ np.cross(Ssuf[i] - self.Msuf[i] * P[i], BM.GRAV))
        return G


_ISO = {}


def _iso(j):
    key = id(j)
    T = _ISO.get(key)
    if T is None:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = j.origin_rot, j.origin_xyz
        _ISO[key] = T
    return T
