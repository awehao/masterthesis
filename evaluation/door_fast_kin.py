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
