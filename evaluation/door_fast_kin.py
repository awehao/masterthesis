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


def _ik_closure_fc_impl(Tg, b, seed, ac):
    q = np.asarray(seed, float).copy()
    res, J = np.inf, None
    T0 = None if ac is None else ac.prefix(b)
    for _ in range(30):
        if ac is None:
            Tc, Jf = FC.fk_jac(np.r_[b, q])
            J = Jf[:, 3:9]
        else:
            Tc, J = ac.fk_jac_arm(T0, q)
        dR = Tg[:3, :3] @ Tc[:3, :3].T
        e = np.r_[Tg[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        res = float(np.linalg.norm(e))
        if res < 1e-10:
            break
        q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
        J = None
    if J is None:
        J = FC.fk_jac(np.r_[b, q])[1][:, 3:9] if ac is None else ac.fk_jac_arm(T0, q)[1]
    return q, res, float(np.linalg.svd(J, compute_uv=False).min())


def ik_closure_fc(Tg, b, seed):
    """回 (q, 殘差, σ_min)；與 door_open_exec_ctrl._ik_closure 同一迭代。A3：正向運動學改手臂專用等值版（_ArmChain；逐位元相同）。"""
    global _AC
    if _AC is None:
        _AC = _ArmChain(FC)
    return _ik_closure_fc_impl(Tg, b, seed, _AC)


class _ArmChain:
    """FastChain.fk_jac 的手臂專用等值版（A3）：底盤三個自由度在一次 IK 內不變 ⇒ 底盤段的前綴變換只算一次；
    各關節的 K、K·K 預先算好。運算順序與 FastChain.fk_jac 相同（T＝(T·O)·B、B＝I＋(sin·K＋(1−cos)·K·K)），結果逐位元相同（door_fast_kin_check3 核）。"""

    def __init__(self, fc):
        self.fc = fc
        self.base = [st for st in fc.steps if st[4] < 3]
        self.arm = []
        for O, k, ax, jt, i in fc.steps:
            if i >= 3:
                assert jt != 'prismatic'
                K_ = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
                self.arm.append((O, ax, K_, K_ @ K_, i))
        assert [st[4] for st in fc.steps] == list(range(9))
        self.tail = fc.tail

    def prefix(self, b):
        T = np.eye(4)
        for O, k, ax, jt, i in self.base:
            Tj = T @ O
            B = np.eye(4)
            if jt == 'prismatic':
                B[:3, 3] = ax * b[i]
            else:
                sn, cs = math.sin(b[i]), math.cos(b[i])
                K_ = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
                B[:3, :3] += sn * K_ + (1.0 - cs) * (K_ @ K_)
            T = Tj @ B
        return T

    def fk_jac_arm(self, T0, qa):
        """回 (T_tcp, J 的手臂六欄)。"""
        T = T0
        A, P = [], []
        for n, (O, ax, K_, KK, i) in enumerate(self.arm):
            Tj = T @ O
            B = np.eye(4)
            sn, cs = math.sin(qa[n]), math.cos(qa[n])
            B[:3, :3] += sn * K_ + (1.0 - cs) * KK
            A.append(Tj[:3, :3] @ ax)
            P.append(Tj[:3, 3])
            T = Tj @ B
        T = T @ self.tail
        pe = T[:3, 3]
        J = np.zeros((6, 6))
        for n in range(6):
            a, r = A[n], pe - P[n]
            J[0, n], J[1, n], J[2, n] = a[1] * r[2] - a[2] * r[1], a[2] * r[0] - a[0] * r[2], a[0] * r[1] - a[1] * r[0]
            J[3:, n] = a
        return T, J


_AC = None


def ik_closure_fc_ref(Tg, b, seed):
    """原版（對照用）。"""
    return _ik_closure_fc_impl(Tg, b, seed, None)

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



class FastGravity3(FastGravity2):
    """FastGravity2 的等值加速（Codex 20261115：重力批次化）：同一串鏈前向與同一公式，
    只把「逐剛體加總 Σm·(R_k·c_b＋p_k)」先合併成每個關節一個向量 R_k·(Σm·c_b)＋M_k·p_k，並把六次 np.cross 合成一次。
    與 FastGravity2 只有加總順序不同（捨入級；door_fast_kin_check3 核最大差）。"""

    def __init__(self, K_, bodies):
        super().__init__(K_, bodies)
        self.ck = np.array([(self.m[self.att == k][:, None] * self.cl[self.att == k]).sum(axis=0) if np.any(self.att == k) else np.zeros(3) for k in range(6)])
        self._arm = [jn in self.arm_j for jn in self.chain]
        self._J = [self.K.joints[jn] for jn in self.chain]
        self._iso = [_iso(j) for j in self._J]
        self._ax = [np.asarray(j.axis, float) for j in self._J]

    def __call__(self, q9):
        q9 = np.asarray(q9, float)
        T = np.eye(4)
        A, P, Sk = np.empty((6, 3)), np.empty((6, 3)), np.empty((6, 3))
        k = 0
        K_ = self.K
        for n, jn in enumerate(self.chain):
            j = self._J[n]
            if self._arm[n]:
                Tj = T @ self._iso[n]
                A[k] = Tj[:3, :3] @ self._ax[n]
                P[k] = Tj[:3, 3]
            T = T @ j.transform(K_._q_of(jn, q9))
            if self._arm[n]:
                Sk[k] = T[:3, :3] @ self.ck[k] + self.Mk[k] * T[:3, 3]
                k += 1
        Ssuf = np.cumsum(Sk[::-1], axis=0)[::-1]
        return np.einsum('ij,ij->i', A, np.cross(Ssuf - self.Msuf[:, None] * P, BM.GRAV))


_ISO = {}


def _iso(j):
    key = id(j)
    T = _ISO.get(key)
    if T is None:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = j.origin_rot, j.origin_xyz
        _ISO[key] = T
    return T
