#!/usr/bin/env python3
"""門機構約束下的全身軌跡最佳化——離線原型（Howard 2026-11-12；reviews/20261112_howard_door_constraint_optimization.md）。
**不沿用舊開門控制**：不給底盤路線、不給手臂關節序列、不給門角時間曲線。只給起始狀態與「門開到 φ_goal」。

問題（離散、等間隔 dt、節點 k＝0…N）
  未知數  x_k＝[底盤 (x,y,θ)、六軸 q、門角 φ]  共 10 維；k＝0 固定為起始狀態（已抓持、靜止）。
  門機構  鉸鏈在世界原點、軸 +z；門角 φ≥0 時門扇繞 +z 轉 −φ（向臥室開）：T_WH(φ)＝Rz(−φ)·T_WH(0)。
  抓持閉鏈（等式）  末端位姿 T_WE(底盤,q)＝T_WH(φ)·T_HE，T_HE＝起始抓持時的把手→末端變換。
      接觸模式 'rigid'：六維全部相等（剛性不滑移**假設**）。
      接觸模式 'bar_revolute'：位置三維＋垂直桿軸的兩個轉動相等，繞桿軸的轉動放開（夾爪可繞圓桿轉；仍假設不沿桿滑）。
      兩者都是運動學假設，不是夾得住／拉得動的證明。
  不等式  關節限位；手臂關節速率；底盤四輪輪速（輪級矩陣，本體速度）；底盤／手臂加速度（二階差分）；門角範圍與門角速率上限；
          全身淨空：手臂各連桿對「轉到 φ 的門扇」、立柱、門框與牆、家具；夾爪殼對橫桿（兩指為指定接觸，豁免）；底盤圓盤對門扇與環境；自碰撞。
  目標    Σ (φ_k−φ_goal)²（越早開到越好，等於讓最佳化自己決定開門進度）＋速度與加速度的平方和（流暢）＋關節遠離限位
          ＋**手臂可操作度**（−log 最小奇異值；Howard 11/12：底盤要跟手臂協同，最佳解不該走到奇異點）。
  另加    手臂 Jacobian 最小奇異值 σ_min ≥ SIG_FLOOR（線性化不等式）：手臂快到奇異時，只能靠底盤移動來維持；
          設定點模型速率：|q̇＋τ·q̈| ≤ 手臂命令上限（一階設定點模型反算的所需命令，線性列）；終點 φ_N ≥ φ_goal（硬性）。
解法    序列二次規劃：每輪在目前軌跡線性化閉鏈（解析 Jacobian：gmpc_wb.FastChain.fk_jac；門角方向為繞鉸鏈的旋量）、輪級列與淨空（有限差分，只對接近的節點），
        解一個帶信賴域的 QP（OSQP）；非線性殘差變差就縮信賴域。收斂後每個節點再以牛頓步把閉鏈殘差壓到數值零（只動手臂），最後**用非線性模型重核**所有條件，
        含節點之間的中點。局部解：不稱全域最佳。QP 解出不代表原問題成立，以重核為準；任一項不過 ⇒ 回報不可行，不退回任何既有開門答案。
未含（照實）：設定點一階落後與 8 步在途命令（只在事後算所需設定點速率）；門的慣性、阻尼、回關；夾持力與摩擦；感測誤差。"""
import math
import os
import sys
import time

import numpy as np
import osqp
from scipy import sparse

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_approach_jnom_ctrl as DA                               # noqa: E402

GW, K, PG, DOC, R1, FAST = DA.GW, DA.K, DA.PG, DA.DOC, DA.R1, DA.FAST
FC = GW.FastChain.of(K, 'link_tcp')
LO, HI = np.asarray(DA.LITE6_SAFE.lower, float) + 0.05, np.asarray(DA.LITE6_SAFE.upper, float) - 0.05
NX = 10
V_ARM, A_ARM = 1.0, 19.984
W_LIM = 0.05 * 5.55                                                # r·ω_max（m/s）
L_W = 0.245
WM = np.array([[0.0, 1.0, L_W], [-1.0, 0.0, L_W], [0.0, -1.0, L_W], [1.0, 0.0, L_W]])
A_BASE = np.array([1.0, 1.0, 2.0])                                 # 世界座標二階差分上限（取本體 1.5／1.0 的較小者＝保守）
A_BODY = np.array([1.5, 1.0, 2.0])                                 # 底盤本體加速度上限（bottle_s0b_ctrl.A_BASE；執行端同值）
PHI_MAX, PHI_RATE = math.radians(105.0), 0.5                       # 門角上限（資產限位 110° 留 5°）、門角速率上限 rad/s（工程設定）
THR = dict(env=R1.ENV_M, base=R1.BASE_M, self=R1.SELF_M, shell_bar=R1.ENV_M)
FINGER_Q = DA.FINGER_Q                                             # 淨空模型的手指開度；綁定後由 set_model() 換成各指實測閉合開度
PHI0 = 0.0                                                         # T_WH0 對應的門角（綁定瞬間）


def set_model(phi0=0.0, finger_q=None):
    """設定把手基準門角與淨空模型的手指開度（模組層級；一次規劃／一趟執行用同一組）。"""
    global PHI0, FINGER_Q
    PHI0 = float(phi0)
    if finger_q is not None:
        FINGER_Q = float(finger_q)
TOOL = ('uflite_gripper_link',) + tuple(DOC.FINGERS)
SIG_FLOOR = 0.03                                                   # 手臂 6×6 Jacobian 最小奇異值下限（非接觸趟的近奇異谷底為 0.009；抓持起始約 0.06–0.09）
TAU_SP = 0.105                                                     # 設定點一階模型時間常數（辨識檔 α＝0.095／步）


def sigma_min(x):
    return float(np.linalg.svd(FC.fk_jac(np.asarray(x[:9], float))[1][:, 3:], compute_uv=False)[-1])


def sigma_grad(x, h=1e-5):
    s0 = sigma_min(x)
    g = np.zeros(6)
    for j in range(6):
        xp = np.array(x, float)
        xp[3 + j] += h
        g[j] = (sigma_min(xp) - s0) / h
    return s0, g


def Rz4(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def handle_pose(T_WH0, phi):
    """T_WH0＝門角 PHI0 時的把手位姿（r3：PHI0 可非零＝綁定瞬間；預設 0 與 r2 相同）。T_WH(φ)＝Rz[−(φ−PHI0)]·T_WH0。"""
    return Rz4(-(phi - PHI0)) @ T_WH0


_STAT = None


def _static():
    global _STAT
    if _STAT is None:
        sh = [s for s in PG.static_shapes() if s[0] != 'leaf']
        leaf = [s for s in PG.static_shapes() if s[0] == 'leaf'][0]
        _STAT = (sh, leaf)
    return _STAT


def clear_door(x, T_WH0):
    """非線性淨空：回 dict(env, base, self, shell_bar, who)。門扇、立柱、橫桿都轉到 φ。"""
    qf, phi = np.asarray(x[:9], float), float(x[9])
    sh, leaf = _static()
    Rp = Rz4(-phi)[:3, :3]                                         # 門扇幾何以絕對門角轉（資產在 φ＝0 定義）
    leaf_p = (leaf[0], Rp @ leaf[1], Rp, leaf[3])
    T_WH = handle_pose(T_WH0, phi)
    posts, bar = PG.handle_shapes(T_WH)
    near = [s for s in sh if float(DOC.box_dist(np.array([[qf[0], qf[1], s[1][2]]]), s[1], s[2], s[3])[0]) < 1.6]
    env, who, shell = np.inf, None, np.inf
    for n in DOC.ARM_LINKS:
        T = K.fk(qf, n)
        P = (T[:3, :3] @ DOC.CL[n].T).T + T[:3, 3]
        if n in DOC.FINGERS:
            P = P + (1.0 if n.endswith('1') else -1.0) * FINGER_Q * T[:3, 1]
        for name, c, R, hs in [leaf_p] + near + posts:
            d = float(DOC.box_dist(P, c, R, hs).min())
            if d < env:
                env, who = d, f'{n}|{name}'
        if n not in TOOL:
            d = float(PG.bar_dist(P, bar).min())
            if d < env:
                env, who = d, f'{n}|bar'
        elif n == 'uflite_gripper_link':
            shell = float(PG.bar_dist(P, bar).min())
    bc, bwho = min((float(DOC.box_dist(np.array([[qf[0], qf[1], c[2]]]), c, R, hs)[0]) - DOC.RB, name) for name, c, R, hs in [leaf_p] + near)
    sc = FAST.clear_full(qf, T_WH, finger_q=FINGER_Q)['self']
    return dict(env=env, base=bc, self=sc, shell_bar=shell, who=dict(env=who, base=bwho))


def margins(x, T_WH0):
    c = clear_door(x, T_WH0)
    return np.array([c[k] - THR[k] for k in ('env', 'base', 'self', 'shell_bar')]), c


class DoorWBOpt:
    def __init__(self, T_WH0, T_WE0, x0, phi_goal, N=60, dt=0.2, mode='rigid', base_fixed=False, w=None, T_HE=None, project_start=True):
        """r3：T_HE 可明確傳入（綁定瞬間的實測抓持變換；之後規劃與重規劃都固定用它）；project_start=False 時起點不做牛頓投影，
        只核閉鏈殘差 <1e-6（否則丟 ValueError）——實測起點不得被改成另一個構型。T_WH0 是門角 PHI0（set_model）時的把手位姿。"""
        self.T_WH0 = np.asarray(T_WH0, float)
        self.T_HE = np.linalg.inv(self.T_WH0) @ np.asarray(T_WE0, float) if T_HE is None else np.asarray(T_HE, float)
        self.mode = mode
        x0 = np.asarray(x0, float)
        self.x0, self.phi_goal, self.N, self.dt, self.mode, self.base_fixed = np.asarray(x0, float), float(phi_goal), int(N), float(dt), mode, bool(base_fixed)
        self.w = dict(phi=4.0, v=1.0, a=1.0, lim=0.3, manip=0.5, slack=1e5, reg=1e-3, col=1e5, term=1e5)
        self.w.update(w or {})
        self.scale = np.r_[0.25, 0.25, 0.6, np.full(6, V_ARM), PHI_RATE]   # 速度正規化（各自的上限量級）
        if project_start:
            self.x0 = self.project(self.x0, iters=30)             # 起始抓持構型的閉鏈殘差壓到數值零（IK 容差 1e-5 m 級）
        else:
            e0 = float(np.abs(self.closure(self.x0)[0]).max())
            if e0 > 1e-6:
                raise ValueError(f'start closure residual {e0:.3e} > 1e-6（實測起點與 T_HE 不一致；不投影）')
        self.X = np.tile(self.x0, (self.N + 1, 1))
        self.act, self.keep = 0.025, 0.004                         # 淨空列：餘裕（距離−門檻）低於 act 才線性化；要求線性化後餘裕 ≥ keep
        self.log = []
        self.init = dict(kind='hold')

    def init_corotate(self, T_ramp):
        """起始猜測（不是答案）：整台機器人跟著門繞鉸鏈剛體轉動——底盤位姿繞原點轉 −φ、手臂不動，閉鏈自動成立。
        φ 以 min-jerk 在 T_ramp 內由 0 到 φ_goal 之後保持。來源只有門機構（鉸鏈位置與轉向），不含任何既有開門軌跡。之後由最佳化自由改動。"""
        ph0 = float(self.x0[9])
        for k in range(self.N + 1):
            u = min(k * self.dt / T_ramp, 1.0)
            dphi = (self.phi_goal - ph0) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)   # 角度**增量**（起點可為非零門角）
            c, s_ = math.cos(-dphi), math.sin(-dphi)
            b = self.x0[:3]
            self.X[k] = np.r_[c * b[0] - s_ * b[1], s_ * b[0] + c * b[1], b[2] - dphi, self.x0[3:9], ph0 + dphi]
        self.X[0] = self.x0
        self.init = dict(kind='corotate', T_ramp=T_ramp)

    # ---- 閉鏈
    def target(self, phi):
        return handle_pose(self.T_WH0, phi) @ self.T_HE

    def rows(self, phi):
        """接觸模式的列選擇矩陣（m×6）。"""
        if self.mode == 'rigid':
            return np.eye(6)
        u = (Rz4(-phi)[:3, :3] @ self.T_WH0[:3, 0])
        u /= np.linalg.norm(u)
        a = np.cross(u, [0, 0, 1.0])
        a /= np.linalg.norm(a)
        B = np.vstack([a, np.cross(u, a)])
        S = np.zeros((5, 6))
        S[:3, :3] = np.eye(3)
        S[3:, 3:] = B
        return S

    def closure(self, x):
        """回 (S·e, S·J_x(6×9), S·J_φ(6,))：e＝目標−目前（世界座標位置差、轉動差）。"""
        Tc, J = FC.fk_jac(x[:9])
        Td = self.target(x[9])
        e = np.r_[Td[:3, 3] - Tc[:3, 3], GW._rot_err_world(Tc[:3, :3], Td[:3, :3])]
        om = np.array([0.0, 0.0, -1.0])
        Jphi = np.r_[np.cross(om, Td[:3, 3]), om]                   # 與 PHI0 無關：dT_WH/dφ 都是繞原點鉸鏈 −z 的旋量
        S = self.rows(x[9])
        return S @ e, S @ J, S @ Jphi

    def residual(self, X):
        r = np.array([self.closure(x)[0] for x in X])
        return float(np.abs(r[:, :3]).max()), float(np.abs(r[:, 3:]).max())

    def project(self, x, iters=12):
        """節點牛頓投影：固定底盤與門角，只動手臂把閉鏈殘差壓到數值零（阻尼最小平方；每步限幅、夾在關節限位內）。"""
        x = x.copy()
        for _ in range(iters):
            e, J, _ = self.closure(x)
            if float(np.abs(e).max()) < 1e-10:
                break
            Ja = J[:, 3:]
            dq = np.linalg.solve(Ja.T @ Ja + 1e-8 * np.eye(6), Ja.T @ e)
            dq *= min(1.0, 0.2 / max(float(np.abs(dq).max()), 1e-12))
            x[3:9] = np.clip(x[3:9] + dq, LO, HI)
        return x

    def merit(self, X):
        """非線性目標＋限制違反的重罰（步長接受用；閉鏈由節點投影維持）。"""
        w, dt, N = self.w, self.dt, self.N
        sc = 1.0 / (self.scale * dt)
        d1 = np.diff(X, axis=0) * sc
        Xe = np.vstack([X[0], X, X[-1]])
        d2 = (Xe[2:] - 2 * Xe[1:-1] + Xe[:-2]) * sc                 # k＝0…N，兩端外視為靜止；與 QP 同式
        half, mid = 0.5 * (HI - LO), 0.5 * (HI + LO)
        sig = np.array([sigma_min(x) for x in X[1:]])
        J = (w['phi'] * float(((X[1:, 9] - self.phi_goal) ** 2).sum()) + w['v'] * float((d1 ** 2).sum()) + w['a'] * float((d2 ** 2).sum())
             + w['lim'] * float((((X[1:, 3:9] - mid) / half) ** 2).sum()) - w['manip'] * float(np.log(np.maximum(sig, 1e-4)).sum()))
        viol = max(0.0, self.phi_goal - X[-1, 9]) + float(np.maximum(min(SIG_FLOOR, sigma_min(X[0])) - sig, 0).sum())
        dX = np.diff(X, axis=0) / dt
        for k in range(N):
            th = 0.5 * (X[k, 2] + X[k + 1, 2])
            c_, s_ = math.cos(th), math.sin(th)
            vb = np.array([c_ * dX[k, 0] + s_ * dX[k, 1], -s_ * dX[k, 0] + c_ * dX[k, 1], dX[k, 2]])
            viol += float(np.maximum(np.abs(WM @ vb) - W_LIM, 0).sum())
        qdd = np.diff(np.vstack([np.zeros(6), dX[:, 3:9]]), axis=0) / dt
        viol += float(np.maximum(np.abs(dX[:, 3:9] + TAU_SP * qdd) - V_ARM, 0).sum())
        mg = np.array([margins(x, self.T_WH0)[0][:3 if self.mode == 'rigid' else 4].min() for x in X[1:]])
        viol += 10.0 * float(np.maximum(-mg, 0).sum())
        return J + 200.0 * viol, J, viol

    # ---- 一輪 QP
    def qp(self, rho, col):
        N, dt, w = self.N, self.dt, self.w
        n = N * NX
        m_c = 6 if self.mode == 'rigid' else 5
        ns = N * m_c
        nc = len(col)
        nz = n + ns + nc + 1                                        # 最後一個＝終點門角鬆弛 s_T
        X = self.X
        idx = lambda k: slice((k - 1) * NX, k * NX)                # noqa: E731  節點 k（1…N）
        P = sparse.lil_matrix((nz, nz))
        qv = np.zeros(nz)
        sc = 1.0 / (self.scale * dt)
        # 目標以「絕對量」寫成 δ 的二次式：x_k＝X̄_k＋δ_k
        for k in range(1, N + 1):
            s = idx(k)
            i_phi = s.start + 9
            P[i_phi, i_phi] += 2 * w['phi']
            qv[i_phi] += 2 * w['phi'] * (X[k, 9] - self.phi_goal)
            for j in range(6):                                     # 關節遠離限位（以中心為零點的正規化二次）
                i = s.start + 3 + j
                half = 0.5 * (HI[j] - LO[j])
                P[i, i] += 2 * w['lim'] / half ** 2
                qv[i] += 2 * w['lim'] * (X[k, 3 + j] - 0.5 * (HI[j] + LO[j])) / half ** 2
            for i in range(s.start, s.stop):
                P[i, i] += 2 * w['reg']
            sg, gg = self.sig[k - 1]
            for j in range(6):                                     # −w·log σ_min 的一階項（二階以 reg 與信賴域處理）
                qv[s.start + 3 + j] += -w['manip'] * gg[j] / max(sg, 1e-4)
        def quad_diff(coefs, wt, scl, ks):                             # Σ |Σ_c c·x_{k+o}|² 的 P、q（含固定的 x_0）
            for k in ks:
                terms = [(min(max(k + o, 0), N), c) for o, c in coefs]   # 兩端外視為靜止：x_{-1}＝x_0、x_{N+1}＝x_N
                const = sum(c * X[t] for t, c in terms)
                for d_ in range(NX):
                    g = wt * scl[d_] ** 2
                    for (ta, ca) in terms:
                        if ta == 0:
                            continue
                        ia = (ta - 1) * NX + d_
                        qv[ia] += 2 * g * ca * const[d_]
                        for (tb, cb) in terms:
                            if tb == 0:
                                continue
                            P[ia, (tb - 1) * NX + d_] += 2 * g * ca * cb
        quad_diff([(0, 1.0), (-1, -1.0)], w['v'], sc, range(1, N + 1))
        quad_diff([(1, 1.0), (0, -2.0), (-1, 1.0)], w['a'], sc, range(0, N + 1))   # 中心二階差分，k＝0…N（含起步與收尾兩端）；與 merit 同式
        for i in range(n, n + ns):
            P[i, i] = 2 * w['slack']
        for i in range(n + ns, nz - 1):
            P[i, i] = 2 * w['col']
        P[nz - 1, nz - 1] = 2 * w['term']
        A, lo, hi = [], [], []

        def add(row_idx, vals, l_, h_):
            A.append((row_idx, vals))
            lo.append(l_)
            hi.append(h_)
        # 閉鏈：J δx − Jφ δφ + s＝e
        for k in range(1, N + 1):
            e, J, Jp = self.closure(X[k])
            s0 = idx(k).start
            for r in range(len(e)):
                cols = list(range(s0, s0 + 9)) + [s0 + 9, n + (k - 1) * m_c + r]
                add(cols, list(J[r]) + [-Jp[r], 1.0], e[r], e[r])
        # 界限與信賴域（合併成 δ 的箱）
        tr = np.r_[0.05, 0.05, 0.10, np.full(6, 0.15), 0.15] * rho
        for k in range(1, N + 1):
            s0 = idx(k).start
            for d_ in range(NX):
                l_, h_ = -tr[d_], tr[d_]
                if 3 <= d_ < 9:
                    l_, h_ = max(l_, LO[d_ - 3] - X[k, d_]), min(h_, HI[d_ - 3] - X[k, d_])
                if d_ == 9:
                    l_, h_ = max(l_, 0.0 - X[k, 9]), min(h_, PHI_MAX - X[k, 9])
                if d_ < 3 and self.base_fixed:
                    l_ = h_ = self.x0[d_] - X[k, d_]
                add([s0 + d_], [1.0], min(l_, h_), max(l_, h_))
        # 速率：手臂、門角（一階差分）；底盤輪級（本體速度，θ 取目前軌跡）
        for k in range(1, N + 1):
            s0, sp = idx(k).start, (idx(k - 1).start if k > 1 else None)
            for d_ in list(range(3, 9)) + [9]:
                lim = (V_ARM if d_ < 9 else PHI_RATE) * dt
                base = X[k, d_] - X[k - 1, d_]
                cols, vals = [s0 + d_], [1.0]
                if sp is not None:
                    cols.append(sp + d_)
                    vals.append(-1.0)
                add(cols, vals, -lim - base, lim - base)
            th = 0.5 * (X[k, 2] + X[k - 1, 2])
            c_, s_ = math.cos(th), math.sin(th)
            Rt = np.array([[c_, s_, 0.0], [-s_, c_, 0.0], [0.0, 0.0, 1.0]])
            M = WM @ Rt                                             # 輪面速度＝W·本體速度；本體速度＝Rᵀ·Δ/dt
            base = M @ (X[k, :3] - X[k - 1, :3])
            for r in range(4):
                cols, vals = [s0, s0 + 1, s0 + 2], list(M[r])
                if sp is not None:
                    cols += [sp, sp + 1, sp + 2]
                    vals += list(-M[r])
                add(cols, vals, -0.97 * W_LIM * dt - base[r], 0.97 * W_LIM * dt - base[r])   # 0.97：θ 取目前軌跡的線性化誤差餘量
        # 加速度（二階差分；k＝0 之前與 k＝N 之後視為靜止）
        for k in range(0, N + 1):                                   # 底盤加速度：本體座標（與執行端同語意：vx 1.5、vy 1.0、ωz 2.0），θ 取目前軌跡；0.9 為線性化餘量
            th = X[k, 2]
            c_, s_ = math.cos(th), math.sin(th)
            Rt = np.array([[c_, s_, 0.0], [-s_, c_, 0.0], [0.0, 0.0, 1.0]])
            terms = [(min(k + 1, N), 1.0), (k, -2.0), (max(k - 1, 0), 1.0)]
            base = Rt @ sum(c * X[t, :3] for t, c in terms)
            for r in range(3):
                cols, vals = [], []
                for t, c in terms:
                    if t >= 1:
                        cols += [(t - 1) * NX, (t - 1) * NX + 1, (t - 1) * NX + 2]
                        vals += list(c * Rt[r])
                if cols:
                    add(cols, vals, -0.9 * A_BODY[r] * dt * dt - base[r], 0.9 * A_BODY[r] * dt * dt - base[r])
        acc = np.r_[A_BASE, np.full(6, A_ARM), 5.0] * dt * dt
        for k in range(0, N + 1):
            for d_ in (9,):                                        # 門角二階差分；手臂由設定點速率列限制、底盤由上面的本體座標列限制
                terms = [(k + 1, 1.0), (k, -2.0), (k - 1, 1.0)]
                terms = [(min(max(t, 0), N), c) for t, c in terms]
                base = sum(c * X[t, d_] for t, c in terms)
                cols, vals = [], []
                for t, c in terms:
                    if t >= 1:
                        cols.append((t - 1) * NX + d_)
                        vals.append(c)
                if cols:
                    add(cols, vals, -acc[d_] - base, acc[d_] - base)
        # 可操作度下限（線性化；與淨空共用重罰鬆弛的作法：這裡直接硬列，起始已低於下限時改為「不得再降」）
        for k in range(1, N + 1):
            sg, gg = self.sig[k - 1]
            s0 = idx(k).start
            add(list(range(s0 + 3, s0 + 9)), list(gg), min(SIG_FLOOR - sg, 0.0), np.inf)
        # 設定點模型所需命令速率：(q_{k+1}−q_k)/dt＋τ·(q_{k+1}−2q_k＋q_{k−1})/dt² ∈ ±V_ARM（k＝0…N−1；k−1<0 取 x_0）
        for k in range(0, N):
            for d_ in range(3, 9):
                terms = [(k + 1, 1.0 / dt + TAU_SP / dt ** 2), (k, -1.0 / dt - 2 * TAU_SP / dt ** 2), (max(k - 1, 0), TAU_SP / dt ** 2)]
                base = sum(c * X[t, d_] for t, c in terms)
                cols, vals = [], []
                for t, c in terms:
                    if t >= 1:
                        cols.append((t - 1) * NX + d_)
                        vals.append(c)
                add(cols, vals, -V_ARM - base, V_ARM - base)
        # 終點門角（硬性）：φ_N ≥ φ_goal
        add([idx(N).start + 9, nz - 1], [1.0, 1.0], self.phi_goal + 2e-4 - X[N, 9], np.inf)   # 2e-4 rad：大於 QP 容差 1e-6 與投影誤差的餘量（同輪速 0.97 的作法）   # 彈性：φ_N＋s_T ≥ φ_goal，s_T 重罰；起始猜測未達目標時 QP 不會必然無解
        add([nz - 1], [1.0], 0.0, np.inf)
        # 淨空（線性化；鬆弛重罰）
        for j, (k, mg, g) in enumerate(col):
            s0 = idx(k).start
            add(list(range(s0, s0 + NX)) + [n + ns + j], list(g) + [1.0], self.keep - mg, np.inf)
            add([n + ns + j], [1.0], 0.0, np.inf)
        rows, cols, vals = [], [], []
        for i, (c_, v_) in enumerate(A):
            rows += [i] * len(c_)
            cols += c_
            vals += v_
        Am = sparse.csc_matrix((vals, (rows, cols)), shape=(len(A), nz))
        pr = osqp.OSQP()
        pr.setup(P=sparse.triu(P.tocsc()).tocsc(), q=qv, A=Am, l=np.array(lo), u=np.array(hi), verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=True)
        r = pr.solve()
        if r.info.status_val not in (1, 2) or r.x is None:
            return None, r.info.status
        return r.x[:n].reshape(N, NX), r.info.status

    def active_collisions(self):
        """接近門檻的節點：回 [(k, 該類餘裕, 對 10 維的梯度)]（有限差分）。"""
        out, worst = [], np.inf
        h = np.r_[1e-4, 1e-4, 1e-4, np.full(6, 1e-4), 1e-4]
        for k in range(1, self.N + 1):
            m0, _ = margins(self.X[k], self.T_WH0)
            worst = min(worst, float(m0.min()))
            for c in np.where(m0 < self.act)[0]:
                if c == 3 and self.mode == 'rigid':                 # 剛性抓持下夾爪殼對橫桿的距離由抓持幾何固定，不是可最佳化的量；只在重核時核門檻
                    continue
                g = np.zeros(NX)
                for d_ in range(NX):
                    xp = self.X[k].copy()
                    xp[d_] += h[d_]
                    g[d_] = (margins(xp, self.T_WH0)[0][c] - m0[c]) / h[d_]
                out.append((k, float(m0[c]), g))
        return out, worst

    def solve(self, max_iter=80, verbose=False):
        t0 = time.perf_counter()
        rho, ok, why = 1.0, False, 'max_iter'
        m_cur = self.merit(self.X)
        self.log.append(dict(it=-1, merit=round(m_cur[0], 4), J=round(m_cur[1], 4), viol=round(m_cur[2], 5)))
        for it in range(max_iter):
            col, worst = self.active_collisions()
            self.sig = [sigma_grad(x) for x in self.X[1:]]
            D, status = self.qp(rho, col)
            if D is None:
                rho *= 0.5
                self.log.append(dict(it=it, qp=status, rho=rho))
                if rho < 1e-3:
                    why = 'qp_infeasible'
                    break
                continue
            Xn = self.X.copy()
            Xn[1:] += D
            rp, rr = self.residual(Xn[1:])
            step = float(np.abs(D / (np.r_[0.05, 0.05, 0.10, np.full(6, 0.15), 0.15])).max())
            if rp > 0.02 or rr > 0.08:                              # 線性化失準：縮信賴域重解
                rho *= 0.5
                self.log.append(dict(it=it, reject=True, rp=rp, rr=rr, rho=rho))
                if rho < 1e-3:
                    why = 'trust_region_collapsed'
                    break
                continue
            Xn[1:] = np.array([self.project(x) for x in Xn[1:]])
            m_new = self.merit(Xn)
            if m_new[0] >= m_cur[0] - 1e-6:                         # 目標沒有下降：不接受，縮信賴域
                rho *= 0.5
                self.log.append(dict(it=it, reject='merit', merit=round(m_new[0], 4), rho=rho))
                if rho < 0.02:
                    ok, why = False, 'stalled_trust_region'        # 步長縮到下限仍無下降：停滯，不視為已證明局部最佳
                    break
                continue
            dec = m_cur[0] - m_new[0]
            self.X, m_cur = Xn, m_new
            rho = min(1.0, rho * 1.3)
            self.log.append(dict(it=it, step=round(step, 4), merit=round(m_new[0], 4), J=round(m_new[1], 4), viol=round(m_new[2], 5), dec=round(dec, 5), worst_margin=round(worst, 4), phi_end_deg=round(math.degrees(self.X[-1, 9]), 2), n_col=len(col), rho=rho))
            if verbose:
                print(self.log[-1], flush=True)
            if dec < 1e-3 * max(1.0, abs(m_new[0])):
                ok, why = True, 'converged'
                break
        self.info = dict(init=self.init, ok=ok, why=why, iters=len(self.log), wall_s=round(time.perf_counter() - t0, 1))
        return self.info

    # ---- 可執行參考與非線性重核（以此為準）
    def dense_reference(self, sub=4):
        """**實際要執行的參考**（r2 定義）：以節點為控制點的均勻二次 B 樣條（兩端各補一個重複控制點＝起步、收尾靜止），每 dt/sub 取樣；
        底盤位姿與門角直接取樣條值，手臂以樣條值為起點做牛頓投影到閉鏈殘差數值零。這條密集序列本身就是參考（不是驗收時才修正）。
        樣條性質：速度連續、各段加速度為常數且恰等於控制點二階差分／dt²（＝QP 內受限的量）；樣條不通過節點（在節點處為相鄰三點的 1/8、6/8、1/8 加權）。
        總時長 (N+1)·dt。回 (X_dense, dt_dense, 手臂投影調整量最大值)。"""
        P = np.vstack([self.X[0], self.X, self.X[-1]])             # P[i]＝x_{i-1}，i＝0…N+2
        out, adj = [], 0.0
        for k in range(self.N + 1):                                # 第 k 段：控制點 x_{k-1}, x_k, x_{k+1}
            a_, b_, c_ = P[k], P[k + 1], P[k + 2]
            for j in range(sub if k < self.N else sub + 1):
                u = j / sub
                x = 0.5 * ((1 - u) ** 2 * a_ + (-2 * u * u + 2 * u + 1) * b_ + u * u * c_)
                xp = self.project(x, iters=12)
                adj = max(adj, float(np.abs(xp[3:9] - x[3:9]).max()))
                out.append(xp)
        return np.array(out), self.dt / sub, adj

    def verify(self, sub=4):
        Xd, h, adj = self.dense_reference(sub)
        n = len(Xd) - 1
        r = np.array([self.closure(x)[0] for x in Xd])
        # 未投影的線性內插（對照；r1 驗收器漏掉這項）
        raw = np.array([self.closure(0.5 * (self.X[k] + self.X[k + 1]))[0] for k in range(self.N)])
        mn, who = {k: np.inf for k in THR}, {}
        for x in Xd:
            c = clear_door(x, self.T_WH0)
            for k in THR:
                if c[k] < mn[k]:
                    mn[k] = c[k]
                    if k in c['who']:
                        who[k] = c['who'][k]
        dX = np.diff(Xd, axis=0) / h
        vb = np.zeros((n, 3))
        for k in range(n):
            th = 0.5 * (Xd[k, 2] + Xd[k + 1, 2])
            c_, s_ = math.cos(th), math.sin(th)
            vb[k] = [c_ * dX[k, 0] + s_ * dX[k, 1], -s_ * dX[k, 0] + c_ * dX[k, 1], dX[k, 2]]
        wheel = np.abs(vb @ WM.T)
        vbe = np.vstack([np.zeros(3), vb, np.zeros(3)])            # 起步前與收尾後靜止
        ab = np.abs(np.diff(vbe, axis=0)) / h                      # 本體座標加速度（與執行端逐拍有界減速同語意）
        wa = np.abs(np.diff(vbe @ WM.T, axis=0)) / h               # 輪面加速度 m/s²；上限 r·α_max
        qd = dX[:, 3:9]
        qdd = np.diff(np.vstack([np.zeros(6), qd]), axis=0) / h
        s_rate = np.abs(qd + TAU_SP * qdd)
        jm = np.minimum(Xd[:, 3:9] - (LO - 0.05), (HI + 0.05) - Xd[:, 3:9])
        sig = np.array([sigma_min(x) for x in Xd])
        phi = Xd[:, 9]
        pd = np.abs(np.diff(phi)) / h
        t_at = lambda th_: (lambda w_: None if not len(w_) else round(float(w_[0] * h), 2))(np.where(phi >= th_)[0])   # noqa: E731
        v = dict(reference='密集投影參考（dt %.3f s；底盤與門角＝節點二次 B 樣條、手臂投影）' % h, n_dense=n + 1, proj_adjust_max_rad=round(adj, 5),
                 closure_pos_mm=round(float(np.abs(r[:, :3]).max()) * 1e3, 6), closure_rot_deg=round(math.degrees(float(np.abs(r[:, 3:]).max())), 6),
                 raw_midpoint_pos_mm=round(float(np.abs(raw[:, :3]).max()) * 1e3, 4), raw_midpoint_rot_deg=round(math.degrees(float(np.abs(raw[:, 3:]).max())), 4),
                 min_clear_m={k: round(float(v_), 4) for k, v_ in mn.items()}, min_clear_who=who, clear_ok=bool(all(mn[k] >= THR[k] for k in THR)),
                 wheel_max=round(float(wheel.max()), 4), wheel_ok=bool(wheel.max() <= W_LIM + 1e-4), wheel_sat_frac=round(float((wheel.max(axis=1) >= W_LIM - 2e-3).mean()), 3),
                 body_acc_max=[round(float(x), 3) for x in ab.max(axis=0)], body_acc_limit=A_BODY.tolist(), body_acc_ok=bool((ab.max(axis=0) <= A_BODY + 1e-6).all()),
                 wheel_acc_max=round(float(wa.max()), 3), wheel_acc_limit=0.05 * 125.0, wheel_acc_ok=bool(wa.max() <= 0.05 * 125.0 + 1e-6),
                 arm_rate_max=round(float(np.abs(qd).max()), 4), arm_rate_ok=bool(np.abs(qd).max() <= V_ARM + 1e-4),
                 setpoint_rate_needed_max=round(float(s_rate.max()), 4), setpoint_rate_ok=bool(s_rate.max() <= V_ARM + 1e-3), setpoint_sat_frac=round(float((s_rate.max(axis=1) >= V_ARM - 1e-2).mean()), 3),
                 joint_margin_min_rad=round(float(jm.min()), 4), joint_margin_min_joint=int(np.unravel_index(jm.argmin(), jm.shape)[1]) + 1, joint_ok=bool(jm.min() >= 0.05 - 1e-6),
                 sigma_min=round(float(sig.min()), 4), sigma_start=round(float(sig[0]), 4), sigma_floor=SIG_FLOOR, sigma_ok=bool(sig.min() >= min(SIG_FLOOR, sig[0]) - 1e-3),
                 sigma_note=None if sig[0] >= SIG_FLOOR else '起始低可操作度例外：起始值已低於下限，只保證不再降低；不得稱全程 σ_min ≥ 下限',
                 phi_end_deg=round(math.degrees(phi[-1]), 3), phi_end_ok=bool(phi[-1] >= self.phi_goal - 1e-6), phi_range_ok=bool(phi.min() >= -1e-9 and phi.max() <= PHI_MAX + 1e-9),
                 phi_rate_max=round(float(pd.max()), 4), phi_rate_ok=bool(pd.max() <= PHI_RATE + 1e-4), phi_monotone=bool((np.diff(phi) >= -1e-6).all()),
                 t_phi_95pct_s=t_at(0.95 * self.phi_goal), t_phi_84p5deg_s=t_at(self.phi_goal - math.radians(0.5)), t_phi_first_goal_s=t_at(self.phi_goal), t_ref_end_s=round(n * h, 2),
                 time_note='各時刻為 0.05 s 取樣序列上的首次到達；參考結束≠實際停穩（後者要執行後量）',
                 base_travel_m=round(float(np.linalg.norm(np.diff(Xd[:, :2], axis=0), axis=1).sum()), 3), base_net_m=round(float(np.linalg.norm(Xd[-1, :2] - Xd[0, :2])), 3), base_yaw_change_deg=round(math.degrees(Xd[-1, 2] - Xd[0, 2]), 1))
        keys = ('clear_ok', 'wheel_ok', 'body_acc_ok', 'wheel_acc_ok', 'arm_rate_ok', 'setpoint_rate_ok', 'joint_ok', 'sigma_ok', 'phi_end_ok', 'phi_range_ok', 'phi_rate_ok')
        v['closure_ok'] = bool(v['closure_pos_mm'] < 1e-3 and v['closure_rot_deg'] < 1e-3)
        v['failed'] = [k for k in keys + ('closure_ok',) if not v[k]]
        v['feasible'] = not v['failed']
        return v


def fixed_base_sweep(T_WH0, T_WE0, x0, phi_goal, mode='rigid', step_deg=0.5):
    """固定底盤可達性掃描（不經最佳化；Codex 163641：r1 的固定底盤案只是初始化失敗，不能當不可行證明）。
    底盤釘在起始位姿，門角每 step_deg 前進一步，手臂以連續牛頓投影跟隨；任一步閉鏈解不到、關節到限位、淨空不過、或 σ_min 低於 min(下限, 起始值) 就停。
    回最大可達門角與停止原因。這是「沿這條連續分支」的結果：不排除另一個 IK 分支可行，不稱已證明不可行。"""
    o = DoorWBOpt(T_WH0, T_WE0, x0, phi_goal, N=1, mode=mode, base_fixed=True)
    x, s0 = o.x0.copy(), sigma_min(o.x0)
    last, why = 0.0, 'reached_goal'
    for phi in np.arange(math.radians(step_deg), phi_goal + 1e-9, math.radians(step_deg)):
        xn = x.copy()
        xn[9] = phi
        xn = o.project(xn, iters=30)
        e = o.closure(xn)[0]
        if float(np.abs(e).max()) > 1e-6:
            why = 'closure_unreachable（手臂到限位或構型無解）'
            break
        if float(np.minimum(xn[3:9] - LO, HI - xn[3:9]).min()) <= 1e-9:
            why = 'joint_limit'
            break
        m, c = margins(xn, T_WH0)
        if m.min() < 0:
            why = 'clearance:' + str({k: round(float(c[k]), 4) for k in THR})
            break
        if sigma_min(xn) < min(SIG_FLOOR, s0) - 1e-3:
            why = 'sigma_below_floor'
            break
        x, last = xn, phi
    return dict(phi_max_deg=round(math.degrees(last), 1), why=why, sigma_at_stop=round(sigma_min(x), 4), joint_margin_at_stop=round(float(np.minimum(x[3:9] - LO, HI - x[3:9]).min()), 4), note='沿起始 IK 分支的連續掃描；不是全分支不可行證明')
