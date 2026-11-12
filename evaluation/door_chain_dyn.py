#!/usr/bin/env python3
"""剛性閉鏈＋關節 PD 柔順的局部動力學模型（開門起步診斷用；Codex 20261112_234431）。不是接觸引擎、不是把 FixedJoint 換成彈簧。

廣義座標 x＝[底盤 x, y, θ（世界）, 手臂 q1..q6, 門角 φ]（10）。每個物理步（dt＝0.01 s）依 Isaac 執行器的語意：
  1. 底盤速度在步首重設為命令（本體速度以模型當下偏航轉到世界）——同 isaac_m8_house_door_open.py 的 set_linear_velocity／set_angular_velocity；
     步內底盤當自由剛體，受閉鏈與手臂反力改變速度（不當運動學底盤）。
  2. 手臂關節 PD 驅動隱式：τ＝kp(s−q⁺)−kd·q̇⁺，q⁺＝q＋dt·q̇⁺；超過 URDF 力矩上限則夾住（主動集再解）。
  3. 門：鉸鏈單自由度，鉸鏈慣量 I_h（資產門扇方盒＋平行軸）；PhysX 角阻尼 c 只作用於繞質心角速度 ⇒ 等效 c·I_cz（隱式）。
  4. 剛性閉鏈：TCP 位姿＝把手位姿(φ)·T_HE（綁定時的相對位姿；同計畫 plan.npz）；速度層 J_c v⁺＝−e/dt（位置殘差每步完全修正）。
  5. 重力：各連桿質心；底盤平面（法向接觸、地面摩擦＝0 不建模）。
速度層 KKT：[M̃  −J_cᵀ; J_c  0][v⁺; λ·dt]＝[M v⁻＋dt(Kp(s−q)＋G); −e/dt]，M̃＝M＋dt²Kp＋dt(Kd＋門阻尼)。x⁺＝x＋dt·v⁺。
手腕 joint6 力旋量（父對子，link6 座標、link6 原點）：W_j＝d(動量_S)/dt − W_c − W_g，S＝link6＋link_eef＋夾爪殼＋兩指。
未建模（揭露）：關節摩擦（USD physxJoint:jointFriction＝1，PhysX 依關節反力的摩擦）、科氏／離心項（低速）、輪子轉動自由度與地面接觸、PhysX TGS 迭代與位置修正細節、手指驅動柔順。
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO                                     # noqa: E402

K, W = XO.K, XO.W
WS = os.path.dirname(HERE)
URDF = os.path.join(WS, 'evaluation', 'models', 'omni_bot_manip.urdf')
DT, KP, KD = 0.01, 600.0, 60.0
GRAV = np.array([0.0, 0.0, -9.81])
S_LINKS = ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def parse_urdf(path=URDF):
    root = ET.parse(path).getroot()
    inert, effort = {}, {}
    for l in root.findall('link'):
        i = l.find('inertial')
        if i is None:
            continue
        m = float(i.find('mass').get('value'))
        o = i.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        a = i.find('inertia').attrib
        I = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])], [float(a['ixy']), float(a['iyy']), float(a['iyz'])], [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
        Rc = rpy_R(*rpy)
        if m > 0:
            inert[l.get('name')] = (m, xyz, Rc @ I @ Rc.T)
    for j in root.findall('joint'):
        lim = j.find('limit')
        if lim is not None and lim.get('effort') is not None:
            effort[j.get('name')] = float(lim.get('effort'))
    return inert, effort


INERT, EFFORT = parse_urdf()
KNOWN = set(sorted({j.child for j in K.joints.values()}))
BODIES = [b for b in INERT if b in KNOWN]
MISSING = [b for b in INERT if b not in KNOWN]
TAU_MAX = np.array([EFFORT[f'joint{i}'] for i in range(1, 7)])
# 門（資產 door_local_v3.yaml）
M_LEAF, L_LEAF, T_LEAF, C_ANG = 8.0, 0.95, 0.04, 2.0
I_CZ = M_LEAF * (L_LEAF ** 2 + T_LEAF ** 2) / 12.0
I_H = I_CZ + M_LEAF * ((L_LEAF / 2) ** 2 + (T_LEAF / 2) ** 2)


def body_terms(q9):
    """回 M9（9×9）、G9（9）與各剛體 (J, 世界慣量, 質量, 質心)；只含 K 認得的剛體。"""
    M9, G9, parts = np.zeros((9, 9)), np.zeros(9), {}
    for b in BODIES:
        m, com, Ib = INERT[b]
        J = K.jacobian(q9, b, offset=com)
        R = K.fk(q9, b)[:3, :3]
        Iw = R @ Ib @ R.T
        Jv, Jw = J[:3], J[3:]
        M9 += m * Jv.T @ Jv + Jw.T @ Iw @ Jw
        G9 += Jv.T @ (m * GRAV)
        parts[b] = (J, Iw, m, (K.fk(q9, b) @ np.r_[com, 1.0])[:3])
    return M9, G9, parts


class ChainModel:
    def __init__(self, T_WH_bind, phi_bind, T_HE):
        self.T_WH_bind, self.phi_bind, self.T_HE = np.asarray(T_WH_bind, float), float(phi_bind), np.asarray(T_HE, float)
        self.delta = np.eye(4)                                       # 診斷用：夾爪相對門扇的偏移（TCP 座標；預設無＝理想剛性）

    def target(self, phi):
        return W.handle_pose(self.T_WH_bind, phi) @ self.T_HE @ self.delta

    def residual(self, x):
        Tt, Tg = K.fk(x[:9], 'link_tcp'), self.target(x[9])
        dR = Tt[:3, :3] @ Tg[:3, :3].T
        return np.r_[Tt[:3, 3] - Tg[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]

    def Jc(self, x):
        J = np.zeros((6, 10))
        J[:, :9] = K.jacobian(x[:9], 'link_tcp')
        eps = 1e-6                                                     # 目標對 φ 的速度（把手繞鉸鏈）：有限差分（含方向慣例）
        T0, T1 = self.target(x[9]), self.target(x[9] + eps)
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[:, 9] = -np.r_[(T1[:3, 3] - T0[:3, 3]) / eps, 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / eps]
        return J

    def step(self, x, v_prev, s, u_body, base_prescribed=None):
        """一個物理步。u_body＝本體速度命令 (vx, vy, ω)；回 (x⁺, v⁺, info)。
        base_prescribed＝世界座標底盤速度（診斷對照用：底盤照給定速度、不受力；None＝Isaac 語意的步首重設＋步內自由）。"""
        q9 = x[:9]
        M9, G9, parts = body_terms(q9)
        M = np.zeros((10, 10)); M[:9, :9] = M9; M[9, 9] = I_H
        G = np.r_[G9, 0.0]
        th = x[2]
        vm = v_prev.copy()
        vm[:3] = [u_body[0] * math.cos(th) - u_body[1] * math.sin(th), u_body[0] * math.sin(th) + u_body[1] * math.cos(th), u_body[2]]   # 步首重設底盤速度
        Kp, Kd = np.zeros(10), np.zeros(10)
        Kp[3:9], Kd[3:9] = KP, KD
        Dd = np.zeros(10); Dd[9] = C_ANG * I_CZ
        e = self.residual(x)
        Jc = self.Jc(x)
        free = np.ones(6, bool)
        tau_fix = np.zeros(6)
        for _ in range(4):                                             # 力矩上限主動集
            Kp_e, Kd_e = Kp.copy(), Kd.copy()
            rhs_tau = np.zeros(10)
            for i in range(6):
                if free[i]:
                    rhs_tau[3 + i] = KP * (s[i] - x[3 + i])
                else:
                    Kp_e[3 + i] = Kd_e[3 + i] = 0.0
                    rhs_tau[3 + i] = tau_fix[i]
            Mt = M + np.diag(DT * DT * Kp_e + DT * (Kd_e + Dd))
            if base_prescribed is None:
                A = np.block([[Mt, -Jc.T], [Jc, np.zeros((6, 6))]])
                b = np.r_[M @ vm + DT * (rhs_tau + G), -e / DT]
                sol = np.linalg.solve(A, b)
                vp, lam = sol[:10], sol[10:] / DT
            else:                                                        # 底盤速度給定：底盤三列改為速度等式（反力由給定者吸收）
                Sb = np.zeros((3, 10)); Sb[:, :3] = np.eye(3)
                A = np.block([[Mt, -Jc.T, -Sb.T], [Jc, np.zeros((6, 9))], [Sb, np.zeros((3, 9))]])
                b = np.r_[M @ vm + DT * (rhs_tau + G), -e / DT, np.asarray(base_prescribed, float)]
                sol = np.linalg.solve(A, b)
                vp, lam = sol[:10], sol[10:16] / DT
            tau = np.where(free, KP * (s - x[3:9] - DT * vp[3:9]) - KD * vp[3:9], tau_fix)
            over = free & (np.abs(tau) > TAU_MAX)
            if not over.any():
                break
            free &= ~over
            tau_fix = np.where(over, np.sign(tau) * TAU_MAX, tau_fix)
        xp = x + DT * vp
        # 手腕 joint6 力旋量（父對子；link6 原點、link6 座標）
        To = K.fk(q9, 'link6')
        po, Ro = To[:3, 3], To[:3, :3]
        acc = (vp - vm) / DT
        Fd, Nd, Fg, Ng = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for b in S_LINKS:
            if b not in parts:
                continue
            J, Iw, m, pc = parts[b]
            a_lin, a_ang = J[:3] @ acc[:9], J[3:] @ acc[:9]
            r = pc - po
            Fd += m * a_lin
            Nd += Iw @ a_ang + np.cross(r, m * a_lin)
            Fg += m * GRAV
            Ng += np.cross(r, m * GRAV)
        pt = K.fk(q9, 'link_tcp')[:3, 3]
        Fc, Nc = lam[:3], lam[3:] + np.cross(pt - po, lam[:3])       # 門對夾爪（作用在 TCP 的 λ）移到 link6 原點
        Fj, Nj = Fd - Fc - Fg, Nd - Nc - Ng
        info = dict(lam=lam, tau=tau, sat=(~free).tolist(), e_pre=e, Wj_link6=np.r_[Ro.T @ Fj, Ro.T @ Nj], v_reset=vm[:3].copy())
        return xp, vp, info
