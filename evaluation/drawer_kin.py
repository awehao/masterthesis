"""抽屜案例的運動學與餘裕評估 —— **檢查端與軌跡產生端共用同一份**。

分開寫兩份的下場這個專案已經付過代價（外接圓半徑 vs 真實幾何）。這裡把
IK 策略、設計接觸對的排除規則、手指開度的處理都放在一個地方。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import solve_ik                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                     # noqa: E402
from verify_self_collision import link_clouds, DESIGNED_CONTACT          # noqa: E402
import drawer_asset as DA                                                # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]

# 工具座標在**世界**的朝向，對應底盤 yaw 90°：
#   工具 z（接近）= 世界 +y   工具 y（手指閉合）= 世界 +z   工具 x = 世界 −x
# 底盤座標下就是工具 z = +x、工具 y = +z，也就是 joint6 = −π/2 的分支。
# 這個矩陣只在「底盤 yaw = 90°」時成立；換停放朝向必須重算，不能照抄。
R_DES_WORLD = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])

# 手指在 URDF 零位是全閉；WholeBodyKinematics 的 dof_names 不含 finger_joint，
# FK 一律當全閉。實際開度要自己把點雲沿各自的軸平移。
FINGER_AXIS = {'uflite_finger1': np.array([0.0, 1.0, 0.0]),
               'uflite_finger2': np.array([0.0, -1.0, 0.0])}
FINGERS = tuple(FINGER_AXIS)


class DrawerKin:
    def __init__(self, urdf_path: str, spec: dict, pose, park):
        self.xml = open(urdf_path).read()
        self.K = WholeBodyKinematics.from_urdf_string(self.xml)
        self.clouds = link_clouds(self.xml)
        self.spec = spec
        self.pose = (float(pose[0]), float(pose[1]))
        self.park = (float(park[0]), float(park[1]), float(park[2]))
        self.idx = [self.K.dof_names.index(j) for j in ARM]

        adj, rigid = set(), {}

        def find(x):
            while rigid.get(x, x) != x:
                x = rigid[x]
            return x
        for j in self.K.joints.values():
            adj.add(frozenset((j.parent, j.child)))
            if j.jtype not in ('revolute', 'prismatic', 'continuous'):
                p, c = find(j.parent), find(j.child)
                if p != c:
                    rigid[p] = c
        self.names = [n for n in self.clouds
                      if n in self.K.parent_of or n == 'base_link']
        self.pairs = [(x, y) for i, x in enumerate(self.names)
                      for y in self.names[i + 1:]
                      if frozenset((x, y)) not in adj
                      and frozenset((x, y)) not in DESIGNED_CONTACT
                      and find(x) != find(y)]
        self.arm_links = [n for n in self.names
                          if n.startswith('link') or n.startswith('uflite')]
        rng = np.random.default_rng(0)
        self.seeds = [np.array([0.0, 0.5, 1.0, 0.0, -1.0, -math.pi / 2]),
                      np.array([0.0, 0.1, 0.2, 0.0, -1.5, -math.pi / 2]),
                      np.array([0.0, -0.3, 0.6, 0.0, -0.4, -math.pi / 2])] + [
                      rng.uniform(LITE6_SAFE.lower, LITE6_SAFE.upper)
                      for _ in range(7)]

    # ------------------------------------------------------------------ IK
    def _solve(self, tcp_w, seed, R=None):
        T = np.eye(4)
        T[:3, :3] = R_DES_WORLD if R is None else np.asarray(R, float)
        T[:3, 3] = tcp_w
        q = np.zeros(len(self.K.dof_names))
        q[0], q[1], q[2] = self.park
        q[self.idx] = seed
        return solve_ik(self.K, q, T)

    def margin(self, qa):
        v = [min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
             for k in range(6)]
        return float(min(v)), int(np.argmin(v)) + 1

    def ik_pose(self, T_des, prev, min_margin=0.05):
        """對完整 4x4 目標位姿解 IK（對準版的拉開段用）。"""
        T_des = np.asarray(T_des, float)
        return self.ik_at(T_des[:3, 3], prev, min_margin, R=T_des[:3, :3])

    def ik_at(self, tcp_w, prev, min_margin=0.05, R=None):
        """能延續上一點的解就延續（保持關節空間連續），不能才換種子。

        單一種子沿路徑暖啟動會卡在一個 IK 分支；但每點獨立取「餘裕最大」的解
        又可能在相鄰點之間跳分支，接起來是一條不能執行的軌跡。
        回傳 (IKResult, 是否換了分支)。
        """
        if prev is not None:
            r = self._solve(tcp_w, prev, R)
            m, _ = self.margin(r.q[self.idx])
            if r.ok and r.pos_err < 1e-3 and m >= min_margin:
                return r, False
        best, bm = None, -1.0
        for sd in self.seeds:
            r = self._solve(tcp_w, sd, R)
            if not (r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2):
                continue
            m, _ = self.margin(r.q[self.idx])
            if m > bm:
                best, bm = r, m
        if best is None:
            return self._solve(tcp_w, prev if prev is not None else self.seeds[0],
                               R), True
        return best, True

    def q_full(self, qa):
        q = np.zeros(len(self.K.dof_names))
        q[0], q[1], q[2] = self.park
        q[self.idx] = np.asarray(qa, float)
        return q

    def tcp_world(self, qa):
        return self.K.fk(self.q_full(qa), 'link_tcp')

    # ------------------------------------------------------------- 餘裕評估
    def evaluate(self, qa, q_d, q_finger):
        """回傳 dict：自碰、對櫃體／抽屜、夾爪殼對橫桿。

        手指與把手橫桿是**設計接觸對**，全程從環境餘裕排除（一般環境的 5 mm
        門檻是給「不該碰到的東西」用的），另以解析包覆餘裕判定。
        """
        q = self.q_full(qa)
        shapes = DA.shapes_world(self.spec, self.pose, q_d)
        world = {}
        for n in self.names:
            Tm = self.K.fk(q, n)
            P = self.clouds[n]
            if n in FINGER_AXIS:
                P = P + FINGER_AXIS[n] * q_finger
            world[n] = (Tm[:3, :3] @ P.T).T + Tm[:3, 3]
        ds, dsw = np.inf, None
        for x, y in self.pairs:
            d = cKDTree(world[x]).query(world[y], k=1)[0].min()
            if d < ds:
                ds, dsw = float(d), f'{x}|{y}'
        de, dew = np.inf, None
        for n in self.arm_links:
            for sh in shapes:
                if n in FINGERS and sh[0] == 'handle/bar':
                    continue
                d, _ = DA.min_distance(world[n], [sh])
                if d < de:
                    de, dew = d, f'{n}|{sh[0]}'
        bar = [s for s in shapes if s[0] == 'handle/bar']
        shell, _ = DA.min_distance(world['uflite_gripper_link'], bar)
        return {'self': ds, 'self_pair': dsw, 'env': de, 'env_pair': dew,
                'shell_bar': float(shell)}


def enclosure_margin(spec: dict) -> float:
    """手指全開時對橫桿的**解析**包覆餘裕（單側）。

    不用點雲量：900 點不保證有點落在最近處，量到的會高估。
    """
    return (float(spec['grasp_surface']['finger_gap_open_m']) / 2.0
            - float(spec['drawer']['handle']['bar']['radius']))
