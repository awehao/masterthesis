"""以**實際抓取關係**生成拉開參考。

定義（全部在世界座標，engage 當下的同一個物理時刻取一次快照）：

    ᴳT_D = (ᵂT_G,0)⁻¹ ᵂT_D,0          連接鎖住的相對位姿，**只取一次**
    ᵂT_D^ref(s) = [ R_D,0 , p_D,0 + â s ]   滑軌真正允許的抽屜路徑
    ᵂT_G^ref(s) = ᵂT_D^ref(s) (ᴳT_D)⁻¹     反推的夾爪參考
    ᵂT_TCP^ref(s) = ᵂT_G^ref(s) ᴳT_TCP     IK 用 TCP 時再套固定變換

s = 0 時 ᵂT_G^ref 就等於連接當下的**實際**夾爪位姿。
**不是直接扣掉那 4.44 mm，也不是把物體瞬移到命令位置。**

抓取關係只取一次：每步重設等於把誤差掩蓋掉，那樣就量不到殘差了。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

# link_tcp 相對 uflite_gripper_link 的固定變換（URDF joint_tcp，無旋轉）
G_TO_TCP_Z = 0.0836


def quat_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-8:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    q = np.zeros(4)
    j, k = (i + 1) % 3, (i + 2) % 3
    t = math.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k]))
    q[i + 1] = t / 2
    q[0] = (R[k, j] - R[j, k]) / (2 * t)
    q[j + 1] = (R[j, i] + R[i, j]) / (2 * t)
    q[k + 1] = (R[k, i] + R[i, k]) / (2 * t)
    return q.tolist()


def iso(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t, float); return T


def ang_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0,
        (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2))))


def snapshot_from_run(run_dir: str) -> dict:
    """從一趟已完成的試驗取出 engage 當下的同步快照。

    夾爪／抽屜位姿取自 attach_frames（模擬器在同一個物理時刻讀回的實際值），
    關節角取自同一個模擬時刻的 log 列。底盤位姿與模擬時間一併帶出。
    """
    J = json.load(open(os.path.join(run_dir, 'sim', 'drawer_run.json')))
    ev = [e for e in J['events'] if e['event'] == 'engage']
    if not ev or 'attach_frames' not in ev[0]:
        raise ValueError(f'{run_dir} 沒有記錄 engage 的連接框架')
    AF = ev[0]['attach_frames']
    t_eng = float(ev[0]['sim_t'])
    i = {c: k for k, c in enumerate(J['log_cols'])}
    arm = [f'joint{n}' for n in range(1, 7)]
    ts = np.array([r[i['t']] for r in J['log']])
    k = int(np.argmin(np.abs(ts - t_eng)))
    q_meas = [float(J['log'][k][i[j]]) for j in arm]
    return {
        'source_run': os.path.basename(os.path.abspath(run_dir)),
        'sim_t': t_eng,
        'log_row_t': float(ts[k]),
        'gripper_world_pos': list(AF['gripper_world_pos']),
        'gripper_world_rot_wxyz': list(AF['gripper_world_rot_wxyz']),
        'drawer_world_pos': list(AF['drawer_world_pos']),
        'drawer_world_rot_wxyz': list(AF['drawer_world_rot_wxyz']),
        'q_measured': q_meas,
        'park': list(J['park']),
        'case': J['case'],
    }


class Alignment:
    """由快照建立的參考產生器。抓取關係只在建構時取一次。"""

    def __init__(self, snap: dict, axis_world):
        self.snap = snap
        self.a = np.asarray(axis_world, float)
        self.a = self.a / np.linalg.norm(self.a)
        self.T_WG0 = iso(quat_R(snap['gripper_world_rot_wxyz']),
                         snap['gripper_world_pos'])
        self.T_WD0 = iso(quat_R(snap['drawer_world_rot_wxyz']),
                         snap['drawer_world_pos'])
        self.T_GD = np.linalg.inv(self.T_WG0) @ self.T_WD0      # 只取一次
        self.T_GD_inv = np.linalg.inv(self.T_GD)
        self.T_G_TCP = iso(np.eye(3), [0.0, 0.0, G_TO_TCP_Z])

    def drawer_ref(self, s):
        return iso(self.T_WD0[:3, :3], self.T_WD0[:3, 3] + self.a * float(s))

    def gripper_ref(self, s):
        return self.drawer_ref(s) @ self.T_GD_inv

    def tcp_ref(self, s):
        return self.gripper_ref(s) @ self.T_G_TCP

    def residual(self, T_G_cmd):
        """一個夾爪命令位姿，對滑軌允許集合的絕對殘差。

        回傳 (沿軸 s, 橫向偏差向量, 橫向模長, 姿態誤差 deg)。
        """
        T_D = np.asarray(T_G_cmd) @ self.T_GD
        v = T_D[:3, 3] - self.T_WD0[:3, 3]
        s = float(self.a @ v)
        perp = v - self.a * s
        return s, perp, float(np.linalg.norm(perp)), ang_deg(T_D[:3, :3],
                                                             self.T_WD0[:3, :3])

    def describe(self):
        t = self.T_GD[:3, 3]
        return {'T_GD_pos': t.tolist(), 'T_GD_rot_wxyz': R_quat(self.T_GD[:3, :3]),
                'T_GD_pos_norm_m': float(np.linalg.norm(t)),
                'axis_world': self.a.tolist(),
                'drawer_ref_pos': self.T_WD0[:3, 3].tolist(),
                'drawer_ref_rot_wxyz': R_quat(self.T_WD0[:3, :3])}
