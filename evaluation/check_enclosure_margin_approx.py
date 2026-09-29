"""合併餘裕近似式的離線幾何核對（不跑模擬、不讀趟次資料）。

近似式（草案 M1）：
    m_approx = (finger_gap_open/2 − r_bar) − |t_y| − (L/2)·|a_y|

本檔用**真實手指網格**與**有限長圓柱**重算最小間距 m_num，在允許的相對位姿
範圍內比較兩者。目的是界定近似式可用的範圍，**不是**宣稱它等於真實間距。

模型與假設（每一項都會影響結論，逐條列出）：
  * 手指以 finger1.stl 的三角面表示；finger2 以 y 鏡射近似（未讀 finger2.stl）
  * 手指置於全開位置：finger_joint = 0.0089，內面在工具 y = ±0.0089
  * 工具座標原點為 link_tcp；uflite_gripper_link 在 z = −0.0836，
    finger_joint1 原點再 +0.0543 ⇒ 手指基準在 z = −0.0293
  * 橫桿：半徑 0.005、長 0.200 的**有限**圓柱，名目軸為工具 x
  * 間距為**無號**距離：一旦貫穿即為 0，不區分貫穿深度
"""
from __future__ import annotations
import itertools, math, os, struct, sys
import numpy as np

MESH = os.path.expanduser('~/masterthesis/install/xarm_description/share/'
                          'xarm_description/meshes/gripper/lite/visual/finger1.stl')
GAP_HALF, R_BAR, L_APPROX = 0.0089, 0.005, 0.020
Z_FINGER_BASE = -0.0836 + 0.0543
Z_BAR_NOM = -0.0147


def load_stl(path):
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    v = np.frombuffer(data[:, 12:48].tobytes(), dtype='<f4').reshape(n, 3, 3)
    return v.astype(float)


def fingers_tool_frame():
    """兩根手指的三角面，置於工具座標的全開位置。"""
    tri = load_stl(MESH)
    f1 = tri.copy()
    f1[:, :, 1] += GAP_HALF                      # 內面移到 +0.0089
    f1[:, :, 2] += Z_FINGER_BASE
    f2 = tri.copy()
    f2[:, :, 1] = -f2[:, :, 1] - GAP_HALF        # y 鏡射
    f2[:, :, 2] += Z_FINGER_BASE
    return np.concatenate([f1, f2], axis=0)


def bar_points(t, axis, n_ax=81, n_th=72, half_len=0.015):
    """橫桿表面取樣；只取工具 x 方向 ±half_len（手指涵蓋範圍稍外擴）。"""
    a = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(a @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u)
    w = np.cross(a, u)
    s = np.linspace(-half_len, half_len, n_ax)
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False)
    P = (t[None, None, :] + s[:, None, None] * a[None, None, :]
         + R_BAR * (np.cos(th)[None, :, None] * u[None, None, :]
                    + np.sin(th)[None, :, None] * w[None, None, :]))
    return P.reshape(-1, 3)


def pt_tri_dist(P, T, chunk=200):
    """點到三角面的最短距離（無號），分塊計算。"""
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    AB, AC = B - A, C - A
    best = np.full(len(P), np.inf)
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk][:, None, :]
        ap = p - A[None]
        d1 = np.einsum('ijk,jk->ij', ap, AB)
        d2 = np.einsum('ijk,jk->ij', ap, AC)
        a_ = np.einsum('jk,jk->j', AB, AB)[None]
        b_ = np.einsum('jk,jk->j', AB, AC)[None]
        c_ = np.einsum('jk,jk->j', AC, AC)[None]
        den = np.maximum(a_ * c_ - b_ * b_, 1e-18)
        s = (c_ * d1 - b_ * d2) / den
        t = (a_ * d2 - b_ * d1) / den
        s = np.clip(s, 0, 1); t = np.clip(t, 0, 1)
        over = s + t > 1
        ss = np.where(over, s / np.maximum(s + t, 1e-18), s)
        tt = np.where(over, t / np.maximum(s + t, 1e-18), t)
        q = A[None] + ss[..., None] * AB[None] + tt[..., None] * AC[None]
        best[i:i + chunk] = np.linalg.norm(p - q, axis=2).min(axis=1)
    return best


def m_approx(t_y, a_y):
    return (GAP_HALF - R_BAR) - abs(t_y) - (L_APPROX / 2) * abs(a_y)


def rot(ax, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if ax == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if ax == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main() -> int:
    T = fingers_tool_frame()
    print(f'手指三角面 {len(T)}（含鏡射）；名目餘裕 '
          f'{1000*(GAP_HALF-R_BAR):.1f} mm')
    print('掃描 |t_y|≤2.9、|t_z|≤4、傾角≤3°、|t_x|≤5 mm')
    rows = []
    cases = []
    # 允許範圍：|t_y| ≤ 2.9 mm（m_approx ≥ 1.0）、插入深度偏差 |t_z| ≤ 4 mm
    # （擬議 H6 為 2 mm，測到 4 mm 以涵蓋邊界）、傾角 ≤ 3°、沿桿軸 ≤ 5 mm
    for ty in (0.0, 1.0, 2.0, 2.9):
        for tz in (-4.0, -2.0, 0.0, 2.0, 4.0):
            for tilt_deg, tilt_ax in ((0.0, 'z'), (3.0, 'z'), (3.0, 'y'), (3.0, 'x')):
                for tx in (0.0, 5.0):
                    cases.append((ty, tx, tz, tilt_deg, tilt_ax))
    for ty, tx, tz, tilt_deg, tilt_ax in cases:
        R = rot(tilt_ax, tilt_deg)
        axis = R @ np.array([1.0, 0.0, 0.0])
        t = np.array([tx / 1000.0, ty / 1000.0, Z_BAR_NOM + tz / 1000.0])
        P = bar_points(t, axis)
        num = float(pt_tri_dist(P, T).min())
        ap = m_approx(t[1], axis[1])
        rows.append((ty, tx, tz, tilt_deg, tilt_ax, ap, num, ap - num))
    rows.sort(key=lambda r: -(r[7]))
    print('  最樂觀（近似高估最多）的 6 組：')
    for ty, tx, tz, tilt_deg, tilt_ax, ap, num, df in rows[:6]:
        print(f'   t_y {ty:4.1f} t_x {tx:4.1f} t_z {tz:+5.1f} {tilt_deg:4.1f}°@{tilt_ax}'
              f'  近似 {1000*ap:7.3f}  數值 {1000*num:7.3f}  差 {1000*df:+7.3f}')
    d = np.array([r[7] for r in rows])
    print(f'\n近似 − 數值：最大 {1000*d.max():+.3f} mm、最小 {1000*d.min():+.3f} mm')
    print('正值 = 近似式高估間距（樂觀），負值 = 低估（保守）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
