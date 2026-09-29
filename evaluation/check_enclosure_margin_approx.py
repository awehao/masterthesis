"""合併餘裕近似式的離線幾何核對（不跑模擬、不讀趟次資料）。

近似式（草案 M1）：
    m_approx = (finger_gap_open/2 − r_bar) − |t_y| − (L/2)·|a_y|

本檔用**兩根手指的實際網格**與**完整有限圓柱**重算最小間距 m_num，
在指定的相對位姿範圍內比較，用以界定近似式可用的範圍。
**不宣稱 m_approx 等於真實最小間距。**

幾何來源與限制（逐條列出，結論只在這些條件下成立）
--------------------------------------------------
  * 手指：URDF 的 collision 直接引用 visual STL（finger1.stl／finger2.stl），
    兩根各自讀取，**不用鏡射近似**。
    **但模擬引擎實際使用的凸分解形狀未經本檔驗證。**
  * 手指全開：finger_joint1 = +0.0089（axis +y）、finger_joint2 = −0.0089
    （axis −y，mimic finger_joint1）。手指基準在工具 z = −0.0836 + 0.0543。
  * 橫桿：半徑 0.005、長 0.200 的完整有限圓柱，**含側面與兩個端面**。
  * 距離為**無號**：貫穿後為 0，不表示貫穿深度。

數值方法
--------
  * 點到三角面：完整區域判定（面內／三邊／三頂點），見 closest_point_triangle，
    並有可手算的單元測試（--selftest）。
  * 圓柱表面離散取樣：距離函數對位置是 1-Lipschitz，因此若任一表面點
    與最近取樣點的距離不超過 h，則真實最小距離 ≥ 取樣最小距離 − h。
    h 由取樣間距算出並隨結果一併印出；比較時使用下界 (m_num − h)。
  * 取樣排除：與手指三角面 AABB 距離 > CULL 的取樣點直接略過。
    凡本檔報出的最小值都 < CULL，故排除不影響結果。
"""
from __future__ import annotations

import argparse
import math
import os
import struct

import numpy as np

MESH_DIR = os.path.expanduser('~/masterthesis/install/xarm_description/share/'
                              'xarm_description/meshes/gripper/lite/visual')
GAP_HALF, R_BAR, BAR_LEN, L_APPROX = 0.0089, 0.005, 0.200, 0.020
Z_FINGER_BASE = -0.0836 + 0.0543
Z_BAR_NOM = -0.0147
CULL = 0.010            # 取樣排除半徑；所有報出的距離都小於它


# ---------------------------------------------------------------- 幾何工具
def closest_point_triangle(P, A, B, C):
    """點集 P 對單一三角面的最近點（Ericson 區域判定，逐區處理）。"""
    ab, ac = B - A, C - A
    ap = P - A
    d1 = ap @ ab
    d2 = ap @ ac
    bp = P - B
    d3 = bp @ ab
    d4 = bp @ ac
    cp = P - C
    d5 = cp @ ab
    d6 = cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = np.empty_like(P)
    done = np.zeros(len(P), dtype=bool)

    def put(mask, pts):
        m = mask & ~done
        if m.any():
            out[m] = pts[m] if pts.ndim == 2 else pts
            done[m] = True

    put((d1 <= 0) & (d2 <= 0), np.broadcast_to(A, P.shape))          # 頂點 A
    put((d3 >= 0) & (d4 <= d3), np.broadcast_to(B, P.shape))         # 頂點 B
    put((d6 >= 0) & (d5 <= d6), np.broadcast_to(C, P.shape))         # 頂點 C
    den = np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + (d1 / den)[:, None] * ab)   # 邊 AB
    den = np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + (d2 / den)[:, None] * ac)   # 邊 AC
    num = d4 - d3
    den = np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30,
                   (d4 - d3) + (d5 - d6))
    put((va <= 0) & (num >= 0) & ((d5 - d6) >= 0),
        B + (num / den)[:, None] * (C - B))                                # 邊 BC
    s = va + vb + vc
    s = np.where(np.abs(s) < 1e-30, 1e-30, s)
    put(np.ones(len(P), bool), A + (vb / s)[:, None] * ab
        + (vc / s)[:, None] * ac)                                          # 面內
    return out


def _closest_batch(P, A, B, C):
    """P:(n,3) 對一批三角面 A/B/C:(m,3) 的最近距離，回傳 (n,) 最小值。

    與 closest_point_triangle 同一組區域判定，只是對 (n,m) 廣播。
    單元測試比對兩者結果一致。
    """
    p = P[:, None, :]
    ab = (B - A)[None]; ac = (C - A)[None]
    ap = p - A[None]
    d1 = np.einsum('nmk,nmk->nm', ap, np.broadcast_to(ab, ap.shape))
    d2 = np.einsum('nmk,nmk->nm', ap, np.broadcast_to(ac, ap.shape))
    bp = p - B[None]
    d3 = np.einsum('nmk,nmk->nm', bp, np.broadcast_to(ab, bp.shape))
    d4 = np.einsum('nmk,nmk->nm', bp, np.broadcast_to(ac, bp.shape))
    cp = p - C[None]
    d5 = np.einsum('nmk,nmk->nm', cp, np.broadcast_to(ab, cp.shape))
    d6 = np.einsum('nmk,nmk->nm', cp, np.broadcast_to(ac, cp.shape))
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    q = np.broadcast_to(A[None], ap.shape).copy()
    done = (d1 <= 0) & (d2 <= 0)
    m = (~done) & (d3 >= 0) & (d4 <= d3)
    q[m] = np.broadcast_to(B[None], ap.shape)[m]; done |= m
    m = (~done) & (d6 >= 0) & (d5 <= d6)
    q[m] = np.broadcast_to(C[None], ap.shape)[m]; done |= m
    den = np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    m = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    q[m] = (A[None] + (d1 / den)[..., None] * ab)[m]; done |= m
    den = np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    m = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    q[m] = (A[None] + (d2 / den)[..., None] * ac)[m]; done |= m
    num = d4 - d3
    den = np.where(np.abs(num + (d5 - d6)) < 1e-30, 1e-30, num + (d5 - d6))
    m = (~done) & (va <= 0) & (num >= 0) & ((d5 - d6) >= 0)
    q[m] = (B[None] + (num / den)[..., None] * (C - B)[None])[m]; done |= m
    s_ = va + vb + vc
    s_ = np.where(np.abs(s_) < 1e-30, 1e-30, s_)
    m = ~done
    q[m] = (A[None] + (vb / s_)[..., None] * ab + (vc / s_)[..., None] * ac)[m]
    return np.linalg.norm(p - q, axis=2).min(axis=1)


def min_dist_points_mesh(P, T, chunk=24):
    """點集對三角面集的最小距離（無號）。"""
    best = np.full(len(P), np.inf)
    for i in range(0, len(T), chunk):
        S = T[i:i + chunk]
        best = np.minimum(best, _closest_batch(P, S[:, 0], S[:, 1], S[:, 2]))
    return best


def load_stl(name):
    with open(os.path.join(MESH_DIR, name), 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(raw[:, 12:48].tobytes(),
                         dtype='<f4').reshape(n, 3, 3).astype(float)


def fingers_tool_frame():
    """兩根手指的三角面，置於工具座標的全開位置（各讀自己的網格）。"""
    f1 = load_stl('finger1.stl')
    f1[:, :, 1] += GAP_HALF                      # finger_joint1 軸 +y
    f1[:, :, 2] += Z_FINGER_BASE
    f2 = load_stl('finger2.stl')
    f2[:, :, 1] -= GAP_HALF                      # finger_joint2 軸 −y（mimic）
    f2[:, :, 2] += Z_FINGER_BASE
    return np.concatenate([f1, f2], axis=0)


def bar_samples(t, axis, d_ax=0.00025, n_th=144, n_r=24):
    """完整有限圓柱的表面取樣：側面 ＋ 兩端面。回傳 (點集, Lipschitz 界 h)。"""
    a = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(a @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u)
    w = np.cross(a, u)
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False)
    ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * w
    s = np.arange(-BAR_LEN / 2, BAR_LEN / 2 + 1e-12, d_ax)
    side = (t[None, None, :] + s[:, None, None] * a[None, None, :]
            + R_BAR * ring[None, :, :]).reshape(-1, 3)
    rr = np.linspace(0.0, R_BAR, n_r)
    cap = []
    for sgn in (-1.0, 1.0):
        c = t + sgn * (BAR_LEN / 2) * a
        cap.append((c[None, None, :] + rr[:, None, None] * ring[None, :, :])
                   .reshape(-1, 3))
    P = np.concatenate([side] + cap, axis=0)
    # h 取自**側面**取樣間距：端面距手指約 100 mm，必被 cull 排除，不參與最小值
    h = 0.5 * math.hypot(d_ax, 2 * math.pi * R_BAR / n_th)
    return P, h


def sample_mesh_surface(T, spacing):
    """三角面表面取樣，保證任一表面點與最近取樣點距離 ≤ spacing。

    每個三角面以重心座標格點取樣，格距沿兩邊都 ≤ spacing。
    取樣只與手指幾何有關，**與掃描組數無關，只算一次**。
    """
    pts = []
    for A, B, C in T:
        n1 = max(int(np.ceil(np.linalg.norm(B - A) / spacing)), 1)
        n2 = max(int(np.ceil(np.linalg.norm(C - A) / spacing)), 1)
        n = max(n1, n2)
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        m = (i + j) <= n
        u = (i[m] / n)[:, None]
        v = (j[m] / n)[:, None]
        pts.append(A[None] + u * (B - A)[None] + v * (C - A)[None])
    return np.unique(np.round(np.concatenate(pts, axis=0), 9), axis=0)


def dist_points_to_capped_cylinder(P, t, axis):
    """點到**實心有限圓柱**的最短距離（解析，無號；點在內部為 0）。"""
    a = axis / np.linalg.norm(axis)
    d = P - t[None]
    s = d @ a
    rad = np.linalg.norm(d - s[:, None] * a[None], axis=1)
    dr = np.maximum(rad - R_BAR, 0.0)
    ds = np.maximum(np.abs(s) - BAR_LEN / 2.0, 0.0)
    return np.hypot(dr, ds)


def cull(P, T):
    """排除距手指 AABB 超過 CULL 的取樣點（排除者真實距離 > CULL）。"""
    lo = T.reshape(-1, 3).min(axis=0)
    hi = T.reshape(-1, 3).max(axis=0)
    d = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
    return P[d <= CULL]


def m_approx(t_y, a_y):
    return (GAP_HALF - R_BAR) - abs(t_y) - (L_APPROX / 2) * abs(a_y)


def rot(ax, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if ax == 'z':
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if ax == 'y':
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ---------------------------------------------------------------- 單元測試
def selftest() -> int:
    """可手算的最近點案例：面內、三邊、三頂點。"""
    A = np.array([0.0, 0.0, 0.0]); B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])
    cases = [
        ('面內',   [0.25, 0.25, 2.0],  [0.25, 0.25, 0.0], 2.0),
        ('邊 AB',  [0.5, -1.0, 0.0],   [0.5, 0.0, 0.0],   1.0),
        ('邊 AC',  [-1.0, 0.5, 0.0],   [0.0, 0.5, 0.0],   1.0),
        ('邊 BC',  [1.0, 1.0, 0.0],    [0.5, 0.5, 0.0],   math.sqrt(0.5)),
        ('頂點 A', [-1.0, -1.0, 0.0],  [0.0, 0.0, 0.0],   math.sqrt(2.0)),
        ('頂點 B', [2.0, -0.5, 0.0],   [1.0, 0.0, 0.0],   math.hypot(1, 0.5)),
        ('頂點 C', [-0.5, 2.0, 1.0],   [0.0, 1.0, 0.0],   math.sqrt(0.25 + 1 + 1)),
    ]
    P = np.array([c[1] for c in cases])
    Q = closest_point_triangle(P, A, B, C)
    d = np.linalg.norm(P - Q, axis=1)
    bad = 0
    for i, (name, _, q_exp, d_exp) in enumerate(cases):
        okq = np.allclose(Q[i], q_exp, atol=1e-12)
        okd = abs(d[i] - d_exp) < 1e-12
        bad += (not okq) or (not okd)
        print(f'  {name:7s} 最近點 {np.round(Q[i],6).tolist()} '
              f'（應為 {q_exp}）距離 {d[i]:.6f}（應為 {d_exp:.6f}）'
              f' {"ok" if okq and okd else "**錯**"}')
    # 解析圓柱距離：可手算案例
    t0 = np.array([0.0, 0.0, 0.0]); ax = np.array([1.0, 0.0, 0.0])
    cyl = [('側面外', [0.0, 0.010, 0.0], 0.010 - R_BAR),
           ('內部',   [0.0, 0.002, 0.0], 0.0),
           ('端面外', [0.150, 0.0, 0.0], 0.150 - BAR_LEN / 2),
           ('端角',   [0.150, 0.010, 0.0],
            math.hypot(0.150 - BAR_LEN / 2, 0.010 - R_BAR))]
    Pc = np.array([c[1] for c in cyl])
    dc = dist_points_to_capped_cylinder(Pc, t0, ax)
    for i, (nm, _, exp) in enumerate(cyl):
        ok = abs(dc[i] - exp) < 1e-12
        bad += not ok
        print(f'  圓柱 {nm:5s} 距離 {dc[i]:.6f}（應為 {exp:.6f}） '
              f'{"ok" if ok else "**錯**"}')
    # 批次版與單面版必須一致
    rng = np.random.default_rng(0)
    Pr = rng.normal(size=(200, 3))
    Tr = rng.normal(size=(7, 3, 3))
    ref = np.full(len(Pr), np.inf)
    for A_, B_, C_ in Tr:
        q = closest_point_triangle(Pr, A_, B_, C_)
        ref = np.minimum(ref, np.linalg.norm(Pr - q, axis=1))
    bat = _closest_batch(Pr, Tr[:, 0], Tr[:, 1], Tr[:, 2])
    dmax = float(np.abs(ref - bat).max())
    print(f'  批次版對單面版：最大差 {dmax:.3e} '
          f'{"ok" if dmax < 1e-12 else "**錯**"}')
    bad += dmax >= 1e-12
    # 退化三角面不得產生 NaN
    Z = closest_point_triangle(np.array([[0.0, 0.0, 1.0]]), A, A, A)
    if not np.isfinite(Z).all():
        print('  退化三角面產生非有限值 **錯**'); bad += 1
    print('單元測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


# ---------------------------------------------------------------- 掃描
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if selftest():
        print('**單元測試未過，停止**'); return 1

    T = fingers_tool_frame()
    SPACING = 0.0003
    S = sample_mesh_surface(T, SPACING)
    h = SPACING              # 取樣保證：任一表面點距最近取樣 ≤ spacing
    print(f'\n手指三角面 {len(T)}（finger1.stl ＋ finger2.stl，各自讀取）')
    print(f'手指表面取樣 {len(S)} 點，格距 ≤ {1000*SPACING:.2f} mm ⇒ Lipschitz 界 h = {1000*h:.2f} mm')

    # 交叉比對：三組用**另一條獨立路徑**（圓柱取樣 → 點到三角面）重算
    print('  交叉比對（兩條獨立路徑）：')
    for ty, tz, rdeg in ((0.0, 0.0, 0.0), (2.9, 0.0, 0.0), (2.0, -4.0, 3.0)):
        R = rot('z', rdeg)
        ax = R @ np.array([1.0, 0.0, 0.0])
        tt = np.array([0.0, ty / 1000.0, Z_BAR_NOM + tz / 1000.0])
        d_new = float(dist_points_to_capped_cylinder(S, tt, ax).min())
        P2, h2 = bar_samples(tt, ax)
        P2 = cull(P2, T)
        d_old = float(min_dist_points_mesh(P2, T).min())
        print(f'   t_y {ty:4.1f} t_z {tz:+5.1f} rot {rdeg:3.1f}°  '
              f'新路徑 {1000*d_new:7.3f} mm（h {1000*h:.2f}）  '
              f'舊路徑 {1000*d_old:7.3f} mm（h {1000*h2:.2f}）  '
              f'差 {1000*abs(d_new-d_old):.3f} mm')
    tys = [0.0, 2.9] if a.quick else [-2.9, -2.0, -1.0, 0.0, 1.0, 2.0, 2.9]
    tzs = [0.0, -4.0] if a.quick else [-4.0, -2.0, 0.0, 2.0, 4.0]
    txs = [0.0] if a.quick else [-5.0, 0.0, 5.0]
    rots = [((0, 0, 0))] if a.quick else [
        (0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0),
        (0, 0, 3), (0, 0, -3), (3, 3, 0), (3, 0, 3), (0, 3, 3),
        (3, 3, 3), (-3, -3, -3)]
    rows = []
    for ty in tys:
        for tz in tzs:
            for tx in txs:
                for rx, ry, rz in rots:
                    R = rot('x', rx) @ rot('y', ry) @ rot('z', rz)
                    axis = R @ np.array([1.0, 0.0, 0.0])
                    t = np.array([tx / 1000.0, ty / 1000.0,
                                  Z_BAR_NOM + tz / 1000.0])
                    num = float(dist_points_to_capped_cylinder(S, t, axis).min())
                    apx = m_approx(t[1], axis[1])
                    rows.append((ty, tz, tx, (rx, ry, rz), apx, num, h,
                                 apx - (num - h)))
    d = np.array([r[7] for r in rows])
    print(f'\n掃描 {len(rows)} 組；取樣 Lipschitz 界 h = {1000*h:.3f} mm')
    print('比較用保守下界 = m_num − h')
    print('  最樂觀（近似最接近下界）的 6 組：')
    for ty, tz, tx, r, apx, num, h, diff in sorted(rows, key=lambda r: -r[7])[:6]:
        print(f'   t_y {ty:5.1f} t_z {tz:+5.1f} t_x {tx:+5.1f} rot {r}  '
              f'近似 {1000*apx:7.3f}  數值 {1000*num:7.3f}  下界 {1000*(num-h):7.3f}  '
              f'差 {1000*diff:+7.3f}')
    print(f'\n近似 − 下界：最大 {1000*d.max():+.3f} mm、最小 {1000*d.min():+.3f} mm')
    print('全部為負 ⇒ 近似式在此範圍內保守；任一為正即該處樂觀，不得放行')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
