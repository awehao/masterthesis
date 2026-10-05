#!/usr/bin/env python3
"""DL1：物體局部座標的工具目標生成（純函式；第二階段計畫 §4.4、§9.13.3；規格 results/vision/DL1_object_target_spec.md）。

記號：`T_AB` 把 **B 座標表示的點轉到 A**（p_A = T_AB · p_B）。
抓取約束 T_WE·T_EG = T_WH·T_HG ⇒ 工具目標

    T_WE_pre(s) = T_WH · Trans(s · a_H) · T_HG · T_EG⁻¹

* `a_H`（把手座標）為接近／退讓軸，s ≥ 0 為退讓距離（公尺）。**本檔不含任何世界軸常數或固定旋轉。**
* 輸出一律附 `geometry_contract: 'unchecked'`：夾爪相容與碰撞模型一起轉換尚未核對，不能據此放行控制或宣稱可達。
* 任何不合法輸入回傳拒絕（ok=False、why），不默默修正。
"""
from __future__ import annotations

import math

import numpy as np

T_MAX_M = 2.0          # 抓取／工具轉換的平移上限（公尺）——防 mm 數值當 m（12.26、300 會被攔下）
S_MAX_M = 0.30         # 退讓距離上限（公尺）
AXIS_SIGN_COS = math.cos(math.radians(60.0))     # 軸向參考與估計軸夾角超過 60° ⇒ 號無法消歧
PARALLEL_COS = math.cos(math.radians(15.0))      # 法向與橫桿軸夾角小於 15° ⇒ 無法定滾轉
TIE_RAD = 1e-9


class Reject(Exception):
    def __init__(self, why):
        super().__init__(why)
        self.why = why


# ------------------------------------------------------------------ 驗證
def rigid(T, name, t_max=None):
    try:
        T = np.asarray(T, float)
    except (TypeError, ValueError):
        raise Reject(f'{name}_not_numeric')
    if T.shape != (4, 4):
        raise Reject(f'{name}_shape_{T.shape}')
    if not np.isfinite(T).all():
        raise Reject(f'{name}_nonfinite')
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise Reject(f'{name}_bottom_row')
    R = T[:3, :3]
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-6 or np.abs(R.T @ R - np.eye(3)).max() > 1e-6:
        raise Reject(f'{name}_not_SO3')
    if t_max is not None and float(np.linalg.norm(T[:3, 3])) > t_max:
        raise Reject(f'{name}_translation_exceeds_{t_max}m（疑似單位誤用）')
    return T


def unit(v, name):
    try:
        v = np.asarray(v, float).reshape(3)
    except (TypeError, ValueError):
        raise Reject(f'{name}_not_3vector')
    if not np.isfinite(v).all():
        raise Reject(f'{name}_nonfinite')
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise Reject(f'{name}_zero')
    return v / n


def scalar(x, name, lo, hi):
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise Reject(f'{name}_not_numeric')
    if not math.isfinite(x):
        raise Reject(f'{name}_nonfinite')
    if not (lo <= x <= hi):
        raise Reject(f'{name}_out_of_range_[{lo},{hi}]（疑似單位誤用或越界）')
    return x


def trans(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


# ------------------------------------------------------------------ 目標
def object_target(T_WH, T_HG, a_H, s, T_EG=None, outward_H=None, object_id=None, locked_object_id=None):
    """回傳 dict：ok、T_WE（4×4 list）或 why；diag 含檢查項。"""
    diag = {'geometry_contract': 'unchecked', 'formula': 'T_WH·Trans(s·a_H)·T_HG·T_EG⁻¹'}
    try:
        if locked_object_id is not None and object_id != locked_object_id:
            raise Reject('object_identity_changed')
        T_WH = rigid(T_WH, 'T_WH')
        T_HG = rigid(T_HG, 'T_HG', T_MAX_M)
        T_EG = np.eye(4) if T_EG is None else rigid(T_EG, 'T_EG', T_MAX_M)
        a = unit(a_H, 'a_H')
        s = scalar(s, 's', 0.0, S_MAX_M)
        if outward_H is not None:
            o = unit(outward_H, 'outward_H')
            if float(a @ o) <= 0.0:
                raise Reject('retreat_not_outward')
            diag['retreat_side'] = 'checked_outward'
        else:
            diag['retreat_side'] = 'retreat_side_unchecked'
        T_WE = T_WH @ trans(s * a) @ T_HG @ np.linalg.inv(T_EG)
        return {'ok': True, 'T_WE': T_WE, 'diag': diag}
    except Reject as r:
        return {'ok': False, 'why': r.why, 'diag': diag}


# ------------------------------------------------------------------ 機構（滑軌）
def opened_handle_pose(T_WH0, u_H, q, q_min, q_max):
    """滑軌：T_WH(q) = T_WH0 · Trans(q·u_H)。T_WH0＝零開度把手位姿；q 公尺、須在 [q_min, q_max]。"""
    try:
        T_WH0 = rigid(T_WH0, 'T_WH0')
        u = unit(u_H, 'u_H')
        qmin = scalar(q_min, 'q_min', -10.0, 10.0)
        qmax = scalar(q_max, 'q_max', -10.0, 10.0)
        if qmax < qmin:
            raise Reject('q_range_inverted')
        q = scalar(q, 'q', qmin, qmax)
        return {'ok': True, 'T_WH': T_WH0 @ trans(q * u)}
    except Reject as r:
        return {'ok': False, 'why': r.why}


# ------------------------------------------------------------------ 部分觀測
def handle_pose_from_partial(center_W, axis_W, normal_W=None, axis_ref_W=None, center_is_center=True,
                             return_candidates=False):
    """只有中心＋橫桿軸時組把手姿態：x＝橫桿軸、y＝前板法向（指向外側）、z＝x×y。

    * 法向只補繞橫桿軸的滾轉，**不決定橫桿軸號**；號由 `axis_ref_W` 決定（夾角 > 60° ⇒ 歧義）。
    * 歧義時：return_candidates=True ⇒ 回兩個等價候選（不選號）；否則拒絕。
    """
    try:
        if not center_is_center:
            raise Reject('point_not_center')
        c = np.asarray(center_W, float).reshape(3)
        if not np.isfinite(c).all():
            raise Reject('center_nonfinite')
        ax = unit(axis_W, 'axis_W')
        if normal_W is None:
            raise Reject('underdetermined_roll')
        n = unit(normal_W, 'normal_W')
        if abs(float(ax @ n)) > PARALLEL_COS:
            raise Reject('normal_parallel_axis')

        def build(x):
            y = n - (n @ x) * x
            y /= np.linalg.norm(y)
            z = np.cross(x, y)
            T = np.eye(4)
            T[:3, :3] = np.column_stack([x, y, z])
            T[:3, 3] = c
            return T
        if axis_ref_W is None:
            sign_ok = False
        else:
            r = unit(axis_ref_W, 'axis_ref_W')
            sign_ok = abs(float(ax @ r)) >= AXIS_SIGN_COS
            if sign_ok and float(ax @ r) < 0:
                ax = -ax
        if not sign_ok:
            if return_candidates:
                return {'ok': True, 'candidates': [build(ax), build(-ax)], 'prior_used': ['normal_W'],
                        'note': 'axis_sign_ambiguous：兩個等價候選，未選號'}
            raise Reject('axis_sign_ambiguous')
        return {'ok': True, 'T_WH': build(ax), 'prior_used': ['normal_W', 'axis_ref_W']}
    except Reject as r:
        return {'ok': False, 'why': r.why}


# ------------------------------------------------------------------ 對稱候選
def _rot_about(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def geodesic(Ra, Rb):
    """SO(3) 測地距離（rad）。用 atan2(sin, cos)：acos 在角度近 0 時精度只有約 √ε（實測 3.7e-8），會讓同分門檻失效。"""
    R = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    c = (np.trace(R) - 1.0) / 2.0
    s = 0.5 * float(np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))
    return math.atan2(s, c)


def symmetric_equivalents(T_HG, sym_axis_H, order):
    """依給定對稱模型（繞把手座標 sym_axis_H 的 order 重旋轉對稱）產生**幾何等價候選**（不是已驗證可抓取）。"""
    T_HG = rigid(T_HG, 'T_HG', T_MAX_M)
    ax = unit(sym_axis_H, 'sym_axis_H')
    if int(order) != order or order < 1:
        raise Reject('order_invalid')
    out = []
    for k in range(int(order)):
        S = np.eye(4)
        S[:3, :3] = _rot_about(ax, 2 * math.pi * k / order)
        out.append(S @ T_HG)
    return out


def choose_candidate(cands_WE, R_ref):
    """與參考旋轉測地距離最小者；同分（差 ≤ 1e-9 rad）取索引最小者。回傳 (index, distances)。"""
    d = [geodesic(np.asarray(R_ref, float), np.asarray(T, float)[:3, :3]) for T in cands_WE]
    best = 0
    for i in range(1, len(d)):
        if d[i] < d[best] - TIE_RAD:
            best = i
    return best, d
