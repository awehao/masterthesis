#!/usr/bin/env python3
"""車尾收納盒（Howard 2026-11-19「在車子後面放一個盒子 放夾取物 並可以同時做茶几抽屜任務」；Claude 自行裁定，Codex 用量用完期間）。
資產：src/my_omnibot_description/config/rear_bin_v1.yaml（底盤座標方盒，隨底盤移動）。
  box_world(b)        盒外形（世界）＝(中心, R, 半尺寸)；b＝底盤 (x, y, θ)。
  bin_dist(qf)        手臂所有連桿點雲（含手指）到盒外形的最小符號距離（盒當實心方盒；不含盒內物體）。
  install(CF)         包住 clear_door_fast／clear_door_fast_multi：'self'＝min(原自碰撞, bin_dist)（盒是機器人本體的一部分，走自碰撞門檻）；
                      未安裝＝逐值不變。who['bin'] 另記最近連桿。
  place_T()           放置 TCP（底盤座標）：盒中心正上方 tcp_z、夾爪朝下、工具 x 沿底盤 y（物體躺下沿盒長邊）。
  q_place()           放置構型：固定種子的數值 IK（關節限位內縮 0.05），取 min(自身, 離盒) 最大者；結果與底盤位姿無關（底盤座標），只算一次。
不宣稱：物體放入盒內靠物理落下（解除理想連接＋張爪），沒有盒內定位保證；淨空不含盒內物體與持物本體。"""
import math
import os

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, '..', 'src', 'my_omnibot_description', 'config', os.environ.get('D2_BIN_SPEC', 'rear_bin_v1.yaml'))
SPEC = yaml.safe_load(open(SPEC_PATH))
C_LOC = np.array(SPEC['outer']['center'], float)
HS = np.array(SPEC['outer']['half'], float)
PLACE = SPEC['place']
DROP_WAIT_S = float(PLACE.get('drop_wait_s', 0.6))
_ORIG = {}
_QP = None


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_world(b):
    R = _Rz(float(b[2]))
    return R @ C_LOC + np.array([float(b[0]), float(b[1]), 0.0]), R, HS


def bin_dist(qf, T=None):
    """T：可傳入已算好的連桿世界變換 dict（含手指平移）；否則用 K.fk（手指閉合位）。"""
    import door_wbopt as WB
    DOC, K = WB.DOC, WB.K
    c, R, hs = box_world(qf[:3])
    m, who = np.inf, None
    for n in DOC.ARM_LINKS:
        Tn = T[n] if (T is not None and n in T) else K.fk(np.asarray(qf, float), n)
        P = (Tn[:3, :3] @ DOC.CL[n].T).T + Tn[:3, 3]
        d = float(DOC.box_dist(P, c, R, hs).min())
        if d < m:
            m, who = d, n
    return m, who


def install(CF):
    """包住 CF.clear_door_fast／clear_door_fast_multi（模組屬性；呼叫端以 CF.xxx 取用 ⇒ 生效）。重複安裝無作用。"""
    if 'single' in _ORIG:
        return
    _ORIG['single'], _ORIG['multi'] = CF.clear_door_fast, CF.clear_door_fast_multi

    def single(x, T_WH0):
        r = _ORIG['single'](x, T_WH0)
        d, who = bin_dist(np.asarray(x[:9], float))
        r['who'] = dict(r.get('who') or {}, bin=who)
        r['self'] = min(float(r['self']), d)
        return r

    def multi(x, T_WH0, fqs):
        r = _ORIG['multi'](x, T_WH0, fqs)
        d, who = bin_dist(np.asarray(x[:9], float))
        r['who'] = dict(r.get('who') or {}, bin=who)
        r['self'] = min(float(r['self']), d)
        return r
    CF.clear_door_fast, CF.clear_door_fast_multi = single, multi


def place_T(dz=0.0):
    """底盤座標放置位姿：工具 z 朝下、工具 x ∥ 底盤 −y（物體軸沿盒長邊）。"""
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]) @ _Rz(math.pi / 2)
    T[:3, 3] = [float(PLACE['tcp_xy'][0]), float(PLACE['tcp_xy'][1]), float(PLACE['tcp_z']) + dz]
    return T


def _self_and_bin(q):
    import door_clear_fast as FAST
    import door_clear_door_fast as CF0
    DOC = FAST.DOC
    qf = np.r_[0.0, 0.0, 0.0, np.asarray(q, float)]
    pairs = FAST._pairs()
    names = sorted(set(DOC.ARM_LINKS) | {n for p in pairs for n in p})
    T = CF0.fk_all(qf, names)
    W_ = {}

    def pts(n):
        if n not in W_:
            W_[n] = (T[n][:3, :3] @ DOC.CL[n].T).T + T[n][:3, 3]
        return W_[n]
    return float(CF0._self_clear(T, pts)), bin_dist(qf, T)[0]


SEEDS = [np.array(s, float) for s in ([-2.65, 0.2, 0.45, 1.6, -0.95, -1.2], [-2.73, 0.2, 0.44, 1.14, -0.97, -0.84], [2.6, 0.2, 0.5, -1.6, -0.95, 1.2],
                                       [-2.86, 0.08, 0.58, 1.52, -1.01, 0.05], [-2.7, 0.04, 0.62, 1.68, -0.92, 0.05], [2.7, 0.1, 0.6, -1.6, -0.95, -1.5],
                                       [-2.5, 0.4, 0.8, 1.5, -1.0, 0.0], [2.5, 0.4, 0.8, -1.5, -1.0, 0.0])]


def q_place(dz=0.0):
    global _QP
    if _QP is not None and dz == 0.0:
        return _QP
    import door_wbopt as WB
    K, LIM = WB.K, WB.DA.LITE6_SAFE
    lo, hi = np.asarray(LIM.lower, float) + 0.05, np.asarray(LIM.upper, float) - 0.05
    T = place_T(dz)

    def res(q):
        Tc = K.fk(np.r_[0.0, 0.0, 0.0, q], 'link_tcp')
        Re = T[:3, :3].T @ Tc[:3, :3]
        return np.r_[(Tc[:3, 3] - T[:3, 3]) * 10.0, 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * 2.0]
    best = None
    for sd in SEEDS:
        r = least_squares(res, np.clip(sd, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=400)
        if float(np.linalg.norm(r.fun)) > 1e-5:
            continue
        sc, bd = _self_and_bin(r.x)
        if best is None or min(sc, bd) > best[1]:
            best = (np.asarray(r.x, float), min(sc, bd), sc, bd)
    if best is None:
        return None
    if dz == 0.0:
        _QP = best
    return best


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    b = q_place()
    print('q_place', None if b is None else (np.round(b[0], 4).tolist(), 'self', round(b[2], 4), 'bin', round(b[3], 4)))
