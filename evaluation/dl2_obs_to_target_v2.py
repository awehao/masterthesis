#!/usr/bin/env python3
"""【PC2 新版】dl2_obs_to_target.py（sha256 71d5d3e5ef3e4a03225705ae91c7675ef437ad13f75362a5ac597511e535126c，凍結，不改）的複本，唯一差異：匯入 d1_handle_detect_v2（像素中心反投影）。

原說明：DL2：視覺觀測 → 物體局部座標目標候選（估計端；規格 results/vision/DL2_obs_to_target_spec.md draft-2）。

定位：檢查既有把手觀測結合明示抓取與方向先驗，能否生成一致的物體局部座標目標候選；
尚未驗收碰撞、可達性、新鮮度或閉迴路執行。輸出一律 geometry_contract: unchecked，不可直接執行。

**估計端不讀真值**：`estimate()` 只接受白名單鍵（見 OBS_KEYS），其他鍵一律拒絕。真值只在評估端（dl2_eval.py）。
先驗（逐筆記於 prior_used）：
* design_grasp_reference —— 軸號以設計抓取的工具旋轉為參考選定（橫桿對稱，觀測無法決定號）
* gravity_vertical_prior —— N-obs 的水平法向篩選（|n_z| ≤ 0.2）
* vertical_front_prior   —— N-prior：前板為鉛直面，n ∝ axis × ẑ
* camera_side_outward    —— 外側＝朝相機側
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect_v2 as D1                    # noqa: E402  PC2：像素中心契約
import object_target_geometry as OT                 # noqa: E402

OBS_KEYS = {'path', 'L1', 'L2', 'reject', 'reject_L2', 'depth', 'K', 'T_cam', 't_obs'}
# DL2 新登錄參數（draft-2 §3；登錄後凍結，不調）
NEIGH_R = 0.15        # 鄰域半徑（到觀測中心）
BAR_EXCL = 0.025      # 橫桿排除半徑（到觀測軸線段）
BAR_HALF = 0.11       # 觀測軸線段半長（觀測中心 ± 0.11 m）
# 沿用 D1 既有值
NZ_MAX = D1.P['nz_max']                  # 0.2
BAND = D1.BAND                           # (0.020, 0.060)
PLANE_MIN = D1.P['plane_min']            # 400
MAX_PLANES = D1.P['max_planes']          # 6
S_RETREAT = 0.03
SIDE_COS_MIN = 1e-6   # |n·u_cam| 小於此值 ⇒ 朝相機側無法判定（相機在平面上）


class Reject(Exception):
    def __init__(self, why):
        super().__init__(why)
        self.why = why


def baseline_grasp(R_grasp, grasp_depth=0.0068):
    """基準抓取參數（DL1 資產 prim 約定）：T_HG = [R_grasp | (0, +grasp_depth, 0)]、a_H = (0, −1, 0)。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_grasp, float)
    T[:3, 3] = [0.0, grasp_depth, 0.0]
    return T, np.array([0.0, -1.0, 0.0])


# ------------------------------------------------------------------ 法向
def _orient_toward_camera(n, d, c, cam_pos):
    u = np.asarray(cam_pos, float) - c
    nu = float(np.linalg.norm(u))
    if nu < 1e-9 or abs(float(n @ u)) / nu < SIDE_COS_MIN:
        raise Reject('outward_side_undetermined')
    return (n, d) if n @ u > 0 else (-n, -d)


def normal_from_points(X, center, axis, cam_pos, seed=0):
    """N-obs 核心（世界點已體素化）：回傳 (外側法向 n, 診斷)；失敗拋 Reject。"""
    X = np.asarray(X, float)
    c = np.asarray(center, float)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    X = X[np.linalg.norm(X - c, axis=1) <= NEIGH_R]
    # 扣除橫桿：到觀測軸線段（c ± BAR_HALF·ax）的距離 ≤ BAR_EXCL
    t = np.clip((X - c) @ ax, -BAR_HALF, BAR_HALF)
    X = X[np.linalg.norm(X - (c + t[:, None] * ax), axis=1) > BAR_EXCL]
    diag = {'n_points': int(len(X))}
    if len(X) < PLANE_MIN:
        raise Reject('normal_unobservable:few_points')
    rng = np.random.default_rng(seed)
    rest = np.arange(len(X))
    qual = []
    for _ in range(MAX_PLANES):
        if len(rest) < PLANE_MIN:
            break
        pl, inl = D1.ransac_plane(X[rest], rng)
        if pl is None or inl.sum() < PLANE_MIN:
            break
        n, d = _orient_toward_camera(np.asarray(pl[0], float), float(pl[1]), c, cam_pos)   # 定向為朝相機側
        sd = float(n @ c + d)                                # 觀測中心到已定向平面的有號距離
        if abs(n[2]) <= NZ_MAX and BAND[0] <= sd <= BAND[1]:
            qual.append({'n': n, 'signed_dist_m': sd, 'n_inl': int(inl.sum())})
        rest = rest[~inl]
    diag['n_qualified'] = len(qual)
    if not qual:
        raise Reject('normal_unobservable')
    if len(qual) >= 2:
        raise Reject('normal_ambiguous')
    diag['signed_dist_m'] = qual[0]['signed_dist_m']
    return qual[0]['n'], diag


def _check_depth_K(depth, K):
    try:
        dep = np.asarray(depth, float)
        Km = np.asarray(K, float)
    except (TypeError, ValueError):
        raise Reject('depth_or_K_not_numeric')
    if dep.ndim != 2 or min(dep.shape) < 2:
        raise Reject('depth_shape')
    if Km.shape != (3, 3) or not np.isfinite(Km).all():
        raise Reject('K_invalid')
    if Km[0, 0] <= 0 or Km[1, 1] <= 0:
        raise Reject('K_focal_nonpositive')
    return dep, Km


def normal_obs(depth, K, T_cam, center, axis):
    depth, K = _check_depth_K(depth, K)
    X = D1.backproject(np.asarray(depth, float), np.asarray(K, float), np.asarray(T_cam, float), D1.P['stride'])
    if len(X):
        _, keep = np.unique(np.floor(X / D1.P['voxel']).astype(np.int64), axis=0, return_index=True)
        X = X[np.sort(keep)]
    return normal_from_points(X, center, axis, np.asarray(T_cam, float)[:3, 3])


def normal_prior(axis, center, cam_pos):
    n = np.cross(np.asarray(axis, float), [0.0, 0.0, 1.0])
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        raise Reject('normal_prior_axis_vertical')
    n /= nn
    n, _ = _orient_toward_camera(n, 0.0, np.asarray(center, float), cam_pos)
    return n, {}


# ------------------------------------------------------------------ 估計
def _finite_time(x, name):
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise Reject(f'{name}_not_numeric')
    if not math.isfinite(x):
        raise Reject(f'{name}_nonfinite')
    return x


def estimate(obs, normal_mode, R_ref_tool, T_HG, a_H, query_t, max_age_s=None):
    """觀測 → 把手位姿 → 目標候選（s = 0、0.03）。回傳 dict（ok、why、prior_used、freshness …）。"""
    out = {'ok': False, 'geometry_contract': 'unchecked', 'prior_used': [], 'normal_mode': normal_mode}
    try:
        if not isinstance(obs, dict):
            raise Reject('obs_not_dict')
        extra = sorted(set(obs) - OBS_KEYS)
        if extra:
            raise Reject('unexpected_input:' + ','.join(map(str, extra)))
        if normal_mode not in ('N-obs', 'N-prior'):
            raise Reject('normal_mode_invalid')
        # 新鮮度：只驗介面，不訂操作門檻
        out['time_raw'] = {'t_obs': repr(obs.get('t_obs')), 'query_t': repr(query_t)}   # 缺失或非法照實記錄，不補造
        t_obs = _finite_time(obs.get('t_obs'), 't_obs')
        q_t = _finite_time(query_t, 'query_t')
        age = q_t - t_obs
        out.update({'t_obs': t_obs, 'query_t': q_t, 'age_s': age})        # 時間檢查後即保存（之後的拒絕列也保留）
        if age < 0:
            raise Reject('observation_from_future')
        if max_age_s is None:
            out['freshness'] = 'not_accepted'
        else:
            m = _finite_time(max_age_s, 'max_age_s')
            if age > m:
                raise Reject('observation_stale')
            out['freshness'] = f'age {age:.3f} s ≤ max_age_s {m}（介面核對，非操作門檻）'
        if obs.get('reject'):
            raise Reject('detector:' + str(obs['reject']))
        if not obs.get('L1'):
            raise Reject('axis_missing')
        if not obs.get('L2'):
            raise Reject('center_missing' + (':' + str(obs['reject_L2']) if obs.get('reject_L2') else ''))
        try:
            c = np.asarray(obs['L2']['center'], float)
            ax = np.asarray(obs['L1']['axis'], float)
        except (TypeError, ValueError, KeyError, IndexError):
            raise Reject('obs_geometry_invalid')
        if c.shape != (3,) or ax.shape != (3,) or not (np.isfinite(c).all() and np.isfinite(ax).all()):
            raise Reject('obs_geometry_invalid')
        if float(np.linalg.norm(ax)) < 1e-9:
            raise Reject('axis_zero')
        T_cam = OT.rigid(obs.get('T_cam'), 'T_cam')
        cam = T_cam[:3, 3]
        if normal_mode == 'N-obs':
            if obs.get('depth') is None or obs.get('K') is None:
                raise Reject('depth_missing')
            n, nd = normal_obs(obs['depth'], obs['K'], T_cam, c, ax)
            out['prior_used'] += ['gravity_vertical_prior', 'camera_side_outward']
        else:
            n, nd = normal_prior(ax, c, cam)
            out['prior_used'] += ['vertical_front_prior', 'camera_side_outward']
        out['normal_diag'] = nd
        out['normal_W'] = n.tolist()
        hp = OT.handle_pose_from_partial(c, ax, n, return_candidates=True)
        if not hp['ok']:
            raise Reject('pose:' + hp['why'])
        cands = hp['candidates']
        tool0 = []
        for Th in cands:
            r = OT.object_target(Th, T_HG, a_H, 0.0)
            if not r['ok']:
                raise Reject('target:' + r['why'])
            tool0.append(r['T_WE'])
        ch = OT.choose_candidate(tool0, R_ref_tool)          # 工具目標旋轉 vs 工具參考旋轉
        if not ch['ok']:
            raise Reject('choose:' + ch['why'])
        i = ch['index']
        out['prior_used'].append('design_grasp_reference')
        rs = OT.object_target(cands[i], T_HG, a_H, S_RETREAT)
        if not rs['ok']:
            raise Reject('target:' + rs['why'])
        out.update({'ok': True, 'T_WH': cands[i], 'candidate_index': i, 'candidate_distances': ch['distances'],
                    'T_WE_s0': tool0[i], 'T_WE_s': rs['T_WE'], 's_m': S_RETREAT})
        return out
    except (Reject, OT.Reject) as r:
        out['why'] = r.why
        return out
