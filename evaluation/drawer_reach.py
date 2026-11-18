#!/usr/bin/env python3
"""抽屜任務線上選位（監督用；Howard 11/18：開門→開抽屜→取筆→返回）。座標＝里程（模擬內里程＝世界，同 cyl_reach_map.select 的作法）。
同一份 IK（cyl_reach_map.ik：閉式＋限位 −0.05＋σ≥0.02）與同一份全身淨空（clear_door_fast＋抽屜方盒（drawer_scene）；env／self 用執行門檻、底盤多留 CYL_BASE_MARGIN）。
  select_handle(b_now)          把手抓取底盤位：抓得到關閉時的把手，且底盤沿拉出方向平移 OPEN_M 後抓得到拉開後的把手（全身拉開的起終點）。
  select_observe(q_open, b_now) 俯視觀察：相機（手腕，光軸∥工具 z）正對拉開後露出的抽屜內部中心、高 OBS_H；同時要求該底盤能正上方夾露出區中心（找到筆後多半不必再移）。
  select_marker(c, yaw, q_open, b_now) 正上方夾筆底盤位：抓取位與預抓（上方 PRE_M）都要 IK＋淨空可行。
把手位置來自場景登錄（已知家具）＋理想里程；筆位置由手腕深度偵測（不讀登錄的筆位置）。"""
import math
import os

import numpy as np

import cyl_reach_map as M
import door_wbopt as WB
import door_clear_door_fast as CF
import drawer_scene as D

D.install(WB)
OPEN_M = float(D.SPEC['target_open_m']) if D.SPEC else 0.2
PRE_M = 0.08
OBS_H = float(os.environ.get('DR_OBS_H', '0.20'))                     # 觀察時 TCP 高於底板頂
GRASP_DEPTH = float(os.environ.get('PICK_GRASP_DEPTH', '0.004'))
CAM_OFF = np.array([0.0675, -0.0325])                                 # 相機光學框原點在 link_eef 的 x, y（URDF；光軸∥工具 z）
BASE_MARGIN = M.BASE_MARGIN
ENV_SEL = float(os.environ.get('DR_ENV_SEL', '0.03'))                 # 俯視位、預抓位的手臂淨空選位門檻（s41：俯視位只 11 mm ⇒ 接近中 link2 擦面板 4.8 mm 被停）
W_ENV = 2.0                                                           # 評分：每多 1 cm 淨空抵 2 cm 底盤距離


def _clear(b, q, q_open):
    D.set_open(q_open)
    c = CF.clear_door_fast(np.r_[b, q, math.radians(85.0)], M._T_WH0)
    ok = all(float(c[k]) >= WB.THR[k] + (BASE_MARGIN if k == 'base' else 0.0) for k in ('env', 'base', 'self'))
    return ok, c


def _ik_ok(T, b, q_open, env_min=0.0):
    for fl in (False, True):
        Tf = T(fl) if callable(T) else T
        g = M.ik(Tf, b)
        if g is None:
            continue
        ok, c = _clear(b, g[0], q_open)
        if ok and float(c['env']) >= env_min:
            return dict(q=g[0], sigma=float(g[2]), flip=fl, env=float(c['env']), base=float(c['base']))
    return None


def _yaw_face():
    p = D.PULL
    return math.atan2(-p[1], -p[0])                                   # 底盤面向櫃（與拉出方向相反）


def _frame_bases(c_xy, dists, lats, yaws):
    p = D.PULL[:2]; n = np.array([-p[1], p[0]])
    for d in dists:
        for la in lats:
            for dy in yaws:
                xy = np.asarray(c_xy, float)[:2] + d * p + la * n
                yield np.array([xy[0], xy[1], _yaw_face() + math.radians(dy)])


def select_handle(b_now, w_dist=1.0, w_sig=0.02):
    best = None
    h0 = D.handle_T(0.0)[:3, 3]
    for b in _frame_bases(h0, np.arange(0.40, 0.66, 0.025), np.arange(-0.20, 0.201, 0.05), (0.0,)):
        g0 = _ik_ok(lambda f: D.handle_T(0.0, f), b, 0.0)
        if g0 is None:
            continue
        b1 = b + np.r_[OPEN_M * D.PULL[:2], 0.0]
        g1 = _ik_ok(lambda f: D.handle_T(OPEN_M, f), b1, OPEN_M)
        if g1 is None:
            continue
        sc = w_dist * float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + w_sig / min(g0['sigma'], g1['sigma'])
        if best is None or sc < best['score']:
            best = dict(base=b.tolist(), base_end=b1.tolist(), flip=bool(g0['flip']), sigma_min=min(g0['sigma'], g1['sigma']), score=sc, env=min(g0['env'], g1['env']), base_clear=min(g0['base'], g1['base']),
                        T_grasp=D.handle_T(0.0, g0['flip']).tolist())
    D.set_open(0.0)
    return best


def grasp_T(c, yaw, flip, s=0.0):
    T = D.marker_T(c, yaw, flip)
    T[:3, 3] = T[:3, 3] + GRASP_DEPTH * T[:3, 2] - s * T[:3, 2]
    return T


def obs_T(c_xy, z_bottom, yaw_tool):
    """相機在 c_xy 正上方：工具 z 朝下、工具 x 朝 yaw_tool；TCP＝c − R·(cam 偏移)。"""
    T = D.marker_T(np.r_[c_xy[:2], z_bottom + OBS_H], yaw_tool + math.pi / 2)
    T[:3, 3] = T[:3, 3] - T[:3, :2] @ CAM_OFF
    return T


def select_marker(c, yaw, q_open, b_now, w_dist=1.0, w_sig=0.02):
    best = None
    for b in _frame_bases(c, np.arange(0.30, 0.56, 0.025), np.arange(-0.30, 0.301, 0.05), (0.0, -20.0, 20.0)):
        if not M.base_free(b):
            continue
        g = _ik_ok(lambda f: grasp_T(c, yaw, f), b, q_open)
        if g is None:
            continue
        gp = _ik_ok(grasp_T(c, yaw, g['flip'], PRE_M), b, q_open, ENV_SEL)
        if gp is None:
            continue
        sc = w_dist * float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + w_sig / min(g['sigma'], gp['sigma']) - W_ENV * gp['env']
        if best is None or sc < best['score']:
            best = dict(base=b.tolist(), flip=bool(g['flip']), sigma_min=min(g['sigma'], gp['sigma']), score=sc, env=min(g['env'], gp['env']), base_clear=min(g['base'], gp['base']), T_grasp=grasp_T(c, yaw, g['flip']).tolist())
    return best


def exposed_center(q_open):
    it = D.SPEC['interior']
    y_lo = float(it['y'][0])
    y_hi = min(float(it['y'][1]), -0.150 + float(q_open) + y_lo + 0.150)   # 露出區：抽屜本地 y ∈ [y_lo, y_lo＋q_open]（櫃前緣 −0.150 以外）
    return D.to_world([0.0, (y_lo + y_lo + float(q_open)) / 2.0, float(it['bottom_top_z'])], q_open, True)


def select_observe(q_open, b_now):
    c = exposed_center(q_open)
    z_b = float(D.SPEC['interior']['bottom_top_z'])
    yaw_m = math.atan2(D.PULL[0], -D.PULL[1]) if False else _yaw_face() + math.pi / 2   # 筆名目平躺方向（沿把手橫桿）只用來定夾爪開合方向，不讀筆位置
    best = None
    for b in _frame_bases(c, np.arange(0.30, 0.56, 0.025), np.arange(-0.20, 0.201, 0.05), (0.0, -20.0, 20.0)):
        if not M.base_free(b):
            continue
        To = obs_T(c, z_b, _yaw_face())
        go = _ik_ok(To, b, q_open, ENV_SEL)
        if go is None:
            continue
        cg = np.r_[c[:2], z_b + 0.014]
        g = _ik_ok(lambda f: grasp_T(cg, yaw_m, f), b, q_open)
        if g is None:
            continue
        gp = _ik_ok(grasp_T(cg, yaw_m, g['flip'], PRE_M), b, q_open, ENV_SEL)
        if gp is None:
            continue
        sc = float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + 0.02 / min(go['sigma'], g['sigma']) - W_ENV * min(go['env'], gp['env'])
        if best is None or sc < best['score']:
            best = dict(base=b.tolist(), sigma_min=min(go['sigma'], g['sigma']), score=sc, env=min(go['env'], g['env']), base_clear=min(go['base'], g['base']), T_obs=To.tolist(), c=c.tolist())
    return best


if __name__ == '__main__':
    import time
    w = time.perf_counter(); h = select_handle(np.array([0.0, 9.0, math.pi]))
    print('handle', None if h is None else dict(base=np.round(h['base'], 3).tolist(), end=np.round(h['base_end'], 3).tolist(), sig=round(h['sigma_min'], 3), env=round(h['env'], 3), bc=round(h['base_clear'], 3)), round((time.perf_counter() - w) * 1e3), 'ms')
    w = time.perf_counter(); o = select_observe(OPEN_M, np.array(h['base_end']) if h else np.array([-3.9, 10.3, math.pi]))
    print('observe', None if o is None else dict(base=np.round(o['base'], 3).tolist(), sig=round(o['sigma_min'], 3), env=round(o['env'], 3), bc=round(o['base_clear'], 3), c=np.round(o['c'], 3).tolist()), round((time.perf_counter() - w) * 1e3), 'ms')
    mk = D.marker_nominal(OPEN_M)
    w = time.perf_counter(); m = select_marker(mk, _yaw_face() + math.pi / 2, OPEN_M, np.array(o['base']) if o else np.array([-4.2, 10.3, math.pi]))
    print('marker', mk.round(3), None if m is None else dict(base=np.round(m['base'], 3).tolist(), sig=round(m['sigma_min'], 3), env=round(m['env'], 3), bc=round(m['base_clear'], 3)), round((time.perf_counter() - w) * 1e3), 'ms')
