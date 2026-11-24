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
PRE_TIERS = [float(x) for x in os.environ.get('DR_PRE_TIERS', '0.08,0.12').split(',') if x]   # 取筆預抓高度（依序試）
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


def _ik_ok(T, b, q_open, env_min=0.0, flips=(False, True)):
    for fl in flips:
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
                        T_grasp=D.handle_T(0.0, g0['flip']).tolist(), q_grasp=[float(x) for x in g0['q']])
    D.set_open(0.0)
    return best


def grasp_T(c, yaw, flip, s=0.0):
    T = D.marker_T(c, yaw, flip)
    T[:3, 3] = T[:3, 3] + GRASP_DEPTH * T[:3, 2] - s * T[:3, 2]
    return T


def _rotvec(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def obs_R(flip):
    """俯視姿態：自抓把手姿態（退出保持）做最短轉動使工具 z 朝下（s42：任意定的繞垂直軸角 ⇒ 途中 j4 轉到 −2.886 碰限位）。"""
    R0 = D.handle_T(0.0, flip)[:3, :3]
    z0, zt = R0[:, 2], np.array([0.0, 0.0, -1.0])
    ax = np.cross(z0, zt); s_ = float(np.linalg.norm(ax))
    ang = math.atan2(s_, float(z0 @ zt))
    return _rotvec(ax / max(s_, 1e-12) * ang) @ R0


def obs_T(c_xy, z_bottom, R):
    """相機在 c_xy 正上方（光軸∥工具 z 朝下）：TCP＝c − R·(cam 偏移)。"""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.r_[c_xy[:2], z_bottom + OBS_H]
    T[:3, 3] = T[:3, 3] - R[:, :2] @ CAM_OFF
    return T


def _rot_angle(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0)))


PATH_ENV = float(os.environ.get('DR_PATH_ENV', '0.008'))             # 俯視位→預抓位接近路徑的手臂淨空下限（s88d：移動中拉開後俯視底盤轉 22°，接近時夾爪殼擦抽屜側板 3.1 mm）


def _path_ok(T_from, T_to, b, q_open, q_seed, n=5):
    """同一底盤位、TCP 由 T_from 直線（姿態等角內插）到 T_to：逐點 IK（同分支連續）＋手臂淨空 ≥ PATH_ENV。"""
    A, Bm = np.asarray(T_from, float), np.asarray(T_to, float)
    from scipy.spatial.transform import Rotation as Rr, Slerp
    sl = Slerp([0.0, 1.0], Rr.from_matrix([A[:3, :3], Bm[:3, :3]]))
    prev = np.asarray(q_seed, float)
    for u in np.linspace(0.0, 1.0, n)[1:-1]:
        T = np.eye(4); T[:3, :3] = sl([u]).as_matrix()[0]; T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]
        q, r, sm = DFK.ik_closure_fc(T, b, prev)
        q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi
        if r > 1e-4 or sm < 0.02:
            return False
        ok, c = _clear(b, q, q_open)
        if not ok or float(c['env']) < PATH_ENV:
            return False
        prev = q
    return True


YAW_TIERS = ((0.0,), (-20.0, 20.0))                                    # s89d：MotM 拉開後底盤橫向漂 0.1 m，±20° 底盤位夾筆擦側板 3.4 mm ⇒ 先限正對櫃面，無解才放 ±20°


def select_marker(c, yaw, q_open, b_now, w_dist=1.0, w_sig=0.02, R_ref=None, T_from=None):
    for pre in PRE_TIERS:                                              # Isaac 98：筆滑近抽屜前板 9 mm ⇒ 預抓 0.08 m 時手臂淨空最多 28 mm（門檻 30）全不可行；改從較高預抓（0.12 m）下降淨空 38.5 mm
        for yt in YAW_TIERS:
            r = _select_marker(c, yaw, q_open, b_now, w_dist, w_sig, R_ref, T_from, yt, pre)
            if r is not None:
                r['pre_m'] = float(pre)
                return r
    return None


def _select_marker(c, yaw, q_open, b_now, w_dist, w_sig, R_ref, T_from, yaws, pre=PRE_M):
    best = None
    flips = (False, True)
    if R_ref is not None:                                              # 先試與俯視姿態轉角較小的夾爪方向（少轉手腕）
        flips = tuple(sorted((False, True), key=lambda f: _rot_angle(np.asarray(R_ref, float), grasp_T(c, yaw, f)[:3, :3])))
    for b in _frame_bases(c, np.arange(0.30, 0.56, 0.025), np.arange(-0.30, 0.301, 0.05), yaws):
        if not M.base_free(b):
            continue
        g = _ik_ok(lambda f: grasp_T(c, yaw, f), b, q_open, flips=flips[:1])
        if g is None:
            continue
        gp = _ik_ok(grasp_T(c, yaw, g['flip'], pre), b, q_open, ENV_SEL)
        if gp is None:
            continue
        if T_from is not None and not _path_ok(T_from, grasp_T(c, yaw, g['flip'], pre), b, q_open, gp['q']):
            continue
        dyaw = abs((float(b[2]) - float(b_now[2]) + math.pi) % (2 * math.pi) - math.pi)
        sc = w_dist * (float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + 0.3 * dyaw) + w_sig / min(g['sigma'], gp['sigma']) - W_ENV * gp['env']   # 偏航差折 0.3 m/rad（少轉底盤、手腕分支連續）
        if best is None or sc < best['score']:
            best = dict(base=b.tolist(), flip=bool(g['flip']), sigma_min=min(g['sigma'], gp['sigma']), score=sc, env=min(g['env'], gp['env']), base_clear=min(g['base'], gp['base']), T_grasp=grasp_T(c, yaw, g['flip']).tolist())
    return best


def exposed_center(q_open):
    it = D.SPEC['interior']
    y_lo = float(it['y'][0])
    y_hi = min(float(it['y'][1]), -0.150 + float(q_open) + y_lo + 0.150)   # 露出區：抽屜本地 y ∈ [y_lo, y_lo＋q_open]（櫃前緣 −0.150 以外）
    return D.to_world([0.0, (y_lo + y_lo + float(q_open)) / 2.0, float(it['bottom_top_z'])], q_open, True)


import door_fast_kin as DFK
OBS_SEEDS = list(M.SEEDS) + [np.array([0.0, 0.3, 1.2, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 1.2, -1.5]), np.array([0.0, 0.0, 1.0, 0.0, 1.2, 1.5]), np.array([0.3, 0.2, 1.4, 0.0, 1.2, -1.6])]


def ik_branch(T, b, margin=0.15):
    """俯視構型：多種子閉式 IK，留限位餘裕 margin，取 σ 最大者（s42：抓把手分支 j5<0 做正上方需 j4＝±π 超限 ⇒ 俯視改關節空間換到 j4≈0、j5>0 分支）。"""
    lo, hi = np.asarray(M.LIM.lower, float), np.asarray(M.LIM.upper, float)
    best = None
    for sd in OBS_SEEDS:
        q, r, sm = DFK.ik_closure_fc(T, b, sd)
        q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi
        if r <= 1e-4 and np.all(q >= lo + margin) and np.all(q <= hi - margin) and sm >= 0.02 and (best is None or sm > best[1]):
            best = (q, float(sm))
    return best


def select_observe(q_open, b_now, flip=False):
    for yt in YAW_TIERS:
        r = _select_observe(q_open, b_now, flip, yt)
        if r is not None:
            return r
    return None


def _select_observe(q_open, b_now, flip, yaws):
    c = exposed_center(q_open)
    z_b = float(D.SPEC['interior']['bottom_top_z'])
    yaw_m = math.atan2(D.PULL[0], -D.PULL[1]) if False else _yaw_face() + math.pi / 2   # 筆名目平躺方向（沿把手橫桿）只用來定夾爪開合方向，不讀筆位置
    best = None
    for b in _frame_bases(c, np.arange(0.30, 0.56, 0.025), np.arange(-0.20, 0.201, 0.05), yaws):
        if not M.base_free(b):
            continue
        go = None
        for of in (False, True):                                      # 俯視繞垂直軸兩種（關節空間過去，不受抓把手姿態限制）
            Ro_ = obs_R(of); To_ = obs_T(c, z_b, Ro_)
            qb = ik_branch(To_, b)
            if qb is None:
                continue
            ok_, c_ = _clear(b, qb[0], q_open)
            if ok_ and float(c_['env']) >= ENV_SEL and (go is None or qb[1] > go['sigma']):
                go, Ro, To = dict(q=qb[0], sigma=qb[1], env=float(c_['env']), base=float(c_['base'])), Ro_, To_
        if go is None:
            continue
        cg = np.r_[c[:2], z_b + 0.014]
        fl_ = min((False, True), key=lambda f: _rot_angle(Ro, grasp_T(cg, yaw_m, f)[:3, :3]))
        g = _ik_ok(lambda f: grasp_T(cg, yaw_m, f), b, q_open, flips=(fl_,))
        if g is None:
            continue
        gp = _ik_ok(grasp_T(cg, yaw_m, g['flip'], PRE_M), b, q_open, ENV_SEL)
        if gp is None:
            continue
        sc = float(np.hypot(*(b[:2] - np.asarray(b_now[:2], float)))) + 0.02 / min(go['sigma'], g['sigma']) - W_ENV * min(go['env'], gp['env'])
        if best is None or sc < best['score']:
            best = dict(base=b.tolist(), sigma_min=min(go['sigma'], g['sigma']), score=sc, env=min(go['env'], g['env']), base_clear=min(go['base'], g['base']), T_obs=To.tolist(), q_obs=[float(x) for x in go['q']], c=c.tolist(), flip_grasp=bool(g['flip']))
    return best


if __name__ == '__main__':
    import time
    w = time.perf_counter(); h = select_handle(np.array([0.0, 9.0, math.pi]))
    print('handle', None if h is None else dict(base=np.round(h['base'], 3).tolist(), end=np.round(h['base_end'], 3).tolist(), sig=round(h['sigma_min'], 3), env=round(h['env'], 3), bc=round(h['base_clear'], 3)), round((time.perf_counter() - w) * 1e3), 'ms')
    w = time.perf_counter(); o = select_observe(OPEN_M, np.array(h['base_end']) if h else np.array([-3.9, 10.3, math.pi]), flip=h['flip'] if h else False)
    print('observe', None if o is None else dict(base=np.round(o['base'], 3).tolist(), sig=round(o['sigma_min'], 3), env=round(o['env'], 3), bc=round(o['base_clear'], 3), c=np.round(o['c'], 3).tolist()), round((time.perf_counter() - w) * 1e3), 'ms')
    mk = D.marker_nominal(OPEN_M)
    w = time.perf_counter(); m = select_marker(mk, _yaw_face() + math.pi / 2, OPEN_M, np.array(o['base']) if o else np.array([-4.2, 10.3, math.pi]), R_ref=np.array(o['T_obs'])[:3, :3] if o else None)
    print('marker', mk.round(3), None if m is None else dict(base=np.round(m['base'], 3).tolist(), sig=round(m['sigma_min'], 3), env=round(m['env'], 3), bc=round(m['base_clear'], 3)), round((time.perf_counter() - w) * 1e3), 'ms')


HANDLE_PAR_COS = math.cos(math.radians(70.0))


def path_ext_handle(T_grasp, p_now, th_now, dirv, b0_sel, v_now, extra_dirs=None, path_pts=None, ds=0.05, s_max=1.2, t_need=1.6, v_min=0.03, v_cap=float(os.environ.get('PATH_V_CAP', '0.15'))):
    """抽屜把手依導航路徑的移動中抓取（Howard 11/20 不要預設路線）：同 cyl_reach_map.path_ext，但淨空用抽屜模型（_ik_ok＝IK＋_clear）、底盤用 M.base_free。
    來向多半垂直櫃面 ⇒ 可行段短、經過速度自然降到很低（減速→閉爪→拉回，速度連續）；拉開段不沿來向橫移（tau＝0）。"""
    p = np.asarray(p_now, float)[:2]; b0 = np.asarray(b0_sel, float)
    to_b = b0[:2] - p
    dm = np.asarray(dirv, float)[:2]; nm = float(np.linalg.norm(dm))
    dirs = []
    if nm > 1e-3:
        dirs.append(dm / nm)                                           # 實際行進方向
    if float(np.linalg.norm(to_b)) > 1e-3:
        dirs.append(to_b / float(np.linalg.norm(to_b)))
    for xd in (extra_dirs or []):                                       # 導航路徑最後一段切向（實際會走的方向）
        xd = np.asarray(xd, float)[:2]
        if float(np.linalg.norm(xd)) > 1e-3:
            dirs.append(xd / float(np.linalg.norm(xd)))               # 朝抓取位（s119d：路徑還在轉彎，瞬時行進方向側向偏 0.59 m 永遠進不了可達區）
    Tg = np.asarray(T_grasp, float)
    n_ = Tg[:2, 2] / max(float(np.linalg.norm(Tg[:2, 2])), 1e-9)       # 櫃面法向（TCP 接近軸水平分量）
    t_ = np.array([-n_[1], n_[0]])
    dirs += [t_, -t_]                                                   # 平行櫃面的兩向（兩指跨在橫桿上只能沿桿方向經過）
    best = None
    ths = [float(th_now)]
    if abs((float(b0[2]) - float(th_now) + math.pi) % (2 * math.pi) - math.pi) > 0.15:
        ths.append(float(b0[2]))
    polys = [np.array([p, p + s_max * d_]) for d_ in dirs]
    if path_pts is not None and len(path_pts) >= 2:                      # 導航路徑折線（目前位置起）＋末段切向再延 0.6 m（Howard 11/20：配合導航實際走出的路徑）
        Pp = np.vstack([p, np.asarray(path_pts, float)[:, :2]])
        tg = Pp[-1] - Pp[-2]; tg = tg / max(float(np.linalg.norm(tg)), 1e-9)
        polys.append(np.vstack([Pp, Pp[-1] + 0.6 * tg]))
    for poly, th in [(P_, th_) for P_ in polys for th_ in ths]:
        cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
        feas = []
        for s_ in np.arange(0.0, min(float(cum[-1]), s_max + 0.6) + 1e-9, ds):
            pp_, d = M._poly_at(poly, cum, s_)
            b = np.array([pp_[0], pp_[1], th])
            if not M.base_clear(b, M.PATH_BASE_MARGIN):
                break
            g = _ik_ok(Tg, b, 0.0, flips=(False,))
            feas.append((float(s_), g is not None, g))
        i0 = next((i for i, f in enumerate(feas) if f[1]), None)
        if i0 is None:
            continue
        i1 = i0
        while i1 + 1 < len(feas) and feas[i1 + 1][1]:
            i1 += 1
        s_a, s_b = feas[i0][0], feas[i1][0]
        L = s_b - s_a
        s_g = s_a + min(0.15, L / 3.0)
        if s_g < 0.2:
            continue                                                    # s131d：已到站才算出經過線（閉爪點只剩 0.1 m 跑道）⇒ 1.75 s 後錯過退回；跑道不足就直接收斂
        dyaw = abs((th - float(th_now) + math.pi) % (2 * math.pi) - math.pi)
        pg_, dg_ = M._poly_at(poly, cum, s_g)
        if any(abs(float(np.dot(M._poly_at(poly, cum, s2)[1], n_))) > HANDLE_PAR_COS for s2 in np.arange(s_g, s_b + 1e-9, ds)):
            continue                                                    # Isaac 48：經過線斜向衝櫃面（與法向夾角 31°）⇒ 夾爪撞把手受力閂鎖；閉爪窗內行進方向須與櫃面法向夾角 ≥70°
        dev = (1.0 - float(np.dot(dg_, dm / nm))) if nm > 1e-3 else 0.0    # 閉爪處切向偏離實際行進方向（0＝同向）
        sc = s_a - 0.5 * L + 0.3 * dyaw + 0.5 * dev
        if best is None or sc < best[0]:
            ig = min(range(i0, i1 + 1), key=lambda i: abs(feas[i][0] - s_g))
            best = (sc, th, s_a, s_b, s_g, feas[ig][2], dg_.copy(), pg_.copy(), poly.copy())
    if best is None:
        return None
    _, th, s_a, s_b, s_g, g, d, pg, poly = best
    v = float(np.clip((s_b - s_g) / t_need, v_min, min(v_cap, max(float(v_now), 0.08))))
    return dict(b0=[float(pg[0]), float(pg[1]), float(th)], e=[float(d[0]), float(d[1])], v_g=v, s0=-0.02, t_arm=1.0, tau=0.0, poly=[[float(x), float(y)] for x, y in poly],
                q_grasp=[float(x) for x in g['q']], no_align=True, path=True, s_feas=[round(s_a, 3), round(s_b, 3)], s_grasp=round(s_g, 3), s_end=round(s_b - s_g, 3))
