#!/usr/bin/env python3
"""第八次 r7 候選 B：由線上確認航跡決定的近垂直穿越（Codex 20261110_221334）。純邏輯、不依賴 ROS。
穿越方向：取原路第一個衝突所屬的確認航跡；只有 WindowLayer2 判為 stable（方向穩定）才據此建立，方向不穩時不建立、照既有保護。
  航跡運動方向 vh、法向 n；原路與航跡中心線（過航跡位置、沿 vh）的交點 xc 決定穿越位置（不用真值行人路線、端點或折返排程）。
幾何：等待點 W＝xc 近側、側向 hb＋db；穿越終點 W'＝xc 遠側、側向 hb＋db；hb＝rz(T_LONG)（現行預測在 T_LONG 內的最大帶半寬），
  db＝d_brake(V_UB_R)（煞停掃掠）。接回：交點後原路第一個在遠側、側向 ≥ hb 的點；原路之後沒出帶就接到終點。
  W、W 朝帶內 db 的煞停掃掠都須在帶外（目前所有確認航跡 T_LONG 內不覆蓋）；接近、穿越、接回三段都核靜態可通行（代價圖）。
時間界（模型修正，須揭露）：window_m()。t_lb 同 WindowLayer2；t_ub＝WindowLayer2 加速剖面
  ＋本體轉向：計畫朝向（plan_yaw；GMPC heading_source=plan_yaw 時實際追的朝向）自目前朝向起的累積變化／W_TURN
  ＋平移轉角：相鄰段方向變化 θ 以 2·v·sin(θ/2)/A_UB（v＝V_UB_R；第一段用目前速度）計。
  原 robot_window 把路徑切線變化當本體轉向（chord 朝向時成立）；plan_yaw 下本體轉向依計畫朝向，平移轉角另計，兩項都保留、不刪。
參數全部沿用 m8_window2（DT_MARGIN、rz、V_UB、V_UB_R、A_UB、W_TURN…），不新增調整量。
"""
import math

import numpy as np

import m8_window2 as W2


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def band_half():
    return float(W2.rz(W2.T_LONG))


def window_m(path, yaw, rp, ryaw, vel):
    """回傳 s, t_lb, t_ub, 分項（rot_s 本體轉向、corner_s 平移轉角）。"""
    path, rp, vel = np.asarray(path, float), np.asarray(rp, float), np.asarray(vel, float)
    seg = np.r_[np.linalg.norm(path[0] - rp), np.linalg.norm(np.diff(path, axis=0), axis=1)]
    s = np.cumsum(seg)
    d = np.diff(np.vstack([rp[None, :], path]), axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    sp = float(np.linalg.norm(vel))
    h_prev = math.atan2(vel[1], vel[0]) if sp > 0.05 else None
    for i in range(len(h)):                                             # 零長度段沿用前一方向
        if seg[i] < 1e-6:
            h[i] = h[i - 1] if i else (h_prev if h_prev is not None else 0.0)
    h0 = h_prev if h_prev is not None else h[0]
    th = np.abs(wrap(np.diff(np.r_[h0, h])))
    vc = np.r_[sp if h_prev is not None else 0.0, np.full(len(h) - 1, W2.V_UB_R)]
    corner = np.cumsum(2.0 * vc * np.sin(th / 2.0) / W2.A_UB)
    rot = np.cumsum(np.abs(wrap(np.diff(np.r_[float(ryaw), np.asarray(yaw, float)])))) / W2.W_TURN
    v_along = max(0.0, float(vel @ np.array([math.cos(h[0]), math.sin(h[0])])))
    t_lb = W2._profile(s, v_along, W2.T_DELAY_LB, W2.A_LB, W2.V_LB)
    t_ub = W2._profile(s, v_along, W2.T_DELAY, W2.A_UB, W2.V_UB_R) + rot + corner
    return s, t_lb, t_ub, dict(rot_s=round(float(rot[-1]), 3), corner_s=round(float(corner[-1]), 3))


def conflicts(P, t_lb, t_ub, M, margin):
    """逐點時空核對：conf（衝突）、cov_any（T_LONG 內被覆蓋）、第一個衝突的細節。"""
    P = np.atleast_2d(np.asarray(P, float))
    conf = np.zeros(len(P), bool)
    cov_any = np.zeros(len(P), bool)
    first = None
    for x, k in M:
        cov, taus = W2.cover_times(P, x, k)
        ca = cov.any(axis=1)
        cov_any |= ca
        for i in np.flatnonzero(ca):
            tt = taus[cov[i]]
            hit = (tt >= t_lb[i] - margin) & (tt <= t_ub[i] + margin)
            if hit.any():
                conf[i] = True
                if first is None or i < first['i']:
                    first = dict(i=int(i), tid=int(x.get('tid', -1)), kind=k, tau_cover=round(float(tt[hit][0]), 2),
                                 robot_win=[round(float(t_lb[i] - margin), 2), round(float(t_ub[i] + margin), 2)],
                                 trk=[round(float(x['x']), 2), round(float(x['y']), 2), round(float(x['vx']), 2), round(float(x['vy']), 2)], pt=[round(float(P[i, 0]), 2), round(float(P[i, 1]), 2)])
    return conf, cov_any, first


def beyond(t_lb, t_ub, margin):
    """各點機器人時窗 [t_lb−margin, t_ub＋margin] 是否超出預測範圍 T_LONG：0＝全在範圍內、1＝部分超出、2＝整段超出（該點未核對）。"""
    lo, hi = np.asarray(t_lb, float) - margin, np.asarray(t_ub, float) + margin
    return np.where(lo > W2.T_LONG, 2, np.where(hi > W2.T_LONG, 1, 0))


def status(conf, bey):
    """conflict｜clear_within_horizon（範圍內無衝突且全在範圍內）｜clear_beyond_horizon（範圍內無衝突，但有點的時窗超出 T_LONG＝未知，不算已驗證）。"""
    if np.asarray(conf).any():
        return 'conflict'
    return 'clear_within_horizon' if not (np.asarray(bey) > 0).any() else 'clear_beyond_horizon'


def line_of(x):
    v = np.array([x['vx'], x['vy']], float)
    vh = v / max(np.linalg.norm(v), 1e-9)
    return np.array([x['x'], x['y']], float), vh, np.array([-vh[1], vh[0]])


def band_span(P, line, hb=None, start=0):
    """路徑（自索引 start 起）相對航跡中心線的幾何帶（|側向| < hb）：第一個入帶與其後第一個出帶的索引（None＝不入帶／不出帶）。"""
    hb = band_half() if hb is None else hb
    p0, _, n = line
    lat = (np.asarray(P, float) - p0) @ n
    inb = np.abs(lat) < hb
    inb[:start] = False
    ii = np.flatnonzero(inb)
    if not len(ii):
        return None, None
    jj = np.flatnonzero(~inb[ii[0]:])
    return int(ii[0]), (int(ii[0] + jj[0]) if len(jj) else None)


def build_cross(rp, ryaw, path, path_yaw, M, blocked, threat):
    """threat＝(航跡, kind)。回傳 (候選, None) 或 (None, 原因)。"""
    x, k = threat
    if k != 'stable':
        return None, dict(why='direction_unstable', tid=int(x.get('tid', -1)), kind=k)
    sp = math.hypot(x['vx'], x['vy'])
    if sp < W2.V_MOVER:
        return None, dict(why='threat_slow', tid=int(x.get('tid', -1)), speed=round(sp, 3))
    rp, path, path_yaw = np.asarray(rp, float), np.asarray(path, float), np.asarray(path_yaw, float)
    p0, vh, n = line_of(x)
    lat_r = float((rp - p0) @ n)
    side = 1.0 if lat_r >= 0.0 else -1.0                                # 近側
    lat = (path - p0) @ n
    far = np.flatnonzero(side * lat < 0.0)
    if not len(far):
        return None, dict(why='path_does_not_cross')
    i = int(far[0])
    a = path[i - 1] if i > 0 else rp
    la = float((a - p0) @ n)
    lb_ = float(lat[i])
    xc = a + (la / (la - lb_)) * (path[i] - a)
    hb, db = band_half(), W2.d_brake(W2.V_UB_R)
    W = xc + side * n * (hb + db)
    Wp = xc - side * n * (hb + db)
    after = np.flatnonzero((np.arange(len(path)) >= i) & (-side * lat >= hb))
    j = int(after[0]) if len(after) else len(path) - 1
    segs = dict(approach=(rp, W), cross=(W, Wp), reconnect=(Wp, path[j]))
    for nm, (p, q) in segs.items():
        if any(blocked(pt) for pt in W2.densify(p, q)):
            return None, dict(why='static', seg=nm)
    probe = np.vstack([W[None, :], W2.densify(W, W - side * n * db)])
    cov = np.zeros(len(probe), bool)
    for xx, kk in M:
        c, _ = W2.cover_times(probe, xx, kk)
        cov |= c.any(axis=1)
    if cov.any():
        return None, dict(why='wait_point_or_sweep_in_band')
    A = W2.densify(rp, W)[1:]
    C = W2.densify(W, Wp)[1:]
    R = W2.densify(Wp, path[j])[1:]
    P = np.vstack([A, C, R, path[j + 1:]])
    yA = np.full(len(A) + len(C), float(ryaw))                          # 接近與穿越維持目前朝向（plan_yaw 側移）
    dy = float(wrap(path_yaw[j] - ryaw))
    yR = float(ryaw) + dy * np.linspace(0.0, 1.0, len(R) + 1)[1:]
    Y = np.r_[yA, yR, path_yaw[j + 1:]]
    return dict(P=P, yaw=Y, W=W, Wp=Wp, xc=xc, side=side, line=(p0, vh, n), tid=int(x.get('tid', -1)), i_W=len(A) - 1, i_Wp=len(A) + len(C) - 1,
                j_reconnect=j, len=dict(approach=round(float(np.linalg.norm(W - rp)), 3), cross=round(float(np.linalg.norm(Wp - W)), 3),
                                        reconnect=round(float(np.linalg.norm(path[j] - Wp)), 3))), None


def times_at(s, t_lb, t_ub, i):
    return None if i is None else dict(s=round(float(s[i]), 2), t_lb=round(float(t_lb[i]), 2), t_ub=round(float(t_ub[i]), 2))
