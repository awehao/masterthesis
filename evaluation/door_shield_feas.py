#!/usr/bin/env python3
"""離線：開門參考與原 Shield（scan_safety_shield）是否相容（D1 乾跑發現 Shield 把被拉的門扇當障礙物改底盤命令）。
同一起點（door_ahead_case：收斂接近候選終點的 READY 預測狀態、首段請求的把手位姿與 T_HE），照非阻塞執行器的 P2 滾動流程離線串段
（每段 H＝2 s、下一段起點＝本段參考在 σ＝L_LAT 的狀態與中央差分速度；剩餘夠小時試末段），直到門角到 85° 或不可行。
只在本程序內覆寫底盤淨空門檻 W.THR['base']（不改模組）：0.02（現行）與更大的值。
Shield 相容判定（逐參考節點）：以世界節點同一套射線投射（場景方塊＋當下角度門扇）得 360 點，套 Shield 的不等式
  v_n ≤ α·(d − d_stop)，d＝距離−R，d_stop＝d0＋v_app·τ_eff＋v_app²/(2a)＋eps_scan＋eps_footprint（參數讀 D1 的 Shield 參數檔；τ_eff＝τ＋掃描週期 0.1 s）；
  v＝參考的本體速度（命令即參考前饋的近似）。margin＝min_i[α(d−d_stop)−v_n]，<0 ⇒ Shield 會改命令。
    python3 door_shield_feas.py OUT.json [thr1,thr2,…]"""
import json
import math
import os
import sys
import time

os.environ.setdefault('POOL_N', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv)
from door_ahead_case import first, x_pred, BG_V, GOAL, FQ, phi_b, H, L_LAT, helper, SHARED, CSW, BG   # noqa: E402
import numpy as np                                                   # noqa: E402
import door_motm_p2 as P2                                            # noqa: E402
W = P2.WB
sys.path.insert(0, os.path.join(HERE, 'far_dry'))
SHP = dict(robot_radius=0.30, alpha=2.0, d0=0.05, eps_scan=0.03, eps_footprint=0.02, tau=0.15, a_brake=6.25)   # D1 讀回的 Shield 參數（results/motm_dry/d1_n3.side）
LEAF = (0.0, 0.95, 0.0, 0.04)


def scan_pts(b, boxes, WN):
    p = b[:2]
    th = b[2] + WN.SCAN_MIN + np.arange(WN.SCAN_N) * WN.SCAN_INC
    return th


def shield_margin(x, v_world, boxes, WN):
    b, phi = x[:3], float(x[9])
    r = WN.ray_cast(b[:2], b[2], boxes)
    c, s = math.cos(phi), math.sin(phi)
    p = b[:2]
    px, py = c * p[0] - s * p[1], s * p[0] + c * p[1]
    th = b[2] + WN.SCAN_MIN + np.arange(WN.SCAN_N) * WN.SCAN_INC
    dxw, dyw = np.cos(th), np.sin(th)
    dx, dy = c * dxw - s * dyw, s * dxw + c * dyw
    with np.errstate(divide='ignore', invalid='ignore'):
        tx0, tx1 = (LEAF[0] - px) / dx, (LEAF[1] - px) / dx
        ty0, ty1 = (LEAF[2] - py) / dy, (LEAF[3] - py) / dy
        tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin) & (tmin > 0)
    r = np.minimum(r, np.where(hit, tmin, np.inf))
    ok = np.isfinite(r) & (r <= 2.0)
    if not ok.any():
        return np.inf, np.inf
    ang = WN.SCAN_MIN + np.arange(WN.SCAN_N) * WN.SCAN_INC          # 本體座標的射線方向
    n = np.stack([np.cos(ang), np.sin(ang)], 1)[ok]
    rr = r[ok]
    cb, sb = math.cos(b[2]), math.sin(b[2])
    vb = np.array([cb * v_world[0] + sb * v_world[1], -sb * v_world[0] + cb * v_world[1]])
    vn = n @ vb
    va = np.maximum(vn, 0.0)
    d = rr - SHP['robot_radius']
    dstop = SHP['d0'] + va * (SHP['tau'] + 0.1) + va * va / (2 * SHP['a_brake']) + SHP['eps_scan'] + SHP['eps_footprint']
    m = SHP['alpha'] * (d - dstop) - vn
    return float(m.min()), float(rr.min())


def run(thr, boxes, WN):
    W.THR['base'] = float(thr); W.THR_PLAN['base'] = float(thr)
    xs, vs, sig_done, segs, nodes, warm = x_pred.copy(), BG_V.copy(), 0.0, [], [], None
    t0 = time.perf_counter()
    for k in range(14):
        fin = False
        ft = None
        try:
            ft = P2.plan_final(first['T_plan'], first['T_HE_pred'], xs, vs, GOAL, FQ, phi_b)
        except ValueError:
            ft = None
        if ft is not None and ft[2]['feasible']:
            o, fin = ft, True
        elif k == 0 or warm is None:
            o = P2.plan_seg(first['T_plan'], first['T_HE_pred'], xs, vs, H, GOAL, FQ, phi_b)
        else:                                                         # 同規劃程序：已執行參考尾段熱啟；不可行再冷啟
            o = P2.plan_seg(first['T_plan'], first['T_HE_pred'], xs, vs, H, GOAL, FQ, phi_b, warm=P2.RefWarm(warm[0], warm[1]), t_s=warm[2])
            if not o[2]['feasible']:
                o2 = P2.plan_seg(first['T_plan'], first['T_HE_pred'], xs, vs, H, GOAL, FQ, phi_b)
                if o2[2]['feasible']:
                    o = o2
        Xd, h, _ = o[0].dense_reference(4)
        segs.append(dict(k=k, final=fin, feasible=bool(o[2]['feasible']), failed=o[2]['failed'], phi_end_deg=round(math.degrees(float(Xd[-1, 9])), 2), wall_s=round(o[3], 2)))
        if not o[2]['feasible']:
            break
        i_s = len(Xd) - 1 if fin else min(int(round(L_LAT / h)), len(Xd) - 1)
        for i in range(0, i_s + 1 if fin else i_s):
            v = (Xd[min(i + 1, len(Xd) - 1)] - Xd[max(i - 1, 0)]) / (2 * h)
            m, rmin = shield_margin(Xd[i], v[:2], boxes, WN)
            c = W.clear_door(Xd[i], first['T_plan'])
            nodes.append([sig_done + i * h, math.degrees(Xd[i, 9]), m, rmin, c['base'], float(np.hypot(v[0], v[1]))])
        if fin:
            break
        warm = (Xd.copy(), h, i_s * h)
        xs = Xd[i_s].copy()
        vv = (Xd[min(i_s + 1, len(Xd) - 1)] - Xd[max(i_s - 1, 0)]) / (2 * h)
        W.set_model(phi0=phi_b, finger_q=FQ)
        vs = np.r_[vv[:3], helper.closure_velocity(xs, vv[:3], vv[9]), vv[9]]
        sig_done += i_s * h
    N = np.array(nodes) if nodes else np.zeros((0, 6))
    reached = bool(segs and segs[-1]['final'] and segs[-1]['feasible'])
    out = dict(thr_base=thr, reached_goal=reached, segs=segs, wall_s=round(time.perf_counter() - t0, 1))
    if len(N):
        bad = N[:, 2] < 0
        out.update(n_nodes=int(len(N)), shield_margin_min=round(float(N[:, 2].min()), 4), n_shield_violate=int(bad.sum()), first_violate_phi_deg=None if not bad.any() else round(float(N[bad][0, 1]), 1),
                   range_min_m=round(float(N[:, 3].min()), 3), base_clear_min=round(float(N[:, 4].min()), 4), phi_max_deg=round(float(N[:, 1].max()), 2), base_speed_max=round(float(N[:, 5].max()), 3))
    return out, N


class _WN:
    """far_dry/world_node.py 的場景方塊與射線投射（照抄；該檔在模組層匯入 ROS 訊息，本離線程序不載入 ROS）。"""
    DT, SCAN_N, SCAN_MIN, SCAN_MAX = 0.01, 360, -3.14159, 3.14159
    SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
    RANGE_MIN, RANGE_MAX = 0.12, 10.0

    @staticmethod
    def load_boxes(lidar_z):
        reg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
        B = []
        for it in reg['walls'] + reg['furniture']:
            c, s = it['c'], it['s']
            if c[2] - s[2] / 2 <= lidar_z <= c[2] + s[2] / 2:
                B.append((it['name'], c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2))
        return B, reg, None

    @classmethod
    def ray_cast(cls, p, yaw, boxes):
        th = yaw + cls.SCAN_MIN + np.arange(cls.SCAN_N) * cls.SCAN_INC
        dx, dy = np.cos(th), np.sin(th)
        best = np.full(cls.SCAN_N, np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            for _, x0, x1, y0, y1 in boxes:
                tx0, tx1 = (x0 - p[0]) / dx, (x1 - p[0]) / dx
                ty0, ty1 = (y0 - p[1]) / dy, (y1 - p[1]) / dy
                tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
                tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
                hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
                t = np.where(tmin > 0, tmin, np.inf)
                best = np.where(hit, np.minimum(best, t), best)
        return np.where(np.isfinite(best) & (best >= cls.RANGE_MIN) & (best <= cls.RANGE_MAX), best, np.inf)


if __name__ == '__main__':
    WN = _WN
    gm = json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))
    boxes, _, _ = WN.load_boxes(float(gm['lidar_z']))
    thrs = [float(x) for x in (sys.argv_[2] if len(sys.argv_) > 2 else '0.02,0.06,0.10,0.12').split(',')]
    res = []
    for th in thrs:
        o, N = run(th, boxes, WN)
        print(json.dumps({k: v for k, v in o.items() if k != 'segs'}, ensure_ascii=False), flush=True)
        print('   segs', [(s['k'], s['feasible'], s['final'], s['phi_end_deg'], s['failed']) for s in o['segs']], flush=True)
        res.append(o)
        np.save(sys.argv_[1].replace('.json', '_thr%.2f.npy' % th), N)
    json.dump(dict(shield_params=SHP, results=res, note='離線；只覆寫本程序 W.THR[base]；Shield 相容以參考本體速度近似命令'), open(sys.argv_[1], 'w'), ensure_ascii=False, indent=1)
    for x_ in (SHARED, CSW, BG):
        try:
            x_.close()
        except Exception:                                             # noqa: BLE001
            pass
