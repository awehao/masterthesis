#!/usr/bin/env python3
"""V1 估計器的合成資料核對（不開 Isaac；Codex 20261101_045634）：Python 射線投射產生無雜訊深度圖（光軸 Z、公尺），場景＝第一桌（桌面、桌腳）、方塊、第二桌、地面、瓶圓柱；
相機外參同 bottle_v1_cam_visibility（底盤相機），沿 bottle_p1_3 量測軌跡 5 Hz、抓取前（CRUISE..CLOSE）。不含機器人自遮擋；理想資料，不代表渲染深度表現。
1) 約定核對：已知點投影後的像素深度＝光軸 Z（不是沿射線距離），反投影回到原點。
2) 理想資料一次核對：有效率（全部影格、幾何預期可見影格）、xy／z／軸角誤差、依距離分層。真值只在這裡用來產生資料與評估。
    python3 evaluation/test_bottle_v1_est.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_v1_cam_visibility as VZ                   # noqa: E402
import bottle_v1_est as EST                             # noqa: E402
import test_bottle_place as TP                          # noqa: E402

W_, H_ = 640, 480
CAM = VZ.CAMS['base']
FX = (W_ / 2) / math.tan(CAM['hfov'] / 2)
K = np.array([[FX, 0, W_ / 2], [0, FX, H_ / 2], [0, 0, 1.0]])
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def rays_cam():
    v, u = np.mgrid[0:H_, 0:W_]
    d = np.stack([(u + 0.0 - K[0, 2]) / K[0, 0], (v + 0.0 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], axis=-1).reshape(-1, 3)   # z 分量 1：t＝光軸 Z
    return d


def render(T_wc, boxes, bottle, near, far):
    """回傳光軸 Z 深度圖。射線 p＝o＋Z·R·d（d 的相機 z 分量為 1，故參數即光軸 Z）。"""
    d = (T_wc[:3, :3] @ rays_cam().T).T
    o = T_wc[:3, 3]
    best = np.full(len(d), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for c, sz in boxes:
            lo, hi = c - sz / 2, c + sz / 2
            t1, t2 = (lo - o) / d, (hi - o) / d
            tn = np.nanmax(np.minimum(t1, t2), axis=1)
            tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tf >= tn) & (tn > 1e-6)
            best = np.where(hit & (tn < best), tn, best)
        tg = -o[2] / d[:, 2]                                                   # 地面 z＝0
        best = np.where((tg > 1e-6) & (tg < best), tg, best)
        b0, ax, r, h = bottle
        oc = o - b0
        dp = d - np.outer(d @ ax, ax)
        op = oc - (oc @ ax) * ax
        A = np.einsum('ij,ij->i', dp, dp)
        Bq = 2 * (dp @ op)
        Cq = op @ op - r * r
        disc = Bq * Bq - 4 * A * Cq
        t = (-Bq - np.sqrt(np.maximum(disc, 0))) / (2 * A)
        hz = (oc @ ax) + t * (d @ ax)
        hit = (disc >= 0) & (t > 1e-6) & (hz >= 0) & (hz <= h)
        best = np.where(hit & (t < best), t, best)
        tcap = (h - oc @ ax) / (d @ ax)                                        # 瓶頂蓋
        pc = oc + np.outer(tcap, np.ones(3)) * 0 + np.outer(tcap, ax) * 0
        pcap = o[None, :] + tcap[:, None] * d - (b0 + h * ax)
        rcap = np.linalg.norm(pcap - np.outer(pcap @ ax, ax), axis=1)
        hitc = (tcap > 1e-6) & (rcap <= r)
        best = np.where(hitc & (tcap < best), tcap, best)
    z = np.where((best >= near) & (best <= far), best, 0.0)
    return z.reshape(H_, W_)


T1 = [(n, c_, sz) for n, c_, sz in G.table_boxes() if n == 'table_top'][0]
TABLE1 = dict(top_z=float(T1[1][2] + T1[2][2] / 2), lo=(T1[1][:2] - T1[2][:2] / 2).tolist(), hi=(T1[1][:2] + T1[2][:2] / 2).tolist())   # 地圖先驗（第一桌）


def scene_boxes():
    return [(c_, sz) for _, c_, sz in G.table_boxes()] + [(SC.OBST['c'], SC.OBST['sz'])] + [(c_, sz) for _, c_, sz in TP.table2_boxes()]


def t_convention():
    T_wc = np.eye(4)
    T_wc[:3, :3] = VZ.rpy(-math.pi / 2, 0, -math.pi / 2)                     # 光學座標：z 朝世界 +x
    T_wc[:3, 3] = [0.0, 0.0, 0.5]
    p = np.array([2.0, 0.3, 0.7])
    box = [(p, np.array([0.02, 0.02, 0.02]))]
    Z = render(T_wc, box, (np.array([9.0, 9.0, 0.0]), np.array([0, 0, 1.0]), 0.01, 0.01), 0.05, 10.0)
    Ti = np.linalg.inv(T_wc)
    pc = Ti[:3, :3] @ p + Ti[:3, 3]
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    zpix = Z[int(round(v)), int(round(u))]
    chk('約定：像素深度＝光軸 Z（立方體前面 = 中心 Z − 0.01）', abs(zpix - (pc[2] - 0.01)) < 2e-3, (zpix, pc[2]))
    ray_len = np.linalg.norm(pc)
    chk('約定：光軸 Z 與沿射線距離確實不同（此點差 > 1 mm，代表沒有混用）', abs(ray_len - pc[2]) > 1e-3, (ray_len, pc[2]))
    P, uv = EST.backproject(Z, K)
    i = int(np.argmin(np.hypot(uv[:, 0] - round(u), uv[:, 1] - round(v))))
    pw = T_wc[:3, :3] @ P[i] + T_wc[:3, 3]
    chk('約定：反投影回世界（立方體前面 x＝1.99）', abs(pw[0] - 1.99) < 2e-3 and abs(pw[1] - 0.3) < 0.01 and abs(pw[2] - 0.7) < 0.01, pw)


def t_trajectory(run='bottle_p1_3'):
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    names = [str(x) for x in Z['phases']]
    boxes = scene_boxes()
    rows = []
    for i in range(0, len(c['t']), 20):                                        # 5 Hz
        ph = names[int(c['phase_index'][i])]
        if ph not in ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE'):
            continue
        q = np.array([c[k][i] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        T_wc = G.C.K.fk(q, CAM['parent']) @ CAM['T']
        w, x, y, z = (c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        ax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        b0 = np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
        D = render(T_wc, boxes, (b0, ax, G.BOTTLE['r'], G.BOTTLE['h']), CAM['near'], CAM['far'])
        e = EST.estimate(D, K, T_wc, TABLE1)
        dist = float(np.linalg.norm(T_wc[:3, 3] - (b0 + 0.1 * ax)))
        Ti = np.linalg.inv(T_wc)
        pts = b0 + np.outer(np.linspace(0.01, 0.19, 10), ax)
        pc = (Ti[:3, :3] @ pts.T).T + Ti[:3, 3]
        vis = []
        for pw, pcm in zip(pts, pc):
            u, v = FX * pcm[0] / pcm[2] + W_ / 2, FX * pcm[1] / pcm[2] + H_ / 2
            vis.append(CAM['near'] <= pcm[2] <= CAM['far'] and 0 <= u < W_ and 0 <= v < H_ and not any(VZ.seg_hits_box(T_wc[:3, 3], pw, bc, bs) for bc, bs in boxes))
        row = dict(t=float(c['t'][i]), phase=ph, dist=dist, expected=float(np.mean(vis)) >= 0.5, valid=e['valid'], why=e['why'])
        if e['valid']:
            row.update(err_xy=float(np.hypot(e['center_xy'][0] - b0[0], e['center_xy'][1] - b0[1])), err_z=float(e['z_from_support'] - b0[2]),
                       err_z_top=float(e['z_from_top'] - b0[2]), err_axis_deg=float(math.degrees(math.acos(min(1.0, abs(float(np.dot(e['axis'], ax))))))), n_pts=e['n_pts'])
        rows.append(row)
    return rows


def summarize(rows):
    def s(rr):
        v = [r for r in rr if r['valid']]
        exy = [r['err_xy'] * 1e3 for r in v]
        return dict(n=len(rr), valid=len(v), rate=round(len(v) / max(len(rr), 1), 3),
                    err_xy_mm=None if not exy else [round(float(np.percentile(exy, 50)), 4), round(float(np.percentile(exy, 95)), 4), round(float(max(exy)), 4)],
                    err_z_support_mm=None if not v else round(float(max(abs(r['err_z']) for r in v)) * 1e3, 3),
                    err_z_top_mm=None if not v else round(float(max(abs(r['err_z_top']) for r in v)) * 1e3, 3),
                    err_axis_deg=None if not v else round(float(max(r['err_axis_deg'] for r in v)), 2),
                    why={w: sum(1 for r in rr if r['why'] == w) for w in set(r['why'] for r in rr) if w != 'ok'})
    out = dict(all=s(rows), expected=s([r for r in rows if r['expected']]))
    for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0)):
        out[f'dist_{lo}-{hi}'] = s([r for r in rows if lo <= r['dist'] < hi])
    return out


def main():
    t_convention()
    rows = t_trajectory()
    res = summarize(rows)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(dict(result=res, rows=rows, note='Python 射線投射理想深度（無雜訊、無機器人自遮擋）；非渲染'), open(os.path.join(HERE, 'results', 'bottle', 'v1_est_synth.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    chk('理想資料：幾何預期可見影格有效率 ≥ 0.8', res['expected']['rate'] >= 0.8, res['expected'])
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
