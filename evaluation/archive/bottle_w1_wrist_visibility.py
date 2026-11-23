#!/usr/bin/env python3
"""W1 手腕相機可見度（Codex 20261101_072429；Howard「要夾東西應該要用手腕」）：沿既有實跑軌跡（s1_steps.npz 的量測底盤、關節、手指開度、瓶位姿），
逐步算手腕相機看瓶子的情形。純幾何投影、非渲染；瓶位姿真值只用來評估可見度。
相機：link_eef → 手腕 D435i 彩色光學座標（bottle_v1_cam_visibility.CAMS['wrist']，URDF）；hfov 69°、640×480、fx 465.6（門任務 CP1 讀回）、半像素中心；Isaac 裁切 0.05–10 m。
瓶：圓柱側面＋頂面取樣點，背面剔除（瓶身自遮）；場景遮擋：桌、桌腳、方塊、第二桌（軸對齊盒）。
夾爪遮擋：uflite_gripper_link、兩指（實際開度）的網格點投影成深度緩衝（每點依鄰點間距展開成方塊，1–6 mm），瓶點比夾爪點遠就算被遮。
  兩種口徑：sim＝只算離相機 ≥ 0.05 m 的夾爪點（Isaac 近裁切會把更近的幾何裁掉）；all＝所有在相機前方的夾爪點（較接近實體相機）。
  不含 link_eef（相機本體與座在其中）與 link6（在相機後方）；網格點近似，邊緣 ±1 個展開尺寸。
    python3 evaluation/bottle_w1_wrist_visibility.py [bottle_p1_3]
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
import test_bottle_place as TP                          # noqa: E402

CAM = VZ.CAMS['wrist']
W_, H_ = 640, 480
FX = (W_ / 2) / math.tan(CAM['hfov'] / 2)
OCC = ('uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
NEAR_SIM = 0.05
PHASES = ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE')


def spacing():
    """各夾爪連桿網格點在連桿座標的最近鄰間距（展開尺寸，夾在 1–6 mm）。"""
    out = {}
    for n in OCC:
        P = G.C.CL[n]
        d = np.sort(np.linalg.norm(P[:, None] - P[None], axis=2), axis=1)[:, 1]
        out[n] = np.clip(d, 0.001, 0.006)
    return out


def bottle_pts(b0, ax):
    """側面（90 角 × 30 高）＋頂面（3 圈）取樣點與外法向。"""
    r, h = G.BOTTLE['r'], G.BOTTLE['h']
    ex = np.cross(ax, [1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.cross(ax, [0, 1.0, 0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(ax, ex)
    th = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    zz = np.linspace(0.005, h - 0.005, 30)
    T_, Z_ = np.meshgrid(th, zz)
    nrm = np.cos(T_)[..., None] * ex + np.sin(T_)[..., None] * ey
    P = b0 + Z_[..., None] * ax + r * nrm
    pts, nr = [P.reshape(-1, 3)], [nrm.reshape(-1, 3)]
    for rr in (0.3 * r, 0.6 * r, 0.9 * r):
        th2 = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        pts.append(b0 + h * ax + rr * (np.cos(th2)[:, None] * ex + np.sin(th2)[:, None] * ey))
        nr.append(np.tile(ax, (36, 1)))
    return np.vstack(pts), np.vstack(nr)


def seg_hits_boxes(p0, P, boxes):
    """相機 p0 到每個點 P 的線段是否穿過任一軸對齊盒（向量化 slab）。"""
    hit = np.zeros(len(P), bool)
    d = P - p0
    for c, sz in boxes:
        lo, hi = c - sz / 2, c + sz / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            a, b = (lo - p0) / d, (hi - p0) / d
        t0 = np.nanmax(np.where(np.abs(d) < 1e-12, np.where((p0 >= lo) & (p0 <= hi), -np.inf, np.inf), np.minimum(a, b)), axis=1)
        t1 = np.nanmin(np.where(np.abs(d) < 1e-12, np.where((p0 >= lo) & (p0 <= hi), np.inf, -np.inf), np.maximum(a, b)), axis=1)
        hit |= (np.maximum(t0, 0.0) <= np.minimum(t1, 1.0)) & (t1 > 1e-6) & (t0 < 1.0 - 1e-6)
    return hit


def project(Ti, P):
    pc = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = FX * pc[:, 0] / z + W_ / 2 - 0.5
        v = FX * pc[:, 1] / z + H_ / 2 - 0.5
    return u, v, z


def zbuffer(Ti, W, fq_sp, near):
    zb = np.full((H_, W_), np.inf)
    for n in OCC:
        u, v, z = project(Ti, W[n])
        ok = (z > near) & np.isfinite(u) & np.isfinite(v)
        rad = np.ceil(FX * fq_sp[n] / np.maximum(z, 1e-6) / 2).astype(int)
        for uu, vv, zz, rr in zip(u[ok], v[ok], z[ok], rad[ok]):
            iu, iv = int(round(uu)), int(round(vv))
            u0, u1, v0, v1 = max(iu - rr, 0), min(iu + rr + 1, W_), max(iv - rr, 0), min(iv + rr + 1, H_)
            if u0 < u1 and v0 < v1:
                blk = zb[v0:v1, u0:u1]
                np.minimum(blk, zz, out=blk)
    return zb


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'bottle_p1_3'
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    names = [str(x) for x in Z['phases']]
    ph = np.array([names[int(v)] for v in c['phase_index']])
    t = c['t']
    boxes = [(cc, sz) for _, cc, sz in G.table_boxes() if _ != 'table_top'] + [(SC.OBST['c'], SC.OBST['sz'])] + [(cc, sz) for _, cc, sz in TP.table2_boxes()]
    top = [(cc, sz) for n_, cc, sz in G.table_boxes() if n_ == 'table_top']
    sp = spacing()
    i_unf = int(np.argmax((ph == 'CRUISE') & (c['base_x'] >= -1.55)))
    idx = [i for i in range(i_unf, len(t)) if ph[i] in PHASES]
    idx = [i for k, i in enumerate(idx) if ph[i] != 'CRUISE' or k % 10 == 0]          # CRUISE 每 10 步；PREGRASP 之後每步
    rows = []
    for i in idx:
        q = np.array([c[k][i] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        Tc = G.C.K.fk(q, CAM['parent']) @ CAM['T']
        Ti = np.linalg.inv(Tc)
        w, x, y, z = (c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        ax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        b0 = np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
        P, N = bottle_pts(b0, ax)
        front = np.einsum('ij,ij->i', N, Tc[:3, 3] - P) > 0
        u, v, zc = project(Ti, P)
        inimg = front & (zc >= NEAR_SIM) & (zc <= CAM['far']) & (u >= -0.5) & (u < W_ - 0.5) & (v >= -0.5) & (v < H_ - 0.5)
        scene = ~seg_hits_boxes(Tc[:3, 3], P, boxes + top)
        W = G.link_world(q, float(c['finger1_q'][i]))
        out = dict(t=float(t[i]), phase=str(ph[i]), base_x=float(q[0]), n_front=int(front.sum()))
        for tag, near in (('sim', NEAR_SIM), ('all', 1e-4)):
            zb = zbuffer(Ti, W, sp, near)
            iu, iv = np.clip(np.round(u).astype(int), 0, W_ - 1), np.clip(np.round(v).astype(int), 0, H_ - 1)
            unocc = zc < zb[iv, iu] - 1e-3
            vis = inimg & scene & unocc
            pix = np.unique(iv[vis] * W_ + iu[vis]) if vis.any() else np.zeros(0, int)
            side = vis[:2700].reshape(30, 90)
            out[tag] = dict(vis_frac=float(vis.sum() / max(front.sum(), 1)), n_pix_pts=int(len(pix)), occ_grip=int((inimg & scene & ~unocc).sum()),
                            top_vis=bool(vis[2700:].any()), arc_cols=int(side.any(axis=0).sum()), grip_cover=float(np.isfinite(zb).mean()),
                            zmin=float(zc[vis].min()) if vis.any() else None, zmax=float(zc[vis].max()) if vis.any() else None)
        out['in_fov_frac'] = float(inimg.sum() / max(front.sum(), 1))
        out['axis_dist'] = float(np.linalg.norm(Tc[:3, 3] - (b0 + 0.5 * G.BOTTLE['h'] * ax)))
        rows.append(out)
    # 依相位彙整與連續可見區間（vis_frac ≥ 0.3 且側面弧 ≥ 10 欄：可做圓擬合的粗門檻，只供看區間，不是估計器判準）
    summ = {}
    for tag in ('sim', 'all'):
        s = {}
        for p in PHASES:
            R = [r for r in rows if r['phase'] == p]
            if not R:
                continue
            ok = [r for r in R if r[tag]['vis_frac'] >= 0.3 and r[tag]['arc_cols'] >= 10]
            s[p] = dict(n=len(R), t=[round(R[0]['t'], 2), round(R[-1]['t'], 2)], n_ok=len(ok), ok_t=[round(ok[0]['t'], 2), round(ok[-1]['t'], 2)] if ok else None,
                        vis_frac=[round(min(r[tag]['vis_frac'] for r in R), 3), round(max(r[tag]['vis_frac'] for r in R), 3)],
                        zmin=[None if all(r[tag]['zmin'] is None for r in R) else round(min(r[tag]['zmin'] for r in R if r[tag]['zmin'] is not None), 3),
                              None if all(r[tag]['zmin'] is None for r in R) else round(max(r[tag]['zmin'] for r in R if r[tag]['zmin'] is not None), 3)],
                        grip_cover=[round(min(r[tag]['grip_cover'] for r in R), 3), round(max(r[tag]['grip_cover'] for r in R), 3)],
                        n_pix=[min(r[tag]['n_pix_pts'] for r in R), max(r[tag]['n_pix_pts'] for r in R)], top_vis=sum(r[tag]['top_vis'] for r in R))
        summ[tag] = s
    res = dict(run=run, cam=dict(parent=CAM['parent'], hfov_deg=math.degrees(CAM['hfov']), fx=FX, near_sim=NEAR_SIM), summary=summ, rows=rows,
               note='純幾何投影（網格點展開深度緩衝、軸對齊盒遮擋），非渲染；瓶位姿真值只用來評估；不含相機本體與座（link_eef）遮擋；最短有效深度未套實體 D435i 規格（Isaac 只有近裁切 0.05 m）')
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', f'w1_wrist_visibility_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
