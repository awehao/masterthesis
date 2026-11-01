#!/usr/bin/env python3
"""三維定位採集的固定動作設計與離線核對（Codex 20261101_220915；純運動學、無 Isaac）。六個佈置共用同一套動作，不依實際桌高選姿態。
佈置（凍結）：第一桌桌面頂 TABLE_Z＝0.45／0.60／0.75 m（厚 0.03、桌腳由地面到桌板底、內縮同資產）× 瓶位 BOTTLES＝(0.00, 0.42)、(0.15, 0.55)（世界座標，瓶底在桌面）。
  第二桌（0.60 m）、紅色方塊障礙同 W1 場景且六個佈置一致；不放移動物。
動作（凍結）：
  1 接近：底盤與手臂照名目趟 bottle_w1_compact3_nominal1 的紀錄 0 → T_OBS（14.80 s，手臂剛好到 K2_READY；底盤 (−0.597, −0.009)），之後底盤停住。
  2 展臂到固定俯視：K2_READY → Q_LOOK（關節空間 min-jerk，時長依 v_arm 與整段 max|Δq|），停 HOLD 秒擷取，再回 K2_READY、再回收 q_tuck。
  Q_LOOK 的選法（只用 0.75 m 最高桌當碰撞限制，因此對三個桌高都同一個）：手腕相機光軸對準兩瓶位中點 (0.075, 0.485, 0.70)，
  相機位置在底盤座標格點上搜尋，IK 延續自 K2_READY；約束＝關節餘裕 ≥ 0.05、自碰 ≥ 8.39 mm、連桿點雲對三個桌高的桌面／桌腳與兩瓶位瓶身 ≥ 0.05 m、
  K2_READY→Q_LOOK 直線關節路徑（20 點）同樣滿足；目標＝相機最高（同高取視線較陡）。
輸出 results/vision/seg3d_collect_design.json：Q_LOOK、相機位姿、各佈置的碰撞餘裕（接近段、過渡段、觀察段）、可見性預估
  （瓶身在影像內比例、相機是否高於桌面、瓶頂上緣外側視線是否落在桌面上＝瓶頂可用深度確認的幾何可能、瓶底周圍桌面環是否在影像內）。
可見性預估不計遮擋，只是幾何可能性，不是採集結果。
    python3 evaluation/bottle_seg3d_collect_design.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import wrist_v0_capture as W0                           # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_manip.urdf')
TABLE_Z = (0.45, 0.60, 0.75)
BOTTLES = ((0.00, 0.42), (0.15, 0.55))
SRC_RUN, T_OBS = 'bottle_w1_compact3_nominal1', 14.80
LOOK_AT = np.array([0.075, 0.485, 0.70])
CLEAR, JOINT_MARGIN = 0.05, 0.05
K = G.C.K
W_, H_, HFOV = 640, 480, math.radians(69.0)
FX = (W_ / 2) / math.tan(HFOV / 2)


def table_boxes(top_z):
    return G.table_boxes(dict(G.S1_TABLE, top_z=top_z))


def bottle_box(xy, top_z):
    r, h = G.BOTTLE['r'], G.BOTTLE['h']
    return np.array([xy[0], xy[1], top_z + h / 2]), np.array([2 * r, 2 * r, h])


def clearance(q, top_z):
    """手臂連桿點雲（手指張開）對桌面／桌腳、兩瓶位瓶身的最小距離；自碰；關節餘裕。"""
    W = G.link_world(q, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    boxes = [(c_, s_) for _, c_, s_ in table_boxes(top_z)] + [bottle_box(xy, top_z) for xy in BOTTLES]
    d = min(float(G.box_dist(P, c_, s_).min()) for c_, s_ in boxes)
    sc, who = G.C.self_clear(q, G.Q_OPEN)
    lo, hi = np.asarray(LITE6_SAFE.lower, float), np.asarray(LITE6_SAFE.upper, float)
    return dict(env=d, self=float(sc), self_who=str(who), margin=float(min((q[3:] - lo).min(), (hi - q[3:]).min())))


def cam_T(q, T_mount):
    return K.fk(q, 'link_eef') @ T_mount


def look_T(c, g):
    z = (g - c) / np.linalg.norm(g - c)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    if x[0] * z[1] - x[1] * z[0] < 0:                  # 影像右方取一致方向
        x = -x
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = c
    return T


def project(T_wc, P):
    Ti = np.linalg.inv(T_wc)
    Pc = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    ok = Pc[:, 2] > 0.05
    u = FX * Pc[:, 0] / np.where(ok, Pc[:, 2], 1) + W_ / 2
    v = FX * Pc[:, 1] / np.where(ok, Pc[:, 2], 1) + H_ / 2
    return ok & (u >= 0) & (u < W_) & (v >= 0) & (v < H_), u, v


def visibility(T_wc, xy, top_z):
    r, h = G.BOTTLE['r'], G.BOTTLE['h']
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    body = np.array([[xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), top_z + z] for z in np.linspace(0, h, 6) for t in a])
    inview, u, v = project(T_wc, body)
    cam = T_wc[:3, 3]
    # 瓶頂上緣外側的視線：從相機穿過瓶頂遠側邊緣點、延長後先碰到什麼（桌面平面在桌面投影內 ⇒ 有效深度背景）
    top_far = np.array([xy[0], xy[1], top_z + h]) + r * np.r_[(np.array(xy) - cam[:2]) / max(np.linalg.norm(np.array(xy) - cam[:2]), 1e-9), 0.0]
    dvec = top_far - cam
    tb = table_boxes(top_z)[0]
    lo_, hi_ = tb[1][:2] - tb[2][:2] / 2, tb[1][:2] + tb[2][:2] / 2
    behind = 'none'
    if dvec[2] < -1e-6:
        s_t = (top_z - cam[2]) / dvec[2]
        hit = cam + s_t * dvec
        if s_t > 1.0 and lo_[0] <= hit[0] <= hi_[0] and lo_[1] <= hit[1] <= hi_[1]:
            behind = 'table_top'
        elif s_t > 1.0:
            behind = 'floor_or_beyond'
    ring = np.array([[xy[0] + rr * math.cos(t), xy[1] + rr * math.sin(t), top_z] for rr in (0.03, 0.06, 0.09) for t in a])
    ring_in = project(T_wc, ring)[0]
    return dict(body_in_view=float(inview.mean()), cam_above_table=bool(cam[2] > top_z + 0.02), cam_height_over_table=float(cam[2] - top_z),
                top_background=behind, support_ring_in_view=float(ring_in.mean()), dist=float(np.linalg.norm(np.array([xy[0], xy[1], top_z + 0.1]) - cam)))


def main():
    T_mount = W0.urdf_chain_T(URDF, 'link_eef', 'camera_color_optical_frame')
    Z = np.load(os.path.join(HERE, 'runs', SRC_RUN, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    k_obs = int(np.argmin(np.abs(c['t'] - T_OBS)))
    qrow = lambda k: np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])   # noqa: E731
    q_ready = qrow(k_obs)
    base = q_ready[:3]
    T_eef_tcp = np.linalg.inv(K.fk(q_ready, 'link_eef')) @ K.fk(q_ready, 'link_tcp')
    # 接近段（紀錄 0 → T_OBS）在三個桌高的碰撞核對
    appr = {}
    for tz in TABLE_Z:
        w = dict(env=9.9, self=9.9, margin=9.9)
        for k in range(0, k_obs + 1, 5):
            cl = clearance(qrow(k), tz)
            w = dict(env=min(w['env'], cl['env']), self=min(w['self'], cl['self']), margin=min(w['margin'], cl['margin']))
        appr[str(tz)] = w
    # Q_LOOK 搜尋：底盤座標格點
    best, tried = None, 0
    cb, sb = math.cos(base[2]), math.sin(base[2])
    for zr in np.arange(0.85, 0.55, -0.025):
        for xr in np.arange(0.0, 0.46, 0.05):
            for yr in np.arange(0.0, 0.46, 0.05):
                cw = np.array([base[0] + cb * xr - sb * yr, base[1] + sb * xr + cb * yr, zr])
                Tc = look_T(cw, LOOK_AT)
                T_tcp = Tc @ np.linalg.inv(T_mount) @ T_eef_tcp
                tried += 1
                ok, w_, qa, pe, re, marg = RS.ik(base, T_tcp, [q_ready[3:]], iters=150)
                if not ok or marg < JOINT_MARGIN:
                    continue
                q = np.r_[base, qa]
                good = True
                for tz in TABLE_Z:
                    cl = clearance(q, tz)
                    if cl['env'] < CLEAR or cl['self'] < 0.00839:
                        good = False
                        break
                if not good:
                    continue
                path_ok, worst = True, dict(env=9.9, self=9.9, margin=9.9)
                for s in np.linspace(0.0, 1.0, 21):
                    qs = np.r_[base, q_ready[3:] + (qa - q_ready[3:]) * s]
                    for tz in TABLE_Z:
                        cl = clearance(qs, tz)
                        worst = dict(env=min(worst['env'], cl['env']), self=min(worst['self'], cl['self']), margin=min(worst['margin'], cl['margin']))
                    if worst['env'] < CLEAR or worst['self'] < 0.00839 or worst['margin'] < JOINT_MARGIN:
                        path_ok = False
                        break
                if not path_ok:
                    continue
                Tc_real = cam_T(q, T_mount)
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -Tc_real[2, 2]))))
                key = (round(float(Tc_real[2, 3]), 3), pitch)
                if best is None or key > best['key']:
                    best = dict(key=key, q=q, cam=Tc_real, target_cam=[float(v) for v in cw], path_worst=worst, pe=float(pe), re=float(re))
        if best is not None and zr < best['key'][0] - 0.05:
            break
    out = dict(src_run=SRC_RUN, t_obs=T_OBS, base_obs=base.tolist(), q_ready=q_ready[3:].tolist(), look_at=LOOK_AT.tolist(), table_z=TABLE_Z, bottles=BOTTLES, n_tried=tried,
               approach_clear=appr)
    if best is None:
        out['q_look'] = None
        print('找不到滿足約束的 Q_LOOK')
    else:
        q = best['q']
        dq = float(np.abs(q[3:] - q_ready[3:]).max())
        T_tr = math.ceil(1.875 * dq / B.W_ARM['v_arm'] / 0.05) * 0.05
        lay = {}
        for tz in TABLE_Z:
            for xy in BOTTLES:
                lay[f'{tz:.2f}_{xy[0]:.2f}_{xy[1]:.2f}'] = dict(look=visibility(best['cam'], xy, tz), clear_look=clearance(q, tz))
        out.update(q_look=q[3:].tolist(), cam_pos=best['cam'][:3, 3].tolist(), cam_pitch_deg=best['key'][1], target_cam=best['target_cam'], ik_err=[best['pe'], best['re']],
                   transition=dict(dq_max=dq, T_tr=T_tr, path_worst=best['path_worst']), layouts=lay)
        print(f"Q_LOOK {np.round(q[3:], 3).tolist()}；相機 {np.round(best['cam'][:3, 3], 3).tolist()}、俯角 {best['key'][1]:.1f}°（試 {tried} 個）；過渡 Δq {dq:.3f} rad → {T_tr:.2f} s；路徑最差 {best['path_worst']}")
        for k_, v_ in lay.items():
            lk = v_['look']
            print(f"  桌 {k_}：瓶身在影像 {lk['body_in_view']:.2f}、相機高於桌面 {lk['cam_height_over_table']:+.3f} m、瓶頂外緣背景 {lk['top_background']}、支撐環在影像 {lk['support_ring_in_view']:.2f}、距離 {lk['dist']:.2f} m；"
                  f"觀察姿態對環境 {v_['clear_look']['env'] * 1e3:.0f} mm")
    print('接近段（紀錄 0→14.8 s）最小：' + '；'.join(f"桌 {k_}：環境 {v_['env'] * 1e3:.0f} mm、自碰 {v_['self'] * 1e3:.1f} mm、關節餘裕 {v_['margin']:.3f}" for k_, v_ in appr.items()))
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
