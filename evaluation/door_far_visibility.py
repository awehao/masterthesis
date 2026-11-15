#!/usr/bin/env python3
"""遠距連續接近取像設計的可見性核對（Codex 20261116：(c)；幾何層，不核遮擋、不跑模擬）。
用 door_far_route_check 選出的新分支與進線姿態：各區段取樣機器人構型，算手腕相機（camera_color_optical_frame；內參取 online_1 實測）下把手估計中心的像素與距離。
區段：A 遠距導航（手臂維持起點構型，底盤由起點到進線點）；B 手臂準備（關節直線）；C 沿線到窗口起點（手臂停在窗口起點解）；D 追蹤窗口（連續 IK）；E 插入。
有效影像的幾何條件：中心在畫面內（留 40 px 邊）、深度 >0、距離在 [0.25, 1.20] m（既有偵測的距離範圍 D_RNG）。底盤相機（URDF 有 base_camera_link）不在目前的運動學模型內（fk 回單位矩陣），本檔未核——列為缺口。
    python3 evaluation/door_far_visibility.py ROUTE_JSON OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_motm_grasp_ctrl as G
import door_fast_kin as DFK
import door_far_link as FL
import door_far_route as FR
K = G.K
R = json.load(open(sys.argv[1]))
RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1')
meta = json.load(open(os.path.join(RUN, 'frames_meta.json')))
fr = [f for f in meta['frames'] if f['stem'] == 'far_live'][0]
Kc = np.asarray(meta['K'], float); Wd, Hd = 640, 480
obs = FL.HandleObs()
T_est, _ = obs.intake(dict(dep=np.load(os.path.join(RUN, 'frames', 'far_live_depth_m.npy')), K=Kc, T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float)), float(fr['t_obs']) + 0.01)
Rg, T_HG, a_H = G.PG.baseline_reference()
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE = float(G.PG.S_PRE)
ch, mg = R['select']['chosen'], R['merge']
q0 = np.asarray(fr['q'], float)
p_in, yaw = np.array(mg['p_in']), math.radians(mg['yaw_deg'])
p_g = np.array(ch['p0']); d = (p_g - p_in) / np.linalg.norm(p_g - p_in)
qw, qp = np.array(ch['q_window_start']), np.array(ch['q_pregrasp'])
hc = T_est[:3, 3]


def view(q9, cam='camera_color_optical_frame'):
    Tc = K.fk(q9, cam)
    pc = np.linalg.inv(Tc) @ np.r_[hc, 1.0]
    z = float(pc[2]); dist = float(np.linalg.norm(pc[:3]))
    if z <= 0:
        return dict(in_view=False, why='behind_camera', dist=round(dist, 3))
    u, v = float(Kc[0, 0] * pc[0] / z + Kc[0, 2]), float(Kc[1, 1] * pc[1] / z + Kc[1, 2])
    inimg = 40 <= u <= Wd - 40 and 40 <= v <= Hd - 40
    rng = 0.25 <= dist <= 1.20
    return dict(in_view=bool(inimg and rng), u=round(u, 1), v=round(v, 1), dist=round(dist, 3), why=None if (inimg and rng) else ('out_of_image' if not inimg else 'out_of_range'))


segs = {}
A = [np.r_[q0[:2] + u_ * (p_in - q0[:2]), q0[2] + u_ * FR._wrap(yaw - q0[2]), q0[3:]] for u_ in np.linspace(0, 1, 11)]
B = [np.r_[p_in, yaw, q0[3:] + u_ * FR._wrap(qw - q0[3:])] for u_ in np.linspace(0, 1, 9)]
C = [np.r_[p_in + d * al, yaw, qw] for al in np.linspace(0, FR.L_IN - FR.W_MAX, 8)]
Dq, seed = [], qw
for al in np.arange(FR.W_MAX, -1e-9, -FR.TRACK_STEP):
    b = np.r_[p_g - d * al, yaw]
    qq, r_, _ = DFK.ik_closure_fc(T_at(T_est, S_PRE), b, seed)
    Dq.append(np.r_[b, qq]); seed = qq
E, seed = [], qp
for s_ in np.linspace(S_PRE, 0.0, 5):
    qq, r_, _ = DFK.ik_closure_fc(T_at(T_est, s_), np.r_[p_g, yaw], seed)
    E.append(np.r_[p_g, yaw, qq]); seed = qq
out = dict(note='幾何可見性（不核遮擋）；把手中心用起點實際影像的估計；手腕相機內參取 online_1 實測', K=Kc.tolist(), segments={})
for nm, S in (('A_far_nav_arm_hold', A), ('B_arm_prep_at_entry', B), ('C_along_line_arm_at_window_start', C), ('D_tracking_window', Dq), ('E_insert', E)):
    V = [view(x) for x in S]
    out['segments'][nm] = dict(n=len(V), n_in_view=sum(v_['in_view'] for v_ in V), dist_range=[min(v_['dist'] for v_ in V), max(v_['dist'] for v_ in V)], why=sorted({v_['why'] for v_ in V if v_['why']}), samples=V[::max(len(V) // 4, 1)])
    print(nm, out['segments'][nm]['n_in_view'], '/', len(V), out['segments'][nm]['dist_range'], out['segments'][nm]['why'])
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
