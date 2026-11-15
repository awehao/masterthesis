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


import wrist_v0_capture as W0
import bottle_v1_cam as BV
URDF = os.path.join(HERE, 'models', 'omni_bot_manip.urdf')
T_BASE_MOUNT = W0.urdf_chain_T(URDF, BV.PARENT, BV.OPT)              # 既有底盤 RGB-D（bottle_v1_cam）：URDF 合成的 base_link → base_camera_color_optical_frame
fb = (BV.W_ / 2.0) / math.tan(BV.HFOV / 2.0)
K_BASE = np.array([[fb, 0.0, BV.W_ / 2.0], [0.0, fb, BV.H_ / 2.0], [0.0, 0.0, 1.0]])   # 由 URDF 水平視角與解析度算的名目內參（實跑須讀回）
CAMS = dict(wrist=dict(K=Kc, rng=(0.25, 1.20), T=lambda q9: K.fk(q9, 'camera_color_optical_frame')),
            base=dict(K=K_BASE, rng=BV.DEPTH_RANGE, T=lambda q9: K.fk(q9, 'base_link') @ T_BASE_MOUNT))
bar = T_est[:3, 0] / np.linalg.norm(T_est[:3, 0])
ENDS = [hc - 0.1 * bar, hc + 0.1 * bar]                              # 橫桿兩端（長 200 mm；偵測需要端點可辨識）
DOC = G.W.DOC


def proj(Tc, Kc_, p):
    pc = np.linalg.inv(Tc) @ np.r_[p, 1.0]
    if pc[2] <= 0:
        return None
    return float(Kc_[0, 0] * pc[0] / pc[2] + Kc_[0, 2]), float(Kc_[1, 1] * pc[1] / pc[2] + Kc_[1, 2]), float(pc[2])


def self_occluded(q9, c_pos, target, rad=0.02):
    """相機到目標的線段是否被手臂／夾爪的碰撞點雲擋住（點到線段距離 < rad；只算目標前方 3 cm 以外的部分）。"""
    v = target - c_pos
    L_ = float(np.linalg.norm(v)); v = v / L_
    for n in DOC.ARM_LINKS:
        T = K.fk(q9, n)
        P = (T[:3, :3] @ DOC.CL[n].T).T + T[:3, 3]
        s_ = (P - c_pos) @ v
        m = (s_ > 0.05) & (s_ < L_ - 0.03)
        if m.any() and float(np.linalg.norm((P[m] - c_pos) - np.outer(s_[m], v), axis=1).min()) < rad:
            return n
    return None


def view(q9, cam='wrist'):
    C_ = CAMS[cam]
    Tc = C_['T'](q9)
    pr = [proj(Tc, C_['K'], p_) for p_ in [hc] + ENDS]
    dist = float(np.linalg.norm(hc - Tc[:3, 3]))
    if pr[0] is None:
        return dict(in_view=False, why='behind_camera', dist=round(dist, 3))
    ok_img = all(p_ is not None and 20 <= p_[0] <= 640 - 20 and 20 <= p_[1] <= 480 - 20 for p_ in pr)
    ok_rng = C_['rng'][0] <= pr[0][2] <= C_['rng'][1]
    occ = self_occluded(q9, Tc[:3, 3], hc) if (ok_img and ok_rng) else None
    why = None if (ok_img and ok_rng and occ is None) else ('feature_out_of_image' if not ok_img else ('out_of_depth_range' if not ok_rng else 'self_occluded:' + occ))
    px_len = None if (pr[1] is None or pr[2] is None) else round(math.hypot(pr[1][0] - pr[2][0], pr[1][1] - pr[2][1]), 1)
    return dict(in_view=why is None, u=round(pr[0][0], 1), v=round(pr[0][1], 1), depth=round(pr[0][2], 3), dist=round(dist, 3), bar_px=px_len, why=why)


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
out = dict(note='幾何可見性：把手中心與橫桿兩端都在畫面內（留 20 px）、深度在各相機有效範圍、相機到把手中心的視線未被手臂／夾爪碰撞點雲擋住（2 cm）；門框／門板遮擋與偵測是否成立未核。把手位姿＝起點實際影像估計',
           cameras=dict(wrist=dict(K=Kc.tolist(), depth_range=[0.25, 1.20], source='online_1 實測內參'), base=dict(K=K_BASE.tolist(), depth_range=list(BV.DEPTH_RANGE), T_mount=T_BASE_MOUNT.tolist(), source='bottle_v1_cam 的 URDF 掛載與名目內參（實跑須讀回）')), segments={})
for nm, S in (('A_far_nav_arm_hold', A), ('B_arm_prep_at_entry', B), ('C_along_line_arm_at_window_start', C), ('D_tracking_window', Dq), ('E_insert', E)):
    rec = {}
    for cam in ('wrist', 'base'):
        V = [view(x, cam) for x in S]
        rec[cam] = dict(n=len(V), n_in_view=sum(v_['in_view'] for v_ in V), depth_range=[min(v_.get('depth', v_['dist']) for v_ in V), max(v_.get('depth', v_['dist']) for v_ in V)], why=sorted({v_['why'] for v_ in V if v_['why']}),
                        bar_px=[v_.get('bar_px') for v_ in V][::max(len(V) // 3, 1)], samples=V[::max(len(V) // 3, 1)])
    out['segments'][nm] = rec
    print(nm, '| wrist', rec['wrist']['n_in_view'], '/', rec['wrist']['n'], rec['wrist']['why'], '| base', rec['base']['n_in_view'], '/', rec['base']['n'], rec['base']['depth_range'], rec['base']['why'], 'bar_px', rec['base']['bar_px'])
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)

# ---- 觀察姿態的最小調整取證：TCP 沿接近軸後退 s（抓取點底盤位置、同一 IK 分支連續解）時，腕部相機是否看得到把手中心與兩端
ret = []
seed = qp
for s_ in np.arange(S_PRE, 0.4501, 0.02):
    Tg_ = T_at(T_est, 0.0).copy()
    Tg_[:3, 3] += float(s_) * (T_est[:3, :3] @ np.asarray(a_H, float))     # 抓取位姿沿接近軸後退 s（object_target 的 s 上限較小，這裡直接平移）
    qq, r_, sm = DFK.ik_closure_fc(Tg_, np.r_[p_g, yaw], seed)
    if not (r_ < 1e-6):
        ret.append(dict(s=round(float(s_), 3), ik=False)); continue
    seed = qq
    v_ = view(np.r_[p_g, yaw, qq], 'wrist')
    ret.append(dict(s=round(float(s_), 3), ik=True, sigma=round(float(sm), 4), in_view=v_['in_view'], why=v_['why'], depth=v_.get('depth'), u=v_.get('u'), v=v_.get('v'), bar_px=v_.get('bar_px')))
out['wrist_view_vs_retreat'] = ret
ok_s = [r_['s'] for r_ in ret if r_.get('in_view')]
out['wrist_min_retreat_in_view'] = min(ok_s) if ok_s else None
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print('wrist vs retreat:', [(r_['s'], r_.get('in_view'), r_.get('why'), r_.get('depth'), r_.get('v')) for r_ in ret][::2])
print('base cam sample D:', out['segments']['D_tracking_window']['base']['samples'][0])
