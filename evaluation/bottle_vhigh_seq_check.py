#!/usr/bin/env python3
"""移動接近中的高位手腕視角（V_HIGH）參考序列：最小離線核對（Howard／Codex reviews/20261102_howard_moving_gaze_no_parked_grasp.md 第 1、2、4 項；純運動學、無 Isaac）。
底盤：名目趟 bottle_w1_segtarget_nominal1 的實測底盤軌跡（真實接近路徑：x_lane −0.9 之前在規劃曲線上，y 約 −0.8→0，不是 y＝0 車道），既有速度設定（v_nav 0.35、x_slow −0.90 後 v_c 0.10）。
手臂序列（底盤座標關節參考，不跳到封存關節角，全部由目前關節以 min-jerk 連續過渡）：
  U  K2_READY（q_tuck）→ V_HIGH，時長 T_U＝1.875·max|Δq|/v_arm；結束時刻由可見窗開始倒推（再留 T_SETTLE）。
  H  保持 V_HIGH 觀察，直到漸變起點 x_blend（−0.60，原抓取目標取得期限）。
  B  轉入預抓，兩案：
     A 直接漸變：TCP 由 V_HIGH 的 TCP（隨底盤）min-jerk 到 T_pre（世界），t_blend 3 s（原設定），IK 延續 0.05 s 一步；
     B 先關節直線 V_HIGH→名目備妥（T_VR），再走原漸變（備妥→T_pre）。B 的觀察窗要提早 T_VR 結束。
每一取樣：可見性（CD.visibility：瓶身入畫、支撐環入畫、相機高於桌面；另算瓶身投影到影像邊界的最小像素餘裕），淨空（第一桌 0.60／0.75、第二桌、兩瓶位瓶身 ≥ 0.05 m 用 CD.clearance；
  自碰 ≥ 8.39 mm；關節餘裕 ≥ 0.05），|j5|（手腕奇異），關節速度（差分）／加速度（二階差分）對 v_arm、A_ARM。
停止路徑：關節直線段停在路徑上任一點都已涵蓋；漸變段停止構型近似為取樣點（減速距離 v²/2A_ARM ≤ 0.03 rad，另列）。
第 4 項：0.75 m 量測目標下，同一分支在底盤 x 上連續可行（預抓、插入線、抓取、抬升線全部 IK 延續）的最長區間 L，對插入＋閉爪＋證據＋抬升＋穩定所需時長 T_MANIP 的最高底盤速度 L/T_MANIP（與 v_c 比）。
輸出 results/vision/vhigh_seq_check.json。投影入畫比例不是估計成功率；遮擋與遮罩品質要另以渲染核對。
    python3 evaluation/bottle_vhigh_seq_check.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_branch as BB                              # noqa: E402
import bottle_ref as BRF                                # noqa: E402
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_segtarget_plan as SP                      # noqa: E402
import bottle_seg3d_collect_design as CD                # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import wrist_v0_capture as W0                           # noqa: E402

SRC = 'bottle_w1_segtarget_nominal1'
Q_TUCK = np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
X_BLEND, X_PRE, T_BLEND, DT = -0.60, -0.30, 3.0, 0.05
T_SETTLE, BOTTLE = 0.5, (0.0, 0.42)
VIS_BODY, VIS_RING = 0.90, 0.80
T_OBS_MEAS = 1.5          # 實測：bottle_w1_segtarget_obs_z075_p1_1 手臂在俯視點停穩（LOOK 25.05 s）到合格 known_type 候選（26.55 s）的時間（n=1）
T_OBS_NEED = 2.0 * T_OBS_MEAS   # 觀察窗需求：實測的兩倍（工程餘量，事前定、不依結果調）
OBST = (np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))   # 場景靜態方塊（名目趟 s1_report obstacle）
T_MANIP = 1.5 + B.T_CLOSE + B.T_EV + B.T_LIFT + B.T_LIFT_SETTLE      # 插入 t_insert 1.5＋閉爪＋證據＋抬升＋穩定（同 heuristic_speed 的時間項）
TABLES = (0.60, 0.75)


def base_traj():
    Z = np.load(os.path.join(HERE, 'runs', SRC, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    return c['t'], np.column_stack([c['base_x'], c['base_y'], c['base_yaw']])


def margin_px(Tc, xy, tz):
    r, h = G.BOTTLE['r'], G.BOTTLE['h']
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    P = np.array([[xy[0] + r * math.cos(t), xy[1] + r * math.sin(t), tz + z] for z in np.linspace(0, h, 6) for t in a])
    ok, u, v = CD.project(Tc, P)
    if not ok.any():
        return -1.0
    return float(min(u[ok].min(), CD.W_ - 1 - u[ok].max(), v[ok].min(), CD.H_ - 1 - v[ok].max()))


def clear_all(q9, tz):
    """桌（第一桌 tz、第二桌）、場景方塊、P1 瓶身（場景只有 P1）分開回報；自碰（開爪）、關節餘裕、|j5|。"""
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    tb = [(c_, s_) for _, c_, s_ in CD.table_boxes(tz)] + [(c_, s_) for _, c_, s_ in TP.table2_boxes()]
    bc, bs = CD.bottle_box(BOTTLE, tz)
    sc = G.C.self_clear(q9, G.Q_OPEN)[0]
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for c_, s_ in tb), obst=float(G.box_dist(P, *OBST).min()), bottle=float(G.box_dist(P, bc, bs).min()),
                self=float(sc), margin=BB.margin(q9[3:]), sing=abs(float(q9[7])))


def jline(qa, qb, s):
    return qa + (qb - qa) * BRF.min_jerk(s)


def rates(Q, dt):
    Q = np.asarray(Q, float)
    v = np.abs(np.diff(Q, axis=0)).max(axis=1) / dt if len(Q) > 1 else np.zeros(1)
    a = np.abs(np.diff(Q, 2, axis=0)).max(axis=1) / dt ** 2 if len(Q) > 2 else np.zeros(1)
    return float(v.max()), float(a.max())


def main():
    L = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_view_library.json')))
    q_vh = np.array(L['views']['V_HIGH']['q'])
    D = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))
    q_ready = np.array(D['q_ready'])                       # 名目備妥（名目趟 14.8 s 實測，0.60 分支）
    T_mount = W0.urdf_chain_T(CD.URDF, 'link_eef', 'camera_color_optical_frame')
    t_all, base_all = base_traj()
    bx = lambda t: np.array([float(np.interp(t, t_all, base_all[:, i])) for i in range(3)])   # noqa: E731
    t_blend = float(np.interp(X_BLEND, base_all[:, 0], t_all))
    out = dict(src=SRC, q_vhigh=q_vh.tolist(), q_ready=q_ready.tolist(), t_x_blend=t_blend, T_manip=T_MANIP)
    # 1 V_HIGH 沿真實接近軌跡的可見窗（固定手臂姿態）
    vis = []
    for t in np.arange(6.0, t_blend + 3.0 + 1e-9, DT):
        b = bx(t)
        Tc = CD.cam_T(np.r_[b, q_vh], T_mount)
        row = dict(t=round(float(t), 3), x=round(float(b[0]), 4), y=round(float(b[1]), 4))
        d_ = float(np.linalg.norm(Tc[:3, 3] - np.array([BOTTLE[0], BOTTLE[1], 0.75 + 0.10])))
        row['dist_075'], row['px_h_075'] = round(d_, 3), round(CD.FX * G.BOTTLE['h'] / d_, 1)
        for tz in TABLES:
            v = CD.visibility(Tc, BOTTLE, tz)
            row[f'ok_{tz:.2f}'] = bool(v['body_in_view'] >= VIS_BODY and v['support_ring_in_view'] >= VIS_RING and v['cam_above_table'])
            row[f'body_{tz:.2f}'], row[f'ring_{tz:.2f}'], row[f'px_{tz:.2f}'] = round(v['body_in_view'], 3), round(v['support_ring_in_view'], 3), round(margin_px(Tc, BOTTLE, tz), 1)
        vis.append(row)
    win = {}
    for tz in TABLES:
        k = f'ok_{tz:.2f}'
        seg, cur = [], None
        for r in vis:
            if r[k] and cur is None:
                cur = [r['t'], r['t'], r['x'], r['x']]
            elif r[k]:
                cur[1], cur[3] = r['t'], r['x']
            elif cur is not None:
                seg.append(cur)
                cur = None
        if cur is not None:
            seg.append(cur)
        win[f'{tz:.2f}'] = [dict(t0=s[0], t1=s[1], x0=s[2], x1=s[3]) for s in seg]
    out['vhigh_windows'] = win
    w75 = next((s for s in win['0.75'] if s['t1'] >= t_blend - 1.0), (win['0.75'] or [None])[-1])
    # 2 展臂 U：倒推起點
    dq_u = float(np.abs(q_vh - Q_TUCK).max())
    T_U = math.ceil(1.875 * dq_u / B.W_ARM['v_arm'] / DT) * DT
    dq_vr0 = float(np.abs(q_ready - q_vh).max())
    T_VR0 = math.ceil(1.875 * dq_vr0 / B.W_ARM['v_arm'] / DT) * DT
    t_v0 = (t_blend - T_VR0) - T_OBS_NEED                     # B 案：離開 V_HIGH 前要有 T_OBS_NEED 的觀察窗 ⇒ 到達並停穩的最晚時刻
    t_u0 = t_v0 - T_SETTLE - T_U
    self_floor = min(G.C.self_clear(np.r_[0.0, 0.0, 0.0, Q_TUCK + (q_ready - Q_TUCK) * s_], G.Q_OPEN)[0] for s_ in np.linspace(0.0, 1.0, 41))
    out['self_floor_open'] = self_floor                      # 已驗證名目展臂（K2_READY→名目備妥，物理多趟）開爪自碰最小值；8.39 mm 是閉爪值
    out['unfold'] = dict(dq_max=dq_u, T_U=T_U, t_start=round(t_u0, 3), t_need_at_vhigh=round(t_v0, 3), x_need_at_vhigh=round(float(bx(t_v0)[0]), 4), x_start=round(float(bx(t_u0)[0]), 4), y_start=round(float(bx(t_u0)[1]), 4), t_at_vhigh=round(t_u0 + T_U, 3),
                         note='原連續展臂起點 x_unfold −1.55（名目趟 8.25 s）')
    # 3 序列取樣（A、B 兩案），0.60 與 0.75 目標
    dq_vr = float(np.abs(q_ready - q_vh).max())
    T_VR = math.ceil(1.875 * dq_vr / B.W_ARM['v_arm'] / DT) * DT
    res = {}
    for tz in TABLES:
        P = SP.poses(BOTTLE[0], BOTTLE[1], tz)
        for case in ('A', 'B'):
            t_dep = t_blend if case == 'A' else t_blend - T_VR
            Q, T = [], []
            for t in np.arange(t_u0, t_u0 + T_U + 1e-9, DT):              # U
                Q.append(np.r_[bx(t), jline(Q_TUCK, q_vh, (t - t_u0) / T_U)])
                T.append(t)
            for t in np.arange(t_u0 + T_U + DT, t_dep + 1e-9, DT):         # H
                Q.append(np.r_[bx(t), q_vh])
                T.append(t)
            q_from = q_vh
            if case == 'B':
                for t in np.arange(t_dep + DT, t_blend + 1e-9, DT):
                    Q.append(np.r_[bx(t), jline(q_vh, q_ready, (t - t_dep) / T_VR)])
                    T.append(t)
                q_from = q_ready
            T_rel = C0.S0_K().fk(np.r_[0.0, 0.0, 0.0, q_from], 'link_tcp') if hasattr(C0, 'S0_K') else G.C.K.fk(np.r_[0.0, 0.0, 0.0, q_from], 'link_tcp')
            q, ik_fail = q_from.copy(), None
            for t in np.arange(t_blend + DT, t_blend + T_BLEND + 1e-9, DT):  # 漸變（底盤用名目趟實測軌跡）
                b = bx(t)
                Tt = BRF.interp_pose(BRF.pose_world(b, T_rel), P['pre'], BRF.min_jerk((t - t_blend) / T_BLEND))
                ok, _, qn, pe, re, m = RS.ik(b, Tt, [q], iters=200)
                if not ok or float(np.abs(qn - q).max()) > SP.MAX_STEP:
                    ik_fail = dict(t=round(float(t), 3), ok=bool(ok), jump=round(float(np.abs(qn - q).max()), 3), margin=round(float(m), 3))
                    break
                Q.append(np.r_[b, qn])
                T.append(t)
                q = qn
            Qa = np.array(Q)
            wst = {ph: dict(table=9.9, obst=9.9, bottle=9.9, self=9.9, margin=9.9, sing=9.9) for ph in ('view', 'blend')}
            where = {ph: {} for ph in wst}
            for i in range(0, len(Qa), 2):
                ph = 'blend' if T[i] > t_blend + 1e-9 else 'view'
                w = clear_all(Qa[i], tz)
                for k_ in wst[ph]:
                    if w[k_] < wst[ph][k_]:
                        wst[ph][k_], where[ph][k_] = w[k_], round(float(T[i]), 2)
            lim = dict(view=dict(table=CD.CLEAR, obst=CD.CLEAR, bottle=CD.CLEAR), blend=dict(table=SP.CLEAR_TRUTH, obst=CD.CLEAR, bottle=SP.CLEAR_BOTTLE))
            fails = [f'{ph}:{k_}' for ph in wst for k_, v_ in wst[ph].items() if (k_ in lim[ph] and v_ < lim[ph][k_]) or (k_ == 'self' and v_ < self_floor - 1e-6) or (k_ == 'margin' and v_ < 0.05)]
            v_max, a_max = rates(Qa[:, 3:], DT)
            obs_t0 = max(t_v0, t_u0 + T_U + T_SETTLE)
            vis_obs = [r for r in vis if obs_t0 <= r['t'] <= t_dep and r[f'ok_{tz:.2f}']]
            res[f'{tz:.2f}_{case}'] = dict(t_depart=round(t_dep, 3), x_depart=round(float(bx(t_dep)[0]), 4), obs_window_s=round(len(vis_obs) * DT, 2),
                                           obs_x=[vis_obs[0]['x'], vis_obs[-1]['x']] if vis_obs else None,
                                           px_margin_min=min((r[f'px_{tz:.2f}'] for r in vis_obs), default=None), blend_ik_fail=ik_fail, worst=wst, worst_t=where,
                                           v_max=round(v_max, 3), a_max=round(a_max, 2), v_lim=B.W_ARM['v_arm'], a_lim=C0.A_ARM, fails=fails + ([] if v_max <= B.W_ARM['v_arm'] + 1e-3 else ['v_arm']),
                                           dist_obs=[min((r['dist_075'] for r in vis_obs), default=None), max((r['dist_075'] for r in vis_obs), default=None)],
                                           px_h_obs=[min((r['px_h_075'] for r in vis_obs), default=None), max((r['px_h_075'] for r in vis_obs), default=None)],
                                           ok=bool(ik_fail is None and not fails and v_max <= B.W_ARM['v_arm'] + 1e-3))
    out['T_VR'], out['sequences'] = T_VR, res
    # 4 0.75 量測目標的移動進度剖面（同分支連續區間）
    P = SP.poses(BOTTLE[0], BOTTLE[1], 0.75)
    path = SP.grasp_path(P)
    xs = np.round(np.arange(-0.35, 0.15 + 1e-9, 0.01), 4)
    best = None
    for i0, x0 in enumerate(xs):
        for s in SP._seeds():
            ok, _, qp, *_r = RS.ik(np.array([x0, 0.0, 0.0]), P['pre'], [s])
            if not ok:
                continue
            bg = SP.Budget(max_ik=10 ** 6, max_s=1e9)
            qs, _m = SP._chain(bg, [x0, 0.0, 0.0], path, qp)
            if qs is None:
                continue
            cur, j = qs, i0
            while j + 1 < len(xs):
                nxt = []
                okn = True
                for qa, Tp in zip(cur, path):                        # 每個路徑位姿沿 x 延續（同 window_branch 的掃描）
                    g = RS.ik(np.array([xs[j + 1], 0.0, 0.0]), Tp, [qa])
                    if not g[0] or g[5] < SP.MIN_MARGIN or float(np.abs(g[2] - qa).max()) > BB.MAX_STEP:
                        okn = False
                        break
                    nxt.append(g[2])
                if not okn:
                    break
                cur, j = nxt, j + 1
            span = (float(xs[i0]), float(xs[j]))
            if best is None or span[1] - span[0] > best['L']:
                best = dict(L=round(span[1] - span[0], 4), x0=span[0], x1=span[1], q5_pre=round(float(qp[4]), 3))
        if best is not None and best['L'] >= xs[-1] - x0:
            break
    out['reach_075'] = dict(best, v_max=round(best['L'] / T_MANIP, 4) if best else None, v_c=0.10, T_manip=T_MANIP,
                            note='同一分支在 x 上連續可行的最長區間（預抓、插入線、抓取、抬升線全部延續）；移動抓取在插入到抬升穩定期間底盤要留在區間內 ⇒ 最高速度 L/T_manip')
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'vhigh_seq_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"x_blend 時刻 {t_blend:.2f} s；V_HIGH 可見窗 {win}")
    print(f"展臂 U：Δq {dq_u:.3f} rad → {T_U:.2f} s；需在 t {t_v0:.2f}（x {out['unfold']['x_need_at_vhigh']}）停穩於 V_HIGH ⇒ 起點 t {t_u0:.2f}（x {out['unfold']['x_start']}, y {out['unfold']['y_start']}）；V_HIGH→備妥 {T_VR:.2f} s；開爪自碰下限 {self_floor * 1e3:.2f} mm")
    for k, v in res.items():
        fw = lambda w: {k_: round(v_ * (1e3 if k_ not in ('margin', 'sing') else 1), 1 if k_ not in ('margin', 'sing') else 3) for k_, v_ in w.items()}   # noqa: E731
        print(f"  {k}：離開 V_HIGH t {v['t_depart']}（x {v['x_depart']}）觀察窗 {v['obs_window_s']} s x {v['obs_x']} 距離 {v['dist_obs']} m 瓶高 {v['px_h_obs']} px 邊界餘裕 {v['px_margin_min']} px；漸變 IK {v['blend_ik_fail'] or 'ok'}；"
              f"觀察段 {fw(v['worst']['view'])}；漸變段 {fw(v['worst']['blend'])}（mm／rad）；速度峰 {v['v_max']} 加速度峰 {v['a_max']}；不過 {v['fails']}；ok {v['ok']}")
    print(f"0.75 移動可達：{out['reach_075']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
