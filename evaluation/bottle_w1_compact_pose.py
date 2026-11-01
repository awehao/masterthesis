#!/usr/bin/env python3
"""低、近身、直立的運送姿態設計 v2（加「從側持沿過渡路徑 IK 延續可達」條件；v1 只核終點 IK，選到手腕翻轉 2.7 rad 的分支，GMPC 局部追蹤未必走得到）。
過渡路徑：TCP 位置 min-jerk 直線、偏航 Rz(ψ·s)（瓶身鉛直），20 步；每步 IK 以前一步解為唯一種子，須收斂、滿足全部約束、相鄰步任一關節變化 ≤ 0.5 rad（不跳分支）。
候選先依不需 IK 的包絡半徑與瓶底高度排序（包絡只由 TCP 位姿決定），依序做延續核對，第一個通過者＝延續可達候選中的字典序最優。
以下為 v1 說明。
低、近身、直立的運送姿態設計（Howard 修正 reviews/20261101_howard_compact_carry_and_motion.md；設計工具，非實驗掃描）。
輸入：名目紀錄 bottle_w1_cont_nominal1 在 t_lift_ok 的側持關節與讀回瓶位姿 ⇒ 瓶相對 TCP 的位姿 T_rel、側持 TCP 底盤座標位姿 T_side。
候選：TCP 姿態＝Rz(ψ)·R_side（繞鉛直軸改偏航，瓶身仍鉛直），ψ ∈ {0, −30, −60, −90}°；TCP 底盤座標位置 x ∈ [−0.10, 0.45]、y ∈ [−0.35, 0.45]、z ∈ [0.40, 0.75] 每 0.05 m。
逐點阻尼最小平方 IK（bottle_reach_scan.ik，鄰點暖啟動）：位置誤差 < 1 mm、姿態 < 1°、關節餘裕 ≥ 0.05 rad（同 W_ARM margin）。
約束：自碰（閉爪）≥ 8.39 mm（已驗證側持與名目的最小值，不比它差）；持瓶對非夾爪連桿 ≥ 5.5 mm（既有瓶淨空拒絕線）；
  車體（碰撞圓柱半徑 0.30 m、頂面 z 0.33 m）：包絡點水平距底盤中心 < 0.32 m 時 z ≥ 0.38 m（頂面＋既有三維本體距離 0.05）；
目標：包絡（瓶身取樣點＋夾爪包圍盒 8 角，同 S1._obb_tcp 的連桿集合）相對底盤中心的最大水平半徑 R_env 最小；R_env 差 1 cm 內取瓶底最低，再取可操作度大。
輸出 results/bottle/w1_compact_pose.json（選定 TCP 底盤座標位姿、IK 關節、R_env、瓶底高度、各約束餘裕）。
    python3 evaluation/bottle_w1_compact_pose.py
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
import bottle_s1_geom as G                              # noqa: E402

K = G.C.K
Q_CLOSED = 0.0038
SELF_MIN, BOT_LINK_MIN = 0.00839, 0.0055
DECK_R, DECK_Z = 0.32, 0.38
GRIP = ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


def quat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rz(a):
    R = np.eye(4)
    R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return R


def reference(run='bottle_w1_cont_nominal1'):
    r = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    k = int(np.argmin(np.abs(c['t'] - r['t_lift_ok'])))
    qb = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]])
    qa = np.array([c[f'q{j}'][k] for j in range(1, 7)])
    Tt = K.fk(np.r_[qb, qa], 'link_tcp')
    Tb = np.eye(4)
    Tb[:3, :3] = quat(c['bottle_qw'][k], c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k])
    Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
    T_rel = np.linalg.inv(Tt) @ Tb
    T_side = K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
    return qa, T_rel, T_side, float(r['t_lift_ok'])


def grip_corners(qa_ready):
    q0 = np.r_[0.0, 0.0, 0.0, qa_ready]
    Ti = np.linalg.inv(K.fk(q0, 'link_tcp'))
    W = G.link_world(q0, G.Q_OPEN)
    P = np.vstack([v for k, v in W.items() if k in GRIP])
    Pt = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    lo, hi = Pt.min(axis=0), Pt.max(axis=0)
    return np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def envelope(T_tcp, T_rel, GC):
    BP = G.bottle_points()
    Tb = T_tcp @ T_rel
    Pb = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
    Pg = (T_tcp[:3, :3] @ GC.T).T + T_tcp[:3, 3]
    return Pb, Pg


def evaluate(qa, T_rel, GC):
    q = np.r_[0.0, 0.0, 0.0, qa]
    Tt = K.fk(q, 'link_tcp')
    Pb, Pg = envelope(Tt, T_rel, GC)
    P = np.vstack([Pb, Pg])
    rad = np.hypot(P[:, 0], P[:, 1])
    deck = np.where(rad < DECK_R, P[:, 2] - DECK_Z, 1.0)
    sc, who = G.C.self_clear(q, Q_CLOSED)
    W = G.link_world(q, Q_CLOSED)
    bl = min(float(G.cyl_dist((np.linalg.inv(Tt @ T_rel) @ np.c_[P_, np.ones(len(P_))].T).T[:, :3], 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min())
             for n_, P_ in W.items() if n_ not in GRIP and n_ != 'uflite_gripper_link')
    return dict(R_env=float(rad.max()), R_bottle=float(np.hypot(Pb[:, 0], Pb[:, 1]).max()), bottle_bottom=float(Pb[:, 2].min()), env_zmin=float(P[:, 2].min()),
                deck_margin=float(deck.min()), self=float(sc), self_who=str(who), bottle_link=float(bl), tcp=Tt[:3, 3].tolist())


def continuation(qa0, T_side, T_c, psi, T_rel, GC, n=20):
    q, path = qa0.copy(), []
    p0, p1 = T_side[:3, 3], T_c[:3, 3]
    for k in range(1, n + 1):
        s_ = k / n
        T = np.eye(4)
        T[:3, :3] = (rz(math.radians(psi) * s_) @ T_side)[:3, :3]
        T[:3, 3] = p0 + (p1 - p0) * s_
        ok, w, qn, pe, re, marg = RS.ik(np.zeros(3), T, [q], iters=60)
        if not ok or float(np.abs(qn - q).max()) > 0.5:
            return None, dict(step=k, why='ik' if not ok else 'branch_jump', jump=float(np.abs(qn - q).max()))
        ev = evaluate(qn, T_rel, GC)
        if ev['self'] < SELF_MIN or ev['bottle_link'] < BOT_LINK_MIN or ev['deck_margin'] < 0:
            return None, dict(step=k, why='constraint', self=ev['self'], bottle_link=ev['bottle_link'], deck=ev['deck_margin'])
        path.append(dict(s=s_, q=qn.tolist(), self=ev['self'], bottle_link=ev['bottle_link'], deck=ev['deck_margin'], margin=float(marg)))
        q = qn
    return path, None


def main():
    qa0, T_rel, T_side, t_ok = reference()
    GC = grip_corners(qa0)
    base_eval = evaluate(qa0, T_rel, GC)
    print(f"側持參考（t_lift_ok {t_ok}）：R_env {base_eval['R_env']:.3f} m、瓶底 {base_eval['bottle_bottom']:.3f} m、自碰 {base_eval['self'] * 1e3:.1f} mm")
    xs, ys, zs = np.arange(-0.10, 0.4501, 0.05), np.arange(-0.35, 0.4501, 0.05), np.arange(0.40, 0.7501, 0.05)
    cands = []
    for psi in (0.0, -30.0, -60.0, -90.0):
        R = (rz(math.radians(psi)) @ T_side)[:3, :3]
        for x in xs:
            for y in ys:
                for z in zs:
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = [x, y, z]
                    Pb, Pg = envelope(T, T_rel, GC)
                    P = np.vstack([Pb, Pg])
                    rad = np.hypot(P[:, 0], P[:, 1])
                    if np.where(rad < DECK_R, P[:, 2] - DECK_Z, 1.0).min() < 0:
                        continue
                    cands.append((round(float(rad.max()) / 0.01), float(Pb[:, 2].min()), psi, T))
    cands.sort(key=lambda c_: (c_[0], c_[1]))
    print(f"候選（車體頂淨空先過）{len(cands)}，依包絡半徑等級與瓶底排序後逐一做延續核對")
    tried, sel, fails = 0, None, {}
    for grade, bb, psi, T in cands:
        tried += 1
        path, why = continuation(qa0, T_side, T, psi, T_rel, GC)
        if path is None:
            fails[why['why']] = fails.get(why['why'], 0) + 1
            continue
        qe = np.array(path[-1]['q'])
        ev = evaluate(qe, T_rel, GC)
        dq = np.abs(qe - qa0)
        sel = dict(psi_deg=psi, tcp_base=T[:3, 3].tolist(), q=qe.tolist(), dq=dq.tolist(), T_tr=1.875 * float(dq.max()) / 1.0, path_self_min=min(p['self'] for p in path),
                   path_bottle_link_min=min(p['bottle_link'] for p in path), path_deck_min=min(p['deck'] for p in path), path_margin_min=min(p['margin'] for p in path), **ev)
        break
    out = dict(version='v2_continuation', reference=dict(run='bottle_w1_cont_nominal1', t_lift_ok=t_ok, q_side=qa0.tolist(), T_rel=T_rel.tolist(), T_side=T_side.tolist(), side_eval=base_eval),
               n_candidates=len(cands), n_tried=tried, fail_reasons=fails, selected=sel,
               rule='包絡半徑最小（1 cm 等級）→ 瓶底最低，且從側持沿過渡路徑 IK 延續可達（每步約束全滿足、不跳分支）')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"試 {tried} 個；失敗原因 {fails}")
    if sel:
        print(f"選定：ψ {sel['psi_deg']}°、TCP 底盤座標 {np.round(sel['tcp_base'], 3).tolist()}；R_env {sel['R_env']:.3f} m（瓶 {sel['R_bottle']:.3f}）、瓶底 {sel['bottle_bottom']:.3f} m；"
              f"終點自碰 {sel['self'] * 1e3:.1f} mm、持瓶↔連桿 {sel['bottle_link'] * 1e3:.1f} mm、車體頂餘裕 {sel['deck_margin']:.3f}；路徑最小 自碰 {sel['path_self_min'] * 1e3:.1f} mm、"
              f"持瓶↔連桿 {sel['path_bottle_link_min'] * 1e3:.1f} mm、關節餘裕 {sel['path_margin_min']:.3f}；Δq {np.round(sel['dq'], 3).tolist()}；T_tr {sel['T_tr']:.2f} s")
    return 0 if sel else 1


if __name__ == '__main__':
    sys.exit(main())
