#!/usr/bin/env python3
"""MotM 開門第一包（2）：可抓窗口的運動學初算。把手＝full_2 本趟近距估計；底盤以固定偏航沿直線通過（方向 dir_deg），夾爪保持在抓取位姿（世界固定）。
沿底盤位移 s 連續解 IK（前一解為種子），記：有解、關節限位、σ_min≥0.03、淨空（閉門、手指張開；兩指—橫桿豁免）、所需關節速率（底盤速度 v 下）。
窗口＝全部成立的連續區段；可用時間＝長度／v，對照需求時間（插入 1.0＋閉爪 0.5＋確認 0.3＋管線 0.08 s）。只是運動學初算：未含追蹤動態、停止掃掠、估計誤差。
    python3 evaluation/door_motm_window.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
import door_insert_ctrl as DI
W, K, PG = XO.W, XO.K, XO.W.DA.PG
E = DI.R4.E
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P = np.load(os.path.join(RUN, 'plan.npz')); L = np.load(os.path.join(RUN, 'log.npz'))['rows']
T_near = P['T_near']; Rg, T_HG, a_H = PG.baseline_reference()
Tg = np.asarray(E.OT.object_target(T_near, T_HG, a_H, 0.0)['T_WE'], float)
E4 = L[L[:, 1] == 4][-1]; q_bind = np.r_[E4[5:8], E4[8:14]]            # 綁定靜置末實測（抓取構型）
W.set_model(phi0=0.0, finger_q=0.0089)
lo, hi = np.asarray(XO.LITE6_SAFE.lower), np.asarray(XO.LITE6_SAFE.upper)
T_NEED = 1.0 + 0.5 + 0.3 + 0.08


def ik(b, seed):
    q = seed.copy()
    for _ in range(40):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = Tg[:3, :3] @ Tc[:3, :3].T
        e = np.r_[Tg[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-9:
            return q, True
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]
        q = q + np.linalg.solve(J.T @ J + 1e-10 * np.eye(6), J.T @ e)
    return q, False


out = dict(handle='full_2 本趟近距估計', base_at_bind=q_bind[:3].round(4).tolist(), need_s=T_NEED, dirs={})
for dname, ddeg in (('沿門面 +x', 0.0), ('沿門面 −x', 180.0), ('遠離門 −y', -90.0), ('斜向 −x−y', -135.0)):
    d = np.array([math.cos(math.radians(ddeg)), math.sin(math.radians(ddeg))])
    rows = []
    for sgn in (1.0, -1.0):                                          # 由綁定點向前、向後各掃
        seed = q_bind[3:].copy()
        for s in np.arange(0.0, 0.61, 0.01):
            b = np.r_[q_bind[:2] + sgn * s * d, q_bind[2]]
            q, ok = ik(b, seed)
            if not ok or np.abs(q - seed).max() > 0.3:
                break
            seed = q
            q9 = np.r_[b, q]
            c = W.clear_door(np.r_[q9, 0.0], T_near)
            sig = float(np.linalg.svd(K.jacobian(q9, 'link_tcp')[:, 3:9], compute_uv=False).min())
            good = bool(np.all(q >= lo + 0.05) and np.all(q <= hi - 0.05) and sig >= W.SIG_FLOOR and all(c[k] >= W.THR[k] for k in ('env', 'base', 'self')))
            rows.append((sgn * s, good, q.copy(), sig, min(c[k] - W.THR[k] for k in ('env', 'base', 'self'))))
            if not good:
                break
    rows.sort(key=lambda r: r[0])
    rows = [r for i, r in enumerate(rows) if i == 0 or abs(r[0] - rows[i - 1][0]) > 1e-9]   # 兩向掃描在 s＝0 重複
    S = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows]); Q = np.array([r[2] for r in rows])
    gi = np.where(G)[0]
    s_lo, s_hi = (float(S[gi[0]]), float(S[gi[-1]])) if len(gi) else (0.0, 0.0)
    dqds = np.abs(np.gradient(Q[gi], S[gi], axis=0)).max(axis=0) if len(gi) > 2 else np.zeros(6)
    res = dict(window_m=[round(s_lo, 3), round(s_hi, 3)], length_m=round(s_hi - s_lo, 3), dq_ds_max_rad_per_m=dqds.round(2).tolist(), sigma_min=round(float(min(r[3] for r in rows if r[1])), 4) if len(gi) else None,
               clear_margin_min_mm=round(float(min(r[4] for r in rows if r[1])) * 1e3, 1) if len(gi) else None, speeds={})
    for v in (0.05, 0.10, 0.15):
        res['speeds'][str(v)] = dict(time_s=round((s_hi - s_lo) / v, 2), enough=bool((s_hi - s_lo) / v >= T_NEED), joint_rate_max=round(float(dqds.max() * v), 3), rate_ok=bool(dqds.max() * v <= 1.0))
    out['dirs'][dname] = res
    print(dname, res['window_m'], res['length_m'], 'σmin', res['sigma_min'], {k: (x['time_s'], x['enough'], x['joint_rate_max']) for k, x in res['speeds'].items()})
out['note'] = '運動學初算：底盤偏航固定、直線通過、夾爪世界位姿固定在抓取位姿；未含插入段位移、追蹤動態、停止掃掠、估計誤差、輪級限制'
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
