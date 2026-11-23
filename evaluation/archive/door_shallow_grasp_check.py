#!/usr/bin/env python3
"""B3 柔順綁定的單一「較淺抓持」候選（Codex 20261115）：TCP 沿接近軸後退 s 時的幾何核對（離線，只算幾何）。
以第八趟綁定瞬間的狀態（把手位姿、門角、底盤）為基準：
  shell_bar(s)＝夾爪殼對橫桿最近距離；保守需求＝門檻 5 mm＋平移預算 3 mm＋轉動預算造成的點位移 2r·sin(0.75°)（r＝該最近點到變形原點＝夾爪連桿原點的距離）。
  兩指：指片碰撞點對橫桿的最近距離（兩指各自、實測閉合開度）、橫桿軸線在指片沿接近軸方向的覆蓋餘量（橫桿中心到指尖的距離扣橫桿半徑與同一包絡）。
  其餘淨空（env／self／base）同式加包絡後是否仍過。
    python3 evaluation/door_shallow_grasp_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as W
import door_open_exec_ctrl as XO
import door_fast_kin as DFK
import door_motm_grasp_ctrl as G
PG, K, DOC = W.PG, XO.K, W.DOC
BUD_POS, BUD_ROT = 0.003, 1.5
RUN = os.path.join(HERE, 'runs', 'm8house_door_motm_8')
FS = np.load(os.path.join(RUN, 'first_segment.npz'))
T_WH_b, phi_b = FS['T_WH_b'], float(FS['phi_b'])
M = np.load(os.path.join(RUN, 'open_steps.npz'), allow_pickle=True)['meas']
q_b = M[0, 1:10].copy()
res8 = json.load(open(os.path.join(RUN, 'result.json')))
fq = [e for e in res8['events'] if e['what'] == 'handover_attempt'][-1]['bind_values']['finger_q']
Rg, T_HG, a_H = PG.baseline_reference()
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
_, bar = PG.handle_shapes(W.handle_pose(T_WH_b, phi_b))
aW = (T_WH_b[:3, :3] @ np.asarray(a_H, float)); aW /= np.linalg.norm(aW)
W.set_model(phi0=phi_b, finger_q=0.5 * (float(fq[0]) + float(fq[1])))
out = dict(basis='m8house_door_motm_8 綁定瞬間', budget=dict(pos_m=BUD_POS, rot_deg=BUD_ROT), finger_q=fq, thr=W.THR, rows=[])
for s in (0.0, 0.002, 0.004, 0.005, 0.006, 0.008, 0.010):
    Tg = T_at(W.handle_pose(T_WH_b, phi_b), s)
    qa, r_, sm = DFK.ik_closure_fc(Tg, q_b[:3], q_b[3:])
    q9 = np.r_[q_b[:3], qa]
    c = W.clear_door(np.r_[q9, phi_b], T_WH_b)
    Tgr = K.fk(q9, 'uflite_gripper_link')
    P = (Tgr[:3, :3] @ DOC.CL['uflite_gripper_link'].T).T + Tgr[:3, 3]
    d = PG.bar_dist(P, bar)
    i_min = int(np.argmin(d))
    r_shell = float(np.linalg.norm(P[i_min] - Tgr[:3, 3]))
    env_rot = 2 * r_shell * math.sin(math.radians(BUD_ROT / 2))
    need_shell = W.THR['shell_bar'] + BUD_POS + env_rot
    fing = {}
    for n in ('uflite_finger1', 'uflite_finger2'):
        T = K.fk(q9, n)
        fqn = float(fq[0] if n.endswith('1') else fq[1])
        Pf = (T[:3, :3] @ DOC.CL[n].T).T + T[:3, 3] + (1.0 if n.endswith('1') else -1.0) * fqn * T[:3, 1]
        db = PG.bar_dist(Pf, bar)
        along = (Pf - Tgr[:3, 3]) @ (-aW)                              # 指片各點沿「插入方向」（−接近軸）到夾爪原點的距離
        fing[n] = dict(bar_dist_mm=round(float(db.min()) * 1e3, 3), tip_along_mm=round(float(along.max()) * 1e3, 2), r_tip_mm=round(float(np.linalg.norm(Pf - Tgr[:3, 3], axis=1).max()) * 1e3, 1))
    bar_c = W.handle_pose(T_WH_b, phi_b)[:3, 3]
    bar_along = float((bar_c - Tgr[:3, 3]) @ (-aW))
    tip = min(fing[n]['tip_along_mm'] for n in fing) * 1e-3
    r_tip = max(fing[n]['r_tip_mm'] for n in fing) * 1e-3
    cover = tip - bar_along                                         # 指尖超出橫桿軸線多少（沿插入方向）
    need_cover = PG.BAR_R if hasattr(PG, 'BAR_R') else 0.013
    cover_margin = cover - (BUD_POS + 2 * r_tip * math.sin(math.radians(BUD_ROT / 2)))
    row = dict(s_mm=round(s * 1e3, 1), ik_resid=float(r_), sigma_min=round(float(sm), 4), shell_bar_mm=round(c['shell_bar'] * 1e3, 3), r_shell_mm=round(r_shell * 1e3, 1), shell_need_mm=round(need_shell * 1e3, 3), shell_ok=bool(c['shell_bar'] >= need_shell),
               env_mm=round(c['env'] * 1e3, 2), env_who=c['who']['env'], self_mm=round(c['self'] * 1e3, 2), base_mm=round(c['base'] * 1e3, 2),
               fingers=fing, bar_center_along_mm=round(bar_along * 1e3, 2), fingertip_beyond_bar_axis_mm=round(cover * 1e3, 2), fingertip_beyond_after_envelope_mm=round(cover_margin * 1e3, 2),
               bar_radius_mm=round(need_cover * 1e3, 1))
    out['rows'].append(row)
    print(json.dumps(row, ensure_ascii=False))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
