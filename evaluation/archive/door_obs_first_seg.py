#!/usr/bin/env python3
"""完整候選的第 4 項：由接手點（a_w、門前線、沿門 −a、V）預測 READY 狀態（抓持排程 2.05 s 後；底盤沿 −a 前推、手臂＝該底盤對抓取位的門旁分支 IK、閉鏈一致速度），
以同一把手估計（far_door_9 近距鎖定參考）請求 P2 開門首段（plan_seg，H＝2），核可行性與門角。運動學層離線。
    python3 door_obs_first_seg.py RUN_DIR OUT.json"""
import json, math, os, sys, time
os.environ.setdefault('P2_TERM', '1'); os.environ.setdefault('P2_FINAL', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
import door_motm_p2 as P2
sys.argv = sys.argv_
WB, G = P2.WB, TS.G
V, A_W, SB, T_READY, H = 0.05, -0.10, 0.4607, 2.05, 2.0
GOAL, FQ = math.radians(85.0), G.FINGER_CLOSED
T_H = TS.T_H
phi_b = 0.0
a_r = A_W - V * T_READY
b_r = TS.base(a_r, SB)
q_pre = TS.end_config(A_W, SB)
q_g, sm = TS.solve(b_r, 0.0, TS.solve(TS.base(a_r, SB), TS.S_PRE, q_pre)[0])
x_pred = np.r_[b_r, q_g, phi_b]
vw = -TS.d * V
T_HE = np.linalg.inv(T_H) @ TS.T_at(T_H, 0.0)
WB.set_model(phi0=phi_b, finger_q=FQ)
helper = WB.DoorWBOpt(T_H, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE, project_start=False)
v_pred = np.r_[vw, 0.0, helper.closure_velocity(x_pred, np.r_[vw, 0.0], 0.0), 0.0]
t0 = time.perf_counter()
o, info, ver, wall = P2.plan_seg(T_H, T_HE, x_pred, v_pred, H, GOAL, FQ, phi_b)
out = dict(note='離線；接手點 a_w 預測 READY 狀態的 P2 開門首段', a_w=A_W, a_ready=round(a_r, 4), sb=SB, sigma_ready=round(float(sm), 4), feasible=bool(ver['feasible']), failed=ver['failed'], why=info['why'],
           phi_end_deg=ver.get('phi_end_deg'), wall_s=round(wall, 3), min_clear=ver.get('min_clear_m'), thr_plan=dict(WB.THR_PLAN))
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=str)
print(out)
