#!/usr/bin/env python3
"""(a) 含 v0 初猜一版（Codex 20261113_120619）：door_wbopt r7 init_corotate(T_ramp, v0_tau＝0.4 s，跑前登錄) 對 r6 初猜（v0_tau＝None），同題同驗收（grad＝analytic）。
案（事前登錄）：移動抓持 READY 狀態冷啟——0.05 m/s 的 nominal／combined／v095／yawdist，0.08 m/s 的 nominal。起點語意同 door_motm_reuse_check（管線清空後、閉鏈一致 v0）。
記：求解狀態、迭代（含拒絕）、時間、verify 可行與失敗項、最小淨空、終點門角、J；兩版解的節點最大差。只當 P2 初值與 P1 對照，不開新一輪 P1 調整。
    python3 evaluation/door_wbopt_init_compare.py READY_DIR OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
TAU = 0.4
rd, OUT = sys.argv[1], sys.argv[2]
src = open(os.path.join(HERE, 'door_motm_reuse_check.py')).read()
sys.argv = ['x', rd]
ns = dict(__file__=os.path.join(HERE, 'door_motm_reuse_check.py'), __name__='rc_head')
exec(compile(src[:src.index("Zn = np.load(")], 'reuse_head', 'exec'), ns)
cases = []
for tag in ('nominal_0.05', 'combined_0.05', 'v095_0.05', 'yawdist_0.05', 'nominal_0.08'):
    Z = np.load(os.path.join(rd, f'{tag}_ready.npz'))
    T_e, x0, v0, T_HE = ns['start_of'](Z)
    b = x0[:3]
    Tr = 1.875 * (ns['GOAL'] - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
    N = int(math.ceil((Tr + 2.0) / 0.2))
    r = dict(tag=tag, N=N, v0_base=np.round(v0[:3], 4).tolist())
    Xs = {}
    for name, tau in (('r6_init', None), ('r7_v0decay', TAU)):
        WB.set_model(phi0=float(Z['phi']), finger_q=ns['FQ'])
        o = WB.DoorWBOpt(T_e, np.eye(4), x0, ns['GOAL'], N=N, dt=0.2, T_HE=T_HE, project_start=False, v0=v0, grad='analytic')
        o.init_corotate(Tr, v0_tau=tau)
        t0 = time.perf_counter(); info = o.solve(max_iter=45); ws = time.perf_counter() - t0
        v = o.verify(); m = o.merit(o.X)
        r[name] = dict(why=info['why'], iters=info['iters'], accepted=sum(1 for z in o.log if 'step' in z), rejected=sum(1 for z in o.log if 'reject' in z or 'qp' in z), wall_s=round(ws, 2),
                       feasible=v['feasible'], failed=v['failed'], min_clear=v['min_clear_m'], phi_end_deg=round(math.degrees(float(o.X[-1, 9])), 3), J=round(float(m[1]), 4), viol=round(float(m[2]), 6))
        Xs[name] = o.X.copy()
    d = np.abs(Xs['r6_init'] - Xs['r7_v0decay']).max(axis=0)
    r['X_maxdiff'] = dict(base_xy_mm=round(float(d[:2].max()) * 1e3, 3), yaw_deg=round(math.degrees(float(d[2])), 4), arm_rad=round(float(d[3:9].max()), 5), phi_deg=round(math.degrees(float(d[9])), 4))
    cases.append(r)
    print(tag, 'r6', r['r6_init'], '\n   r7', r['r7_v0decay'], '\n   dX', r['X_maxdiff'], flush=True)
json.dump(dict(v0_tau=TAU, cases=cases), open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
