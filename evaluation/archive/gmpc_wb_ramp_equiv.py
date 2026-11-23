#!/usr/bin/env python3
"""A：gmpc_wb_ramp 選項關閉（cmd_ramp_w＝None）時與原 gmpc_wb 逐值相同。在開門執行層名目假世界上，每控制拍把同一組輸入同時送進兩個求解器比較。
    python3 evaluation/gmpc_wb_ramp_equiv.py PLAN.npz OUT.json"""
import dataclasses, json, sys
import numpy as np
import test_door_open_exec as T
X = T.X
GW, GR, B = X.GW, X.GR, X.B
P = np.load(sys.argv[1])
ident = json.load(open(T.os.path.join(T.HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q)
nc = B.nav_cfg()
g_orig = GW.GMPC(nc)
rc = GR.GMPCConfig(**{f.name: getattr(nc, f.name) for f in dataclasses.fields(nc)}); rc.cmd_ramp_w = None
g_off = GR.GMPC(rc)
real = c.mpc
diffs = []

def both(**kw):
    a = g_orig.solve(**kw); b = g_off.solve(**kw)
    da = float(np.abs(np.asarray(a.u_opt) - np.asarray(b.u_opt)).max())
    dq = 0.0 if a.u_arm_seq is None and b.u_arm_seq is None else float(np.abs(np.asarray(a.u_arm_seq) - np.asarray(b.u_arm_seq)).max())
    diffs.append((da, dq))
    return real.solve(**kw)
c.mpc = type('M', (), {'solve': staticmethod(both)})()
F = T.Fake(c, q0)
F.run(t_end=1.6)
d = np.array(diffs)
out = dict(n_solves=len(d), u_opt_max_abs_diff=float(d[:, 0].max()), u_arm_seq_max_abs_diff=float(d[:, 1].max()), equal=bool(d.max() == 0.0))
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(out)
