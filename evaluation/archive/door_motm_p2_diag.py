import sys, math, json, numpy as np
sys.path.insert(0, '/home/howardchen/masterthesis/evaluation')
import door_motm_p2 as P2
WB = P2.WB
rd = sys.argv[1]
ns, Z, T_e, x0, v0, T_HE = P2.load_start(rd, 'nominal_0.05')
goal, fq, phm = ns['GOAL'], ns['FQ'], float(Z['phi'])
# 起點：P2（含 (i)）1 s 段續接到 t≈20 s 的參考狀態——重跑續接取狀態太貴；改用「整段規劃」密集參考上 φ≈84.2° 的構型、v0＝0（診斷起點，照實標示）
WB.set_model(phi0=phm, finger_q=fq)
b = x0[:3]; Tr = 1.875 * (goal - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM); N = int(math.ceil((Tr + 2.0) / 0.2))
o = WB.DoorWBOpt(T_e, np.eye(4), x0, goal, N=N, dt=0.2, T_HE=T_HE, project_start=False, v0=v0, grad='analytic'); o.init_corotate(Tr); o.solve(max_iter=45)
Xd, h, _ = o.dense_reference(4)
i = int(np.argmin(np.abs(Xd[:, 9] - math.radians(84.2)))); xs = o.project(Xd[i], iters=30)
out = dict(start_phi_deg=round(math.degrees(xs[9]), 3))
for H in (1.0, 2.0):
    Nh = int(round(H / 0.2)); M = P2.term_nodes(xs, goal, Nh)
    for tag, tol, wmod in (('default', 1e-3, {}), ('tight_tol', 1e-8, {}), ('no_lim_manip', 1e-3, dict(lim=0.0, manip=0.0)), ('tight+no_lim_manip', 1e-8, dict(lim=0.0, manip=0.0))):
        WB.set_model(phi0=phm, finger_q=fq)
        s = WB.DoorWBOpt(T_e, np.eye(4), xs, goal, N=Nh, dt=0.2, T_HE=T_HE, project_start=False, v0=np.zeros(10), grad='analytic', segment=True, term_nodes=M, w=wmod)
        s.init_corotate(max(1.0, H))
        info = s.solve(max_iter=200, tol_rel=tol); v = s.verify()
        out[f'H{H:g}_{tag}'] = dict(M=M, why=info['why'], iters=info['iters'], phi_N_deg=round(math.degrees(s.X[-1, 9]), 4), dphi_deg=round(math.degrees(s.X[-1, 9] - xs[9]), 4), feasible=v['feasible'], failed=v['failed'])
        print(H, tag, out[f'H{H:g}_{tag}'], flush=True)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
