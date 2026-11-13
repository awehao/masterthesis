import json, os, sys, time, math
import numpy as np
sys.path.insert(0, '/home/howardchen/masterthesis/evaluation')
import door_wbopt as WB
if os.environ.get('FAST_CLEAR') == '1':
    import door_clear_door_fast as _F; WB.clear_door = _F.clear_door_fast
import test_door_open_exec as T
X = T.X
SP = sys.argv[1]   # scratchpad（含 ex4/plan_nominal.npz）
P = np.load(SP + '/ex4/plan_nominal.npz')
ident = json.load(open('/home/howardchen/masterthesis/evaluation/results/wgmpc_arm_sp_ident_free4.json'))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q, arm_mode='closure')
F = T.Fake(c, q0); F.run(t_end=3.0)
r = c.report()['timing']
# 退出控制器
import door_exit_ctrl as XE
RUN = '/home/howardchen/masterthesis/evaluation/runs/m8house_door_full_2'
L = np.load(RUN + '/log.npz')['rows']; Pp = np.load(RUN + '/plan.npz')
R7 = L[L[:, 1] == 7]; qx = np.r_[R7[-1, 5:8], R7[-1, 8:14]]; phx = float(R7[-1, 2]); om = float((R7[-1, 2] - R7[-11, 2]) / 0.1)
sx = np.load(RUN + '/open_steps.npz')['ctrl'][-1, 32:38]
xc = XE.DoorExitCtrl(ident, Pp['T_WH_bind'], float(Pp['phi_bind']), sx, [np.zeros(9)] * 8, np.zeros(9), np.zeros(9), al * (qx[3:] - sx), 0.0089, phx, om, np.zeros((1, 2)), qx[3:], scope_end='X2')
q = qx.copy(); ticks = []
for k in range(300):
    t0 = time.perf_counter(); o = xc.step(k * 0.01, q.copy(), phx); dt = time.perf_counter() - t0
    if k % 5 == 0: ticks.append(dt * 1e3)
    cs, sn = math.cos(q[2]), math.sin(q[2]); vx, vy, wz = o['base_twist']
    q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz]); q[3:] += al * (o['arm_target'] - q[3:] + (qx[3:] - sx))
print(json.dumps(dict(fast=os.environ.get('FAST_CLEAR'), open_exec=r, exit_tick_ms=dict(p50=round(float(np.median(ticks)), 1), p95=round(float(np.percentile(ticks, 95)), 1), max=round(float(max(ticks)), 1)), exit_phase=xc.phase), ensure_ascii=False))
