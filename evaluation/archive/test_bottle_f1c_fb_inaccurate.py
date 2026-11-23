#!/usr/bin/env python3
"""返回段後備擴到 solved inaccurate 的針對性核對（Codex 20261103_145515；不跑模擬）。輸入是 bottle_f1c_1 的完整紀錄與當拍掃描（實跑紀錄，非近似）。
  1 重現原停止：預設關閉時 72.90 s 狀態 solved inaccurate、命令與紀錄全精度相同。
  2 開啟後備：KKT 過、回 solved；全時域全部 QP 列與後處理後第一拍限制滿足；報全精度差與完整 ε 序列。
  3 正常 solved 拍不觸發，命令不變；原迭代上限後備仍有效。
  4 新狀態下後備未得解、非有限值、維度不符：保留失敗狀態（返回段照原規則停止）。
    python3 evaluation/test_bottle_f1c_fb_inaccurate.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402
import bottle_f1_dyn_report as FD                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

D = os.path.join(HERE, 'runs', 'bottle_f1c_1')
REAL = GW.osqp.OSQP
CAP = {}


class Spy(REAL):
    def setup(self, *a, **k):
        if k.get('max_iter') == 8000:
            CAP.update({n: k[n] for n in ('P', 'q', 'A', 'l', 'u')})
        return super().setup(*a, **k)


GW.osqp.OSQP = Spy
c, _ = T.build_ctrl()
cfg = c.mpc_nav.cfg
SL = np.load(os.path.join(D, 'scan_log.npz'))
ts, rr = SL['t'], SL['ranges']
L = [e for e in FD.load_rows(os.path.join(D, 's1_cbf.npz')) if float(e['w_p']) == 0.0]
fails, OUT = [], {}


def chk(name, ok, detail=''):
    print(('通過 ' if ok else '未過 ') + name, detail)
    if not ok:
        fails.append(name)


def solve(e, arm):
    c.t = float(e['t'])
    i_ = int(np.searchsorted(ts, c.t + 1e-9) - 1)
    c.scan_last = (float(ts[i_]), np.array(rr[i_], float))
    return c.mpc_nav.solve(obstacles=list(e['obs']), X_now=from_xytheta(*np.asarray(e['qp'], float)[:3]), X_ref_win=e['X_ref'], xi_ref_win=e['xi_ref'], xi_prev=np.asarray(e['u_prev'], float)[:3], arm=arm)


OFF, ON = dict(mode='hold'), dict(mode='hold', mi_gi_fallback=True)
stop = L[-1]
chk('末拍是 72.90 s', abs(float(stop['t']) - 72.9) < 1e-6, float(stop['t']))
r0 = solve(stop, OFF)
d0 = float(np.abs(np.asarray(r0.u_opt, float) - np.asarray(stop['u'], float)[:3]).max())
TOL_REPRO = 1e-12                                       # 全精度比對容差（浮點捨入）；實際差值照印
chk('1 預設關閉重現原停止', str(r0.status) == 'solved inaccurate' and d0 <= TOL_REPRO and r0.hold_fb is None, f'status {r0.status}；命令全精度差 {d0:.3e}')
r1 = solve(stop, ON)
fb = r1.hold_fb
chk('2 後備觸發並過 KKT', bool(fb and fb['gi_ok']) and fb['osqp_status'] == 'solved inaccurate' and str(r1.status) == 'solved', json.dumps({k_: fb[k_] for k_ in ('osqp_status', 'osqp_iter', 'osqp_ms', 'gi_steps', 'gi_ms', 'kkt')}, ensure_ascii=False, default=float))
gx, gy, gi = GW.qp_dual_active(CAP['P'], CAP['q'], CAP['A'], CAP['l'], CAP['u'])
Ax = CAP['A'] @ gx
viol = float(np.maximum(np.maximum(CAP['l'] - Ax, Ax - CAP['u']), 0.0).max())
Nm = cfg.N * 3
eps = gx[Nm:]
chk('2 全時域全部 QP 列', viol <= 1e-8, f'最大違反 {viol:.3e}；{CAP["A"].shape[0]} 列、{len(gx)} 變數')
u = np.asarray(r1.u_opt, float)
up = np.asarray(stop['u_prev'], float)[:3]
xi0 = np.asarray(stop['xi_ref'], float)[0]
W = GW.wheel_matrix(cfg)
i_ = int(np.searchsorted(ts, float(stop['t']) + 1e-9) - 1)
rows = c._u0_rows()
A0, l0, _, _ = GW._build_cbf_horizon(cfg, from_xytheta(*np.asarray(stop['qp'], float)[:3]), stop['X_ref'], stop['xi_ref'], list(stop['obs']), cfg.N)
A0 = A0.tocsr()
k0 = np.array([r_ for r_ in range(A0.shape[0]) if A0[r_, :3].nnz], int)
res0 = A0[k0][:, :3] @ (u - xi0) - l0[k0]
DT = cfg.dt
v = dict(vel=float(np.maximum(np.maximum(u - cfg.u_max, cfg.u_min - u), 0.0).max()), acc=float(np.maximum(np.abs(u - up) - np.asarray(cfg.a_max) * DT, 0.0).max()), wheel_vel=float(max(np.abs(W @ u).max() - cfg.wheel_radius * cfg.wheel_w_max, 0.0)), wheel_acc=float(max(np.abs(W @ (u - up)).max() - cfg.wheel_radius * cfg.wheel_a_max * DT, 0.0)), shield=float(max((rows[0] @ u - rows[1]).max(), 0.0)), k0_slack=float(max(-(res0 + float(r1.eps0)).min(), 0.0)))
chk('2 後處理後第一拍限制', all(x_ <= 1e-6 for x_ in v.values()), f'accept {r1.accept_action}；違反 {v}')
du = u - np.asarray(stop['u'], float)[:3]
OUT['stop_tick'] = dict(t=float(stop['t']), osqp=dict(status=fb['osqp_status'], iter=fb['osqp_iter'], ms=fb['osqp_ms']), gi=dict(steps=fb['gi_steps'], ms=fb['gi_ms'], kkt=fb['kkt']), command_logged=np.asarray(stop['u'], float)[:3].tolist(), command_gi=u.tolist(), command_diff_full=du.tolist(), command_diff_max=float(np.abs(du).max()), qp_viol_max=viol, first_tick_viol=v, k0_resid_noslack_min=float(res0.min()), eps0=float(r1.eps0), eps_seq_max=float(eps.max()), eps_seq_nonzero=[[int(k_), float(x_)] for k_, x_ in enumerate(eps) if x_ > 1e-8], n_eps=len(eps), accept=str(r1.accept_action))
print('   全精度命令差', [f'{x_:.3e}' for x_ in du], '；ε 序列最大', f'{eps.max():.3e}', '；非零', OUT['stop_tick']['eps_seq_nonzero'][:6])
for e in L[-5:-1]:
    a_, b_ = solve(e, OFF), solve(e, ON)
    dl_ = float(np.abs(np.asarray(b_.u_opt) - np.asarray(e['u'], float)[:3]).max())
    chk(f'3 正常拍不觸發 t={float(e["t"]):.2f}', str(b_.status) == 'solved' and b_.hold_fb is None and np.array_equal(np.asarray(a_.u_opt), np.asarray(b_.u_opt)) and dl_ <= TOL_REPRO, f'對紀錄差 {dl_:.3e}')
hit = None
for e in L:
    if 70.40 <= float(e['t']) <= 70.60:
        rr_ = solve(e, ON)
        if rr_.hold_fb is not None and rr_.hold_fb['osqp_status'] == 'maximum iterations reached':
            hit = (float(e['t']), rr_)
            break
chk('3 原迭代上限後備仍有效', hit is not None and hit[1].hold_fb['gi_ok'] and str(hit[1].status) == 'solved', None if hit is None else f't {hit[0]:.2f}；ε₀ {hit[1].eps0:.4f}')
orig = GW.HOLD_GI
for why, inj in (('未得解', lambda *a_, **k_: (None, None, dict(ok=False, why='injected', steps=0))), ('非有限值', lambda *a_, **k_: (gx * np.nan, gy, dict(ok=True, steps=1))), ('維度不符', lambda *a_, **k_: (gx[:-1], None, dict(ok=True, steps=1)))):
    GW.HOLD_GI = inj
    rf = solve(stop, ON)
    chk(f'4 後備{why}保留失敗狀態', rf.hold_fb is not None and rf.hold_fb['gi_ok'] is False and str(rf.status) == 'solved inaccurate', f'status {rf.status}')
GW.HOLD_GI = orig
json.dump(OUT, open(os.path.join(HERE, 'results', 'bottle', 'f1c_fb_inaccurate_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print('全部通過' if not fails else f'未過 {fails}')
sys.exit(1 if fails else 0)
