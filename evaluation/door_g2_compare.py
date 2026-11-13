#!/usr/bin/env python3
"""G2 整趟對照：開門執行層 r8 在假世界（test_door_open_exec.Fake，名目計畫、closure 模式）整趟開門，g2 各組合對 r7 行為（全關）比：
逐物理步送進管線的命令（cmd_hist）最大差、事件序列、停止原因、閉鏈統計與最小淨空是否相同；整趟每拍耗時（首拍＝冷、p50／p95／max、>50 ms 拍數）、快取命中。
組合（事先登記）：off（＝r7）、gravity、gravity+ik_fc、gravity+ik_fc+sweep_cache、all（再加 ik_warm）。
    python3 evaluation/door_g2_compare.py PLAN_DIR OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
import test_door_open_exec as T
X = T.X
P = np.load(os.path.join(sys.argv[1], 'plan_nominal.npz'))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
COMBOS = [('off', {}), ('gravity', dict(gravity=True)), ('gravity+ik_fc', dict(gravity=True, ik_fc=True)), ('gravity+ik_fc+sweep_cache', dict(gravity=True, ik_fc=True, sweep_cache=True)),
          ('all', dict(gravity=True, ik_fc=True, sweep_cache=True, ik_warm=True))]
res, base = {}, None
for tag, g2 in COMBOS:
    c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q, arm_mode='closure', g2=g2)
    F = T.Fake(c, q0)
    t0 = time.perf_counter(); F.run(t_end=40.0); wall = time.perf_counter() - t0
    rep = c.report()
    tk = np.array(c.tick_ms)
    U = np.array(c.cmd_hist)
    ev = [(e['what'], e['t']) for e in c.events]
    r = dict(wall_s=round(wall, 1), n_ticks=int(len(tk)), tick_ms=dict(first_cold=round(float(tk[0]), 1), p50=round(float(np.median(tk)), 1), p95=round(float(np.percentile(tk, 95)), 1), max=round(float(tk.max()), 1), n_over_50=int((tk > 50).sum())),
             solve_ms_p50=round(float(np.median(c.solve_ms)), 1), sweep_ms_p50=round(float(np.median(c.sweep_ms)), 1) if c.sweep_ms else None,
             stop=c.stop, done=c.done, n_events=len(ev), cl_stats={k: v for k, v in c.cl_stats.items() if k != 'fails'}, n_cl_fails=len(c.cl_stats['fails']), mins=c.mins,
             cache=None if c._ccache is None else dict(hit=c._ccache.hit, miss=c._ccache.miss))
    if base is None:
        base = dict(U=U, ev=ev, stop=c.stop, mins=c.mins, cl=r['cl_stats'], log=np.array(F.log))
        r['vs_off'] = None
    else:
        L = np.array(F.log)
        r['vs_off'] = dict(cmd_len_same=bool(U.shape == base['U'].shape), cmd_max_abs_diff=float(np.abs(U - base['U']).max()) if U.shape == base['U'].shape else None,
                           events_same=bool(ev == base['ev']), stop_same=bool(c.stop == base['stop']),
                           mins_max_abs_diff=max(abs(float(c.mins[g][k]) - float(base['mins'][g][k])) for g in ('act', 'stop') for k in X.KEYS),
                           n_sweep_same=bool(c.mins['n_sweep'] == base['mins']['n_sweep']),
                           state_log_max_abs_diff=float(np.abs(L - base['log']).max()) if L.shape == base['log'].shape else None)
    res[tag] = r
    print(tag, json.dumps({k: r[k] for k in ('wall_s', 'n_ticks', 'tick_ms', 'solve_ms_p50', 'sweep_ms_p50', 'done', 'cache', 'vs_off')}, ensure_ascii=False, default=float), flush=True)
json.dump(res, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
