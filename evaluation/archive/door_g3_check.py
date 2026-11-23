#!/usr/bin/env python3
"""G3 核對（事前登錄）：
  U 計算池單元：(1) 工作 A 在工作程序 0 注入 0.3 s 延遲、收集逾時 0.05 s ⇒ 'timeout'；(2) 撤延遲後工作 B ⇒ ok，A 的遲到結果被丟棄（stale_dropped≥1），B 結果與直接計算逐值相同；
     (3) 終止工作程序 1 後工作 C ⇒ 'worker_dead'（不得回結果）。
  E 開門執行層假世界整趟（名目計畫、closure）：r8 最佳組合（gravity+ik_fc+sweep_cache）對 r9（gravity+ik_fc+pool=N）：命令、事件、停止、掃掠樣本數、最小淨空逐值比；每拍耗時與分項（準備／派工／等待／彙整）。
  F 失效注入（r9）：t＝1.0 s 終止一個工作程序；t＝1.0 s 起一個工作程序注入 0.8 s 延遲（> POOL_TIMEOUT_S）⇒ 應模擬中止、中止後不再出命令。
    python3 evaluation/door_g3_check.py PLAN_DIR N_WORKERS OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
import door_sweep_pool as DSP
import test_door_open_exec as T
X = T.X
NW = int(sys.argv[2])
out = {}
# ---- U
P2 = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
Tb, phib = P2['T_WH_bind'], float(P2['phi_bind'])
Xs = P2['Xd'][::10][:24]
pool = DSP.SweepPool(2)
pool.inject_delay = {0: 0.3}
hA = pool.submit(Tb, 0.00174, phib, Xs)
rA, wA, _ = pool.collect(hA, 0.05)
pool.inject_delay = {}
hB = pool.submit(Tb, 0.00174, phib, Xs)
rB, wB, tB = pool.collect(hB, 2.0)
WB.set_model(phi0=phib, finger_q=0.00174)
direct = [_F.clear_door_fast(x, Tb) for x in Xs]
eqB = rB is not None and all(all(a[k] == b[k] for k in ('env', 'base', 'self', 'shell_bar')) and a['who'] == b['who'] for a, b in zip(rB, direct))
pool.procs[1].terminate(); pool.procs[1].join()
hC = pool.submit(Tb, 0.00174, phib, Xs)
rC, wC, _ = pool.collect(hC, 1.0)
out['U'] = dict(A=wA, A_none=rA is None, B=wB, B_equal_direct=bool(eqB), stale_dropped=pool.stats['stale_dropped'], C=wC, C_none=rC is None, stats=pool.stats)
pool.close()
print('U', out['U'], flush=True)
# ---- E
P = np.load(os.path.join(sys.argv[1], 'plan_nominal.npz'))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)


def mk(g2):
    return X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q, arm_mode='closure', g2=g2)


def st(a):
    a = np.asarray(a, float)
    return dict(n=int(a.size), first_cold=round(float(a[0]), 1), p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1), n_over_50=int((a > 50).sum()))


runs = {}
for tag, g2 in (('r8_best', dict(gravity=True, ik_fc=True, sweep_cache=True)), (f'r9_pool{NW}', dict(gravity=True, ik_fc=True, pool=NW))):
    c = mk(g2)
    F = T.Fake(c, q0)
    F.run(t_end=40.0)
    runs[tag] = dict(U=np.array(c.cmd_hist), ev=[(e['what'], e['t']) for e in c.events], stop=c.stop, mins=c.mins, log=np.array(F.log), tick=st(c.tick_ms), sweep=st(c.sweep_ms), done=c.done,
                     pool=None if c._pool is None else dict(stats=c._pool.stats, dispatch=st([d['dispatch'] for d in c.pool_ms]), wait=st([d['wait'] for d in c.pool_ms]), gather=st([d['gather'] for d in c.pool_ms]), prep=st(c.prep_ms)))
    c.close_pool()
    print(tag, runs[tag]['tick'], runs[tag]['sweep'], runs[tag]['pool'], flush=True)
a, b = runs['r8_best'], runs[f'r9_pool{NW}']
out['E'] = dict(r8_best=dict(tick=a['tick'], sweep=a['sweep'], done=a['done']), r9=dict(tick=b['tick'], sweep=b['sweep'], done=b['done'], pool=b['pool']),
                cmd_equal=bool(a['U'].shape == b['U'].shape and np.array_equal(a['U'], b['U'])), events_equal=bool(a['ev'] == b['ev']), stop_equal=bool(a['stop'] == b['stop']),
                mins_equal=bool(all(a['mins'][g][k] == b['mins'][g][k] for g in ('act', 'stop') for k in X.KEYS) and a['mins']['n_sweep'] == b['mins']['n_sweep']),
                state_log_equal=bool(a['log'].shape == b['log'].shape and np.array_equal(a['log'], b['log'])))
print('E', {k: v for k, v in out['E'].items() if k not in ('r8_best', 'r9')}, flush=True)
# ---- F
out['F'] = {}
for tag in ('kill_worker', 'delay_worker'):
    c = mk(dict(gravity=True, ik_fc=True, pool=NW))
    F = T.Fake(c, q0)

    def hook(f, c=c, tag=tag, done=[False]):
        if f.t >= 1.0 and not done[0]:
            done[0] = True
            if tag == 'kill_worker':
                c._pool.procs[0].terminate(); c._pool.procs[0].join()
            else:
                c._pool.inject_delay = {0: 0.8}
    F.run(t_end=4.0, hook=hook)
    out['F'][tag] = dict(stop=c.stop, abort=c.abort, aborted_fake=F.aborted, abort_t=F.abort_t, n_cmd_after_abort=F.n_cmd_after_abort, n_phys_after_abort=F.n_phys_after_abort, pool_stats=c._pool.stats)
    c.close_pool()
    print('F', tag, out['F'][tag], flush=True)
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=float)
