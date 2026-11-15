#!/usr/bin/env python3
"""閉鏈一致備援（開門執行層 r16＋door_cstop）最小核對（Codex 20261115_005224 §3）：假世界名目計畫，pool 6、deadline 1000 ms＋注入晚到。
  early：t≥0.05 s（同 motm_3 晚到時點）強制晚到；mid：t≥3.0 s（開門中段）強制晚到。
記：備援依據（模擬拍數、淨空、閉鏈）、停止模式、是否中止、停止有效與耗時、停止期間實測閉鏈最大值、設定點—實測差終值、命令上限使用、預測終點對實際終點差。
    python3 evaluation/door_cstop_check.py PLAN_NPZ OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_sweep_pool as DSP
import door_cstop as DCS
POOL, CSW = DSP.SweepPool(6), DCS.CStopWorker()
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
import test_door_open_exec as T
X = T.X
P = np.load(sys.argv[1])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
out = {}
# r2（Codex 20261115_131250）：abort_in_cstop＝備援停止 0.2 s 後超力（須同拍中止、之後不再出命令）；missing＝現在停核對結果缺失（須中止，不得當核過）；
#   infeasible_late＝模型判備援不可行且同拍晚到（須中止）
for case, t_late in (('early', 0.05), ('mid', 3.0), ('abort_in_cstop', 3.0), ('missing', 3.0), ('infeasible_late', 3.0)):
    c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q,
                       arm_mode='closure', g2=dict(gravity2=True, ik_fc=True, pool=POOL, prep_fc=True, deadline_ms=1000.0, cstop=CSW))
    F = T.Fake(c, q0)
    c.warmup(q0, F.door()[0])
    st = dict(done=False, pred=None)

    def dl(cc, st=st, t_late=t_late):
        if cc.t >= t_late - 1e-9 and not st['done']:
            st['done'] = True
            return True
        return False
    if case != 'missing':
        c.deadline_inject = dl
    if case == 'abort_in_cstop':
        F.force_fn = lambda F_: 50.0 if (F_.c.stop_mode == 'cstop' and F_.c.stopped_t is not None and F_.t - F_.c.stopped_t >= 0.2) else 5.0
    if case in ('missing', 'infeasible_late'):
        c.cs_inject = lambda cc, tag, case=case, t_late=t_late: ({'missing': 'missing', 'infeasible_late': 'infeasible'}[case] if (tag == 'stop_now' and cc.t >= t_late - 1e-9) else None)
    F.run(t_end=t_late + 8.0)
    arm_cmd = max((x[2] for x in c.cs_log), default=None)
    out[case] = dict(stop=c.stop, abort=c.abort, stop_mode=c.stop_mode, stop_ok=c.stop_ok, n_cstop_ticks=len(c.cs_log), arm_cmd_max=arm_cmd, phi_end_deg=round(math.degrees(F.door()[0]), 3),
                     after_stop_closure_pos_mm=round(F.after_stop['closure_pos'] * 1e3, 3), after_stop_min_clear={k: round(float(v), 4) for k, v in F.after_stop['min'].items()},
                     setpoint_minus_meas_end=[round(float(x), 5) for x in (F.q[3:] - c.s)], d_true=T.D_TRUE.tolist(), n_cmd_after_abort=F.n_cmd_after_abort, escalations=(c.stop or {}).get('escalations'), cs_ticks_after_abort=(None if c.abort is None else sum(1 for x in c.cs_log if x[0] > c.abort['t'] + 1e-9)))
    print(case, json.dumps(out[case], ensure_ascii=False, default=float)[:6000], flush=True)
POOL.close(); CSW.close()
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
