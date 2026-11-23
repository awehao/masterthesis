#!/usr/bin/env python3
"""開門執行層 r11 期限備援核對（Codex 20261113_130916 §1；事前登錄）：pool＝6、prep_fc、gravity2、ik_fc、deadline 50 ms、交接前預熱；假世界名目計畫。
  A nominal        不注入
  B late_valid     t≥3.0 s 第一拍強制晚到；備援照常評估
  C late_badcl     同 B，且該拍備援評估的預測閉鏈位置偏差放大為 20 mm（淨空照過）⇒ 應模擬中止
  D path_valid     t≥3.0 s 外部 forced_stop（路徑停止），t≥3.2 s 第一拍強制晚到
  E path_badcl     同 D，晚到那拍備援閉鏈偏差放大為 20 mm ⇒ 應模擬中止
注入案（B–E）期限設 1000 ms，只有注入的那一拍晚到（邏輯核對）；A 用 50 ms（實際計時）。第一次執行時 B 用 50 ms，於 t＝0.1 s 自然晚到（52.1 ms），保留在 natural_miss_first_run。
記：停止原因與模式、中止、晚到紀錄（備援依據）、停止有效、停止期間實測閉鏈偏差最大值（假世界 after_stop）、中止後命令數。
    python3 evaluation/door_deadline_check.py PLAN_DIR OUT.json"""
import json, math, os, sys
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
out = {}
for case in ('nominal', 'late_valid', 'late_badcl', 'path_valid', 'path_badcl'):
    c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q,
                       arm_mode='closure', g2=dict(gravity2=True, ik_fc=True, pool=6, prep_fc=True, deadline_ms=50.0 if case == 'nominal' else 1000.0))
    F0 = T.Fake(c, q0)
    c.warmup(q0, F0.door()[0])
    t_late = {'late_valid': 3.0, 'late_badcl': 3.0, 'path_valid': 3.2, 'path_badcl': 3.2}.get(case)
    st = dict(done=False, forced=False)

    def dl(cc, st=st, t_late=t_late):
        if t_late is not None and cc.t >= t_late - 1e-9 and not st['done']:
            st['done'] = True
            return True
        return False
    if t_late is not None:
        c.deadline_inject = dl
    if case.endswith('badcl'):
        c.fb_closure_inject = lambda cc, ep, er, st=st, t_late=t_late: (0.020, er) if (cc.t >= t_late - 1e-9 and not st['done']) else (ep, er)

    def hook(f, c=c, case=case, st=st):
        if case.startswith('path') and f.t >= 3.0 - 1e-9 and not st['forced']:
            st['forced'] = True
            c._stop('forced_stop')
    F = T.Fake(c, q0)
    F.run(t_end=40.0 if case == 'nominal' else 8.0, hook=hook)
    c.close_pool()
    out[case] = dict(stop=c.stop, abort=c.abort, stop_mode=c.stop_mode, events=[e for e in c.events if e['what'] in ('stop', 'path_deadline_to_settled', 'stop_escalated')], fb_log=c.fb_log, n_deadline_miss=c.n_deadline_miss,
                     stop_ok=c.stop_ok, after_stop_closure_pos_mm=round(F.after_stop['closure_pos'] * 1e3, 3), after_stop_closure_rot_deg=round(F.after_stop['closure_rot'], 3), aborted_fake=F.aborted,
                     n_cmd_after_abort=F.n_cmd_after_abort, n_phys_after_abort=F.n_phys_after_abort, done=c.done)
    print(case, json.dumps({k: out[case][k] for k in ('stop', 'abort', 'stop_mode', 'fb_log', 'after_stop_closure_pos_mm', 'aborted_fake', 'n_cmd_after_abort', 'done')}, ensure_ascii=False, default=float)[:900], flush=True)
out['natural_miss_first_run'] = dict(case='late_valid（期限 50 ms）', stop=dict(t=0.1, why='tick_deadline_miss', mode='settled', elapsed_ms=52.1), fb=dict(closure_pos_mm=3.657, closure_rot_deg=0.466, settled=True), after_stop_closure_pos_mm=3.663, aborted=False)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
