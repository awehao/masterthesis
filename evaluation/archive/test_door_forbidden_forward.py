#!/usr/bin/env python3
"""禁止接觸轉送核對（Codex 20261112_232809 第 3 點）：真 BindGate 鎖存 → 真 forward_forbidden → 真 DoorOpenExec._stop／_ev。
核：舊寫法（**gate.latched）確實崩潰；新寫法鎖存模擬中止、保留原事件時間與原因、之後 step 不再產生命令、report 可經執行器自己的 clean() 存成 JSON。
    python3 evaluation/test_door_forbidden_forward.py PLAN.npz OUT.json"""
import ast, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_bind_logic as BL
import door_open_exec_ctrl as XO
import test_door_open_exec as T

src = open(os.path.join(HERE, 'isaac_m8_house_door_open.py')).read()
fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'clean'][0]
ns = dict(np=np, math=math)
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'clean', 'exec'), ns)            # 執行器自己的 clean()
clean = ns['clean']

P = np.load(sys.argv[1])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)   # 同執行器 COND


def mk():
    q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
    c = XO.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q)
    return c, q0


out = {}
# 1 舊寫法：確認會崩潰
c, q0 = mk()
g = BL.BindGate(COND, 0.01)
g.step_seq(1.23, np.eye(4), 0.8, 0.0)                                    # 非手指接觸 0.8 N > 0.05 ⇒ 鎖存
assert g.latched is not None and g.latched['why'] == 'nonfinger_contact', g.latched
try:
    c._stop('forbidden_contact', **g.latched)
    out['old_form'] = 'no_error（未重現）'
except TypeError as e:
    out['old_form'] = f'TypeError: {e}'
# 2 新寫法：開門中途（跑 0.6 s 後）鎖存
c, q0 = mk()
F = T.Fake(c, q0)
F.run(t_end=0.6)
n_before = len(c.step_log)
g = BL.BindGate(COND, 0.01)
g.step_seq(F.t, np.eye(4), 0.8, 0.0)
info = BL.forward_forbidden(g.latched, 'link4', ctrl=None, ex=c)
o = c.step(F.t + 0.01, F.q, F.phi_hist[-1] if F.phi_hist else 0.0, 5.0)
rep = c.report()
s = json.dumps(clean(dict(O_open=rep, forbidden=dict(latched=g.latched, events=[info]))), ensure_ascii=False)
out['new_form'] = dict(abort=c.abort, stop=c.stop, stop_mode=c.stop_mode, next_step_abort=bool(o['abort']), next_step_base_twist=o['base_twist'].tolist(), step_log_grew=len(c.step_log) - n_before,
                       stop_ok=c.stop_ok, events_tail=c.events[-2:], json_len=len(s), bound=c.bound, unbind_t=c.unbind_t)
ab = c.abort or {}
ci = ab.get('contact_info', {})
out['checks'] = dict(old_form_crashes=out['old_form'].startswith('TypeError'), abort_latched=bool(c.abort is not None and c.stop_mode == 'abort' and c.stop['why'] == 'forbidden_contact'),
                     original_time_kept=bool(ci.get('t') == g.latched['t']), original_why_kept=bool(ci.get('why') == 'nonfinger_contact'), who_kept=bool(ci.get('who') == 'link4'),
                     next_step_no_command=bool(o['abort'] and not np.any(o['base_twist']) and len(c.step_log) == n_before), not_reported_success=bool(c.stop_ok is not None and c.stop_ok.get('ok') is None),
                     no_unbind=bool(c.bound and c.unbind_t is None), json_ok=bool(len(s) > 100), config_readback=XO.DoorOpenExec.config_readback(c))
out['checks']['config_readback'] = {k: v for k, v in out['checks']['config_readback'].items() if 'sha' not in k}
out['pass'] = bool(all(v for k, v in out['checks'].items() if k != 'config_readback'))
json.dump(clean(out), open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(clean(out['checks']), ensure_ascii=False), 'PASS' if out['pass'] else 'FAIL', out['old_form'])
