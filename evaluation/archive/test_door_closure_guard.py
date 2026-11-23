#!/usr/bin/env python3
"""A（arm_mode='closure'）真鏈核對（Codex 20261113_003032 第 2 點）：
1 替換命令確實進管線與提交前掃掠：包住真 _sweep，記錄 'candidate' 收到的 u；核它的手臂部分＝閉鏈輸出、＝之後提交的 u_last、＝進管線的命令。
2 反例「舊 QP 合格、替換命令違反限制」：開門中途把可操作度下限 W.SIG_FLOOR 提到目前 IK 構型 σ 之上（QP 的 arm_check 不含可操作度 ⇒ QP 照樣合格），
  核：同一拍提交前攔下（closure_cmd_invalid、模擬中止鎖存）、沒有新命令進管線、之後不再步進物理、不解綁、不報停止成功。
    python3 evaluation/test_door_closure_guard.py PLAN.npz OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
import test_door_open_exec as T
P = np.load(sys.argv[1])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))


def mk():
    q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
    c = XO.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q, arm_mode='closure')
    return c, q0


out = {}
# 1 替換命令進管線／掃掠
c, q0 = mk()
seen = []
orig = c._sweep
def spy(q, u, extras, tag, **kw):
    if tag == 'candidate':
        seen.append(np.asarray(u, float).copy())
    return orig(q, u, extras, tag, **kw)
c._sweep = spy
orig_cl = c._closure_arm
cl_out = []
def spy_cl(*a, **k):
    r = orig_cl(*a, **k)
    cl_out.append(None if r is None else r.copy())
    return r
c._closure_arm = spy_cl
F = T.Fake(c, q0)
commits, pipes = [], []
for _ in range(int(round(1.0 / 0.01))):
    tick = c.n_step % XO.NPH == 0
    F.one()
    if tick and c.stop is None:
        commits.append(c.u_last.copy()); pipes.append(np.asarray(c.cmd_hist[-1]).copy() if c.cmd_hist else None)
n = min(len(seen), len(cl_out), len(commits))
d_sweep = max(float(np.abs(seen[i][3:] - cl_out[i]).max()) for i in range(n))
d_commit = max(float(np.abs(commits[i][3:] - cl_out[i]).max()) for i in range(n))
d_pipe = max(float(np.abs(np.asarray(c.step_log)[k][14 + 3:14 + 9] - np.asarray(c.step_log)[k][5 + 3:5 + 9]).max()) for k in range(len(c.step_log)))   # 進管線命令的手臂部分＝提交命令手臂部分（手臂不經斜坡）
out['replaced_cmd_flow'] = dict(ticks=n, max_diff_candidate_sweep_vs_closure=d_sweep, max_diff_commit_vs_closure=d_commit, max_diff_pipe_in_vs_commit_arm=d_pipe, closure_stats={k: v for k, v in c.cl_stats.items() if k != 'fails'}, anchor=c.anchor)
ok1 = d_sweep == 0.0 and d_commit == 0.0 and d_pipe == 0.0 and n >= 15
# 2 反例
c, q0 = mk()
F = T.Fake(c, q0)
F.run(t_end=1.0)
assert c.stop is None, c.stop
sig_now = c.cl_stats['sigma_min']
floor_save = XO.W.SIG_FLOOR
XO.W.SIG_FLOOR = 1.0                                                  # 收緊到任何構型之上（σ_min 遠小於 1；QP 的 arm_check 不核可操作度）
u_before, pipe_before, nlog = c.u_last.copy(), [p.copy() for p in c.pipe], len(c.step_log)
qp_ok_seen = []
orig_solve = c.mpc.solve
def spy_solve(*a, **kw):
    r = orig_solve(*a, **kw)
    if not (kw.get('arm') or {}).get('mode') == 'active':
        return r
    ck = dict(r.arm_check or {})
    qp_ok_seen.append(bool(not ck.get('fallback') and not ck.get('failed') and ck.get('seq_consistent', False) and r.u_arm_seq is not None))
    return r
c.mpc.solve = spy_solve
while c.stop is None and F.t < 2.0:
    F.one()
XO.W.SIG_FLOOR = floor_save
t_before = F.t
F.one()                                                              # 中止後執行器入口再呼叫一次：不得步進物理、不得送命令
if F.t != t_before:
    F.n_phys_after_abort += 1
rep = c.report()
k_stop = next(i for i, r in enumerate(c.step_log) if i >= nlog and r[1] > 0.5)     # 觸發那一拍
row = np.asarray(c.step_log[k_stop])
out['counterexample'] = dict(stop=c.stop, abort=c.abort, qp_arm_check_ok_at_trigger=qp_ok_seen[-1] if qp_ok_seen else None, sigma_now=sig_now, floor_injected=1.0,
                             u_last_unchanged=bool(np.array_equal(c.u_last, u_before)),
                             phys_after_abort=F.n_phys_after_abort, cmd_after_abort=F.n_cmd_after_abort, bound=c.bound, unbind_t=c.unbind_t, stop_effect=rep['stop_effect'])
ok2 = (c.stop is not None and c.stop['why'] == 'closure_cmd_invalid' and c.stop_mode == 'abort' and any(b['what'] == 'manipulability' for b in c.stop['bad']) and qp_ok_seen and qp_ok_seen[-1]
       and np.array_equal(c.u_last, u_before) and F.aborted and F.n_phys_after_abort == 0 and F.n_cmd_after_abort == 0 and c.bound and c.unbind_t is None and rep['stop_effect'].get('ok') is None)
out['checks'] = dict(replaced_cmd_into_sweep_commit_pipe=bool(ok1), counterexample_blocked_before_commit=bool(ok2))
out['pass'] = bool(ok1 and ok2)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out['checks']), 'PASS' if out['pass'] else 'FAIL'); print(json.dumps(out['counterexample'], ensure_ascii=False, default=float)[:900]); print(json.dumps({k: v for k, v in out['replaced_cmd_flow'].items() if k != 'anchor'}, default=float)[:600])
