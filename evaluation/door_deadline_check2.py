#!/usr/bin/env python3
"""抓持 r7、退出 r7 期限政策核對（Codex 20261113_130916 §4；事前登錄）。注入案期限設 1000 ms（只有注入那拍晚到）；自然計時案 50 ms。
  抓持（0.05 m/s 名目，pool＝6、fastkin）：g_ins_valid 相對 0.8 s（INSERT、未鎖定）強制晚到 ⇒ 原停法停止；g_conf_valid 1.9 s（CONFIRM、已鎖定）⇒ 接觸停法停止；
     g_conf_invalid 同 1.9 s 且備援視為未驗證 ⇒ 模擬中止；g_natural50 不注入、50 ms。
  退出（full_2 開門終態承接，pool＝6、預熱）：x_valid t≥2.0 s 強制晚到 ⇒ 停止；x_invalid 同時備援未驗證 ⇒ 中止；x_natural50。
記：停止原因、中止、停止有效、晚到次數、正常拍／停止拍耗時分列、TCP 對原抓取位姿終值（抓持）。
    python3 evaluation/door_deadline_check2.py OUT.json"""
import json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SP = os.environ.get('SP', '/tmp/claude-1000/-home-howardchen-masterthesis/2a47f71f-1a91-4ff0-9dbf-1892d8540042/scratchpad')
OUT = sys.argv[1]
W = os.path.join(SP, 'dl2'); os.makedirs(W, exist_ok=True)


def tstats(tl, stopped):
    a = np.array([m for _, m, s in tl if s == stopped], float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1), n_over_50=int((a > 50).sum()))


res = {}
for tag, env in (('g_ins_valid', dict(DEADLINE_MS='1000', DL_INJECT_T='0.8')), ('g_conf_valid', dict(DEADLINE_MS='1000', DL_INJECT_T='1.9')),
                 ('g_conf_invalid', dict(DEADLINE_MS='1000', DL_INJECT_T='1.9', FB_INVALID='1')), ('g_natural50', dict(DEADLINE_MS='50'))):
    e = dict(os.environ, POOL='6', FASTKIN='1', RQ_CASE='r_delay', **env)                 # r_delay：不接受交接，READY 後逾時停止；晚到注入在其前
    e.pop('VIRTUAL_ENV', None); e['PATH'] = '/usr/local/bin:/usr/bin:/bin'
    subprocess.run(['systemd-run', '--user', '--scope', '-q', '-p', 'CPUQuota=700%', '-p', 'MemoryMax=6G', 'env'] + [f'{k}={v}' for k, v in e.items() if k in ('POOL', 'FASTKIN', 'RQ_CASE', 'DEADLINE_MS', 'DL_INJECT_T', 'FB_INVALID', 'PATH')] +
                   ['/usr/bin/python3', os.path.join(HERE, 'test_door_motm_grasp_fake.py'), 'nominal', '0.05', f'{W}/{tag}.json'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    d = json.load(open(f'{W}/{tag}.json'))['deadline']
    res[tag] = dict(stop=(d['stop'] or {}), abort=d['abort'], stop_effect=d['stop_effect'], n_miss=d['n_miss'], tcp_vs_original_end_mm=d['tcp_vs_original_end_mm'],
                    ticks_normal=tstats(d['tick_log'], False), ticks_stop=tstats(d['tick_log'], True))
    print(tag, json.dumps(res[tag], ensure_ascii=False, default=float)[:600], flush=True)
# 退出
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
import door_exit_ctrl as XE
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
L = np.load(RUN + '/log.npz')['rows']; Pp = np.load(RUN + '/plan.npz')
R7 = L[L[:, 1] == 7]; qx = np.r_[R7[-1, 5:8], R7[-1, 8:14]]; phx = float(R7[-1, 2]); om = float((R7[-1, 2] - R7[-11, 2]) / 0.1)
sx = np.load(RUN + '/open_steps.npz')['ctrl'][-1, 32:38]
for tag, dl, inj, inval in (('x_valid', 1000.0, 2.0, False), ('x_invalid', 1000.0, 2.0, True), ('x_natural50', 50.0, None, False)):
    xc = XE.DoorExitCtrl(ident, Pp['T_WH_bind'], float(Pp['phi_bind']), sx, [np.zeros(9)] * 8, np.zeros(9), np.zeros(9), al * (qx[3:] - sx), 0.0089, phx, om, np.zeros((1, 2)), qx[3:], scope_end='X2', pool=6, deadline_ms=dl)
    xc.warmup(qx, phx)
    st = dict(done=False)
    if inj is not None:
        def _inj(cc, st=st, inj=inj):
            if not st['done'] and cc.t >= inj - 1e-9:
                st['done'] = True
                return True
            return False
        xc.deadline_inject = _inj
        if inval:
            xc.fb_inject = lambda cc, st=st, inj=inj: not (cc.t >= inj - 1e-9 and not st['done'])
    q = qx.copy(); n_after = 0
    for k in range(1500):
        o = xc.step(k * 0.01, q.copy(), phx)
        if o.get('abort'):
            n_after += int(bool(np.any(o['base_twist'])))
            break
        if (xc.stop is not None and xc.stop_ok is not None) or (xc.done and xc.stop_ok is not None):
            break
        cs, sn = math.cos(q[2]), math.sin(q[2]); vx, vy, wz = o['base_twist']
        q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz]); q[3:] += al * (o['arm_target'] - q[3:] + (qx[3:] - sx))
    xc.close_pool()
    res[tag] = dict(stop=xc.stop, abort=xc.abort, stop_effect=xc.stop_ok, phase=xc.phase, n_miss=xc.n_deadline_miss, n_cmd_after_abort=n_after,
                    ticks_normal=tstats(xc.tick_log, False), ticks_stop=tstats(xc.tick_log, True))
    print(tag, json.dumps(res[tag], ensure_ascii=False, default=float)[:600], flush=True)
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
