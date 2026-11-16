#!/usr/bin/env python3
"""停止核對的離線核對（Codex 20261116 條件）：(1) 處置分支單元測試；(2) 用已錄的乾跑趟（每拍實際的位姿、設定點、在途命令、放行命令、提交的手臂命令）重放現版 stop_check。
    python3 evaluation/far_dry/stop_check_replay.py DIR PREFIX OUT.json"""
import json, multiprocessing as mp, os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'far_dry')); sys.path.insert(0, HERE)
import numpy as np
import merge_node as MN

SD = MN.stop_disposition
ok, bad = dict(ok=True), dict(ok=False)
cases = [('both_ok', ('tick', dict(now=ok, next=ok)), 'commit'), ('next_fail_now_ok', ('tick', dict(now=ok, next=bad)), 'stop_now'), ('now_fail', ('tick', dict(now=bad, next=ok)), 'abort'), ('both_fail', ('tick', dict(now=bad, next=bad)), 'abort'),
         ('late', ('late', None), 'abort'), ('worker_error', ('error', 'x'), 'abort'), ('result_missing', ('tick', None), 'abort'), ('sweep_unavailable', ('tick', dict(now=ok, next=ok, why='sweep_timeout')), 'abort'), ('partial', ('tick', dict(now=ok)), 'abort'),
         ('ok_not_bool', ('tick', dict(now=dict(ok=1), next=dict(ok=1))), 'abort')]
unit = [dict(case=n, got=SD(*x), want=w, **{'pass': SD(*x) == w}) for n, x, w in cases]
D, PF = sys.argv[1], sys.argv[2]
J = json.load(open(os.path.join(D, PF + '_merge.json')))
Z = np.load(os.path.join(D, PF + '_merge.npz')); R, c = Z['R'], [str(x) for x in Z['cols']]; i = c.index
Wd = np.load(os.path.join(D, PF + '_world.npz')); L, wc = Wd['L'], [str(x) for x in Wd['cols']]
iu = wc.index('uin_x')
ctx = mp.get_context('fork'); pc, cc = ctx.Pipe(); wp = ctx.Process(target=MN.worker, args=(cc,)); wp.start(); pc.recv()
an = J['anchor']
pc.send(('prior', dict(ver=1))); pc.recv()
pc.send(('anchor', dict(ver=2, T_odom_base=MN._T3(an['odom_base']).tolist(), T_base_handle=an['T_base_handle'], T_mo=an['map_odom']))); pc.recv()
arm, st = R[:, i('arm(0HOLD,1PREP,2TRACK)')], R[:, i('stop')]
rows = np.where((arm > 0) & (st == 0) & np.isfinite(R[:, i('stop_ok')]))[0]
out_rows, t_ms = [], []
for r_ in rows:
    k = int(round(R[r_, i('t_world')] / 0.01)) - 1
    if k < 8 or abs(L[k, 0] - R[r_, i('t_world')]) > 1e-6:
        continue
    pipe = L[k - 7:k + 1, iu:iu + 9]
    pc.send(('stop_check', dict(ver=3, q=R[r_, 2:11].tolist(), pipe=pipe.tolist(), rel=R[r_, i('rel_x'):i('rel_x') + 3].tolist(), s=R[r_, 11:17].tolist(), ua=R[r_, i('ua0'):i('ua0') + 6].tolist(), T_mo=R[r_, i('mo_x'):i('mo_x') + 3].tolist())))
    tag, _, sc = pc.recv()
    if tag != 'stop_check':
        out_rows.append(dict(t=float(R[r_, 0]), error=str(sc)[:200])); continue
    t_ms.append(sc['ms'])
    out_rows.append(dict(t=float(R[r_, 0]), arm=int(arm[r_]), disposition=SD('tick', sc), ok=sc['ok'], why=sc.get('why'), now_ok=sc['now']['ok'], next_ok=sc['next']['ok'], settled=[sc['now']['settled'], sc['next']['settled']], coverage=[sc['now']['coverage_ok'], sc['next']['coverage_ok']],
                         margin=sc['margin_after_interval_bound'], min_clear=sc['min_clear'], n_eval=sc['n_eval'], n_refine=sc['n_refine'], ms=sc['ms'], recorded_ok=bool(R[r_, i('stop_ok')] == 1)))
pc.send(None)
from collections import Counter
good = [o for o in out_rows if 'error' not in o]
summ = dict(source=os.path.join(D, PF), n_replayed=len(out_rows), n_error=len(out_rows) - len(good), disposition=dict(Counter(o['disposition'] for o in good)), by_arm={str(a_): dict(Counter(o['disposition'] for o in good if o['arm'] == a_)) for a_ in (1, 2)},
            n_refine_ticks=sum(1 for o in good if o['n_refine']), margin_lower_bound_min={k_: round(min(o['margin'][k_] for o in good), 5) for k_ in ('env', 'base', 'self')} if good else None,
            ms=dict(p50=round(float(np.median(t_ms)), 1), p95=round(float(np.percentile(t_ms, 95)), 1), max=round(float(np.max(t_ms)), 1), note='離線重放的牆鐘（無 ROS 負載）；期限 45 ms') if t_ms else None,
            non_commit=[o for o in good if o['disposition'] != 'commit'][:10], note='自碰撞區間界：逐對、只計兩連桿間關節，槓桿長用對所有構型成立的上界 LEVER（URDF 固定長度三角不等式，不超過 0.75 m）；逐對精確距離快取鍵為實際位元組')
json.dump(dict(unit_stop_disposition=unit, unit_all_pass=all(u['pass'] for u in unit), replay=summ, rows=out_rows), open(sys.argv[3], 'w'), ensure_ascii=False, indent=1)
print('unit_all_pass', all(u['pass'] for u in unit), [u['case'] for u in unit if not u['pass']])
print(json.dumps(summ, ensure_ascii=False)[:1500])
