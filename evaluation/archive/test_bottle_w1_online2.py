#!/usr/bin/env python3
"""區域導引在線 MotM 第二版（bottle_w1_online2_ctrl）承接新鮮度的小核對（Codex 20261102_075357；假世界、假候選，非物理）。
幾何同 test_bottle_w1_online（B1 位置，第一批已用點）；假世界節流 RTF ≤ 0.6；工作者是真的子程序。
  closed  正常：新鮮結果照常承接、完整抓放；報承接當下來源年齡
  （gap／stale 的抓取結果延到請求後 0.7 s 才交付，模擬規劃較慢；否則結果約 0.1 s 就到，影像中斷前已承接、測不到）
  gap     抓取請求送出後影像（候選）停 1.0 s 再恢復 ⇒ 期望：結果到時觀測年齡 > 0.5 s 不採用（grasp_plan_wait_obs），恢復後以新鮮觀測承接（承接時刻 ≥ 恢復時刻、年齡 ≤ 0.5 s）、完整抓放
  stale   抓取請求送出後 1.5 s 內只重送請求前最後一筆的擷取（age_s 隨時間增加）再恢復 ⇒ 期望：重送不刷新年齡（入口拒收 stale、不承接），恢復後承接、完整抓放
    python3 evaluation/test_bottle_w1_online2.py all
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'online2_offline.json')
MODES = ('closed', 'gap', 'stale')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(mode=k, error=p.stderr[-2500:]))
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(one, MODES))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in MODES:
        print(k, {kk: res[k].get(kk) for kk in ('pass', 'expect', 'outcome', 't_done', 'gp', 'resume', 'rejected', 'error')})
    sys.exit(0)

MODE = sys.argv[1]
Z1, DXB, DYB, D_U, RTF_CAP = 0.55, -0.04, 0.02, 3.6, 0.6
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_w1_online2_ctrl as ON2                    # noqa: E402

T = TV.T
LOGP = os.path.join(HERE, 'results', 'bottle', f'online2_offline_{MODE}_worker.log')


class Ctl(ON2.W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, wall0=None, t_gap0=None, resume=None, last=None, n_rej0=None)
    step0 = c.step
    if MODE in ('gap', 'stale'):                                               # 結果延到請求後 0.7 s 才交付（模擬規劃較慢），使結果到時觀測已過期
        _p = c.pc.poll
        c.pc.poll = lambda rid: None if (c.gp is not None and rid == c.gp['id'] and c.t < c.gp['t_req'] + 0.7) else _p(rid)

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = time.time() - t
        lag = t - (time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            time.sleep(lag / RTF_CAP)
        if st['t_gap0'] is None and c.gp is not None:
            st['t_gap0'] = t
            st['n_rej0'] = dict(c.seg_rej)
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                hold = st['t_gap0'] is not None and MODE in ('gap', 'stale') and t < st['t_gap0'] + (1.0 if MODE == 'gap' else 1.5)
                if hold and MODE == 'gap':
                    pass                                                       # 影像停：不送候選
                elif hold and MODE == 'stale':
                    L = st['last']                                             # 只重送請求前最後一筆（擷取時刻固定，年齡照實增加）
                    c.candidate_intake(t, dict(L, age_s=float(t - L['_t_cap'])))
                else:
                    if st['t_gap0'] is not None and st['resume'] is None and MODE in ('gap', 'stale'):
                        st['resume'] = t
                    xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                    if st['snap'] is None or t - st['t_seq'] >= 3.0:
                        st['seq'] += 1
                        st['t_seq'], st['snap'] = t, xy.copy()
                    cnd = dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                               snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0)
                    st['last'] = dict(cnd, _t_cap=t - 0.05)
                    c.candidate_intake(t, cnd)
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=80.0)
    return c, f, r, st


EXPECT = dict(closed='新鮮結果照常承接、完整抓放', gap='影像停 1 s：結果到時不採用、恢復後以新鮮觀測承接、完整抓放',
              stale='重送舊擷取不刷新年齡（入口拒收）、恢復後承接、完整抓放')


def main():
    c, f, r, st = run()
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    on = r.get('online') or {}
    gps = on.get('grasp_plans') or []
    ga = next((g for g in gps if g.get('state') == 'accepted'), None)
    rej = {k: v - (st['n_rej0'] or {}).get(k, 0) for k, v in c.seg_rej.items()}
    out = dict(mode=MODE, expect=EXPECT[MODE], outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None),
               gp=[{k: g.get(k) for k in ('id', 'state', 't_req', 't_res', 't_wait_obs', 't_acc', 'obs_age_at_accept', 'snap_age_at_accept', 'compute_wall')} for g in gps],
               resume=st['resume'], t_gap0=st['t_gap0'], rejected=rej, latency=on.get('grasp_latency'))
    ok_age = ga is not None and ga.get('obs_age_at_accept') is not None and ga['obs_age_at_accept'] <= 0.5 + 1e-9
    ok = {'closed': done and ok_age,
          'gap': done and ok_age and ga.get('t_wait_obs') is not None and st['resume'] is not None and ga['t_acc'] >= st['resume'] - 1e-9,
          'stale': done and ok_age and st['resume'] is not None and ga['t_acc'] >= st['resume'] - 1e-9 and rej.get('stale', 0) > 0}[MODE]
    out['pass'] = bool(ok and T.finite_cmds(f))
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
