#!/usr/bin/env python3
"""多任務 v2：U 段採用回歸＋新排程換段附近逐控制拍停止核對（Codex 20261102_053914 第 1、3 項；假世界、假候選，非物理）。
物理 A1／A2 都在 V_HIGH 上升段 U 開始後約 3.6 s 採用（離線批次只在保持段 H 送候選）。本檔讓假候選從 U 開始後 d 秒起送（U、H 段皆送，直到採用），
d＝3.6（物理實測時機）與 0.5（U 剛開始），案例 A1、A2、A3、A4、B2（C1／C2 另含動態物核對）。
第 1 項（U 段採用相容）：V_HIGH 序列恰為 U→H→R→DONE 各一次（展臂不重置、U 時長不變）、採用時段＝U、R 由平移後 x_dep 觸發、DONE 在漸變前、
  漸變起點位置對平移後 x_blend、採用前後 ±0.5 s 的設定點速率與命令變化率對全程峰值（參考連續）、最終結果。
第 3 項（停止核對範圍如實）：模擬時逐控制拍存控制器狀態（q、設定點、管線、上一命令、手指命令、持瓶相對位姿），跑完後挑控制拍做兩條 stop_traj（現在停／延後一拍停）：
  基底＝U 開始到離桌成立＋0.5 s 每 10 拍；另加逐拍窗（±5 拍）於：relwin_shift、採用、V_HIGH 各段、漸變起、各相位切換、liftb_switch、目標鎖定、閉爪、離桌成立；
  非手指連桿對桌淨空的全程最小與距最小 5 mm 內的局部最小；命令變化率 ≥ 本趟峰值 60% 的控制拍。判準同 test_bottle_w1_relwin_batch（不放寬）。
這仍是取樣幾何核對，不是全流程停止保證。
    python3 evaluation/test_bottle_w1_relwin_uadopt.py all
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, 'experiments', '20261102_multi_motm_tasks.json')))
CASES = {c['id']: (c['table1_z_m'], c['bottle_dx_m'], c['bottle_dy_m']) for c in M['stage_1']['cases']}
OUT = os.path.join(HERE, 'results', 'bottle', 'relwin_uadopt_offline.json')
RUNS = [(k, d) for k in ('A1', 'A2', 'A3', 'A4', 'B2') for d in (3.6, 0.5)]

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(kd):
        p = subprocess.run([sys.executable, __file__, kd[0], str(kd[1])], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return f'{kd[0]}@U+{kd[1]}', (json.loads(line[-1]) if line else dict(case=kd[0], d=kd[1], error=p.stderr[-2000:]))
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(one, RUNS))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, r in res.items():
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 'adopt', 'u_compat', 'continuity', 'stop_check_n', 'stop_fails', 'error')})
    sys.exit(0)

CASE, D_U = sys.argv[1], float(sys.argv[2])
Z1, DXB, DYB = CASES[CASE]
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402  （依 argv 的 Z1 在匯入時改第一桌幾何）
import bottle_branch as BB                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_vhigh_relwin_ctrl as RW                # noqa: E402

T, TC = TV.T, TV.TC
TB = [(c_, s_) for _, c_, s_ in G.table_boxes()]
BPT = G.bottle_points()


def chk(q9, T_rel, f_cmd):
    """同 test_bottle_w1_relwin_batch.chk（該檔匯入期即依 argv 選案例，故不匯入、照抄）。"""
    W = G.link_world(q9, f_cmd)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for c_, s_ in TB), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        Pb = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
        out['held_bottle_table'] = min(float(G.box_dist(Pb, c_, s_).min()) for c_, s_ in TB)
    return out
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
EV_HOT = ('relwin_shift', 'segtarget_freeze', 'vhigh_seq', 'blend_start', 'phase', 'liftb_switch', 'segtarget_lock', 'close_request', 'unfold_start')
REC = []        # 每控制拍：t、相位、q、s、pipe、u_last、f_cmd、T_rel（離桌成立後）、命令


class Ctl(RW.W1PlaceController):
    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all():
            REC.append(dict(t=float(self.t), n=int(self.n_step), ph=self.st.phase, q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.t_lift_ok is None else np.asarray(self.judge.T_rel, float).copy()))
        super()._advance_inner(T_tcp, close_done, q)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, n_sent=0, ph_at_freeze=None, t_first_send=None)
    step0 = c.step

    def step(t, q, **kw):
        sq = c.vh_seq
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if sq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
                st['n_sent'] += 1
                st['t_first_send'] = st['t_first_send'] if st['t_first_send'] is not None else t
        out = step0(t, q, **kw)
        if c.vfix is not None and st['ph_at_freeze'] is None:
            st['ph_at_freeze'] = c.vh_seq['ph']
        return out
    c.step = step
    r, _ = T.run(c, f, t_end=75.0)
    return c, f, r, st


def main():
    c, f, r, st = run()
    ev = r['events']
    vs = r['vhigh_seq']['log']
    phs = [e[1] for e in vs]
    t_u = next((e[0] for e in vs if e[1] == 'U'), None)
    u_T = next((e[3] for e in vs if e[1] == 'U'), None)
    rw = r.get('relwin') or {}
    fz = next((e for e in ev if e['what'] == 'segtarget_freeze'), None)
    bl = next((e for e in ev if e['what'] == 'blend_start'), None)
    t_of = {e[1]: e[0] for e in vs}
    t_lok = r.get('t_lift_ok')
    done = bool(r.get('mission_done'))
    why = (r['stop'] or {}).get('why')
    # 第 1 項：U 段採用相容
    t_r = t_of.get('R')
    x_r = next((e[2] for e in vs if e[1] == 'R'), None)
    u_compat = dict(order=phs, order_ok=phs == ['U', 'H', 'R', 'DONE'], n_U=phs.count('U'), U_duration=u_T, ph_at_freeze=st['ph_at_freeze'], t_first_send=st['t_first_send'],
                    t_adopt=None if fz is None else fz['t'], x_dep_new=(rw.get('new') or {}).get('x_dep'), x_at_R=x_r, R_after_x_dep=None if x_r is None or not rw else bool(x_r >= rw['new']['x_dep'] - 1e-3),
                    t_done_seq=t_of.get('DONE'), t_blend=None if bl is None else bl['t'], done_before_blend=None if bl is None or 'DONE' not in t_of else bool(t_of['DONE'] <= bl['t'] + 1e-9),
                    x_blend_new=(rw.get('new') or {}).get('x_blend'), x_at_blend=None if bl is None else bl.get('x'),
                    blend_lag_m=None if bl is None or not rw else round(float(bl['x']) - float(rw['new']['x_blend']), 4))
    # 參考連續：設定點速率與命令變化率，採用前後 ±0.5 s vs 全程（U 開始到離桌成立）
    tt = np.array([x['t'] for x in REC])
    S = np.array([x['s'] for x in REC])
    U = np.array([x['u_last'] for x in REC])
    dt = np.diff(tt)
    ok_dt = (dt > 1e-6) & (dt < 0.11)
    ds = np.where(ok_dt[:, None], np.abs(np.diff(S[:, 3:], axis=0)) / np.maximum(dt, 1e-6)[:, None], 0.0).max(axis=1)
    du = np.where(ok_dt[:, None], np.abs(np.diff(U[:, 3:], axis=0)) / np.maximum(dt, 1e-6)[:, None], 0.0).max(axis=1)
    t_end = t_lok + 0.5 if t_lok is not None else tt[-1]
    span = (tt[1:] >= (t_u or 0.0)) & (tt[1:] <= t_end)
    near = span & (np.abs(tt[1:] - (fz['t'] if fz else -99)) <= 0.5)
    continuity = dict(ds_peak_span=round(float(ds[span].max()), 4), ds_peak_adopt=round(float(ds[near].max()), 4) if near.any() else None,
                      du_peak_span=round(float(du[span].max()), 4), du_peak_adopt=round(float(du[near].max()), 4) if near.any() else None,
                      tick_dt=[round(float(dt[ok_dt].min()), 4), round(float(np.median(dt[ok_dt])), 4), round(float(dt[ok_dt].max()), 4)], n_rec=len(REC))
    # 第 3 項：挑控制拍
    i_span = [i for i, x in enumerate(REC) if (t_u or 0.0) <= x['t'] <= t_end]
    sel = set(i_span[::10])
    ev_t = sorted({float(e['t']) for e in ev if e['what'] in EV_HOT} | ({float(t_lok)} if t_lok is not None else set()))
    for te in ev_t:
        k = int(np.argmin(np.abs(tt - te)))
        sel.update(i for i in range(k - 5, k + 6) if i in set(i_span))
    clr = np.full(len(REC), np.inf)
    for i in i_span:
        clr[i] = chk(REC[i]['q'], None, REC[i]['f_cmd'])['nonfinger_table']
    m0 = float(clr.min())
    loc = [i for i in i_span if clr[i] <= m0 + 0.005 and clr[i] <= clr[max(i - 1, 0)] and clr[i] <= clr[min(i + 1, len(REC) - 1)]]
    for k in loc:
        sel.update(i for i in range(k - 5, k + 6) if i in set(i_span))
    du_full = np.r_[0.0, du]
    dmax = float(du_full[i_span].max())
    hot = [i for i in i_span if du_full[i] >= 0.6 * dmax]
    for k in hot:
        sel.update(i for i in range(k - 5, k + 6) if i in set(i_span))
    worst, where, nonfin, unfin = {}, {}, 0, 0
    for i in sorted(sel):
        x = REC[i]
        for e in (0, 1):
            Q = C21.stop_traj(x['q'], x['s'], x['pipe'], x['u_last'], c.cfg, e)
            if len(Q) >= 2000 // C21.SAMPLE_EVERY + 1:
                unfin += 1
            if not np.isfinite(Q).all():
                nonfin += 1
                continue
            for qq in Q[::2]:
                for kk, v in chk(qq, x['T_rel'], x['f_cmd']).items():
                    key = f'e{e}:{kk}'
                    if v < worst.get(key, 9.9):
                        worst[key], where[key] = v, (round(x['t'], 3), x['ph'])
    fails = sorted(k for k, v in worst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
    p_ok = bool(done and u_compat['order_ok'] and u_compat['n_U'] == 1 and u_compat['ph_at_freeze'] == 'U' and u_compat['done_before_blend'] and u_compat['R_after_x_dep'] and not fails
                and T.finite_cmds(f))
    out = dict(case=CASE, d=D_U, z1=Z1, bottle=[DXB, 0.42 + DYB], pass_=p_ok, outcome='完整抓放' if done else f'停止 {why}', t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               adopt=dict(t=None if fz is None else fz['t'], ph=st['ph_at_freeze'], dx=rw.get('dx'), x_now=rw.get('x_now')), u_compat=u_compat, continuity=continuity,
               stop_check_n=len(sel), stop_check_span=[t_u, t_end], stop_check_sel=dict(base_every10=len(i_span[::10]), events=len(ev_t), clear_local_min=len(loc), clear_min_m=round(m0, 4),
                                                                                         du_hot=len(hot), du_max=round(dmax, 3)),
               stop_worst={k: round(v, 5) for k, v in worst.items()}, stop_where=where, stop_fails=fails, limits=LIM,
               note='假世界、假候選（U 段起送）；取樣控制拍核對，非全流程停止保證')
    out['pass'] = out.pop('pass_')
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
