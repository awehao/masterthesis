#!/usr/bin/env python3
"""多任務 v2w：C1／C2 含封存動態物的離線核對（Codex 20261102_053914 第 2 項；假受控對象＋離線射線，非物理）。
C1＝瓶 (−0.06, 0.44)＋w1_dyn_pass1（迎面錯身），C2＝瓶 (+0.06, 0.40)＋w1_dyn_overtake1（後方追越）；第一桌 0.60。
障礙軌跡、速度、起動時刻照 bottle_w1_dyn_case 封存值（任務時刻排程，不追機器人）；掃描＝靜態盒（兩桌）＋移動圓柱（同 test_bottle_w1_dyn）。
控制器＝bottle_w1_vhigh_relwin_slipwarn_ctrl（v2w）；假候選同 test_bottle_w1_relwin_uadopt（U 段開始後 3.6 s 起送）。假世界沒有 Shield 節點（物理另有）。
核對（結果可以是完成或受控停止，不要求必成功）：
  1 實際軌跡：底盤圓盤（半徑 0.39）、全部連桿點、持瓶（假受控對象實際瓶位姿）到障礙圓柱表面（高 1.0 m 以下的點）的最小距離與時刻；
  2 停止路徑：會遇窗（障礙表面距底盤中心 < 1.5 m 的控制拍，前後各 5 拍）逐控制拍兩條 stop_traj（現在停／延後一拍停），
    軌跡每個取樣點對應時刻（t＋取樣偏移）的障礙位置，算連桿點與持瓶（判定器 T_rel＝當下量測）到障礙表面最小距離，判準 > 0（另報最小值）；
    同時照舊核對連桿／手指／持瓶對兩桌與自碰（判準同 test_bottle_w1_relwin_batch）；持瓶對桌在放置段（PARKED 起）不核——瓶本來就要放上第二桌
    （第一版未排除，C2 因障礙起動前停在起點附近使會遇窗涵蓋放置段而誤判失敗，已更正）；
  3 結果、停止原因、會遇窗內最低車速與 0.1 s 速度變化率峰、動態障礙進入控制週期數、命令有限。
    python3 evaluation/test_bottle_w1_relwin_dyn.py all
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, 'experiments', '20261102_multi_motm_tasks.json')))
CASES = {c['id']: (c['table1_z_m'], c['bottle_dx_m'], c['bottle_dy_m'], c['dyn_case']) for c in M['stage_1']['cases']}
OUT = os.path.join(HERE, 'results', 'bottle', 'relwin_dyn_offline.json')

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2000:]))
    with ThreadPoolExecutor(2) as ex:
        res = dict(ex.map(one, ('C1', 'C2')))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, r in res.items():
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 'actual_min', 'stop_dyn', 'stop_static_fails', 'encounter', 'error')})
    sys.exit(0)

CASE = sys.argv[1]
Z1, DXB, DYB, DYN = CASES[CASE]
D_U = 3.6
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_vhigh_relwin_slipwarn_ctrl as SW       # noqa: E402
import test_bottle_place as TP                          # noqa: E402

T = TV.T
CA = DC.CASES[DYN]
TB = [(c_, s_) for _, c_, s_ in G.table_boxes()] + [(c_, s_) for _, c_, s_ in TP.table2_boxes()]
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
REC, TBLOG = [], []
R_BASE = 0.39
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')   # 放置段持瓶本來就要接觸第二桌：持瓶對桌不核（連桿／手指照核）


def obs_d(P, t):
    oc, _ = DC.state(t, CA)
    P = P[P[:, 2] <= CA['height']]
    return float((np.hypot(P[:, 0] - oc[0], P[:, 1] - oc[1]) - CA['radius']).min()) if len(P) else 9.9


def pts(q9, f_cmd, T_rel):
    W = G.link_world(q9, f_cmd)
    P_l = np.vstack(list(W.values()))
    P_b = None
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        P_b = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
    return W, P_l, P_b


def chk_static(q9, T_rel, f_cmd, W, P_b):
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for c_, s_ in TB), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if P_b is not None:
        out['held_bottle_table'] = min(float(G.box_dist(P_b, c_, s_).min()) for c_, s_ in TB)
    return out


class Ctl(SW.W1PlaceController):
    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all():
            REC.append(dict(t=float(self.t), ph=self.st.phase, pph=getattr(self, 'pph', None), q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.t_lift_ok is None or self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy()))
        super()._advance_inner(T_tcp, close_done, q)


def run():
    scene = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']

    def scan_fn(t, q):
        r = NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z)
        oc, _ = DC.state(t, CA)
        return np.minimum(r, DC.ray_cyl(q[:2], NC.scan_angles() + float(q[2]), oc, CA['radius'], NC.RANGE_MAX))
    orig, orig_ml = T.W1.W1PlaceController, SC.make_lidar
    T.W1.W1PlaceController = Ctl
    SC.make_lidar = lambda **kw: orig_ml(scan_fn=scan_fn, **kw)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, SC.make_lidar = orig, orig_ml
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None)
    step0, meas0 = c.step, f.meas

    def meas(**ov):
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy(), bool(f.grasped and not getattr(f, 'released', False))))
        return m
    f.meas = meas

    def step(t, q, **kw):
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=80.0)
    return c, f, r


def main():
    c, f, r = run()
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    ev = r['events']
    # 1 實際軌跡
    H = f.hist
    tt = np.array([h['t'] for h in H])
    Q = np.array([h['q'] for h in H])
    ob = np.array([DC.state(x, CA)[0] for x in tt])
    db = np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - CA['radius'] - R_BASE
    tb_t = np.array([x[0] for x in TBLOG])
    best_l, best_b = (9.9, None), (9.9, None)
    for k in range(0, len(H), 5):
        if db[k] > 1.5:
            continue
        W = G.link_world(Q[k], 0.0038 if H[k]['phase'] in ('LIFT', 'CARRY') else G.Q_OPEN)
        d_l = obs_d(np.vstack(list(W.values())), tt[k])
        if d_l < best_l[0]:
            best_l = (d_l, round(float(tt[k]), 2))
        j = int(np.argmin(np.abs(tb_t - tt[k])))
        if TBLOG[j][2]:
            Tb = TBLOG[j][1]
            d_b = obs_d((Tb[:3, :3] @ BPT.T).T + Tb[:3, 3], tt[k])
            if d_b < best_b[0]:
                best_b = (d_b, round(float(tt[k]), 2))
    k10 = np.flatnonzero(np.abs(tt * 10 - np.round(tt * 10)) < 1e-6)
    sp = np.hypot(np.diff(Q[k10, 0]), np.diff(Q[k10, 1])) / 0.1
    acc = np.abs(np.diff(sp)) / 0.1
    near_k = db[k10][1:-1] < 1.5
    enc = dict(t_first_near=round(float(tt[np.argmax(db < 1.5)]), 2) if (db < 1.5).any() else None, t_last_near=round(float(tt[len(db) - 1 - np.argmax(db[::-1] < 1.5)]), 2) if (db < 1.5).any() else None,
               v_min_near=round(float(sp[1:][near_k].min()), 4) if near_k.any() else None, dv_max_near=round(float(acc[near_k].max()), 3) if near_k.any() else None,
               phases_near=sorted({H[k]['phase'] for k in np.flatnonzero(db < 1.5)}))
    actual_min = dict(base_disc_surface=(round(float(db.min()), 4), round(float(tt[int(np.argmin(db))]), 2)), links=best_l, held_bottle=best_b)
    # 2 停止路徑（會遇窗逐控制拍）
    rt = np.array([x['t'] for x in REC])
    dnear = np.array([float(np.hypot(x['q'][0] - DC.state(x['t'], CA)[0][0], x['q'][1] - DC.state(x['t'], CA)[0][1]) - CA['radius'] - R_BASE) for x in REC])
    sel = set()
    for i in np.flatnonzero(dnear < 1.5):
        sel.update(range(max(i - 5, 0), min(i + 6, len(REC))))
    dyn_worst, stat_worst, where, nonfin, unfin = {}, {}, {}, 0, 0
    dt_s = C21.SAMPLE_EVERY * 0.01
    for i in sorted(sel):
        x = REC[i]
        for e in (0, 1):
            Qs = C21.stop_traj(x['q'], x['s'], x['pipe'], x['u_last'], c.cfg, e)
            if len(Qs) >= 2000 // C21.SAMPLE_EVERY + 1:
                unfin += 1
            if not np.isfinite(Qs).all():
                nonfin += 1
                continue
            for j, qq in enumerate(Qs):
                W, P_l, P_b = pts(qq, x['f_cmd'], x['T_rel'])
                tj = x['t'] + j * dt_s
                for key, v in (('links_obs', obs_d(P_l, tj)), ('held_obs', obs_d(P_b, tj) if P_b is not None else 9.9),
                               ('base_disc_obs', float(np.hypot(qq[0] - DC.state(tj, CA)[0][0], qq[1] - DC.state(tj, CA)[0][1]) - CA['radius'] - R_BASE))):
                    kk = f'e{e}:{key}'
                    if v < dyn_worst.get(kk, 9.9):
                        dyn_worst[kk], where[kk] = v, (round(x['t'], 2), x['pph'] or x['ph'], round(j * dt_s, 2))
                if j % 2 == 0:
                    placing = (x['pph'] or '') in PLACING
                    for kk, v in chk_static(qq, x['T_rel'], x['f_cmd'], W, P_b if not placing else None).items():
                        key = f'e{e}:{kk}'
                        if v < stat_worst.get(key, 9.9):
                            stat_worst[key], where[key] = v, (round(x['t'], 2), x['pph'] or x['ph'], round(j * dt_s, 2))
    dyn_fails = sorted(k for k, v in dyn_worst.items() if v <= 0.0 and 'base_disc' not in k)
    stat_fails = sorted(k for k, v in stat_worst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
    nd = [len([o for o in (cc.get('obs') or []) if not o.get('static')]) for cc in c.compact] if getattr(c, 'compact', None) else []
    pb = [e for e in ev if 'blocked' in str(e.get('why', '')) or e['what'] == 'path_blocked']
    ok = (done or why is not None) and not dyn_fails and not stat_fails and T.finite_cmds(f)
    sw = (r.get('slip_warn') or {})
    out = dict(case=CASE, dyn=DYN, z1=Z1, bottle=[DXB, 0.42 + DYB], **{'pass': bool(ok)}, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               adopt=next(({'t': e['t'], 'dx': e['dx']} for e in ev if e['what'] == 'relwin_shift'), None), slip_max=r['slip_max_mm_deg'], slip_warn=sw.get('warn'),
               actual_min=actual_min, encounter=enc, stop_dyn={k: round(v, 4) for k, v in dyn_worst.items()}, stop_dyn_where=where, stop_dyn_fails=dyn_fails,
               stop_static={k: round(v, 4) for k, v in stat_worst.items()}, stop_static_fails=stat_fails, n_stop_ticks=len(sel), n_dyn_cycles=int(sum(1 for x in nd if x > 0)), path_blocked=pb,
               note='假世界無 Shield；障礙照封存任務時刻排程；停止路徑判準：連桿／持瓶到障礙表面 > 0（底盤圓盤值另報不判）；取樣核對，非全流程保證')
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
