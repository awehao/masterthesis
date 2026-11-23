#!/usr/bin/env python3
"""兩瓶連續任務 F1 前置第 1 項（Codex 20261102 F1 裁定；假受控對象、假候選，非物理、不用 Isaac）：每輪全程（含鬆爪、退離、保持）對另一瓶的掃掠距離。
兩輪都由原起點 (−3.6, 0) 接近（路徑＝bottle_online5_u3_1 實跑規劃）；第二輪重啟點＝原起點，不越過展臂觸發點（約 −2.65）。控制器＝bottle_w1_carrygov_ctrl（基線）；觀看點 x 取該輪區域中心。
  c1 第一輪：抓 (−0.15, 0.42)，放遠端 (2.55, −0.58)；另一瓶＝第一桌 (+0.15, 0.42)（生成位置，偏差 0）。
  c2 第二輪：抓 (+0.15, 0.42)，放近端 (2.25, −0.58)；另一瓶＝已放第一瓶 (2.55, −0.58)，**含允許落點偏差 PLACE_DEV**（半徑加大）。
  r1／r2 對調抓取順序（python3 evaluation/test_bottle_f1_geom.py rev）：先抓 (+0.15, 0.42)，另一瓶＝(−0.15, 0.42)，放遠端；再抓 (−0.15, 0.42)，放近端，另一瓶＝已放第一瓶。
持瓶段：F1_PLAN 未設 ⇒ 直線假路徑（第一版；只取起訖點給 bottle_f1_offline_plan）；F1_PLAN＝結果檔 ⇒ 用該檔的真 Smac 路徑（首點換成當趟實際起點，末點換成當趟目標，差須 ≤ 5 cm，否則拒跑）。
距離＝全臂（非手指）、手指、持瓶到另一瓶圓柱表面；實際軌跡逐 5 物理步；停止軌跡＝compactcarry21.stop_traj 現在停／延後一拍停，只在實際距離 < GATE 的控制拍（前後各 5 拍）。
判準（事前）：全部 ≥ LIM_OTHER 0.05 m（同接近段對瓶淨空判準）；另一瓶不進 QP：這是指定靜態配置的核對，不是在線避障。同時照舊核對連桿／手指／持瓶對兩桌與自碰（判準同 test_bottle_w1_rb1_dyn）。
    python3 evaluation/test_bottle_f1_geom.py all
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLACE_DEV = 0.030                                       # 第一瓶允許落點偏差＝既有 place_tol_report
CASES = dict(c1=dict(b=(-0.15, 0.42), target=(2.55, -0.58), other=(0.15, 0.42), z_other=0.60, dev=0.0, x_look=-0.15),
             c2=dict(b=(0.15, 0.42), target=(2.25, -0.58), other=(2.55, -0.58), z_other=0.60, dev=PLACE_DEV, x_look=0.15),
             r1=dict(b=(0.15, 0.42), target=(2.55, -0.58), other=(-0.15, 0.42), z_other=0.60, dev=0.0, x_look=0.15),
             r2=dict(b=(-0.15, 0.42), target=(2.25, -0.58), other=(2.55, -0.58), z_other=0.60, dev=PLACE_DEV, x_look=-0.15))
PLAN_FILE = os.environ.get('F1_PLAN')
BODY = os.environ.get('F1_BODY') == '1'                 # 另一瓶以既有三維障礙列（bottle_s1_ctrl._arm_extra，d_body 0.05、S3 同設定）進 QP；另列，不覆蓋未進 QP 的結果
REV = len(sys.argv) > 1 and sys.argv[1] == 'rev'
OUT = os.path.join(HERE, 'results', 'bottle', ('f1_geom_rev_plan.json' if PLAN_FILE else 'f1_geom_rev.json') if REV else (('f1_geom_plan_body.json' if BODY else 'f1_geom_plan.json') if PLAN_FILE else 'f1_geom.json'))

if REV or (len(sys.argv) > 1 and sys.argv[1] == 'all'):
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:]))
    with ThreadPoolExecutor(2) as ex:
        res = dict(ex.map(one, ('r1', 'r2') if REV else ('c1', 'c2')))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k, r in res.items():
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 't_done', 'actual_other', 'stop_other', 'fails', 'stop_static_fails', 'place_err', 'accept', 'error')})
    sys.exit(0 if all(r.get('pass') for r in res.values()) else 1)

CASE = sys.argv[1]
CF = CASES[CASE]
D_U, GATE, LIM_OTHER = 3.6, 0.60, 0.05
sys.argv = [sys.argv[0], '0.600', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = CF['b']
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_carrygov_ctrl as CG                    # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
TP.DX, TP.DY = 2.4, -1.0
PC.PLACE['target'] = CF['target']
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', 'bottle_online5_u3_1', 's1_report.json')))['plan']['joined'], float)
RTF_CAP = 0.6
LOGP = os.path.join(HERE, 'results', 'bottle', f'f1_geom_{CASE}_worker.log')
T = TV.T
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
TB = [(c_, s_) for _, c_, s_ in TBN]
TB_NO_T2TOP = [(c_, s_) for n, c_, s_ in TBN if n != 't2:table_top']
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
REC, TBLOG = [], []
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
OX, OY, OZ0, OZ1, OR = CF['other'][0], CF['other'][1], CF['z_other'], CF['z_other'] + G.BOTTLE['h'], G.BOTTLE['r'] + CF['dev']


def d_other(P):
    return float(G.cyl_dist(P, OX, OY, OZ0, OZ1, OR).min())


def split(W):
    return np.vstack([v for k, v in W.items() if k not in G.C.FINGERS]), np.vstack([v for k, v in W.items() if k in G.C.FINGERS])


def chk_static(q9, T_rel, f_cmd, W, P_b, placing=False):
    P_nf, P_f = split(W)
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for c_, s_ in TB), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for c_, s_ in TB),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if P_b is not None:
        out['held_bottle_table'] = min(float(G.box_dist(P_b, c_, s_).min()) for c_, s_ in (TB_NO_T2TOP if placing else TB))
    return out


class Ctl(CG.W1PlaceController):
    look = np.array([CF['x_look'], 0.4275, 0.70])

    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _place_arm_extra(self):
        return self._arm_extra() if BODY else super()._place_arm_extra()   # 持瓶導航段也帶三維障礙列（同 bottle_w1_shrow3d_ctrl 的掛鉤）

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all():
            REC.append(dict(t=float(self.t), ph=self.st.phase, pph=getattr(self, 'pph', None), q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.t_lift_ok is None or self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy()))
        super()._advance_inner(T_tcp, close_done, q)


class PlanFile(TP.FakePlanner):
    """離線 Smac 路徑（同 bottle_f1_offline_plan 的結果）；只換首末點對齊當趟實際起點與目標。"""

    def __init__(self, P):
        super().__init__()
        self.P0 = np.array(P, float)
        self.info_ = None

    def poll(self):
        res = super().poll()
        if res is None or not res.get('ok'):
            return res
        a, b, _ = self.req
        d0, d1 = float(np.linalg.norm(self.P0[0] - a)), float(np.linalg.norm(self.P0[-1] - b))
        if max(d0, d1) > 0.05:
            raise SystemExit(f'離線路徑首末點差太大 {d0:.3f}／{d1:.3f} m ⇒ 拒跑（重產 bottle_f1_offline_plan）')
        P = self.P0.copy()
        P[0], P[-1] = a, b
        self.info_ = dict(d_start=round(d0, 4), d_goal=round(d1, 4), n=len(P))
        res.update(P=P, info=dict(planned=True, **self.info_))
        return res


def run():
    scene = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']

    def scan_fn(t, q):
        return NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z)
    orig, orig_ml = T.W1.W1PlaceController, SC.make_lidar
    T.W1.W1PlaceController = Ctl
    SC.make_lidar = lambda **kw: orig_ml(scan_fn=scan_fn, **kw)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, SC.make_lidar = orig, orig_ml
    if BODY:
        c.body_boxes = [(np.array([OX, OY, OZ0 + G.BOTTLE['h'] / 2.0]), np.array([2.0 * OR, 2.0 * OR, G.BOTTLE['h']]))]
    if PLAN_FILE:
        pj = json.load(open(PLAN_FILE))
        c.carry_planner = PlanFile(pj[{'c1': 'carry_far', 'c2': 'carry_near', 'r1': 'carry_far', 'r2': 'carry_near'}[CASE]]['P_region'])
    c.path = NP.PlannedPath(J_RUN, -0.9)
    c.s_ref = c.sig_robot = c.path.sig0
    f.q[0], f.q[1] = J_RUN[0][0], J_RUN[0][1]
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, wall0=None)
    step0, meas0 = c.step, f.meas
    import time as _time

    def meas(**ov):
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy(), bool(f.grasped and not getattr(f, 'released', False))))
        return m
    f.meas = meas

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        else:
            st['wall0'] = _time.time() - t / RTF_CAP
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=0.60, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=90.0)
    return c, f, r


def main():
    c, f, r = run()
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    ev = r['events']
    # 1 實際軌跡
    H = f.hist
    tt, Q = np.array([h['t'] for h in H]), np.array([h['q'] for h in H])
    tb_t = np.array([x[0] for x in TBLOG])
    act = dict(nonfinger=(9.9, None, None), finger=(9.9, None, None), held=(9.9, None, None))
    rt = np.array([x['t'] for x in REC])
    dmin_t = {}
    for k in range(0, len(H), 5):
        ph = H[k]['phase']
        W = G.link_world(Q[k], 0.0038 if ph in ('LIFT', 'CARRY') else G.Q_OPEN)
        P_nf, P_f = split(W)
        dn, df = d_other(P_nf), d_other(P_f)
        j = int(np.argmin(np.abs(tb_t - tt[k])))
        Tb = TBLOG[j][1]
        db = d_other((Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]) if TBLOG[j][2] else 9.9
        for nm, v in (('nonfinger', dn), ('finger', df), ('held', db)):
            if v < act[nm][0]:
                act[nm] = (round(v, 4), round(float(tt[k]), 2), ph)
        dmin_t[round(float(tt[k]), 2)] = min(dn, df, db)
    # 2 停止軌跡（靠近另一瓶的控制拍，前後各 5 拍）
    near = np.zeros(len(REC), bool)
    for i, x in enumerate(REC):
        near[i] = min(dmin_t.get(round(x['t'], 2), 9.9), dmin_t.get(round(x['t'] + 0.01, 2), 9.9), dmin_t.get(round(x['t'] + 0.02, 2), 9.9)) < GATE or (x['pph'] in PLACING)
    sel = set()
    for i in np.flatnonzero(near):
        sel.update(range(max(i - 5, 0), min(i + 6, len(REC))))
    worst, stat_worst, where, nonfin, unfin = {}, {}, {}, 0, 0
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
                if j % 2:
                    continue
                W = G.link_world(qq, x['f_cmd'])
                P_nf, P_f = split(W)
                P_b = None
                if x['T_rel'] is not None:
                    Tb = G.C.K.fk(qq, 'link_tcp') @ x['T_rel']
                    P_b = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
                for nm, v in (('nonfinger', d_other(P_nf)), ('finger', d_other(P_f)), ('held', d_other(P_b) if P_b is not None else 9.9)):
                    kk = f'e{e}:{nm}'
                    if v < worst.get(kk, 9.9):
                        worst[kk], where[kk] = v, (round(x['t'], 2), x['pph'] or x['ph'], round(j * dt_s, 2))
                for kk, v in chk_static(qq, x['T_rel'], x['f_cmd'], W, P_b, (x['pph'] or '') in PLACING).items():
                    key = f'e{e}:{kk}'
                    if v < stat_worst.get(key, 9.9):
                        stat_worst[key], where[key] = v, (round(x['t'], 2), x['pph'] or x['ph'], round(j * dt_s, 2))
    stat_fails = sorted(k for k, v in stat_worst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
    fails = sorted([k for k, v in act.items() if v[0] < LIM_OTHER]) + sorted(k for k, v in worst.items() if v < LIM_OTHER)
    ok = done and not fails and not stat_fails and T.finite_cmds(f)
    tb_end = TBLOG[-1][1]
    pl = dict(target=CF['target'], final=[round(float(tb_end[0, 3]), 4), round(float(tb_end[1, 3]), 4)])
    pl_err = round(float(np.hypot(tb_end[0, 3] - CF['target'][0], tb_end[1, 3] - CF['target'][1])), 4)
    out = dict(case=CASE, bottle=CF['b'], target=CF['target'], other=CF['other'], dev=CF['dev'], **{'pass': bool(ok)}, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               actual_other=act, stop_other={k: round(v, 4) for k, v in worst.items()}, stop_other_where={k: where[k] for k in worst}, fails=fails, stop_static={k: round(v, 4) for k, v in stat_worst.items()}, stop_static_fails=stat_fails,
               stop_static_where={k: where[k] for k in stat_worst}, n_stop_ticks=len(sel), place=pl, place_err=pl_err, park=np.asarray(c.park).tolist() if getattr(c, 'park', None) is not None else None, q_end=np.asarray(Q[-1]).tolist(),
               body=dict(on=BODY, bind=r.get('body_bind'), rows_max=r.get('body_rows_max'), planes=r.get('body_planes')), solve_ms=r.get('solve_ms'), n_over_50ms=r.get('n_over_50ms'), n_fallback=r.get('n_fallback'),
               carry_req=(r['place'] or {}).get('carry_req'), carry_plan=getattr(c.carry_planner, 'info_', None),
               accept=next(({'t': e['t'], 'x': e.get('x')} for e in ev if e['what'] in ('online_plan_accept', 'grasp_plan_accept', 'relwin_shift')), None), adopt=[{k: e.get(k) for k in ('what', 't', 'dx', 'x')} for e in ev if e['what'] in ('relwin_shift', 'online_gp_accept', 'online_obs_first')],
               gaze=(r.get('gaze') or {}).get('look'), phases={e['to']: e['t'] for e in ev if e['what'] in ('phase', 'place_phase')}, slip_max=r['slip_max_mm_deg'],
               carry_gov=dict(hold_s=(r.get('carry_gov') or {}).get('hold_s'), min_stop_margin=(r.get('carry_gov') or {}).get('min_stop_margin')),
               note='假受控對象、假候選（真值＋雜訊）、持瓶段直線假路徑；取樣核對，非全流程保證；另一瓶不進 QP')
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
