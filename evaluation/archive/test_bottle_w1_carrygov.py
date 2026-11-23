#!/usr/bin/env python3
"""運送進度協調（bottle_w1_carrygov_ctrl）的最小核對（Codex 20261102_175049；假世界、假候選，非物理；不做速度或位置掃描）。
案例：
  rb2    R-B2 短路徑（接近＝rb2_offline_plan 離線規劃、O＝(0.8, −0.8)、第一桌 0.645、瓶 區域 (0.0607, 0.4048)）：期望完整抓放；記持守位置、減速過程。
  rb1    R-B1 原路徑（接近＝bottle_rb1_1 實跑規劃、第一桌 0.58）：期望持守點從不限速（無 bind 事件）、完成時刻同視向版（48.4 s）。
  delay  R-B2＋轉回計畫結果晚 3.0 s 交付（模擬時間）：期望底盤在直線段入口前有界減速、停在持守點等待，收到後照常完成。
  fail   R-B2＋轉回計畫結果永不交付（工作者仍活）：期望持守點等待到 HOLD_MAX 後 carry_hold_timeout 受控停止，持瓶、無接觸、底盤停在持守點前。
  recede 延遲案＋收近開始後 0.6 s 把控制器的第二桌先驗（place['c']）沿 −x 移動，使幾何危險邊界落在底盤目前進度前 0.04 m（持守點突然後退，來不及停）：
         期望停止模型監看觸發 carry_hold_overrun 受控停止；持瓶段停止軌跡對實際兩桌照核。
  recede_mild 同上但邊界在前 0.12 m：期望停止模型監看觸發，或來得及停住且停止模型餘裕始終 ≥ 0。
  nonfinite 延遲案＋收近開始後 0.6 s 起持守包絡回傳非有限：期望 carry_gov_invalid 受控停止、不放行到停車位。
每案：持瓶段（離桌確立 t_lift_ok＋0.2 s 起，同實際瓶對桌；到鬆爪或停止）每 5 控制拍＋持守 bind／release、收近、外伸等事件 ±5 拍，做現在停（e0）與延後一拍停（e1）的完整取樣停止軌跡（compactcarry21.stop_traj），
  核非手指／手指對兩桌、持瓶對兩桌（放置相位排除第二桌桌面）、自碰、關節餘裕；另記實際瓶對兩桌最小距離、持守紀錄、參考車速最小值與減速度。
假世界節流 RTF 0.6、落後重新錨定（同 test_bottle_w1_gaze）。
    python3 evaluation/test_bottle_w1_carrygov.py all       （依序，寫 results/bottle/carrygov_check.json）
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'carrygov_check.json')
ORDER = ('rb2', 'rb1', 'delay', 'fail', 'recede', 'recede_mild', 'nonfinite')
GEO = dict(rb2=dict(run=None, plan='rb2_offline_plan.json', z1=0.645, b=(0.0607, 0.4048), O=(0.80, -0.80)),
           rb1=dict(run='bottle_rb1_1', z1=0.58, b=(-0.03, 0.44), O=(0.60, 0.80)))
DELAY_S = 3.0

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    res = {}
    for k in ORDER:
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        res[k] = json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:])
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('pass', 'carry_ok', 'outcome', 't_done', 'gov_events', 'hold_s', 'hold_pos', 'min_hold_minus_brake', 'min_stop_margin', 'inject', 'park_tol_log', 'stop_fails', 'bottle_table', 'error')}, flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    sys.exit(0 if all(r.get('pass') for r in res.values()) else 1)

CASE = sys.argv[1]
CF = GEO['rb1' if CASE == 'rb1' else 'rb2']
Z1, D_U = CF['z1'], 3.6
sys.argv = [sys.argv[0], f'{Z1:.3f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = CF['b']
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_carrygov_ctrl as CG                    # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import test_bottle_place as TP                          # noqa: E402

T = TV.T
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_region as BRG                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
REG = BRG.RegionFrame(*CF['O'])
PLACE_R = REG.place_to_region(PC.PLACE)
PC.PLACE.clear()
PC.PLACE.update(PLACE_R)
TP.DX, TP.DY = 2.4 - CF['O'][0], -1.0 - CF['O'][1]
SC.X0 = -3.6 - CF['O'][0]
SC.OBST['c'] = np.array([-2.0 - CF['O'][0], 0.0 - CF['O'][1], 0.25])
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', CF['run'], 's1_report.json')))['plan']['joined'] if CF.get('run') else
                 json.load(open(os.path.join(HERE, 'results', 'bottle', CF['plan'])))['approach']['joined_region'], float)
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
EV_HOT = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned', 'xh_back', 'xh_tcp_converged', 'place_phase',
          'return_plan_accept', 'carry_gov_bind', 'carry_gov_release')
REC, TBLOG = [], []
RTF_CAP = float(os.environ.get('CG_RTF', '0.6'))
DBG = os.environ.get('CG_DEBUG')
LOGP = os.path.join(HERE, 'results', 'bottle', f'carrygov_{CASE}_worker.log')
N_MAX = 2000 // C21.SAMPLE_EVERY + 1


def chk(q9, T_rel, f_cmd, placing):
    W = G.link_world(q9, f_cmd)
    P_nf = np.vstack([v for k, v in W.items() if k not in G.C.FINGERS])
    P_f = np.vstack([v for k, v in W.items() if k in G.C.FINGERS])
    out = dict(nonfinger_table=min(float(G.box_dist(P_nf, c_, s_).min()) for _, c_, s_ in TBN), finger_table=min(float(G.box_dist(P_f, c_, s_).min()) for _, c_, s_ in TBN),
               self=float(G.C.self_clear(q9, f_cmd)[0]), margin=BB.margin(q9[3:]))
    if T_rel is not None:
        Tb = G.C.K.fk(q9, 'link_tcp') @ T_rel
        Pb = (Tb[:3, :3] @ BPT.T).T + Tb[:3, 3]
        out['held_bottle_table'] = min(float(G.box_dist(Pb, c_, s_).min()) for n, c_, s_ in TBN if not (placing and n == 't2:table_top'))
    return out


class Ctl(CG.W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all() and self.judge.t_lift_ok is not None and self.t >= self.judge.t_lift_ok + 0.2:   # 同實際瓶對桌：離桌確立後 0.2 s 起
            REC.append(dict(t=float(self.t), pph=self.pph, stage=None if self.cc is None else self.cc['stage'], q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy(), v_b=float(self.v_b), hold=self.gov_hold))
        super()._advance_inner(T_tcp, close_done, q)
        if DBG and self.pph in ('CARRY_NAV', 'PLACE_APPROACH') and self.n_step % 20 == 0:
            print('DBG', round(float(self.t), 2), self.pph, None if self.cc is None else self.cc['stage'], self.gov_hold, self.gov_info, self.csig, self.s_ref, self.v_b,
                  None if self.cpath is None else self.cpath.x_lane, None if self.park is None else self.park[0], self.judge.T_rel is None, flush=True)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    c.path = NP.PlannedPath(J_RUN, -0.9)
    c.s_ref = c.sig_robot = c.path.sig0
    f.q[0], f.q[1] = J_RUN[0][0], J_RUN[0][1]
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, held={}, wall0=None)
    if CASE in ('delay', 'fail', 'recede', 'recede_mild', 'nonfinite'):
        _poll = c.pc.poll

        def poll(rid):
            ret = c.rp is not None and rid == c.rp.get('id')
            if not ret:
                return _poll(rid)
            if rid not in st['held']:
                m = _poll(rid)
                if m is None:
                    return None
                st['held'][rid] = (float(c.t), m)
            t0_, m = st['held'][rid]
            if CASE == 'fail' or c.t < t0_ + DELAY_S - 1e-9:   # recede／nonfinite 沿用延遲案
                return None
            return m
        c.pc.poll = poll
    step0, meas0 = c.step, f.meas
    if os.environ.get('CG_TIME'):                                    # 計時：持守每拍計算（_gov_compute＋_gov_stop_check），測試端包裝，不改受測控制器
        import time as _tm
        st['gov_ms'] = []
        _gc, _gs = c._gov_compute, c._gov_stop_check

        def gc_(q, _f=_gc):
            t0_ = _tm.perf_counter()
            ok_ = _f(q)
            st['_t0'] = t0_
            return ok_

        def gs_(q, _f=_gs):
            r_ = _f(q)
            st['gov_ms'].append((float(c.t), (_tm.perf_counter() - st.pop('_t0', _tm.perf_counter())) * 1e3))
            return r_
        c._gov_compute, c._gov_stop_check = gc_, gs_
    if DBG:                                                         # 除錯：記每次 GMPC 外層結果（後處理動作、輪級、CBF）
        for nm in ('mpc_park', 'mpc_nav'):
            m_ = getattr(c, nm, None)
            if m_ is None:
                continue
            _s = m_.solve

            def sv(*a_, _s=_s, _nm=nm, **k_):
                r_ = _s(*a_, **k_)
                if k_.get('arm', {}).get('mode') == 'active' and not k_.get('arm', {}).get('_inner'):
                    st.setdefault('mpc_last', []).append(dict(t=round(float(c.t), 3), nm=_nm, acc=getattr(r_, 'accept_action', None), fb=(r_.arm_check or {}).get('fallback'),
                                                             u=np.round(np.asarray(r_.u_opt, float), 6).tolist(), cbf=int(getattr(r_, 'cbf_active', -1)), n_obs=len(k_.get('obstacles') or []),
                                                             status=getattr(r_, 'status', None)))
                    st['mpc_last'] = st['mpc_last'][-6:]
                return r_
            m_.solve = sv

    def meas(**ov):
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy()))
        return m
    f.meas = meas
    import time as _time

    def inject():
        cs = next((e['t'] for e in c.events if e['what'] == 'compact_start'), None)
        if cs is None or st.get('injected') or c.t < cs + 0.6 - 1e-9 or c.judge.stop is not None:
            return
        st['injected'] = float(c.t)
        if CASE in ('recede', 'recede_mild'):
            gi = c.gov_info or {}
            geom = gi.get('geom', float('inf'))
            ahead = 0.04 if CASE == 'recede' else 0.12
            dx = (geom + CG.GOV_BACK) - (float(c.csig) + ahead) if np.isfinite(geom) else 0.0
            c.place['c'] = (float(c.place['c'][0]) - dx, float(c.place['c'][1]))
            st['inject_info'] = dict(t=float(c.t), csig=float(c.csig), v_b=float(c.v_b), geom_before=geom, shift_x=-dx)
        elif CASE == 'nonfinite':
            env0 = c._gov_env
            c._gov_env = lambda q_arms, _e=env0: _e(q_arms) * np.nan
            st['inject_info'] = dict(t=float(c.t), csig=float(c.csig), v_b=float(c.v_b))

    def step(t, q, **kw):
        if CASE in ('recede', 'recede_mild', 'nonfinite'):
            inject()
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
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=85.0)
    return c, f, r, st


def main():
    c, f, r, st = run()
    ev = r['events']
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    on = r.get('online') or {}
    ph = {}
    for e in ev:
        if e['what'] in ('phase', 'place_phase'):
            ph.setdefault(e['to'], e['t'])
    pm = (r.get('place') or {}).get('metrics') or {}
    cg = r.get('carry_gov') or {}
    out = dict(case=CASE, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None, slip_max=r['slip_max_mm_deg'],
               t_grasp=r.get('t_grasp'), t_lift_ok=r.get('t_lift_ok'), phases=ph,
               rp=[{k: x.get(k) for k in ('id', 'state', 't_req', 't_res', 't_acc', 'compute_wall')} for x in on.get('return_plans') or []], held_return=list(st['held'].keys()),
               compact=[(e['t'], e['what']) for e in ev if e['what'].startswith('compact_')], gov_events=cg.get('events'), hold_s=cg.get('hold_s'),
               min_hold_minus_brake=cg.get('min_hold_minus_brake'), min_stop_margin=cg.get('min_stop_margin'), park_tol_log=cg.get('park_tol_log'), inject=st.get('inject_info'))
    # 持守位置與減速
    lg = cg.get('log') or []
    held = [x for x in lg if x[5] == 0.0 and x[6] is not None and x[6] - CG.GOV_BUF - x[4] <= 0.03 and x[7] != 'park']
    out['hold_pos'] = None if not held else dict(t_first=held[0][0], t_last=held[-1][0], csig=held[0][3], hold=held[0][6], why=sorted({x[7] for x in held}), stage_first=held[0][2], stage_last=held[-1][2],
                                                 x_lane=None if c.cpath is None else round(float(c.cpath.x_lane), 4), park=None if c.park is None else round(float(c.park[0]), 4))
    vb = [(x[0], x[5]) for x in lg]
    if len(vb) > 2:
        tt, vv = np.array(vb).T
        dv = np.diff(vv) / np.maximum(np.diff(tt), 1e-9)
        out['ref_decel_max'] = round(float(-dv.min()), 4)
        out['ref_acc_max'] = round(float(dv.max()), 4)
    # 持瓶段停止軌跡
    rt = np.array([x['t'] for x in REC])
    t_end = ph.get('RELEASE', (r['stop'] or {}).get('t', 1e9))
    span = [i for i, x in enumerate(REC) if x['t'] <= t_end]
    sel = set(span[::5])
    for te in sorted({float(e['t']) for e in ev if e['what'] in EV_HOT}):
        k = int(np.argmin(np.abs(rt - te))) if len(rt) else 0
        sel.update(i for i in range(k - 5, k + 6) if i in set(span))
    worst, where, nonfin, unfin = {}, {}, 0, 0
    for i in sorted(sel):
        x = REC[i]
        placing = (x['pph'] or '') in PLACING
        for e in (0, 1):
            Qs = C21.stop_traj(x['q'], x['s'], x['pipe'], x['u_last'], c.cfg, e)
            if len(Qs) >= N_MAX:
                unfin += 1
            if not np.isfinite(Qs).all():
                nonfin += 1
                continue
            for qq in Qs[::2]:
                for kk, v in chk(qq, x['T_rel'], x['f_cmd'], placing).items():
                    key = f'e{e}:{kk}'
                    if v < worst.get(key, 9.9):
                        worst[key], where[key] = v, (round(x['t'], 3), x['pph'], x['stage'])
    out['stop_n'] = len(sel)
    out['stop_worst'] = {k: round(v, 5) for k, v in worst.items()}
    out['stop_where'] = where
    out['stop_fails'] = sorted(k for k, v in worst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
    t_lok, t_park = r.get('t_lift_ok'), ph.get('PARKED', 1e9)
    bt = dict(v=9.9, t=None, box=None)
    for tt_, Tb_ in TBLOG[::5]:
        if t_lok is None or tt_ < t_lok + 0.2:
            continue
        P_ = (Tb_[:3, :3] @ BPT.T).T + Tb_[:3, 3]
        for n_, c_, s_ in TBN:
            if tt_ >= t_park and n_ == 't2:table_top':
                continue
            d_ = float(G.box_dist(P_, c_, s_).min())
            if d_ < bt['v']:
                bt.update(v=round(d_, 4), t=round(tt_, 2), box=n_)
    out['bottle_table'] = bt
    th = np.array([h['t'] for h in f.hist])
    Qh = np.array([h['q'][:3] for h in f.hist])
    out['base_final'] = Qh[-1].round(4).tolist()
    base_ok = bt['v'] >= LIM['held_bottle_table'] and not out['stop_fails'] and T.finite_cmds(f)
    ok = {'rb2': done,
          'rb1': done and not cg.get('events') and out['t_done'] is not None and abs(out['t_done'] - 48.4) < 0.051,
          'delay': done and bool(cg.get('events')) and (cg.get('hold_s') or 0) > 0.0,
          'fail': why == 'carry_hold_timeout' and r.get('t_grasp') is not None and not ph.get('PARKED'),
          'recede': st.get('inject_info') is not None and why == 'carry_hold_overrun',
          'recede_mild': st.get('inject_info') is not None and (why == 'carry_hold_overrun' or (((cg.get('min_stop_margin') or {}).get('v', -1) >= 0.0) and 'RELEASE' in ph)),
          'nonfinite': st.get('inject_info') is not None and why == 'carry_gov_invalid' and not ph.get('PARKED')}[CASE]
    reached_release = 'RELEASE' in ph
    out['carry_ok'] = bool(base_ok and (reached_release if CASE != 'fail' else ok))   # 持瓶段（持守、減速、停止軌跡、瓶對桌）是否成立，與整趟結果分列
    out['gov_log_cols'] = cg.get('log_cols')
    out['gov_log'] = cg.get('log')
    if st.get('gov_ms'):
        g_ = np.array([x[1] for x in st['gov_ms']])
        out['gov_ms'] = dict(n=len(g_), p50=round(float(np.percentile(g_, 50)), 3), p95=round(float(np.percentile(g_, 95)), 3), max=round(float(g_.max()), 3))
        out['timing'] = r.get('timing')
    out['pass'] = bool(ok and base_ok)
    out['expect'] = dict(rb2='完整抓放', rb1='持守從不限速、完成 48.4 s（同視向版）', delay='入口前減速並持守等待、收到後完成', fail='持守等待到上限後 carry_hold_timeout、持瓶無接觸',
                         recede='停止模型監看 carry_hold_overrun 受控停止', recede_mild='停止模型監看觸發，或停住且停止模型餘裕 ≥ 0', nonfinite='carry_gov_invalid 受控停止、不放行')[CASE]
    _ = th
    if DBG:
        print('EV', [e for e in ev if e['t'] >= (r['stop'] or {}).get('t', 1e9) - 1.0], flush=True)
        print('QP', r.get('qp_log', [])[-4:], flush=True)
        for x_ in st.get('mpc_last', []):
            print('MPC', x_, flush=True)
        H = [h for h in f.hist if h['t'] >= (r['stop'] or {}).get('t', 1e9) - 0.6]
        for h in H[::10]:
            print('H', round(h['t'], 2), h['phase'], np.round(h['q'][:3], 4).tolist(), np.round(h['u'], 4).tolist(), flush=True)
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
