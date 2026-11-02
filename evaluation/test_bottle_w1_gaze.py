#!/usr/bin/env python3
"""在線視向候選（bottle_w1_gaze_ctrl）的假世界閉迴路核對（Codex 20261102_092601；假世界、假候選，非物理）。
兩條既有接近軌跡各跑「視向開」與「視向關（＝online5 基線、對照與退化）」：
  u3   原區域 U3：接近路徑＝bottle_online5_u3_1 實跑規劃（joined）、第一桌 0.598、瓶 (0.0042, 0.4564)、第二桌 (2.4, −1.0)、原障礙 (−2.0, 0)；
  rb1  R-B1：接近路徑＝bottle_rb1_1 實跑規劃（區域 joined）、第一桌 0.58、瓶 區域 (−0.03, 0.44)、第二桌區域 (1.8, −1.8)、原障礙區域 (−2.6, −0.8)、放置先驗換區域。
假候選同 test_bottle_w1_rb1（U 段開始後 3.6 s 起送；本檔不估計取得時刻，取得時刻要看實跑；這裡核動態）。規劃工作者是真子程序，假世界節流 RTF 0.6。
核對：
  接近段（CRUISE 未漸變、序列 U／H／R）每個控制拍：現在停（e0）與延後一拍停（e1）的完整取樣停止軌跡（compactcarry21.stop_traj），
    每構型核第一桌（該案桌高）、第二桌、原障礙方塊、瓶 ≥ 0.05 m；自碰（開爪）≥ 8.017 mm；關節餘裕 ≥ 0.05；非有限、未停完。
  參考連續：每拍手臂關節參考（序列＋g·e1）的一階差分／DT 最大值（對 v_arm）、二階差分／DT² 最大值；MPC 手臂命令的相鄰拍差分／DT 最大值（對 A_ARM）。
  視向：|g|、|ġ|、|g̈| 最大值、R 段開始時 g、歸零時刻、DONE 是否早於漸變起點、序列等候紀錄。
  覆蓋（評估用）：假世界量測構型下實際瓶身＋支撐環在畫面內的比例（U 開始到 R 開始）。
  移動抓取：U 開始到閉爪請求之間底盤最低速度（> 0 ⇒ 未停車）；完整抓放；收近起到鬆爪的停止路徑（同 test_bottle_w1_rb1）；實際瓶對兩桌。
    python3 evaluation/test_bottle_w1_gaze.py all       （四組依序；節流在落後時重新錨定，見 step；寫 results/bottle/gaze_closed.json）
    python3 evaluation/test_bottle_w1_gaze.py rb1 on
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'gaze_closed.json')
COMBOS = [('u3', 'on'), ('u3', 'off'), ('rb1', 'on'), ('rb1', 'off')]
CASES = dict(u3=dict(run='bottle_online5_u3_1', z1=0.598, b=(0.0042, 0.4564), O=(0.0, 0.0)),
             rb1=dict(run='bottle_rb1_1', z1=0.58, b=(-0.03, 0.44), O=(0.60, 0.80)))

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, *k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return '_'.join(k), (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:]))
    with ThreadPoolExecutor(1) as ex:
        res = dict(ex.map(one, COMBOS))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in res:
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 't_done', 'place_mm', 'gaze_sum', 'ref_rate', 'u_rate', 'approach_stop_fails', 'approach_stop_worst', 'stop_fails', 'coverage',
                                           'v_base_min_U_close', 'seq', 'done_before_blend', 'error')})
    sys.exit(0 if all(r.get('pass') for r in res.values()) else 1)

CASE, GZON = sys.argv[1], sys.argv[2] == 'on'
CF = CASES[CASE]
Z1, D_U = CF['z1'], 3.6
sys.argv = [sys.argv[0], f'{Z1:.3f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = CF['b']
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_gaze_ctrl as GZ                        # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import bottle_gaze_offline as GO                        # noqa: E402

T = TV.T
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_region as BRG                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
REG = BRG.RegionFrame(*CF['O'])
if CASE == 'rb1':
    PLACE_R = REG.place_to_region(PC.PLACE)
    PC.PLACE.clear()
    PC.PLACE.update(PLACE_R)
TP.DX, TP.DY = 2.4 - CF['O'][0], -1.0 - CF['O'][1]
SC.X0 = -3.6 - CF['O'][0]
SC.OBST['c'] = np.array([-2.0 - CF['O'][0], 0.0 - CF['O'][1], 0.25])
J_RUN = np.array(json.load(open(os.path.join(HERE, 'runs', CF['run'], 's1_report.json')))['plan']['joined'], float)
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
OBX = (SC.OBST['c'], SC.OBST['sz'])
BOXC, BOXS = np.array([CF['b'][0], CF['b'][1], Z1 + G.BOTTLE['h'] / 2]), np.array([2 * G.BOTTLE['r'], 2 * G.BOTTLE['r'], G.BOTTLE['h']])
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
ALIM = dict(table=0.05, obst=0.05, bottle=0.05, self=0.008017 - 1e-6, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
EV_HOT = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned', 'xh_back', 'xh_tcp_converged', 'place_phase', 'contact_first',
          'return_plan_accept')
REC, TBLOG, AREC, UREC = [], [], [], []
AST = dict(n=0, n_cfg=0, worst={}, where={}, nonfinite=0, unfinished=0)
RTF_CAP = float(os.environ.get('GZ_RTF_CAP', '0.6'))
LOGP = os.path.join(HERE, 'results', 'bottle', f"gaze_closed_{CASE}_{'on' if GZON else 'off'}_worker.log")
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


def achk(q9):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for _, c_, s_ in TBN), obst=float(G.box_dist(P, *OBX).min()), bottle=float(G.box_dist(P, BOXC, BOXS).min()),
                self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=BB.margin(q9[3:]))


class Ctl(GZ.W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, gaze_on=GZON, **kw)

    def _advance_inner(self, T_tcp, close_done, q):
        ok_ = self.judge.stop is None and np.isfinite(q).all()
        if ok_ and self.st.phase == 'CRUISE' and not self.blending and self.vh_seq['ph'] in ('U', 'H', 'R'):
            AST['n'] += 1
            for e in (0, 1):
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                if len(Q) >= N_MAX:
                    AST['unfinished'] += 1
                if not np.isfinite(Q).all():
                    AST['nonfinite'] += 1
                    continue
                for qq in Q:
                    AST['n_cfg'] += 1
                    for k, v in achk(qq).items():
                        key = f'e{e}:{k}'
                        if v < AST['worst'].get(key, 9.9):
                            AST['worst'][key], AST['where'][key] = v, (round(float(self.t), 3), self.vh_seq['ph'], round(float(q[0]), 4))
        if ok_ and self.cc is not None:
            REC.append(dict(t=float(self.t), pph=self.pph, stage=self.cc['stage'], q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy()))
        super()._advance_inner(T_tcp, close_done, q)
        if ok_ and self.st.phase == 'CRUISE' and not self.blending:
            AREC.append((float(self.t), self.vh_seq['ph'], self._qseq(self.t) + (self.gz['g'] if self.gaze_on else 0.0) * GZ.E1, np.asarray(q, float).copy()))

    def _solve(self, q):
        u = super()._solve(q)
        if u is not None and self.st.phase == 'CRUISE' and not self.blending:
            UREC.append((float(self.t), np.asarray(u, float).copy()))
        return u


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
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None)
    if os.environ.get('GZ_DUMP'):                                   # 診斷：記錄送給工作者的請求內容
        _sub = c.pc.submit

        def sub(kind, inputs):
            with open(os.environ['GZ_DUMP'], 'a') as fd:
                fd.write(json.dumps(dict(kind=kind, inputs=inputs), default=lambda o: np.asarray(o).tolist()) + '\n')
            return _sub(kind, inputs)
        c.pc.submit = sub
    step0, meas0 = c.step, f.meas

    def meas(**ov):
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy()))
        return m
    f.meas = meas
    import time as _time
    st['wall0'] = None

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        else:                                                       # 落後（本檔接近段每拍算停止軌跡）時重新錨定：之後任何時段都不快於牆鐘×RTF_CAP，
            st['wall0'] = _time.time() - t / RTF_CAP               # 不再以累積落後在持瓶段數倍速追趕（否則工作者 1.7 s 牆鐘變成 6 s 以上模擬時間）
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
    ev = r['events']
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    on = r.get('online') or {}
    ph = {}
    for e in ev:
        if e['what'] in ('phase', 'place_phase'):
            ph.setdefault(e['to'], e['t'])
    pm = (r.get('place') or {}).get('metrics') or {}
    gps = on.get('grasp_plans') or []
    rps = on.get('return_plans') or []
    vs = r['vhigh_seq']
    t_of = {e[1]: e[0] for e in vs['log']}
    t_blend = next((e['t'] for e in ev if e['what'] == 'blend_start'), None)
    out = dict(case=CASE, gaze=GZON, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               close_req=c.t_close_req, place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None,
               gp=[{k: g.get(k) for k in ('id', 'state', 't_acc', 'x_req', 'x_latest', 'margin')} for g in gps], rp=[{k: x.get(k) for k in ('id', 'state', 't_req', 't_res', 't_acc', 'compute_wall')} for x in rps],
               rp_wall=[dict(id=x.get('id'), wall_req=x.get('wall_req'), wall_recv=x.get('wall_recv')) for x in c.rp_hist], t_grasp=r.get('t_grasp'), rtf_cap=RTF_CAP, client=on.get('client'),
               slip_max=r['slip_max_mm_deg'], seq=vs['log'], waits=vs['waits'], t_blend=t_blend, done_before_blend=None if t_blend is None or 'DONE' not in t_of else bool(t_of['DONE'] <= t_blend + 1e-9))
    gz = r.get('gaze') or {}
    out['gaze_sum'] = dict(max={k: round(v, 4) for k, v in (gz.get('max') or {}).items()}, ev=gz.get('ev'), counts=gz.get('counts'))
    # 參考連續與命令變化率（接近段）
    A = [x for x in AREC]
    tq = np.array([x[0] for x in A])
    QR = np.array([x[2] for x in A])
    d1 = np.abs(np.diff(QR, axis=0)) / B.DT
    d2 = np.abs(np.diff(QR, 2, axis=0)) / B.DT ** 2
    out['ref_rate'] = dict(v_max_per_joint=d1.max(axis=0).round(4).tolist(), v_arm=B.W_ARM['v_arm'], a_max_per_joint=d2.max(axis=0).round(3).tolist(), n=len(A))
    tu = np.array([x[0] for x in UREC])
    UA = np.array([x[1][3:] for x in UREC])
    du = np.abs(np.diff(UA, axis=0)) / np.maximum(np.diff(tu), 1e-9)[:, None]
    kpk = du.argmax(axis=0)
    out['u_rate'] = dict(arm_max_per_joint=du.max(axis=0).round(3).tolist(), A_ARM=C0.A_ARM, arm_u_max=np.abs(UA).max(axis=0).round(4).tolist(),
                         peak_t=[round(float(tu[k_ + 1]), 3) for k_ in kpk], top_j1=[(round(float(tu[k_ + 1]), 3), round(float(du[k_, 0]), 3)) for k_ in np.argsort(-du[:, 0])[:6]],
                         n_over_2=[int((du[:, j] > 2.0).sum()) for j in range(6)])
    # 接近段停止軌跡
    out['approach_stop_n'] = AST['n']
    out['approach_stop_cfg'] = AST['n_cfg']
    out['approach_stop_worst'] = {k: round(v, 5) for k, v in AST['worst'].items()}
    out['approach_stop_where'] = AST['where']
    out['approach_stop_fails'] = sorted(k for k, v in AST['worst'].items() if v < ALIM[k.split(':')[1]]) + (['nonfinite'] if AST['nonfinite'] else []) + (['unfinished'] if AST['unfinished'] else [])
    # 覆蓋（假世界量測構型）
    tU, tR = t_of.get('U'), t_of.get('R')
    cv = [GO.cover(x[3], CF['b'], Z1) for x in A if tU is not None and tR is not None and tU - 1e-9 <= x[0] <= tR + 1e-9]
    out['coverage'] = dict(U_to_R_real=round(float(np.mean([x['real'] for x in cv])), 3) if cv else None, U_to_R_area=round(float(np.mean([x['area'] for x in cv])), 3) if cv else None, n=len(cv))
    # 移動抓取：U 開始到閉爪請求之間底盤最低速度
    th = np.array([h['t'] for h in f.hist])
    Qh = np.array([h['q'][:3] for h in f.hist])
    vb = np.r_[0.0, np.linalg.norm(np.diff(Qh[:, :2], axis=0), axis=1) / np.maximum(np.diff(th), 1e-9)]
    if tU is not None and c.t_close_req is not None:
        m_ = (th >= tU) & (th <= c.t_close_req)
        vsm = np.convolve(vb[m_], np.ones(10) / 10, mode='valid')
        out['v_base_min_U_close'] = round(float(vsm.min()), 4) if len(vsm) else None
    # 收近起到鬆爪停止路徑（同 test_bottle_w1_rb1）
    rt = np.array([x['t'] for x in REC])
    t_rel = ph.get('RELEASE', 1e9)
    span = [i for i, x in enumerate(REC) if x['t'] <= t_rel]
    sel = set(span[::10])
    for te in sorted({float(e['t']) for e in ev if e['what'] in EV_HOT}):
        k = int(np.argmin(np.abs(rt - te))) if len(rt) else 0
        sel.update(i for i in range(k - 5, k + 6) if i in set(span))
    worst, nonfin, unfin = {}, 0, 0
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
                    worst[key] = min(worst.get(key, 9.9), v)
    out['stop_n'] = len(sel)
    out['stop_worst'] = {k: round(v, 5) for k, v in worst.items()}
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
    acc_g = [g for g in gps if g.get('state') == 'accepted']
    acc_r = [x for x in rps if x.get('state') == 'accepted']
    gm = (gz.get('max') or {})
    gaze_ok = (not GZON) or (gm.get('v', 9) <= GZ.V_G + 1e-9 and gm.get('a', 9) <= GZ.A_G + 1e-6 and (gz.get('state') or {}).get('g') == 0.0)
    ok = (done and not out['stop_fails'] and not out['approach_stop_fails'] and bt['v'] >= LIM['held_bottle_table'] and len(acc_g) == 1 and len(acc_r) == 1 and gaze_ok
          and out['done_before_blend'] is True and (out.get('v_base_min_U_close') or 0) > 0.0)
    out['pass'] = bool(ok and T.finite_cmds(f))
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
