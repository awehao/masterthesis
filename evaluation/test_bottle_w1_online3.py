#!/usr/bin/env python3
"""在線 MotM 第三版（提前安全轉回、與運送重疊；bottle_w1_online3_ctrl）對照第二版的針對性閉迴路（Codex 20261102_080657；假世界、假候選，非物理）。
幾何：U3 位置（第一桌 0.598、瓶 (+0.0042, 0.4564)）——U3 物理已用第二版跑過，這裡只核排程差異。
  base   第二版（online2）；early  第三版（online3）。兩者都報：底盤就緒（首次進停車位 20 mm 內）、手臂就緒（TCP 收斂）、實際下放（PARKED／LOWER）、完成；
  外伸暴露（離桌成立到鬆爪，持瓶＋夾爪點到底盤中心的最大水平距離 > 0.39 m 的總時間與最大值）；外伸淨空閘觸發餘裕與監看最小；停止路徑與實際瓶對桌（同下）。
以下為原說明。
區域導引在線 MotM（bottle_w1_online_ctrl）的最小離線核對（Codex 20261102_072949：接口反例、必要退化、一次最小閉迴路；假世界、假候選，非物理）。
幾何：B1 位置（第一桌 0.55、瓶 (−0.04, 0.44)；第一批已用過的設計域點，不是之後的三個新位置）。假候選同 test_bottle_w1_relwin_uadopt（U 段開始後 3.6 s 起送）。
規劃工作者是真的子程序（同物理接線）；延遲以牆鐘與假世界模擬時間各記。假世界節流到模擬時間不快於牆鐘×0.6（Isaac 實跑 RTF 約 0.58），
否則運送段假世界比真實快約 3.8 倍，工作者 1.8 s 牆鐘會變成 6.7 s 模擬時間（第一版即如此）。
  closed    最小閉迴路：完整抓放、抓取計畫延遲、轉回計畫是否在第一次收近閘前就緒、時長＝工作者所需×1.05 進位（不低於原值）、
            停止路徑（收近起到鬆爪，基底每 10 控制拍＋換段 ±5 拍；放置段持瓶只排除第二桌桌面）、實際瓶對桌。
  dead      第一個抓取請求送出後殺掉工作者 ⇒ 期望 planner_dead 受控停止、未閉爪。
  slow      抓取結果不交給控制器（模擬工作者過慢）⇒ 期望在最晚處置點 grasp_plan_late 受控停止、未閉爪、停止時底盤 x < 新漸變起點。
  moved     抓取請求送出後假候選目標改 +5 mm ⇒ 期望舊請求作廢、重送（gp_n＝2）、以新目標承接、完整抓放。
  envelope  轉回請求送出後持瓶相對位姿再平移 4 mm（模擬滑移）⇒ 期望承接時 envelope_changed 拒收、有界重送後承接、完整抓放。
    python3 evaluation/test_bottle_w1_online.py all
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'online3_offline.json')
MODES = ('base', 'early')

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
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 't_done', 'timeline', 'exposure', 'reach', 'stop_fails', 'bottle_table', 'slip_max', 'place_mm', 'error')})
    sys.exit(0)

MODE = sys.argv[1]
Z1, DXB, DYB, D_U = 0.598, 0.0042, 0.0364, 3.6
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_online2_ctrl as ON2                    # noqa: E402
import bottle_w1_online3_ctrl as ON3                    # noqa: E402
import test_bottle_place as TP                          # noqa: E402

T = TV.T
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
EV_HOT = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned', 'xh_back', 'xh_tcp_converged', 'place_phase', 'contact_first',
          'return_plan_accept')
REC, TBLOG = [], []
RTF_CAP = 0.6                                                           # Isaac 實跑約 0.58
LOGP = os.path.join(HERE, 'results', 'bottle', f'online3_offline_{MODE}_worker.log')


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


class Ctl((ON3 if MODE == 'early' else ON2).W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all() and self.cc is not None:
            REC.append(dict(t=float(self.t), pph=self.pph, stage=self.cc['stage'], q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy()))
        super()._advance_inner(T_tcp, close_done, q)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None, shift=np.zeros(2), killed=False, env_done=False, Trel0=None)
    step0, meas0 = c.step, f.meas
    if MODE == 'slow':
        c.pc.poll = lambda rid, _p=c.pc.poll: (None if any(g['id'] == rid for g in c.gp_hist) else _p(rid))   # 抓取結果永不交付
    if MODE == 'moved':
        c.pc.poll = lambda rid, _p=c.pc.poll: (None if c.gp_hist and rid == c.gp_hist[0]['id'] else _p(rid))   # 第一個請求的結果扣住（確定被作廢，不因時序先承接舊目標）

    def meas(**ov):
        if MODE == 'envelope' and not st['env_done'] and c.rp is not None and c.rp['state'] in ('pending', 'ready') and f.grasped:
            f.Trel = np.array(f.Trel, float).copy()
            f.Trel[0, 3] -= 0.004                                       # 轉回請求送出後持瓶相對位姿再平移 4 mm
            st['env_done'] = True
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy()))
        return m
    f.meas = meas

    import time as _time
    st['wall0'] = None

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP                  # 節流：模擬時間不快於牆鐘×RTF_CAP（假世界有時比真實快數倍，否則工作者延遲在模擬時間上失真）
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        if MODE == 'dead' and not st['killed'] and c.gp is not None:
            c.pc.proc.kill()
            st['killed'] = True
        if MODE == 'moved' and c.gp is not None and c.gp_n == 1 and not st['shift'].any():
            st['shift'] = np.array([0.005, 0.0])
            st['snap'] = None                                            # 下一個快照（新序號）帶新目標
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_t'] - 1e-9:
            st['next_t'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array([T.T.BX, T.T.BY]) + st['shift'] + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
        return step0(t, q, **kw)
    c.step = step
    r, _ = T.run(c, f, t_end=80.0)
    return c, f, r


EXPECT = dict(base='第二版：完整抓放（對照）', early='第三版：完整抓放、手臂就緒提前、下放提前；停止路徑與淨空照核')
GRIPL = ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


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
    out = dict(mode=MODE, expect=EXPECT[MODE], outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               close_req=c.t_close_req, place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None,
               gp=[{k: g.get(k) for k in ('id', 'state', 'seq', 'xy', 't_req', 't_res', 't_acc', 'x_req', 'x_latest', 'margin', 'compute_wall')} for g in gps],
               rp=[{k: x.get(k) for k in ('id', 'state', 't_req', 't_res', 't_acc', 'compute_wall')} for x in rps], latency=dict(grasp=on.get('grasp_latency'), ret=on.get('return_latency')),
               dur=on.get('durations'), need={k: round(v['T_need'], 3) for k, v in ((next((x for x in rps if x['state'] == 'accepted'), {}).get('res') or {}).get('seg') or {}).items()},
               t1=on.get('t1_measured'), xh=on.get('xh'), client=on.get('client'), slip_max=r['slip_max_mm_deg'], x_blend=getattr(c, 'x_blend', None))
    if True:
        rt = np.array([x['t'] for x in REC])
        t_rel = ph.get('RELEASE', 1e9)
        span = [i for i, x in enumerate(REC) if x['t'] <= t_rel]
        sel = set(span[::10])
        for te in sorted({float(e['t']) for e in ev if e['what'] in EV_HOT}):
            k = int(np.argmin(np.abs(rt - te)))
            sel.update(i for i in range(k - 5, k + 6) if i in set(span))
        worst, nonfin, unfin = {}, 0, 0
        for i in sorted(sel):
            x = REC[i]
            placing = (x['pph'] or '') in PLACING
            for e in (0, 1):
                Qs = C21.stop_traj(x['q'], x['s'], x['pipe'], x['u_last'], c.cfg, e)
                if len(Qs) >= 2000 // C21.SAMPLE_EVERY + 1:
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
        # 時序：底盤就緒＝首次進停車位 20 mm 內；手臂就緒＝TCP 收斂；實際下放＝PARKED
        H = f.hist
        park = getattr(c, 'park', None)
        t_carry = ph.get('CARRY_NAV', 0.0)
        tb_ready = next((h['t'] for h in H if park is not None and h['t'] >= t_carry and float(np.hypot(h['q'][0] - park[0], h['q'][1] - park[1])) <= 0.02), None)
        evt = {}
        for e in ev:
            evt.setdefault(e['what'] + (':' + e['to'] if e.get('to') else ''), e['t'])
        out['timeline'] = dict(compact_start=evt.get('compact_start'), compact_reached=evt.get('compact_reached'), raise_start=evt.get('compact_raise_start'), raised=evt.get('compact_raised'),
                               out_start=evt.get('compact_out_start'), out_done=evt.get('compact_returned'), place_approach=evt.get('place_phase:PLACE_APPROACH'),
                               base_ready=tb_ready, arm_ready=(on.get('xh_conv') or {}).get('t'), lower=evt.get('place_phase:PARKED'), release=evt.get('place_phase:RELEASE'), done=out['t_done'],
                               raise_why=next((e.get('why') for e in ev if e['what'] == 'compact_raise_start'), None))
        tl = out['timeline']
        out['timeline']['table_wait'] = None if tl['lower'] is None or tl['base_ready'] is None else round(tl['lower'] - tl['base_ready'], 3)
        # 外伸暴露：離桌成立到鬆爪，持瓶（實際瓶位姿）＋夾爪點到底盤中心最大水平距離 > 0.39 m 的總時間
        tbt = np.array([x[0] for x in TBLOG])
        exp_s, rmax, t_rmax = 0.0, 0.0, None
        t_rel_ = ph.get('RELEASE', 1e9)
        for k in range(0, len(H), 5):
            h = H[k]
            if t_lok is None or h['t'] < t_lok or h['t'] > t_rel_:
                continue
            j = int(np.argmin(np.abs(tbt - h['t'])))
            Tb_ = TBLOG[j][1]
            W = G.link_world(h['q'], 0.0038)
            P_ = np.vstack([(Tb_[:3, :3] @ BPT.T).T + Tb_[:3, 3]] + [v for kk, v in W.items() if kk in GRIPL])
            rr = float(np.hypot(P_[:, 0] - h['q'][0], P_[:, 1] - h['q'][1]).max())
            if rr > 0.39:
                exp_s += 0.05
            if rr > rmax:
                rmax, t_rmax = rr, h['t']
        out['exposure'] = dict(time_beyond_0p39_s=round(exp_s, 2), r_max_m=round(rmax, 3), t_r_max=t_rmax)
        rg = (r.get('compact_carry') or {}).get('reach_gate') or {}
        out['reach'] = dict(trigger=rg.get('trigger'), monitor_min=rg.get('monitor_min'))
    acc_g = [g for g in gps if g.get('state') == 'accepted']
    acc_r = [x for x in rps if x.get('state') == 'accepted']
    ok = done and not out.get('stop_fails') and (out.get('bottle_table') or {}).get('v', 0) >= LIM['held_bottle_table'] and len(acc_g) == 1 and len(acc_r) == 1
    out['pass'] = bool(ok and T.finite_cmds(f))
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
