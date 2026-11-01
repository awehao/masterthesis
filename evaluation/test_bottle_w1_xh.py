#!/usr/bin/env python3
"""多任務 v3（bottle_w1_vhigh_relwin_xh_ctrl，跨桌高）離線核對（Codex 20261102_070114；假世界、假候選，非物理）。
案例：B1（第一桌 0.55、瓶 (−0.04, 0.44)，目標案）、B2（0.65、(+0.04, 0.40)）、A4（0.60、(+0.08, 0.46)，名目高度；另跑 v2w 同條件 A4w 對照退化）。
假候選同 test_bottle_w1_relwin_uadopt（U 段開始後 3.6 s 起送）。核對：
1 結果與三段（抓起／運送／放置）、z_back 與側持原高度、瓶底在轉回高度的位置（應 ≥ 第二桌面＋Z_PAD＋H_CLEAR）、外伸淨空閘觸發餘裕與監看最小、TCP 收斂時刻；
2 時長重核：以實際收近起點構型與 T_rel，沿 收近（原 T_side → 緊湊）、升高（緊湊 → z_back）、外伸（→ T_back）三段做 40 步 IK 延續
  （bottle_w1_compact_timing.path_q：關節速度／加速度所需 T、自碰、持瓶↔連桿、車體頂、關節餘裕），對 T_TO／T_UP／T_OUT；
3 停止路徑：收近開始到鬆爪，基底每 10 控制拍＋各換段事件 ±5 拍逐拍，兩條 stop_traj（現在停／延後一拍），對兩桌（桌面＋桌腳）、自碰、關節餘裕、
  持瓶（判定器 T_rel）對桌；放置段（PARKED 起）持瓶只排除第二桌桌面（指定支撐面），第一桌與桌腳照核。判準同 test_bottle_w1_relwin_batch。
4 實際瓶（假受控對象位姿）對兩桌：離桌成立＋0.2 s 起，PARKED 起排除第二桌桌面。
取樣核對，非全流程保證。
    python3 evaluation/test_bottle_w1_xh.py all
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'bottle', 'xh_offline.json')
CASES = dict(B1=(0.55, -0.04, 0.02, 'xh'), B2=(0.65, 0.04, -0.02, 'xh'), A4=(0.60, 0.08, 0.04, 'xh'), A4w=(0.60, 0.08, 0.04, 'v2w'))

if len(sys.argv) > 1 and sys.argv[1] == 'all':
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        p = subprocess.run([sys.executable, __file__, k], capture_output=True, text=True, timeout=5400)
        line = [l_ for l_ in p.stdout.splitlines() if l_.startswith('{')]
        return k, (json.loads(line[-1]) if line else dict(case=k, error=p.stderr[-2500:]))
    with ThreadPoolExecutor(3) as ex:
        res = dict(ex.map(one, list(CASES)))
    a, b = res.get('A4', {}), res.get('A4w', {})
    res['degeneracy_A4'] = dict(t_done=[a.get('t_done'), b.get('t_done')], place_mm=[a.get('place_mm'), b.get('place_mm')], same_phases=a.get('phases') == b.get('phases'))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in CASES:
        r = res[k]
        print(k, {kk: r.get(kk) for kk in ('pass', 'outcome', 't_done', 'place_mm', 'xh', 'reach', 'conv', 'timing', 'stop_fails', 'bottle_table', 'error')})
    print('A4 退化', res['degeneracy_A4'])
    sys.exit(0)

CASE = sys.argv[1]
Z1, DXB, DYB, VER = CASES[CASE]
D_U = 3.6
sys.argv = [sys.argv[0], f'{Z1:.2f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = 0.0 + DXB, 0.42 + DYB
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compact_timing as TM                   # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2m              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_vhigh_relwin_slipwarn_ctrl as SW       # noqa: E402
import bottle_w1_vhigh_relwin_xh_ctrl as XH             # noqa: E402
import test_bottle_place as TP                          # noqa: E402

T = TV.T
TBN = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes()] + [('t2:' + n, c_, s_) for n, c_, s_ in TP.table2_boxes()]
BPT = G.bottle_points()
LIM = dict(nonfinger_table=0.01, finger_table=0.005, held_bottle_table=0.0055, self=0.0, margin=0.05)
PLACING = ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')
EV_HOT = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned', 'xh_back', 'xh_tcp_converged', 'place_phase', 'contact_first')
REC, TBLOG = [], []
BASE = XH.W1PlaceController if VER == 'xh' else SW.W1PlaceController


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


class Ctl(BASE):
    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and np.isfinite(q).all() and self.cc is not None:
            REC.append(dict(t=float(self.t), pph=self.pph, stage=self.cc['stage'], q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(),
                            pipe=[np.asarray(u, float).copy() for u in self.pipe], u_last=np.asarray(self.u_last, float).copy(), f_cmd=copy.deepcopy(self.f_cmd),
                            T_rel=None if self.judge.T_rel is None else np.asarray(self.judge.T_rel, float).copy(), T_side=np.array(self.cc['T_side'], float)))
        super()._advance_inner(T_tcp, close_done, q)


def run():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    rng = np.random.default_rng(0)
    st = dict(next_t=0.0, seq=0, t_seq=-99.0, snap=None)
    step0, meas0 = c.step, f.meas

    def meas(**ov):
        m = meas0(**ov)
        TBLOG.append((float(f.t), np.array(f.Tb, float).copy()))
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


def timing(c):
    """沿實際收近起點構型重核三段所需時長（T_TO／T_UP／T_OUT）。"""
    if not REC:
        return None
    x0 = REC[0]
    T_side0 = x0['T_side']
    back = (getattr(c, 'xh', None) or {}).get('z_back')
    T_back = T_side0.copy()
    if back is not None:
        T_back[2, 3] = back
    T_c = T_side0.copy()
    T_c[:3, 3] = C2m.TCP_C
    T_mid = T_c.copy()
    T_mid[2, 3] = T_back[2, 3]
    GC = CP.grip_corners(x0['q'][3:])
    Tr = x0['T_rel']
    out = {}
    try:
        Q1, s1 = TM.path_q(x0['q'][3:], T_side0, T_c, Tr, GC)
        Q2, s2 = TM.path_q(Q1[-1], T_c, T_mid, Tr, GC)
        Q3, s3 = TM.path_q(Q2[-1], T_mid, T_back, Tr, GC)
    except Exception as e:                                                 # IK 延續失敗照實記
        return dict(error=repr(e))
    for nm, sg, Tset in (('to_compact', s1, XH.T_TO_XH), ('raise', s2, XH.T_UP_XH), ('out', s3, XH.T_OUT_XH)):
        out[nm] = dict(T_need=round(sg['T'], 3), T_used=Tset, ok=bool(sg['T'] <= Tset + 1e-9), self_mm=round(sg['worst']['self'] * 1e3, 1), bottle_link_mm=round(sg['worst']['bottle_link'] * 1e3, 1),
                       deck=round(sg['worst']['deck'], 3), margin=round(sg['worst']['margin'], 3))
    return out


def main():
    c, f, r = run()
    ev = r['events']
    why = (r['stop'] or {}).get('why')
    done = bool(r.get('mission_done'))
    ph = {}
    for e in ev:
        if e['what'] in ('phase', 'place_phase'):
            ph.setdefault(e['to'], e['t'])
    xh = (r.get('xh') or {})
    cc = r.get('compact_carry') or {}
    rg = cc.get('reach_gate') or {}
    pm = (r.get('place') or {}).get('metrics') or {}
    # 停止路徑
    rt = np.array([x['t'] for x in REC])
    t_rel = ph.get('RELEASE', 1e9)
    span = [i for i, x in enumerate(REC) if x['t'] <= t_rel]
    sel = set(span[::10])
    et = sorted({float(e['t']) for e in ev if e['what'] in EV_HOT})
    for te in et:
        k = int(np.argmin(np.abs(rt - te)))
        sel.update(i for i in range(k - 5, k + 6) if i in set(span))
    worst, where, nonfin, unfin = {}, {}, 0, 0
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
                    if v < worst.get(key, 9.9):
                        worst[key], where[key] = v, (round(x['t'], 2), x['pph'], x['stage'])
    fails = sorted(k for k, v in worst.items() if v < LIM[k.split(':')[1]]) + (['nonfinite'] if nonfin else []) + (['unfinished'] if unfin else [])
    # 實際瓶對桌
    t_lok, t_park = r.get('t_lift_ok'), ph.get('PARKED', 1e9)
    bt = dict(v=9.9, t=None, box=None)
    if t_lok is not None:
        for tt_, Tb_ in TBLOG[::5]:
            if tt_ < t_lok + 0.2:
                continue
            P_ = (Tb_[:3, :3] @ BPT.T).T + Tb_[:3, 3]
            for n_, c_, s_ in TBN:
                if tt_ >= t_park and n_ == 't2:table_top':
                    continue
                d_ = float(G.box_dist(P_, c_, s_).min())
                if d_ < bt['v']:
                    bt.update(v=round(d_, 4), t=round(tt_, 2), box=n_)
    tm = timing(c) if VER == 'xh' else None
    tm_ok = tm is None or ('error' not in tm and all(v['ok'] for v in tm.values()))
    back = xh.get('back') or {}
    ok = bool(done and not fails and T.finite_cmds(f) and bt['v'] >= LIM['held_bottle_table'] and tm_ok
              and (VER != 'xh' or (back and back['bottom_at_back'] >= back['top2'] + back['z_pad'] + back['h_clear'] - 1e-6)))
    out = dict(case=CASE, ver=VER, z1=Z1, bottle=[DXB, 0.42 + DYB], **{'pass': ok}, outcome='完整抓放' if done else f'停止 {why}', stop=r['stop'], t_done=next((e['t'] for e in ev if e['what'] == 'mission_done'), None),
               phases=ph, place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None, place=pm,
               xh={k: (round(v, 4) if isinstance(v, float) else v) for k, v in back.items()} if back else None, conv=xh.get('converged'),
               reach=dict(trigger=rg.get('trigger'), monitor_min=rg.get('monitor_min')), timing=tm, slip_max=r['slip_max_mm_deg'], slip_warn=(r.get('slip_warn') or {}).get('warn'),
               stop_n=len(sel), stop_worst={k: round(v, 5) for k, v in worst.items()}, stop_where=where, stop_fails=fails, bottle_table=bt, limits=LIM,
               note='假世界、假候選；取樣核對，非全流程保證')
    print(json.dumps(out, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
