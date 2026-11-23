#!/usr/bin/env python3
"""緊湊持物 v2.1（停止軌跡核對）與 v3（提前升高的轉回排程）的針對性離線核對（Codex 20261101_204716 第 2、4 項；假受控對象，非物理）。
A v2.1 名目：完成且放置判準通過；閘起點與收近中（含兩條停止軌跡）餘裕 ≥ 0；閘與停止軌跡單拍耗時。
B 停止軌跡模型 vs 假世界：v2.1 收近中 0.5×T_to 起掃描中斷 ⇒ 受控停止；以停止那一步開始時的控制器狀態積分模型停止軌跡；
  判準＝實際停止期間每個 TCP 到交叉集合（底盤停止路徑取樣位姿 × 手臂停止路徑取樣構型，同閘）、底盤到模型底盤路徑的最近距離（涵蓋，取樣間距 4 物理步）；逐點差與底盤停止距離（實際 vs 模型）另報。
C v3 名目 vs v2 名目：完成且放置判準通過；升高在 PLACE_APPROACH 前開始（early）、外伸在 PLACE_APPROACH 且外伸閘餘裕 ≥ 0；
  停車前近停（平移 < 0.01 m/s 連續 ≥ 0.1 s，抓後到 PARKED）、DONE、外伸持續時間（持瓶到 LOWER 前，瓶身＋夾爪水平半徑 > 0.30 m 的累計秒數）、升高後近身持瓶秒數；
  持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限、設定點速率峰 ≤ v_arm、滑移。
D v3 停止：升高開始後 1.0 s 起、外伸開始後 1.9 s 起各注入掃描中斷 ⇒ 受控停止且停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm。
E 外伸閘非有限 ⇒ None。
    python3 evaluation/test_bottle_w1_compactcarry3.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
import bottle_w1_compactcarry3_ctrl as C3               # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_compactcarry as TC                # noqa: E402
import test_bottle_w1_compactcarry2 as T2               # noqa: E402
import test_bottle_w1_frontcarry as TF                  # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(cls, cut_at=None, capture=False):
    """cut_at(c) → 注入起點（None＝尚未決定）；capture＝在受控停止當拍記下控制器狀態。"""
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    st = dict(t_cut=None, cap=None, q=None)
    if cut_at is not None:
        fn0 = c.scan_fn

        def scan_cut(t, q):
            if st['t_cut'] is None:
                st['t_cut'] = cut_at(c)
            if st['t_cut'] is not None and t >= st['t_cut'] - 1e-9:
                return None
            return fn0(t, q)
        c.scan_fn = scan_cut
    if capture:
        step0 = c.step

        def step(t, q, **kw):
            pre = dict(t=float(t), q=np.asarray(q, float).copy(), s=c.s.copy(), pipe=[u.copy() for u in c.pipe], u=c.u_last.copy(), phase=c.n_step % 5)
            was = c.judge.stop
            out = step0(t, q, **kw)
            if was is None and c.judge.stop is not None and st['cap'] is None:
                st['cap'] = pre                             # 停止那一步「開始時」的狀態（減速與管線推進都還沒做）
            return out
        c.step = step
    r, _ = T.run(c, f)
    return c, f, r, st


def near_stops(f, r):
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    tg, tp = r['t_grasp'], pe.get('PARKED', 1e9)
    H = [h for h in f.hist if tg <= h['t'] < tp]
    t = np.array([h['t'] for h in H])
    xy = np.array([h['q'][:2] for h in H])
    v = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T) / np.diff(t)]
    out, i = [], 0
    while i < len(t):
        if v[i] < 0.01:
            j = i
            while j + 1 < len(t) and v[j + 1] < 0.01:
                j += 1
            if t[j] - t[i] >= 0.1:
                out.append((round(float(t[i]), 2), round(float(t[j]), 2)))
            i = j + 1
        else:
            i += 1
    return out


def reach_time(c, f, r):
    """持瓶（LIFT 起）到 LOWER 前：瓶身＋夾爪水平半徑 > 0.30 m 的累計秒數；另回傳升高後（緊湊 xy、側持高度）的持瓶秒數。"""
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    ev = {e['what']: e['t'] for e in r['events']}
    t_low = pe.get('LOWER', 1e9)
    Pt = c._pts_tcp() if c.judge.T_rel is not None or c._Pt is not None else None
    H = [h for h in f.hist if h['phase'] in ('LIFT', 'CARRY') and h['t'] <= t_low]
    dt = H[1]['t'] - H[0]['t'] if len(H) > 1 else 0.0
    n_out = 0
    for h in H:
        q = h['q']
        Tw = T.T.K.fk(q, 'link_tcp')
        P = (Tw[:3, :3] @ Pt.T).T + Tw[:3, 3]
        if float(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]).max()) > 0.30:
            n_out += 1
    raised = (ev.get('compact_out_start') or ev.get('compact_return') or 0.0) - (ev.get('compact_raised') or ev.get('compact_return') or 0.0) if 'compact_raised' in ev else 0.0
    return round(n_out * dt, 2), round(max(raised, 0.0), 2)


def main():
    res = {}
    # A＋B：v2.1
    c, f, r, _ = run(C21.W1PlaceController)
    a = TC.summarize('v2.1 名目', c, f, r)
    g = r['compact_carry']['gate']
    print(f"   閘：起點 {g['trigger']['t']:.2f} s 餘裕 {g['trigger']['margin']:.4f}；收近中最小 {g['monitor_min']}；停止軌跡最小 {g['stop_traj']['min_margin']}、狀態數 ≤ {g['stop_traj']['max_states']}；"
          f"耗時 閘 {g['ms_max']:.2f}／停止軌跡 {g['stop_traj']['ms_max']:.2f} ms（最大）；v_ahead {g['v_ahead']:.4f}")
    chk('A v2.1 完成且放置判準通過', a['done'] and bool((a['place'] or {}).get('ok')), (a['stop'], a['place']))
    chk('A 閘起點、收近中與停止軌跡餘裕 ≥ 0', g['trigger']['margin'] >= 0 and (g['monitor_min'] or 0) >= 0 and (g['stop_traj']['min_margin'] if g['stop_traj']['min_margin'] is not None else 1.0) >= 0, g)
    res['v21'] = dict(summary=a, gate={k: v for k, v in g.items() if k != 'log'}, near_stops=near_stops(f, r))
    cb, fb, rb, st = run(C21.W1PlaceController, cut_at=lambda c_: None if c_.cc is None else c_.cc['t0'] + 0.5 * C2.T_TO, capture=True)
    cap = st['cap']
    dev = None
    if cap is not None:
        Q = C21.stop_traj(cap['q'], cap['s'], cap['pipe'], cap['u'], cb.cfg, 0, cap['phase'])
        H = [h for h in fb.hist if h['t'] > cap['t'] - 1e-9]
        n = min(len(Q) - 1, len(H) // C21.SAMPLE_EVERY)
        dtcp, dbase = 0.0, 0.0
        for k in range(1, n + 1):
            qa, qm = H[k * C21.SAMPLE_EVERY - 1]['q'], Q[k]          # 假世界 hist 第 j 筆＝第 j＋1 個物理步之後的狀態
            dtcp = max(dtcp, float(np.linalg.norm(T.T.K.fk(qa, 'link_tcp')[:3, 3] - T.T.K.fk(qm, 'link_tcp')[:3, 3])))
            dbase = max(dbase, float(np.hypot(*(np.asarray(qa[:2]) - qm[:2]))))
        # 涵蓋：實際停止期間（到 1.5 s 收尾）每個狀態到模型軌跡取樣點集合的最近距離，取最大（單向 Hausdorff）
        A_ = np.array([T.T.K.fk(np.r_[0.0, 0.0, 0.0, qm[3:]], 'link_tcp')[:3, 3] for qm in Q])          # 手臂停止路徑上的 TCP（底盤座標）
        cth, sth = np.cos(Q[:, 2])[:, None], np.sin(Q[:, 2])[:, None]
        Ptcp = np.stack([Q[:, 0:1] + cth * A_[:, 0] - sth * A_[:, 1], Q[:, 1:2] + sth * A_[:, 0] + cth * A_[:, 1], np.broadcast_to(A_[:, 2], (len(Q), len(A_)))], axis=-1).reshape(-1, 3)   # 交叉集合
        cov_t, cov_b = 0.0, 0.0
        for h in H:
            cov_t = max(cov_t, float(np.linalg.norm(Ptcp - T.T.K.fk(h['q'], 'link_tcp')[:3, 3], axis=1).min()))
            cov_b = max(cov_b, float(np.hypot(*(Q[:, :2] - np.asarray(h['q'][:2])).T).min()))
        d_act = float(np.hypot(*(np.asarray(H[-1]['q'][:2]) - cap['q'][:2])))
        d_mod = float(np.hypot(*(Q[-1, :2] - cap['q'][:2])))
        dev = dict(t_stop=cap['t'], n=n, tcp_mm=round(dtcp * 1e3, 2), base_mm=round(dbase * 1e3, 2), span_s=round(n * C21.SAMPLE_EVERY * 0.01, 2),
                   cover_tcp_mm=round(cov_t * 1e3, 2), cover_base_mm=round(cov_b * 1e3, 2), base_travel_actual_mm=round(d_act * 1e3, 1), base_travel_model_mm=round(d_mod * 1e3, 1))
    print(f"   停止軌跡模型 vs 假世界：{dev}；停止 {rb['stop']}")
    chk('B 收近中掃描中斷 ⇒ 受控停止；實際停止期間每個 TCP 到交叉集合（底盤停止位姿 × 手臂停止構型）、底盤到模型底盤路徑的最近距離 < 10 mm（涵蓋；逐點差另報）', rb['stop'] is not None and dev is not None and dev['cover_tcp_mm'] < 10.0 and dev['cover_base_mm'] < 10.0, (rb['stop'], dev))
    res['stop_model'] = dev
    # C：v3 vs v2
    c2, f2, r2, _ = run(C2.W1PlaceController)
    o2 = TC.summarize('v2 名目（對照）', c2, f2, r2)
    c3, f3, r3, _ = run(C3.W1PlaceController)
    o3 = TC.summarize('v3 名目', c3, f3, r3)
    ev3 = {}
    for e in r3['events']:
        ev3.setdefault(e['what'], e)
    pe3 = {e.get('to'): e['t'] for e in r3['events'] if e['what'] == 'place_phase'}
    ns2, ns3 = near_stops(f2, r2), near_stops(f3, r3)
    rt2, rt3 = reach_time(c2, f2, r2), reach_time(c3, f3, r3)
    rg = r3['compact_carry']['reach_gate']
    print(f"   v3：升高 {ev3.get('compact_raise_start', {}).get('t')}（{ev3.get('compact_raise_start', {}).get('why')}，距入口 {ev3.get('compact_raise_start', {}).get('rem')} m）→ {ev3.get('compact_raised', {}).get('t')}；"
          f"PLACE_APPROACH {pe3.get('PLACE_APPROACH')}；外伸 {ev3.get('compact_out_start', {}).get('t')}（閘 {rg['trigger']}）→ {ev3.get('compact_returned', {}).get('t')}；PARKED {pe3.get('PARKED')}；外伸中閘最小 {rg['monitor_min']}")
    print(f"   近停 v2 {ns2}／v3 {ns3}；DONE v2 {o2['t_done']}／v3 {o3['t_done']}；外伸（半徑 > 0.30 m）累計 v2 {rt2[0]} s／v3 {rt3[0]} s；升高後近身持瓶 v3 {rt3[1]} s；滑移 v2 {o2['slip'][0]:.2f}／v3 {o3['slip'][0]:.2f} mm")
    chk('C v3 完成且放置判準通過', o3['done'] and bool((o3['place'] or {}).get('ok')), (o3['stop'], o3['place']))
    chk('C 升高在 PLACE_APPROACH 前開始（early）、外伸在 PLACE_APPROACH 且閘餘裕 ≥ 0、外伸中閘 ≥ 0',
        ev3.get('compact_raise_start', {}).get('why') == 'early' and ev3['compact_raise_start']['t'] < pe3.get('PLACE_APPROACH', 0) and ev3.get('compact_out_start', {}).get('t', -1) >= pe3.get('PLACE_APPROACH', 1e9) - 1e-9
        and (rg['trigger'] or {}).get('margin', -1) >= 0 and (rg['monitor_min'] if rg['monitor_min'] is not None else 1.0) >= 0, (ev3.get('compact_raise_start'), ev3.get('compact_out_start'), rg['trigger']))
    chk('C 持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限、設定點速率峰 ≤ v_arm', all(v[0] >= TL.REJ * 1e3 for v in o3['held'].values()) and o3['self_min_mm'] > 0 and o3['forb'] == 0 and o3['finite']
        and o3['rate'] <= B.W_ARM['v_arm'] + 1e-6, (o3['held'], o3['self_min_mm'], o3['forb'], o3['rate']))
    w2 = sum(b - a_ for a_, b in ns2 if b > (pe3.get('PLACE_APPROACH') or 0) - 5)
    w3 = sum(b - a_ for a_, b in ns3 if b > (pe3.get('PLACE_APPROACH') or 0) - 5)
    chk('C 停車前近停時間 v3 < v2（比較用，不是通過門檻的放寬）', w3 < w2, (ns2, ns3))
    res.update(v2=dict(summary=o2, near_stops=ns2, reach_s=rt2[0]), v3=dict(summary=o3, near_stops=ns3, reach_s=rt3[0], raised_close_s=rt3[1], reach_gate={k: v for k, v in rg.items() if k != 'log'},
                                                                              events={k: ev3.get(k) for k in ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned')}))
    # D：v3 停止
    for nm, fn in (('升高中', lambda c_: None if c_.cc is None or c_.cc.get('t_raise') is None else c_.cc['t_raise'] + 1.0),
                   ('外伸中', lambda c_: None if c_.cc is None or c_.cc.get('t_out') is None else c_.cc['t_out'] + 1.9)):
        cd, fd, rd, sd = run(C3.W1PlaceController, cut_at=fn)
        od = TC.summarize(f'v3 {nm}掃描中斷', cd, fd, rd)
        t_low = next((e['t'] for e in rd['events'] if e['what'] == 'place_phase' and e.get('to') == 'LOWER'), 1e9)
        held, tilt, _, _ = TF.held_tables(fd, sd['t_cut'] or 0.0, t_low)
        se = od['stop_effect'] or {}
        print(f"      注入起 {sd['t_cut']}；停止 {od['stop'] and (round(od['stop']['t'], 2), od['stop']['why'])}；停止有效 {se.get('ok')}；注入後持瓶對桌 {held}；傾角 {tilt:.2f}°")
        chk(f'D v3 {nm}掃描中斷 ⇒ 受控停止且停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm',
            sd['t_cut'] is not None and (not od['done']) and (od['stop'] or {}).get('why') in ('scan_stale', 'shield_intervened') and se.get('ok') is True and all(v[0] >= TL.REJ * 1e3 for v in held.values())
            and od['self_min_mm'] > 0 and od['margin'] > 0 and od['forb'] == 0 and od['slip'][0] < 5.0, (sd['t_cut'], od['stop'], se, held))
        res[f'stop_{nm}'] = dict(summary=od, t_cut=sd['t_cut'], held_after=held, tilt_after=tilt)
    # E
    c3.judge.T_rel = np.array(T2.POSE['reference']['T_rel'])
    c3.cc = dict(stage='raised', t0=0.0, t_ret=None, t_raise=0.0, t_out=None, T_side=np.array(T2.POSE['reference']['T_side']))
    m_nan = c3._reach_gate(np.full(9, np.nan), 1.0)[0]
    chk('E 外伸閘：非有限 q ⇒ None', m_nan is None, m_nan)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_compactcarry3_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
