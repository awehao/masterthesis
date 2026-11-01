#!/usr/bin/env python3
"""緊湊持物候選（bottle_w1_compactcarry_ctrl）的最小離線核對（Howard 修正；假受控對象，非物理）。
0 轉回路徑幾何：由設計終點（w1_compact_pose_v2）沿「先升高再外伸」路徑 IK 延續回側持（20 步，每步收斂、關節餘裕 ≥ 0.05、自碰 ≥ 8.39 mm、持瓶↔連桿 ≥ 5.5 mm、車體頂淨空、不跳分支）。
1 名目閉迴路（連續展臂基線 vs 緊湊候選）：完成且放置判準通過；收近晚於抓持、離桌判定且在清出第一桌之後；轉回完成早於 PARKED；
  持瓶（LIFT 起剛性附著）對兩桌（到 LOWER 前）≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限；
  包絡（瓶身＋夾爪包圍盒）相對底盤中心的前向／側向最大投影與最大水平半徑、瓶底高度：過渡期與緊湊穩態分開報告。
2 停止路徑：收近開始後 0.5×T_tr 起掃描不發布 ⇒ 既有 scan_stale 受控停止；停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm。
    python3 evaluation/test_bottle_w1_compactcarry.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compactcarry_ctrl as CC                # noqa: E402
import bottle_w1_continuous_approach_ctrl as CA         # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_frontcarry as TF                  # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def case_return_path():
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))
    T_rel, T_side = np.array(d['reference']['T_rel']), np.array(d['reference']['T_side'])
    GC = CP.grip_corners(np.array(d['reference']['q_side']))
    q = np.array(d['selected']['q'])
    pa, pb = np.array(d['selected']['tcp_base']), T_side[:3, 3]
    worst = dict(self=9.9, bottle_link=9.9, deck=9.9, margin=9.9, jump=0.0)
    for k in range(1, 21):
        s = k / 20
        p = [pa[0], pa[1], pa[2] + (pb[2] - pa[2]) * (s / 0.5)] if s <= 0.5 else [pa[0] + (pb[0] - pa[0]) * (s - 0.5) / 0.5, pa[1] + (pb[1] - pa[1]) * (s - 0.5) / 0.5, pb[2]]
        Tt = T_side.copy()
        Tt[:3, 3] = p
        ok, w, qn, pe, re, marg = RS.ik(np.zeros(3), Tt, [q], iters=60)
        if not ok:
            chk('0 轉回路徑（先升後伸）IK 延續可達', False, dict(step=k, pe=pe, re=re, marg=marg))
            return None
        ev = CP.evaluate(qn, T_rel, GC)
        worst = dict(self=min(worst['self'], ev['self']), bottle_link=min(worst['bottle_link'], ev['bottle_link']), deck=min(worst['deck'], ev['deck_margin']),
                     margin=min(worst['margin'], marg), jump=max(worst['jump'], float(np.abs(qn - q).max())))
        q = qn
    end_err = float(np.abs(q - np.array(d['reference']['q_side'])).max())
    print(f"   轉回路徑：自碰最小 {worst['self'] * 1e3:.1f} mm、持瓶↔連桿 {worst['bottle_link'] * 1e3:.1f} mm、車體頂餘裕 {worst['deck']:.3f}、關節餘裕 {worst['margin']:.3f}、單步最大關節變化 {worst['jump']:.3f}；終點對側持關節差 {end_err:.3f} rad")
    chk('0 轉回路徑（先升後伸）IK 延續可達、約束全滿足、不跳分支', worst['self'] >= CP.SELF_MIN and worst['bottle_link'] >= CP.BOT_LINK_MIN and worst['deck'] >= 0 and worst['margin'] >= 0.05 and worst['jump'] <= 0.5, worst)
    return dict(worst=worst, end_joint_err=end_err)


def run(cls, cut=None):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    st = dict(t_cut=None, last=None)
    if cut is not None:
        fn0 = c.scan_fn

        def scan_cut(t, q):
            if st['t_cut'] is None and getattr(c, 'cc', None) is not None:
                st['t_cut'] = c.cc['t0'] + cut * c.cc['T_tr']
            if st['t_cut'] is not None and t >= st['t_cut'] - 1e-9:
                return None
            st['last'] = float(t)
            return fn0(t, q)
        c.scan_fn = scan_cut
    r, _ = T.run(c, f)
    return c, f, r, st


def envelope_stats(c, f, r):
    """瓶身＋夾爪包圍盒相對底盤中心：前向／側向投影最大、水平半徑最大、瓶底最低；分期：收近過渡、緊湊穩態、轉回過渡、其餘持瓶段。"""
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))
    GC = CP.grip_corners(np.array(d['reference']['q_side']))
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    t_low = pe.get('LOWER', 1e9)
    seg = lambda t: ('to_compact' if ev.get('compact_start', 1e9) <= t < ev.get('compact_reached', 1e9) else   # noqa: E731
                     'compact' if ev.get('compact_reached', 1e9) <= t < ev.get('compact_return', 1e9) else
                     'to_side' if ev.get('compact_return', 1e9) <= t < ev.get('compact_returned', 1e9) else 'other_held')
    rel, out = None, {}
    for h in f.hist[::5]:
        if h['phase'] not in ('LIFT', 'CARRY') or h['t'] > t_low:
            continue
        q = h['q']
        Tt = T.T.K.fk(q, 'link_tcp')
        if rel is None:
            T0 = np.eye(4)
            T0[:3, 3] = [T.T.BX, T.T.BY, T.T.TOP]
            rel = np.linalg.inv(Tt) @ T0
        Pb, Pg = CP.envelope(Tt, rel, GC)
        P = np.vstack([Pb, Pg])
        cth, sth = math.cos(q[2]), math.sin(q[2])
        dx, dy = P[:, 0] - q[0], P[:, 1] - q[1]
        fw, sd = cth * dx + sth * dy, -sth * dx + cth * dy
        k = seg(h['t'])
        o = out.setdefault(k, dict(fwd=-9.0, side=0.0, rad=0.0, bottom=9.9, n=0))
        o.update(fwd=max(o['fwd'], float(fw.max())), side=max(o['side'], float(np.abs(sd).max())), rad=max(o['rad'], float(np.hypot(dx, dy).max())), bottom=min(o['bottom'], float(Pb[:, 2].min())), n=o['n'] + 1)
    return {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}


def summarize(nm, c, f, r):
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    mins, sm, ns, rate, marg = PCK.check(f)
    sw = (r.get('paired_strategy') or {}).get('switch')
    held, tilt, fwd, side = TF.held_tables(f, sw['t'] if sw else 0.0, pe.get('LOWER', 1e9))
    o = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), t_grasp=r.get('t_grasp'), t_lift_ok=r.get('t_lift_ok'),
             cc=[ev.get(k) for k in ('compact_start', 'compact_reached', 'compact_return', 'compact_returned')], parked=pe.get('PARKED'), place=r['place'].get('metrics'),
             held=held, tilt=tilt, self_min_mm=round(sm[0] * 1e3, 2), forb=max(v['peak'] for v in r['forbidden_pulses'].values()), finite=bool(T.finite_cmds(f)), slip=r['slip_max_mm_deg'],
             rate=rate, margin=marg, env=envelope_stats(c, f, r), stop_effect=r.get('stop_effect'))
    print(f"   [{nm}] 完成 {o['done']} 阻點 {o['stop'] and (round(o['stop']['t'], 2), o['stop']['why'])} DONE {o['t_done']}；抓持 {o['t_grasp']}、離桌判定 {o['t_lift_ok']}；收近 {o['cc'][0]}→{o['cc'][1]}；轉回 {o['cc'][2]}→{o['cc'][3]}；PARKED {o['parked']}")
    print(f"      包絡（前向／側向／半徑最大、瓶底最低）{o['env']}；持瓶對桌 {held}；傾角 {tilt:.2f}°；自碰 {o['self_min_mm']}；禁止接觸 {o['forb']}；滑移 {np.round(o['slip'], 2).tolist()}；設定點速率峰 {rate:.3f}；關節餘裕 {marg:.3f}")
    return o


def main():
    res = dict(return_path=case_return_path())
    b0 = summarize('連續展臂基線', *run(CA.W1PlaceController)[:3])
    c, f, r, _ = run(CC.W1PlaceController)
    b = summarize('緊湊候選', c, f, r)
    res.update(baseline=b0, compact=b)
    chk('1 候選完成且放置判準通過', b['done'] and bool((b['place'] or {}).get('ok')), (b['stop'], b['place']))
    chk('1 收近不早於抓持與離桌判定；轉回完成早於 PARKED', b['cc'][0] is not None and b['cc'][0] >= max(b['t_grasp'], b['t_lift_ok']) - 1e-9 and b['cc'][3] is not None and b['parked'] is not None and b['cc'][3] <= b['parked'] + 1e-9, (b['cc'], b['parked']))
    chk('1 持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限', all(v[0] >= TL.REJ * 1e3 for v in b['held'].values()) and b['self_min_mm'] > 0 and b['forb'] == 0 and b['finite'], (b['held'], b['self_min_mm'], b['forb']))
    cs = b['env'].get('compact', {})
    chk('1 緊湊穩態包絡水平半徑小於側持穩態', cs.get('rad', 9.9) < b0['env'].get('other_held', {}).get('rad', 0.0), (cs, b0['env']))
    c2, f2, r2, st = run(CC.W1PlaceController, cut=0.5)
    s2 = summarize('收近過渡中掃描中斷', c2, f2, r2)
    held2, tilt2, _, _ = TF.held_tables(f2, st['t_cut'] or 0.0)
    s2.update(t_cut=st['t_cut'], last_scan=st['last'], held_after_cut=held2, tilt_after_cut=tilt2)
    res['stop_path'] = s2
    se = s2['stop_effect'] or {}
    print(f"      中斷起 {st['t_cut']}（最後掃描 {st['last']}）；停止 {s2['stop']}；停止有效 {se.get('ok')} @ {se.get('t')}；注入後持瓶對桌 {held2}；傾角 {tilt2:.2f}°")
    chk('2 收近過渡中掃描中斷 ⇒ 受控停止且停止有效、注入後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、關節餘裕 > 0、禁止接觸 0、滑移 < 5 mm',
        (not s2['done']) and (s2['stop'] or {}).get('why') in ('scan_stale', 'shield_intervened') and se.get('ok') is True and all(v[0] >= TL.REJ * 1e3 for v in held2.values())
        and s2['self_min_mm'] > 0 and s2['margin'] > 0 and s2['forb'] == 0 and s2['slip'][0] < 5.0, (s2['stop'], se, held2))
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_compactcarry_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
