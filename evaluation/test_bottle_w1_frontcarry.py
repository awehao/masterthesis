#!/usr/bin/env python3
"""前持候選（bottle_w1_frontcarry_ctrl）的最小離線核對（Codex 20261101_191812；假受控對象，非物理）。
0 幾何：以物理名目 bottle_w1_cont_nominal1 在 t_lift_ok 的側持關節，沿 φ∈[0, φ_front] 每 2° 只改 q1：自碰最小距離（閉爪手指）、q1 限位餘裕、瓶軸傾角（應維持鉛直）、
  TCP 高度與到第一軸半徑不變；瓶底高度對底盤頂（link_base 高度）的餘裕。
1 閉迴路名目（連續展臂基線 vs 前持候選，同假世界）：完成；前持啟動晚於抓持與離桌判定；到前持、轉回側持都完成且轉回完成早於 PARKED；放置判準；
  持瓶（LIFT 起剛性附著）對第一桌、第二桌（含桌面與桌腳）在切換後到 LOWER 開始前的最小距離 ≥ 5.5 mm；自碰 > 0（PCK 取樣，實測與設定點構型）；禁止接觸 0；命令有限；
  瓶軸傾角最大、瓶心在底盤座標的前向／側向外伸、過渡時長、滑移、DONE。
    python3 evaluation/test_bottle_w1_frontcarry.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_continuous_approach_ctrl as CA         # noqa: E402
import bottle_w1_frontcarry_ctrl as FC                  # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402

BAD = []
Q_CLOSED = 0.0038


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def case_geom():
    r = json.load(open(os.path.join(HERE, 'runs', 'bottle_w1_cont_nominal1', 's1_report.json')))
    Z = np.load(os.path.join(HERE, 'runs', 'bottle_w1_cont_nominal1', 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    k = int(np.argmin(np.abs(c['t'] - r['t_lift_ok'])))
    qa = np.array([c[f'q{j}'][k] for j in range(1, 7)])
    K = G.C.K
    q0 = np.r_[0.0, 0.0, 0.0, qa]
    Lb = K.fk(q0, 'link_base')
    T0 = K.fk(q0, 'link_tcp')
    p = np.linalg.inv(Lb) @ T0[:, 3]
    phi_f = -math.atan2(p[1], p[0])
    rows = []
    for phi in np.linspace(0.0, phi_f, int(abs(math.degrees(phi_f)) // 2) + 2):
        q = q0.copy()
        q[3] += phi
        Tt = K.fk(q, 'link_tcp')
        d, who = G.C.self_clear(q, Q_CLOSED)
        rows.append(dict(phi_deg=float(math.degrees(phi)), self_mm=float(d * 1e3), who=str(who), q1_margin=float(min(q[3] - LITE6_SAFE.lower[0], LITE6_SAFE.upper[0] - q[3])),
                         tcp_z=float(Tt[2, 3]), radius=float(np.hypot(*(np.linalg.inv(Lb) @ Tt[:, 3])[:2]))))
    worst = min(rows, key=lambda x: x['self_mm'])
    print(f"   幾何（cont_nominal1 t_lift_ok {r['t_lift_ok']} 的側持關節，只改 q1）：φ_front {math.degrees(phi_f):.2f}°；自碰最小 {worst['self_mm']:.1f} mm @ φ {worst['phi_deg']:.1f}°（{worst['who']}）；"
          f"q1 餘裕最小 {min(x['q1_margin'] for x in rows):.3f} rad；TCP 高度 {min(x['tcp_z'] for x in rows):.4f}–{max(x['tcp_z'] for x in rows):.4f} m；半徑 {min(x['radius'] for x in rows):.4f}–{max(x['radius'] for x in rows):.4f} m")
    chk('0 幾何：沿前持旋轉自碰 > 0、q1 在限位內、TCP 高度與半徑不變（只改 q1）', worst['self_mm'] > 0 and min(x['q1_margin'] for x in rows) > 0
        and max(x['tcp_z'] for x in rows) - min(x['tcp_z'] for x in rows) < 1e-9 and max(x['radius'] for x in rows) - min(x['radius'] for x in rows) < 1e-9, worst)
    return dict(phi_front_deg=math.degrees(phi_f), self_min_mm=worst['self_mm'], self_who=worst['who'], q1_margin_min=min(x['q1_margin'] for x in rows), link_base=Lb[:3, 3].tolist())


def run(cls):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    r, _ = T.run(c, f)
    return c, f, r


def held_tables(f, t0, t1=1e9):
    """持瓶（LIFT 起剛性附著）對第一桌與第二桌（桌面、桌腳）在 t0 ≤ t ≤ t1 的最小距離（t1＝LOWER 開始：之後是刻意下放接觸、鬆爪後瓶不再附著）；另回傳瓶軸傾角最大、瓶心底盤座標前向／側向外伸最大。"""
    boxes = [(n, c_, sz) for n, c_, sz in G.table_boxes()] + [('t2_' + n, c_, sz) for n, c_, sz in TP.table2_boxes()]
    BP, rel, best, tilt, fwd, side = G.bottle_points(), None, {}, 0.0, -9.0, -9.0
    for h in f.hist[::5]:
        if h['phase'] not in ('LIFT', 'CARRY'):
            continue
        Tt = T.T.K.fk(h['q'], 'link_tcp')
        if rel is None:
            T0 = np.eye(4)
            T0[:3, 3] = [T.T.BX, T.T.BY, T.T.TOP]
            rel = np.linalg.inv(Tt) @ T0
        Tb = Tt @ rel
        tilt = max(tilt, math.degrees(math.acos(min(1.0, abs(float(Tb[2, 2]))))))
        q = h['q']
        cth, sth = math.cos(q[2]), math.sin(q[2])
        dx, dy = Tb[0, 3] - q[0], Tb[1, 3] - q[1]
        fwd, side = max(fwd, cth * dx + sth * dy), max(side, abs(-sth * dx + cth * dy))
        if h['t'] < t0 - 1e-9 or h['t'] > t1 + 1e-9:
            continue
        Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
        for nm, c_, sz in boxes:
            d = float(G.box_dist(Pw, c_, sz).min())
            k = ('table2' if nm.startswith('t2_') else 'table1') + ('_top' if nm.endswith('table_top') else '_leg')
            if k not in best or d < best[k][0]:
                best[k] = (d, round(float(h['t']), 2), h['phase'])
    return {k: (round(v[0] * 1e3, 2), v[1], v[2]) for k, v in best.items()}, tilt, fwd, side


def summarize(nm, c, f, r):
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'], e['t'])
    pe = {p: t for p, t in ((e.get('to'), e['t']) for e in r['events'] if e['what'] == 'place_phase')}
    mins, sm, ns, rate, marg = PCK.check(f)
    sw = (r.get('paired_strategy') or {}).get('switch')
    held, tilt, fwd, side = held_tables(f, sw['t'] if sw else 0.0, pe.get('LOWER', 1e9))
    fc = r.get('front_carry') or {}
    o = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), t_grasp=r.get('t_grasp'), t_lift_ok=r.get('t_lift_ok'),
             fc_start=ev.get('front_carry_start'), fc_reached=ev.get('front_carry_reached'), fc_return=ev.get('front_carry_return'), fc_returned=ev.get('front_carry_returned'),
             parked=pe.get('PARKED'), place_metrics=r['place'].get('metrics'), held=held, tilt_max_deg=tilt, fwd_max=fwd, side_max=side, self_min_mm=round(sm[0] * 1e3, 2),
             self_where=sm[2:], forb=max(v['peak'] for v in r['forbidden_pulses'].values()), finite=bool(T.finite_cmds(f)), slip=r['slip_max_mm_deg'], rate=rate, margin=marg,
             phi_front_deg=fc.get('phi_front_deg'), solve_p95=float(np.percentile(c.solve_ms, 95)) if c.solve_ms else None)
    print(f"   [{nm}] 完成 {o['done']} 阻點 {o['stop']} DONE {o['t_done']}；抓持 {o['t_grasp']}、離桌判定 {o['t_lift_ok']}；前持 {o['fc_start']}→到位 {o['fc_reached']}；轉回 {o['fc_return']}→完成 {o['fc_returned']}；PARKED {o['parked']}")
    print(f"      持瓶對桌（切換後）{held}；瓶軸傾角最大 {tilt:.2f}°；瓶心外伸 前向 {fwd:.3f}／側向 {side:.3f} m；自碰 {o['self_min_mm']} mm {o['self_where']}；禁止接觸 {o['forb']}；滑移 {np.round(o['slip'], 2).tolist()}；"
          f"放置 {o['place_metrics'] and {k: o['place_metrics'][k] for k in ('ok', 'offset', 'tilt')}}；設定點速率峰 {rate:.3f} rad/s；關節餘裕 {marg:.3f} rad；求解 p95 {o['solve_p95']:.1f} ms")
    return o


def main():
    res = dict(geom=case_geom())
    a = summarize('連續展臂基線', *run(CA.W1PlaceController))
    b = summarize('前持候選', *run(FC.W1PlaceController))
    res.update(baseline=a, front=b)
    chk('1 候選完成且放置判準通過', b['done'] and bool((b['place_metrics'] or {}).get('ok')), (b['stop'], b['place_metrics']))
    chk('1 前持啟動不早於抓持與離桌判定', b['fc_start'] is not None and b['t_grasp'] is not None and b['t_lift_ok'] is not None and b['fc_start'] >= max(b['t_grasp'], b['t_lift_ok']) - 1e-9,
        (b['fc_start'], b['t_grasp'], b['t_lift_ok']))
    chk('1 到前持、轉回側持都完成，且轉回完成早於 PARKED', b['fc_reached'] is not None and b['fc_returned'] is not None and b['parked'] is not None and b['fc_returned'] <= b['parked'] + 1e-9,
        (b['fc_reached'], b['fc_returned'], b['parked']))
    chk('1 切換後持瓶對兩桌 ≥ 5.5 mm、自碰 > 0、禁止接觸 0、命令有限', all(v[0] >= TL.REJ * 1e3 for v in b['held'].values()) and b['self_min_mm'] > 0 and b['forb'] == 0 and b['finite'],
        (b['held'], b['self_min_mm'], b['forb']))
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_frontcarry_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
