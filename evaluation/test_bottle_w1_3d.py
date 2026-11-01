#!/usr/bin/env python3
"""W1 整合趟（視覺抓瓶＋規劃運送＋三維全身避障＋第二桌放置）的最小離線閉迴路核對（Codex 20261101_142546；假受控對象，非物理）。
障礙＝封存的懸空方塊（bottle_w1_box3d_case.BOX，掃描面之上，LiDAR 看不到；模擬真值進控制器）。追蹤器用候選 2（同物理趟）。
A 反事實（只離線）：SHROW 不給三維約束，方塊在場 ⇒ 全部連桿／持瓶對方塊的最小距離（預期 < 0，確認幾何衝突）。
B 候選：bottle_w1_shrow3d_ctrl（body_boxes＝[BOX]、body_links＝link3–5，既有機制與距離）⇒
  完成或受控停止、命令有限、無禁止接觸；全部連桿與持瓶對方塊最小距離 ≥ 0（列目標 d_body 0.05 m 只報告）；切換後持瓶↔桌 ≥ 5.5 mm；自碰 > 0；
  放置判準（假世界）；誰讓路（方塊 x 處底盤 y 與名目差、TCP 高度差）、三維列貼合時段、耗時。無有效解就停止，不放寬距離。
    python3 evaluation/test_bottle_w1_3d.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s0b_path_check as PCK                     # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_w1_box3d_case as BC                       # noqa: E402
import bottle_w1_shrow3d_ctrl as S3D                    # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BOX = (np.array(BC.BOX[0], float), np.array(BC.BOX[1], float))
LINKS = ('link3', 'link4', 'link5')
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def build(cls, body):
    import bottle_n2_path as NP
    import bottle_s1_cand as SC
    import test_bottle_place as TP
    plan = json.load(open(os.path.join(HERE, 'runs', 'bottle_n2_smoke2', 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    tg0, orig = T.T.T_G, NC.ScanObstacleTracker
    T.T.T_G = np.full((4, 4), np.nan)
    NC.ScanObstacleTracker = KC.make2(orig)                     # 候選 2 追蹤器（同物理趟）
    try:
        kw = dict(body_boxes=[BOX], body_links=LINKS) if body else {}
        c = SC.make_lidar(cls=cls, extra_boxes=[(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top'], carry_planner=TP.FakePlanner(), **kw)
    finally:
        T.T.T_G, NC.ScanObstacleTracker = tg0, orig
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])
    f = TP.FakePlace(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.T.D_TRUE
    return c, f


def box_dist(f):
    """全部連桿（含手指、夾爪）與持瓶（LIFT 起剛性附著，同 PCK）對方塊的最小距離與時刻；底盤圓盤（0.39 m）對方塊 xy 投影（方塊在掃描面之上，只報告）。"""
    BP, rel = G.bottle_points(), None
    per, bot = {}, (9.9, None)
    for h in f.hist[::5]:
        q = h['q']
        for n, P in G.link_world(q, 0.0038).items():
            d = float(G.box_dist(P, *BOX).min())
            if d < per.get(n, (9.9,))[0]:
                per[n] = (d, round(float(h['t']), 2))
        if h['phase'] in ('LIFT', 'CARRY'):
            Tt = T.T.K.fk(q, 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [T.T.BX, T.T.BY, T.T.TOP]
                rel = np.linalg.inv(Tt) @ T0
            Tb = Tt @ rel
            Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
            d = float(G.box_dist(Pw, *BOX).min())
            if d < bot[0]:
                bot = (d, round(float(h['t']), 2))
    worst = min(per.items(), key=lambda x: x[1][0])
    return dict(bottle_mm=round(bot[0] * 1e3, 1), bottle_t=bot[1], link_min=(worst[0], round(worst[1][0] * 1e3, 1), worst[1][1]),
                per_link_mm={k: round(v[0] * 1e3, 1) for k, v in sorted(per.items(), key=lambda x: x[1][0])[:6]})


def at_box(f):
    """底盤在方塊 x 處的 y、TCP 世界高度、底盤偏航（誰讓路）。"""
    H = [h for h in f.hist if h['phase'] == 'CARRY']
    if not H:
        return None
    k = int(np.argmin([abs(h['q'][0] + 0.116 - BOX[0][0]) for h in H]))   # 名目瓶心在底盤前方約 0.116 m（同紀錄）
    h = H[k]
    tcp = T.T.K.fk(h['q'], 'link_tcp')[:3, 3]
    return dict(t=round(float(h['t']), 2), base=[round(float(v), 3) for v in h['q'][:3]], tcp_z=round(float(tcp[2]), 3), tcp_y=round(float(tcp[1]), 3))


def run(cls, body):
    c, f = build(cls, body)
    r, _ = T.run(c, f)
    ev = {e['what']: e['t'] for e in r['events']}
    mins, sm, ns, rate, marg = PCK.check(f)
    sw = (r.get('paired_strategy') or {}).get('switch')
    held = TL.held_after(f, sw['t']) if sw else None
    pl = r['place']
    o = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), finite=bool(T.finite_cmds(f)), forb=max(v['peak'] for v in r['forbidden_pulses'].values()),
             slip=r['slip_max_mm_deg'], self_min_mm=round(sm[0] * 1e3, 2), held_after_switch=held, place_metrics=pl.get('metrics'), box=box_dist(f), at_box=at_box(f),
             body_rows_max=r.get('body_rows_max'), body_bind=r.get('body_bind'), bind_span=(r.get('paired_strategy') or {}).get('body_bind_span'),
             solve_p95=float(np.percentile(c.solve_ms, 95)) if c.solve_ms else None, solve_max=float(max(c.solve_ms)) if c.solve_ms else None,
             n_fallback=r.get('n_fallback'), shield=r['nav_chain']['shield'])
    return o


def main():
    res = {}
    a = run(SR.W1PlaceController, False)
    res['A_no_constraint'] = a
    print(f"   [A 反事實：無三維約束] 完成 {a['done']} DONE {a['t_done']}；對方塊 瓶 {a['box']['bottle_mm']} mm @ {a['box']['bottle_t']}、連桿最近 {a['box']['link_min']}；方塊處 {a['at_box']}")
    chk('A 反事實：無約束時持瓶或連桿穿入方塊（確認幾何衝突）', a['box']['bottle_mm'] < 0 or a['box']['link_min'][1] < 0, a['box'])
    b = run(S3D.W1PlaceController, True)
    res['B_shrow3d'] = b
    print(f"   [B SHROW-3D] 完成 {b['done']} 阻點 {b['stop']} DONE {b['t_done']}；對方塊 瓶 {b['box']['bottle_mm']} mm @ {b['box']['bottle_t']}、連桿最近 {b['box']['link_min']}、各連桿 {b['box']['per_link_mm']}")
    print(f"      方塊處 {b['at_box']}（A：{a['at_box']}）；三維列峰 {b['body_rows_max']}、貼合 {b['body_bind']}、時段 {b['bind_span']}；切換後持瓶↔桌 {b['held_after_switch']}；自碰 {b['self_min_mm']} mm；"
          f"滑移 {np.round(b['slip'], 2).tolist()}；放置 {b['place_metrics'] and {k: b['place_metrics'][k] for k in ('ok', 'offset', 'tilt')}}；求解 p95／max {b['solve_p95']:.1f}／{b['solve_max']:.1f} ms；fallback {b['n_fallback']}；Shield {b['shield']['active']}")
    ha = b['held_after_switch']
    chk('B 候選：完成或受控停止、命令有限、無禁止接觸、自碰 > 0、切換後持瓶↔桌 ≥ 5.5 mm', (b['done'] or b['stop'] is not None) and b['finite'] and b['forb'] == 0 and b['self_min_mm'] > 0
        and (ha is None or all(v[0] >= TL.REJ * 1e3 for v in ha.values())), (b['stop'], b['forb'], b['self_min_mm'], ha))
    chk('B 候選：全部連桿與持瓶對方塊不穿入（≥ 0 mm）', b['box']['bottle_mm'] >= 0 and b['box']['link_min'][1] >= 0, b['box'])
    chk('B 候選：任務完成且放置判準通過（假世界）', b['done'] and bool((b['place_metrics'] or {}).get('ok')), (b['stop'], b['place_metrics']))
    json.dump(dict(box=[BOX[0].tolist(), BOX[1].tolist()], links=LINKS, **res), open(os.path.join(HERE, 'results', 'bottle', 'w1_3d_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
