#!/usr/bin/env python3
"""前持候選在兩個移動物案例的完整包絡離線核對（Codex 20261101_193038；假受控對象＋候選 2 追蹤器，非物理）。
每案：完成或受控停止、命令有限、禁止接觸 0、自碰 > 0、切換後持瓶對兩桌（到 LOWER 前）≥ 5.5 mm；
包絡：全部連桿（含夾爪、手指）與持瓶（LIFT 起剛性附著，到 LOWER 前）對移動物圓柱（讀軌跡 DC.state，鉛直圓柱 z 0–高度）的最小三維距離；
全部連桿對第二桌（桌面＋桌腳）的最小距離；底盤中心到移動物表面。只報告並要求 > 0（不接觸）；不移障礙、不放寬距離。
    python3 evaluation/test_bottle_w1_frontcarry_dyn.py
"""
import json
import math
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
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_frontcarry_ctrl as FC                  # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402
import test_bottle_w1_frontcarry as TF                  # noqa: E402
import test_bottle_w1_liftb as TL                       # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(case):
    CA = DC.CASES[case]
    scene = TD.SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']

    def scan_fn(t, q):
        r = NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z)
        oc, _ = DC.state(t, CA)
        return np.minimum(r, DC.ray_cyl(q[:2], NC.scan_angles() + float(q[2]), oc, CA['radius'], NC.RANGE_MAX))
    o_cls, o_ml, o_tr = T.W1.W1PlaceController, TD.SC.make_lidar, NC.ScanObstacleTracker
    T.W1.W1PlaceController = FC.W1PlaceController
    TD.SC.make_lidar = lambda **kw: o_ml(scan_fn=scan_fn, **kw)
    NC.ScanObstacleTracker = KC.make2(o_tr)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, TD.SC.make_lidar, NC.ScanObstacleTracker = o_cls, o_ml, o_tr
    r, _ = T.run(c, f)
    return CA, c, f, r


def envelope(f, CA, t_lower):
    BP, rel = G.bottle_points(), None
    t2 = [(c_, sz) for n, c_, sz in TP.table2_boxes()]
    out = dict(link_dyn=(9.9, None, None), bottle_dyn=(9.9, None), link_t2=(9.9, None, None), base_dyn=(9.9, None))
    H = float(CA.get('height', 1.0))
    for h in f.hist[::5]:
        q, t = h['q'], float(h['t'])
        oc, _ = DC.state(t, CA)
        held = h['phase'] in ('LIFT', 'CARRY') and t <= t_lower
        for n, P in G.link_world(q, 0.0038 if h['phase'] in ('CLOSE', 'LIFT', 'CARRY') else G.Q_OPEN).items():
            dz = np.maximum(np.maximum(-P[:, 2], P[:, 2] - H), 0.0)
            d = float(np.hypot(np.maximum(np.hypot(P[:, 0] - oc[0], P[:, 1] - oc[1]) - CA['radius'], 0.0), dz).min())
            if d < out['link_dyn'][0]:
                out['link_dyn'] = (d, round(t, 2), n)
            for c_, sz in t2:
                d2 = float(G.box_dist(P, c_, sz).min())
                if d2 < out['link_t2'][0]:
                    out['link_t2'] = (d2, round(t, 2), n)
        bd = float(np.hypot(q[0] - oc[0], q[1] - oc[1]) - CA['radius'])
        if bd < out['base_dyn'][0]:
            out['base_dyn'] = (bd, round(t, 2))
        if h['phase'] in ('LIFT', 'CARRY'):
            Tt = T.T.K.fk(q, 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [T.T.BX, T.T.BY, T.T.TOP]
                rel = np.linalg.inv(Tt) @ T0
            if held:
                Tb = Tt @ rel
                Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
                d = float(np.maximum(np.hypot(Pw[:, 0] - oc[0], Pw[:, 1] - oc[1]) - CA['radius'], 0.0).min())
                if d < out['bottle_dyn'][0]:
                    out['bottle_dyn'] = (d, round(t, 2))
    return {k: ((round(v[0] * 1e3, 1),) + tuple(v[1:])) for k, v in out.items()}


def main():
    res = {}
    for case in ('w1_dyn_pass1', 'w1_dyn_overtake1'):
        CA, c, f, r = run(case)
        ev = {}
        for e in r['events']:
            ev.setdefault(e['what'], e['t'])
        pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
        t_lower = pe.get('LOWER', 1e9)
        mins, sm, ns, rate, marg = PCK.check(f)
        sw = (r.get('paired_strategy') or {}).get('switch')
        held, tilt, fwd, side = TF.held_tables(f, sw['t'] if sw else 0.0, t_lower)
        env = envelope(f, CA, t_lower)
        o = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), fc=[ev.get(k) for k in ('front_carry_start', 'front_carry_reached', 'front_carry_return', 'front_carry_returned')],
                 envelope=env, held_tables=held, tilt=tilt, fwd=fwd, self_min_mm=round(sm[0] * 1e3, 2), forb=max(v['peak'] for v in r['forbidden_pulses'].values()), finite=bool(T.finite_cmds(f)),
                 shield=r['nav_chain']['shield'], slip=r['slip_max_mm_deg'], u0_bind=(r['paired_strategy'].get('u0_rows') or {}).get('n_bind'))
        res[case] = o
        print(f"   [{case}] 完成 {o['done']} 阻點 {o['stop'] and (round(o['stop']['t'], 2), o['stop']['why'])} DONE {o['t_done']}；前持 {o['fc']}；Shield 介入 {o['shield']['active']}；第一拍列貼合 {o['u0_bind']}")
        print(f"      包絡 mm：連桿↔移動物 {env['link_dyn']}、持瓶↔移動物 {env['bottle_dyn']}、連桿↔第二桌 {env['link_t2']}、底盤中心↔移動物表面 {env['base_dyn']}；持瓶↔兩桌 {held}；傾角 {tilt:.2f}°；前向外伸 {fwd:.3f} m；自碰 {o['self_min_mm']}；滑移 {np.round(o['slip'], 2).tolist()}")
        ok_held = all(v[0] >= TL.REJ * 1e3 for v in held.values())
        chk(f'{case}：完成或受控停止、命令有限、禁止接觸 0、自碰 > 0、持瓶對兩桌 ≥ 5.5 mm', (o['done'] or o['stop'] is not None) and o['finite'] and o['forb'] == 0 and o['self_min_mm'] > 0 and ok_held, (o['stop'], o['forb'], held))
        chk(f'{case}：連桿與持瓶對移動物、連桿對第二桌都不接觸（> 0）', env['link_dyn'][0] > 0 and env['bottle_dyn'][0] > 0 and env['link_t2'][0] > 0, env)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_frontcarry_dyn_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
