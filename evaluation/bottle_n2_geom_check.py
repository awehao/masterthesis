#!/usr/bin/env python3
"""bottle_n2_1 前置（Codex 20261101）：用 Isaac 冒煙實際規劃出的路徑（plan.json 完整點），假受控對象跑 S1Planned，補查新路線的
行進／展臂包絡、自碰、桌瓶淨空、障礙餘裕、線上瓶淨空；判準同 bottle_s1_tuck_check（自碰 ≥5 mm、成功、fallback 0），跨既有拒絕線只回報、不改路線或門檻。
掃描鏈用同程序替身（bottle_nav_chain）＋離線射線；非物理。
    python3 evaluation/bottle_n2_geom_check.py <有 plan.json 的 RUN_ID>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_ctrl as N2C                            # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
import bottle_s0b_path_check as PC                      # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_s1_tuck_check as TC                       # noqa: E402
import test_bottle_s0b as T                             # noqa: E402


def main():
    run = sys.argv[1]
    plan = json.load(open(os.path.join(HERE, 'runs', run, 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    c = SC.make_lidar(cls=N2C.S1Planned)
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    f.run(70.0, lambda t, s: (s.c._trigger_stop('sim_limit') if t >= 65.0 else None) or {})
    r = c.report()
    ev = [(round(e['t'], 2), e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'unfold_start', 'blend_start', 'exit_enter', 'mission_done', 'stop')]
    print('路徑', run, f'L {pp.L:.3f} m、y 最低 {pp.P[:, 1].min():.3f}；結果', r['success'], '離開碼', S1.exit_code(r), 'stop', r['stop'], '\n事件', ev)
    H = f.hist
    t_u = next((e['t'] for e in r['events'] if e['what'] == 'unfold_start'), None)
    t_b = next((e['t'] for e in r['events'] if e['what'] == 'blend_start'), None)
    r_nav = max(TC.envelope(h, G.Q_OPEN) for h in H[::20] if t_u is None or h['t'] < t_u)
    r_unf = max((TC.envelope(h, G.Q_OPEN) for h in H[::20] if t_u is not None and t_b is not None and t_u <= h['t'] < t_b), default=float('nan'))
    q_u = next(h['q'][:3] for h in H if t_u is not None and h['t'] >= t_u)
    print(f'包絡：行進 {r_nav:.3f} m、展臂段 {r_unf:.3f} m；展臂開始時底盤 ({q_u[0]:.3f}, {q_u[1]:.3f})，偏航 {np.degrees(q_u[2]):.2f}°')
    mins, sm, ns, rate, marg = PC.check(f)
    for k in sorted(mins):
        if 'LIFT' not in k and not k.startswith('sp:'):
            print(f'  {k:<40} {mins[k][0] * 1e3:7.1f} mm @ {mins[k][1]} s {mins[k][2]}')
    print(f'自碰最小 {sm[0] * 1e3:.1f} mm @ {sm[1]} s {sm[2]} {sm[3]}（{ns} 點）；設定點速率峰 {rate:.3f} rad/s；關節餘裕 {marg:.3f} rad')
    og = SC.obstacle_geom(f, SC.OBST['c'])
    q = np.array([h['q'][:3] for h in H])
    d_c = np.hypot(q[:, 0] - SC.OBST['c'][0], q[:, 1] - SC.OBST['c'][1])
    print('障礙', og, f"中心距最小 {d_c.min():.3f} m、對 r_eff 餘裕 {(d_c.min() - SC.R_OBST - SC.M_STATIC) * 1e3:.1f} mm；線上瓶淨空", r['clear_bottle'])
    print('Shield', r['nav_chain']['shield'], '求解', r['timing']['solve'], 'fallback', r['n_fallback'], 'σ 落後／領先峰', r['planned_path']['lag_max'], r['planned_path']['lead_max'])
    cb = r['clear_bottle']
    ok = dict(success=bool(r['success']), self_ge_5mm=bool(sm[0] >= 0.005), fallback0=r['n_fallback'] == 0,
              bottle_clear_ge_5p5mm=bool(min(cb['meas_min'] or 9, cb['pred_min'] or 9, cb['next_min'] or 9) >= 0.0055))
    print('前置', ok, '全過' if all(ok.values()) else '未過')
    json.dump(dict(run=run, L=pp.L, events=ev, r_nav=r_nav, r_unfold=r_unf, self_mm=sm[0] * 1e3, self_at=[sm[1], sm[2], sm[3]], joint_margin=marg, obstacle=og,
                   mins={k: [round(v[0] * 1e3, 1), v[1], v[2]] for k, v in mins.items() if 'LIFT' not in k and not k.startswith('sp:')}, clear=cb, ok=ok),
              open(os.path.join(HERE, 'results', 'bottle', 'n2_geom_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0 if all(ok.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
