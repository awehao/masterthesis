#!/usr/bin/env python3
"""bottle_s1_2 的兩項前置（Codex 20261031 收臂提速）：完整軌跡幾何（收臂初態、行進包絡、展臂、自碰、桌瓶、障礙、線上淨空）與名目 P2 一次離線回歸（參考、設定點、命令連續；抓持與出口判定）。
另各一次導航與展臂中的停止。非物理：假受控對象。
    python3 evaluation/bottle_s1_tuck_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s0b_path_check as PC                      # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

RB = 0.30


def envelope(h, q_f):
    """全部連桿與夾爪點雲對底盤中心的最大水平半徑（底盤座標）。"""
    W = G.link_world(np.r_[0.0, 0.0, 0.0, h['q'][3:]], q_f)
    P = np.vstack(list(W.values()))
    return float(np.hypot(P[:, 0], P[:, 1]).max())


def main():
    c = SC.make(path=SC.PATH_P2, **SC.FAST)
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE                           # 實跑由收臂靜止出發（含重力偏移），不從舊備妥姿態
    f.run(70.0, lambda t, s: (s.c._trigger_stop('sim_limit') if t >= 65.0 else None) or {})
    r = c.report()
    ev = [(round(e['t'], 2), e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'unfold_start', 'blend_start', 'exit_enter', 'mission_done')]
    print('結果', r['success'], '離開碼', S1.exit_code(r), 'stop', r['stop'], '\n事件', ev)
    t_u = next(e['t'] for e in r['events'] if e['what'] == 'unfold_start')
    t_b = next(e['t'] for e in r['events'] if e['what'] == 'blend_start')
    H = f.hist
    nav = [h for h in H[::20] if h['t'] < t_u]
    r_nav = max(envelope(h, G.Q_OPEN) for h in nav)
    r_unf = max(envelope(h, G.Q_OPEN) for h in H[::20] if t_u <= h['t'] < t_b)
    print(f'行進包絡（實測關節、全部連桿與夾爪）最大半徑 {r_nav:.3f} m（底盤 {RB}）；展臂段 {r_unf:.3f} m；未伸出底盤 {r_nav <= RB}（0.39 m 圓盤 {r_nav <= 0.39}）')
    mins, sm, ns, rate, marg = PC.check(f)
    for k in sorted(mins):
        if 'LIFT' not in k and not k.startswith('sp:'):
            print(f'  {k:<40} {mins[k][0] * 1e3:7.1f} mm @ {mins[k][1]} s {mins[k][2]}')
    print(f'自碰最小 {sm[0] * 1e3:.1f} mm @ {sm[1]} s {sm[2]} {sm[3]}（{ns} 點）；設定點速率峰 {rate:.3f} rad/s；關節餘裕 {marg:.3f} rad')
    print('障礙', SC.obstacle_geom(f, SC.OBST['c']), '線上瓶淨空', r['clear_bottle'])
    # 連續性：相鄰物理步的設定點、控制週期命令、參考
    s_ = np.array([h['s'] for h in H])
    u_ = np.array([h['u_last'] for h in H])[::5]
    sr = np.array([h['v_b'] for h in H])[::5]
    tcp = np.array([h['tcp'] for h in H])
    print(f"連續性：設定點單步 Δ 峰 {np.abs(np.diff(s_, axis=0)).max():.5f} rad；底盤命令週期 Δ 峰 {np.abs(np.diff(u_[:, :3], axis=0)).max():.4f}；手臂命令週期 Δ 峰 {np.abs(np.diff(u_[:, 3:], axis=0)).max():.4f}；v_b 週期 Δ 峰 {np.abs(np.diff(sr)).max():.4f}；TCP 單步 Δ 峰 {np.linalg.norm(np.diff(tcp, axis=0), axis=1).max() * 1e3:.2f} mm")
    q = np.array([h['q'][:3] for h in H])
    t = np.array([h['t'] for h in H])
    sp = np.convolve(np.hypot(np.gradient(q[:, 0], t), np.gradient(q[:, 1], t)), np.ones(5) / 5, mode='same')
    tc, tl = r['t_close'], r['t_lift_start']
    mk = (t >= tc) & (t <= tl)
    ms = np.array(r['solve_ms_all'])
    print(f"速度：峰 {sp.max():.4f}、夾持 {sp[mk].min():.4f}–{sp[mk].max():.4f} m/s；總時長 {t[-1]:.1f} s、滑移 {r['slip_max_mm_deg']}；末 ({q[-1, 0]:.3f}, {q[-1, 1]:.3f})；ms p95 {np.percentile(ms, 95):.1f} max {ms.max():.1f} >50 {r['n_over_50ms']}；{r['qp_counts']}；管外 {r['n_gate']}")
    d_c = np.hypot(q[:, 0] - SC.OBST['c'][0], q[:, 1] - SC.OBST['c'][1])
    print(f'障礙餘裕：r_eff {(d_c.min() - SC.R_OBST - SC.M_STATIC) * 1e3:.1f} mm、0.39 m 圓盤模型 {(d_c.min() - SC.R_OBST - SC.R_DISC) * 1e3:.1f} mm')
    ok = [r['success'], sm[0] >= 0.005, r['n_fallback'] == 0]                # K2_READY 是 Howard 指定，包絡只報告不設門檻
    for name, ts in (('導航中停止', 6.0), ('展臂中停止', t_u + 1.5)):
        c2 = SC.make(path=SC.PATH_P2, **SC.FAST)
        f2 = T.Fake(c2, x0=SC.X0)
        f2.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
        f2.run(ts + 4.0, lambda t_, s_2, ts_=ts: dict(forbidden={T.REQ[0]: 0.2, T.REQ[1]: 0.0}) if t_ >= ts_ else {})
        rr = c2.report()
        print(name, rr['stop'] and rr['stop']['why'], '停止有效', rr['stop_effect'], '離開碼', S1.exit_code(rr))
        ok += [S1.exit_code(rr) == 3]
    print('前置', '全過' if all(ok) else f'未過 {ok}')
    json.dump(dict(success=r['success'], events=ev, r_nav=r_nav, r_unfold=r_unf, self_mm=sm[0] * 1e3, joint_margin=marg, clear=r['clear_bottle'], ok=[bool(v) for v in ok]), open(os.path.join(HERE, 'results', 'bottle', 's1_tuck_check.json'), 'w'), ensure_ascii=False, indent=1)
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
