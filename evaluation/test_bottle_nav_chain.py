#!/usr/bin/env python3
"""S1 感測保護鏈的針對性核對（Codex 20261031_225849）：真值不進控制器、掃描過期、Shield 無旁路、CBF 失明時 Shield 仍擋、名目 P2 一次。假受控對象，非物理。
    python3 evaluation/test_bottle_nav_chain.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

BAD, N = [], 0


def chk(name, c, info=''):
    global N
    N += 1
    print(('  ok ' if c else 'FAIL ') + name, info if not c or info else '')
    if not c:
        BAD.append(name)


def fake(c, t_end, fault=None):
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    return f.run(t_end, fault or (lambda t, s: (s.c._trigger_stop('sim_limit') if t >= t_end - 3.0 else None) or {}))


def clear_disc(f):
    q = np.array([h['q'][:3] for h in f.hist])
    bx = [b for b in SC.scene_boxes() if b[0][2] < 0.5]                # 底盤高度只會碰方塊與桌腳，桌面在底盤之上
    d = [float(min(np.hypot(max(abs(x - c[0]) - sz[0] / 2, 0.0), max(abs(y - c[1]) - sz[1] / 2, 0.0)) for c, sz in bx)) for x, y in q[:, :2]]
    return min(d) - 0.30


def main():
    # 名目 P2：掃描鏈全程
    c = SC.make_lidar(log_cbf=True)
    f = fake(c, 45.0)
    r = c.report()
    n = r['nav_chain']
    q = np.array([h['q'][:3] for h in f.hist])
    t = np.array([h['t'] for h in f.hist])
    sp = np.convolve(np.hypot(np.gradient(q[:, 0], t), np.gradient(q[:, 1], t)), np.ones(5) / 5, mode='same')
    mk = (t >= (r['t_close'] or 1e9)) & (t <= (r['t_lift_start'] or -1))
    d_c = np.hypot(q[:, 0] - SC.OBST['c'][0], q[:, 1] - SC.OBST['c'][1])
    L = [e for e in c.cbf_log if e['cbf_active'] > 0 and e['du'] >= 1e-3]
    ph = {}
    for e in L:
        ph[e['phase']] = ph.get(e['phase'], 0) + 1
    ms = np.array(r['solve_ms_all'])
    print('名目', r['success'], S1.exit_code(r), r['stop'], [(round(e['t'], 1), e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'unfold_start', 'blend_start', 'exit_enter', 'mission_done')])
    print('  掃描鏈', {k: n[k] for k in ('n_scan', 'n_dyn_max', 'n_static_max')}, n['shield'], n['shield_first'], n['shield_last'])
    print(f"  CBF 改命令 {len(L)}／{len(c.cbf_log)} {ph}；真值：對方塊表面 {(d_c.min() - 0.2) * 1e3:.0f} mm 級、底盤圓盤 0.30 對環境淨空 {clear_disc(f) * 1e3:.1f} mm；速度峰 {sp.max():.3f}、夾持 {sp[mk].min() if mk.any() else 0:.3f}–{sp[mk].max() if mk.any() else 0:.3f}；滑移 {r['slip_max_mm_deg']}；瓶 {r['clear_bottle']['meas_min']}；ms p95 {np.percentile(ms, 95):.1f} max {ms.max():.1f} >50 {r['n_over_50ms']}；{r['qp_counts']}；總 {t[-1]:.1f} s；末 ({q[-1, 0]:.3f}, {q[-1, 1]:.3f})")
    chk('C1 名目完整任務成功', r['success'] and S1.exit_code(r) == 0, str(r['stop']))
    chk('C1 標明資料來源', r['nav_source'] == 'lidar_tracker_to_cbf' and r['shield'] is True)
    chk('C2 Shield 無旁路（每物理步都過）', n['shield']['n'] == len(f.hist), f"{n['shield']['n']} {len(f.hist)}")

    # 真值不進控制器：掃描看不到障礙 ⇒ 控制器的障礙清單為空
    c = SC.make_lidar(boxes=[])
    fake(c, 8.0)
    chk('C3 掃描無物 ⇒ 障礙清單空（真值未洩漏）', c.chain.obstacles(8.0) == [] and c.chain.n_static_max == 0, str(c.chain.n_static_max))

    # 掃描過期：不繞過，走停止路徑
    c = SC.make_lidar(scan_fn=lambda t, q, B=SC.scene_boxes(): None if t >= 5.0 else NC.ray_boxes(q[:2], float(q[2]), B, SC.G.LIDAR_Z))
    f = fake(c, 12.0)
    chk('C4 掃描中斷 ⇒ scan_stale 停止', (c.judge.stop or {}).get('why') == 'scan_stale', str(c.judge.stop))
    chk('C4 不報成功、停止有效', c.report()['success'] is False and (c.stop_ok or {}).get('ok') is True, str(c.stop_ok))
    after = [h for h in f.hist if h['t'] >= 5.5]
    chk('C4 Shield 過期限速 ≤0.05 m/s', max(float(np.hypot(*h['u'][:2])) for h in after) <= 0.05 + 1e-9, str(max(float(np.hypot(*h['u'][:2])) for h in after)))
    chk('C4 Shield 記到過期', c.chain.report()['shield']['stale'] > 0)

    # CBF 失明（追蹤結果清空）、方塊在直線路上：Shield 單獨擋住，不報成功
    c = SC.make_lidar(path=(-3.4, -3.3, 0.0), obs=lambda t: [])
    f = fake(c, 18.0)
    dmin = min(math.hypot(max(abs(h['q'][0] + 2.0) - 0.2, 0.0), max(abs(h['q'][1]) - 0.2, 0.0)) for h in f.hist)
    chk('C5 CBF 失明 ⇒ Shield 介入', c.chain.report()['shield']['active'] > 0, str(c.chain.report()['shield']))
    chk('C5 底盤未撞（0.30 圓盤淨空 >0）', dmin - 0.30 > 0.0, str(dmin))
    chk('C5 不報成功', c.report()['success'] is False, str(c.report()['stop']))
    print(f'  C5 底盤中心到方塊面最小 {dmin:.3f} m；x 末 {f.hist[-1]["x"]:.3f}')

    # 停止路徑也經 Shield
    c = SC.make_lidar()
    f = fake(c, 9.0, lambda t, s: dict(forbidden={T.REQ[0]: 0.2, T.REQ[1]: 0.0}) if t >= 6.0 else {})
    chk('C6 停止路徑經 Shield、停止有效', c.chain.report()['shield']['n'] == len(f.hist) and (c.stop_ok or {}).get('ok') is True, str(c.stop_ok))
    # 慢伺服（掃描鏈）：5 mm 拒絕線仍要攔得住
    for al in (0.6, 0.4):
        c = SC.make_lidar()
        f = T.Fake(c, x0=SC.X0, alpha_scale=al)
        f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
        f.run(45.0, lambda t, s: (s.c._trigger_stop('sim_limit') if t >= 42.0 else None) or {})
        tb, tg = T.bottle_T(), c.judge.t_grasp or 1e9
        dm = min(c._bottle_clear(h['q'], tb) for h in f.hist if 18.0 <= h['t'] <= tg)
        print(f"  α×{al}: stop {(c.judge.stop or {}).get('why')} 成功 {c.report()['success']} 瓶實測最小 {dm * 1e3:.2f} mm")
        chk(f'C7 α×{al} 實測 ≥5 mm', dm >= S1.CLEAR_REJECT, str(dm))
    print('全過' if not BAD else f'未過 {BAD}', N)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
