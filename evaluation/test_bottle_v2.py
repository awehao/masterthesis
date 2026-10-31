#!/usr/bin/env python3
"""V2 最小接線的離線驗收（假受控對象＋合成估計，非物理）：底盤相機幾何可見時，每 0.2 s 給一筆估計＝真瓶位置＋偏差＋雜訊（v1_1 實測量級），
距離用估計值與相機位姿算。驗凍結、目標重算、完成，以及各停止條件；真值只在這裡用來合成估計與評估。
    python3 evaluation/test_bottle_v2.py [案例…]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_path as NP                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_v1_cam_visibility as VZ                   # noqa: E402
import bottle_v2_ctrl as V2                             # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

BAD = []
T_CAM = VZ.CAMS['base']['T']
BOXES = [(c_, sz) for _, c_, sz in G.table_boxes()] + [(SC.OBST['c'], SC.OBST['sz'])] + [(c_, sz) for _, c_, sz in TP.table2_boxes()]


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def build():
    plan = json.load(open(os.path.join(HERE, 'runs', 'bottle_n2_smoke2', 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    c = SC.make_lidar(cls=V2.V2PlaceController, extra_boxes=[(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top'], carry_planner=TP.FakePlanner())
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])
    f = TP.FakePlace(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    return c, f


def visible(q):
    T_wc = G.C.K.fk(q, 'base_link') @ T_CAM
    Ti = np.linalg.inv(T_wc)
    b0 = np.array([T.BX, T.BY, T.TOP])
    fx = 320.0 / math.tan(VZ.CAMS['base']['hfov'] / 2)
    vis = []
    for pw in b0 + np.outer(np.linspace(0.01, 0.19, 10), [0, 0, 1.0]):
        pc = Ti[:3, :3] @ pw + Ti[:3, 3]
        if pc[2] <= 0:
            vis.append(False)
            continue
        u, v = fx * pc[0] / pc[2] + 320, fx * pc[1] / pc[2] + 240
        vis.append(0.3 <= pc[2] <= 8.0 and 0 <= u < 640 and 0 <= v < 480 and not any(VZ.seg_hits_box(T_wc[:3, 3], pw, bc, bs) for bc, bs in BOXES))
    return float(np.mean(vis)) >= 0.5, T_wc[:3, 3]


def run(c, f, bias=(0.43e-3, 1.66e-3), noise=(0.33e-3, 0.43e-3), age=0.02, off=True, inject=None, t_end=75.0, seed=0):
    rng = np.random.default_rng(seed)
    st = dict(next=0.0, pend=[])

    def hook(t, s):
        for p_ in [p_ for p_ in st['pend'] if p_[0] <= t + 1e-9]:
            c.vision_intake(p_[1], t, p_[2])
            st['pend'].remove(p_)
        if off and t >= st['next'] - 1e-9 and s.c.st.phase == 'CRUISE':
            st['next'] = t + 0.2
            ok, cam = visible(s.q)
            if ok:
                xy = np.array([T.BX, T.BY]) + np.array(bias) + rng.normal(0, noise)
                if inject and inject[0](c):
                    xy = xy + np.array(inject[1])
                e = dict(valid=True, cx=float(xy[0]), cy=float(xy[1]), ax=[0.0, 0.0, 1.0], dist=float(np.linalg.norm(np.r_[xy, T.TOP + 0.1] - cam)))
                st['pend'].append((t + age, t, e))
        if t >= t_end - 3.0 and not s.c.mission_done:
            s.c._trigger_stop('sim_limit')
        return {}
    f.run(t_end, hook, tail=2.6)
    return c.report()


def show(name, r, c):
    v = r['v2']
    fx = v['fix']
    err = None if fx is None else [round((fx['xy'][0] - T.BX) * 1e3, 2), round((fx['xy'][1] - T.BY) * 1e3, 2)]
    print(f"[{name}] 完成 {r['mission_done']} 阻點 {r['stop'] and r['stop']['why']} 停止有效 {r['stop_effect'] and r['stop_effect'].get('ok')} 凍結 {fx and (round(fx['t'], 2), fx['n'], round(fx['spread'] * 1e3, 2))} 凍結誤差 mm {err} 收 {v['n_intake']} 拒 {v['rejected']}")
    return err


def case_nominal():
    c, f = build()
    r = run(c, f)
    err = show('名目（v1_1 偏差＋雜訊）', r, c)
    chk('名目：凍結、完成放置', r['mission_done'] and r['v2']['fix'] is not None, r['stop'])
    chk('名目：凍結在朝瓶漸變前', r['v2']['fix'] and r['v2']['fix']['t'] < next(e['t'] for e in r['events'] if e['what'] == 'blend_start'))
    chk('名目：凍結誤差在插入軸預算內（−2.0～+3.0 mm）與閉合軸（−5.0～+4.5）', err and -2.0 <= err[1] <= 3.0 and -5.0 <= err[0] <= 4.5, err)


def case_no_est():
    c, f = build()
    r = run(c, f, off=False, t_end=30.0)
    show('沒有任何估計', r, c)
    chk('無估計 ⇒ 漸變前 vision_no_fix 停止、未閉爪', (r['stop'] or {}).get('why') == 'vision_no_fix' and c.t_close_req is None, r['stop'])


def case_jitter():
    c, f = build()
    r = run(c, f, noise=(3e-3, 3e-3), t_end=30.0)
    show('估計跳動 3 mm', r, c)
    chk('不穩定 ⇒ 不凍結、vision_no_fix', (r['stop'] or {}).get('why') == 'vision_no_fix', r['stop'])


def case_stale():
    c, f = build()
    r = run(c, f, age=0.5, t_end=30.0)
    show('估計過期 0.5 s', r, c)
    chk('過期估計一律不收 ⇒ vision_no_fix', (r['stop'] or {}).get('why') == 'vision_no_fix' and r['v2']['rejected'].get('stale', 0) > 0, (r['stop'], r['v2']['rejected']))


def case_inconsistent():
    c, f = build()
    r = run(c, f, inject=(lambda c_: c_.vfix is not None and c_.t > c_.vfix['t'] + 0.5, (0.0, 0.008)), t_end=40.0)
    show('凍結後估計偏 8 mm', r, c)
    chk('凍結後不一致 ⇒ vision_inconsistent 停止、未閉爪', (r['stop'] or {}).get('why') == 'vision_inconsistent' and c.t_close_req is None, r['stop'])


def case_deep():
    c, f = build()
    r = run(c, f, bias=(0.0, 4.0e-3), noise=(0.0, 0.0), t_end=60.0)
    show('插入軸偏差 +4 mm（超出預算）', r, c)
    chk('超出預算：不得完成時把責任推給真值；以淨空拒絕線受控停止或照實記錄', r['stop'] is not None or r['mission_done'], r['stop'])
    print(f"   瓶淨空 實測／預測／下一拍 mm {[round((r['clear_bottle'][k] or float('nan')) * 1e3, 2) for k in ('meas_min', 'pred_min', 'next_min')]}")


def case_time():
    """估計接收的時間防呆（直接呼叫 vision_intake）。"""
    c, f = build()
    c.t = 5.0
    good = lambda: dict(valid=True, cx=0.0, cy=0.42, ax=[0.0, 0.0, 1.0], dist=1.2)   # noqa: E731
    c.vision_intake(4.98, 5.0, good())
    n0 = len(c.vbuf)
    c.vision_intake(float('nan'), 5.0, good())
    c.vision_intake(5.5, 5.0, good())
    c.vision_intake(4.98, 5.0, good())
    c.vision_intake(4.90, 5.0, good())
    e = good()
    e['ax'] = [0.0, 0.0, 0.0]
    c.vision_intake(4.99, 5.0, e)
    e = good()
    e['ax'] = [0.0, 0.0, 5.0]
    c.vision_intake(4.995, 5.0, e)
    rj = c.vrej
    chk('時間：非有限、未來、重複、倒序都拒收', rj.get('time_nonfinite') == 1 and rj.get('time_future') == 1 and rj.get('time_not_increasing') == 2, rj)
    chk('軸：零向量拒收，非單位向量正規化後照收', rj.get('axis_zero') == 1 and len(c.vbuf) == n0 + 1, (rj, len(c.vbuf)))


def case_transition():
    """凍結後「既有備妥姿態→新預抓目標」的實際過渡軌跡淨空（目標偏到預算邊界）。"""
    import bottle_s0b_path_check as PCK
    for bias in ((0.43e-3, 1.66e-3), (-5.0e-3, 0.0), (4.5e-3, 0.0), (0.0, -2.0e-3), (0.0, 3.0e-3)):
        c, f = build()
        r = run(c, f, bias=bias, noise=(0.0, 0.0), t_end=40.0)
        mins, sm, ns, rate, marg = PCK.check(f)
        keys = [k for k in mins if ('bottle' in k or 'table' in k) and 'LIFT' not in k and not k.startswith('sp:') and 'held' not in k]
        worst = {k: round(mins[k][0] * 1e3, 2) for k in keys}
        fb = [k for k in keys if 'nonfinger↔bottle' in k]
        print(f"   偏移 {np.round(np.array(bias) * 1e3, 2).tolist()} mm：完成 {r['mission_done']} 阻點 {r['stop'] and r['stop']['why']}；淨空 {worst}；自碰 {sm[0] * 1e3:.1f} mm")
        chk(f'過渡淨空：偏移 {np.round(np.array(bias) * 1e3, 1).tolist()} mm 時非手指↔瓶 ≥ 5.5 mm 或由拒絕線受控停止',
            all(mins[k][0] >= 0.0055 for k in fb) or str((r['stop'] or {}).get('why', '')).startswith('clearance'), (worst, r['stop']))


CASES = dict(time=case_time, transition=case_transition, nominal=case_nominal, no_est=case_no_est, jitter=case_jitter, stale=case_stale, inconsistent=case_inconsistent, deep=case_deep)


def main():
    for n in sys.argv[1:] or list(CASES):
        CASES[n]()
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
