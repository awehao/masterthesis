#!/usr/bin/env python3
"""W1 最小離線驗收（Codex 20261101_074314；假受控對象＋合成估計，非物理）：底盤估計同 test_bottle_v2（v1_2 半像素量級），
手腕估計＝真瓶位置＋偏差＋雜訊（偏差用 bottle_w1c_2 實測量級 (0.15, 0.54) mm），在手腕相機到瓶 ≤ 0.35 m、閉爪前每 0.2 s 一筆；
兩路都經 V2Link（位姿互核閘）進控制器，目標建構為 NaN。距離用估計值與相機位姿算。真值只用來合成估計與評估。
    python3 evaluation/test_bottle_w1.py [案例…]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_v1_cam_visibility as VZ                   # noqa: E402
import bottle_v2_link as VL                             # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_s0b as T                             # noqa: E402
import test_bottle_v2 as TV2                            # noqa: E402

BAD = []
T_WR = VZ.CAMS['wrist']['T']
W_BIAS, W_NOISE = (0.15e-3, 0.54e-3), (0.05e-3, 0.05e-3)


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def build():
    import json
    import bottle_n2_path as NP
    import bottle_s1_cand as SC
    plan = json.load(open(os.path.join(HERE, 'runs', 'bottle_n2_smoke2', 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    tg0 = T.T_G
    T.T_G = np.full((4, 4), np.nan)                      # 不先放真值
    try:
        c = SC.make_lidar(cls=W1.W1PlaceController, extra_boxes=[(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top'], carry_planner=TP.FakePlanner())
    finally:
        T.T_G = tg0
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])
    f = TP.FakePlace(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    return c, f


def msg(t, xy, cam, ok=True):
    nan = float('nan')
    if not ok:
        return [t, 0.0] + [nan] * 14
    dist = float(np.linalg.norm(np.r_[xy, T.TOP + 0.1] - cam))
    return [t, 1.0, float(xy[0]), float(xy[1]), T.TOP, nan, 0.0, 0.0, 1.0, 1000.0, 2.4e-4, 50.0, 0.0, 0.0, 0.0, dist]


def run(c, f, w_bias=W_BIAS, w_noise=W_NOISE, w_on=True, w_until=None, t_end=75.0, seed=0, base_bias=TV2.BIAS_R2, base_noise=TV2.NOISE_R2):
    """w_until：只在手腕可見後的前 w_until 秒送手腕估計（之後完全不送）。"""
    rng = np.random.default_rng(seed)
    capb, capw = TV2.FakeCap(), TV2.FakeCap()
    lb, lw = VL.V2Link(capb, c), VL.V2Link(capw, c, sink=c.wrist_intake)
    st = dict(nb=0.0, nw=0.0, pend=[], w0=None)

    def hook(t, s):
        for p_ in [p_ for p_ in st['pend'] if p_[0] <= t + 1e-9]:
            p_[1].ests[p_[2]] = (p_[0], p_[3])
            st['pend'].remove(p_)
        lb.tick(t)
        lw.tick(t)
        ph = s.c.st.phase
        if t >= st['nb'] - 1e-9 and ph == 'CRUISE':
            st['nb'] = t + 0.2
            ok, cam = TV2.visible(s.q)
            if ok:
                xy = np.array([T.BX, T.BY]) + np.array(base_bias) + rng.normal(0, base_noise)
                k = round(t, 6)
                capb.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                st['pend'].append((t + 0.02, capb, k, msg(t, xy, cam)))
        if w_on and t >= st['nw'] - 1e-9 and ph in ('CRUISE', 'PREGRASP', 'INSERT'):
            st['nw'] = t + 0.2
            cam = (G.C.K.fk(s.q, 'link_eef') @ T_WR)[:3, 3]
            if float(np.linalg.norm(np.array([T.BX, T.BY, T.TOP + 0.1]) - cam)) <= 0.35:
                st['w0'] = t if st['w0'] is None else st['w0']
                if w_until is None or t - st['w0'] <= w_until:
                    xy = np.array([T.BX, T.BY]) + np.array(w_bias) + rng.normal(0, w_noise)
                    k = round(t, 6)
                    capw.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                    st['pend'].append((t + 0.05, capw, k, msg(t, xy, cam)))
        if t >= t_end - 3.0 and not s.c.mission_done:
            s.c._trigger_stop('sim_limit')
        return {}
    f.run(t_end, hook, tail=2.6)
    return c.report(), lw


def show(name, r):
    w, v = r['w1'], r['v2']
    gerr = None if w['goal'] is None else [round((w['goal'][0] - T.BX) * 1e3, 3), round((w['goal'][1] - T.BY) * 1e3, 3)]
    print(f"[{name}] 完成 {r['mission_done']} 阻點 {r['stop'] and r['stop']['why']} 停止有效 {r['stop_effect'] and r['stop_effect'].get('ok')} 底盤凍結 {v['fix'] and round(v['fix']['t'], 2)} "
          f"手腕量測組 {w['n_groups']} 修正 {w['corr_mm'] and round(w['corr_mm'], 3)} mm 目標誤差 {gerr} 鎖定 {w['locked']} 收 {w['n_intake']} 拒 {w['rejected']}")
    return gerr


def rate_ok(c):
    H = np.array(c.w_hist)
    if len(H) < 2:
        return True, 0.0
    d = np.linalg.norm(np.diff(H[:, 1:], axis=0), axis=1)
    dt = np.diff(H[:, 0])
    r = float(np.max(d / np.maximum(dt, 1e-9)))
    return r <= c.w1['rate'] + 1e-9, r


def finite_cmds(f):
    return all(np.isfinite(h['u']).all() and np.isfinite(h['s']).all() for h in f.hist)


def ev_t(r, what):
    return next((e['t'] for e in r['events'] if e.get('what') == what), None)


def case_nominal():
    c, f = build()
    r, lw = run(c, f)
    gerr = show('名目（手腕偏差 w1c_2 量級）', r)
    ok, rmax = rate_ok(c)
    t_cl = ev_t(r, 'close_request')
    chk('名目：完成放置、命令有限', r['mission_done'] and finite_cmds(f), r['stop'])
    chk('名目：形成手腕量測、鎖定、閉爪前已到達鎖定目標', r['w1']['n_groups'] >= 1 and r['w1']['locked'] and t_cl is not None and np.allclose(r['w1']['cur'], r['w1']['goal'], atol=1e-9))
    chk('名目：手腕目標誤差 ≈ 手腕偏差（≤ 0.3 mm 差）', gerr and abs(gerr[0] - 0.15) <= 0.3 and abs(gerr[1] - 0.54) <= 0.3, gerr)
    chk('名目：抓取目標＝由施加值算出的位姿', np.allclose(c.task.T_grasp, C0.grasp_target(c.w_cur[0], c.w_cur[1], c.top_z + 0.10), atol=1e-12))
    chk('名目：修正限速 ≤ 5 mm/s', ok, rmax)


def case_jump():
    """底盤無偏差、手腕偏 +4 mm（閉合軸）：修正 4 mm 需 0.8 s；查限速、未來參考依限速預測、閉爪在到達之後。"""
    c, f = build()
    r, lw = run(c, f, w_bias=(4.0e-3, 0.0), base_bias=(0.0, 0.0), base_noise=(0.0, 0.0), w_noise=(0.0, 0.0))
    show('手腕修正 4 mm（閉合軸）', r)
    ok, rmax = rate_ok(c)
    print(f"   最大修正速率 {rmax * 1e3:.3f} mm/s；施加歷程 {len(c.w_hist)} 拍")
    chk('跳 4 mm：限速 ≤ 5 mm/s', ok, rmax)
    chk('跳 4 mm：完成或受控停止且未在未到達時閉爪', (r['mission_done'] or (r['stop'] and r['stop_effect'] and r['stop_effect'].get('ok'))) and
        (ev_t(r, 'close_request') is None or np.allclose(r['w1']['cur'], r['w1']['goal'], atol=1e-9)), r['stop'])
    # 未來參考：設定施加值與量測目標，_w_pred 依限速逐拍前進
    c2, _ = build()
    c2.vfix = dict(t=0.0, xy=np.array([T.BX, T.BY]), spread=0.0, n=5, margin=1.0)
    c2.w_cur, c2.w_goal, c2.w_t, c2.t = np.array([T.BX, T.BY]), np.array([T.BX + 0.004, T.BY]), 0.0, 0.0
    P = np.array([c2._w_pred(k * W1.B.DT) for k in range(W1.B.N + 1)])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    chk('未來參考：逐拍預測的修正每拍 ≤ 限速×DT，且最終到達量測目標', bool((step <= c2.w1['rate'] * W1.B.DT + 1e-12).all() and np.allclose(P[-1], c2.w_goal)), (step.max(), P[-1] - c2.w_goal))


def case_no_wrist():
    c, f = build()
    r, _ = run(c, f, w_on=False, t_end=40.0)
    show('手腕全程無估計', r)
    chk('無手腕估計 ⇒ wrist_no_fix、未閉爪、命令有限（不退回底盤目標）', (r['stop'] or {}).get('why') == 'wrist_no_fix' and c.t_close_req is None and finite_cmds(f), (r['stop'], c.t_close_req))


def case_inconsistent():
    c, f = build()
    r, _ = run(c, f, w_bias=(0.0, 8.0e-3), t_end=40.0)
    show('手腕與底盤差 8 mm', r)
    chk('差 > 5 mm ⇒ wrist_inconsistent、未閉爪', (r['stop'] or {}).get('why') == 'wrist_inconsistent' and c.t_close_req is None, r['stop'])


def case_stale():
    """鎖定前缺測：手腕只在可見後前 0.8 s 送（先形成一組），之後不送 ⇒ wrist_stale。"""
    c, f = build()
    r, _ = run(c, f, w_until=0.8, t_end=40.0)
    show('鎖定前缺測', r)
    chk('鎖定前缺測：先形成手腕量測組', r['w1']['n_groups'] >= 1, r['w1']['n_groups'])
    chk('鎖定前缺測 > 0.5 s ⇒ wrist_stale、未閉爪', (r['stop'] or {}).get('why') == 'wrist_stale' and c.t_close_req is None and not r['w1']['locked'], r['stop'])


def case_expire():
    """鎖定後期限（直接設狀態）：最後有效影格距今 2.6 s、尚未閉爪 ⇒ wrist_fix_expired；2.4 s 不停。"""
    for age, expect in ((2.4, None), (2.6, 'wrist_fix_expired')):
        c, _ = build()
        c.t = 30.0
        c.vfix = dict(t=10.0, xy=np.array([T.BX, T.BY]), spread=0.0, n=5, margin=1.0)
        c.w_goal = c.w_cur = np.array([T.BX, T.BY])
        c.w_locked, c.w_last_valid, c.t_close_req = True, 30.0 - age, None
        c.st.phase = 'INSERT'
        c.st.t_phase = 28.0
        c._w1_checks()
        got = c.judge.stop and c.judge.stop['why']
        chk(f'鎖定後最近有效影格 {age} s、未閉爪 ⇒ {expect}', got == expect, got)


def case_time():
    c, _ = build()
    c.t = 5.0
    c.vfix = dict(t=1.0, xy=np.array([T.BX, T.BY]), spread=0.0, n=5, margin=1.0)
    good = lambda: dict(valid=True, cx=T.BX, cy=T.BY, ax=[0.0, 0.0, 1.0], dist=0.2)   # noqa: E731
    c.wrist_intake(4.98, 5.0, good())
    n0 = len(c.wbuf)
    c.wrist_intake(float('nan'), 5.0, good())
    c.wrist_intake(5.5, 5.0, good())
    c.wrist_intake(4.98, 5.0, good())
    c.wrist_intake(4.99, 5.4, good())
    e = good()
    e['dist'] = 1.2
    c.wrist_intake(4.995, 5.0, e)
    rj = c.wrej
    chk('手腕接收：非有限、未來、重複拒收；年齡 > 0.3 s 拒收；遠距（> 0.30 m）拒收', rj.get('time_nonfinite') == 1 and rj.get('time_future') == 1 and rj.get('time_not_increasing') == 1
        and rj.get('stale') == 1 and rj.get('distance') == 1 and len(c.wbuf) == n0, rj)


CASES = dict(time=case_time, expire=case_expire, nominal=case_nominal, jump=case_jump, no_wrist=case_no_wrist, inconsistent=case_inconsistent, stale=case_stale)


def main():
    for n in sys.argv[1:] or list(CASES):
        CASES[n]()
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
