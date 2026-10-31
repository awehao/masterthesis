#!/usr/bin/env python3
"""第二桌放置第一包（Codex 20261101_034611）：假受控對象＋同程序掃描鏈替身，驗流程、判定與停止路徑。
合成支撐（瓶底碰到第二桌就由桌面承接、鬆爪後留在原位）只能驗流程，不能證明物理放置成功。
導航段用 Isaac 冒煙實際規劃出的路徑（bottle_n2_smoke2/plan.json）；持瓶段用假規劃器（起點到入口的直線，取樣 0.05 m）。
    python3 evaluation/test_bottle_place.py [案例…]
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
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

DX, DY = PC.PLACE['target'][0] - G.BOTTLE['x'], PC.PLACE['lane_y']


def table2_boxes():
    return [(n, c + np.array([DX, DY, 0.0]), sz) for n, c, sz in G.table_boxes()]


def empty_grid():
    return NP.Grid(np.zeros(170 * 120 * 2), 0.05, -5.0, -3.0, 170 * 2, 120, 0.0)


class FakePlanner:
    def __init__(self, ok=True, grid=None, monitor=False):
        self.ok, self.grid, self.monitor, self.req = ok, grid, monitor, None

    def request(self, start, goal, t):
        self.req = (np.array(start[:2], float), np.array(goal, float), t)

    def poll(self):
        if self.req is None:
            return None
        if not self.ok:
            return dict(ok=False, why='no_valid_path')
        a, b, _ = self.req
        n = max(int(np.linalg.norm(b - a) / 0.05), 2)
        P = a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]
        return dict(ok=True, P=P, frame='map', grid=self.grid if self.grid is not None else empty_grid(), info=dict(fake=True),
                    monitor=(lambda cp: NP.PathMonitor(cp)) if self.monitor else None)


class FakePlace(T.Fake):
    """T.Fake 加第二桌支撐：持瓶時瓶隨 TCP；瓶底低於第二桌面且在桌面範圍內 ⇒ 桌面承接（瓶底夾在桌面高度）、給法向力；
    手指張到 0.6 全開以上 ⇒ 解除夾持、瓶留在原位。mode 改變假世界：low_table（桌面比控制器以為的低 30 mm）、side（只有水平力）、sticky（張開後手指仍有力）。"""

    def __init__(self, ctrl, mode='nominal', **k):
        super().__init__(ctrl, **k)
        self.mode, self.released, self.f_place = mode, False, np.zeros(3)
        self.top2 = PC.PLACE['top'] - (0.03 if mode == 'low_table' else 0.0)

    def meas(self, **ov):
        c = self.c
        T_tcp = T.K.fk(self.q, 'link_tcp')
        closed = c.f_cmd <= 0.0040
        near = np.linalg.norm(T_tcp[:3, 3] - np.array([T.BX, T.BY, T.HG])) < 0.03 or self.grasped
        if closed and near and not self.grasped:
            self.grasped, self.Trel = True, np.linalg.inv(T_tcp) @ self.Tb
        if self.grasped and not self.released and c.f_cmd >= 0.6 * C0.FINGER_OPEN and c.t_release is not None:
            self.released = True
        self.f_place = np.zeros(3)
        if self.grasped and not self.released and c.st.phase in ('LIFT', 'CARRY'):
            cand = T_tcp @ self.Trel
            cx, cy = PC.PLACE['c']
            sx, sy = PC.PLACE['size']
            over = abs(cand[0, 3] - cx) <= sx / 2 and abs(cand[1, 3] - cy) <= sy / 2
            pen = self.top2 - PC.bottle_bottom_z(cand, T.R_B)
            if self.mode == 'early' and c.pph == 'PLACE_APPROACH':      # 假世界：持瓶接近時瓶身擦到第二桌面
                self.f_place = np.array([0.0, 0.3, 0.2])
            if over and pen >= 0.0:
                cand[2, 3] += pen
                fz = T.MASS * 9.81 + 2000.0 * pen
                k_ = int(round(self.t / 0.01))
                if self.mode == 'share':                                # 夾爪仍握持：桌面只承約 1/3 重量（bottle_p1_2 實測）
                    fz = T.MASS * 9.81 / 3.0
                elif self.mode == 'flicker':                            # 逐步交替 0 與 2/3 mg（平均 1/3 mg）
                    fz = 2.0 * T.MASS * 9.81 / 3.0 if k_ % 2 else 0.0
                elif self.mode == 'dropout':                            # 每 30 步掉力 10 步；其餘 mg/2，使窗口平均仍高於 0.25 mg，單獨驗連續掉力
                    fz = 0.0 if k_ % 30 < 10 else T.MASS * 9.81 / 2.0
                elif self.mode == 'low':                                # 桌面只承 0.15 mg：低於首次接觸門檻
                    fz = 0.15 * T.MASS * 9.81
                self.f_place = np.array([0.0, fz, 0.0]) if self.mode == 'side' else np.array([0.0, 0.0, -fz if self.mode == 'reverse' else fz])
                if self.mode == 'side':
                    cand[2, 3] -= pen
            self.Tb = cand
        if self.released:
            self.f_place = np.array([0.0, 0.0, T.MASS * 9.81])
        f = (self.f_hold, self.f_hold) if (closed and near and not self.released) or (self.released and self.mode == 'sticky') else (0.0, 0.0)
        if self.released and self.mode != 'sticky':
            f = (0.0, 0.0)
        gap = self.Tb[2, 3] - T.TOP
        m = dict(f_fingers=f, t_contact=self.t, T_bottle=self.Tb.copy(), t_bottle=self.t, f_table=(T.MASS * 9.81 if gap <= 1e-6 and not self.grasped else 0.0),
                 forbidden={k: 0.0 for k in T.REQ}, penetration=0.0, f_place=self.f_place.copy(), q_fingers=(c.f_cmd, c.f_cmd))
        m.update(ov)
        return m


def build(planner=None, mode='nominal'):
    plan = json.load(open(os.path.join(HERE, 'runs', 'bottle_n2_smoke2', 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    c = SC.make_lidar(cls=PC.PlaceController, extra_boxes=[(c_, sz) for n, c_, sz in table2_boxes() if n != 'table_top'], carry_planner=planner or FakePlanner())
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])                   # 舊出口區不用
    f = FakePlace(c, mode=mode, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    return c, f


def run(c, f, t_end=75.0, hook=None):
    def fault(t, s):
        if hook:
            hook(t, s)
        if t >= t_end - 3.0 and not s.c.mission_done:
            s.c._trigger_stop('sim_limit')
        return {}
    f.run(t_end, fault, tail=2.0)                        # 停穩判定要停止後 1.5 s 的資料
    return c.report()


def summary(name, r, c, f):
    ev = [(round(e['t'], 2), e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'place_phase', 'carry_plan_request', 'contact_first', 'mission_done', 'stop')]
    q = np.array([h['q'][:3] for h in f.hist])
    print(f"[{name}] 完成 {r['mission_done']} 判定成功 {r['success']} 阻點 {r['stop'] and r['stop']['why']} 停止有效 {r['stop_effect'] and r['stop_effect']['ok']}")
    print(f"   事件 {ev}")
    pl = r['place']
    print(f"   停車 {pl['park'] and np.round(pl['park'], 3).tolist()} 抬升 {pl['h_lift'] and round(pl['h_lift'], 4)} 凍結下降 {pl['d_freeze'] and round(pl['d_freeze'], 4)} 承接 {pl['carry'] and {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pl['carry'].items() if k != 'plan'}}")
    print(f"   指標 {pl['metrics'] and {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pl['metrics'].items()}}；滑移 {[round(v, 2) for v in r['slip_max_mm_deg']]}；求解 {r['timing']['solve']}；fallback {r['n_fallback']}；底盤末 {q[-1].round(3).tolist()}")


BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def case_nominal():
    c, f = build()
    r = run(c, f)
    summary('名目', r, c, f)
    seq = [e['to'] for e in r['events'] if e['what'] == 'place_phase']
    chk('名目：放置完成', r['mission_done'] is True and r['stop'] is None, r['stop'])
    chk('名目：相位依序', seq == list(PC.PPH), seq)
    chk('名目：支撐先於鬆爪', r['place']['events']['RELEASE']['t'] > next(e['t'] for e in r['events'] if e['what'] == 'contact_first'))
    ta = r['place']['events']['CARRY_NAV']['t']
    U = np.array([h['u'] for h in f.hist])[::5]
    tt = np.array([h['t'] for h in f.hist])[::5]
    dU = np.abs(np.diff(U[:, :2], axis=0)).max(axis=1)
    near = (tt[1:] >= ta - 0.5) & (tt[1:] <= ta + 0.5)
    car = (tt[1:] >= ta) & (tt[1:] <= r['place']['events']['PLACE_APPROACH']['t'])
    print(f'   底盤命令每週期跳變：承接 ±0.5 s 內最大 {dU[near].max():.4f}、持瓶導航段最大 {dU[car].max():.4f} m/s')
    chk('名目：承接前後命令跳變不大於持瓶段其他時候', dU[near].max() <= dU[car].max() + 1e-9, (dU[near].max(), dU[car].max()))
    chk('名目：停車後底盤命令為 0', max(float(np.abs(h['u']).max()) for h in f.hist if h['t'] > r['place']['events']['LOWER']['t'] + 0.2) < 1e-6)
    return r, c, f


def case_plan_fail():
    c, f = build(planner=FakePlanner(ok=False))
    r = run(c, f, 40.0)
    summary('持瓶規劃失敗', r, c, f)
    chk('規劃失敗 ⇒ 受控停止、不退回直線', (r['stop'] or {}).get('why') == 'carry_plan_invalid' and r['place']['pph'] is None and (r['stop_effect'] or {}).get('ok'), r['stop'])


def case_blocked_grid():
    g = empty_grid()
    a = g.a
    ix, iy = int((1.0 + 5.0) / 0.05), int((-0.55 + 3.0) / 0.05)
    a[iy - 3:iy + 3, ix - 3:ix + 3] = 100
    c, f = build(planner=FakePlanner(grid=g))
    r = run(c, f, 40.0)
    summary('持瓶路徑上有障礙', r, c, f)
    chk('路徑有致命格 ⇒ 拒絕採用、停止', (r['stop'] or {}).get('why') == 'carry_plan_rejected', r['stop'])


def case_blocked_mid():
    pl = FakePlanner(monitor=True)
    c, f = build(planner=pl)
    done = dict(x=False)

    def hook(t, s):
        if c.pph == 'CARRY_NAV' and not done['x'] and t > c.place_ev['CARRY_NAV']['t'] + 1.0:
            g = empty_grid()
            p = c.cpath.xy(c.csig + 0.8)[0]
            ix, iy = int((p[0] + 5.0) / 0.05), int((p[1] + 3.0) / 0.05)
            g.a[iy - 2:iy + 3, ix - 2:ix + 3] = 99
            g.stamp = c.t
            c.cmon.on_grid(g, c.csig)
            done['x'] = True
    r = run(c, f, 50.0, hook)
    summary('持瓶途中路徑被擋', r, c, f)
    chk('途中路徑被擋 ⇒ path_blocked 停止', (r['stop'] or {}).get('why') == 'path_blocked' and (r['stop_effect'] or {}).get('ok'), r['stop'])


def case_no_support():
    c, f = build(mode='low_table')
    r = run(c, f, 70.0)
    summary('下降到上限仍無支撐', r, c, f)
    chk('無支撐 ⇒ no_support 停止，未鬆爪', (r['stop'] or {}).get('why') == 'no_support' and c.t_release is None, r['stop'])


def case_side():
    c, f = build(mode='side')
    r = run(c, f, 70.0)
    summary('只有水平接觸（側碰）', r, c, f)
    chk('側碰不算支撐，且為第二桌禁止接觸 ⇒ 未鬆爪、停止', c.t_release is None and (r['stop'] or {}).get('why') in ('forbidden_contact:bottle_table2', 'no_support'), r['stop'])


def case_sticky():
    c, f = build(mode='sticky')
    r = run(c, f, 70.0)
    summary('張開後手指仍有力', r, c, f)
    chk('鬆爪未確認 ⇒ 不退離、停止', (r['stop'] or {}).get('why') == 'release_not_confirmed' and r['place']['pph'] == 'RELEASE', r['stop'])


def case_gap():
    c, f = build()
    st = dict(n=0)

    def hook(t, s):
        if c.pph == 'LOWER' and c.t_sup0 is not None and c.t_release is None and st['n'] < 25:
            st['n'] += 1
            s.gap = True
        else:
            s.gap = False
    orig = f.meas

    def meas(**ov):
        m = orig(**ov)
        if getattr(f, 'gap', False):
            m['f_place'] = np.full(3, np.nan)
        return m
    f.meas = meas
    r = run(c, f, 70.0, hook)
    summary('支撐確認中力資料中斷 0.25 s', r, c, f)
    chk('資料中斷 > 0.1 s ⇒ data_stale_place_force 停止、未鬆爪', (r['stop'] or {}).get('why') == 'data_stale_place_force' and c.t_release is None, r['stop'])


def case_gap_short():
    c, f = build()
    st = dict(n=0)

    def hook(t, s):
        s.gap = c.pph == 'LOWER' and c.t_sup0 is not None and c.t_release is None and st.__setitem__('n', st['n'] + 1) is None and st['n'] <= 5
    orig = f.meas

    def meas(**ov):
        m = orig(**ov)
        if getattr(f, 'gap', False):
            m['f_place'] = np.full(3, np.nan)
        return m
    f.meas = meas
    r = run(c, f, 75.0, hook)
    summary('支撐確認中力資料中斷 0.05 s', r, c, f)
    t_rel = r['place']['events'].get('RELEASE', {}).get('t')
    chk('短暫中斷不跨過：計數歸零後重新累計 0.3 s 才鬆爪', t_rel is not None and t_rel - c.t_sup0 >= 0.3 + 0.05 - 1e-6, (t_rel, c.t_sup0))


def case_reverse():
    c, f = build(mode='reverse')
    r = run(c, f, 70.0)
    summary('支撐力方向相反', r, c, f)
    chk('反向力不成立支撐 ⇒ 未鬆爪、停止', c.t_release is None and r['stop'] is not None and c.d_freeze is None, r['stop'])


def case_early():
    c, f = build(mode='early')
    r = run(c, f, 60.0)
    summary('LOWER 前瓶碰第二桌面', r, c, f)
    chk('LOWER 前接觸 ⇒ forbidden_contact:bottle_table2', (r['stop'] or {}).get('why') == 'forbidden_contact:bottle_table2' and c.pph in ('CARRY_NAV', 'PLACE_APPROACH'), (r['stop'], c.pph))


def case_share():
    c, f = build(mode='share')
    r = run(c, f)
    summary('桌面只承 1/3 mg（夾爪仍握持）', r, c, f)
    chk('分擔重量：接觸就緒後鬆爪，HOLD 驗到獨立承重而完成', r['mission_done'] is True and r['stop'] is None, r['stop'])


def case_flicker():
    c, f = build(mode='flicker')
    r = run(c, f)
    summary('接觸力逐步交替 0／2/3 mg', r, c, f)
    chk('逐步交替（單步掉力）不阻擋就緒，任務完成', r['mission_done'] is True and r['stop'] is None, r['stop'])


def case_dropout():
    c, f = build(mode='dropout')
    r = run(c, f, 70.0)
    summary('接觸中每 30 步掉力 10 步', r, c, f)
    chk('連續掉力超過 5 步 ⇒ 不就緒、未鬆爪、停止', c.t_release is None and r['stop'] is not None and r['place']['ready_why'] == 'dropout', (r['stop'], r['place']['ready_why']))


def case_low():
    c, f = build(mode='low')
    r = run(c, f, 70.0)
    summary('桌面只承 0.15 mg', r, c, f)
    chk('低於首次接觸門檻 ⇒ 不凍結、繼續下降，以 no_support 或滑移受控停止，未鬆爪', ((r['stop'] or {}).get('why') == 'no_support' or str((r['stop'] or {}).get('why', '')).startswith('slip')) and c.t_release is None and c.d_freeze is None, r['stop'])


def case_stop_unit():
    """有界微動停穩：合成停止後軌跡。"""
    R = PC.REG
    q0 = np.r_[2.3, -0.99, 0.0, 0.0, 0.3, 1.1, 0.0, -1.4, 0.0]
    s0 = q0[3:].copy()

    def rows(fn, cmd=0.0, pipe=0.0, sp_drift=0.0, gap=None):
        out = []
        for k in range(0, 201):
            t = 10.0 + k * 0.01
            if gap and gap[0] <= k < gap[1]:
                continue
            q = q0 + fn(k * 0.01)
            out.append((t, q, s0 + sp_drift * k, np.r_[cmd, np.zeros(8)], pipe))
        return out
    rng = np.random.default_rng(0)
    jit = lambda tt: np.r_[0, 0, 0, rng.normal(0, 1e-5, 6)]                     # noqa: E731
    cases = [('微抖 1e-5 rad', rows(jit), True),
             ('持續漂移 2e-3 rad/s', rows(lambda tt: np.r_[0, 0, 0, 0, 0, 0, 2e-3 * tt, 0, 0]), False),
             ('往返抵消但峰峰 3e-3 rad', rows(lambda tt: np.r_[0, 0, 0, 0, 0, 0, 1.5e-3 * np.sin(2 * np.pi * 4 * tt), 0, 0]), False),
             ('底盤漂移 0.02 m/s', rows(lambda tt: np.r_[0.02 * tt, 0, 0, 0, 0, 0, 0, 0, 0]), False),
             ('資料中斷 0.2 s', rows(jit, gap=(80, 100)), None),
             ('命令仍非零', rows(jit, cmd=1e-4), False),
             ('在途佇列非零', rows(jit, pipe=1e-4), False),
             ('設定點仍在變', rows(jit, sp_drift=1e-6), False)]
    for name, rr, want in cases:
        res = PC.stop_settled(rr, 10.0, R)
        chk(f'停穩：{name} ⇒ {want}', res['ok'] is want, {k: v for k, v in res.items() if k in ('ok', 'fails', 'why', 'arm_rate', 'arm_ptp', 'base_rate')})


CASES = dict(stop_unit=case_stop_unit, share=case_share, flicker=case_flicker, dropout=case_dropout, low=case_low, reverse=case_reverse, early=case_early, nominal=case_nominal, gap=case_gap, gap_short=case_gap_short, plan_fail=case_plan_fail, blocked_grid=case_blocked_grid, blocked_mid=case_blocked_mid, no_support=case_no_support, side=case_side, sticky=case_sticky)


def main():
    names = sys.argv[1:] or list(CASES)
    for n in names:
        CASES[n]()
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
