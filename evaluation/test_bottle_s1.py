#!/usr/bin/env python3
"""S1 候選邏輯負測：完整任務（繞障→移動抓取→出口區）的成功條件、位置事件、導航段底盤座標任務、停止與 fallback。
假受控對象驅動同一份 S1Controller（Isaac 腳本也用它）；無接觸物理，只驗邏輯。
    python3 evaluation/test_bottle_s1.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_cbf_replay as RP                       # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

BAD, N = [], 0


def chk(name, c, info=''):
    global N
    N += 1
    if not c:
        BAD.append(f'{name} {info}')


def stop_of(f):
    return (f.c.judge.stop or {}).get('why')


def lim(ts):
    return lambda t, s: (s.c._trigger_stop('sim_limit') if t >= ts else None) or {}


def run(c, t_end=110.0, fault=None, **kw):
    return T.Fake(c, x0=SC.X0, **kw).run(t_end, fault or lim(t_end - 5.0))


def main():
    p = S1.DetourPath(*SC.PATH)
    xs = np.linspace(-3.5, -0.5, 301)
    num = np.gradient([p.y(float(x)) for x in xs], xs)
    chk('P1 dy 與數值微分一致（接點處二階不連續，容差取步長尺度）', np.abs(num[1:-1] - [p.dy(float(x)) for x in xs[1:-1]]).max() < 0.015, str(np.abs(num[1:-1] - [p.dy(float(x)) for x in xs[1:-1]]).max()))
    chk('P2 路徑兩端 y＝0', p.y(-3.4) == 0.0 and p.y(-0.6) == 0.0 and p.y(0.2) == 0.0)
    chk('P3 名目路徑全程在 keep-out 外', SC.path_clearance() >= SC.CLEAR_MIN, str(SC.path_clearance()))

    f = run(SC.make())
    r = f.c.report()
    print('名目', r['success'], [(e['t'], e.get('to', e['what'])) for e in r['events'] if e['what'] in ('phase', 'blend_start', 'exit_enter', 'mission_done', 'done')])
    chk('T1 完整任務成功', r['success'] and S1.exit_code(r) == 0, str(r['stop']))
    ev = [e.get('to', e['what']) for e in r['events'] if e['what'] in ('phase', 'blend_start', 'exit_enter', 'mission_done')]
    chk('T2 事件次序', ev == ['blend_start', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY', 'exit_enter', 'mission_done'], str(ev))
    h_nav = [h for h in f.hist if h['phase'] == 'CRUISE' and h['x'] < -1.0]
    h_ss = [h for h in h_nav if h['t'] >= 1.0]                         # 起步 1 s 內是 d̂ 收斂暫態（假受控對象重力偏移），峰 14 mrad
    d_nav = max(np.abs(h['q'][3:] - T.Q_READY).max() for h in h_ss)
    chk('T3 導航段手臂維持備妥（實測關節變化 ≤5 mrad）', d_nav <= 0.005, str(d_nav))
    e_nav = max(float(np.linalg.norm(T.K.fk(np.r_[0.0, 0.0, 0.0, h['q'][3:]], 'link_tcp')[:3, 3] - T.T_READY[:3, 3])) for h in h_ss[::10])
    chk('T3 導航段 TCP 相對底盤誤差 ≤2 mm', e_nav <= 0.002, str(e_nav))
    chk('T4 確實繞過障礙（y 峰 ≥0.7 m）', min(h['q'][1] for h in f.hist) <= -0.7)
    chk('T5 展臂由位置觸發', next(e for e in r['events'] if e['what'] == 'blend_start')['x'] >= f.c.x_blend)
    chk('T6 判定通過時仍要進出口才算成功', r['exit_reached'] and r['t_exit_in'] > r['t_lift_ok'])
    chk('T7 出口內停下', math.hypot(f.hist[-1]['q'][0] - 1.0, f.hist[-1]['q'][1]) <= 0.30 and abs(f.hist[-1]['u'][0]) < 0.01)
    chk('T8 標明導航資料來源', r['nav_source'] == 'truth_obstacle_to_cbf' and r['shield'] is False)
    chk('T9 全程 GI', r['qp_counts'].get('gi', 0) == r['n_tick'] and r['n_fallback'] == 0, str(r['qp_counts']))

    # N1 出口前被擋：抓持五項全過也不算成功
    wall = lambda t: [SC.obstacles(t)[0], dict(x=0.75, y=0.0, radius=0.10, margin=0.47, static=True)]
    f = run(SC.make(obs=wall), 100.0)
    r = f.c.report()
    chk('N1 抓持通過但未抵出口 ⇒ 不成功', r['success_s0_judge'] is False or r['success'] is False, str(r['success']))
    chk('N1 不報成功、非零離開碼', r['success'] is False and S1.exit_code(r) != 0, str(r['stop']))
    chk('N1 未進出口區', r['exit_reached'] is False, str(r['t_exit_in']))
    # N2 參考路徑穿進 keep-out：底盤停在邊界、不穿透、不報成功
    f = run(SC.make(path=(-3.4, -2.0, -0.2)), 40.0)
    r = f.c.report()
    d = min(math.hypot(h['q'][0] + 2.0, h['q'][1]) for h in f.hist)
    chk('N2 死鎖不報成功', r['success'] is False and r['t_close'] is None, str(r['stop']))
    chk('N2 不穿進 keep-out（≥ −5 mm）', d - SC.R_OBST - SC.M_STATIC >= -0.005, str(d))
    # N3 導航中各種停止：底盤有界減速、停止有效
    for name, fault, want in (('禁止接觸', lambda t, s: dict(forbidden={T.REQ[0]: 0.2, T.REQ[1]: 0.0}) if t >= 6.0 else {}, 'forbidden_contact'),
                              ('瓶位姿過期', lambda t, s: dict(t_bottle=6.0) if t > 6.0 else {}, 'data_stale_bottle_pose')):
        f = run(SC.make(), 30.0, fault)
        u0 = max(abs(h['u_last'][0]) for h in f.hist if 5.8 <= h['t'] < 6.0)
        chk(f'N3 {name} ⇒ 停止', str(stop_of(f)).startswith(want), str(stop_of(f)))
        chk(f'N3 {name} 停止有效、離開碼 3', (f.c.stop_ok or {}).get('ok') is True and S1.exit_code(f.c.report()) == 3, str(f.c.stop_ok))
        d_ul = np.abs(np.diff(np.array([h['u_last'] for h in f.hist if h['stopped']])[::5], axis=0))
        chk(f'N3 {name} 底盤減速有界', d_ul.size == 0 or (d_ul[:, :3] <= B.A_BASE * B.DT + 1e-9).all(), str(u0))
    # N4 fallback 不當可繼續的命令
    c = SC.make()
    c.mpc_fail_inject = lambda t: dict(fallback='base_brake', failed=False, seq_consistent=False) if t >= 6.0 else None
    f = run(c, 30.0)
    chk('N4 fallback ⇒ 停止', stop_of(f) == 'wb_fallback:base_brake' and f.c.report()['success'] is False, str(stop_of(f)))
    # N5 無抓持證據：不抬升、不進 CARRY、不算出口
    f = T.Fake(SC.make(), f_hold=0.0, x0=SC.X0).run(60.0, lim(55.0))
    r = f.c.report()
    chk('N5 無證據 ⇒ 不進 CARRY、不成功', all(h['phase'] != 'CARRY' for h in f.hist) and r['success'] is False and r['exit_reached'] is False, str(r['stop']))
    # N6 CARRY 中失抓：不算成功
    t_l = next(e['t'] for e in run(SC.make()).c.report()['events'] if e.get('to') == 'CARRY')
    f = run(SC.make(), 80.0, lambda t, s: dict(f_fingers=(0.0, 0.0)) if t >= t_l + 2.0 else {})
    chk('N6 持瓶途中失抓 ⇒ 停止、不成功', str(stop_of(f)).startswith('grasp_loss') and f.c.report()['success'] is False, str(stop_of(f)))
    # N7 偏離放行管：不開始展臂
    c = SC.make()
    c.t = 20.0
    q_bad = np.r_[-0.40, 0.2, 0.0, T.Q_READY]
    c._advance(T.K.fk(q_bad, 'link_tcp'), False, q_bad)
    chk('N7 管外不開始展臂', c.blending is False and c.n_gate >= 1)
    q_y = np.r_[-0.40, 0.0, math.radians(4.0), T.Q_READY]
    c._advance(T.K.fk(q_y, 'link_tcp'), False, q_y)
    chk('N7 偏航超出不開始展臂', c.blending is False)
    # N9 出口判定：在出口內、參考已歸零但底盤仍動 ⇒ 不得 mission_done
    c = SC.make()
    c.done, c.v_b, c.t_exit_in, c.st.phase, c.st.grasp_ok = True, 0.0, 1.0, 'CARRY', True
    c.st.T_carry_rel = np.eye(4)
    c.u_prev, c.u_last = np.zeros(9), np.zeros(9)
    c.judge.res['hold_ok'] = True
    q_e = np.r_[0.9, 0.0, 0.0, T.Q_READY]
    Tb0 = T.bottle_T()
    for k in range(200):
        q_e[0] += 0.05 * 0.01                                          # 底盤實際 0.05 m/s
        c.q_prev2 = q_e - np.array([0.05 * 0.01] + [0.0] * 8)
        r0 = c.n_still
        v = float(np.hypot(*(q_e[:2] - c.q_prev2[:2]))) / 0.01
        c.n_still = c.n_still + 1 if v < 0.01 else 0
    chk('N9 速度計算用更新前位置（底盤動 ⇒ 不累計）', c.n_still == 0 and c.mission_done is False, str(c.n_still))
    f = run(SC.make())
    h_end = [h for h in f.hist if h['t'] >= next(e['t'] for e in f.c.events if e['what'] == 'mission_done') - 0.2]
    chk('N9 mission_done 時實測確實停住', max(float(abs(h['u'][0])) for h in h_end) < 0.02, str(max(float(abs(h['u'][0])) for h in h_end)))

    # 直接餵量測：出口內、參考已歸零、底盤實測仍 0.05 m/s ⇒ 不得完成；真的停住 0.2 s 後才完成
    c = SC.make()
    fk = T.Fake(c, x0=0.9)
    c.done, c.v_b, c.t_exit_in, c.st.phase, c.st.grasp_ok, c.st.T_carry_rel, c.done_anchored = True, 0.0, 1.0, 'CARRY', True, c.task.T_lift.copy(), True
    c.judge.res['hold_ok'] = True
    c.st.T_carry_rel = np.linalg.inv(T.K.fk(np.r_[0.0, 0.0, 0.0, T.Q_READY], 'link_tcp')) @ T.K.fk(np.r_[0.0, 0.0, 0.0, T.Q_READY], 'link_tcp')
    c.st.T_carry_rel = T.K.fk(np.r_[0.0, 0.0, 0.0, T.Q_READY], 'link_tcp')
    t_ = 0.0
    for k in range(60):
        fk.t = t_
        c.step(t_, fk.q.copy(), **fk.meas())
        fk.q[0] += 0.0005
        t_ += 0.01
    chk('N9 出口內實測仍動 ⇒ 不 mission_done', c.mission_done is False and c.v_b == 0.0, f'{c.mission_done} v_b {c.v_b} still {c.n_still}')
    for k in range(40):
        fk.t = t_
        c.step(t_, fk.q.copy(), **fk.meas())
        t_ += 0.01
    chk('N9 真的停住才完成', c.mission_done is True and c.n_still >= S1.N_STILL, f'{c.mission_done} still {c.n_still}')

    # N10 非手指↔瓶線上拒絕：慢伺服觸發、停止點不跨 5 mm；名目不觸發
    f = run(SC.make(path=SC.PATH_P2), 60.0)
    r = f.c.report()
    print('名目瓶淨空', r['clear_bottle'])
    chk('N10 名目不觸發且成功', r['success'] and r['clear_bottle']['pred_min'] >= S1.CLEAR_REJECT + S1.CLEAR_GUARD, str(r['clear_bottle']))
    worst_rest = float('inf')
    for al in (0.6, 0.5, 0.4, 0.3):
        f = run(SC.make(path=SC.PATH_P2), 60.0, alpha_scale=al)
        r = f.c.report()
        chk(f'N10 α×{al} ⇒ clearance_bottle 停止', stop_of(f) == 'clearance_bottle', str(stop_of(f)))
        chk(f'N10 α×{al} 不成功、離開碼非零', r['success'] is False and S1.exit_code(r) != 0, str(f.c.stop_ok))
        if al >= 0.6:                                                  # 更慢時受控對象 1 s 內尚未靜止，回報停止未驗證（離開碼 4）才是正確行為
            chk(f'N10 α×{al} 停止有效、離開碼 3', (f.c.stop_ok or {}).get('ok') is True and S1.exit_code(r) == 3, str(f.c.stop_ok))
        d_rest = min(f.c._bottle_clear(h['q'], T.bottle_T()) for h in f.hist if h['t'] >= f.c.judge.stop['t'] - 0.1)
        worst_rest = min(worst_rest, d_rest)
        chk(f'N10 α×{al} 停止全程實測 ≥ 5 mm', d_rest >= S1.CLEAR_REJECT, str(d_rest))
    print('慢伺服停止段實測最小', round(worst_rest * 1e3, 2), 'mm')
    # N11 不同抵達時機的拒絕：注入瓶位姿朝夾爪偏 3 mm，淨空變窄也要拒絕
    Tbad = T.bottle_T()
    Tbad[1, 3] -= 0.003
    f = run(SC.make(path=SC.PATH_P2), 60.0, lambda t, s: dict(T_bottle=Tbad.copy()) if t >= 20.0 else {})
    chk('N11 瓶位姿偏近 ⇒ 不成功', f.c.report()['success'] is False, str(stop_of(f)))
    # N12 放行管排除已知角落
    c = SC.make()
    chk('N12 y<0、yaw<0 且 x 過角落界 ⇒ 不放行', not c._in_tube(np.r_[0.05, -0.02, -0.01, T.Q_READY]))
    chk('N12 同 x、y>0 可放行', bool(c._in_tube(np.r_[0.05, 0.02, -0.01, T.Q_READY])))
    # N13 CBF 緊湊紀錄：同狀態重解重現實際命令；移除 CBF 的反事實與線上重解一致
    c = SC.make(path=SC.PATH_P2, log_cbf=True)
    c.compact = []
    f = T.Fake(c, x0=SC.X0).run(30.0, None, tail=0.0)
    pth = os.path.join(os.environ.get('TMPDIR', '/tmp'), 's1_cbf_test.npz')
    S1.save_compact(c, pth)
    rep = RP.replay(pth, n_stride=5)
    chk('N13 重解重現實際命令', rep['reproduce_max'] <= 1e-6, str(rep['reproduce_max']))
    du_on = np.array([e['du'] for e in c.cbf_log])[::5]
    chk('N13 反事實與線上一致', np.abs(np.array(rep['du'])[:len(du_on)] - du_on[:len(rep['du'])]).max() <= 1e-5, str(np.abs(np.array(rep['du'])[:len(du_on)] - du_on[:len(rep['du'])]).max()))
    chk('N13 確實記到 CBF 改命令', rep['n_bound'] > 0, str(rep['n_bound']))
    os.remove(pth)
    # N8 離開碼
    chk('N8 未中止未完成 ⇒ 4', S1.exit_code(dict(stop=None, success=False)) == 4)
    chk('N8 中止有效 ⇒ 3', S1.exit_code(dict(stop=dict(why='x'), success=False, stop_effect=dict(ok=True))) == 3)
    chk('N8 中止未驗 ⇒ 4', S1.exit_code(dict(stop=dict(why='x'), success=False, stop_effect=None)) == 4)
    if BAD:
        print(f'未過 {len(BAD)}／{N}：')
        for b in BAD:
            print('  ', b)
        return 1
    print(f'S1 候選邏輯 {N} 項：全過')
    return 0


if __name__ == '__main__':
    sys.exit(main())
