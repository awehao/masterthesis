#!/usr/bin/env python3
"""第一拍 Shield 一致列候選的真節點假世界（bottle_w1_liftc_fake.py 的版本化複本，2026-11-01；Codex 20261101_130612）：控制器換成 bottle_w1_shrow_ctrl，其餘相同。
ROS 鏈的 Shield 在命令送出時評估（同物理）；離線 NC 鏈在命令生效時逐步評估，兩者時序不同。
以下為原說明。
W1 分段抬升＋進 CARRY 重規劃候選的真節點假世界（bottle_place_fake.py 的版本化複本，2026-11-01；Codex 20261101_125000 條件 4）。
控制器＝bottle_w1_liftc_ctrl；假受控對象＋合成第二桌支撐＋離線射線取代 Isaac；relay、追蹤器、Shield、map_server、planner_server、RSP 都是真節點與真話題。
視覺量測照 test_bottle_w1.run 合成（底盤／手腕估計經 V2Link，與離線同偏差與雜訊）；導航段監看進度用 bottle_monitor_progress（同 isaac_bottle_w1dynm）。
模式：nominal（無移動物）、dyn2（原第二案 w1_dyn_overtake1：離線圓柱進掃描，只供驅動與評估，不進控制器）。
只驗流程：等待持瓶段規劃期間安全鏈不中斷、目標不跳、回覆核對；合成支撐與假受控對象不能當物理證據。
  PLAN=1 MAP_ARGS="--table2 2.4,-1.0" SIM_MODE=fake FAKE_SCRIPT=evaluation/bottle_w1_liftc_fake.py FAKE_MODE=nominal|dyn2 bash evaluation/run_bottle_ros.sh
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--mode', default='nominal', choices=['nominal', 'dyn2'])
ap.add_argument('--t-end', type=float, default=float(os.environ.get('FAKE_T_END', '75')))
ap.add_argument('--step-s', type=float, default=0.02)
ap.add_argument('--plan', action='store_true')
ap.add_argument('--goal', default='-0.9,0.0')
a = ap.parse_args()


def main():
    T, S1, SC, TP, LC, TV2, VL, VZ, G, DC = RC.import_with_real_ros('test_bottle_s0b', 'bottle_s1_ctrl', 'bottle_s1_cand', 'test_bottle_place', 'bottle_w1_shrow_ctrl', 'test_bottle_v2',
                                                                     'bottle_v2_link', 'bottle_v1_cam_visibility', 'bottle_s1_geom', 'bottle_w1_dyn_case')
    import bottle_monitor_progress as MPG
    import bottle_n2_path as NP
    import bottle_n2_ros as NR
    import bottle_place_ros as PR
    import test_bottle_w1 as TW
    import rclpy
    rclpy.init()
    chain = RC.RosChain()
    node = RC.make_bridge(chain)
    chain.tx = node
    CA = DC.CASES['w1_dyn_overtake1'] if a.mode == 'dyn2' else None
    bx = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']

    def scan_fn(t, q):
        r = SC.NC.ray_boxes(q[:2], float(q[2]), bx, SC.G.LIDAR_Z)
        if CA is not None:
            oc, _ = DC.state(t, CA)
            r = np.minimum(r, DC.ray_cyl(q[:2], SC.NC.scan_angles() + float(q[2]), oc, CA['radius'], SC.NC.RANGE_MAX))
        return r
    pr = NR.PlanRos(node)
    q0 = np.r_[SC.X0, 0.0, 0.0, np.array(SC.Q_TUCK) + T.D_TRUE]

    def state0():
        return q0[:2], (1.0, 0.0, 0.0, 0.0), q0[3:]
    T0, why = RC.handshake(node, chain, lambda: time.sleep(a.step_s), state0, lambda: scan_fn(0.0, q0), wall_limit_s=40.0)
    if T0 is None:
        print(f'[fake] 握手未過 {json.dumps(why, ensure_ascii=False, default=str)[:900]}', flush=True)
        return 70
    goal = tuple(float(v) for v in a.goal.split(','))
    pp, info = NR.plan_before_start(node, chain, pr, lambda: time.sleep(a.step_s), state0, lambda: scan_fn(0.0, q0), goal, -0.9, T0 + RC.PHYS)
    if pp is None:
        print(f'[fake] 導航段無有效路徑 ⇒ 不起跑：{json.dumps(info, ensure_ascii=False, default=float)[:600]}', flush=True)
        return 72
    T0 = info['t_ready'] + RC.PHYS
    clock = dict(t=0.0)
    mon = NP.PathMonitor(pp)
    mon.on_grid(pr.grid, pp.sig0)
    mprog = MPG.MonitorProgress(pp, S1.B.Y_TUBE)
    tg0 = T.T_G
    T.T_G = np.full((4, 4), np.nan)                      # 不先放真值（同 test_bottle_w1.build）
    try:
        c = SC.make(cls=LC.W1PlaceController, path=SC.PATH_P2, obs=chain.obstacles, chain=chain, scan_fn=scan_fn, monitor=mon, carry_planner=None, **SC.FAST)
    finally:
        T.T_G = tg0
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])
    cp = PR.RosCarryPlanner(pr, lambda: T0 + clock['t'], lambda: c.csig)
    c.carry_planner = cp
    pr.monitor, pr.sig_fn = mon, (lambda: mprog.sig)
    chain.ctrl = c
    f = TP.FakePlace(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    print(f"[fake] 握手 T0 {T0:.2f}；導航段 L {pp.L:.3f} m；模式 {a.mode}", flush=True)

    rng = np.random.default_rng(0)
    capb, capw = TV2.FakeCap(), TV2.FakeCap()
    lb, lw = VL.V2Link(capb, c), VL.V2Link(capw, c, sink=c.wrist_intake)
    vs = dict(nb=0.0, nw=0.0, pend=[])
    T_WR = VZ.CAMS['wrist']['T']

    def vision(t, s):
        """同 test_bottle_w1.run：底盤相機每 0.2 s（CRUISE、可見）、手腕相機每 0.2 s（≤0.35 m）合成估計，延遲 0.02／0.05 s 到達。"""
        for p_ in [p_ for p_ in vs['pend'] if p_[0] <= t + 1e-9]:
            p_[1].ests[p_[2]] = (p_[0], p_[3])
            vs['pend'].remove(p_)
        lb.tick(t)
        lw.tick(t)
        ph = s.c.st.phase
        if t >= vs['nb'] - 1e-9 and ph == 'CRUISE':
            vs['nb'] = t + 0.2
            ok, cam = TV2.visible(s.q)
            if ok:
                xy = np.array([T.BX, T.BY]) + np.array(TV2.BIAS_R2) + rng.normal(0, TV2.NOISE_R2)
                k = round(t, 6)
                capb.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                vs['pend'].append((t + 0.02, capb, k, TW.msg(t, xy, cam)))
        if t >= vs['nw'] - 1e-9 and ph in ('CRUISE', 'PREGRASP', 'INSERT'):
            vs['nw'] = t + 0.2
            cam = (G.C.K.fk(s.q, 'link_eef') @ T_WR)[:3, 3]
            if float(np.linalg.norm(np.array([T.BX, T.BY, T.TOP + 0.1]) - cam)) <= 0.35:
                xy = np.array([T.BX, T.BY]) + np.array(TW.W_BIAS) + rng.normal(0, TW.W_NOISE)
                k = round(t, 6)
                capw.frames[k] = dict(task_t=t, dev_mm=0.0, dev_deg=0.0)
                vs['pend'].append((t + 0.05, capw, k, TW.msg(t, xy, cam)))

    def state():
        y = float(f.q[2])
        return f.q[:2], (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)), f.q[3:]
    chain.start(T0)
    st = dict(w=time.perf_counter(), wait=[])

    def hook(t, s):
        clock['t'] = t
        mprog.update(f.q, t)
        p, qt, qa = state()
        node.publish_state(T0 + t, p, qt, qa)
        chain.set_now(T0 + t)
        node.drain()
        cp.tick()
        vision(t, s)
        if c.pph is None:                                # 承接後代價圖只餵持瓶段監看器；導航段監看器到此結束
            mon.check_age(T0 + t, mprog.sig)
        elif c.cmon is not None:
            c.cmon.check_age(T0 + t, c.csig)
        if c.cw is not None and c.cw.get('t_accept') is None and c.judge.stop is None:   # 等待期間逐步：掃描年齡、Shield 診斷、參考速度
            st['wait'].append(dict(t=round(t, 3), scan_age=chain.age(t), v_b=float(c.v_b), q=[round(float(v), 4) for v in f.q[:3]]))
        if t >= a.t_end - 3.0 and not c.mission_done:
            c._trigger_stop('sim_limit')
        rest = a.step_s - (time.perf_counter() - st['w'])
        if rest > 0:
            time.sleep(rest)
        st['w'] = time.perf_counter()
        return {}

    f.run(a.t_end, hook, tail=2.6)
    r = c.report()
    pl = r['place']
    tt = np.array([h['t'] for h in f.hist])
    Q = np.array([h['q'] for h in f.hist])
    sp = np.convolve(np.hypot(np.gradient(Q[:, 0], tt), np.gradient(Q[:, 1], tt)), np.ones(5) / 5, mode='same')
    ev = {e['what']: e['t'] for e in r['events']}
    t_end = ev.get('mission_done') or (r['stop'] or {}).get('t', float(tt[-1]))
    cw = r['paired_strategy'].get('carry_wait') or {}
    t1 = cw.get('t_accept') or t_end
    m_w = (tt >= (cw.get('t_req') or 1e9)) & (tt <= t1)
    extra = dict(mode=a.mode, wait_log=st['wait'], near_run_s=float((sp[(tt >= 0.2) & (tt <= t_end)] < 0.01).sum() * 0.01), near_wait_s=float((sp[m_w] < 0.01).sum() * 0.01),
                 scan_age_wait_max=max((w['scan_age'] for w in st['wait']), default=None), monitor_progress=mprog.report())
    if CA is not None:
        ob = np.array([DC.state(x, CA)[0] for x in tt])
        extra['base_center_to_surface_min_m'] = float((np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - CA['radius']).min())
    extra['u0_rows'] = {k: v for k, v in r['paired_strategy'].get('u0_rows', {}).items() if k != 'log_tail'}
    r['fake_liftc'] = extra
    print(f"[fake] 第一拍列 {extra['u0_rows']}", flush=True)
    print(f"[fake] 模式 {a.mode} 完成 {r['mission_done']} 碼 {S1.exit_code(r)} 阻點 {r['stop']} 停止有效 {r['stop_effect'] and r['stop_effect'].get('ok')}", flush=True)
    print(f"[fake] 等待持瓶段規劃 {cw}；等待步數 {len(st['wait'])}、掃描年齡最大 {extra['scan_age_wait_max']}；近停 等待中 {extra['near_wait_s']:.2f} s／全程 {extra['near_run_s']:.2f} s；底盤中心到障礙表面最小 {extra.get('base_center_to_surface_min_m')}", flush=True)
    print(f"[fake] 請求 {pl['carry_req']}；承接 {pl['carry'] and {k: v for k, v in pl['carry'].items() if k != 'plan'}}", flush=True)
    print(f"[fake] 持瓶規劃 {pl['carry'] and {k: v for k, v in (pl['carry'].get('plan') or {}).items() if k in ('id', 'status', 'wall_ms', 'grid_age')}}；放置事件 {[(k, v['t']) for k, v in pl['events'].items()]}；指標 {pl['metrics']}", flush=True)
    print(f"[fake] 掃描鏈 { {k: r['nav_chain'][k] for k in ('n_scan', 'n_static_max', 'shield')} }；求解 {r['timing']['solve']}；fallback {r['n_fallback']}；切換 {r['paired_strategy'].get('switch')}", flush=True)
    json.dump(r, open(os.path.join(a.out, 'fake_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return S1.exit_code(r)


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
            fh.write(str(rc))
    sys.exit(rc)
