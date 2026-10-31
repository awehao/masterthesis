#!/usr/bin/env python3
"""第二桌放置的真節點假世界（bottle_ros_fake.py 的放置版複本）：假受控對象（合成第二桌支撐）＋離線射線取代 Isaac，
其餘是真節點與真話題（relay、追蹤器、Shield、map_server、planner_server、RSP）。導航段與持瓶段都向 planner_server 要路徑。
合成支撐只驗流程與停止，不能當物理放置證據。
  PLAN=1 MAP_ARGS="--table2 2.4,-1.0" SIM_MODE=fake FAKE_SCRIPT=evaluation/bottle_place_fake.py FAKE_MODE=full|carry_block bash evaluation/run_bottle_ros.sh
carry_block：LIFT 開始時在持瓶目標（直線入口）附近放一個掃描看得到的方塊 ⇒ 規劃失敗或採用核對不過 ⇒ 受控停止。
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
ap.add_argument('--mode', default='full')
ap.add_argument('--t-end', type=float, default=float(os.environ.get('FAKE_T_END', '75')))
ap.add_argument('--step-s', type=float, default=0.02)
ap.add_argument('--plan', action='store_true')
ap.add_argument('--goal', default='-0.9,0.0')
a = ap.parse_args()


def main():
    T, S1, SC, PCm, TP = RC.import_with_real_ros('test_bottle_s0b', 'bottle_s1_ctrl', 'bottle_s1_cand', 'bottle_place_ctrl', 'test_bottle_place')
    import bottle_n2_path as NP
    import bottle_n2_ros as NR
    import bottle_place_ros as PR
    import rclpy
    rclpy.init()
    chain = RC.RosChain()
    node = RC.make_bridge(chain)
    chain.tx = node
    bx = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    scan_fn = lambda t, q: SC.NC.ray_boxes(q[:2], float(q[2]), bx, SC.G.LIDAR_Z)     # noqa: E731
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
    c = SC.make(cls=PCm.PlaceController, path=SC.PATH_P2, obs=chain.obstacles, chain=chain, scan_fn=scan_fn, monitor=mon,
                carry_planner=None, **SC.FAST)
    c.path, c.s_ref, c.sig_robot = pp, pp.sig0, pp.sig0
    c.exit_c = np.array([99.0, 99.0])
    cp = PR.RosCarryPlanner(pr, lambda: T0 + clock['t'], lambda: c.csig)
    c.carry_planner = cp
    pr.monitor, pr.sig_fn = mon, (lambda: c.sig_robot)
    chain.ctrl = c
    f = TP.FakePlace(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE
    print(f"[fake] 握手 T0 {T0:.2f}；導航段 L {pp.L:.3f} m；模式 {a.mode}", flush=True)

    def state():
        y = float(f.q[2])
        return f.q[:2], (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)), f.q[3:]
    chain.start(T0)
    st = dict(w=time.perf_counter(), blocked=False)

    def hook(t, s):
        clock['t'] = t
        p, qt, qa = state()
        node.publish_state(T0 + t, p, qt, qa)
        chain.set_now(T0 + t)
        node.drain()
        cp.tick()
        if a.mode == 'carry_block' and not st['blocked'] and c.st.phase == 'LIFT':
            st['blocked'] = True
            gx = c.park[0] - c.place['entry_back'] if c.park is not None else 1.7
            bx.append((np.array([gx, float(c.park[1]) if c.park is not None else -1.0, 0.25]), np.array([0.5, 0.5, 0.5])))
            print(f'[fake] carry_block：t {t:.2f} 在持瓶目標附近 ({gx:.2f}) 加 0.5 m 方塊', flush=True)
        mon.check_age(T0 + t, c.sig_robot)
        if c.cmon is not None:
            c.cmon.check_age(T0 + t, c.csig)
        if t >= a.t_end - 3.0 and not c.mission_done:
            c._trigger_stop('sim_limit')
        rest = a.step_s - (time.perf_counter() - st['w'])
        if rest > 0:
            time.sleep(rest)
        st['w'] = time.perf_counter()
        return {}

    f.run(a.t_end, hook)
    r = c.report()
    pl = r['place']
    print(f"[fake] 模式 {a.mode} 完成 {r['mission_done']} 碼 {S1.exit_code(r)} 阻點 {r['stop']} 停止有效 {r['stop_effect']}", flush=True)
    print(f"[fake] 放置事件 {[(k, v['t']) for k, v in pl['events'].items()]}；停車 {pl['park']}；承接 {pl['carry'] and {k: v for k, v in pl['carry'].items() if k != 'plan'}}", flush=True)
    print(f"[fake] 持瓶規劃 {pl['carry'] and {k: v for k, v in (pl['carry'].get('plan') or {}).items() if k in ('id', 'status', 'wall_ms', 'grid_age')}}；監看 {pl['monitor']}；指標 {pl['metrics']}", flush=True)
    print(f"[fake] 掃描鏈 { {k: r['nav_chain'][k] for k in ('n_scan', 'n_static_max', 'shield')} }；求解 {r['timing']['solve']}；fallback {r['n_fallback']}", flush=True)
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
