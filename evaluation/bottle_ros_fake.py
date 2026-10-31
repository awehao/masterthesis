#!/usr/bin/env python3
"""瓶子 N1 上 Isaac 前的真節點驗收：假受控對象＋離線射線取代 Isaac，其餘全是真節點與真話題（relay、追蹤器、Shield、map_server、RSP）。
跑法把這支當模擬端：SIM_MODE=fake FAKE_MODE=full|kill_shield|kill_tracker|stop_scan|block_path [PLAN=1] bash evaluation/run_bottle_ros.sh。
N2（--plan）：起跑前向 planner_server 要路徑，控制器換 S1Planned；block_path 在路徑前方加一個掃描看得到的方塊，驗路徑失效停止；--goal 可改規劃目標（驗無有效路徑）。不是物理，步調壓到約 RTF 0.5 以貼近 Isaac。
"""
import argparse
import json
import math
import os
import signal
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
ap.add_argument('--t-fault', type=float, default=5.0)
ap.add_argument("--t-end", type=float, default=float(os.environ.get("FAKE_T_END", "45")))
ap.add_argument('--step-s', type=float, default=0.02)
ap.add_argument('--plan', action='store_true')
ap.add_argument('--goal', default='-0.9,0.0', help='N2 規劃目標＝直線段入口')
a = ap.parse_args()


def kill_node(name):
    """只動本趟 procs.tsv 記下的群組。"""
    for ln in open(os.path.join(a.out, 'procs.tsv')):
        f = ln.split()
        if f[0] == name:
            os.killpg(int(f[2]), signal.SIGTERM)
            print(f'[fake] 已停 {name} 群組 {f[2]}', flush=True)


def main():
    T, S1, SC, N2C = RC.import_with_real_ros('test_bottle_s0b', 'bottle_s1_ctrl', 'bottle_s1_cand', 'bottle_n2_ctrl')
    import bottle_n2_path as NP
    import bottle_n2_ros as NR
    import rclpy
    rclpy.init()
    chain = RC.RosChain()
    node = RC.make_bridge(chain)
    chain.tx = node
    bx = SC.scene_boxes()
    dead = dict(scan=False)
    scan_fn = lambda t, q: None if dead['scan'] else SC.NC.ray_boxes(q[:2], float(q[2]), bx, SC.G.LIDAR_Z)     # noqa: E731
    pr = NR.PlanRos(node) if a.plan else None
    q0 = np.r_[SC.X0, 0.0, 0.0, np.array(SC.Q_TUCK) + T.D_TRUE]

    def state0():
        return q0[:2], (1.0, 0.0, 0.0, 0.0), q0[3:]
    if not a.plan:
        c = SC.make(path=SC.PATH_P2, obs=chain.obstacles, chain=chain, scan_fn=scan_fn, **SC.FAST)

    T0, why = RC.handshake(node, chain, lambda: time.sleep(a.step_s), state0, lambda: scan_fn(0.0, q0), wall_limit_s=40.0)
    if T0 is None:
        print(f'[fake] 握手未過 {json.dumps(why, ensure_ascii=False, default=str)[:900]}', flush=True)
        return 70
    print(f'[fake] 握手通過 T0 {T0:.2f}；發布者 {chain.graph["publishers"]}；TF {chain.graph["tf_pairs"].keys()}', flush=True)
    if a.plan:
        goal = tuple(float(v) for v in a.goal.split(','))
        pp, info = NR.plan_before_start(node, chain, pr, lambda: time.sleep(a.step_s), state0, lambda: scan_fn(0.0, q0), goal, -0.9, T0 + RC.PHYS)
        json.dump(dict(plan=info), open(os.path.join(a.out, 'plan.json'), 'w'), ensure_ascii=False, indent=1, default=float)
        if pp is None:
            print(f'[fake] 無有效路徑 ⇒ 不起跑：{json.dumps(info, ensure_ascii=False, default=float)[:600]}', flush=True)
            return 72
        print(f"[fake] 規劃 L {pp.L:.3f} m、σ0 {pp.sig0:.3f}、原始 {info['n_raw']} 點、代價峰 {info['cost_max']}、規劃 {info['planning_time'] * 1e3:.0f} ms（牆鐘 {info['wall_ms']:.0f} ms）", flush=True)
        T0 = info['t_ready'] + RC.PHYS
        mon = NP.PathMonitor(pp)
        mon.on_grid(pr.grid, pp.sig0)                    # 先用規劃時那張代價圖；之後每張新圖都核對剩餘段
        c = SC.make(cls=N2C.S1Planned, path=SC.PATH_P2, obs=chain.obstacles, chain=chain, scan_fn=scan_fn, monitor=mon, **SC.FAST)
        c.path = pp                                      # 換成規劃路徑；DetourPath 只為了建構，不會被用到
        c.s_ref = c.sig_robot = pp.sig0
        pr.monitor, pr.sig_fn = mon, (lambda: c.sig_robot)
        if a.mode == 'block_path':
            a.t_fault = min(a.t_fault, 2.0)              # 早放、放遠：要驗路徑監看，不是讓 Shield 在 6 cm 處先擋
            pb = pp.xy(pp.sig0 + 2.6)[0]
            print(f'[fake] block_path：t≥{a.t_fault} 在路徑 σ0＋2.6 m 處 {pb.round(3).tolist()} 加 0.3 m 方塊', flush=True)
    chain.ctrl = c
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE

    def state():
        y = float(f.q[2])
        return f.q[:2], (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)), f.q[3:]
    chain.start(T0)
    st = dict(done=False, w=time.perf_counter())
    mon = c.monitor if a.plan else None

    def hook(t, s):
        p, qt, qa = state()
        node.publish_state(T0 + t, p, qt, qa)
        chain.set_now(T0 + t)
        node.drain()
        if not st['done'] and t >= a.t_fault:
            st['done'] = True
            if a.mode == 'kill_shield':
                kill_node('shield')
            elif a.mode == 'kill_tracker':
                kill_node('tracker')
            elif a.mode == 'stop_scan':
                dead['scan'] = True
            elif a.mode == 'block_path':
                bx.append((np.array([pb[0], pb[1], 0.25]), np.array([0.3, 0.3, 0.5])))
        if mon is not None:
            mon.check_age(T0 + t, c.sig_robot)
        rest = a.step_s - (time.perf_counter() - st['w'])
        if rest > 0:
            time.sleep(rest)
        st['w'] = time.perf_counter()
        return {}

    f.run(a.t_end, hook)
    r = c.report()
    q = np.array([h['u'] for h in f.hist])
    print(f"[fake] 模式 {a.mode} 成功 {r['success']} 完成 {r['mission_done']} 碼 {S1.exit_code(r)} 阻點 {r['stop']} 停止有效 {r['stop_effect']} 出口 {r['t_exit_in']}", flush=True)
    print(f"[fake] 末命令 {np.round(q[-1], 4).tolist()}；事件 {[(round(e['t'], 2), e.get('to', e['what'])) for e in r['events']][-6:]}", flush=True)
    print(f"[fake] 掃描鏈 { {k: r['nav_chain'][k] for k in ('n_scan', 'n_dyn_max', 'n_static_max', 'shield')} }", flush=True)
    print(f"[fake] ROS { {k: v for k, v in r['nav_chain']['ros'].items() if k not in ('graph', 'note')} }", flush=True)
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
