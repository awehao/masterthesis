#!/usr/bin/env python3
"""瓶子 N1 上 Isaac 前的真節點驗收：假受控對象＋離線射線取代 Isaac，其餘全是真節點與真話題（relay、追蹤器、Shield、map_server、RSP）。
跑法把這支當模擬端：SIM_MODE=fake FAKE_MODE=full|kill_shield|kill_tracker|stop_scan bash evaluation/run_bottle_ros.sh。不是物理，步調壓到約 RTF 0.5 以貼近 Isaac。
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
a = ap.parse_args()


def kill_node(name):
    """只動本趟 procs.tsv 記下的群組。"""
    for ln in open(os.path.join(a.out, 'procs.tsv')):
        f = ln.split()
        if f[0] == name:
            os.killpg(int(f[2]), signal.SIGTERM)
            print(f'[fake] 已停 {name} 群組 {f[2]}', flush=True)


def main():
    T, S1, SC = RC.import_with_real_ros('test_bottle_s0b', 'bottle_s1_ctrl', 'bottle_s1_cand')
    import rclpy
    rclpy.init()
    chain = RC.RosChain()
    node = RC.make_bridge(chain)
    chain.tx = node
    bx = SC.scene_boxes()
    dead = dict(scan=False)
    scan_fn = lambda t, q: None if dead['scan'] else SC.NC.ray_boxes(q[:2], float(q[2]), bx, SC.G.LIDAR_Z)     # noqa: E731
    c = SC.make(path=SC.PATH_P2, obs=chain.obstacles, chain=chain, scan_fn=scan_fn, **SC.FAST)
    chain.ctrl = c
    f = T.Fake(c, x0=SC.X0)
    f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE

    def state():
        y = float(f.q[2])
        return f.q[:2], (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)), f.q[3:]

    T0, why = RC.handshake(node, chain, lambda: time.sleep(a.step_s), state, lambda: scan_fn(0.0, f.q), wall_limit_s=40.0)
    if T0 is None:
        print(f'[fake] 握手未過 {json.dumps(why, ensure_ascii=False, default=str)[:900]}', flush=True)
        return 70
    print(f'[fake] 握手通過 T0 {T0:.2f}；發布者 {chain.graph["publishers"]}；TF {chain.graph["tf_pairs"].keys()}', flush=True)
    chain.start(T0)
    st = dict(done=False, w=time.perf_counter())

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
