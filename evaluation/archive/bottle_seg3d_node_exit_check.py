#!/usr/bin/env python3
"""分割節點收尾核對（Codex 20261102_092601：SIGTERM 核心傾印另修，以獨立節點啟停核對，不為此重跑任務）。
獨立 ROS domain（預設 77）、牆鐘時間：啟動 bottle_seg3d_node（.venv-dl0 python，--ns w1，use_sim_time false），本程序以 15 Hz 發布同戳四則訊息
（/w1/color＝R-B1 實錄手腕影格 rgb8、/w1/depth＝常數 1.0 m 32FC1、/w1/camera_info fx 465.6、/w1/cam_pose frame odom），讓工作執行緒持續推論；
收到 ≥ 3 筆 /w1/bottle_seg3d 後，等隨機 0–0.25 s（落在推論中或閒置）再對節點程序群組送 SIGTERM（同 run_bottle_ros.sh cleanup），最多等 8 s。
每趟記：結束碼、是否需要 KILL、stderr 是否含「terminate called」或 Traceback、結束耗時；正常結束（rc 0）與強制結束（rc 3＋「強制結束」訊息）分開計數。對照：git HEAD 版本（修正前）與工作目錄版本各 N 趟（CHECK_VERS 可只跑其一）。CHECK_SLOW＝秒數 ⇒ 節點推論後多睡（模擬收尾時推論卡住）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_seg3d_node_exit_check.py 6
"""
import json
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
DOMAIN = os.environ.get('CHECK_DOMAIN', '77')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PY = os.path.join(WS, '.venv-dl0', 'bin', 'python')
W = os.path.join(HERE, 'dl0_ckpt', 'bottle_seg_v1_continue3', 'best.pth')
OUT = os.environ.get('CHECK_OUT', os.path.join(HERE, 'results', 'vision', 'seg3d_node_exit_check.json'))
TMP = os.environ.get('CHECK_TMP', '/tmp')


def node_files():
    old = os.path.join(TMP, 'bottle_seg3d_node_head.py')
    with open(old, 'w') as f:
        f.write(subprocess.run(['git', '-C', WS, 'show', 'HEAD:evaluation/bottle_seg3d_node.py'], capture_output=True, text=True, check=True).stdout)
    return dict(head=old, work=os.path.join(HERE, 'bottle_seg3d_node.py'))


def main():
    os.environ['ROS_DOMAIN_ID'] = DOMAIN
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float64MultiArray
    F = np.load(os.path.join(HERE, 'runs', 'bottle_rb1_1', 'w1_frames.npz'), allow_pickle=True)
    rgb = np.ascontiguousarray(np.asarray(F['rgb'][len(F['rgb']) // 3], np.uint8))
    H, Wd = rgb.shape[:2]
    depth = np.full((H, Wd), 1.0, np.float32)
    rclpy.init()
    n = Node('seg3d_exit_check')
    pubs = dict(c=n.create_publisher(Image, '/w1/color', 10), d=n.create_publisher(Image, '/w1/depth', 10), i=n.create_publisher(CameraInfo, '/w1/camera_info', 10),
                p=n.create_publisher(PoseStamped, '/w1/cam_pose', 10))
    got = []
    n.create_subscription(Float64MultiArray, '/w1/bottle_seg3d', lambda m: got.append(time.time()), 50)

    def publish():
        t = time.time()
        st = n.get_clock().now().to_msg()
        st.sec, st.nanosec = int(t), int((t - int(t)) * 1e9)
        c = Image(height=H, width=Wd, encoding='rgb8', is_bigendian=0, step=3 * Wd, data=rgb.tobytes())
        d = Image(height=H, width=Wd, encoding='32FC1', is_bigendian=int(sys.byteorder == 'big'), step=4 * Wd, data=depth.tobytes())
        i = CameraInfo(height=H, width=Wd)
        i.k = [465.6, 0.0, Wd / 2, 0.0, 465.6, H / 2, 0.0, 0.0, 1.0]
        p = PoseStamped()
        p.pose.position.z = 1.0
        p.pose.orientation.w = 1.0
        for m, fr in ((c, 'cam'), (d, 'cam'), (i, 'cam'), (p, 'odom')):
            m.header.stamp, m.header.frame_id = st, fr
        pubs['c'].publish(c)
        pubs['d'].publish(d)
        pubs['i'].publish(i)
        pubs['p'].publish(p)
    files = node_files()
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN, PYTHONPATH=HERE + os.pathsep + os.environ.get('PYTHONPATH', ''))
    if os.environ.get('CHECK_SLOW'):                                 # 模擬收尾時推論卡住：節點每格推論後多睡 CHECK_SLOW 秒
        env['SEG3D_EXIT_TEST_DELAY'] = os.environ['CHECK_SLOW']
    env.pop('VIRTUAL_ENV', None)
    res = {}
    rnd = random.Random(20261102)
    try:
        for ver, path in [(v_, files[v_]) for v_ in os.environ.get('CHECK_VERS', 'head,work').split(',')]:
            res[ver] = []
            for k in range(N):
                errp = os.path.join(TMP, f'seg3d_exit_{ver}_{k}.err')
                with open(errp, 'w') as ef:
                    pr = subprocess.Popen([PY, '-u', path, '--weights', W, '--ns', 'w1', '--ros-args', '-p', 'use_sim_time:=false'], env=env, stdout=subprocess.DEVNULL, stderr=ef,
                                          start_new_session=True)
                got.clear()
                t0 = time.time()
                nxt = t0
                while len(got) < 3 and time.time() - t0 < 90 and pr.poll() is None:
                    if time.time() >= nxt:
                        publish()
                        nxt += 1.0 / 15
                    rclpy.spin_once(n, timeout_sec=0.005)
                n_before = len(got)
                dly = rnd.uniform(0.0, 0.25)
                t1 = time.time()
                while time.time() - t1 < dly:
                    if time.time() >= nxt:
                        publish()
                        nxt += 1.0 / 15
                    rclpy.spin_once(n, timeout_sec=0.005)
                tk = time.time()
                try:
                    os.killpg(pr.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                killed = False
                try:
                    rc = pr.wait(timeout=float(os.environ.get('CHECK_GRACE', '8.0')))
                except subprocess.TimeoutExpired:
                    os.killpg(pr.pid, signal.SIGKILL)
                    rc, killed = pr.wait(), True
                dt_exit = time.time() - tk
                err = open(errp).read()
                res[ver].append(dict(trial=k, outputs_before=n_before, delay_s=round(dly, 3), rc=rc, killed=killed, exit_s=round(dt_exit, 3),
                                     terminate=('terminate called' in err), traceback=('Traceback' in err), core=('core dumped' in err) or rc in (-6, 134), err_tail=err[-300:]))
                print(ver, res[ver][-1]['trial'], {kk: res[ver][-1][kk] for kk in ('outputs_before', 'delay_s', 'rc', 'killed', 'exit_s', 'terminate', 'traceback')}, flush=True)
                time.sleep(1.0)
    finally:
        n.destroy_node()
        rclpy.shutdown()
    summ = {v: dict(n=len(x), clean=sum(1 for y in x if y['rc'] == 0 and not y['killed'] and not y['terminate'] and not y['traceback']), terminate=sum(y['terminate'] for y in x),
                    forced=sum(1 for y in x if y['rc'] == 3 and '強制結束' in y['err_tail'] and not y['killed'] and not y['terminate']),
                    abort=sum(1 for y in x if y['rc'] in (-6, 134)), killed=sum(y['killed'] for y in x), rcs=sorted({y['rc'] for y in x}), exit_s_max=max(y['exit_s'] for y in x)) for v, x in res.items()}
    json.dump(dict(domain=DOMAIN, n=N, weights=os.path.relpath(W, WS), summary=summ, trials=res), open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
