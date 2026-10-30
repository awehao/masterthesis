#!/usr/bin/env python3
"""nav1 r11 START_TURN 的假端測試（無物理）：假執行端以運動學積分節點輸出、發 /odom；控制者先 wholebody 後轉 nav；
假 GMPC 平滑器輸出固定命令。核：轉向期間 /nav1/state 仍是第二段 CRUISE 且執行端備妥核對可過、線速度降到 0、|ω| ≤ turn_w、|α| ≤ turn_a；
轉到 ≤ turn_deg 後依 bumpless 變化率接到導航命令。--stuck：底盤不轉 ⇒ 逾時 NAV_TIMEOUT（start_turn_timeout）且輸出零。
    ROS_DOMAIN_ID=99 python3 evaluation/test_nav1_start_turn.py [--stuck]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))


def nav_ready(d):                                                              # 與 ev9r8z _nav_ready 同條件（第二段）
    return d is not None and d.get('mission') == 2 and d.get('seg') == 'CRUISE' and d.get('cin_age') is not None \
        and 0.0 <= float(d['cin_age']) <= 0.25 and bool(d.get('plan_n'))


def win_rate(U, i, w=0.2):
    """≥ w 秒視窗的最大 |Δu_i|/Δt（收訊時刻有排程抖動，逐筆差分不代表節點自身的模擬步長）。"""
    r, j = 0.0, 0
    for k in range(len(U)):
        while j < k and U[k][0] - U[j + 1][0] >= w:
            j += 1
        if U[k][0] - U[j][0] >= w:
            r = max(r, abs(U[k][i] - U[j][i]) / (U[k][0] - U[j][0]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stuck', action='store_true')
    ap.add_argument('--ref', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60m.json'))
    ap.add_argument('--out', default='/tmp/test_nav1_start_turn.json')
    a = ap.parse_args()
    P = json.load(open(a.ref))
    b = P['segments']['4']['base'][-1]
    m4 = P['m4r']
    if os.path.exists(a.out):
        os.remove(a.out)
    pr = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_nav1_mission_node_r11.py'), '--bumpless', '--start-turn', '--rolling',
                           '--goal=0.6,-0.5332,1.570796', '--out', a.out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rclpy.init()
    n = Node('fake_nav1_env')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    po = n.create_publisher(Odometry, '/odom', 10)
    pc = n.create_publisher(Twist, '/cmd_vel_pre_mux', 10)
    ps = n.create_publisher(LaserScan, '/scan', 10)
    ph = n.create_publisher(String, '/handover/state', 10)
    pa = n.create_publisher(Float64MultiArray, '/coman/applied_cmd', 1)
    pm = n.create_publisher(String, '/nav1/mission', lat)
    out, states = [], []
    n.create_subscription(Twist, '/cmd_vel_pre_shield', lambda m: out.append((tsim[0], m.linear.x, m.linear.y, m.angular.z)), 50)
    n.create_subscription(String, '/nav1/state', lambda m: states.append((tsim[0], json.loads(m.data))), 50)
    x, y, th = float(b[0]), float(b[1]), float(b[2])
    tsim, owner, u_app = [100.0], 'wholebody', (0.0, 0.0, 0.0)
    U_NAV = (0.25, 0.0, -0.3)
    t_nav = None
    sent_m = False
    k = 0
    T0 = time.monotonic()
    while time.monotonic() - T0 < 40.0:
        k += 1
        dt = 0.02
        tsim[0] += dt
        if owner == 'nav' and out:
            u_app = out[-1][1:]
        if not (a.stuck and owner == 'nav'):
            c, s_ = math.cos(th), math.sin(th)
            x += (c * u_app[0] - s_ * u_app[1]) * dt
            y += (s_ * u_app[0] + c * u_app[1]) * dt
            th += u_app[2] * dt
        o = Odometry()
        o.header.stamp.sec, o.header.stamp.nanosec = int(tsim[0]), int((tsim[0] % 1) * 1e9)
        o.pose.pose.position.x, o.pose.pose.position.y = x, y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        po.publish(o)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = U_NAV
        pc.publish(tw)
        pa.publish(Float64MultiArray(data=[float(k), tsim[0], *u_app] + [0.0] * 13))
        ph.publish(String(data=json.dumps({'owner': owner})))
        if k % 5 == 0:
            sc = LaserScan()
            sc.header.frame_id = 'base_scan'
            ps.publish(sc)
        if not sent_m and k > 100:
            pm.publish(String(data=json.dumps({'id': 2, 'vertices': m4['R2']['vertices'], 'goal': m4['R2']['goal']})))
            sent_m = True
        last = states[-1][1] if states else None
        if owner == 'wholebody' and sent_m and last and last.get('mission') == 2 and k > 200:
            if not nav_ready(last):
                print('FAIL 交棒前備妥核對不過', last)
                break
            owner, t_nav = 'nav', tsim[0]
            print(f'[fake] 控制者 → nav @ {t_nav:.2f}；pose {x:.3f},{y:.3f},{math.degrees(th):.1f}°')
        rclpy.spin_once(n, timeout_sec=0.0)
        time.sleep(0.02)
        if t_nav is not None and tsim[0] - t_nav > (9.0 if a.stuck else 8.0):
            break
    pr.terminate()
    log = pr.communicate(timeout=10)[0]
    print('\n'.join(ln for ln in log.splitlines() if 'START_TURN' in ln or 'NAV_TIMEOUT' in ln))
    rep = json.load(open(a.out))
    st = rep.get('start_turn')
    print('start_turn', st)
    ok = True

    def expect(name, cond, det=''):
        nonlocal ok
        ok = ok and bool(cond)
        print(('PASS ' if cond else 'FAIL ') + name, det)
    U = [u for u in out if t_nav is not None and u[0] >= t_nav]
    S = [s for s in states if t_nav is not None and s[0] >= t_nav]
    if a.stuck:
        expect('逾時 ⇒ NAV_TIMEOUT start_turn_timeout', rep.get('result') == 'start_turn_timeout' and any(s[1].get('seg') == 'NAV_TIMEOUT' for s in S))
        t_to = next(s[0] for s in S if s[1].get('seg') == 'NAV_TIMEOUT') if any(s[1].get('seg') == 'NAV_TIMEOUT' for s in S) else None
        expect('逾時後輸出零', t_to is not None and all(abs(v) < 1e-12 for u in U if u[0] > t_to + 0.1 for v in u[1:]))
    else:
        turning = [s for s in S if s[1].get('turn') == 'turning']
        expect('轉向期間仍是第二段 CRUISE 且備妥核對可過', turning and all(nav_ready(s[1]) for s in turning), f'n={len(turning)}')
        expect('轉向完成且末差 ≤ 45°', st and st.get('t_end') is not None and abs(st['e_end_deg']) <= 45.0 + 1e-6, st and st.get('e_end_deg'))
        t_end = st.get('t_end') if st else None
        Ut = [u for u in U if t_end is not None and u[0] <= t_end]
        expect('轉向中 |ω| ≤ 0.8', all(abs(u[3]) <= 0.8 + 1e-9 for u in Ut))
        dw = win_rate(Ut, 3)
        expect('轉向中 |α| ≤ 2.0（0.2 s 視窗）', dw <= 2.0 * 1.05, f'{dw:.3f}')
        expect('轉向中線速度 0', all(abs(u[1]) < 1e-9 and abs(u[2]) < 1e-9 for u in Ut[3:]))
        Ua = [u for u in U if t_end is not None and u[0] > t_end]
        dv, dw2 = win_rate(Ua, 1), win_rate(Ua, 3)
        expect('轉向後線加速度 ≤ 0.5、角加速度 ≤ 1.2（0.2 s 視窗）', dv <= 0.5 * 1.05 and dw2 <= 1.2 * 1.05, f'{dv:.3f} {dw2:.3f}')
        expect('轉向後收斂到導航命令', Ua and max(abs(Ua[-1][i + 1] - U_NAV[i]) for i in range(3)) < 1e-6, Ua[-1] if Ua else None)
        expect('bumpless 記錄 after_turn', any(e.get('after_turn') for e in rep.get('bumpless', [])))
    print('全過' if ok else '有失敗')
    n.destroy_node()
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
