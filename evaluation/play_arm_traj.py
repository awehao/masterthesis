"""播放經 check_arm_path.py 檢查過的那條關節軌跡。

軌跡由 evaluation/arm_traj.trapezoid 產生 —— **與檢查端 import 同一個函式**，
所以「沿途已檢查」指的就是這裡送出去的每一點。本節點不做任何平滑、內插或
重新參數化；模擬器端只收位置目標，不在裡面積分。

    /arm/joint_position_cmd   Float64MultiArray, 6
"""
import argparse, json, math, os, sys, time
import numpy as np, yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from arm_traj import trapezoid

ARM = [f'joint{i}' for i in range(1, 7)]
ap = argparse.ArgumentParser()
ap.add_argument('--case', default='')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--hold-s', type=float, default=3.0, help='走完後持續送終點的秒數')
ap.add_argument('--out', default='')
ap.add_argument('--wait-subs', type=float, default=15.0,
                help='等訂閱者計數的秒數；等不到也照發')
a = ap.parse_args()

C = yaml.safe_load(open(a.cases)); name = a.case or C['default_case']
case = C['cases'][name]; P = yaml.safe_load(open(a.poses))
q0 = np.array([float(P[case['pregrasp']['start_config']][j]) for j in ARM])
q1 = np.array([float(P[case['pregrasp']['arm_config']][j]) for j in ARM])
tr = case['trajectory']
Q, T = trapezoid(q0, q1, tr['joint_vel_max_rps'], tr['joint_acc_max_rps2'], tr['rate_hz'])

rclpy.init()
n = Node('play_arm_traj')
# 單變數測試：endpoint_check（看得到 Isaac 的那個行程）沒有設 use_sim_time，
# 播放端有設且看不到。本節點用 time.sleep 控制節奏，不依賴 ROS 時鐘，
# 所以拿掉它不改變送出的軌跡內容或時序。
pub = n.create_publisher(Float64MultiArray, '/arm/joint_position_cmd', 10)

print(f'案例 {name}：{len(Q)} 點，{T:.2f} s，{tr["rate_hz"]:.0f} Hz '
      f'（v≤{tr["joint_vel_max_rps"]} rad/s，a≤{tr["joint_acc_max_rps2"]} rad/s²）')
# 訂閱者計數只作參考，不作為放行條件。
# 實測：Isaac 行程的節點名在圖上解析不出來（ros2 CLI 甚至看不到任何節點），
# 但它的端點確實在線上 —— endpoint_check 在同一趟就看得到 /cmd_vel 的訂閱。
# 因此改為「等一段時間、把看到的數字記下來，然後照發」，由接收端實際收到
# 幾則來判定傳輸是否成立，而不是用圖查詢當閘門。
t0 = time.monotonic()
while n.count_subscribers('/arm/joint_position_cmd') < 1 and time.monotonic() - t0 < a.wait_subs:
    rclpy.spin_once(n, timeout_sec=0.1)
nsub = n.count_subscribers('/arm/joint_position_cmd')
print(f'訂閱者計數 {nsub}（等待 {time.monotonic()-t0:.1f} s）'
      + ('' if nsub else ' —— 計數為 0，仍照發，由接收端計數判定'))

dt = 1.0 / tr['rate_hz']
sent = []
nxt = time.monotonic()
for i, q in enumerate(Q):
    m = Float64MultiArray(); m.data = [float(v) for v in q]
    pub.publish(m); sent.append([float(v) for v in q])
    rclpy.spin_once(n, timeout_sec=0.0)
    nxt += dt
    s = nxt - time.monotonic()
    if s > 0:
        time.sleep(s)
    else:
        nxt = time.monotonic()
hold_n = int(a.hold_s / dt)
for _ in range(hold_n):
    m = Float64MultiArray(); m.data = [float(v) for v in Q[-1]]
    pub.publish(m); rclpy.spin_once(n, timeout_sec=0.0)
    time.sleep(dt)
published = len(Q) + hold_n
print(f'播放完成：實際發布 {published} 則（軌跡 {len(Q)} + 保持 {hold_n}）')
if a.out:
    json.dump(dict(schema='arm_traj_sent/1', case=name, n=len(Q), T=T,
                   rate_hz=tr['rate_hz'], hold_s=a.hold_s,
                   published_msgs=published, subs_seen_before_play=nsub,
                   traj=sent), open(a.out, 'w'), ensure_ascii=False)
    print(f'-> {a.out}')
n.destroy_node(); rclpy.shutdown()
