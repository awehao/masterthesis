"""播放經 check_arm_path.py 檢查過的關節軌跡，**以模擬時間排程**。

為什麼改時鐘：手臂在模擬器裡運動，軌跡的 4.52 s 理應是**模擬時間**。
前一版用 time.monotonic()/time.sleep() 排程（牆鐘），於是「每牆鐘秒 50 則」
與接收端「每模擬秒 11.5 則」兩個數字分屬不同時鐘，既不能相除，
也無法說軌跡是否按預定時間執行。

訊息格式（仍用 std_msgs/Float64MultiArray，不新增介面）：

    [0] seq       序號，從 0 起
    [1] kind      0 = 軌跡設定點，1 = 終點保持
    [2] t_sched   預定執行的模擬時間，相對軌跡起點
    [3..8]        joint1..joint6 目標

軌跡由 evaluation/arm_traj.trapezoid 產生 —— 與 check_arm_path.py import
同一個函式。本節點不做平滑、內插或重新參數化。
"""
import argparse, json, math, os, sys, time
import numpy as np, yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
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
ap.add_argument('--hold-s', type=float, default=3.0, help='走完後保持終點的模擬秒數')
ap.add_argument('--lead-s', type=float, default=1.0, help='起點比現在晚多少模擬秒')
ap.add_argument('--clock-wait', type=float, default=60.0)
ap.add_argument('--stall-s', type=float, default=30.0,
                help='等單一設定點的牆鐘上限；超過視為模擬時間停止')
ap.add_argument('--out', default='')
a = ap.parse_args()

C = yaml.safe_load(open(a.cases)); name = a.case or C['default_case']
case = C['cases'][name]; P = yaml.safe_load(open(a.poses))
q0 = np.array([float(P[case['pregrasp']['start_config']][j]) for j in ARM])
q1 = np.array([float(P[case['pregrasp']['arm_config']][j]) for j in ARM])
tr = case['trajectory']
Q, T = trapezoid(q0, q1, tr['joint_vel_max_rps'], tr['joint_acc_max_rps2'], tr['rate_hz'])
dt = 1.0 / tr['rate_hz']
n_hold = int(round(a.hold_s / dt))

rclpy.init()
n = Node('play_arm_traj')
sim = {'t': None}
n.create_subscription(Clock, '/clock',
                      lambda m: sim.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9),
                      QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST))
pub = n.create_publisher(Float64MultiArray, '/arm/joint_position_cmd', 10)

t0 = time.monotonic()
while sim['t'] is None and time.monotonic() - t0 < a.clock_wait:
    rclpy.spin_once(n, timeout_sec=0.05)
if sim['t'] is None:
    print('!! 等不到 /clock，無法以模擬時間排程'); sys.exit(1)
print(f'/clock 已就緒（等待 {time.monotonic()-t0:.1f} s），現在模擬時間 {sim["t"]:.3f} s')

try:
    matched = pub.get_subscription_count()
except Exception:
    matched = None
qos = pub.qos_profile
qos_txt = (f'reliability={qos.reliability.name} durability={qos.durability.name} '
           f'history={qos.history.name} depth={qos.depth}')
print(f'發布器看見的匹配訂閱數 {matched}；型別 std_msgs/msg/Float64MultiArray')
print(f'  QoS {qos_txt}')

# 排程表：軌跡點 + 終點保持，各帶序號、類別、預定模擬時間
T_START = sim['t'] + a.lead_s
sched = []
for i, q in enumerate(Q):
    sched.append((i, 0.0, i * dt, q))
for k in range(n_hold):
    sched.append((len(Q) + k, 1.0, T + (k + 1) * dt, Q[-1]))

print(f'案例 {name}：軌跡 {len(Q)} 點（{T:.2f} s 模擬時間）+ 保持 {n_hold} 則'
      f'（{a.hold_s:.1f} s），起點 sim {T_START:.3f} s')

sent = []
late = []
for seq, kind, ts, q in sched:
    due = T_START + ts
    # 牆鐘上限：模擬器結束後 /clock 會停，sim['t'] 就此凍結。沒有這個上限時
    # 迴圈會永遠等下去（實測卡住 9 分鐘才被人工中止）。
    w0 = time.monotonic()
    stalled = False
    while sim['t'] is not None and sim['t'] < due:
        if time.monotonic() - w0 > a.stall_s:
            stalled = True
            break
        rclpy.spin_once(n, timeout_sec=0.005)
    if stalled:
        print(f'!! 等待模擬時間 {due:.3f} s 超過 {a.stall_s:.0f} s 牆鐘仍未到達'
              f'（sim 停在 {sim["t"]:.3f} s）——模擬器可能已結束，中止播放')
        break
    m = Float64MultiArray()
    m.data = [float(seq), float(kind), float(ts)] + [float(v) for v in q]
    pub.publish(m)
    now = sim['t']
    sent.append(dict(seq=seq, kind=int(kind), t_sched=ts, pub_sim_t=now,
                     late_s=(now - due) if now is not None else None))
    if now is not None and now - due > 2 * dt:
        late.append(seq)
    rclpy.spin_once(n, timeout_sec=0.0)

try:
    matched_end = pub.get_subscription_count()
except Exception:
    matched_end = None
n_traj = sum(1 for s in sent if s['kind'] == 0)
n_hold_sent = sum(1 for s in sent if s['kind'] == 1)
lates = [s['late_s'] for s in sent if s['late_s'] is not None]
print(f'發布完成：軌跡 {n_traj} 則 + 保持 {n_hold_sent} 則 = {len(sent)} 則')
if lates:
    print(f'  發布相對預定時刻的延遲：中位 {np.median(lates)*1000:.1f} ms  '
          f'max {max(lates)*1000:.1f} ms；超過 2 個週期的 {len(late)} 則')
print(f'  匹配訂閱數 前 {matched} / 後 {matched_end}')

if a.out:
    json.dump(dict(schema='arm_traj_sent/3', case=name,
                   n_traj=n_traj, n_hold=n_hold_sent, T_traj=T,
                   rate_hz=tr['rate_hz'], dt=dt, t_start_sim=T_START,
                   clock='sim', matched_subs_before=matched,
                   matched_subs_after=matched_end,
                   msg_type='std_msgs/msg/Float64MultiArray', qos=qos_txt,
                   msgs=sent,
                   traj=[[float(v) for v in q] for q in Q]),
              open(a.out, 'w'), ensure_ascii=False)
    print(f'  -> {a.out}')
n.destroy_node(); rclpy.shutdown()
