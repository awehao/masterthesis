"""播放抽屜案例的軌跡，**以模擬時間排程**，並在正確時刻送出 engage / release。

三個話題，刻意分開：

    /arm/joint_position_cmd  Float64MultiArray [seq, kind, t_sched, q1..q6]
                             與第一階段**同一個格式**，沿用已驗收的命令鏈
    /manip/gripper_cmd       Float64MultiArray [seq, t_sched, finger_target]
    /manip/phase_cmd         String(JSON) {seq, t_sched, phase, event}

事件（engage / release）用 RELIABLE + KEEP_ALL 發，漏掉 release 的後果是
「手臂帶著抽屜退出」而不是退開。模擬端收到後必須在 /manip/status 回報實際
套用的事件與當時的模擬時間，播放端不假設它一定收到。

軌跡由 gen_drawer_traj.py 產生並已逐點驗證；**本節點不做平滑、內插或重新
參數化**，只照表發。
"""
import argparse, csv, json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray, String

ap = argparse.ArgumentParser()
ap.add_argument('--traj', required=True, help='gen_drawer_traj.py 產生的 traj.csv')
ap.add_argument('--lead-s', type=float, default=1.5, help='起點比現在晚多少模擬秒')
ap.add_argument('--clock-wait', type=float, default=90.0)
ap.add_argument('--sub-wait', type=float, default=60.0, help='等訂閱者的牆鐘上限')
ap.add_argument('--stall-s', type=float, default=30.0,
                help='等單一設定點的牆鐘上限；超過視為模擬時間停止')
ap.add_argument('--out', default='')
a = ap.parse_args()

ARM = [f'joint{i}' for i in range(1, 7)]
rows = []
with open(a.traj) as f:
    for r in csv.DictReader(f):
        rows.append({'seq': int(r['seq']), 't': float(r['t']), 'phase': r['phase'],
                     'event': r['event'], 'q': [float(r[j]) for j in ARM],
                     'finger': float(r['finger']),
                     'opening': float(r['expected_opening'])})
if not rows:
    print('!! 軌跡為空'); sys.exit(1)
meta_p = os.path.join(os.path.dirname(os.path.abspath(a.traj)), 'traj_meta.json')
META = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
if META and not META.get('pass', False):
    print('!! traj_meta.json 標示驗證未通過，拒絕播放'); sys.exit(2)

rclpy.init()
n = Node('play_drawer_traj')
be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST)
rel = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_ALL,
                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
sim = {'t': None}
n.create_subscription(Clock, '/clock',
                      lambda m: sim.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), be)
acks = []
n.create_subscription(String, '/manip/status',
                      lambda m: acks.append(m.data), 10)
pub_q = n.create_publisher(Float64MultiArray, '/arm/joint_position_cmd', 10)
pub_g = n.create_publisher(Float64MultiArray, '/manip/gripper_cmd', 10)
pub_p = n.create_publisher(String, '/manip/phase_cmd', rel)

t0 = time.monotonic()
while sim['t'] is None and time.monotonic() - t0 < a.clock_wait:
    rclpy.spin_once(n, timeout_sec=0.05)
if sim['t'] is None:
    print('!! 等不到 /clock'); sys.exit(1)
print(f'/clock 就緒（等 {time.monotonic()-t0:.1f} s），模擬時間 {sim["t"]:.3f} s')

t0 = time.monotonic()
while time.monotonic() - t0 < a.sub_wait:
    if (pub_q.get_subscription_count() > 0 and pub_g.get_subscription_count() > 0
            and pub_p.get_subscription_count() > 0):
        break
    rclpy.spin_once(n, timeout_sec=0.05)
cnt = (pub_q.get_subscription_count(), pub_g.get_subscription_count(),
       pub_p.get_subscription_count())
print(f'匹配訂閱數 arm/gripper/phase = {cnt}（等 {time.monotonic()-t0:.1f} s）')
if min(cnt) == 0:
    print('!! 有話題沒有訂閱者，中止（不在沒人收的情況下空跑）'); sys.exit(3)

T_START = sim['t'] + a.lead_s
print(f'{len(rows)} 個設定點，總時長 {rows[-1]["t"]:.3f} s，起點 sim {T_START:.3f} s')
sent, late, events = [], [], []
last_phase = None
for r in rows:
    due = T_START + r['t']
    w0 = time.monotonic(); stalled = False
    while sim['t'] is not None and sim['t'] < due:
        if time.monotonic() - w0 > a.stall_s:
            stalled = True; break
        rclpy.spin_once(n, timeout_sec=0.005)
    if stalled:
        print(f'!! 等模擬時間 {due:.3f} s 超過 {a.stall_s:.0f} s（sim 停在 '
              f'{sim["t"]:.3f} s）——模擬器可能已結束，中止播放')
        break
    # 事件與階段先送，設定點後送：模擬端要先知道這一步屬於哪個階段
    if r['event'] or r['phase'] != last_phase:
        s = String()
        s.data = json.dumps({'seq': r['seq'], 't_sched': r['t'],
                             'phase': r['phase'], 'event': r['event']})
        pub_p.publish(s)
        if r['event']:
            events.append({'seq': r['seq'], 'event': r['event'],
                           't_sched': r['t'], 'pub_sim_t': sim['t']})
            print(f'  事件 {r["event"]:8s} seq {r["seq"]} t_sched {r["t"]:.3f} '
                  f'sim {sim["t"]:.3f}')
        last_phase = r['phase']
    g = Float64MultiArray()
    g.data = [float(r['seq']), float(r['t']), float(r['finger'])]
    pub_g.publish(g)
    m = Float64MultiArray()
    m.data = [float(r['seq']), 0.0, float(r['t'])] + [float(v) for v in r['q']]
    pub_q.publish(m)
    now = sim['t']
    sent.append({'seq': r['seq'], 't_sched': r['t'], 'pub_sim_t': now,
                 'late_s': (now - due) if now is not None else None,
                 'phase': r['phase']})
    if now is not None and now - due > 0.04:
        late.append(r['seq'])
    rclpy.spin_once(n, timeout_sec=0.0)

lates = [s['late_s'] for s in sent if s['late_s'] is not None]
print(f'發布完成 {len(sent)} / {len(rows)} 則')
if lates:
    print(f'  相對預定時刻延遲：中位 {np.median(lates)*1000:.1f} ms  '
          f'max {max(lates)*1000:.1f} ms；超過 2 週期的 {len(late)} 則')
applied = [x for x in acks if '"event_applied"' in x]
print(f'  模擬端回報的事件套用訊息 {len(applied)} 則')
for x in applied[-4:]:
    print(f'    {x}')
if a.out:
    json.dump({'schema': 'drawer_traj_sent/1', 'traj': os.path.abspath(a.traj),
               'traj_meta': META, 'n_sent': len(sent), 'n_rows': len(rows),
               't_start_sim': T_START, 'clock': 'sim',
               'matched_subs': list(cnt), 'events': events,
               'sim_event_acks': applied, 'msgs': sent},
              open(a.out, 'w'), ensure_ascii=False)
    print(f'  -> {a.out}')
n.destroy_node(); rclpy.shutdown()
