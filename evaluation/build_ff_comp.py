"""由無連接試驗的實測偏離，建立**逐指令序號**的前饋補償表（離線）。

定義（全部在世界座標）：

    ᵂT_D,ach(k) = ᵂT_G,free(k) · ᴳT_D          無連接時實際達到的抽屜位姿
    s(k)   = â^T ( p_ach(k) − p_D0 )            沿滑軌分量（**不補償、不更動配時**）
    e⊥(k)  = ( p_ach(k) − p_D0 ) − â s(k)       垂直滑軌的位置偏離（逐分量保留）
    ΔR(k)  = R_ach(k) · R_D0ᵀ                   **世界座標**下，由理想姿態到實際姿態
                                                的旋轉（左乘慣例）

補償只改命令的目標位姿，**沿軸分量原封不動**：

    R_D,cmd_new(k) = ΔR(k)⁻¹ · R_D,cmd_old(k)   （世界座標左乘）
    p_D,cmd_new(k) = p_D,cmd_old(k) − e⊥(k)     （只減垂直分量）
    ᵂT_G,cmd_new(k) = ᵂT_D,cmd_new(k) · (ᴳT_D)⁻¹

平滑：對逐序號的量測值取**置中移動平均**（視窗見輸出），不做參數擬合 ——
實測形狀不是線性（前段快速上升後趨緩），擬合會把這個形狀抹掉。

**適用條件**：同一模型、固定底座、同一路徑與配時、同一增益。
這是**本案例的致動校準**，不是通用方法。
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import drawer_align as AL                                   # noqa
import yaml                                                 # noqa

ARM = [f'joint{i}' for i in range(1, 7)]
G_TCP_Z = 0.0836
ap = argparse.ArgumentParser()
ap.add_argument('--free', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--smooth', type=int, default=9, help='置中移動平均視窗（序號數）')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
a = ap.parse_args()

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped


def tcp_poses(run):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=os.path.join(run, 'bag'), storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    out = []
    while rd.has_next():
        t, d, _ = rd.read_next()
        if t != '/manip/tcp_pose':
            continue
        m = deserialize_message(d, PoseStamped)
        out.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                    m.pose.position.x, m.pose.position.y, m.pose.position.z,
                    m.pose.orientation.w, m.pose.orientation.x,
                    m.pose.orientation.y, m.pose.orientation.z])
    return np.array(out)


def iso(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


JF = json.load(open(os.path.join(a.free, 'sim', 'drawer_run.json')))
JB = json.load(open(os.path.join(a.base, 'sim', 'drawer_run.json')))
CASE = yaml.safe_load(open(a.cases))['cases'][JB['case']]
AXIS = np.array(CASE['force']['drawer_axis_world'], float); AXIS /= np.linalg.norm(AXIS)
AF = [e for e in JB['events'] if e['event'] == 'engage'][0]['attach_frames']
T_WG0 = iso(AL.quat_R(AF['gripper_world_rot_wxyz']), np.array(AF['gripper_world_pos']))
T_WD0 = iso(AL.quat_R(AF['drawer_world_rot_wxyz']), np.array(AF['drawer_world_pos']))
T_GD = np.linalg.inv(T_WG0) @ T_WD0
R_D0, p_D0 = T_WD0[:3, :3], T_WD0[:3, 3]

P = tcp_poses(a.free)
i = {c: k for k, c in enumerate(JF['log_cols'])}
assert len(JF['log_cols']) == len(JF['log'][0])
t = np.array([r[i['t']] for r in JF['log']])
ph = np.array([r[i['phase']] for r in JF['log']])
seq = np.array([r[i['applied_seq']] if r[i['applied_seq']] is not None else -1
                for r in JF['log']])
k = np.clip(np.searchsorted(P[:, 0], t), 0, len(P) - 1)
ok = np.abs(P[k, 0] - t) < 0.011
m = (ph == 'pull') & ok & (seq >= 0)

per = {}
for idx in np.where(m)[0]:
    row = P[k[idx]]
    R = AL.quat_R(row[4:8])
    p_G = np.array(row[1:4]) - R @ np.array([0.0, 0.0, G_TCP_Z])
    T_D = iso(R, p_G) @ T_GD
    v = T_D[:3, 3] - p_D0
    s = float(AXIS @ v)
    e = v - AXIS * s
    dR = T_D[:3, :3] @ R_D0.T
    per.setdefault(int(seq[idx]), []).append((e, dR, s))

seqs = sorted(per)
E = np.array([np.median(np.array([x[0] for x in per[q]]), axis=0) for q in seqs])
S = np.array([np.median([x[2] for x in per[q]]) for q in seqs])
# 旋轉取平均：轉成旋轉向量再平均（角度都 < 0.1°，線性化誤差可忽略）
def rotvec(R):
    c = float(np.clip((np.trace(R) - 1) / 2, -1, 1)); th = math.acos(c)
    if th < 1e-12: return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * \
        (th / (2 * math.sin(th)))
V = np.array([np.median(np.array([rotvec(x[1]) for x in per[q]]), axis=0) for q in seqs])


def smooth(A, w):
    if w <= 1: return A.copy()
    pad = w // 2
    B = np.pad(A, ((pad, pad), (0, 0)), mode='edge')
    ker = np.ones(w) / w
    return np.stack([np.convolve(B[:, j], ker, mode='valid') for j in range(A.shape[1])],
                    axis=1)


Es, Vs = smooth(E, a.smooth), smooth(V, a.smooth)
print(f'補償表：{len(seqs)} 個指令序號（{seqs[0]}–{seqs[-1]}），'
      f'置中移動平均視窗 {a.smooth}')
print(f'{"seq":>6}{"s mm":>9}{"e⊥x mm":>10}{"e⊥y mm":>10}{"e⊥z mm":>10}{"rotvec ° (x,y,z)":>28}')
for j in range(0, len(seqs), max(len(seqs) // 10, 1)):
    d = np.degrees(Vs[j])
    print(f'{seqs[j]:6d}{S[j]*1000:9.3f}{Es[j,0]*1000:10.4f}{Es[j,1]*1000:10.4f}'
          f'{Es[j,2]*1000:10.4f}   ({d[0]:+7.4f},{d[1]:+7.4f},{d[2]:+7.4f})')
print(f'\n平滑前後 e⊥z 最大差 {np.abs(Es[:,2]-E[:,2]).max()*1000:.5f} mm；'
      f'rotvec 最大差 {np.degrees(np.abs(Vs-V)).max():.6f}°')

json.dump({'schema': 'ff_comp/1',
           'free_run': os.path.basename(os.path.abspath(a.free)),
           'base_run': os.path.basename(os.path.abspath(a.base)),
           'smooth_window': a.smooth,
           'axis_world': AXIS.tolist(),
           'frame': ('世界座標；ΔR = R_ach · R_D0ᵀ，補償以 ΔR⁻¹ **左乘**命令姿態；'
                     '位置只減 e⊥（垂直滑軌分量），沿軸分量不動'),
           'applicability': ('同一模型、固定底座、同一路徑與配時、同一增益；'
                             '**本案例的致動校準，非通用方法**'),
           'seq': seqs, 's_m': S.tolist(),
           'e_perp_m': Es.tolist(), 'rotvec_rad': Vs.tolist()},
          open(a.out, 'w'), indent=1)
print(f'-> {a.out}')
