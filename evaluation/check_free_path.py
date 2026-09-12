"""無連接試驗的**實際路徑**是否符合滑軌（離線，不跑模擬）。

沿用基準趟連接當下的實際抓取關係 ᴳT_D，用**無連接試驗量到的實際夾爪位姿**推算

    ᵂT_D,pred(t) = ᵂT_G,free(t) · ᴳT_D

再與滑軌允許的位姿集合比較。回答的是：
「手臂自己走這條路時，本來想把抽屜帶到哪裡？」

位姿來源（依序優先）：
  1. bag 的 /manip/tcp_pose —— **模擬器 stage 讀出的實際 TCP 位姿**。
     夾爪位姿由固定變換 ᴳT_TCP = 平移(0,0,0.0836) 精確反推，不經 FK。
  2. 由量測關節角重建的 FK —— 只作交叉核對，並標明已知的
     stage 與 FK 差（該趟記錄的 stage_vs_fk_err）。

**不只比較誤差模長**：垂直滑軌的位置誤差**逐分量**列出，姿態誤差取完整旋轉夾角。
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
import drawer_align as AL                                            # noqa
import yaml                                                          # noqa
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa

ARM = [f'joint{i}' for i in range(1, 7)]
G_TCP_Z = 0.0836

ap = argparse.ArgumentParser()
ap.add_argument('--free', required=True, help='無連接試驗目錄')
ap.add_argument('--base', required=True, help='有固定連接的基準試驗目錄')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
a = ap.parse_args()

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped


def tcp_poses(run):
    """bag 的 /manip/tcp_pose：(sim_t, p(3), q_wxyz(4))，stage 讀出的實際值。"""
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=os.path.join(run, 'bag'), storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    out = []
    while rd.has_next():
        topic, data, _ = rd.read_next()
        if topic != '/manip/tcp_pose':
            continue
        m = deserialize_message(data, PoseStamped)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p = m.pose.position; o = m.pose.orientation
        out.append([t, p.x, p.y, p.z, o.w, o.x, o.y, o.z])
    return np.array(out)


def iso(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def grip_from_tcp(row):
    """ᵂT_G = ᵂT_TCP · (ᴳT_TCP)⁻¹ —— 沿 TCP 自身 z 軸退 0.0836 m。"""
    R = AL.quat_R(row[4:8])
    p = np.array(row[1:4]) - R @ np.array([0.0, 0.0, G_TCP_Z])
    return iso(R, p)


JF = json.load(open(os.path.join(a.free, 'sim', 'drawer_run.json')))
JB = json.load(open(os.path.join(a.base, 'sim', 'drawer_run.json')))
CASE = yaml.safe_load(open(a.cases))['cases'][JB['case']]
AXIS = np.array(CASE['force']['drawer_axis_world'], float); AXIS /= np.linalg.norm(AXIS)

# 抓取關係與滑軌基準：**取自基準趟**連接當下的實際位姿
AF = [e for e in JB['events'] if e['event'] == 'engage'][0]['attach_frames']
T_WG0 = iso(AL.quat_R(AF['gripper_world_rot_wxyz']), np.array(AF['gripper_world_pos']))
T_WD0 = iso(AL.quat_R(AF['drawer_world_rot_wxyz']), np.array(AF['drawer_world_pos']))
T_GD = np.linalg.inv(T_WG0) @ T_WD0
R_D0, p_D0 = T_WD0[:3, :3], T_WD0[:3, 3]

PF, PB = tcp_poses(a.free), tcp_poses(a.base)
print(f'無連接 /manip/tcp_pose {len(PF)} 則；基準 {len(PB)} 則')


def joinlog(J, P):
    i = {c: k for k, c in enumerate(J['log_cols'])}
    assert len(J['log_cols']) == len(J['log'][0])
    t = np.array([r[i['t']] for r in J['log']])
    ph = np.array([r[i['phase']] for r in J['log']])
    seq = np.array([r[i['applied_seq']] if r[i['applied_seq']] is not None else -1
                    for r in J['log']])
    q = np.array([[r[i[x]] for x in ARM] for r in J['log']])
    sfk = np.array([r[i['stage_vs_fk_err']] for r in J['log']])
    k = np.clip(np.searchsorted(P[:, 0], t), 0, len(P) - 1)
    ok = np.abs(P[k, 0] - t) < 0.011
    return t, ph, seq, q, sfk, P[k], ok


tF, phF, sqF, qF, sfkF, poF, okF = joinlog(JF, PF)
tB, phB, sqB, qB, sfkB, poB, okB = joinlog(JB, PB)
mF = (phF == 'pull') & okF & (sqF >= 0)
mB = (phB == 'pull') & okB & (sqB >= 0)
print(f'pull 段可用：無連接 {mF.sum()}，基準 {mB.sum()}')

rows = {int(r['seq']): float(r['expected_opening']) * 1000
        for r in csv.DictReader(open(os.path.join(a.free, 'traj', 'traj.csv')))}
cmdF = np.array([rows.get(int(s), np.nan) for s in sqF])

print(f'\nstage 與 FK 的已知差（本趟記錄）：無連接 中位 '
      f'{np.median(sfkF[mF])*1000:.4f} max {sfkF[mF].max()*1000:.4f} mm；'
      f'基準 中位 {np.median(sfkB[mB])*1000:.4f} max {sfkB[mB].max()*1000:.4f} mm')
print('（本分析主來源是 stage 讀出的 /manip/tcp_pose，未經 FK；此值僅標明限制）')

print(f'\n=== 無連接：實際夾爪位姿推出的抽屜位姿 vs 滑軌允許集合 ===')
print(f'{"指令開度 mm":>13}{"沿軸 s mm":>12}{"e⊥ x mm":>11}{"e⊥ y mm":>11}'
      f'{"e⊥ z mm":>11}{"|e⊥| mm":>10}{"姿態誤差 °":>12}')
res = []
for k in np.where(mF)[0]:
    T_G = grip_from_tcp(poF[k])
    T_D = T_G @ T_GD
    v = T_D[:3, 3] - p_D0
    s = float(AXIS @ v)
    e = v - AXIS * s
    ang = AL.ang_deg(T_D[:3, :3], R_D0)
    res.append([cmdF[k], s * 1000, e[0] * 1000, e[1] * 1000, e[2] * 1000,
                float(np.linalg.norm(e)) * 1000, ang])
R = np.array(res)
for lo in range(0, 80, 10):
    m = (R[:, 0] >= lo) & (R[:, 0] < lo + 10)
    if m.sum() < 2: continue
    r = np.median(R[m], axis=0)
    print(f'{lo:5d}–{lo+10:<6d}{r[1]:12.3f}{r[2]:11.4f}{r[3]:11.4f}{r[4]:11.4f}'
          f'{r[5]:10.4f}{r[6]:12.5f}')
print(f'\n  沿軸 s：{R[:,1].min():.3f} … {R[:,1].max():.3f} mm'
      f'（**手臂自己走出來的等效開度**）')
for j, nm in ((2, 'e⊥ x'), (3, 'e⊥ y'), (4, 'e⊥ z'), (5, '|e⊥|'), (6, '姿態 °')):
    print(f'  {nm:<8} 中位 {np.median(R[:,j]):+9.4f}  範圍 {R[:,j].min():+9.4f} … '
          f'{R[:,j].max():+9.4f}  與 s 相關 {np.corrcoef(R[:,j],R[:,1])[0,1]:+.4f}')

print(f'\n=== 相同指令序號下，兩趟的**實際**夾爪位姿差 ===')
bseq = {int(s): k for k, s in enumerate(sqB) if mB[k]}
print(f'{"指令開度 mm":>13}{"Δ位置 mm":>11}{"Δx":>9}{"Δy":>9}{"Δz":>9}{"Δ姿態 °":>11}')
dd = []
for k in np.where(mF)[0]:
    s = int(sqF[k])
    if s not in bseq: continue
    kb = bseq[s]
    TF_ = grip_from_tcp(poF[k]); TB_ = grip_from_tcp(poB[kb])
    d = TF_[:3, 3] - TB_[:3, 3]
    dd.append([cmdF[k], np.linalg.norm(d) * 1000, d[0] * 1000, d[1] * 1000,
               d[2] * 1000, AL.ang_deg(TF_[:3, :3], TB_[:3, :3])])
DD = np.array(dd)
for lo in range(0, 80, 10):
    m = (DD[:, 0] >= lo) & (DD[:, 0] < lo + 10)
    if m.sum() < 2: continue
    r = np.median(DD[m], axis=0)
    print(f'{lo:5d}–{lo+10:<6d}{r[1]:11.4f}{r[2]:9.4f}{r[3]:9.4f}{r[4]:9.4f}{r[5]:11.5f}')
print(f'  Δ位置模長 中位 {np.median(DD[:,1]):.4f} max {DD[:,1].max():.4f} mm；'
      f'Δ姿態 中位 {np.median(DD[:,5]):.5f} max {DD[:,5].max():.5f}°')
print(f'  Δz 中位 {np.median(DD[:,4]):+.4f} mm，與指令開度相關 '
      f'{np.corrcoef(DD[:,4],DD[:,0])[0,1]:+.4f}')

out = os.path.join(a.free, 'free_path.json')
json.dump({'schema': 'free_path/1', 'free': os.path.basename(os.path.abspath(a.free)),
           'base': os.path.basename(os.path.abspath(a.base)),
           'pose_source': '/manip/tcp_pose（stage 讀出），夾爪由固定變換 0.0836 m 反推',
           'stage_vs_fk_err_free_max_m': float(sfkF[mF].max()),
           'axis_world': AXIS.tolist(), 'T_GD': T_GD.tolist(),
           'cols_res': ['cmd_open_mm', 's_mm', 'ex_mm', 'ey_mm', 'ez_mm',
                        'eperp_mm', 'att_deg'], 'res': R.tolist(),
           'cols_dd': ['cmd_open_mm', 'dpos_mm', 'dx', 'dy', 'dz', 'datt_deg'],
           'dd': DD.tolist()}, open(out, 'w'))
print(f'\n-> {out}')
