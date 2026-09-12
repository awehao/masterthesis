"""連接框架與參考路徑的**絕對**相容性檢查（離線，不跑模擬）。

第一版錯在基準取錯：它算的是

    d = T_pred(q) − T_pred(q_engage)

也就是「後續命令預測的抽屜位姿」減去「engage 命令預測的抽屜位姿」。
整條命令路徑若一直帶著固定的高度／姿態偏差，這個減法會把它**整個消掉**，
所以那些數字只支持「命令推導的抽屜運動增量接近純 y 平移」，
**不支持「命令路徑與實際滑軌相容」**。

本版改為與滑軌真正允許的位姿集合比較：

    T_pred(q)    = T_夾爪,命令(q) · T_夾爪→抽屜,連接
    T_allowed(s) = [ R_d0 , p_d0 + â s ; 0 1 ]

其中 R_d0、p_d0 取自**連接當下抽屜的實際位姿**（模擬器讀回值），
â 為滑軌軸。沿軸位置 s 自由；垂直軸的位置偏差與姿態偏差**完整保留**，
不做任何相對消去。

同時量出所用 FK 模型與 Isaac 實際模型的差異，讓小於該差異的偏差不被過度解讀。

用法：
    python3 evaluation/check_attach_compat.py --run evaluation/runs/<id>
"""
import argparse, csv, json, math, os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
GRIP = 'uflite_gripper_link'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--urdf-fk', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'),
    help='離線 FK 用的模型（9-DOF 展開檔）')
ap.add_argument('--urdf-sim', default=os.path.join(
    WS, 'evaluation/models/omni_bot_manip.urdf'),
    help='Isaac 實際載入的模型，用來量模型差異')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
a = ap.parse_args()

J = json.load(open(os.path.join(a.run, 'sim', 'drawer_run.json')))
CASE = yaml.safe_load(open(a.cases))['cases'][J['case']]
AXIS = np.array(CASE['force']['drawer_axis_world'], float)
AXIS = AXIS / np.linalg.norm(AXIS)
ev = [e for e in J['events'] if e['event'] == 'engage']
if not ev or 'attach_frames' not in ev[0]:
    print('!! 這趟沒有記錄連接框架'); sys.exit(2)
AF = ev[0]['attach_frames']
PARK = J['park']


def quat_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def iso(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def ang(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0,
        (np.trace(Ra.T @ Rb) - 1) / 2))))


# ---- 兩個模型 ----
K9 = WholeBodyKinematics.from_urdf_string(open(a.urdf_fk).read())
I9 = [K9.dof_names.index(j) for j in ARM]
K6 = WholeBodyKinematics.from_urdf_string(open(a.urdf_sim).read(), dof_names=ARM)


def grip_cmd(qa):
    """命令關節角對應的夾爪世界位姿，用離線 FK 模型（9-DOF 展開檔）。"""
    q = np.zeros(len(K9.dof_names)); q[0], q[1], q[2] = PARK
    q[I9] = np.asarray(qa, float)
    return K9.fk(q, GRIP)


def model_delta(qa):
    """同一組關節角下，兩個模型算出的夾爪位姿差（位置 m、姿態 deg）。

    兩邊都取**底盤座標**，排除停放位姿本身。
    """
    q9 = np.zeros(len(K9.dof_names)); q9[I9] = np.asarray(qa, float)
    T9 = K9.fk(q9, GRIP)
    T6 = K6.fk(np.asarray(qa, float), GRIP)
    return float(np.linalg.norm(T9[:3, 3] - T6[:3, 3])), ang(T9[:3, :3], T6[:3, :3])


# ---- 連接鎖住的相對位姿：兩端都取**實際**位姿 ----
T_wg0_act = iso(quat_R(AF['gripper_world_rot_wxyz']), np.array(AF['gripper_world_pos']))
T_wd0_act = iso(quat_R(AF['drawer_world_rot_wxyz']), np.array(AF['drawer_world_pos']))
T_gd = np.linalg.inv(T_wg0_act) @ T_wd0_act
R_d0 = T_wd0_act[:3, :3]
p_d0 = T_wd0_act[:3, 3]

rows = [r for r in csv.DictReader(open(os.path.join(a.run, 'traj', 'traj.csv')))]
q_eng = next(([float(r[f'joint{i}']) for i in range(1, 7)]
              for r in rows if r['event'] == 'engage'), None)
pull = [r for r in rows if r['phase'] == 'pull']

print(f'趟次 {os.path.basename(a.run)}   案例 {J["case"]}')
print(f'離線 FK 模型 : {os.path.basename(a.urdf_fk)}')
print(f'Isaac 實際模型: {os.path.basename(a.urdf_sim)}')
print(f'滑軌軸 â = {AXIS.tolist()}（世界座標）')
print(f'連接於 sim {ev[0]["sim_t"]:.3f} s')
print(f'  抽屜實際位姿（基準）位置 {np.round(p_d0,6).tolist()}  '
      f'姿態 wxyz {np.round(AF["drawer_world_rot_wxyz"],6).tolist()}')

# ---- 模型差異 ----
md_p, md_a = [], []
for r in ([rows[0]] + pull[::max(len(pull)//40, 1)]):
    qa = [float(r[f'joint{i}']) for i in range(1, 7)]
    dp, da = model_delta(qa)
    md_p.append(dp); md_a.append(da)
MDP, MDA = max(md_p), max(md_a)
print(f'\n模型差異（同一組關節角，兩個 URDF 的夾爪位姿差，取樣 {len(md_p)} 點）：')
print(f'  位置 max {MDP*1000:.4f} mm     姿態 max {MDA:.5f}°')
print(f'  ⇒ 小於 {MDP*1000:.3f} mm 的位置偏差**不能與模型差異區分**。')

# ---- engage 當下：命令 FK 的夾爪位姿 vs 實際 ----
T_cmd0 = grip_cmd(q_eng)
dv0 = T_cmd0[:3, 3] - T_wg0_act[:3, 3]
s0 = float(AXIS @ dv0)
perp0 = dv0 - AXIS * s0
print(f'\nengage 當下，命令 FK 的夾爪位姿 vs 實際位姿：')
print(f'  位置差向量 {np.round(dv0*1000,4).tolist()} mm   模長 {np.linalg.norm(dv0)*1000:.4f} mm')
print(f'    沿軸 {s0*1000:+.4f} mm   **垂直軸 {np.linalg.norm(perp0)*1000:.4f} mm**')
print(f'  姿態差 {ang(T_cmd0[:3,:3], T_wg0_act[:3,:3]):.5f}°')

# ---- 絕對相容性：T_pred(q) vs 滑軌允許集合 ----
print(f'\n絕對相容性（基準 = 連接當下抽屜的實際位姿，**未做相對消去**）：')
print(f'{"開度指令mm":>11}{"沿軸 s mm":>12}{"橫向偏差 mm":>13}{"  橫向分量 (x,y,z) mm":>26}{"姿態誤差 deg":>13}')
res = []
sel = [rows.index(next(r for r in rows if r['event'] == 'engage'))]
for k, r in enumerate(pull):
    qa = [float(r[f'joint{i}']) for i in range(1, 7)]
    T_pred = grip_cmd(qa) @ T_gd
    v = T_pred[:3, 3] - p_d0
    s = float(AXIS @ v)
    perp = v - AXIS * s
    pn = float(np.linalg.norm(perp))
    ra = ang(T_pred[:3, :3], R_d0)
    res.append((float(r['expected_opening']) * 1000, s * 1000, pn * 1000,
                perp * 1000, ra))
    if k % max(len(pull) // 8, 1) == 0 or k == len(pull) - 1:
        print(f'{res[-1][0]:11.2f}{s*1000:12.3f}{pn*1000:13.4f}'
              f'   ({perp[0]*1000:+7.4f},{perp[1]*1000:+7.4f},{perp[2]*1000:+7.4f})'
              f'{ra:13.5f}')
# engage 那一點
T_pred_e = grip_cmd(q_eng) @ T_gd
ve = T_pred_e[:3, 3] - p_d0
se = float(AXIS @ ve); perpe = ve - AXIS * se
rae = ang(T_pred_e[:3, :3], R_d0)
print(f'\nengage 設定點本身：沿軸 {se*1000:+.3f} mm，'
      f'**橫向偏差 {np.linalg.norm(perpe)*1000:.4f} mm**'
      f'（{perpe[0]*1000:+.4f}, {perpe[1]*1000:+.4f}, {perpe[2]*1000:+.4f}），'
      f'姿態誤差 {rae:.5f}°')

pn_all = np.array([r[2] for r in res]); ra_all = np.array([r[4] for r in res])
print(f'\n拉開全段（{len(res)} 點）：')
print(f'  橫向偏差  中位 {np.median(pn_all):.4f}  min {pn_all.min():.4f}  '
      f'max {pn_all.max():.4f} mm')
print(f'  姿態誤差  中位 {np.median(ra_all):.5f}  max {ra_all.max():.5f}°')
print(f'  變化量（max − min）橫向 {pn_all.max()-pn_all.min():.4f} mm，'
      f'姿態 {ra_all.max()-ra_all.min():.5f}°')
verdict = ('固定偏差存在：橫向偏差全段維持在同一量級且遠大於模型差異'
           if pn_all.min() > MDP * 1000 * 2 else
           '橫向偏差與模型差異同量級，無法據此斷定有固定偏差')
print(f'\n判讀：{verdict}')
print(f'（沿軸位置 s 自由，不列入不相容；模型差異上界 {MDP*1000:.3f} mm 已標明。）')
json.dump({'schema': 'attach_compat/2', 'run': os.path.basename(a.run),
           'fk_model': os.path.basename(a.urdf_fk),
           'sim_model': os.path.basename(a.urdf_sim),
           'model_delta_pos_max_m': MDP, 'model_delta_rot_max_deg': MDA,
           'axis_world': AXIS.tolist(),
           'drawer_ref_pos': p_d0.tolist(),
           'drawer_ref_rot_wxyz': AF['drawer_world_rot_wxyz'],
           'engage_cmd_vs_actual_gripper': {
               'delta_m': dv0.tolist(), 'along_axis_m': s0,
               'perp_m': float(np.linalg.norm(perp0)),
               'rot_deg': ang(T_cmd0[:3, :3], T_wg0_act[:3, :3])},
           'engage_setpoint': {'s_m': se, 'perp_m': float(np.linalg.norm(perpe)),
                               'perp_vec_m': perpe.tolist(), 'rot_deg': rae},
           'pull': {'perp_median_mm': float(np.median(pn_all)),
                    'perp_min_mm': float(pn_all.min()),
                    'perp_max_mm': float(pn_all.max()),
                    'rot_median_deg': float(np.median(ra_all)),
                    'rot_max_deg': float(ra_all.max())},
           'trace': [[r[0], r[1], r[2], list(r[3]), r[4]] for r in res]},
          open(os.path.join(a.run, 'attach_compat.json'), 'w'),
          ensure_ascii=False, indent=2)
print(f'-> {os.path.join(a.run, "attach_compat.json")}')
