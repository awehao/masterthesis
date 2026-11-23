"""長行程的負載分量與位姿誤差分析（**純離線**，不跑模擬）。

回答三件事：

  A 按開度分解世界座標的 F_x, F_y, F_z，以及沿滑軌／垂直滑軌分量
    —— 只看模長分不出是哪個方向在累積負載。

  B 把力矩換算到把手點，明確標示為**等效力矩**：
        F_H = F_O ,  M_H = M_O − (p_H − p_O) × F_O
    換點只改力矩、**力的模長不變**；而且這只是同一反作用力系的換點表示，
    未處理中間連桿的重量、慣性與其他作用力，**不能稱為「把手接觸力」**。

  C 幾何參考 q_geo(s)、帶偏置命令 q_cmd(s) = q_geo(s) + b0、實際位姿三者的
    **位置與完整姿態誤差**隨開度的變化。
    b0 固定，但其笛卡兒影響一般隨構形改變：δx(s) ≈ J(q_geo(s)) b0。
    **20 mm 通過不代表同一偏置適用於 200 mm。**

用法：
    python3 evaluation/analyze_drawer_load.py --run evaluation/runs/<id> --traj <traj.csv>
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
import drawer_align as AL                                                # noqa
import drawer_asset as DA                                                # noqa
import yaml                                                              # noqa

ARM = [f'joint{i}' for i in range(1, 7)]
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--traj', default='')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
ap.add_argument('--out', default='')
a = ap.parse_args()

J = json.load(open(os.path.join(a.run, 'sim', 'drawer_run.json')))
CASE = yaml.safe_load(open(a.cases))['cases'][J['case']]
SPEC = DA.load(a.spec)
AXIS = np.array(CASE['force']['drawer_axis_world'], float)
AXIS /= np.linalg.norm(AXIS)
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
PARK = J['park']
traj_p = a.traj or os.path.join(a.run, 'traj', 'traj.csv')
meta_p = os.path.join(os.path.dirname(traj_p), 'traj_meta.json')
META = json.load(open(meta_p))
HO = META['handover']
OFFS = np.array(HO['offset_rad']) if HO and HO.get('mode') == 'offset' else np.zeros(6)

K = WholeBodyKinematics.from_urdf_string(open(a.urdf).read())
IDX = [K.dof_names.index(j) for j in ARM]


def qfull(qa):
    q = np.zeros(len(K.dof_names)); q[0], q[1], q[2] = PARK
    q[IDX] = np.asarray(qa, float); return q


# ---- bag：/manip/contact 的世界座標力分量（payload[0] 為模擬時間）----
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float64MultiArray
rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=os.path.join(a.run, 'bag'), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''))
contact = []
while rd.has_next():
    topic, data, _ = rd.read_next()
    if topic == '/manip/contact':
        m = deserialize_message(data, Float64MultiArray)
        contact.append(list(m.data))
C = np.array(contact)
print(f'讀入 /manip/contact {len(C)} 則（欄位：t, Fx, Fy, Fz, |F|, F_pull, '
      f'fcx, fcy, fcz, |fc|, F̂_drawer）')

i = {c: k for k, c in enumerate(J['log_cols'])}
Lg = J['log']
lt = np.array([r[i['t']] for r in Lg])
lop = np.array([r[i['opening']] for r in Lg])
lph = np.array([r[i['phase']] for r in Lg])
lseq = np.array([r[i['applied_seq']] if r[i['applied_seq']] is not None else -1
                 for r in Lg])
lq = np.array([[r[i[j]] for j in ARM] for r in Lg])
ltq = np.array([[r[i[c]] for c in ('tq_x', 'tq_y', 'tq_z')] for r in Lg])

rows = {int(r['seq']): np.array([float(r[j]) for j in ARM])
        for r in csv.DictReader(open(traj_p))}

# 以模擬時間把 contact 對到 log
idxc = np.searchsorted(C[:, 0], lt)
idxc = np.clip(idxc, 0, len(C) - 1)
ok = np.abs(C[idxc, 0] - lt) < 0.011
F = C[idxc, 1:4]

sel = (lph == 'pull') & ok & (lseq >= 0)
print(f'pull 段可用樣本 {sel.sum()} / {(lph=="pull").sum()}')

op = lop[sel] * 1000.0
Fs = F[sel]
Fpar = Fs @ AXIS
Fperp = Fs - np.outer(Fpar, AXIS)
Fperp_n = np.linalg.norm(Fperp, axis=1)
Fn = np.linalg.norm(Fs, axis=1)

# ---- B 力矩換點：M_H = M_O − (p_H − p_O) × F_O ----
M_O = ltq[sel]
bar = SPEC['drawer']['handle']['bar']['center']
M_H, pOH = [], []
for k, (qa, o) in enumerate(zip(lq[sel], lop[sel])):
    p_O = K.fk(qfull(qa), 'link6')[:3, 3]
    p_H = np.array([bar[0] + POSE[0], bar[1] - o + POSE[1], bar[2]])
    d = p_H - p_O
    M_H.append(M_O[k] - np.cross(d, Fs[k]))
    pOH.append(np.linalg.norm(d))
M_H = np.array(M_H); pOH = np.array(pOH)

# ---- C 位姿誤差：幾何參考 / 帶偏置命令 / 實際 ----
q_cmd = np.array([rows[s] for s in lseq[sel]])
q_geo = q_cmd - OFFS
q_act = lq[sel]


def pose(q):
    T = K.fk(qfull(q), 'uflite_gripper_link'); return T[:3, 3], T[:3, :3]


def perr(qa, qb):
    pa, Ra = pose(qa); pb, Rb = pose(qb)
    return float(np.linalg.norm(pa - pb)), AL.ang_deg(Ra, Rb)


e_cmd_act = np.array([perr(q_cmd[k], q_act[k]) for k in range(len(op))])
e_geo_act = np.array([perr(q_geo[k], q_act[k]) for k in range(len(op))])
e_geo_cmd = np.array([perr(q_geo[k], q_cmd[k]) for k in range(len(op))])

print(f'\n=== A 世界座標力分量 vs 開度 ===')
print(f'{"開度 mm":>10}{"Fx":>9}{"Fy":>9}{"Fz":>9}{"|F|":>9}'
      f'{"沿軸 F·â":>11}{"垂直 |F⊥|":>11}')
for lo in range(0, 100, 10):
    m = (op >= lo) & (op < lo + 10)
    if m.sum() < 2: continue
    print(f'{lo:4d}–{lo+10:<5d}{np.median(Fs[m,0]):9.2f}{np.median(Fs[m,1]):9.2f}'
          f'{np.median(Fs[m,2]):9.2f}{np.median(Fn[m]):9.2f}'
          f'{np.median(Fpar[m]):11.2f}{np.median(Fperp_n[m]):11.2f}')
for lbl, v in (('Fx', Fs[:, 0]), ('Fy', Fs[:, 1]), ('Fz', Fs[:, 2]),
               ('沿軸 F·â', Fpar), ('垂直 |F⊥|', Fperp_n), ('|F|', Fn)):
    print(f'  {lbl:<10} 與開度相關 {np.corrcoef(v, op)[0,1]:+.4f}   '
          f'範圍 {v.min():+8.2f} … {v.max():+8.2f} N')

print(f'\n=== B 力矩換點（等效力矩，非把手接觸力）===')
print(f'{"開度 mm":>10}{"|M_O| (link6)":>15}{"|M_H| (把手)":>15}{"|p_H−p_O| m":>14}')
for lo in range(0, 100, 10):
    m = (op >= lo) & (op < lo + 10)
    if m.sum() < 2: continue
    print(f'{lo:4d}–{lo+10:<5d}{np.median(np.linalg.norm(M_O[m],axis=1)):15.3f}'
          f'{np.median(np.linalg.norm(M_H[m],axis=1)):15.3f}{np.median(pOH[m]):14.4f}')
print(f'  |M_O| 相關 {np.corrcoef(np.linalg.norm(M_O,axis=1),op)[0,1]:+.4f}；'
      f'|M_H| 相關 {np.corrcoef(np.linalg.norm(M_H,axis=1),op)[0,1]:+.4f}')
print(f'  力臂 |p_H−p_O| 由 {pOH.min():.4f} 變到 {pOH.max():.4f} m'
      f'（變化 {(pOH.max()-pOH.min())*1000:.2f} mm）')
print(f'  **換點只改力矩；|F| 不變**（同一組 F 用於兩個參考點）')

print(f'\n=== C 位姿誤差 vs 開度（夾爪連桿）===')
print(f'{"開度 mm":>10}{"命令−實際 mm":>15}{"命令−實際 °":>13}'
      f'{"幾何−實際 mm":>15}{"幾何−實際 °":>13}{"幾何−命令 mm":>15}{"幾何−命令 °":>13}')
for lo in range(0, 100, 10):
    m = (op >= lo) & (op < lo + 10)
    if m.sum() < 2: continue
    print(f'{lo:4d}–{lo+10:<5d}{np.median(e_cmd_act[m,0])*1000:15.4f}'
          f'{np.median(e_cmd_act[m,1]):13.5f}{np.median(e_geo_act[m,0])*1000:15.4f}'
          f'{np.median(e_geo_act[m,1]):13.5f}{np.median(e_geo_cmd[m,0])*1000:15.4f}'
          f'{np.median(e_geo_cmd[m,1]):13.5f}')
for lbl, v in (('命令−實際 位置', e_cmd_act[:,0]*1000), ('命令−實際 姿態', e_cmd_act[:,1]),
               ('幾何−命令 位置', e_geo_cmd[:,0]*1000), ('幾何−命令 姿態', e_geo_cmd[:,1])):
    print(f'  {lbl:<16} 相關 {np.corrcoef(v,op)[0,1]:+.4f}  '
          f'{v.min():.4f} … {v.max():.4f}')

out = a.out or os.path.join(a.run, 'load_analysis')
os.makedirs(out, exist_ok=True)
json.dump({'schema': 'drawer_load/1', 'run': os.path.basename(os.path.abspath(a.run)),
           'axis_world': AXIS.tolist(), 'offset_rad': OFFS.tolist(),
           'n_samples': int(sel.sum()),
           'opening_mm': op.tolist(),
           'F_world': Fs.tolist(), 'F_par': Fpar.tolist(),
           'F_perp_norm': Fperp_n.tolist(),
           'M_O': M_O.tolist(), 'M_H': M_H.tolist(), 'lever_m': pOH.tolist(),
           'e_cmd_act': e_cmd_act.tolist(), 'e_geo_act': e_geo_act.tolist(),
           'e_geo_cmd': e_geo_cmd.tolist(),
           'note': ('M_H 為同一反作用力系換到把手點的**等效力矩**；'
                    '未扣除中間連桿重量與慣性，不是把手接觸力。'
                    '換點不改變 |F|。')},
          open(os.path.join(out, 'load_analysis.json'), 'w'))
print(f'\n-> {os.path.join(out, "load_analysis.json")}')
