"""連接框架與參考路徑的相容性檢查（離線）。

問題：固定連接鎖住的是**連接當下的實際相對位姿** T_夾爪→抽屜。之後參考路徑
命令夾爪走一條笛卡兒直線，那條路徑透過這個鎖住的關係，會把抽屜帶到哪裡？

抽屜只有一個自由度（沿世界 −y 平移）。所以相容的必要條件是：

    T_世界→抽屜(t) = T_世界→夾爪_命令(t) · T_夾爪→抽屜

在整段路徑上都只有 y 方向的平移，x、z 位移與轉角都必須是 0。
任何不為 0 的分量都是滑動關節必須剛性抵抗的量 —— 那就是約束負載的來源。

用法：
    python3 evaluation/check_attach_compat.py --run evaluation/runs/<id>
"""
import argparse, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='evaluation/runs/<id>')
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
a = ap.parse_args()

J = json.load(open(os.path.join(a.run, 'sim', 'drawer_run.json')))
ev = [e for e in J['events'] if e['event'] == 'engage']
if not ev or 'attach_frames' not in ev[0]:
    print('!! 這趟沒有記錄連接框架（attach_frames）'); sys.exit(2)
AF = ev[0]['attach_frames']
PARK = J['park']
K = WholeBodyKinematics.from_urdf_string(open(a.urdf).read())
IDX = [K.dof_names.index(f'joint{i}') for i in range(1, 7)]
GRIP = 'uflite_gripper_link'


def quat_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def iso(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


T_wg0 = iso(quat_R(AF['gripper_world_rot_wxyz']), np.array(AF['gripper_world_pos']))
T_wd0 = iso(quat_R(AF['drawer_world_rot_wxyz']), np.array(AF['drawer_world_pos']))
T_gd = np.linalg.inv(T_wg0) @ T_wd0          # 連接鎖住的相對位姿

# 參考路徑（命令值）
rows = [r for r in csv.DictReader(open(os.path.join(a.run, 'traj', 'traj.csv')))]
pull = [r for r in rows if r['phase'] == 'pull']
print(f'趟次 {os.path.basename(a.run)}  拉開段 {len(pull)} 點')
print(f'連接於 sim {ev[0]["sim_t"]:.3f} s')
print(f'  夾爪世界位置 {np.round(AF["gripper_world_pos"],6).tolist()}')
print(f'  抽屜世界位置 {np.round(AF["drawer_world_pos"],6).tolist()}')

# FK 的夾爪位姿要與連接當下的實際位姿對齊：先量兩者的差（模型 + 追蹤落後）
def fk_grip(qa):
    q = np.zeros(len(K.dof_names)); q[0], q[1], q[2] = PARK
    q[IDX] = np.asarray(qa, float)
    return K.fk(q, GRIP)


q_eng = None
for r in rows:
    if r['event'] == 'engage':
        q_eng = [float(r[f'joint{i}']) for i in range(1, 7)]
        break
if q_eng is not None:
    T_fk0 = fk_grip(q_eng)
    d0 = T_fk0[:3, 3] - np.array(AF['gripper_world_pos'])
    ang0 = math.degrees(math.acos(max(-1.0, min(1.0,
        (np.trace(T_fk0[:3, :3].T @ T_wg0[:3, :3]) - 1) / 2))))
    print(f'  命令 FK 的夾爪位姿 vs 連接當下實際：位置差 '
          f'{np.round(d0*1000,4).tolist()} mm（模長 {np.linalg.norm(d0)*1000:.4f}），'
          f'轉角差 {ang0:.4f}°')

print(f'\n沿參考路徑推出的抽屜位姿（相對連接時刻）：')
print(f'{"開度指令mm":>11}{"Δx mm":>10}{"Δy mm":>10}{"Δz mm":>10}{"轉角 deg":>11}')
worst = {'x': 0.0, 'z': 0.0, 'ang': 0.0}
tr = []
for k, r in enumerate(pull):
    qa = [float(r[f'joint{i}']) for i in range(1, 7)]
    T_wd = fk_grip(qa) @ T_gd
    d = T_wd[:3, 3] - (fk_grip(q_eng) @ T_gd)[:3, 3]
    R0 = (fk_grip(q_eng) @ T_gd)[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0,
        (np.trace(T_wd[:3, :3].T @ R0) - 1) / 2))))
    worst['x'] = max(worst['x'], abs(d[0])); worst['z'] = max(worst['z'], abs(d[2]))
    worst['ang'] = max(worst['ang'], ang)
    tr.append([float(r['expected_opening']) * 1000, d[0] * 1000, d[1] * 1000,
               d[2] * 1000, ang])
    if k % max(len(pull) // 10, 1) == 0 or k == len(pull) - 1:
        print(f'{float(r["expected_opening"])*1000:11.2f}{d[0]*1000:10.4f}'
              f'{d[1]*1000:10.3f}{d[2]*1000:10.4f}{ang:11.5f}')
print(f'\n不相容量（滑動關節必須剛性抵抗的部分）：')
print(f'  |Δx| max {worst["x"]*1000:.4f} mm')
print(f'  |Δz| max {worst["z"]*1000:.4f} mm')
print(f'  轉角  max {worst["ang"]:.5f}°')
print(f'\n說明：抽屜只能沿世界 −y 平移。上面三個量若不為 0，就是參考路徑透過'
      f'鎖住的連接關係，要求抽屜做它做不到的運動 —— 那部分只能由滑動關節'
      f'與固定連接以內力承擔。')
json.dump({'schema': 'attach_compat/1', 'run': os.path.basename(a.run),
           'T_gd': T_gd.tolist(), 'worst': worst, 'trace': tr},
          open(os.path.join(a.run, 'attach_compat.json'), 'w'),
          ensure_ascii=False, indent=2)
