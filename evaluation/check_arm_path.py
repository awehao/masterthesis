"""預抓取軌跡的**沿途**碰撞檢查：自碰 ＋ 環境，底盤固定。

為什麼不能只驗終點：終點無碰撞不代表整段運動安全。arm_initial_pose.yaml 自己
就記著這件事的證據 —— 從 all-zeros 出發，joint3 往負向 0.05 rad 內就會相交，
往正向則開到 20 mm。一個只檢查兩端的流程會放行這種軌跡。

也不是檢查直線內插：實際執行的是有速度與加速度上限的軌跡，所以這裡先產生
**要執行的那條軌跡**，再逐點檢查它。

用法：
    python3 evaluation/check_arm_path.py --case box12_south
"""
import argparse, math, os, re, sys
import numpy as np, yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics, DOF_NAMES
from verify_self_collision import link_clouds, DESIGNED_CONTACT
from arm_traj import trapezoid   # 與 play_arm_traj.py 共用，不各自實作

ARM = [f'joint{i}' for i in range(1, 7)]


def static_boxes(world_path, exclude=()):
    s = re.sub(r'<!--.*?-->', '', open(world_path).read(), flags=re.S)
    out = []
    for m in re.finditer(r'<model name="(known_obs_\d+)">(.*?)</model>', s, re.S):
        n, b = m.group(1), m.group(2)
        if n in exclude:
            continue
        p = [float(v) for v in re.search(r'<pose>([^<]+)</pose>', b).group(1).split()]
        sz = [float(v) for v in re.search(r'<box>\s*<size>([^<]+)</size>', b).group(1).split()]
        out.append((n, np.array(p[:3]), np.array(sz)))
    return out


def box_dist(P, c, sz):
    """點雲到軸對齊方塊表面的最短距離（方塊內為負，取最深）。"""
    d = np.abs(P - c) - sz / 2.0
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(d.max(axis=1), 0.0)
    return outside + inside


ap = argparse.ArgumentParser()
ap.add_argument('--case', default='')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--urdf', default='')
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf'))
# 門檻對齊專案既有數字，不自訂：
#   0.020 m  verify_self_collision.py 的 MARGIN_WARN（舒適線）
#   0.005 m  §10.9 記載的規劃器自碰拒絕門檻
# 兩者都不是「軌跡驗收」的正式門檻——本檔只如實標示落在哪一段。
ap.add_argument('--warn', type=float, default=0.020, help='舒適線 m（低於此標注）')
ap.add_argument('--hard', type=float, default=0.005, help='規劃器拒絕門檻 m')
a = ap.parse_args()

CASES = yaml.safe_load(open(a.cases))
case = CASES['cases'][a.case or CASES['default_case']]
POSES = yaml.safe_load(open(a.poses))
q0a = np.array([float(POSES[case['pregrasp']['start_config']][j]) for j in ARM])
q1a = np.array([float(POSES[case['pregrasp']['arm_config']][j]) for j in ARM])
tr = case['trajectory']

urdf = a.urdf
if not urdf:
    import subprocess
    urdf = os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf')
    src = os.path.join(WS, 'src/my_omnibot_description/urdf/omni_bot_wholebody.urdf.xacro')
    open(urdf, 'w').write(subprocess.run(['xacro', src], capture_output=True,
                                         text=True, check=True).stdout)
xml = open(urdf).read()
K = WholeBodyKinematics.from_urdf_string(xml)
clouds = link_clouds(xml)

# 配對集合：與 verify_self_collision 相同的排除規則（相鄰、同剛體群、設計接觸）
adj, rigid = set(), {}
def find(x):
    while rigid.get(x, x) != x:
        x = rigid[x]
    return x
for j in K.joints.values():
    adj.add(frozenset((j.parent, j.child)))
    if j.jtype not in ('revolute', 'prismatic', 'continuous'):
        a_, b_ = find(j.parent), find(j.child)
        if a_ != b_:
            rigid[a_] = b_
names = [n for n in clouds if n in K.parent_of or n == 'base_link']
pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
         if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT
         and find(x) != find(y)]

Q, T = trapezoid(q0a, q1a, tr['joint_vel_max_rps'], tr['joint_acc_max_rps2'],
                 tr['rate_hz'])
px, py = case['parking']['x'], case['parking']['y']
th = math.radians(case['parking']['yaw_deg'])
boxes = static_boxes(a.world)
obj = case['object']['name']

print(f'案例 {a.case or CASES["default_case"]}   URDF {os.path.basename(urdf)}')
print(f'  起點 {case["pregrasp"]["start_config"]} -> 終點 {case["pregrasp"]["arm_config"]}')
print(f'  每關節變化量 (rad): {np.round(q1a - q0a, 4).tolist()}')
print(f'  軌跡 {len(Q)} 點，時長 {T:.2f} s（v≤{tr["joint_vel_max_rps"]} rad/s，'
      f'a≤{tr["joint_acc_max_rps2"]} rad/s²，{tr["rate_hz"]:.0f} Hz）')
print(f'  底盤固定於 ({px:.3f}, {py:.3f})，車頭 {case["parking"]["yaw_deg"]:.1f}°')
print(f'  {len(clouds)} 個碰撞幾何，{len(pairs)} 對自碰組合，'
      f'{len(boxes)} 個靜態方塊（含目標 {obj}）\n')

worst_self, worst_self_at = 9.9, None
worst_env, worst_env_at = 9.9, None
rows = []
for i, qa in enumerate(Q):
    q = np.zeros(9); q[0], q[1], q[2] = px, py, th
    q[3:9] = qa
    world = {}
    for n, P in clouds.items():
        if n not in K.parent_of and n != 'base_link':
            continue
        Tm = K.fk(q, n)
        world[n] = (Tm[:3, :3] @ P.T).T + Tm[:3, 3]
    ds, who = 9.9, None
    for x, y in pairs:
        if x in world and y in world:
            d = cKDTree(world[x]).query(world[y], k=1)[0].min()
            if d < ds: ds, who = d, (x, y)
    de, whoe = 9.9, None
    for n, P in world.items():
        if n in ('base_link',) or n.startswith('rim') or n.startswith('roller'):
            continue                     # 底盤與輪子不是本檢查的對象
        for bn, c, sz in boxes:
            d = box_dist(P, c, sz).min()
            if d < de: de, whoe = d, (n, bn)
    rows.append((i * T / max(len(Q) - 1, 1), ds, de))
    if ds < worst_self: worst_self, worst_self_at = ds, (i, who)
    if de < worst_env: worst_env, worst_env_at = de, (i, whoe)

def flag(d):
    if d <= 0:       return '✗ 碰撞'
    if d < a.hard:   return f'✗ 低於 {a.hard*1000:.0f} mm 規劃器門檻'
    if d < a.warn:   return f'⚠ 介於 {a.hard*1000:.0f}–{a.warn*1000:.0f} mm'
    return '✓'

print(f'  沿途最小自碰餘裕  {worst_self*1000:8.2f} mm  {flag(worst_self)}'
      f'   第 {worst_self_at[0]}/{len(Q)-1} 點  {worst_self_at[1]}')
print(f'  沿途最小環境餘裕  {worst_env*1000:8.2f} mm  {flag(worst_env)}'
      f'   第 {worst_env_at[0]}/{len(Q)-1} 點  {worst_env_at[1]}')
print(f'\n  {"t (s)":>7}{"自碰 mm":>10}{"環境 mm":>10}')
step = max(len(rows) // 12, 1)
for t, ds, de in rows[::step] + [rows[-1]]:
    print(f'  {t:7.2f}{ds*1000:10.2f}{de*1000:10.2f}')

ok = worst_self > a.hard and worst_env > a.hard
print(f'\n  判定：{"通過（沿途皆高於規劃器門檻）" if ok else "**未通過**"}')
print(f'  幾何與算法：URDF <collision> 的 mesh 頂點與基本形取樣點雲，'
      f'每 link ≤900 點；以控制器自己的 FK 轉到世界座標後做 KD-tree 最近鄰。')
print(f'  **{len(Q)} 個離散取樣點，不是連續碰撞證明**；取樣間隔 '
      f'{T/max(len(Q)-1,1)*1000:.1f} ms。')
sys.exit(0 if ok else 1)
