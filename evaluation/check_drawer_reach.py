"""抽屜案例的**全行程**幾何檢查：不是只確認起始把手可達。

檢查三段（底盤全程固定）：
    B 接近   預抓取 → 夾持位（沿工具 z 前進）
    D 拉開   開度 0 → 目標開度，TCP 與把手**同步**移動
    F 退出   夾持位 → 預抓取（B 的反向，開度停在目標值）

每一點都報：IK 殘差、六軸離安全限位的最小餘裕、自碰餘裕、手臂對「櫃體＋抽屜」
的餘裕。手指與把手橫桿是**設計接觸**，從環境餘裕裡排除，另外單獨列出夾爪殼
與橫桿的間隙（那個不該接觸）。

用法：
    python3 evaluation/check_drawer_reach.py --case drawer_open_a
"""
import argparse, math, os, sys
import numpy as np, yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
from ammr_wholebody_mpc.arm_pregrasp import solve_ik
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE
from verify_self_collision import link_clouds, DESIGNED_CONTACT
import drawer_asset as DA

ARM = [f'joint{i}' for i in range(1, 7)]
# 工具座標在世界的朝向（joint6 = −π/2 的分支）：
#   工具 z（接近）= 世界 +y   工具 y（手指閉合）= 世界 +z   工具 x = 世界 −x
# 手指垂直閉合，所以把手可以是常規的水平橫桿。
R_DES = np.array([[-1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0]])

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='drawer_open_a')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf'))
ap.add_argument('--n-pull', type=int, default=41, help='拉開段取樣點數')
ap.add_argument('--n-appr', type=int, default=21, help='接近段取樣點數')
# 門檻沿用專案既有數字，不另訂：0.005 規劃器拒絕、0.020 舒適線
ap.add_argument('--hard', type=float, default=0.005)
ap.add_argument('--warn', type=float, default=0.020)
# 限位餘裕門檻：arm_pregrasp 的 IK 就以 LITE6_SAFE 為界，這裡要求再留一段
ap.add_argument('--limit-margin', type=float, default=0.050)
a = ap.parse_args()

CASE = yaml.safe_load(open(a.cases))['cases'][a.case]
SPEC = DA.load(a.spec)
xml = open(a.urdf).read()
K = WholeBodyKinematics.from_urdf_string(xml)
clouds = link_clouds(xml)
IDX = [K.dof_names.index(j) for j in ARM]

POSE = (CASE['object']['pose'][0], CASE['object']['pose'][1])
PARK = (CASE['parking']['x'], CASE['parking']['y'],
        math.radians(CASE['parking']['yaw_deg']))
TARGET = float(CASE['drawer']['target_opening_m'])
Q_FINGER = float(SPEC['grasp_surface']['finger_joint_open'])
ENCL = float(SPEC['grasp_surface']['enclosure_margin_min_m'])
# 包覆餘裕的解析值：全開時單側內距 − 桿半徑。這是幾何真值，
# 不受點雲取樣密度影響。
ENCL_ACTUAL = (float(SPEC['grasp_surface']['finger_gap_open_m']) / 2.0
               - float(SPEC['drawer']['handle']['bar']['radius']))
APPROACH = float(CASE['drawer']['approach_backoff_m'])
RETREAT = float(CASE['drawer']['retreat_backoff_m'])

# 自碰配對（與 verify_self_collision / check_arm_path 同一套排除規則）
adj, rigid = set(), {}
def find(x):
    while rigid.get(x, x) != x:
        x = rigid[x]
    return x
for j in K.joints.values():
    adj.add(frozenset((j.parent, j.child)))
    if j.jtype not in ('revolute', 'prismatic', 'continuous'):
        p, c = find(j.parent), find(j.child)
        if p != c:
            rigid[p] = c
names = [n for n in clouds if n in K.parent_of or n == 'base_link']
pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
         if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT
         and find(x) != find(y)]

ARM_LINKS = [n for n in names
             if n.startswith('link') or n.startswith('uflite')]
FINGERS = ('uflite_finger1', 'uflite_finger2')


_RNG = np.random.default_rng(0)
_SEEDS = [np.array([0.0, 0.5, 1.0, 0.0, -1.0, -math.pi / 2]),
          np.array([0.0, 0.1, 0.2, 0.0, -1.5, -math.pi / 2]),
          np.array([0.0, -0.3, 0.6, 0.0, -0.4, -math.pi / 2])] + [
          _RNG.uniform(LITE6_SAFE.lower, LITE6_SAFE.upper) for _ in range(7)]


def _solve(tcp_w, seed):
    T = np.eye(4); T[:3, :3] = R_DES; T[:3, 3] = tcp_w
    q = np.zeros(9); q[0], q[1], q[2] = PARK
    q[IDX] = seed
    return solve_ik(K, q, T)


def ik_at(tcp_w, prev):
    """先沿用上一點的解（保持關節空間連續），失敗才換種子。

    單一種子沿路徑暖啟動會卡在一個 IK 分支上 —— 本案例原本記錄的可達區間
    0.350–0.610 就是這樣來的，實際上是錯的。但反過來每點獨立取「限位餘裕最大」
    的解也不行：相鄰點可能落在不同分支，接起來就是一個不能執行的跳動。
    所以規則是：能延續就延續；不能才換，並把換分支造成的關節跳動報出來。
    """
    if prev is not None:
        r = _solve(tcp_w, prev)
        qa = r.q[IDX]
        m = min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
                for k in range(6))
        if r.ok and r.pos_err < 1e-3 and m >= a.limit_margin:
            return r, False
    best, bm = None, -1.0
    for sd in _SEEDS:
        r = _solve(tcp_w, sd)
        if not (r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2):
            continue
        qa = r.q[IDX]
        m = min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
                for k in range(6))
        if m > bm:
            best, bm = r, m
    if best is None:
        return _solve(tcp_w, prev if prev is not None else _SEEDS[0]), True
    return best, True


# 手指在 URDF 的零位是**全閉**（兩片在 y = 0 相接），而 WholeBodyKinematics 的
# dof_names 不含 finger_joint，_q_of 對它回傳 0 —— 也就是 FK 一律把手指當全閉。
# 進入與退出時手指是全開的，用全閉的幾何去量就會量出假的侵入。
# 關節平移發生在子連桿座標，且關節 origin 無旋轉，所以直接把點雲沿軸平移即可。
FINGER_AXIS = {'uflite_finger1': np.array([0.0, 1.0, 0.0]),
               'uflite_finger2': np.array([0.0, -1.0, 0.0])}


def evaluate(q_full, q_d, q_finger):
    """一個姿態的所有餘裕。q_finger 為 finger_joint 位置（0 = 全閉）。

    手指與把手橫桿是**設計接觸對**，全程從環境餘裕裡排除，另以包覆餘裕單獨判定：
    一般環境的 5 mm 門檻是給「不該碰到的東西」用的，套在要夾的物體上沒有意義。
    """
    shapes = DA.shapes_world(SPEC, POSE, q_d)
    world = {}
    for n in names:
        Tm = K.fk(q_full, n)
        P = clouds[n]
        if n in FINGER_AXIS:
            P = P + FINGER_AXIS[n] * q_finger
        world[n] = (Tm[:3, :3] @ P.T).T + Tm[:3, 3]
    ds, dswho = np.inf, None
    for x, y in pairs:
        d = cKDTree(world[x]).query(world[y], k=1)[0].min()
        if d < ds: ds, dswho = float(d), f'{x}|{y}'
    de, dewho = np.inf, None
    for n in ARM_LINKS:
        for sh in shapes:
            if n in FINGERS and sh[0] == 'handle/bar':
                continue                      # 設計接觸對，另行判定
            d, _ = DA.min_distance(world[n], [sh])
            if d < de: de, dewho = d, f'{n}|{sh[0]}'
    bar = [s for s in shapes if s[0] == 'handle/bar']
    shell, _ = DA.min_distance(world['uflite_gripper_link'], bar)
    fing = min(DA.min_distance(world[f], bar)[0] for f in FINGERS)
    return ds, dswho, de, dewho, shell, fing


def margins(qa):
    return min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
               for k in range(6)), \
           int(np.argmin([min(qa[k] - LITE6_SAFE.lower[k],
                              LITE6_SAFE.upper[k] - qa[k]) for k in range(6)])) + 1


print(f'案例 {a.case}   規格 {SPEC["name"]}   URDF {os.path.basename(a.urdf)}')
print(f'  櫃體世界位置 ({POSE[0]:.3f}, {POSE[1]:.3f})，本地 +y 指向櫃內，抽屜沿 −y 拉出')
print(f'  底盤固定於 ({PARK[0]:.4f}, {PARK[1]:.4f}) yaw {CASE["parking"]["yaw_deg"]:.1f}°')
tcp0 = DA.grasp_tcp_world(SPEC, POSE, 0.0)
tcpT = DA.grasp_tcp_world(SPEC, POSE, TARGET)
print(f'  夾持 TCP 世界：關閉 {np.round(tcp0,4).tolist()} → 開 {TARGET:.3f} m '
      f'{np.round(tcpT,4).tolist()}')
print(f'  夾持 TCP 底盤 x：{tcp0[1]-PARK[1]:.4f} → {tcpT[1]-PARK[1]:.4f} m\n')

rows, fail = [], []
seed = None

segs = [('B 接近', [(t, 0.0) for t in np.linspace(-APPROACH, 0.0, a.n_appr)]),
        ('D 拉開', [(0.0, d) for d in np.linspace(0.0, TARGET, a.n_pull)]),
        ('F 退出', [(t, TARGET) for t in np.linspace(0.0, -RETREAT, a.n_appr)])]

for seg, pts in segs:
    worst = dict(ik=0.0, lm=np.inf, ds=np.inf, de=np.inf, sh=np.inf, fg=np.inf, step=-1.0)
    wat = {}
    for backoff, q_d in pts:
        tcp = DA.grasp_tcp_world(SPEC, POSE, q_d).copy()
        tcp[1] += backoff                       # 沿工具 z（世界 +y）退開
        r, switched = ik_at(tcp, seed)
        qa = r.q[IDX]
        step = 0.0 if seed is None else float(np.abs(qa - seed).max())
        if seed is not None and step > worst.get('step', -1):
            worst['step'], wat['step'] = step, (q_d, switched)
        seed = qa.copy()
        lm, jb = margins(qa)
        ds, dsw, de, dew, shell, fing = evaluate(r.q, q_d, Q_FINGER)
        rows.append((seg, backoff, q_d, r.pos_err, r.rot_err, lm, jb,
                     ds, de, shell, fing, dsw, dew))
        if r.pos_err > worst['ik']: worst['ik'] = r.pos_err
        if lm < worst['lm']: worst['lm'], wat['lm'] = lm, (q_d, jb)
        if ds < worst['ds']: worst['ds'], wat['ds'] = ds, dsw
        if de < worst['de']: worst['de'], wat['de'] = de, dew
        if shell < worst['sh']: worst['sh'], wat['sh'] = shell, q_d
        if fing < worst['fg']: worst['fg'], wat['fg'] = fing, q_d
        if not r.ok:
            fail.append((seg, backoff, q_d, 'IK 未收斂'))
    print(f'[{seg}] {len(pts)} 點')
    print(f'   IK 位置殘差最大   {worst["ik"]*1000:8.4f} mm')
    print(f'   限位餘裕最小      {worst["lm"]:8.4f} rad   '
          f'（開度 {wat["lm"][0]:.3f} m，joint{wat["lm"][1]}）')
    print(f'   自碰餘裕最小      {worst["ds"]*1000:8.2f} mm   {wat["ds"]}')
    print(f'   對櫃體／抽屜最小  {worst["de"]*1000:8.2f} mm   {wat["de"]}')
    print(f'   夾爪殼對橫桿最小  {worst["sh"]*1000:8.2f} mm   （開度 {wat["sh"]:.3f} m）')
    if worst['step'] >= 0:
        print(f'   相鄰點關節跳動最大{worst["step"]:8.4f} rad  '
              f'（開度 {wat["step"][0]:.3f} m'
              f'{"，換了 IK 分支" if wat["step"][1] else ""}）')
    if worst['lm'] < a.limit_margin: fail.append((seg, None, None, '限位餘裕不足'))
    if worst['ds'] < a.hard: fail.append((seg, None, None, '自碰低於門檻'))
    if worst['de'] < a.hard: fail.append((seg, None, None, '環境低於門檻'))
    # 取樣值只當交叉檢查：指墊內面在 900 點的取樣下可能沒有點正好落在最近處，
    # 量到的距離會**高估**。判定用解析值（見下方的包覆餘裕）。
    print(f'   手指對橫桿取樣值  {worst["fg"]*1000:8.2f} mm   '
          f'（解析值 {ENCL_ACTUAL*1000:.2f} mm；取樣會高估，僅供交叉檢查）')
    if worst['sh'] < a.hard: fail.append((seg, None, None, '夾爪殼碰到橫桿'))
    print()

# 夾持關係的幾何前提：橫桿必須整根落在指墊的 tool z 範圍內
b = SPEC['drawer']['handle']['bar']
off = float(SPEC['grasp_surface']['tcp_offset_along_tool_z'])
# 三個數字量自 URDF，各有不同用途，**不可互相代用**：
#   指墊範圍      能產生夾持力的面（指基部退進殼裡，所以起點比殼前緣更後面）
#   殼前緣        真正會先撞到橫桿的幾何
#   指尖          另一側的邊界
pad_lo, pad_hi = -0.0320, -0.0025
shell_face = -0.0269
bar_lo, bar_hi = -off - b['radius'], -off + b['radius']
print('夾持關係的幾何前提（tool z，相對 TCP）')
print(f'   指墊範圍   [{pad_lo:+.4f}, {pad_hi:+.4f}]  長 {(pad_hi-pad_lo)*1000:.1f} mm')
print(f'   可用窗口   [{shell_face:+.4f}, {pad_hi:+.4f}]  '
      f'長 {(pad_hi-shell_face)*1000:.1f} mm（殼前緣到指尖）')
print(f'   橫桿範圍   [{bar_lo:+.4f}, {bar_hi:+.4f}]  徑 {b["radius"]*2*1000:.1f} mm')
print(f'   窗口內餘裕 殼前緣側 {(bar_lo-shell_face)*1000:+.2f} mm   '
      f'指尖側 {(pad_hi-bar_hi)*1000:+.2f} mm')
print(f'   指墊涵蓋   橫桿後方還有 {(bar_lo-pad_lo)*1000:+.2f} mm 的指墊（在殼內，不接觸）')
gap = float(SPEC['grasp_surface']['finger_gap_open_m'])
print(f'   全開內距 {gap*1000:.1f} mm 對 {b["radius"]*2*1000:.1f} mm 的桿：'
      f'每側 {ENCL_ACTUAL*1000:.2f} mm  '
      f'{"✓" if ENCL_ACTUAL >= ENCL else "✗ 低於 %.1f mm 判準" % (ENCL*1000)}')
if ENCL_ACTUAL < ENCL:
    fail.append(('夾持', None, None, '手指全開時包覆餘裕不足'))
qf_touch = b['radius']
print(f'   接觸時 finger_joint = {qf_touch:.4f}（上限 {gap/2:.4f}）；'
      f'命令 0.000 時過閉合 {qf_touch*1000:.1f} mm')

print(f'\n判定：{"通過" if not fail else "**未通過**"}')
for f in fail:
    print(f'   ✗ {f[0]}  {f[3]}')
print(f'取樣 {len(rows)} 點，**離散取樣，不是連續碰撞證明**。')
sys.exit(0 if not fail else 1)
