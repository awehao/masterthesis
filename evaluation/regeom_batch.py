"""以 SDF 碰撞幾何重評本批 40 趟的距離相關指標（評估改版，不動控制與原始資料）。

舊度量：clearance = ‖p_robot − p_obs‖ − r_YAML − 0.30
        r_YAML 是軌跡設定裡供 RViz Marker 用的**外接圓**半徑。
新度量：「依 SDF 碰撞幾何與真值位姿計算的底盤圓盤最小取樣間距」
        = 點到 SDF collision 形狀聯集表面的距離 − 0.30

每趟都要通過執行期姿態檢查與資料完整性檢查；不通過就標記**不可評估**，
不退回舊公式、不略過物件。
"""
import csv, glob, hashlib, json, math, os, re, sys
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obstacle_geometry import (load_dyn_obstacles, surface_distance,
                               check_height_cover, assert_no_rotation_in_run)
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from geometry_msgs.msg import PoseStamped

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, 'evaluation/results/geom_v2')
SDF = os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf')
TRAJ = os.path.join(WS, 'src/ammr_bringup/config/'
                        'dynamic_trajectories_bigarena_traffic_v3.yaml')
R_ROB = 0.30
D_ENTER, D_CORE, MIN_N, MOVE_EPS = 1.00, 0.50, 10, 0.05
os.makedirs(OUT, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


SHAPES = load_dyn_obstacles(SDF)
YR = {d['name']: float(d['radius'])
      for d in yaml.safe_load(open(TRAJ))['dynamic_obstacles']}
for n, sh in SHAPES.items():
    ok, msg = check_height_cover(sh, 0.0, 0.45)
    if not ok:
        raise SystemExit(f'{n} 高度不覆蓋：{msg}')

version = dict(
    schema='geom_reeval/1',
    metric='依 SDF 碰撞幾何與真值位姿計算的底盤圓盤最小取樣間距',
    robot_disc_radius_m=R_ROB,
    sdf=os.path.relpath(SDF, WS), sdf_sha=sha(SDF),
    traj_yaml_sha=sha(TRAJ),
    obstacle_geometry_sha=sha(os.path.join(WS, 'evaluation/obstacle_geometry.py')),
    this_script_sha=sha(os.path.abspath(__file__)),
    event_thresholds=dict(d_enter=D_ENTER, d_core=D_CORE, min_n=MIN_N,
                          move_eps=MOVE_EPS),
    note='舊度量保留為 old_disc_*，新度量為 new_sdf_*；本檔不覆寫任何舊輸出')
json.dump(version, open(os.path.join(OUT, 'version.json'), 'w'),
          ensure_ascii=False, indent=1)
print('評估版本：')
for k in ('metric', 'sdf_sha', 'traj_yaml_sha', 'obstacle_geometry_sha',
          'this_script_sha'):
    print(f'  {k:<24} {version[k]}')

# ---- 1) 清單 ---------------------------------------------------------------
runs = []
for d in sorted(glob.glob(os.path.join(WS, 'evaluation/runs/*c10_s*'))):
    b = os.path.basename(d)
    m = re.search(r'c10_s(\d+)_(r\d)_(off|on)_', b)
    if not m:
        continue
    aborted = os.path.exists(os.path.join(d, 'ABORTED.md'))
    has_bag = os.path.exists(os.path.join(d, 'bag', 'metadata.yaml'))
    has_res = os.path.exists(os.path.join(d, 'isaac_run.json'))
    runs.append(dict(dir=d, name=b, seed=int(m.group(1)), rep=m.group(2),
                     cond=m.group(3), aborted=aborted,
                     has_bag=has_bag, has_result=has_res))
print(f'\n清單：共 {len(runs)} 趟')
print(f'  標記 ABORTED：{sum(r["aborted"] for r in runs)}')
print(f'  缺 bag：{sum(not r["has_bag"] for r in runs)}   '
      f'缺 isaac_run.json：{sum(not r["has_result"] for r in runs)}')


def load_run(d):
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.join(d, 'bag'), storage_id='mcap'),
           ConverterOptions('', ''))
    rob, obs, quats = [], {n: [] for n in SHAPES}, set()
    while r.has_next():
        tn, data, _ = r.read_next()
        if tn == '/model/omni_bot/pose':
            m = deserialize_message(data, PoseStamped); s = m.header.stamp
            rob.append((s.sec + s.nanosec * 1e-9, m.pose.position.x,
                        m.pose.position.y, yaw_of(m.pose.orientation)))
        elif tn.startswith('/model/dyn_obs_') and tn.endswith('/pose'):
            n = tn.split('/')[2]
            if n in obs:
                m = deserialize_message(data, PoseStamped); s = m.header.stamp
                q = m.pose.orientation
                quats.add((round(q.w, 4), round(q.x, 4), round(q.y, 4),
                           round(q.z, 4)))
                obs[n].append((s.sec + s.nanosec * 1e-9, m.pose.position.x,
                               m.pose.position.y))
    return np.array(rob), {k: np.array(v) for k, v in obs.items()}, quats


per, events = [], []
for r in runs:
    tag = f'{r["seed"]}/{r["rep"]}/{r["cond"]}'
    if r['aborted'] or not r['has_bag'] or not r['has_result']:
        r['evaluable'] = False
        r['reason'] = 'ABORTED' if r['aborted'] else '缺檔'
        continue
    try:
        rob, obs, quats = load_run(r['dir'])
        assert_no_rotation_in_run(quats)               # 執行期姿態檢查
        missing = [n for n, v in obs.items() if len(v) < 10]
        if missing:
            raise ValueError(f'障礙物位姿樣本不足：{missing}')
        rj = json.load(open(os.path.join(r['dir'], 'isaac_run.json')))['run']
        t0, t1 = float(rj['motion_start_sim_t']), float(rj['sim_time'])
        s = rob[(rob[:, 0] >= t0) & (rob[:, 0] <= t1)]
        if len(s) < 20:
            raise ValueError(f'任務視窗取樣不足 {len(s)}')
    except Exception as e:
        r['evaluable'] = False; r['reason'] = str(e)[:80]
        print(f'  !! {tag} 不可評估：{r["reason"]}')
        continue
    r['evaluable'] = True; r['reason'] = ''

    old_all, new_all = {}, {}
    for n, A in obs.items():
        ox = np.interp(s[:, 0], A[:, 0], A[:, 1])
        oy = np.interp(s[:, 0], A[:, 0], A[:, 2])
        old_all[n] = np.hypot(s[:, 1] - ox, s[:, 2] - oy) - YR[n] - R_ROB
        new_all[n] = surface_distance(s[:, 1], s[:, 2], ox, oy, SHAPES[n]) - R_ROB
    o_min = {n: float(v.min()) for n, v in old_all.items()}
    n_min = {n: float(v.min()) for n, v in new_all.items()}
    o_who = min(o_min, key=o_min.get); n_who = min(n_min, key=n_min.get)

    # 不受幾何影響的項目（回歸核對用）
    dt = s[2:, 0] - s[:-2, 0]
    vx = (s[2:, 1] - s[:-2, 1]) / dt; vy = (s[2:, 2] - s[:-2, 2]) / dt
    yaw = s[1:-1, 3]; sp = np.hypot(vx, vy); mv = sp > MOVE_EPS
    ang = np.array([abs(math.degrees(wrap(math.atan2(b, a) - c)))
                    for a, b, c in zip(vx, vy, yaw)])
    dyaw = np.array([wrap(a - b) for a, b in zip(s[1:, 3], s[:-1, 3])])
    per.append(dict(
        name=r['name'], seed=r['seed'], rep=r['rep'], cond=r['cond'],
        stop=rj['stop_reason'], arrived=bool(rj['arrived']),
        arrival_time_s=rj.get('arrival_time_s'),
        path_m=float(np.hypot(np.diff(s[:, 1]), np.diff(s[:, 2])).sum()),
        ang_med=float(np.median(ang[mv])), ang_lt15=float((ang[mv] < 15).mean() * 100),
        turn_abs_deg=float(np.degrees(np.abs(dyaw).sum())),
        old_disc_min=o_min[o_who], old_disc_who=o_who,
        new_sdf_min=n_min[n_who], new_sdf_who=n_who,
        delta=n_min[n_who] - o_min[o_who], n_samples=len(s)))

    # 遭遇事件：以**新**距離重新辨識，門檻不變
    vxb = vx * np.cos(yaw) + vy * np.sin(yaw)
    vyb = -vx * np.sin(yaw) + vy * np.cos(yaw)
    tt = s[1:-1, 0]
    for n in SHAPES:
        c = new_all[n][1:-1]
        inside = c <= D_ENTER
        if not inside.any():
            continue
        e = np.diff(inside.astype(int))
        st = list(np.where(e == 1)[0] + 1); en = list(np.where(e == -1)[0] + 1)
        if inside[0]: st = [0] + st
        if inside[-1]: en = en + [len(inside)]
        for a_, b_ in zip(st, en):
            cs = c[a_:b_]
            if cs.min() > D_CORE:
                continue
            core = np.where(cs <= D_CORE)[0]
            i0, i1 = a_ + core[0], a_ + core[-1] + 1
            ph = {'pre': slice(a_, i0), 'core': slice(i0, i1), 'post': slice(i1, b_)}
            if any((ph[k].stop - ph[k].start) < MIN_N for k in ph):
                continue
            rec = dict(run=r['name'], seed=r['seed'], rep=r['rep'], cond=r['cond'],
                       obs=n, c_min=float(cs.min()),
                       t_min=float(tt[a_ + int(np.argmin(cs))] - t0))
            for k, sl in ph.items():
                m2 = sp[sl] > MOVE_EPS
                g = (lambda arr: float(np.median(arr[sl][m2]))
                     if m2.sum() >= 3 else float('nan'))
                rec[f'{k}_vyb'] = g(np.abs(vyb)); rec[f'{k}_ang'] = g(ang)
            events.append(rec)

for f, rows in (('per_run.csv', per), ('events.csv', events)):
    if rows:
        with open(os.path.join(OUT, f), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
json.dump(runs, open(os.path.join(OUT, 'inventory.json'), 'w'),
          ensure_ascii=False, indent=1, default=str)
print(f'\n可評估 {len(per)} / {len(runs)} 趟；遭遇事件 {len(events)} 個')
print(f'輸出 -> {OUT}')
