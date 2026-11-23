"""避障前／中／後的遭遇事件分析。

**事件判準在分析前固定，且只依距離，與朝向誤差完全無關**（避免用結果去挑事件）：

    淨距 c_i(t) = ‖p_robot(t) − p_obs_i(t)‖ − r_i − R_ROBOT,  R_ROBOT = 0.30 m

    事件      對單一障礙物 i，c_i ≤ D_ENTER = 1.00 m 的極大連續區間
    合格事件  該區間內 min c_i ≤ D_CORE = 0.50 m（否則視為遠距通過，捨棄）
    前        由進入 1.00 m 起，到首次跌破 0.50 m 為止
    中        c_i ≤ 0.50 m 的子區間
    後        由最後一次升回 0.50 m 起，到離開 1.00 m 為止
    每個階段至少 MIN_N = 10 個取樣（0.5 s）才納入

D_ENTER 與 D_CORE 只牽涉幾何距離；車頭角、側向速度都**不參與事件挑選**，
只在事件挑好之後才量測。

用法：
    python3 evaluation/analyze_encounters.py RUN_DIR [RUN_DIR ...] [--csv OUT]
"""
import argparse, json, math, os
import numpy as np, yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from geometry_msgs.msg import PoseStamped

R_ROBOT = 0.30
D_ENTER = 1.00
D_CORE  = 0.50
MIN_N   = 10
MOVE_EPS = 0.05                       # m/s，低於此不計入方向類統計

TRAJ = 'src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml'
CFG = yaml.safe_load(open(TRAJ))['dynamic_obstacles']
RAD = {d['name']: float(d['radius']) for d in CFG}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load(run):
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.join(run, 'bag'), storage_id='mcap'),
           ConverterOptions('', ''))
    rob, obs = [], {n: [] for n in RAD}
    while r.has_next():
        tn, raw, _ = r.read_next()
        if tn == '/model/omni_bot/pose':
            m = deserialize_message(raw, PoseStamped); s = m.header.stamp
            rob.append((s.sec + s.nanosec * 1e-9, m.pose.position.x,
                        m.pose.position.y, yaw_of(m.pose.orientation)))
        elif tn.startswith('/model/dyn_obs_') and tn.endswith('/pose'):
            n = tn.split('/')[2]
            if n in obs:
                m = deserialize_message(raw, PoseStamped); s = m.header.stamp
                obs[n].append((s.sec + s.nanosec * 1e-9, m.pose.position.x,
                               m.pose.position.y))
    rj = json.load(open(os.path.join(run, 'isaac_run.json')))['run']
    return np.array(rob), {k: np.array(v) for k, v in obs.items()}, rj


def events(run):
    rob, obs, rj = load(run)
    t0, t1 = float(rj['motion_start_sim_t']), float(rj['sim_time'])
    s = rob[(rob[:, 0] >= t0) & (rob[:, 0] <= t1)]
    if len(s) < 5:
        return []
    dt = s[2:, 0] - s[:-2, 0]
    vx = (s[2:, 1] - s[:-2, 1]) / dt
    vy = (s[2:, 2] - s[:-2, 2]) / dt
    yaw = s[1:-1, 3]
    wz = np.array([wrap(a - b) for a, b in zip(s[2:, 3], s[:-2, 3])]) / dt
    vxb = vx * np.cos(yaw) + vy * np.sin(yaw)
    vyb = -vx * np.sin(yaw) + vy * np.cos(yaw)
    sp = np.hypot(vx, vy)
    ang = np.array([abs(math.degrees(wrap(math.atan2(b, a) - c)))
                    for a, b, c in zip(vx, vy, yaw)])
    tt = s[1:-1, 0]

    out = []
    for n, A in obs.items():
        ox = np.interp(tt, A[:, 0], A[:, 1]); oy = np.interp(tt, A[:, 0], A[:, 2])
        c = np.hypot(s[1:-1, 1] - ox, s[1:-1, 2] - oy) - RAD[n] - R_ROBOT
        inside = c <= D_ENTER
        if not inside.any():
            continue
        edges = np.diff(inside.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if inside[0]:
            starts = [0] + starts
        if inside[-1]:
            ends = ends + [len(inside)]
        for a, b in zip(starts, ends):
            seg = slice(a, b)
            cs = c[seg]
            if cs.min() > D_CORE:
                continue
            core = np.where(cs <= D_CORE)[0]
            i_core0, i_core1 = a + core[0], a + core[-1] + 1
            ph = {'前': slice(a, i_core0), '中': slice(i_core0, i_core1),
                  '後': slice(i_core1, b)}
            if any((ph[k].stop - ph[k].start) < MIN_N for k in ph):
                continue
            rec = dict(obs=n, t_min=float(tt[a + int(np.argmin(cs))] - t0),
                       c_min=float(cs.min()), dur=float(tt[b - 1] - tt[a]))
            for k, sl in ph.items():
                mv = sp[sl] > MOVE_EPS
                g = lambda arr: (float(np.median(arr[sl][mv]))
                                 if mv.sum() >= 3 else float('nan'))
                rec[f'{k}_vyb'] = g(np.abs(vyb))
                rec[f'{k}_ang'] = g(ang)
                rec[f'{k}_wz'] = g(np.abs(wz))
                rec[f'{k}_sp'] = g(sp)
                rec[f'{k}_c'] = float(np.min(c[sl]))
                rec[f'{k}_n'] = int(sl.stop - sl.start)
            out.append(rec)
    return sorted(out, key=lambda r: r['t_min'])


ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--csv', default='')
a = ap.parse_args()

print(f'判準（分析前固定，只依距離）：D_ENTER={D_ENTER} m，D_CORE={D_CORE} m，'
      f'每階段至少 {MIN_N} 取樣，機器人半徑 {R_ROBOT} m')
print(f'方向類統計只取速率 > {MOVE_EPS} m/s 的取樣\n')
rows = []
for run in a.runs:
    ev = events(run)
    base = os.path.basename(run)
    side = 'ON ' if 'heading' in base else 'OFF'
    seed = base.split('__seed')[1].split('__')[0]
    print(f'{base}')
    print(f'  seed {seed} {side}  合格事件 {len(ev)}')
    for e in ev:
        print(f'    {e["obs"]:<11} t={e["t_min"]:6.1f}s 最近 {e["c_min"]:.4f} '
              f'長 {e["dur"]:5.1f}s │ |vy_b| 前 {e["前_vyb"]:.4f} 中 {e["中_vyb"]:.4f} '
              f'後 {e["後_vyb"]:.4f} │ 夾角 前 {e["前_ang"]:5.1f} 中 {e["中_ang"]:5.1f} '
              f'後 {e["後_ang"]:5.1f}')
        rows.append(dict(run=base, seed=int(seed), side=side.strip(), **e))
    print()

if rows:
    import csv as _csv
    print('=' * 78)
    for side in ('OFF', 'ON'):
        R = [r for r in rows if r['side'] == side]
        if not R:
            continue
        print(f'\n== {side}  事件 {len(R)} 個（{len({r["run"] for r in R})} 趟）==')
        for k, lab in (('vyb', '|vy_b| 中位 m/s'), ('ang', '車頭−行進夾角 °'),
                       ('wz', '|wz| 中位 rad/s'), ('sp', '速率 m/s')):
            v = {p: np.array([r[f'{p}_{k}'] for r in R], float) for p in '前中後'}
            v = {p: x[np.isfinite(x)] for p, x in v.items()}
            print(f'  {lab:<18} 前 {np.median(v["前"]):7.4f}   '
                  f'中 {np.median(v["中"]):7.4f}   後 {np.median(v["後"]):7.4f}')
        d = np.array([r['中_vyb'] - r['前_vyb'] for r in R], float)
        d = d[np.isfinite(d)]
        print(f'  中 − 前 的 |vy_b| 差：> 0 的事件 {int((d > 0).sum())}/{len(d)}'
              f'（中位 {np.median(d):+.4f}）')
        d2 = np.array([r['後_ang'] - r['中_ang'] for r in R], float)
        d2 = d2[np.isfinite(d2)]
        print(f'  後 − 中 的夾角差：< 0（恢復對齊）的事件 '
              f'{int((d2 < 0).sum())}/{len(d2)}（中位 {np.median(d2):+.2f}°）')
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'\n逐事件資料 -> {a.csv}')
