"""朝向 OFF／ON 配對的跨 seed 彙整。

每組配對先驗證實際障礙軌跡，再輸出兩趟共同具備的指標。
guard 內部狀態只有在該趟真的錄到時才報；**缺少不等於零次**。

用法：
    python3 evaluation/summarise_heading_pairs.py \
        --pair SEED OFF_RUN_DIR ON_RUN_DIR [--pair ...]
"""
import argparse, json, math, os, sys
import numpy as np, yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from geometry_msgs.msg import PoseArray, PoseStamped, Twist
from std_msgs.msg import Float32MultiArray, Float64, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('BIGARENA', '1'); os.environ.setdefault('SCENE', 'bigarena')
import analyze as AZ
STATIC = AZ.load_static_clearance()
R_ROB = AZ.ROBOT_RADIUS_M                       # 0.30
L, RW, W_MAX, A_MAX = 0.245, 0.05, 5.55, 125.0
W = np.array([[0, 1, L], [-1, 0, L], [0, -1, L], [1, 0, L]], float)
V_LIM = RW * W_MAX
TRAJ = 'src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml'
CFG = yaml.safe_load(open(TRAJ))['dynamic_obstacles']
NAMES = [d['name'] for d in CFG]
RAD = {d['name']: float(d['radius']) for d in CFG}
ACT = {0: 'as_is', 1: 'acc_clipped', 2: 'wheel_scaled', 3: 'brake'}
TOL = 1e-6


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def sched(o, t):
    sx, sy = o['start']; ex, ey = o['end']
    Ln = math.hypot(ex - sx, ey - sy); ux, uy = (ex - sx) / Ln, (ey - sy) / Ln
    s0 = o['phase0_m'] if o.get('direction', 1.0) >= 0 else (2 * Ln - o['phase0_m'])
    s = (s0 + o['speed'] * t) % (2 * Ln)
    d = s if s <= Ln else 2 * Ln - s
    return sx + ux * d, sy + uy * d


def read(run):
    r = SequentialReader()
    r.open(StorageOptions(uri=os.path.join(run, 'bag'), storage_id='mcap'),
           ConverterOptions('', ''))
    d = dict(rob=[], obs={n: [] for n in NAMES}, tgt=[], cmd=[], nav=[],
             guard=[], diag=[], epoch=None, epoch_n=0)
    while r.has_next():
        tn, raw, ts = r.read_next()
        if tn == '/model/omni_bot/pose':
            m = deserialize_message(raw, PoseStamped); s = m.header.stamp
            d['rob'].append((s.sec + s.nanosec * 1e-9, m.pose.position.x,
                             m.pose.position.y, yaw_of(m.pose.orientation)))
        elif tn.startswith('/model/dyn_obs_') and tn.endswith('/pose'):
            n = tn.split('/')[2]
            if n in d['obs']:
                m = deserialize_message(raw, PoseStamped); s = m.header.stamp
                d['obs'][n].append((s.sec + s.nanosec * 1e-9,
                                    m.pose.position.x, m.pose.position.y))
        elif tn == '/dynamic_obstacles/target':
            m = deserialize_message(raw, PoseArray); s = m.header.stamp
            d['tgt'].append((s.sec + s.nanosec * 1e-9,
                             [(p.position.x, p.position.y) for p in m.poses]))
        elif tn == '/dynamic_obstacles/phase_epoch':
            v = deserialize_message(raw, Float64).data
            if np.isfinite(v):
                if d['epoch'] is not None and abs(v - d['epoch']) > 1e-6:
                    d['epoch_n'] += 1
                d['epoch'] = v
        elif tn in ('/cmd_vel', '/cmd_vel_nav'):
            m = deserialize_message(raw, Twist)
            (d['cmd'] if tn == '/cmd_vel' else d['nav']).append(
                (ts * 1e-9, m.linear.x, m.linear.y, m.angular.z))
        elif tn == '/wheel_guard/status':
            d['guard'].append(json.loads(deserialize_message(raw, String).data))
        elif tn == '/gmpc/diag':
            d['diag'].append(list(deserialize_message(raw, Float32MultiArray).data))
    d['rob'] = np.array(d['rob'])
    d['obs'] = {k: np.array(v) for k, v in d['obs'].items()}
    d['run'] = json.load(open(os.path.join(run, 'isaac_run.json')))['run']
    d['dir'] = run
    return d


def scen(d):
    """實際障礙軌跡是否符合排程（趟內）。"""
    ep = d['epoch']
    if ep is None:
        return None
    full = [(t, p) for t, p in d['tgt'] if len(p) == len(NAMES)]
    et, ea = [], []
    for t, p in full:
        for i, n in enumerate(NAMES):
            gx, gy = sched(CFG[i], t - ep)
            et.append(math.hypot(p[i][0] - gx, p[i][1] - gy))
    for n, o in zip(NAMES, CFG):
        A = d['obs'][n]; A = A[A[:, 0] > ep + 1.0]
        for t, x, y in A:
            gx, gy = sched(o, t - ep)
            ea.append(math.hypot(x - gx, y - gy))
    return dict(epoch=ep, resets=d['epoch_n'], n=len(full),
                tgt_max=max(et) * 1000 if et else float('nan'),
                act_med=np.median(ea) * 1000 if ea else float('nan'),
                act_max=max(ea) * 1000 if ea else float('nan'))


def cross(a, b):
    """兩趟實際軌跡以各自相位零點對齊後的差。"""
    if a['epoch'] is None or b['epoch'] is None:
        return None
    worst = 0.0
    for n in NAMES:
        A, B = a['obs'][n], b['obs'][n]
        ta, tb = A[:, 0] - a['epoch'], B[:, 0] - b['epoch']
        lo = max(ta.min(), tb.min(), 0.0); hi = min(ta.max(), tb.max())
        if hi <= lo:
            continue
        g = np.arange(lo, hi, 0.5)
        e = np.hypot(np.interp(g, ta, A[:, 1]) - np.interp(g, tb, B[:, 1]),
                     np.interp(g, ta, A[:, 2]) - np.interp(g, tb, B[:, 2]))
        worst = max(worst, e.max())
    return worst * 1000


def metrics(d):
    rj = d['run']
    t0, t1 = float(rj['motion_start_sim_t']), float(rj['sim_time'])
    s = d['rob'][(d['rob'][:, 0] >= t0) & (d['rob'][:, 0] <= t1)]
    m = dict(stop=rj['stop_reason'], arrived=bool(rj['arrived']),
             t_arr=float(rj['arrival_time_s']), n=len(s))
    m['path'] = float(np.hypot(np.diff(s[:, 1]), np.diff(s[:, 2])).sum())
    # 速度、夾角、轉動
    dt = s[2:, 0] - s[:-2, 0]
    vx = (s[2:, 1] - s[:-2, 1]) / dt; vy = (s[2:, 2] - s[:-2, 2]) / dt
    yaw = s[1:-1, 3]
    wz = np.array([wrap(a - b) for a, b in zip(s[2:, 3], s[:-2, 3])]) / dt
    vxb = vx * np.cos(yaw) + vy * np.sin(yaw)
    vyb = -vx * np.sin(yaw) + vy * np.cos(yaw)
    sp = np.hypot(vx, vy); mv = sp > 0.05
    ang = np.array([abs(math.degrees(wrap(math.atan2(b, a) - c)))
                    for a, b, c in zip(vx, vy, yaw)])
    m['ang_med'] = float(np.median(ang[mv]))
    m['ang_lt15'] = float((ang[mv] < 15).mean() * 100)
    m['fwd'] = float((vxb[mv] > 0.05).mean() * 100)
    m['bwd'] = float((vxb[mv] < -0.05).mean() * 100)
    m['lat'] = float((np.abs(vyb[mv]) > np.abs(vxb[mv])).mean() * 100)
    dyaw = np.array([wrap(a - b) for a, b in zip(s[1:, 3], s[:-1, 3])])
    m['turn_abs'] = float(np.degrees(np.abs(dyaw).sum()))
    m['wz_max'] = float(np.abs(wz).max())
    big = np.abs(wz) > 0.02
    sg = np.sign(wz[big])
    m['sign_ch'] = int((np.diff(sg) != 0).sum()) if len(sg) > 1 else 0
    # 淨距
    clr = np.full(len(s), 1e9)
    for n, A in d['obs'].items():
        ox = np.interp(s[:, 0], A[:, 0], A[:, 1])
        oy = np.interp(s[:, 0], A[:, 0], A[:, 2])
        clr = np.minimum(clr, np.hypot(s[:, 1] - ox, s[:, 2] - oy)
                         - RAD[n] - R_ROB)
    m['dyn_clr'] = float(clr.min())
    m['stat_clr'] = (float(np.nanmin([STATIC(x, y) - R_ROB
                                      for _, x, y, _ in s]))
                     if STATIC else float('nan'))
    c = clr[1:-1]
    for tag, msk in (('vy_far', mv & (c > 1.0)), ('vy_near', mv & (c <= 0.5))):
        m[tag] = float(np.median(np.abs(vyb[msk]))) if msk.sum() >= 3 else float('nan')
    # 最終命令輪速
    C = np.array(d['cmd']); wsp = np.abs(C[:, 1:4] @ W.T).max(axis=1)
    m['w_max'] = float(wsp.max() / RW); m['w_over'] = int((wsp > V_LIM + 1e-9).sum())
    m['w_n'] = len(C)
    # 求解器出口
    A = np.array(d['diag'], float)
    ok = np.isfinite(A[:, 13])
    act = A[ok, 23]
    m['n_solve'] = int(ok.sum())
    m['wheel_scaled'] = int((act == 2).sum())
    m['minh'] = float(np.nanmin(A[ok, 13]))
    rn = A[ok, 17]; m['resid_neg'] = int((rn[np.isfinite(rn)] < -1e-9).sum())
    # guard（只有錄到才報）
    g = d['guard']
    if g:
        okg = [x for x in g if 'input_timeout' not in x.get('faults', [])]
        lam = np.array([x['lam'] for x in okg if x['lam'] is not None], float)
        m['g_n'] = len(g)
        m['g_clip'] = int((lam < 1 - 1e-9).sum()) if len(lam) else 0
        m['g_mod'] = sum(1 for x in g if x['action'] != 'ok' or x.get('faults')
                         or np.max(np.abs(np.array(x['out'])
                                          - np.array(x['target']))) > TOL)
        m['g_sw'] = sum(1 for a, b in zip(g, g[1:]) if a['mode'] != b['mode'])
        m['g_faults'] = sorted({f for x in g for f in x.get('faults', [])})
        dts = np.array([x['dt'] for x in g if x['dt'] is not None], float)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        wo = np.array([np.abs(np.array(x['out']) @ W.T).max() / RW for x in g])
        aw = np.abs(np.diff(wo))[:len(dts) - 1] / dts[1:]
        m['g_acc_p95'] = float(np.percentile(aw, 95)); m['g_acc_max'] = float(aw.max())
        m['g_acc_over'] = int((aw > A_MAX + 1e-6).sum())
    else:
        m['g_n'] = 0
    return m


ap = argparse.ArgumentParser()
ap.add_argument('--pair', nargs=3, action='append', metavar=('SEED','OFF','ON'),
                required=True)
a = ap.parse_args()

STRAIGHT = {int(r.split(',')[0]): float(r.split(',')[5])
            for r in open('evaluation/results/bigarena_poses.csv').read()
                        .strip().split('\n')[1:]}

rows = []
for seed, off_d, on_d in a.pair:
    seed = int(seed)
    A, B = read(off_d), read(on_d)
    sa, sb, cx = scen(A), scen(B), cross(A, B)
    ma, mb = metrics(A), metrics(B)
    rows.append((seed, ma, mb))
    print(f'\n╔══ seed {seed}  直線 {STRAIGHT[seed]:.2f} m ' + '═'*30)
    print(f'║ 場景驗證')
    for tag, sc in (('OFF', sa), ('ON ', sb)):
        if sc is None:
            print(f'║   {tag} 無相位零點'); continue
        print(f'║   {tag} epoch {sc["epoch"]:.3f}s 重設 {sc["resets"]}  '
              f'驅動目標對排程 max {sc["tgt_max"]:.4f} mm  '
              f'真實位置對排程 中位 {sc["act_med"]:.2f} / max {sc["act_max"]:.2f} mm')
    print(f'║   兩趟實際軌跡對齊 最大差 {cx:.1f} mm'
          if cx is not None else '║   無法對齊')
    print(f'║')
    def line(k, fmt, name, better=None):
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None:
            print(f'║ {name:<26} {"—":>12} {"—":>12}'); return
        print(f'║ {name:<26} {format(va,fmt):>12} {format(vb,fmt):>12}')
    print(f'║ {"指標":<24} {"OFF":>12} {"ON":>12}')
    print(f'║ {"-"*52}')
    print(f'║ {"停止原因":<24} {ma["stop"]:>12} {mb["stop"]:>12}')
    print(f'║ {"到達":<26} {str(ma["arrived"]):>12} {str(mb["arrived"]):>12}')
    line('t_arr','.3f','到達時間 s')
    line('path','.3f','實際路徑 m')
    line('ang_med','.2f','夾角中位 °')
    line('ang_lt15','.1f','夾角<15° %')
    line('fwd','.1f','前進 %'); line('bwd','.1f','倒退 %'); line('lat','.1f','側移主導 %')
    line('turn_abs','.2f','累計絕對轉角 °')
    line('wz_max','.4f','|wz| max rad/s'); line('sign_ch','d','角速度變號次數')
    line('dyn_clr','.4f','動態最近距離 m'); line('stat_clr','.4f','靜態最近距離 m')
    line('vy_far','.4f','|vy_b| p50 遠 (>1m)'); line('vy_near','.4f','|vy_b| p50 近 (<=0.5m)')
    line('w_max','.4f','最終命令輪速 max'); line('w_over','d','輪速超限筆數')
    line('wheel_scaled','d','出口 wheel_scaled')
    line('minh','.4f','min_h (感知)'); line('resid_neg','d','無slack殘差<0 筆數')
    for tag, m in (('OFF', ma), ('ON ', mb)):
        if m['g_n']:
            print(f'║ guard {tag}: n={m["g_n"]} 截斷 {m["g_clip"]} 修改 {m["g_mod"]} '
                  f'模式切換 {m["g_sw"]} faults {m["g_faults"] or "無"} '
                  f'輪加速度 p95 {m["g_acc_p95"]:.2f} max {m["g_acc_max"]:.2f} '
                  f'超限 {m["g_acc_over"]}')
        else:
            print(f'║ guard {tag}: **未錄製**（缺少不等於零次）')
    print('╚' + '═'*60)

print('\n\n跨 seed 彙整（ON − OFF）')
print(f'{"seed":>5} {"到達":>9} {"夾角中位°":>18} {"夾角<15%":>16} '
      f'{"路徑 m":>16} {"到達時間 s":>16} {"轉角°":>16} {"淨距 m":>16}')
for seed, ma, mb in rows:
    arr = f'{"Y" if ma["arrived"] else "N"}/{"Y" if mb["arrived"] else "N"}'
    print(f'{seed:5d} {arr:>9} '
          f'{ma["ang_med"]:7.2f}→{mb["ang_med"]:<7.2f} '
          f'{ma["ang_lt15"]:6.1f}→{mb["ang_lt15"]:<6.1f} '
          f'{ma["path"]:7.3f}→{mb["path"]:<7.3f} '
          f'{ma["t_arr"]:7.3f}→{mb["t_arr"]:<7.3f} '
          f'{ma["turn_abs"]:7.2f}→{mb["turn_abs"]:<7.2f} '
          f'{ma["dyn_clr"]:7.4f}→{mb["dyn_clr"]:<7.4f}')
