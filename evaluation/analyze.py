"""Read a Nav2 / GMPC rosbag and emit one row of evaluation metrics.

Reads:
    /odom                      (nav_msgs/Odometry)        actual pose history
    /cmd_vel                   (geometry_msgs/Twist)      applied command
    /plan                      (nav_msgs/Path)            published global path
    (optional) /cmd_vel_nav    pre-velocity-smoother cmd  (for RPP/MPPI runs)

Computes (per run):
    success            : robot reached goal within tolerance before bag ended
    arrival_time_s     : seconds from first /plan to within goal_tolerance_m
    total_time_s       : bag duration
    path_length_m      : cumulative odom motion
    tracking_rmse_m    : RMS distance from odom pose to closest /plan point
                         (using the LATEST /plan available at each odom timestamp)
    smooth_vx/vy/wz    : std of /cmd_vel components  (lower = smoother)
    jerk_vx/vy/wz      : std of finite-difference accel             (lower = smoother)

Usage
-----
    python3 analyze.py BAG_PATH --method NAME --run RUN_ID [--out results/runs.csv]

Each call appends one row to the output CSV (creates header if file is new).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# ROS bag deps (Jazzy)
try:
    from rosbag2_py            import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization   import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f'ERROR: ROS2 python bindings not available — source /opt/ros/jazzy/setup.bash first',
          file=sys.stderr)
    raise


GOAL_TOLERANCE_M = 0.25

# Topics we care about. The GMPC stack publishes:
#   /gmpc/solve_time_ms : std_msgs/Float32 — per-step OSQP wall time
#   /gmpc/obstacles     : std_msgs/Float32MultiArray — flat [x,y,r,...] ground truth
#   /gmpc/min_h         : std_msgs/Float32 — smallest CBF barrier value
TOPICS_OF_INTEREST = {
    '/odom', '/cmd_vel', '/cmd_vel_nav', '/plan',
    '/gmpc/solve_time_ms', '/gmpc/obstacles', '/gmpc/min_h',
}

COLLISION_BUFFER_M = 0.0   # treat distance < radius as collision


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------

def _detect_storage_id(bag_path: str) -> str:
    """Auto-detect rosbag2 storage backend from the file extensions present.

    Jazzy default is MCAP; older bags use SQLite3. Falls back to mcap.
    """
    p = Path(bag_path)
    if any(p.glob('*.mcap')):
        return 'mcap'
    if any(p.glob('*.db3')):
        return 'sqlite3'
    return 'mcap'


def read_bag(bag_path: str) -> dict[str, list[tuple[int, object]]]:
    """Return {topic: [(timestamp_ns, msg), ...]} for topics we care about."""
    reader = SequentialReader()
    storage   = StorageOptions(uri=bag_path, storage_id=_detect_storage_id(bag_path))
    converter = ConverterOptions(input_serialization_format='cdr',
                                 output_serialization_format='cdr')
    reader.open(storage, converter)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    bucket = {topic: [] for topic in type_map}

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in TOPICS_OF_INTEREST:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        bucket[topic].append((int(t_ns), msg))

    return {t: bucket[t] for t in bucket if t in TOPICS_OF_INTEREST}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quaternion_to_yaw(qx, qy, qz, qw) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def extract_odom(messages):
    if '/odom' not in messages:
        return np.zeros(0), np.zeros((0, 3))
    ts, xyth = [], []
    for t_ns, m in messages['/odom']:
        p = m.pose.pose.position
        o = m.pose.pose.orientation
        ts.append(t_ns / 1e9)
        xyth.append((p.x, p.y, quaternion_to_yaw(o.x, o.y, o.z, o.w)))
    return np.array(ts), np.array(xyth)


def extract_cmd(messages, topic='/cmd_vel'):
    if topic not in messages:
        return np.zeros(0), np.zeros((0, 3))
    ts, vec = [], []
    for t_ns, m in messages[topic]:
        ts.append(t_ns / 1e9)
        vec.append((m.linear.x, m.linear.y, m.angular.z))
    return np.array(ts), np.array(vec)


def extract_plans(messages):
    """Return [(t_seconds, path_xy_array_(M,2)), ...] sorted by time."""
    if '/plan' not in messages:
        return []
    out = []
    for t_ns, m in messages['/plan']:
        if len(m.poses) == 0:
            continue
        xy = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])
        out.append((t_ns / 1e9, xy))
    return out


def extract_float32_series(messages, topic: str):
    """Return arrays (ts, vals) from a std_msgs/Float32 topic, or empty if absent."""
    if topic not in messages:
        return np.zeros(0), np.zeros(0)
    ts, vals = [], []
    for t_ns, m in messages[topic]:
        ts.append(t_ns / 1e9)
        vals.append(float(m.data))
    return np.array(ts), np.array(vals)


def extract_obstacles(messages, topic='/gmpc/obstacles'):
    """Time-series of obstacle snapshots from a Float32MultiArray topic.

    Wire format (5 floats per obstacle):
        [x1, y1, r1, vx1, vy1, x2, y2, r2, vx2, vy2, ...]
    Returns list of (t_seconds, np.ndarray of shape (N, 3) with cols x, y, radius).
    The velocity components are kept on the wire for the CBF predictor but
    aren't needed by analyze.py — we drop them here.
    """
    if topic not in messages:
        return []
    out = []
    for t_ns, m in messages[topic]:
        data = np.asarray(m.data, dtype=float)
        if data.size == 0 or data.size % 5 != 0:
            continue
        full = data.reshape(-1, 5)        # (N, [x, y, r, vx, vy])
        xy_r = full[:, :3]                # keep x, y, r only
        out.append((t_ns / 1e9, xy_r))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(messages, goal_tol_m: float = GOAL_TOLERANCE_M) -> dict:
    odom_t,  odom_xyth = extract_odom(messages)
    cmd_t,   cmd_vec   = extract_cmd(messages, '/cmd_vel')
    plans              = extract_plans(messages)
    obs_series         = extract_obstacles(messages)
    st_t, st_vals      = extract_float32_series(messages, '/gmpc/solve_time_ms')

    out = {
        'success'       : False,
        'arrival_time_s': float('nan'),
        'total_time_s'  : float('nan'),
        'path_length_m' : float('nan'),
        'tracking_rmse_m': float('nan'),
        'smooth_vx'     : float('nan'),
        'smooth_vy'     : float('nan'),
        'smooth_wz'     : float('nan'),
        'jerk_vx'       : float('nan'),
        'jerk_vy'       : float('nan'),
        'jerk_wz'       : float('nan'),
        # Safety / timing
        'min_clearance_m'   : float('nan'),
        'collision_count'   : 0,
        'collided'          : False,
        'solve_time_mean_ms': float('nan'),
        'solve_time_p95_ms' : float('nan'),
        'solve_time_max_ms' : float('nan'),
        'n_odom'        : len(odom_t),
        'n_cmd'         : len(cmd_t),
        'n_plan'        : len(plans),
        'n_obstacles'   : len(obs_series),
        'n_solve_time'  : len(st_vals),
    }
    if len(odom_t) < 2 or not plans:
        return out

    first_plan_t = plans[0][0]
    goal_xy = plans[-1][1][-1]                       # endpoint of latest /plan

    # ------- arrival time / success -------
    arrival_t = None
    in_run = odom_t >= first_plan_t
    for i in np.where(in_run)[0]:
        if np.linalg.norm(odom_xyth[i, :2] - goal_xy) < goal_tol_m:
            arrival_t = odom_t[i] - first_plan_t
            break
    out['success']        = arrival_t is not None
    out['arrival_time_s'] = float(arrival_t) if arrival_t is not None else float('nan')
    out['total_time_s']   = float(odom_t[-1] - first_plan_t)

    # ------- path length -------
    out['path_length_m'] = float(np.sum(
        np.linalg.norm(np.diff(odom_xyth[in_run, :2], axis=0), axis=1)
    ))

    # ------- tracking RMSE vs latest /plan available at each odom time -------
    plan_ts = np.array([p[0] for p in plans])
    plan_xys = [p[1] for p in plans]
    errs = []
    for i in np.where(in_run)[0]:
        t = odom_t[i]
        j = int(np.searchsorted(plan_ts, t, side='right') - 1)
        j = max(0, min(j, len(plans) - 1))
        d_min = float(np.min(np.linalg.norm(plan_xys[j] - odom_xyth[i, :2], axis=1)))
        errs.append(d_min)
    if errs:
        out['tracking_rmse_m'] = float(np.sqrt(np.mean(np.square(errs))))

    # ------- smoothness + jerk -------
    if len(cmd_vec) > 0:
        out['smooth_vx'] = float(np.std(cmd_vec[:, 0]))
        out['smooth_vy'] = float(np.std(cmd_vec[:, 1]))
        out['smooth_wz'] = float(np.std(cmd_vec[:, 2]))
    if len(cmd_vec) > 1:
        dt = np.diff(cmd_t)
        ok = dt > 1e-6
        if np.any(ok):
            accel = np.diff(cmd_vec, axis=0)
            accel = accel[ok] / dt[ok, None]
            out['jerk_vx'] = float(np.std(accel[:, 0]))
            out['jerk_vy'] = float(np.std(accel[:, 1]))
            out['jerk_wz'] = float(np.std(accel[:, 2]))

    # ------- safety: per-odom min distance to any (time-aligned) obstacle ----
    # If /gmpc/obstacles is missing (RPP/MPPI bags), the bag won't have it.
    # In that case the safety metrics stay NaN — the user should record with
    # obstacle_aggregator running so all baselines share the same ground truth.
    if obs_series:
        obs_ts = np.array([t for t, _ in obs_series])
        clearances = []
        n_coll = 0
        in_collision = False
        for i in np.where(in_run)[0]:
            t = odom_t[i]
            j = int(np.searchsorted(obs_ts, t, side='right') - 1)
            j = max(0, min(j, len(obs_series) - 1))
            obs_xy_r = obs_series[j][1]                    # (M, 3)
            d_centres = np.linalg.norm(obs_xy_r[:, :2] - odom_xyth[i, :2], axis=1)
            d_surfaces = d_centres - obs_xy_r[:, 2]        # subtract radius
            d_min = float(np.min(d_surfaces))
            clearances.append(d_min)
            # Edge-trigger collision count when crossing zero
            if d_min < COLLISION_BUFFER_M:
                if not in_collision:
                    n_coll += 1
                in_collision = True
            else:
                in_collision = False
        if clearances:
            out['min_clearance_m'] = float(np.min(clearances))
            out['collision_count'] = int(n_coll)
            out['collided']        = bool(n_coll > 0)

    # ------- solve time stats (GMPC only — empty for RPP/MPPI bags) ---------
    if len(st_vals):
        out['solve_time_mean_ms'] = float(np.mean(st_vals))
        out['solve_time_p95_ms']  = float(np.percentile(st_vals, 95))
        out['solve_time_max_ms']  = float(np.max(st_vals))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CSV_HEADER = [
    'method', 'run',
    'success', 'arrival_time_s', 'total_time_s', 'path_length_m',
    'tracking_rmse_m',
    'smooth_vx', 'smooth_vy', 'smooth_wz',
    'jerk_vx',   'jerk_vy',   'jerk_wz',
    'min_clearance_m', 'collision_count', 'collided',
    'solve_time_mean_ms', 'solve_time_p95_ms', 'solve_time_max_ms',
    'n_odom', 'n_cmd', 'n_plan', 'n_obstacles', 'n_solve_time',
    'bag',
]


def append_csv(out_path: Path, row: dict):
    new = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in CSV_HEADER})


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('bag',     help='Path to rosbag directory (contains metadata.yaml + .db3)')
    ap.add_argument('--method', required=True,
                    choices=['rpp', 'mppi', 'gmpc', 'gmpc_cbf'],
                    help='Controller used in this run')
    ap.add_argument('--run',    required=True, help='Run identifier (e.g. seed42_run1)')
    ap.add_argument('--out',    default=str(here / 'results' / 'runs.csv'),
                    help='CSV path to append to')
    args = ap.parse_args()

    bag_path = str(Path(args.bag).expanduser().resolve())
    messages = read_bag(bag_path)
    metrics  = compute_metrics(messages)

    row = {
        'method': args.method,
        'run':    args.run,
        'bag':    bag_path,
        **metrics,
    }
    append_csv(Path(args.out), row)

    # Print human-readable summary too
    def fmt(v): return 'NaN' if (isinstance(v, float) and np.isnan(v)) else f'{v}'
    print(f'[{args.method:>4} / {args.run}]')
    for k in CSV_HEADER[2:-1]:                       # skip method/run/bag
        v = row[k]
        if isinstance(v, float):
            print(f'  {k:<18}  {v:>10.4f}' if not np.isnan(v) else f'  {k:<18}  {"NaN":>10}')
        else:
            print(f'  {k:<18}  {v}')


if __name__ == '__main__':
    main()
