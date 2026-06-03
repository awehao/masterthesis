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
import re
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
    '/tf', '/tf_static',
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


def parse_trial_log_for_goal(log_path: Path) -> dict:
    """Parse run_one_trial.sh's per-trial log for the controller's own
    authoritative goal-reached signal.

    Why this exists: rosbag2 samples /tf at the publisher's rate (~1 Hz for
    AMCL's map→odom updates). When AMCL re-converges briefly close to goal
    and then drifts back, the bag may miss the in-tolerance window, and the
    bag-based success check fails even though the controller itself logged
    'Goal reached'. The controller has access to the live TF buffer at the
    full controller rate and is therefore the more authoritative source.

    Returns dict with possibly-missing keys:
        'controller_goal_reached'   : bool
        'controller_arrival_dist_m' : float    (distance reported in the log)
        'controller_arrival_time_s' : float    (Goal-reached − first-plan timestamp)
    """
    out = {'controller_goal_reached': False}
    if not log_path.is_file():
        return out

    # Example lines (ROS2 launch prefixes each child's stdout with the node tag):
    #   [gmpc_node-7] [INFO] [1780501339.596244635] [gmpc_controller]: Goal reached (within 0.191 m of (17.00, 17.00), tol=0.20 m) -- holding zero twist
    #   [goal_to_plan_relay-5] [INFO] [1780501023.298733212] [goal_to_plan_relay]: Plan request (new goal): (0.00, 0.00) -> (17.00, 17.00)
    goal_re = re.compile(
        r'\[(\d+\.\d+)\].*?gmpc_controller.*?Goal reached.*?within\s+([\d.]+)\s+m'
    )
    plan_re = re.compile(
        r'\[(\d+\.\d+)\].*?goal_to_plan_relay.*?Plan request \(new goal\)'
    )

    plan_t = None
    goal_t = None
    goal_d = None
    try:
        with open(log_path, errors='replace') as f:
            for line in f:
                if plan_t is None:
                    m = plan_re.search(line)
                    if m:
                        plan_t = float(m.group(1))
                m = goal_re.search(line)
                if m:
                    # Take the FIRST Goal-reached event (subsequent log lines
                    # may re-print when the controller is in hold-state).
                    goal_t = float(m.group(1))
                    goal_d = float(m.group(2))
                    break
    except OSError:
        return out

    if goal_t is None:
        return out

    out['controller_goal_reached']   = True
    out['controller_arrival_dist_m'] = goal_d
    if plan_t is not None:
        out['controller_arrival_time_s'] = goal_t - plan_t
    return out


def extract_robot_map_pose(messages):
    """Reconstruct robot pose in **map** frame by composing map→odom (from
    AMCL) with odom→base_footprint (from the chassis controller) along the
    /tf stream.

    Returns (ts, xyth) where xyth is the robot's (x, y, yaw) in map frame at
    each timestamp where a /tf update covers both halves of the chain.
    Empty arrays if /tf is missing or chain incomplete.

    Why this is needed: /odom is in the *odom* frame (dead-reckoning).
    AMCL corrects drift via the map→odom transform. The controller uses
    map-frame pose (via TF lookup) for its goal check, so analyze.py must
    do the same to agree with the runtime "Goal reached" decision.
    """
    if '/tf' not in messages:
        return np.zeros(0), np.zeros((0, 3))

    mo_t, mo_xyth = [], []      # map → odom
    ob_t, ob_xyth = [], []      # odom → base_footprint

    for t_ns, msg in messages['/tf']:
        for tf in msg.transforms:
            tr = tf.transform
            r  = tr.rotation
            x  = float(tr.translation.x)
            y  = float(tr.translation.y)
            yaw = quaternion_to_yaw(r.x, r.y, r.z, r.w)
            t = int(t_ns) / 1e9
            if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                mo_t.append(t); mo_xyth.append((x, y, yaw))
            elif (tf.header.frame_id == 'odom'
                  and tf.child_frame_id == 'base_footprint'):
                ob_t.append(t); ob_xyth.append((x, y, yaw))

    if not mo_t or not ob_t:
        return np.zeros(0), np.zeros((0, 3))

    mo_t    = np.array(mo_t)
    mo_xyth = np.array(mo_xyth)
    ob_t    = np.array(ob_t)
    ob_xyth = np.array(ob_xyth)
    order   = np.argsort(mo_t); mo_t = mo_t[order]; mo_xyth = mo_xyth[order]
    order   = np.argsort(ob_t); ob_t = ob_t[order]; ob_xyth = ob_xyth[order]

    # Sample on the odom→base stream (high rate) and look up latest map→odom.
    out_t, out_xyth = [], []
    for i in range(len(ob_t)):
        t = ob_t[i]
        j = int(np.searchsorted(mo_t, t, side='right') - 1)
        if j < 0:
            continue                                     # no map→odom yet
        mo_x, mo_y, mo_yaw = mo_xyth[j]
        ob_x, ob_y, ob_yaw = ob_xyth[i]
        c, s = np.cos(mo_yaw), np.sin(mo_yaw)
        # Compose: map_T_base = map_T_odom * odom_T_base
        x_map = mo_x + c * ob_x - s * ob_y
        y_map = mo_y + s * ob_x + c * ob_y
        yaw_map = mo_yaw + ob_yaw
        out_t.append(t)
        out_xyth.append((x_map, y_map, yaw_map))

    return np.array(out_t), np.array(out_xyth)


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
    odom_t_raw, odom_xyth_raw = extract_odom(messages)
    map_t, map_xyth           = extract_robot_map_pose(messages)
    # Goals, plans, and obstacles all live in the *map* frame, so when the
    # bag has /tf we must compare against the AMCL-corrected robot pose, not
    # the dead-reckoned /odom (which drifts and was the cause of the bogus
    # success=False even after gmpc_node logged "Goal reached within 0.192m").
    pose_in_map = len(map_t) >= 2
    odom_t   = map_t   if pose_in_map else odom_t_raw
    odom_xyth = map_xyth if pose_in_map else odom_xyth_raw
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
        # Goal diagnostics (for debugging "didn't arrive" claims)
        'goal_xy_x'          : float('nan'),
        'goal_xy_y'          : float('nan'),
        'final_dist_to_goal_m': float('nan'),
        'min_dist_to_goal_m'  : float('nan'),
        'pose_source'   : 'map_tf' if pose_in_map else 'odom_raw',
        'n_odom'        : len(odom_t),
        'n_cmd'         : len(cmd_t),
        'n_plan'        : len(plans),
        'n_obstacles'   : len(obs_series),
        'n_solve_time'  : len(st_vals),
    }
    if len(odom_t) < 2 or not plans:
        return out

    first_plan_t = plans[0][0]

    # ------- robust goal detection -------
    # The very first plan's endpoint is the most reliable: the planner was
    # invoked right after the user published /goal_pose, so plans[0]'s last
    # pose IS the user-supplied goal. Later replans can be short residuals
    # near goal_tolerance, so we don't trust them.
    goal_xy = plans[0][1][-1]
    out['goal_xy_x'] = float(goal_xy[0])
    out['goal_xy_y'] = float(goal_xy[1])

    # ------- arrival time / success -------
    arrival_t = None
    in_run = odom_t >= first_plan_t
    dists_to_goal = np.linalg.norm(odom_xyth[in_run, :2] - goal_xy, axis=1)
    out['final_dist_to_goal_m'] = float(dists_to_goal[-1])
    out['min_dist_to_goal_m']   = float(np.min(dists_to_goal))
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
    'goal_xy_x', 'goal_xy_y', 'final_dist_to_goal_m', 'min_dist_to_goal_m',
    'pose_source',
    'success_source', 'controller_arrival_dist_m',
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

    # ----- log-based ground truth (authoritative for GMPC stacks) -----
    # Try to find the matching run_one_trial.sh log next to the bag, e.g.
    # bags/gmpc_cbf__seed3/ → logs/gmpc_cbf__seed3.log. The bag-based check
    # can falsely fail when AMCL pose-converges briefly close to goal in
    # between published /tf messages (the live controller sees it, but the
    # bag does not). When the controller node itself logged 'Goal reached',
    # we take that as ground truth and override `success`.
    bag_dir = Path(bag_path)
    log_path = bag_dir.parent.parent / 'logs' / f'{bag_dir.name}.log'
    log_info = parse_trial_log_for_goal(log_path)

    if log_info.get('controller_goal_reached'):
        metrics['success']      = True
        metrics['success_source'] = 'controller_log'
        # Prefer log-derived arrival time only if bag couldn't measure one.
        if (isinstance(metrics.get('arrival_time_s'), float)
                and np.isnan(metrics['arrival_time_s'])
                and 'controller_arrival_time_s' in log_info):
            metrics['arrival_time_s'] = log_info['controller_arrival_time_s']
    else:
        metrics['success_source'] = (
            'bag_pose' if metrics.get('success') else 'none'
        )
    # Carry the controller-reported distance through for transparency.
    metrics['controller_arrival_dist_m'] = log_info.get(
        'controller_arrival_dist_m', float('nan')
    )

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
