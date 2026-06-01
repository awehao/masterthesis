# Evaluation framework — RPP vs MPPI Omni vs SE(2) GMPC

Pure-Python scripts (no ROS package — run from anywhere) that:

1. **record** a rosbag with the right topics during a Nav2 run
2. **analyze** the bag → one CSV row per run (success / arrival_time / RMSE / smoothness / jerk / ...)
3. **plot** the CSV → side-by-side bar charts for the report

## Topics recorded

| topic | type | what it gives us |
|-------|------|------------------|
| `/odom` | `nav_msgs/Odometry` | actual pose history (ground truth in Gazebo) |
| `/cmd_vel` | `geometry_msgs/Twist` | the command actually sent to the chassis |
| `/cmd_vel_nav` | `geometry_msgs/Twist` | pre-velocity-smoother cmd (only present for RPP/MPPI) |
| `/plan` | `nav_msgs/Path` | global path emitted by the planner |
| `/goal_pose` | `geometry_msgs/PoseStamped` | the goal that triggered the run |
| `/tf`, `/tf_static` | transforms | for replay / debugging |

## Workflow

For each method × seed × goal:

```bash
# 1. Start Gazebo (same world for all methods!)
ros2 launch ammr_bringup gazebo.launch.py

# 2. Start the controller stack of choice
#    Baseline A:  RPP
ros2 launch ammr_navigation nav2.launch.py
#    Baseline B:  MPPI Omni
ros2 launch ammr_navigation nav2_omni_mppi.launch.py
#    Ours:        SE(2) GMPC
ros2 launch ammr_wholebody_mpc gmpc_nav2.launch.py

# 3. Start recording (T-record)
cd ~/masterthesis/evaluation
./record.sh gmpc seed42_goal1 60         # method, run-tag, duration_seconds

# 4. While recording is on, send a goal
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0}, orientation: {w: 1.0}}}" --once

# 5. Wait for robot to arrive (or recording to time out)
```

## Analysis

```bash
cd ~/masterthesis/evaluation

# Process one bag → append a row to results/runs.csv
python3 analyze.py bags/gmpc__seed42_goal1 --method gmpc --run seed42_goal1

# Once all runs are done, plot
python3 plot.py
# → results/figs/summary.png    (3×3 grid, all metrics)
# → results/figs/<metric>.png   (one panel per metric for the report)
# Also prints a text table to stdout.
```

## Fair-comparison rules

- **Same Gazebo world** for all three controllers (random map seed pinned)
- **Same start pose** (Gazebo spawn at origin)
- **Same goal** (or set of goals)
- **Recording window** covers the full attempt (use generous duration, e.g. 90 s)
- Repeat each method ≥3 times to estimate variance

## Metrics computed by `analyze.py`

| column | meaning | lower = better? |
|--------|---------|-----------------|
| `success` | reached within 0.25 m of goal before bag end | — |
| `arrival_time_s` | time from first `/plan` to within tolerance | yes |
| `total_time_s` | total bag duration | — |
| `path_length_m` | cumulative |Δp| of `/odom` | yes (proxy for efficiency) |
| `tracking_rmse_m` | RMS distance from odom pose to closest point on the *latest* plan | yes |
| `smooth_vx/vy/wz` | std of cmd_vel components | yes |
| `jerk_vx/vy/wz` | std of cmd_vel finite-difference accel | yes |

## Honesty notes

- `tracking_rmse_m` is "distance to *some* point of the latest plan", not "cross-track to the closest path edge" — fine for first cut, swap to true cross-track if a reviewer asks.
- `smooth_*` and `jerk_*` are aggregated `std`, which conflates intentional accelerations (start/stop) with chatter. Acceptable for cross-method comparison if windows are similar lengths.
- `success` does not check yaw alignment at the goal. Add if needed.
