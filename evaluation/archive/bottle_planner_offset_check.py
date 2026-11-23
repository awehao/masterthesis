#!/usr/bin/env python3
"""Smac2D 輸出座標語義與「格心換算」候選的無 Isaac 驗證（Codex 20261102_083817；不改控制器、不實跑任務）。
獨立 ROS domain（預設 77）啟動：map_server（當趟地圖）、planner_server（bottle_n2_planner.yaml 的複本，只把 use_sim_time 改 false）、兩個 lifecycle_manager、
map→odom→base_footprint 靜態 TF。沒有 /scan（障礙層無標記；實跑另有掃描標記，揭露）。訂 /global_costmap/costmap 取實際規劃代價圖（解析度、原點、是否降採樣照讀）。
請求：
  A 既有紀錄：持瓶段（三趟 carry_req 起訖）、接近段（兩趟 plan.json 起訖）。
  B 事前固定種子 20261102：持瓶起點在 x∈[0.00, 0.10]、y∈[−0.05, 0.05] 均勻 20 點（終點＝online3 持瓶終點）；接近起點在 x∈[−3.65, −3.55]、y∈[−0.05, 0.05] 均勻 10 點（終點 (−0.9, 0)）。
每個回應：原始路徑（raw）、格心換算候選（raw＋res/2）。核對：
  首點／末點是否為起點格／目標格的格角或格心；raw→換算的換格點數；
  代價（NP.Grid＋NP.blocked：≥99 或未知即擋）分別查：raw 點、換算點、raw 與換算的 1 cm 加密連線、換算後重取樣＋平滑（同 build_carry）、
  完整承接路徑（持瓶段：build_carry(換算, p_pred＝起點, 方向 +x, x_entry＝終點 x, lane_y＝終點 y) 的 CarryPath 點以 1 cm 加密）；
  validate_plan（raw、換算）；承接偏差（build_carry 的 lateral；raw 與換算各一）。
任一換算後項目擋住 ⇒ 該請求的候選判不通過（不因 raw 通過而放行，不移動代價圖）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_planner_offset_check.py
"""
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_n2_path as NP                             # noqa: E402
import bottle_place_ctrl as PC                          # noqa: E402

DOMAIN = os.environ.get('CHECK_DOMAIN', '77')
OUT = os.path.join(HERE, 'results', 'bottle', 'planner_offset_check.json')
MAP_RUN = 'bottle_online4_u3_1'
CARRY_RUNS = ['bottle_online3_u3_1', 'bottle_online_u3_2', 'bottle_online4_u3_1']
APPROACH_RUNS = ['bottle_online3_u3_1', 'bottle_online4_u3_1']
SEED = 20261102


def launch(tmp):
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    y = open(os.path.join(HERE, 'bottle_n2_planner.yaml')).read().replace('use_sim_time: true', 'use_sim_time: false')
    yp = os.path.join(tmp, 'planner_nosim.yaml')
    open(yp, 'w').write(y)
    mp = os.path.join(HERE, 'runs', MAP_RUN, 'map.yaml')
    cmds = [['ros2', 'run', 'nav2_map_server', 'map_server', '--ros-args', '-p', 'use_sim_time:=false', '-p', f'yaml_filename:={mp}'],
            ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_map', '-p', 'use_sim_time:=false', '-p', 'autostart:=true', '-p', "node_names:=['map_server']"],
            ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'map', '--child-frame-id', 'odom'],
            ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'odom', '--child-frame-id', 'base_footprint'],
            ['ros2', 'run', 'nav2_planner', 'planner_server', '--ros-args', '--params-file', yp, '-p', 'use_sim_time:=false'],
            ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_planner', '-p', 'use_sim_time:=false', '-p', 'autostart:=true', '-p', "node_names:=['planner_server']"]]
    logf = open(os.path.join(tmp, 'planner_check_nodes.log'), 'w')
    return [subprocess.Popen(c, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True) for c in cmds]


def dense(P, step=0.01):
    P = np.asarray(P, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def analyse(kind, start, goal, raw, grid):
    res, ox, oy = grid.res, grid.ox, grid.oy
    raw = np.asarray(raw, float)
    conv = raw + res / 2.0
    cell = lambda p: (math.floor((p[0] - ox) / res), math.floor((p[1] - oy) / res))   # noqa: E731
    sc, gc = cell(start), cell(goal)
    corner = lambda c: np.array([ox + c[0] * res, oy + c[1] * res])                  # noqa: E731
    o = dict(kind=kind, start=np.round(start, 4).tolist(), goal=np.round(goal, 4).tolist(), n_raw=len(raw), res=res,
             first_is_start_corner=bool(np.allclose(raw[0], corner(sc), atol=1e-4)), last_is_goal_corner=bool(np.allclose(raw[-1], corner(gc), atol=1e-4)),
             start_to_raw_first_mm=round(float(np.linalg.norm(raw[0] - start)) * 1e3, 1), start_to_conv_first_mm=round(float(np.linalg.norm(conv[0] - start)) * 1e3, 1),
             on_lattice_frac=round(float(np.mean([abs((v[0] - ox) / res - round((v[0] - ox) / res)) < 1e-3 and abs((v[1] - oy) / res - round((v[1] - oy) / res)) < 1e-3 for v in raw])), 2),
             cell_changes=int(sum(cell(a) != cell(b) for a, b in zip(raw, conv))))
    blk = lambda P: int(NP.blocked(grid.cost(P)).sum())                                  # noqa: E731
    o['blocked'] = dict(raw_pts=blk(raw), conv_pts=blk(conv), raw_dense=blk(dense(raw)), conv_dense=blk(dense(conv)))
    sm = PC.smooth_path(NP.resample(conv), PC.REG['smooth_iters'])
    o['blocked']['conv_smoothed_dense'] = blk(dense(sm))
    o['validate'] = dict(raw=NP.validate_plan('map', raw, np.asarray(start), np.asarray(goal), grid), conv=NP.validate_plan('map', conv, np.asarray(start), np.asarray(goal), grid))
    if kind == 'carry':
        for nm, P in (('raw', raw), ('conv', conv)):
            cp, info = PC.build_carry(P, np.asarray(start, float), np.array([1.0, 0.0]), float(goal[0]), float(goal[1]))
            o[f'join_{nm}'] = dict(ok=cp is not None, lateral_mm=None if info.get('lateral') is None else round(info['lateral'] * 1e3, 1), why=info.get('why'))
            if cp is not None and nm == 'conv':
                o['blocked']['conv_join_path_dense'] = blk(dense(cp.P))
    b = o['blocked']
    o['candidate_pass'] = bool(b['conv_pts'] == 0 and b['conv_dense'] == 0 and b['conv_smoothed_dense'] == 0 and b.get('conv_join_path_dense', 0) == 0 and o['validate']['conv'][0]
                               and (kind != 'carry' or o['join_conv']['ok']))
    return o


def main():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from nav2_msgs.action import ComputePathToPose
    from nav_msgs.msg import OccupancyGrid
    tmp = os.environ.get('CHECK_TMP', '/tmp')
    procs = launch(tmp)
    os.environ['ROS_DOMAIN_ID'] = DOMAIN
    rclpy.init()
    node = rclpy.create_node('planner_offset_check')
    st = dict(grid=None)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def gcb(m):
        i = m.info
        st['grid'] = NP.Grid(m.data, i.resolution, i.origin.position.x, i.origin.position.y, i.width, i.height, 0.0)
    node.create_subscription(OccupancyGrid, '/global_costmap/costmap', gcb, qos)
    ac = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
    out = dict(domain=DOMAIN, map_run=MAP_RUN, requests=[], note='無 /scan（障礙層無標記）；use_sim_time false；planner 參數同 bottle_n2_planner.yaml')
    try:
        t0 = time.time()
        while (st['grid'] is None or not ac.server_is_ready()) and time.time() - t0 < 60:
            rclpy.spin_once(node, timeout_sec=0.2)
        if st['grid'] is None or not ac.server_is_ready():
            raise RuntimeError('costmap 或 planner 未就緒')
        g = st['grid']
        out['costmap'] = dict(res=g.res, origin=[g.ox, g.oy], w=g.w, h=g.h)

        def plan(start, goal):
            from geometry_msgs.msg import PoseStamped
            def pose(x, y):
                p = PoseStamped()
                p.header.frame_id = 'map'
                p.pose.position.x, p.pose.position.y = float(x), float(y)
                p.pose.orientation.w = 1.0
                return p
            gl = ComputePathToPose.Goal()
            gl.goal, gl.start, gl.use_start, gl.planner_id = pose(*goal), pose(*start), True, 'GridBased'
            f = ac.send_goal_async(gl)
            rclpy.spin_until_future_complete(node, f, timeout_sec=10)
            h = f.result()
            if h is None or not h.accepted:
                return None
            rf = h.get_result_async()
            rclpy.spin_until_future_complete(node, rf, timeout_sec=10)
            r = rf.result()
            if r is None or len(r.result.path.poses) < 2:
                return None
            return np.array([[p.pose.position.x, p.pose.position.y] for p in r.result.path.poses])

        reqs = []
        for run in CARRY_RUNS:
            cr = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))['place']['carry_req']
            reqs.append(('carry', f'rec:{run}', cr['start'][:2], cr['goal']))
        for run in APPROACH_RUNS:
            pj = json.load(open(os.path.join(HERE, 'runs', run, 'plan.json')))['plan']
            reqs.append(('approach', f'rec:{run}', pj['start'][:2], pj['goal']))
        rng = np.random.default_rng(SEED)
        g3 = json.load(open(os.path.join(HERE, 'runs', 'bottle_online3_u3_1', 's1_report.json')))['place']['carry_req']['goal']
        for k in range(20):
            reqs.append(('carry', f'seed{k}', [float(rng.uniform(0.0, 0.10)), float(rng.uniform(-0.05, 0.05))], g3))
        for k in range(10):
            reqs.append(('approach', f'seed{k}', [float(rng.uniform(-3.65, -3.55)), float(rng.uniform(-0.05, 0.05))], [-0.9, 0.0]))
        for kind, tag, s_, g_ in reqs:
            raw = plan(s_, g_)
            if raw is None:
                out['requests'].append(dict(kind=kind, tag=tag, start=s_, goal=g_, error='no_path'))
                continue
            o = analyse(kind, np.array(s_, float), np.array(g_, float), raw, g)
            o['tag'] = tag
            out['requests'].append(o)
            print(tag, kind, {k: o.get(k) for k in ('first_is_start_corner', 'start_to_raw_first_mm', 'start_to_conv_first_mm', 'cell_changes', 'blocked', 'candidate_pass')},
                  {k: o.get(k) for k in ('join_raw', 'join_conv')}, flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1.5)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    R = [r for r in out['requests'] if 'error' not in r]
    carry = [r for r in R if r['kind'] == 'carry']
    out['summary'] = dict(n=len(R), n_err=len(out['requests']) - len(R), all_first_corner=all(r['first_is_start_corner'] for r in R), all_last_corner=all(r['last_is_goal_corner'] for r in R),
                          candidate_pass=sum(r['candidate_pass'] for r in R), cell_changes_mean=round(float(np.mean([r['cell_changes'] for r in R])), 1) if R else None,
                          raw_blocked_any=sum(any(v for k, v in r['blocked'].items() if k.startswith('raw')) for r in R),
                          conv_blocked_any=sum(any(v for k, v in r['blocked'].items() if k.startswith('conv')) for r in R),
                          join_raw_fail=sum(not r['join_raw']['ok'] for r in carry), join_conv_fail=sum(not r['join_conv']['ok'] for r in carry),
                          join_raw_lat_max=max((r['join_raw']['lateral_mm'] or 0) for r in carry) if carry else None, join_conv_lat_max=max((r['join_conv']['lateral_mm'] or 0) for r in carry) if carry else None)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('摘要', out['summary'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
