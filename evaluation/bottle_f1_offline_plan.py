#!/usr/bin/env python3
"""F1 兩瓶連續任務的離線規劃核對（不用 Isaac；獨立 ROS domain；bottle_region_offline_plan 的 F1 版）。地圖＝第二桌 (2.4, −1.0)，無區域平移。
請求：持瓶遠端／近端（起訖點取 results/bottle/f1_geom.json 當趟 carry_req）、返回遠端／近端停車位 → 原起點 (−3.6, 0)。
核對：validate_plan（與實跑同一函式）、重取樣後最大代價、阻擋格數。輸出 results/bottle/f1_offline_plan.json。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_f1_offline_plan.py
"""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
DOMAIN = os.environ.get('CHECK_DOMAIN', '79')
GEOM, KEYS, OUT = (('f1_geom_rev.json', ('r1', 'r2'), 'f1_offline_plan_rev.json') if os.environ.get('F1_ORDER') == 'rev' else ('f1_geom.json', ('c1', 'c2'), 'f1_offline_plan.json'))   # F1_ORDER=rev：對調抓取順序（r1 遠端、r2 近端）
OUT = os.path.join(HERE, 'results', 'bottle', OUT.replace('.json', '_box.json') if os.environ.get('F1_BOX') == '1' else OUT)
WALLS = os.environ.get('F1_WALLS', '')                  # 場景 JSON：牆體進地圖；輸出檔名另加 _lchan
if WALLS:
    OUT = OUT.replace('.json', '_lchan.json')
    _pf = json.load(open(WALLS, encoding='utf-8')).get('plan_file')
START = [-3.6, 0.0]
DX = float(os.environ.get('F1_T2_DX', '0'))              # 只供設計比較：第二桌與放置點整體東移（公尺）；輸出檔名另加 _dx
if DX:
    OUT = OUT.replace('.json', f'_dx{DX:g}.json')
if WALLS and _pf:                                       # 場景檔指定輸出檔名（版本化，不覆蓋舊場景的路徑）
    OUT = os.path.join(HERE, 'results', 'bottle', _pf)
CONT = os.environ.get('F1_CONT') == '1'                 # 連續交接：多一條停車位到第二瓶轉向點 G 的路徑；輸出另存
if CONT:
    OUT = OUT.replace('.json', '_cont.json')


def main():
    tmp = os.environ.get('CHECK_TMP', '/tmp')
    mp = os.path.join(tmp, 'f1_map')
    subprocess.run([sys.executable, os.path.join(HERE, 'bottle_map_write.py'), mp, '--table2', f'{2.4 + DX},-1.0'] + (['--wall-box=-2.0,0.0,0.25,0.4,0.4,0.5'] if os.environ.get('F1_BOX') == '1' else []) + (['--walls-file', WALLS] if WALLS else []), check=True)   # F1_BOX：原障礙方塊進地圖（實跑靠掃描進代價圖，離線沒有掃描）
    os.environ['ROS_DOMAIN_ID'] = DOMAIN
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    y = open(os.path.join(HERE, 'bottle_n2_planner.yaml')).read().replace('use_sim_time: true', 'use_sim_time: false')
    yp = os.path.join(tmp, 'planner_nosim_f1.yaml')
    open(yp, 'w').write(y)
    lm = ['-p', 'use_sim_time:=false', '-p', 'autostart:=true', '-p']
    cmds = [['ros2', 'run', 'nav2_map_server', 'map_server', '--ros-args', '-p', 'use_sim_time:=false', '-p', f'yaml_filename:={mp}.yaml'],
            ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_map'] + lm + ['node_names:=[map_server]'],
            ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'map', '--child-frame-id', 'odom'],
            ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'odom', '--child-frame-id', 'base_footprint'],
            ['ros2', 'run', 'nav2_planner', 'planner_server', '--ros-args', '--params-file', yp, '-p', 'use_sim_time:=false'],
            ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_planner'] + lm + ['node_names:=[planner_server]']]
    logf = open(os.path.join(tmp, 'f1_plan_nodes.log'), 'w')
    procs = [subprocess.Popen(c, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True) for c in cmds]
    import rclpy
    from rclpy.node import Node
    import bottle_n2_path as NP
    import bottle_n2_ros_center as NRC
    import bottle_region as BRG

    class N(Node):
        def _w(self, f):
            return f

        def stamp(self, t):
            from builtin_interfaces.msg import Time
            m = Time()
            m.sec, m.nanosec = int(t), int((t - int(t)) * 1e9)
            return m
    G1 = json.load(open(os.path.join(HERE, 'results', 'bottle', GEOM)))
    rclpy.init()
    node = N('f1_offline_plan')
    R = BRG.RegionFrame(0.0, 0.0)
    out = dict(domain=DOMAIN, map=mp + '.yaml')
    try:
        time.sleep(2.0)
        conv = NRC.verify(node, mp + '.yaml', timeout=20.0)
        out['verify'] = conv
        pr = BRG.PlanRosRegion(node, conv, R)
        t0 = time.time()
        while (pr.grid is None or not pr.ac.server_is_ready()) and time.time() - t0 < 40:
            rclpy.spin_once(node, timeout_sec=0.2)
        reqs = {}
        for nm, k in zip(('far', 'near'), KEYS):
            cr = G1[k]['carry_req']
            reqs['carry_' + nm] = (list(cr['start']), [cr['goal'][0] + DX, cr['goal'][1]])
            pk = [G1[k]['park'][0] + DX, G1[k]['park'][1]]
            reqs['return_' + nm] = ([float(pk[0]), float(pk[1]), 0.0], START)
            if nm == 'far':                                    # 短返回：終點＝接近路徑（bottle_online5_u3_1 實跑規劃）上 x＝RESTART_X 的投影點
                J_ = np.array(json.load(open(os.path.join(HERE, 'runs', 'bottle_online5_u3_1', 's1_report.json')))['plan']['joined'], float)
                import bottle_f1_task as FT_
                out['restart'] = dict(zip(('sig', 'xy'), FT_.restart_point(NP.PlannedPath(J_, -0.9))))
                reqs['return_short'] = ([float(pk[0]), float(pk[1]), 0.0], out['restart']['xy'])
                if CONT:
                    import bottle_f1c_path as CP_
                    reqs['transit_far'] = ([float(pk[0]), float(pk[1]), 0.0], list(CP_.G_TRANSIT))
        for nm, (s_, g_) in reqs.items():
            rid = pr.request(tuple(s_), tuple(g_), 0.0)
            t1 = time.time()
            while (pr.result is None or pr.result.get('id') != rid) and time.time() - t1 < 30:
                rclpy.spin_once(node, timeout_sec=0.1)
                pr.poll()
            r = pr.result
            if r is None or not r.get('ok'):
                out[nm] = dict(error='no_path', start=s_, goal=g_)
                print(nm, 'no_path')
                continue
            P = np.asarray(r['P'], float)
            g = pr.grid
            ok_v, why_v = NP.validate_plan('map', P, np.array(s_[:2]), np.array(g_), g)
            RS = NP.resample(P)
            cost = g.cost(RS)
            L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
            out[nm] = dict(start=s_, goal=g_, n=len(P), length_m=round(L, 3), validate=[bool(ok_v), why_v], cost_max=int(cost.max()), n_blocked=int(NP.blocked(cost).sum()), planning_ms=r.get('wall_ms'), x_range=[round(float(P[:, 0].min()), 3), round(float(P[:, 0].max()), 3)],
                           y_range=[round(float(P[:, 1].min()), 3), round(float(P[:, 1].max()), 3)], P_region=P.round(5).tolist())
            print(nm, {k_: out[nm][k_] for k_ in ('n', 'length_m', 'validate', 'cost_max', 'n_blocked', 'x_range', 'y_range')})
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
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
