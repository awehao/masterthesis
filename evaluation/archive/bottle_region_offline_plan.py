#!/usr/bin/env python3
"""R-B1 區域座標的規劃接口與往返一致性核對（Codex 20261102_090007；不用 Isaac、不實跑任務）。R-B2 起以環境變數 REGION_OFS／REGION_TAG 指定區域（預設＝R-B1，輸出不變）。
獨立 ROS domain（預設 79）：map_server（bottle_map_write --table2 2.4,-1.0 --region-offset ox,oy 產生的 R-B1 地圖）＋planner_server（同 bottle_planner_offset_check.launch）。
以 bottle_region.PlanRosRegion（含 bottle_n2_ros_center.verify 的版本限定格心換算）在區域座標請求：
  接近段：起點＝世界 (−3.6, 0) 換區域、終點＝區域 (−0.9, 0)（＝世界 (−0.9＋ox, oy)）；
  持瓶段：起點＝區域 (0.10, 0)（抓取完成附近）、終點＝區域放置停車位 − entry_back（PLACE 換區域後同 PlaceController 的算法；停車 xy 取名目 T_rel 近似，只供假世界）。
核對：verify 啟用；回覆換算各一次（conv、region 各一筆）；P_world ＝ P_region＋O（逐點）；世界首點＝起點格格心（換算後）；RegionGrid.cost(P_region)＝世界格 cost(P_world)；
validate_plan（區域）；join_lane（區域，x_lane −0.9）後代價。輸出 results/bottle/rb1_offline_plan.json（含區域路徑，供假世界閉迴路使用）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_region_offline_plan.py
"""
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
OFS = tuple(float(v) for v in os.environ.get('REGION_OFS', '0.60,0.80').split(','))        # 預設＝R-B1；R-B2：REGION_OFS=0.8,-0.8 REGION_TAG=rb2
TAG = os.environ.get('REGION_TAG', 'rb1')
DOMAIN = os.environ.get('CHECK_DOMAIN', '79')
OUT = os.path.join(HERE, 'results', 'bottle', f'{TAG}_offline_plan.json')


def main():
    import subprocess
    tmp = os.environ.get('CHECK_TMP', '/tmp')
    mp = os.path.join(tmp, f'{TAG}_map')
    subprocess.run([sys.executable, os.path.join(HERE, 'bottle_map_write.py'), mp, '--table2', '2.4,-1.0', '--region-offset', f'{OFS[0]},{OFS[1]}'], check=True)
    os.environ['ROS_DOMAIN_ID'] = DOMAIN
    import bottle_planner_offset_check as CK
    CK.DOMAIN = DOMAIN
    CK.MAP_RUN = None
    _launch = CK.launch

    def launch_map(tmp_):
        # 同 CK.launch，但地圖換成 R-B1
        env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
        y = open(os.path.join(HERE, 'bottle_n2_planner.yaml')).read().replace('use_sim_time: true', 'use_sim_time: false')
        yp = os.path.join(tmp_, f'planner_nosim_{TAG}.yaml')
        open(yp, 'w').write(y)
        cmds = [['ros2', 'run', 'nav2_map_server', 'map_server', '--ros-args', '-p', 'use_sim_time:=false', '-p', f'yaml_filename:={mp}.yaml'],
                ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_map', '-p', 'use_sim_time:=false', '-p', 'autostart:=true', '-p', "node_names:=['map_server']"],
                ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'map', '--child-frame-id', 'odom'],
                ['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '--frame-id', 'odom', '--child-frame-id', 'base_footprint'],
                ['ros2', 'run', 'nav2_planner', 'planner_server', '--ros-args', '--params-file', yp, '-p', 'use_sim_time:=false'],
                ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_planner', '-p', 'use_sim_time:=false', '-p', 'autostart:=true', '-p', "node_names:=['planner_server']"]]
        logf = open(os.path.join(tmp_, f'{TAG}_plan_nodes.log'), 'w')
        return [subprocess.Popen(c, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True) for c in cmds]
    del _launch
    procs = launch_map(tmp)
    import rclpy
    from rclpy.node import Node
    import bottle_n2_path as NP
    import bottle_n2_ros_center as NRC
    import bottle_place_ctrl as PC
    import bottle_region as BRG

    class N(Node):
        def _w(self, f):
            return f

        def stamp(self, t):
            from builtin_interfaces.msg import Time
            m = Time()
            m.sec, m.nanosec = int(t), int((t - int(t)) * 1e9)
            return m
    rclpy.init()
    node = N('rb1_offline_plan')
    R = BRG.RegionFrame(*OFS)
    out = dict(offset=list(OFS), domain=DOMAIN, map=mp + '.yaml')
    try:
        time.sleep(2.0)
        conv = NRC.verify(node, mp + '.yaml', timeout=20.0)
        out['verify'] = conv
        pr = BRG.PlanRosRegion(node, conv, R)
        t0 = time.time()
        while (pr.grid is None or not pr.ac.server_is_ready()) and time.time() - t0 < 40:
            rclpy.spin_once(node, timeout_sec=0.2)
        place_r = R.place_to_region(PC.PLACE)
        park_r = np.array([place_r['target'][0], place_r['lane_y']])            # 近似：停車 y＝放置車道、x＝放置目標 x（假世界用；實跑由 PlaceController 依 T_rel 求）
        reqs = dict(approach=(R.to_region([-3.6, 0.0]).tolist() + [0.0], [-0.9, 0.0]),
                    carry=([0.10, 0.0, 0.0], [float(park_r[0] - place_r['entry_back']), float(park_r[1])]))
        for nm, (s_, g_) in reqs.items():
            rid = pr.request(tuple(s_), tuple(g_), 0.0)
            t1 = time.time()
            while (pr.result is None or pr.result.get('id') != rid) and time.time() - t1 < 20:
                rclpy.spin_once(node, timeout_sec=0.1)
                pr.poll()
            r = pr.result
            if r is None or not r.get('ok'):
                out[nm] = dict(error='no_path', start=s_, goal=g_)
                continue
            P_reg = np.asarray(r['P'], float)
            P_w = np.asarray(r['P_world'], float)
            g = pr.grid
            wg = NP.Grid(g.a.reshape(-1), g.res, g.ox, g.oy, g.w, g.h, g.stamp)
            ok_v, why_v = NP.validate_plan('map', P_reg, np.array(s_[:2]), np.array(g_), g)
            res_ = conv['map_res']
            sw = R.to_world(s_[:2])
            cell_c = np.array([g.ox + (math.floor((sw[0] - g.ox) / res_) + 0.5) * res_, g.oy + (math.floor((sw[1] - g.oy) / res_) + 0.5) * res_])
            o = dict(start_region=s_, goal_region=g_, start_world=sw.tolist(), goal_world=R.to_world(g_).tolist(), n=len(P_reg), conv=r.get('conv'), region=r.get('region'),
                     roundtrip_max_err=float(np.abs(P_w - (P_reg + R.O)).max()), first_world=P_w[0].tolist(), first_is_start_cell_center=bool(np.allclose(P_w[0], cell_c, atol=1e-4)),
                     grid_cost_equal=bool(np.array_equal(g.cost(P_reg), wg.cost(P_w))), validate=[bool(ok_v), why_v],
                     cost_max_region=int(g.cost(NP.resample(P_reg)).max()), P_region=P_reg.round(5).tolist())
            if nm == 'approach':
                J = NP.join_lane(P_reg, -0.9)
                o['joined_cost_blocked'] = int(NP.blocked(g.cost(J)).sum())
                o['joined_region'] = J.round(5).tolist()
            out[nm] = o
            print(nm, {k: o[k] for k in ('n', 'roundtrip_max_err', 'first_is_start_cell_center', 'grid_cost_equal', 'validate', 'cost_max_region')}, o.get('joined_cost_blocked'), flush=True)
        out['place_region'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in place_r.items()}
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
    print('verify', out.get('verify'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
