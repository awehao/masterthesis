#!/usr/bin/env python3
"""整合版（dynsr43＋雙相機候選節點＋站位工作者）不動作接線檢查（Codex 20261111_134437：修補兩項後、凍結前）。不起 Isaac、不動機器人。
A 靜態：執行端 dynsr43 的控制器、轉接器、站位車道參數、耗時紀錄；控制器方法實際解析到哪個類別（子類鏈沒有被誤蓋）；起跑腳本接受 CAND_SCRIPT。
B 真 DDS 冒煙（獨立網域）：以子程序啟動 bottle_seg3d_candidates_dual_node，假 /v1、/w1 列（區域外世界座標）→ 節點實際發布
  /bottle/target_guide 與 /bottle/target_candidates(+checks) → 經 GuideLink／SegTargetLink 驗證與轉送（記拒收原因）；節點收到 SIGINT 正常結束。
    source /opt/ros/jazzy/setup.bash; /usr/bin/python3 evaluation/motm_obs_wiring_check.py OUT.json A          （系統 Python：控制器鏈需 osqp）
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=87 .venv-dl0/bin/python evaluation/motm_obs_wiring_check.py OUT.json B   （候選節點環境）
"""
import inspect
import json
import os
import signal
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src', 'ammr_wholebody_mpc'))

PART = sys.argv[2]
out = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
out[PART] = {}


def part_A():
    # ---------- A 靜態 ----------
    src = open(os.path.join(HERE, 'isaac_bottle_w1dynsr43.py')).read()
    run_sh = open(os.path.join(HERE, 'run_bottle_ros.sh')).read()
    need = {
        'controller_obs': "RC.import_with_real_ros('bottle_w1_obs_ctrl'",
        'guide_link_made': 'GL.GuideLink(node, ctrl, Float64MultiArray, region_offset=RO)',
        'guide_link_T0': 'glink.set_T0(T0)',
        'guide_link_tick': 'glink.tick(t)',
        'guide_link_report': "rep['guide_link'] = glink.report()",
        'seg_link_tick': 'slink.tick(t)',
        'lane_planner_y_reach': 'def _lane_planner(obs, t_task, sig_robot, d_stop, y_reach=None)',
        'lane_planner_injected': 'ctrl.lane_planner = _lane_planner',
        'timing_ctrl_total': "('ctrl_step_total', ctrl_ms)",
        'runner_cand_script': '${CAND_SCRIPT:-evaluation/bottle_seg3d_candidates_node.py}',
    }
    out['A']['static'] = {k: (v in src) if k != 'runner_cand_script' else (v in run_sh) for k, v in need.items()}
    i0 = src.find('def _lane_planner(')
    out['A']['decide_passes_y_reach'] = 'y_reach=y_reach' in src[i0:i0 + 1200]
    import bottle_ros_chain as RC
    OC, = RC.import_with_real_ros('bottle_w1_obs_ctrl')
    C = OC.W1PlaceController
    meths = ('step', '_vstep', '_advance_inner', '_gp_submit', '_lane_decide', '_lane_from_guide', '_standoff', '_pick_standoff', 'guide_intake', 'candidate_intake',
             '_plan_yaw', '_tpa_start', '_v_target', '_rel_target', '_unfolded', 'report')
    res_m = {}
    for m in meths:
        f = getattr(C, m, None)
        res_m[m] = None if f is None else inspect.getmodule(f).__name__
    out['A']['method_owner'] = res_m
    out['A']['mro'] = [k.__module__ for k in C.__mro__ if k.__module__.startswith('bottle')]
    exp = dict(_vstep='bottle_w1_obs_ctrl', _advance_inner='bottle_w1_obs_ctrl', _gp_submit='bottle_w1_obs_ctrl', _lane_decide='bottle_w1_obs_ctrl', guide_intake='bottle_w1_obs_ctrl',
               _plan_yaw='bottle_w1_head_ctrl', _tpa_start='bottle_w1_head_ctrl', _v_target='bottle_w1_head_ctrl')
    out['A']['owner_as_expected'] = {k: res_m.get(k) == v for k, v in exp.items()}
    for k in ('bottle_w1_tpas_ctrl', 'bottle_w1_tpa_cj_ctrl', 'bottle_w1_tpa_ctrl', 'bottle_w1_tlane_ctrl'):
        out['A']['owner_as_expected'][f'mro_has_{k}'] = k in out['A']['mro']
    print('A', json.dumps(dict(static=out['A']['static'], decide=out['A']['decide_passes_y_reach'], owner=out['A']['owner_as_expected']), ensure_ascii=False), flush=True)


def part_B():
    # ---------- B 真 DDS 冒煙 ----------
    import rclpy                                                            # noqa: E402
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension        # noqa: E402
    import bottle_guide_link as GL                                          # noqa: E402
    import bottle_segtarget_link as SL                                      # noqa: E402
    import bottle_seg3d_node as SN                                          # noqa: E402

    log = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'motm_obs_wiring_dual_node.jsonl')
    if os.path.exists(log):
        os.remove(log)
    proc = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'bottle_seg3d_candidates_dual_node.py'), '--log', log, '--ros-args', '-p', 'use_sim_time:=false'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = rclpy.create_node('motm_obs_wiring_check')
    pub = dict(v1=node.create_publisher(Float64MultiArray, '/v1/bottle_seg3d', 50), w1=node.create_publisher(Float64MultiArray, '/w1/bottle_seg3d', 50))
    g_in, c_in = [], []
    ctrl = types.SimpleNamespace(guide_intake=lambda t, g: g_in.append(dict(t=t, **g)), candidate_intake=lambda t, c: c_in.append(dict(t=t, **c)))
    glink = GL.GuideLink(node, ctrl, Float64MultiArray, region_offset=(0.6, 0.8))
    slink = SL.SegTargetLink(node, ctrl, Float64MultiArray)
    glink.set_T0(0.0)
    slink.set_T0(0.0)
    XW, YW = 0.6 - 0.0232, 0.8 + 0.4081                                     # 世界座標（區域偏移 0.6,0.8；名目瓶位）


    def now():
        return node.get_clock().now().nanoseconds * 1e-9


    def msg(cam, sup):
        d = {f: 0.0 for f in SN.FIELDS}
        d.update(stamp=now() - 0.05, valid=1.0, xy_valid=1.0, support_observed=1.0 if sup else 0.0, top_observed=0.0, height_valid=0.0, cx_known_r=XW, cy_known_r=YW, z_support=0.58,
                 z_top=float('nan'))
        m = Float64MultiArray()
        m.layout.dim = [MultiArrayDimension(label=SN.LAYOUT_LABEL, size=len(SN.FIELDS), stride=len(SN.FIELDS))]
        m.data = [float(d[f]) for f in SN.FIELDS]
        return m


    t0 = time.time()
    while time.time() - t0 < 15.0 and node.count_subscribers('/w1/bottle_seg3d') + node.count_subscribers('/v1/bottle_seg3d') < 2:
        rclpy.spin_once(node, timeout_sec=0.1)
    out['B']['subscribers_found_s'] = round(time.time() - t0, 2)
    out['B']['subs'] = dict(v1=node.count_subscribers('/v1/bottle_seg3d'), w1=node.count_subscribers('/w1/bottle_seg3d'))
    out['B']['pubs'] = dict(guide=node.count_publishers('/bottle/target_guide'), cand=node.count_publishers('/bottle/target_candidates'),
                            chk=node.count_publishers('/bottle/target_candidate_checks'))
    t1 = time.time()
    k = 0
    while time.time() - t1 < 6.0:
        k += 1
        pub['v1'].publish(msg('v1', False))                                 # 底盤：只有 xy
        if time.time() - t1 > 1.5:
            pub['w1'].publish(msg('w1', True))                              # 1.5 s 後手腕：xy＋支撐
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.01)
        glink.tick(now())
        slink.tick(now())
        time.sleep(0.05)
    proc.send_signal(signal.SIGINT)
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        rc = 'killed'
    txt = proc.stdout.read().decode(errors='ignore')
    node.destroy_node()
    rclpy.shutdown()
    ready_adopt = [c for c in c_in if c['ready'] == 1.0 and c['snap_ik'] == 2.0]
    out['B'].update(node_rc=rc, node_traceback=('Traceback' in txt), node_tail=txt[-600:], guide=dict(n=len(g_in), rejected=glink.rej, first=g_in[0] if g_in else None,
                    cams=sorted({str(g['cam']) for g in g_in}), states=sorted({g['state'] for g in g_in}), last=g_in[-1] if g_in else None),
                    cand=dict(n=len(c_in), rejected=slink.rej, n_merged=slink.n_merged, n_adoptable=len(ready_adopt), seqs=sorted({int(c['snap_seq']) for c in c_in}),
                              adopt_xy_region=None if not ready_adopt else [round(ready_adopt[-1]['snap_cx'], 4), round(ready_adopt[-1]['snap_cy'], 4)]))
    gc = sorted({str(g.get('grasp_cam')) for g in g_in})
    out['B']['guide']['grasp_cams'] = gc
    ok_g = bool(g_in) and not glink.rej and 'v1' in out['B']['guide']['cams'] and 'w1' in gc        # 導引採用遲滯：先採用底盤 v1 即保持；抓取角色來自手腕 w1
    xy_g = g_in[-1]['xy'] if g_in and 'xy' in g_in[-1] else None
    out['B']['verdict'] = dict(node_clean_exit=(rc == 0 and not out['B']['node_traceback']), guide_v1_grasp_w1=ok_g,
                               guide_xy_region_ok=xy_g is not None and abs(xy_g[0] + 0.0232) < 1e-3 and abs(xy_g[1] - 0.4081) < 1e-3,
                               cand_adoptable=len(ready_adopt) > 0, cand_rejects_expected=set(slink.rej) <= {'no_snapshot_yet', 'snapshot_too_old'})   # 快照 3 s 界線：轉送當下逾期即拒（失效關閉）
    print('B', json.dumps(dict(verdict=out['B']['verdict'], subs=out['B']['subs'], pubs=out['B']['pubs'], guide_rej=glink.rej, cand_rej=slink.rej, cams=out['B']['guide']['cams'], grasp_cams=gc,
                               states=out['B']['guide']['states'], seqs=out['B']['cand']['seqs'], xy=xy_g), ensure_ascii=False, default=str), flush=True)


part_A() if PART == 'A' else part_B()
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)
