#!/usr/bin/env python3
"""右側整合（dynsr44＋dual_side 節點＋bottle_w1_side_ctrl）不動作接線檢查。不起 Isaac、不動機器人。
A 靜態：執行端 dynsr44 的鏡射旗標、區域先驗注入、側別發布與每拍呼叫、車道規劃帶 side、放置先驗鏡射、控制器類別；地圖寫入鏡射旗標；
  右側控制器方法解析；左右兩條鏈的 hook（_tcp_c、_grasp_T、_t1）都能解析。
B 真 DDS（獨立網域）：dual_side 節點；假 /v1、/w1 列（鏡射場景右側瓶，世界 (0.5768, −1.2081)）＋ /bottle/grasp_side：
  未選側 ⇒ 不建快照（候選列無序號）；發布 right ⇒ 建快照、導引訊息 chk_side＝2、chk_side_seq0≥1，候選可達（右側檢查）可採用；
  改發 left ⇒ 起算序號前移、舊快照作廢；節點 SIGINT 正常結束。
    source /opt/ros/jazzy/setup.bash; /usr/bin/python3 evaluation/motm_right_wiring_check.py OUT.json A
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=87 .venv-dl0/bin/python evaluation/motm_right_wiring_check.py OUT.json B
"""
import inspect
import json
import os
import re
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
    src = open(os.path.join(HERE, 'isaac_bottle_w1dynsr44.py')).read()
    mw = open(os.path.join(HERE, 'bottle_map_write.py')).read()
    need = {
        'controller_side': "RC.import_with_real_ros('bottle_w1_side_ctrl'",
        'mirror_arg': "ap.add_argument('--mirror-y'",
        'region_offset_mirrored': "RO = mir([float(v) for v in a.region_offset.split(',')] + [0.0])",
        'place_mirrored_to_ctrl': 'place=REG.place_to_region(PLACE_W)',
        'region_prior_injected': 'ctrl.region_prior = dict(',
        'apply_region_prior': 'ctrl.apply_region_prior()',
        'side_publisher': "node.create_publisher(Float64MultiArray, '/bottle/grasp_side', 10)",
        'side_publish_each_tick': 'publish_side(t)',
        'lane_side': 'y_tube=C2.B.Y_TUBE, y_reach=y_reach, side=side_)',
        'lane_side_unknown_refuse': "why='grasp_side_unknown'",
        'nominal_only_with_mirror': "if a.mirror_y and a.dyn_case != 'none':",
        'map_mirror_flag': "ap.add_argument('--mirror-y'" if "ap.add_argument('--mirror-y'" in mw else None,
    }
    out['A']['static'] = {k: (v is not None and (v in src if k != 'map_mirror_flag' else True)) for k, v in need.items()}
    import bottle_ros_chain as RC
    OC, SC = RC.import_with_real_ros('bottle_w1_obs_ctrl', 'bottle_w1_side_ctrl')
    C = SC.W1PlaceController
    own = {m: [k.__module__ for k in C.__mro__ if m in k.__dict__][0] for m in ('_grasp_T', '_tcp_c', 'candidate_intake', 'guide_intake', '_standoff', '_advance_inner',
                                                                                'apply_region_prior', 'report')}
    out['A']['side_methods'] = own
    hooks = {}
    for K_ in (OC.W1PlaceController, C):
        names = set()
        for k in K_.__mro__:
            try:
                s_ = inspect.getsource(k)
            except (TypeError, OSError):
                continue
            names |= set(re.findall(r'self\.(_tcp_c|_grasp_T|_t1|_seg_valid_t|_seg_expire_info)\(', s_))
        hooks[K_.__module__] = {n: hasattr(K_, n) for n in sorted(names)}
    out['A']['hooks_resolve'] = hooks
    ok = all(out['A']['static'].values()) and all(v == 'bottle_w1_side_ctrl' for k, v in own.items() if k != 'report' or True) and all(all(h.values()) for h in hooks.values())
    out['A']['verdict'] = ok
    print('A', json.dumps(out['A'], ensure_ascii=False), flush=True)


def part_B():
    import rclpy
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension
    import bottle_guide_link as GL
    import bottle_segtarget_link as SL
    import bottle_seg3d_node as SN
    log = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), 'motm_right_wiring_node.jsonl')
    if os.path.exists(log):
        os.remove(log)
    proc = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'bottle_seg3d_candidates_dual_side_node.py'), '--log', log, '--ros-args', '-p', 'use_sim_time:=false'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = rclpy.create_node('motm_right_wiring_check')
    pub = dict(v1=node.create_publisher(Float64MultiArray, '/v1/bottle_seg3d', 50), w1=node.create_publisher(Float64MultiArray, '/w1/bottle_seg3d', 50))
    pub_side = node.create_publisher(Float64MultiArray, '/bottle/grasp_side', 10)
    g_in, c_in = [], []
    ctrl = types.SimpleNamespace(guide_intake=lambda t, g: g_in.append(dict(t=t, **g)), candidate_intake=lambda t, c: c_in.append(dict(t=t, **c)))
    RO = (0.6, -0.8)                                                    # 鏡射後區域偏移
    glink = GL.GuideLink(node, ctrl, Float64MultiArray, region_offset=RO)
    slink = SL.SegTargetLink(node, ctrl, Float64MultiArray)
    glink.set_T0(0.0)
    slink.set_T0(0.0)
    XW, YW = 0.6 - 0.0232, -(0.8 + 0.4081)                              # 鏡射場景右側瓶（世界）

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def msg(sup):
        d = {f: 0.0 for f in SN.FIELDS}
        d.update(stamp=now() - 0.05, valid=1.0, xy_valid=1.0, support_observed=1.0 if sup else 0.0, top_observed=0.0, height_valid=0.0, cx_known_r=XW, cy_known_r=YW, z_support=0.58,
                 z_top=float('nan'))
        m = Float64MultiArray()
        m.layout.dim = [MultiArrayDimension(label=SN.LAYOUT_LABEL, size=len(SN.FIELDS), stride=len(SN.FIELDS))]
        m.data = [float(d[f]) for f in SN.FIELDS]
        return m

    def side_msg(code, ver):
        m = Float64MultiArray()
        m.layout.dim = [MultiArrayDimension(label='bottle_grasp_side schema=1 side=0 none,1 left,2 right fields=t_ros,side,ver', size=3, stride=3)]
        m.data = [now(), float(code), float(ver)]
        return m
    t0 = time.time()
    while time.time() - t0 < 15.0 and node.count_subscribers('/w1/bottle_seg3d') + node.count_subscribers('/v1/bottle_seg3d') + node.count_subscribers('/bottle/grasp_side') < 3:
        rclpy.spin_once(node, timeout_sec=0.1)
    out['B']['subs'] = dict(v1=node.count_subscribers('/v1/bottle_seg3d'), w1=node.count_subscribers('/w1/bottle_seg3d'), side=node.count_subscribers('/bottle/grasp_side'))
    phases = []

    def run(sec, side_code, ver, tag):
        n0g, n0c = len(g_in), len(c_in)
        t1 = time.time()
        while time.time() - t1 < sec:
            if side_code is not None:
                pub_side.publish(side_msg(side_code, ver))
            pub['v1'].publish(msg(False))
            pub['w1'].publish(msg(True))
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.01)
            glink.tick(now())
            slink.tick(now())
            time.sleep(0.05)
        gg, cc = g_in[n0g:], c_in[n0c:]
        phases.append(dict(tag=tag, n_guide=len(gg), chk_side=sorted({g.get('chk_side') for g in gg}, key=str), chk_side_seq0=sorted({g.get('chk_side_seq0') for g in gg}, key=str),
                           n_cand=len(cc), seqs=sorted({int(c['snap_seq']) for c in cc}), adoptable=sum(1 for c in cc if c['ready'] == 1.0 and c['snap_ik'] == 2.0),
                           xy_last=None if not cc else [round(cc[-1]['snap_cx'], 4), round(cc[-1]['snap_cy'], 4)]))
    run(3.0, None, 0, '未選側')
    run(4.0, 2, 1, '選右側')
    run(2.0, 1, 2, '改左側')
    proc.send_signal(signal.SIGINT)
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        rc = 'killed'
    txt = proc.stdout.read().decode(errors='ignore')
    node.destroy_node()
    rclpy.shutdown()
    p0, p1, p2 = phases
    out['B'].update(phases=phases, node_rc=rc, node_traceback='Traceback' in txt, guide_rej=glink.rej, cand_rej=slink.rej, node_tail=txt[-400:])
    out['B']['verdict'] = dict(no_snapshot_before_side=(p0['seqs'] == [] and 0 in p0['chk_side']), right_checked_and_adoptable=(2 in p1['chk_side'] and p1['adoptable'] > 0),
                               xy_region_right=(p1['xy_last'] is not None and p1['xy_last'][1] < -0.3), seq0_advances_on_switch=(max(p2['chk_side_seq0']) > max(p1['chk_side_seq0'])),
                               guide_no_reject=not glink.rej, node_clean_exit=(rc == 0 and 'Traceback' not in txt))
    print('B', json.dumps(dict(verdict=out['B']['verdict'], subs=out['B']['subs'], phases=phases, guide_rej=glink.rej, cand_rej=slink.rej), ensure_ascii=False, default=str), flush=True)


part_A() if PART == 'A' else part_B()
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str)
