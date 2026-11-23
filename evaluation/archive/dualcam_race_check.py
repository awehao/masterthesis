#!/usr/bin/env python3
"""雙相機閘門競態反例（Codex 20261111_054341 第 1 項）：手腕快照背景檢查中 → 底盤相機發現不一致（偏 3 cm）→ 手腕暫無新影格 → 舊檢查完成。
期望：該快照作廢、完成通知不發布、最新候選列為未就緒。對照：底盤一致（偏 0）⇒ 完成通知照常發布。
背景檢查以包裝延遲 0.6 s 製造競態視窗；節點在獨立網域建立、不 spin，直接呼叫回呼並攔截發布。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=86 .venv-dl0/bin/python evaluation/dualcam_race_check.py
"""
import sys, time, types, json, os
sys.path.insert(0, 'evaluation'); sys.path.insert(0, 'src/ammr_wholebody_mpc')
import numpy as np
import bottle_seg3d_candidates as CT
_orig = CT.reach_check
def slow(c, q, iters=150):
    time.sleep(0.6); return _orig(c, q, iters)
CT.reach_check = slow                                                   # 放慢背景檢查以製造競態視窗
import rclpy
import bottle_seg3d_candidates_dual_node as DN
import bottle_seg3d_node as SN
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=false'])
q_seed = np.array(json.load(open('evaluation/results/vision/seg3d_collect_design.json'))['q_ready'])

def run(offset):
    node = DN.DualCandNode(q_seed, None)
    pubs, chks = [], []
    node.pub.publish = lambda m: pubs.append(dict(zip(DN.CN.FIELDS, m.data)))
    node.pub_chk.publish = lambda m: chks.append(list(m.data))
    node.pub_guide.publish = lambda m: None
    def msg(t, cx, cy, sup=True):
        d = {f: 0.0 for f in SN.FIELDS}
        d.update(stamp=t, valid=1.0, xy_valid=1.0, support_observed=1.0 if sup else 0.0, top_observed=0.0, height_valid=0.0, cx_known_r=cx, cy_known_r=cy, z_support=0.58, z_top=float('nan'))
        m = Float64MultiArray(); m.layout.dim = [MultiArrayDimension(label=SN.LAYOUT_LABEL, size=len(SN.FIELDS), stride=len(SN.FIELDS))]
        m.data = [float(d[f]) for f in SN.FIELDS]; return m
    now = lambda: node.get_clock().now().nanoseconds * 1e-9
    xb, yb = 0.577, 1.208
    for k in range(3):                                                 # 手腕 3 列 ⇒ known_type ready ⇒ 快照、背景檢查開始
        node._in(msg(now() - 0.05, xb, yb)); time.sleep(0.02)
    snap_seq = pubs[-1]['k_snap_seq']
    for k in range(3):                                                 # 檢查中：底盤 3 列（偏 offset）
        node._in_v1(msg(now() - 0.05, xb + offset, yb, sup=False)); time.sleep(0.02)
    time.sleep(1.5)                                                    # 等背景檢查完成
    last = pubs[-1]
    out = dict(offset_m=offset, snap_seq=snap_seq, gate=node.gate, n_check_published=len(chks), n_suppressed=node.n_suppressed, void=sorted(node.void_seq),
               last_row=dict(k_ready=last['k_ready'], k_why=last['k_why'], k_snap_seq=last['k_snap_seq'], k_snap_ik=last['k_snap_ik']))
    node.close(); node.destroy_node()
    return out

for off in (0.03, 0.0):
    print(run(off), flush=True)
rclpy.shutdown()
