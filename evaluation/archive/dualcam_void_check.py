#!/usr/bin/env python3
"""雙相機作廢快照不可再被採用（Codex 20261111_134437 第 2 項）：經 SegTargetLink 核對最終可採用的序號。
情境 A（Codex 指定）：手腕 3 列 ⇒ 快照 #1 背景檢查中 → 底盤偏 3 cm ⇒ 不一致（作廢 #1）→ 舊檢查完成（應壓下）→ 底盤偵測中斷一列後偏 0 ⇒ 恢復一致 → 手腕新列。
情境 B（順序對調）：不一致 → 恢復一致 → 舊檢查才完成。
期望：#1 從未以「就緒＋檢查通過」轉送到控制端；恢復後建立新序號快照、其檢查通過並經 SegTargetLink 轉送。
背景檢查以包裝延遲製造競態視窗；節點在獨立網域建立、不 spin，直接呼叫回呼並攔截發布，候選列與完成通知照收到順序餵 SegTargetLink。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=86 .venv-dl0/bin/python evaluation/dualcam_void_check.py OUT.json
"""
import json
import sys
import time
import types

sys.path.insert(0, 'evaluation')
sys.path.insert(0, 'src/ammr_wholebody_mpc')
import numpy as np                                                      # noqa: E402
import bottle_seg3d_candidates as CT                                    # noqa: E402

DELAY = dict(v=0.35)
_orig = CT.reach_check


def slow(c, q, iters=150):
    time.sleep(DELAY['v'])
    return _orig(c, q, iters)


CT.reach_check = slow
import rclpy                                                            # noqa: E402
import bottle_seg3d_candidates_dual_node as DN                         # noqa: E402
import bottle_seg3d_node as SN                                          # noqa: E402
import bottle_segtarget_link as SL                                      # noqa: E402
from std_msgs.msg import Float64MultiArray, MultiArrayDimension        # noqa: E402

rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=false'])
q_seed = np.array(json.load(open('evaluation/results/vision/seg3d_collect_design.json'))['q_ready'])


def run(order):
    node = DN.DualCandNode(q_seed, None)
    fwd = []
    link = SL.SegTargetLink(types.SimpleNamespace(create_subscription=lambda *a, **k: None), None, None, sink=lambda t, c: fwd.append(dict(t=t, **c)))
    link.set_T0(0.0)
    rows, chks = [], []

    def pub_row(m):
        rows.append(dict(zip(DN.CN.FIELDS, m.data)))
        link.q.append((m.layout.dim[0].label, list(m.data)))

    def pub_chk(m):
        chks.append(dict(zip(DN.CN.CHK_FIELDS, m.data)))
        link.qc.append((m.layout.dim[0].label, list(m.data)))
    node.pub.publish, node.pub_chk.publish, node.pub_guide.publish = pub_row, pub_chk, (lambda m: None)

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def msg(cx, cy, sup=True, valid=True):
        d = {f: 0.0 for f in SN.FIELDS}
        d.update(stamp=now() - 0.05, valid=1.0 if valid else 0.0, xy_valid=1.0 if valid else 0.0, support_observed=1.0 if (sup and valid) else 0.0, top_observed=0.0,
                 height_valid=0.0, cx_known_r=cx, cy_known_r=cy, z_support=0.58, z_top=float('nan'))
        m = Float64MultiArray()
        m.layout.dim = [MultiArrayDimension(label=SN.LAYOUT_LABEL, size=len(SN.FIELDS), stride=len(SN.FIELDS))]
        m.data = [float(d[f]) for f in SN.FIELDS]
        return m
    steps = []

    def mark(what):
        link.tick(now())
        steps.append(dict(what=what, gate=node.gate, snap=(node.sc.snap.get('known_type') or {}).get('seq'), n_chk=len(chks), n_supp=node.n_suppressed,
                          last_row_seq=rows[-1]['k_snap_seq'] if rows else None, last_row_ready=rows[-1]['k_ready'] if rows else None))
    xb, yb = 0.577, 1.208

    def w1(n):
        for _ in range(n):
            node._in(msg(xb, yb))
            time.sleep(0.02)

    def v1(n, off, valid=True):
        for _ in range(n):
            node._in_v1(msg(xb + off, yb, sup=False, valid=valid))
            time.sleep(0.02)

    def wait_suppressed(k, t_max=2.0):
        t0 = time.time()
        while node.n_suppressed < k and time.time() - t0 < t_max:
            time.sleep(0.01)
    w1(3)
    mark('w1×3：快照建立、檢查中')
    v1(3, 0.03)
    mark('v1 偏 3 cm ×3：不一致')
    if order == 'A':
        wait_suppressed(1)
        mark('舊檢查完成（期望壓下）')
        v1(1, 0.0, valid=False)
        v1(3, 0.0)
        mark('v1 中斷一列後偏 0 ×3：恢復一致')
        w1(2)
        mark('w1 新列')
    else:
        v1(1, 0.0, valid=False)
        v1(3, 0.0)
        mark('v1 中斷一列後偏 0 ×3：恢復一致')
        w1(2)
        mark('w1 新列（舊檢查仍在跑）')
        wait_suppressed(1)
        mark('舊檢查完成（期望壓下）')
    t0 = time.time()
    while time.time() - t0 < 2.0 and not any(c['seq'] >= 2 for c in chks):
        time.sleep(0.02)
    w1(1)
    mark('新快照檢查完成後手腕再一列')
    void = sorted(node.void_seq)
    adopt = sorted({int(c['snap_seq']) for c in fwd if c['ready'] == 1.0 and c['snap_ik'] == 2.0})
    out = dict(order=order, delay_s=DELAY['v'], void_seq=void, check_published_seqs=[int(c['seq']) for c in chks], n_suppressed=node.n_suppressed,
               forwarded=[dict(seq=int(c['snap_seq']), ready=c['ready'], ik=c['snap_ik'], local=c['snap_local']) for c in fwd],
               adoptable_seqs=adopt, link=dict(n_fwd=link.n_fwd, n_merged=link.n_merged, rejected=link.rej), steps=steps,
               verdict=dict(void_never_adoptable=not (set(void) & set(adopt)), void_check_never_published=not (set(void) & {int(c['seq']) for c in chks}),
                            new_snapshot_adoptable=any(s not in void for s in adopt)))
    node.close()
    node.destroy_node()
    return out


res = {}
for order in ('A', 'B'):
    r = run(order)
    res[order] = r
    print(order, r['verdict'], 'void', r['void_seq'], 'checks', r['check_published_seqs'], 'adoptable', r['adoptable_seqs'], flush=True)
    for s in r['steps']:
        print('   ', s, flush=True)
rclpy.shutdown()
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
