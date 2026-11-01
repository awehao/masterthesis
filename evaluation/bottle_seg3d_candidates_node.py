#!/usr/bin/env python3
"""任務端三維目標候選節點（Codex 20261102_001039；獨立程序、只記錄、不採用、不影響控制）。
訂 /w1/bottle_seg3d（旁路節點 schema ≥ 3，欄位名取自 layout 標籤；座標系 odom＝世界）；每收到一列：收到時年齡＝節點時鐘（use_sim_time）− 擷取戳，
餵 bottle_seg3d_candidates.CandidateTracker，當下評估兩類候選，ik／局部點檢查交給 ReachWorker（單一背景執行緒、最新待算槽，不阻塞回呼）。
發 /bottle/target_candidates（Float64MultiArray；layout 標籤帶 schema、座標系與欄位名），並把每筆輸入列與輸出逐行寫入 --log（JSON lines，每 20 筆 flush，結束時 flush）。
欄位（兩類各一組，前綴 c_＝complete、k_＝known_type）：ready、why（0 ready 1 rows 2 stale 3 inconsistent 4 unobservable）、cx、cy、z_support、z_top、z_grasp、n、age_s、
  以及該類目前的不可變快照（schema 3，Codex 20261102_002139）：snap_seq、snap_t_first／t_last（來源擷取戳）、snap_n、snap 精確目標（cx、cy、z_support、z_grasp）、
  snap_ik／snap_local（0 n/a 1 pending 2 ok 3 fail；屬於該快照版本）、snap_local_min、snap_age_s（以最新來源列計）、snap_dev_mm 與 snap_consistent（目前觀測相對快照，容差 2 mm）；
  快照檢查完成時另發 /bottle/target_candidate_checks（序號、來源戳、精確檢查目標、結果），不等下一張影像。所有時間是 ROS 模擬時鐘，不是任務時刻。另有 t_eval、src_stamp、n_rows、eval_ms、n_check_run。
ik＝假設的名目抓取底盤位置下的抓持終點 IK；local＝瓶心 0.10 m 內的局部點檢查；都不含底盤可達、路徑碰撞、自碰與完整環境碰撞，「候選可達」不是抓取成功。
    source /opt/ros/jazzy/setup.bash; .venv-dl0/bin/python evaluation/bottle_seg3d_candidates_node.py --log RUN_DIR/candidates.jsonl --ros-args -p use_sim_time:=true
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg3d_candidates as CT                    # noqa: E402

import rclpy                                            # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node                             # noqa: E402
from std_msgs.msg import Float64MultiArray, MultiArrayDimension   # noqa: E402

WHY = {None: 0, 'rows': 1, 'stale': 2, 'inconsistent': 3, 'unobservable': 4}
STATE = {'n/a': 0, 'pending': 1, 'ok': 2, 'fail': 3}
PER = ('ready', 'why', 'cx', 'cy', 'z_support', 'z_top', 'z_grasp', 'n', 'age_s',
       'snap_seq', 'snap_t_first', 'snap_t_last', 'snap_n', 'snap_cx', 'snap_cy', 'snap_z_support', 'snap_z_grasp', 'snap_ik', 'snap_local', 'snap_local_min', 'snap_age_s', 'snap_dev_mm', 'snap_consistent')
FIELDS = ('t_eval', 'src_stamp', 'n_rows', 'eval_ms', 'n_check_run') + tuple(f'c_{x}' for x in PER) + tuple(f'k_{x}' for x in PER) + ('schema',)
SCHEMA = 3.0
LABEL = 'bottle_target_candidates schema=3 frame=odom(world, z up, m) stamps=ros_sim_clock fields=' + ','.join(FIELDS)
CHK_FIELDS = ('t_publish', 'kind', 'seq', 't_first', 't_last', 'n', 'cx', 'cy', 'z_support', 'z_grasp', 'ik', 'local', 'local_min', 'schema')
CHK_LABEL = 'bottle_target_candidate_checks schema=3 frame=odom(world, z up, m) kind=0 complete,1 known_type stamps=ros_sim_clock fields=' + ','.join(CHK_FIELDS)


class CandNode(Node):
    def __init__(self, q_seed, log_path):
        super().__init__('bottle_target_candidates')
        self.tr = CT.CandidateTracker()
        self.sc = CT.SnapshotChecker(q_seed, on_done=self._done)
        self.log = open(log_path, 'a') if log_path else None
        self.n_log, self.n_rows = 0, 0
        self.create_subscription(Float64MultiArray, '/w1/bottle_seg3d', self._in, 50)
        self.pub = self.create_publisher(Float64MultiArray, '/bottle/target_candidates', 50)
        self.pub_chk = self.create_publisher(Float64MultiArray, '/bottle/target_candidate_checks', 20)

    def _done(self, s, r):
        """背景工作者完成一個快照的檢查 ⇒ 立即發布該版本結果（不等下一張影像）。"""
        nan = float('nan')
        c = s['cand']
        row = dict(t_publish=self.get_clock().now().nanoseconds * 1e-9, kind=0.0 if c['kind'] == 'complete' else 1.0, seq=float(s['seq']), t_first=c['t_first'], t_last=c['t_last'],
                   n=float(c['n']), cx=c['center'][0], cy=c['center'][1], z_support=c['z_support'], z_grasp=c['z_grasp'], ik=float(STATE[r.get('ik_state', 'n/a')]),
                   local=float(STATE[r.get('local_clear_state', 'n/a')]), local_min=r.get('local_clear_min') if r.get('local_clear_min') is not None else nan, schema=SCHEMA)
        m = Float64MultiArray()
        m.layout.dim = [MultiArrayDimension(label=CHK_LABEL, size=len(CHK_FIELDS), stride=len(CHK_FIELDS))]
        m.data = [float(row[f]) for f in CHK_FIELDS]
        self.pub_chk.publish(m)
        self._write(dict(rec='check', check=row, scope=r.get('scope')))          # 不能用 kind 當紀錄鍵：row 本身有 kind 欄

    def _write(self, d):
        if self.log is None:
            return
        self.log.write(json.dumps(d, ensure_ascii=False, default=float) + '\n')
        self.n_log += 1
        if self.n_log % 20 == 0:
            self.log.flush()

    def _in(self, m):
        if not m.layout.dim or 'fields=' not in m.layout.dim[0].label:
            return
        fs = m.layout.dim[0].label.split('fields=')[1].split(',')
        if len(fs) != len(m.data):
            return
        r = dict(zip(fs, m.data))
        now = self.get_clock().now().nanoseconds * 1e-9
        r['recv_age_s'] = now - r['stamp']
        accepted = self.tr.add(r)
        self.n_rows += 1
        self._write(dict(kind='row', t_recv=now, accepted=accepted, row={k: r[k] for k in ('stamp', 'recv_age_s', 'valid', 'xy_valid', 'top_observed', 'support_observed', 'height_valid',
                                                                                           'cx_known_r', 'cy_known_r', 'z_support', 'z_top', 'age_ms', 'proc_ms') if k in r}))
        w0 = time.perf_counter()
        ev = self.tr.evaluate(now)
        snaps = {k: self.sc.update(c, now) for k, c in ev.items()}
        ms = (time.perf_counter() - w0) * 1e3
        nan = float('nan')
        row = dict(t_eval=now, src_stamp=r['stamp'], n_rows=self.n_rows, eval_ms=ms, n_check_run=self.sc.n_run, schema=SCHEMA)
        for pre, k in (('c_', 'complete'), ('k_', 'known_type')):
            c, q = ev[k], snaps[k]
            ctr = c.get('center') or [nan, nan]
            row.update({f'{pre}ready': float(c['status'] == 'ready'), f'{pre}why': float(WHY.get(c['why'], -1)), f'{pre}cx': ctr[0], f'{pre}cy': ctr[1],
                        f'{pre}z_support': c.get('z_support', nan), f'{pre}z_top': c.get('z_top', nan), f'{pre}z_grasp': c.get('z_grasp', nan), f'{pre}n': c.get('n', 0),
                        f'{pre}age_s': c.get('age_s', nan)})
            sc_ = (q.get('center') or [nan, nan])
            row.update({f'{pre}snap_seq': float(q.get('seq', 0)), f'{pre}snap_t_first': q.get('t_first', nan), f'{pre}snap_t_last': q.get('t_last', nan), f'{pre}snap_n': float(q.get('n', 0)),
                        f'{pre}snap_cx': sc_[0], f'{pre}snap_cy': sc_[1], f'{pre}snap_z_support': q.get('z_support', nan), f'{pre}snap_z_grasp': q.get('z_grasp', nan),
                        f'{pre}snap_ik': float(STATE[q.get('ik_state', 'n/a')]) if q.get('seq') else 0.0, f'{pre}snap_local': float(STATE[q.get('local_clear_state', 'n/a')]) if q.get('seq') else 0.0,
                        f'{pre}snap_local_min': q.get('local_clear_min') if q.get('local_clear_min') is not None else nan, f'{pre}snap_age_s': q.get('age_s', nan),
                        f'{pre}snap_dev_mm': q.get('deviation_mm') if q.get('deviation_mm') is not None else nan,
                        f'{pre}snap_consistent': (1.0 if q.get('consistent') else 0.0) if q.get('consistent') is not None else nan})
        out = Float64MultiArray()
        out.layout.dim = [MultiArrayDimension(label=LABEL, size=len(FIELDS), stride=len(FIELDS))]
        out.data = [float(row[f]) if row[f] is not None else nan for f in FIELDS]
        self.pub.publish(out)
        self._write(dict(kind='cand', **{f: row[f] for f in FIELDS}, priors={k: ev[k].get('priors') for k in ev}, measured={k: ev[k].get('measured') for k in ev}))

    def close(self):
        self.sc.stop()
        if self.log is not None:
            self.log.flush()
            self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    ap.add_argument('--design', default=os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json'))
    a, ros_args = ap.parse_known_args()
    q_seed = np.array(json.load(open(a.design))['q_ready'])
    rclpy.init(args=ros_args)
    node = CandNode(q_seed, a.log)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
