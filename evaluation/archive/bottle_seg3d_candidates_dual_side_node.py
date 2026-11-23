#!/usr/bin/env python3
"""雙相機候選節點（右側整合版；Codex 20261111_163931：候選可達檢查等控制端回傳選側，不以預設左側結果替右側背書）。版本化子類，基線＝bottle_seg3d_candidates_dual_node.DualCandNode。
訂 /bottle/grasp_side（控制端抓取側：0 未定、1 左、2 右；由執行端 bridge 依控制器 grasp_side 發布）：
  未選側 ⇒ known_type 視為未就緒（why＝side_unknown），不建快照、不檢查；
  已選側 ⇒ 候選帶 side 進不可變快照，可達檢查（bottle_seg3d_candidates.reach_check）用同一側的假設底盤位置與抓取姿態；
  側別改變 ⇒ 作廢目前快照（含檢查中者，完成通知壓下），起算序號 side_seq0＝下一個快照序號。
導引訊息（/bottle/target_guide）多兩欄：chk_side（節點檢查用的側別碼）、chk_side_seq0（該側起算的快照序號）；控制端只收同側且序號 ≥ side_seq0 的快照。
    source /opt/ros/jazzy/setup.bash; .venv-dl0/bin/python evaluation/bottle_seg3d_candidates_dual_side_node.py --log RUN_DIR/candidates.jsonl --ros-args -p use_sim_time:=true
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_seg3d_candidates_dual_node as DN          # noqa: E402

import rclpy                                            # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402

SIDE = {0: None, 1: 'left', 2: 'right'}
CODE = {None: 0.0, 'left': 1.0, 'right': 2.0}
GFIELDS2 = DN.GFIELDS + ('chk_side', 'chk_side_seq0')
GLABEL2 = DN.GLABEL.split('fields=')[0] + 'chk_side=0 none,1 left,2 right fields=' + ','.join(GFIELDS2)
SLABEL = 'bottle_grasp_side schema=1 side=0 none,1 left,2 right fields=t_ros,side,ver'


class DualSideNode(DN.DualCandNode):
    def __init__(self, q_seed, log_path):
        super().__init__(q_seed, log_path)
        self.side, self.side_seq0, self.n_side_msg = None, 0, 0
        self.create_subscription(Float64MultiArray, '/bottle/grasp_side', self._in_side, 10)
        ev1 = self.tr.evaluate                                           # 已含雙相機閘門

        def side_gated(t_now):
            ev = ev1(t_now)
            k = ev['known_type']
            if self.side is None:
                if k['status'] == 'ready':
                    ev['known_type'] = dict(k, status='not_ready', why='side_unknown')
            else:
                ev['known_type'] = dict(k, side=self.side)
            return ev
        self.tr.evaluate = side_gated

    def _in_side(self, m):
        lab = m.layout.dim[0].label if m.layout.dim else ''
        if 'bottle_grasp_side' not in lab or 'fields=' not in lab:
            return
        fs = lab.split('fields=')[1].split(',')
        if len(fs) != len(m.data):
            return
        r = dict(zip(fs, m.data))
        c = r.get('side')
        if c is None or not np.isfinite(c) or int(round(c)) not in SIDE:
            return
        self.n_side_msg += 1
        new = SIDE[int(round(c))]
        if new == self.side:
            return
        with self.sc.lock:                                               # 換側：作廢目前快照（同閘門擋住的作廢）
            snap = self.sc.snap.pop('known_type', None)
            if snap is not None:
                self.void_seq.add(snap['seq'])
                if self.sc.queue.get('known_type') == snap['seq']:
                    self.sc.queue.pop('known_type', None)
            self.side_seq0 = self.sc.seq + 1
        self._write(dict(rec='grasp_side', side=new, prev=self.side, side_seq0=self.side_seq0, t_ros=r.get('t_ros'), void_seq=sorted(self.void_seq)))
        self.side = new

    def _done(self, s, r):
        if s['cand'].get('side') != self.side:
            self.n_suppressed += 1
            self._write(dict(rec='check_suppressed', seq=s['seq'], why='side_changed', snap_side=s['cand'].get('side'), side=self.side))
            return
        super()._done(s, r)

    def _guide_fields(self):
        return GFIELDS2, GLABEL2

    def _guide_extra(self):
        return dict(chk_side=CODE[self.side], chk_side_seq0=float(self.side_seq0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    ap.add_argument('--design', default=os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json'))
    a, ros_args = ap.parse_known_args()
    q_seed = np.array(json.load(open(a.design))['q_ready'])
    rclpy.init(args=ros_args)
    node = DualSideNode(q_seed, a.log)
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
