#!/usr/bin/env python3
"""雙相機任務端候選節點（Howard 11/11「要善用兩顆鏡頭」；Codex 20261111_050821 (c)；版本化子類，基線＝bottle_seg3d_candidates_node.CandNode）。
訂 /v1/bottle_seg3d（底盤）與 /w1/bottle_seg3d（手腕）；兩路都餵 bottle_dualcam_select.DualSelect（各來源追蹤、交叉核對、物件層級不一致、擷取時刻年齡）。
  抓取候選：沿用基線的 known_type 追蹤與不可變快照（ik／局部點檢查），只多一道閘——DualSelect 判 inconsistent／reject／ambiguous／adopted_object_lost
    ⇒ 該拍 known_type 視為 not_ready（why＝inconsistent），不產生新快照；發布格式不變（控制端 segtarget 規則照舊：限速更新、凍結、期限）。
    閘門轉為擋住時：目前快照（含背景檢查中者）作廢——從快照檢查器移除（快照與待算槽），其完成通知不發布，之後的候選列也不再帶該序號與其結果
      （恢復一致時另建新序號快照重新檢查；Codex 134437 第 2 項）；若由底盤列觸發，立即以最後手腕訊息重評估並發布一列（未就緒），
    使最新候選列不再是 ready（Codex 054341 競態：檢查中 → 底盤發現不一致 → 手腕暫無新影格 → 舊檢查完成仍被承接）。
  導引候選：另發 /bottle/target_guide（Float64MultiArray；layout 標籤帶欄位名）：t_eval、guide_cam（0 無 1 v1 2 w1）、guide_state、cx、cy、t_first、t_last（擷取戳）、
    age_s、n、grasp_cam、grasp_state、reject、schema。供控制端觀察觸發、車道／側別（不是抓取目標）。
每筆輸入列與 DualSelect 結果寫入 --log（JSON lines；含 camera_id）。所有時間是 ROS 模擬時鐘。
    source /opt/ros/jazzy/setup.bash; .venv-dl0/bin/python evaluation/bottle_seg3d_candidates_dual_node.py --log RUN_DIR/candidates.jsonl --ros-args -p use_sim_time:=true
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_dualcam_select as DS                      # noqa: E402
import bottle_seg3d_candidates_node as CN               # noqa: E402

import rclpy                                            # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from std_msgs.msg import Float64MultiArray, MultiArrayDimension   # noqa: E402

CAM = {None: 0.0, 'v1': 1.0, 'w1': 2.0}
GSTATE = {'none': 0.0, 'selected': 1.0, 'kept': 2.0, 'selected_unchecked': 3.0, 'kept_unchecked': 4.0, 'inconsistent': 5.0, 'reject': 6.0,
          'ambiguous_two_objects': 7.0, 'adopted_object_lost': 8.0}
GFIELDS = ('t_eval', 'guide_cam', 'guide_state', 'cx', 'cy', 't_first', 't_last', 'age_s', 'n', 'grasp_cam', 'grasp_state', 'reject', 'schema')
GLABEL = 'bottle_target_guide schema=1 frame=odom(world, z up, m) stamps=ros_sim_clock cam=0 none,1 v1,2 w1 fields=' + ','.join(GFIELDS)
BLOCK = ('inconsistent', 'reject', 'ambiguous_two_objects', 'adopted_object_lost')


class DualCandNode(CN.CandNode):
    def __init__(self, q_seed, log_path):
        super().__init__(q_seed, log_path)
        self.ds = DS.DualSelect()
        self.gate, self.void_seq, self.n_suppressed, self._last_w1 = None, set(), 0, None
        self.create_subscription(Float64MultiArray, '/v1/bottle_seg3d', self._in_v1, 50)
        self.pub_guide = self.create_publisher(Float64MultiArray, '/bottle/target_guide', 20)
        ev0 = self.tr.evaluate

        def gated(t_now):
            ev = ev0(t_now)
            if self.gate in BLOCK and ev['known_type']['status'] == 'ready':
                ev['known_type'] = dict(ev['known_type'], status='not_ready', why='inconsistent', dual_gate=self.gate)
            return ev
        self.tr.evaluate = gated

    def _parse(self, m):
        if not m.layout.dim or 'fields=' not in m.layout.dim[0].label:
            return None
        fs = m.layout.dim[0].label.split('fields=')[1].split(',')
        if len(fs) != len(m.data):
            return None
        r = dict(zip(fs, m.data))
        r['recv_age_s'] = self.get_clock().now().nanoseconds * 1e-9 - r['stamp']
        return r

    def _done(self, s, r):
        """背景快照檢查完成：該快照已因雙相機不一致作廢、或此刻閘門擋住 ⇒ 不發布完成通知（Codex 054341：檢查中快照不可在衝突後被承接）。"""
        if s['seq'] in self.void_seq or self.gate in BLOCK:
            self.n_suppressed += 1
            self._write(dict(rec='check_suppressed', seq=s['seq'], gate=self.gate, void=s['seq'] in self.void_seq))
            return
        super()._done(s, r)

    def _dual(self, cam, r):
        now = self.get_clock().now().nanoseconds * 1e-9
        acc = self.ds.add(cam, r)
        res = self.ds.evaluate(now)
        was = self.gate
        self.gate = res['grasp_state']
        if self.gate in BLOCK and was not in BLOCK:
            with self.sc.lock:                                                # 作廢目前快照（含背景檢查中者）：從檢查器移除快照與待算槽（Codex 134437 第 2 項）
                snap = self.sc.snap.pop('known_type', None)                   #   ⇒ 之後候選列不再帶舊序號／舊結果；恢復一致時 ready 候選建立新序號快照
                if snap is not None:
                    self.void_seq.add(snap['seq'])
                    if self.sc.queue.get('known_type') == snap['seq']:
                        self.sc.queue.pop('known_type', None)                 #   尚未開始的檢查不再執行；執行中的檢查完成時由 _done 依 void_seq 壓下
            self._write(dict(rec='gate_block', gate=self.gate, void_seq=sorted(self.void_seq)))
            if cam == 'v1' and self._last_w1 is not None:
                CN.CandNode._in(self, self._last_w1)                          # 立即以最後手腕訊息重評估並發布（閘門 ⇒ known_type 未就緒），最新候選列不再是 ready
        u, g = res['guide'], res['grasp']
        nan = float('nan')
        row = dict(t_eval=now, guide_cam=CAM[None if u is None else u['camera_id']], guide_state=GSTATE.get(res['guide_state'], -1.0),
                   cx=nan if u is None else u['center'][0], cy=nan if u is None else u['center'][1], t_first=nan if u is None else u['t_first'],
                   t_last=nan if u is None else u['t_last'], age_s=nan if u is None else u['age_s'], n=0.0 if u is None else float(u['n']),
                   grasp_cam=CAM[None if g is None else g['camera_id']], grasp_state=GSTATE.get(res['grasp_state'], -1.0), reject=1.0 if res.get('reject') else 0.0, schema=1.0)
        out = Float64MultiArray()
        out.layout.dim = [MultiArrayDimension(label=GLABEL, size=len(GFIELDS), stride=len(GFIELDS))]
        out.data = [float(row[f]) for f in GFIELDS]
        self.pub_guide.publish(out)
        self._write(dict(kind='dual', t_eval=now, cam=cam, accepted=acc, stamp=r.get('stamp'), recv_age_s=r.get('recv_age_s'), guide_state=res['guide_state'], grasp_state=res['grasp_state'],
                         guide=None if u is None else dict(cam=u['camera_id'], cx=u['center'][0], cy=u['center'][1], age_s=u['age_s']),
                         grasp=None if g is None else dict(cam=g['camera_id'], cx=g['center'][0], cy=g['center'][1], age_s=g['age_s']), handoffs=self.ds.handoffs[-1:] if self.ds.handoffs else []))

    def _in_v1(self, m):
        r = self._parse(m)
        if r is not None:
            self._dual('v1', r)

    def _in(self, m):
        r = self._parse(m)
        if r is not None:
            self._last_w1 = m
            self._dual('w1', dict(r))
        super()._in(m)                                                    # 基線 known_type 追蹤與快照（經 gated evaluate）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    ap.add_argument('--design', default=os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json'))
    a, ros_args = ap.parse_known_args()
    q_seed = np.array(json.load(open(a.design))['q_ready'])
    rclpy.init(args=ros_args)
    node = DualCandNode(q_seed, a.log)
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
