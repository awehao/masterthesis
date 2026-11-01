#!/usr/bin/env python3
"""候選轉接（Codex 20261102_002139）：/bottle/target_candidates（候選節點 schema ≥ 3）→ bottle_w1_segtarget_ctrl.candidate_intake。
在 Isaac 同程序的 bridge 節點上訂閱；主迴圈每步 tick(t_task) 依收到順序轉送 known_type 欄位。
時間：訊息內都是 ROS 模擬時鐘；轉送時換成任務時刻（ROS − T0）。觀測年齡在轉送當下重算＝(T0＋t_task) − 最新來源列擷取戳（t_eval − k_age_s），含轉送延遲，不沿用節點評估時的年齡。
拒收（分項計數、不送控制）：沒有 layout 標籤、schema < 3、標籤沒有 frame=odom、欄位數不符、t_eval 不遞增、T0 未設定。
"""
import numpy as np

KEYS = ('ready', 'snap_seq', 'snap_t_first', 'snap_t_last', 'snap_n', 'snap_cx', 'snap_cy', 'snap_z_support', 'snap_z_grasp', 'snap_ik', 'snap_local', 'snap_local_min',
        'snap_age_s', 'snap_dev_mm', 'snap_consistent')


class SegTargetLink:
    def __init__(self, node, ctrl, msg_type, topic='/bottle/target_candidates', sink=None):
        self.ctrl, self.sink = ctrl, (ctrl.candidate_intake if sink is None else sink)
        self.q, self.T0, self.last_eval = [], None, None
        self.n_fwd, self.rej, self.log = 0, {}, []
        node.create_subscription(msg_type, topic, self._cb, 50)

    def set_T0(self, T0):
        self.T0 = float(T0)

    def _cb(self, m):
        self.q.append((m.layout.dim[0].label if m.layout.dim else '', list(m.data)))

    def _rej(self, why):
        self.rej[why] = self.rej.get(why, 0) + 1

    def tick(self, t_task):
        q, self.q = self.q, []
        for label, data in q:
            if self.T0 is None:
                self._rej('no_T0')
                continue
            if 'fields=' not in label:
                self._rej('no_label')
                continue
            try:
                schema = float(label.split('schema=')[1].split()[0])
            except (IndexError, ValueError):
                schema = 0.0
            if schema < 3 or 'frame=odom' not in label:
                self._rej('schema_or_frame')
                continue
            fs = label.split('fields=')[1].split(',')
            if len(fs) != len(data):
                self._rej('field_count')
                continue
            d = dict(zip(fs, data))
            if self.last_eval is not None and d['t_eval'] <= self.last_eval:
                self._rej('t_eval_not_increasing')
                continue
            self.last_eval = d['t_eval']
            now_ros = self.T0 + float(t_task)
            src_latest = d['t_eval'] - d['k_age_s'] if np.isfinite(d.get('k_age_s', np.nan)) else np.nan
            c = {k: d[f'k_{k}'] for k in KEYS}
            c['age_s'] = now_ros - src_latest if np.isfinite(src_latest) else np.nan
            for k in ('snap_t_first', 'snap_t_last'):
                c[k] = c[k] - self.T0 if np.isfinite(c[k]) else np.nan
            self.log.append((float(t_task), float(d['t_eval'] - self.T0), int(c['snap_seq']) if np.isfinite(c['snap_seq']) else 0, float(c['age_s']), float(c['ready'])))
            self.sink(float(t_task), c)
            self.n_fwd += 1

    def report(self):
        ages = np.array([x[3] for x in self.log if np.isfinite(x[3])], float)
        return dict(n_forwarded=self.n_fwd, rejected=self.rej, T0=self.T0, n_ready=sum(1 for x in self.log if x[4] > 0.5),
                    age_s=None if not len(ages) else dict(median=round(float(np.median(ages)), 3), max=round(float(ages.max()), 3)),
                    note='時間換成任務時刻（ROS − T0）；年齡在轉送當下重算（含轉送延遲）')
