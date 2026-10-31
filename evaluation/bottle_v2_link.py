#!/usr/bin/env python3
"""V2 轉接（Codex 20261101_071255）：V1Capture 收到的估計 → V2PlaceController.vision_intake。
只轉送「該影格擷取步位姿互核通過」的估計（渲染端回讀 vs 量測底盤×外參，門檻同起跑前第一張影格核對：< 1.0 mm、< 0.1°）；
互核失敗、找不到影格紀錄、訊息欄位不足一律不送控制，分項計數。時間換成任務時間：擷取＝該影格記錄的任務時間，接收＝轉送當下的任務時間（≥ 實際收到時刻，偏保守）。
估計欄位同 bottle_v1_est_node：[擷取戳, valid, cx, cy, z_support, z_top, ax, ay, az, n_pts, rms, proc_ms, why, n_drop, z_top_valid, dist_est]；dist 用 dist_est（估計值＋相機位姿）。
"""
import numpy as np

POSE_MM, POSE_DEG = 1.0, 0.1


class V2Link:
    def __init__(self, cap, ctrl, pose_mm=POSE_MM, pose_deg=POSE_DEG, sink=None):
        """sink：轉送目標（預設 ctrl.vision_intake＝底盤相機；手腕相機用 ctrl.wrist_intake）。"""
        self.cap, self.ctrl, self.pose_mm, self.pose_deg = cap, ctrl, float(pose_mm), float(pose_deg)
        self.sink = ctrl.vision_intake if sink is None else sink
        self.seen, self.n_fwd, self.rej, self.log = set(), 0, {}, []

    def _rej(self, st, why):
        self.rej[why] = self.rej.get(why, 0) + 1
        self.log.append((float(st), why))

    def tick(self, t_now):
        for st in sorted(k for k in self.cap.ests if k not in self.seen):
            self.seen.add(st)
            _, d = self.cap.ests[st]
            f = self.cap.frames.get(st)
            if f is None:
                self._rej(st, 'no_frame_record')
                continue
            if not (np.isfinite(f['dev_mm']) and np.isfinite(f['dev_deg']) and f['dev_mm'] < self.pose_mm and f['dev_deg'] < self.pose_deg):
                self._rej(st, 'pose_check')
                continue
            if len(d) < 16:
                self._rej(st, 'short_message')
                continue
            e = dict(valid=d[1] == 1.0, cx=float(d[2]), cy=float(d[3]), ax=[float(v) for v in d[6:9]], dist=float(d[15]))
            self.sink(float(f['task_t']), float(t_now), e)
            self.n_fwd += 1
            self.log.append((float(st), 'forwarded'))

    def report(self):
        return dict(n_forwarded=self.n_fwd, rejected=dict(self.rej), pose_gate=dict(mm=self.pose_mm, deg=self.pose_deg), n_seen=len(self.seen))
