#!/usr/bin/env python3
"""候選轉接（Codex 20261102_002139、003740）：/bottle/target_candidates＋/bottle/target_candidate_checks（候選節點 schema ≥ 3）→ bottle_w1_segtarget_ctrl.candidate_intake。
在 Isaac 同程序的 bridge 節點上訂閱；主迴圈每步 tick(t_task) 依收到順序處理。所有訊息內時間都是 ROS 模擬時鐘；轉送時換成任務時刻（ROS − T0）。
候選列先完整驗證，全部通過才更新接收進度（last_eval）並轉送（失效關閉）：
  標籤（schema ≥ 3、frame=odom、欄位名）；必要欄位齊全；t_eval／src_stamp 有限、非未來（≤ 現在）、t_eval ≥ src_stamp、t_eval 嚴格遞增；
  狀態碼（k_ready ∈ {0,1}、k_snap_ik／k_snap_local ∈ {0,1,2,3}）須是精確整數，不用 int() 截斷；
  尚無快照（k_snap_seq＝0）⇒ 允許欄位未設定，但不轉送、不拋例外；有快照時目標與來源戳須有限、來源戳不晚於 t_eval，快照年齡由來源戳重算（現在 − snap_t_last）≤ SNAP_MAX_AGE（3 s）。
  觀測年齡在轉送當下重算＝現在 − 最新來源列擷取戳（t_eval − k_age_s），含轉送延遲。
完成通知（checks）同樣先驗證（kind＝1 known_type、序號、目標、狀態碼、t_publish 有限且非未來）；以「序號＋精確目標」與最新候選列配對：
  最新候選列該序號仍是 pending ⇒ 在通知到達當下補送一筆（ik／local 取通知結果，年齡在此刻重算），不必等下一張影像；採用仍由控制器核對新鮮度與一致性，
  觀測已逾期就不會採用。目標不符的通知不配對。
"""
import math

import numpy as np

SNAP_MAX_AGE = 3.0
REQ = ('t_eval', 'src_stamp', 'k_ready', 'k_age_s', 'k_snap_seq', 'k_snap_t_last', 'k_snap_cx', 'k_snap_cy', 'k_snap_z_support', 'k_snap_ik', 'k_snap_local', 'k_snap_dev_mm',
       'k_snap_consistent')
CHK_REQ = ('t_publish', 'kind', 'seq', 'cx', 'cy', 'z_support', 'ik', 'local')
STATES = (0, 1, 2, 3)


def _code(v, allowed):
    """精確整數狀態碼；否則 None。"""
    if v is None or not np.isfinite(v) or abs(v - round(v)) > 1e-9 or int(round(v)) not in allowed:
        return None
    return int(round(v))


def _parse(label, data, req):
    if 'fields=' not in label:
        return None, 'no_label'
    try:
        schema = float(label.split('schema=')[1].split()[0])
    except (IndexError, ValueError):
        schema = 0.0
    if schema < 3 or 'frame=odom' not in label:
        return None, 'schema_or_frame'
    fs = label.split('fields=')[1].split(',')
    if len(fs) != len(data):
        return None, 'field_count'
    d = dict(zip(fs, data))
    if any(k not in d for k in req):
        return None, 'missing_field'
    return d, None


class SegTargetLink:
    def __init__(self, node, ctrl, msg_type, topic='/bottle/target_candidates', check_topic='/bottle/target_candidate_checks', sink=None):
        self.ctrl, self.sink = ctrl, (ctrl.candidate_intake if sink is None else sink)
        self.q, self.qc, self.T0, self.last_eval = [], [], None, None
        self.latest, self.checks, self.merged = None, {}, set()
        self.n_fwd, self.n_merged, self.rej, self.log = 0, 0, {}, []
        node.create_subscription(msg_type, topic, lambda m: self.q.append((m.layout.dim[0].label if m.layout.dim else '', list(m.data))), 50)
        node.create_subscription(msg_type, check_topic, lambda m: self.qc.append((m.layout.dim[0].label if m.layout.dim else '', list(m.data))), 20)

    def set_T0(self, T0):
        self.T0 = float(T0)

    def _rej(self, why):
        self.rej[why] = self.rej.get(why, 0) + 1

    def _row(self, d, now_ros):
        """驗證候選列；回傳 (record 或 None, 拒收原因)。"""
        te, ss = d['t_eval'], d['src_stamp']
        if not (np.isfinite(te) and np.isfinite(ss)):
            return None, 'time_nonfinite'
        if te > now_ros + 1e-6 or ss > te + 1e-9:
            return None, 'time_future'
        if self.last_eval is not None and te <= self.last_eval:
            return None, 't_eval_not_increasing'
        rd, ik, lc = _code(d['k_ready'], (0, 1)), _code(d['k_snap_ik'], STATES), _code(d['k_snap_local'], STATES)
        if rd is None or ik is None or lc is None:
            return None, 'bad_state_code'
        seq = d['k_snap_seq']
        if not np.isfinite(seq) or abs(seq - round(seq)) > 1e-9 or seq < 0:
            return None, 'bad_seq'
        rec = dict(t_eval=te, seq=int(round(seq)), ready=rd, ik=ik, local=lc)
        if rec['seq'] == 0:
            return rec, None                                          # 尚無快照：記錄進度，不轉送
        vals = [d['k_snap_t_last'], d['k_snap_cx'], d['k_snap_cy'], d['k_snap_z_support']]
        if not all(np.isfinite(v) for v in vals):
            return None, 'snapshot_nonfinite'
        if rd == 1 and not np.isfinite(d['k_age_s']):               # 目前觀測 ready 時年齡必須有限；未 ready 時年齡可為 NaN（照轉送，控制器以 not_ready 拒收）
            return None, 'age_nonfinite'
        if d['k_snap_t_last'] > te + 1e-9 or (np.isfinite(d['k_age_s']) and d['k_age_s'] < 0):
            return None, 'snapshot_time_order'
        if now_ros - d['k_snap_t_last'] > SNAP_MAX_AGE:
            return None, 'snapshot_too_old'
        rec.update(src_latest=te - d['k_age_s'] if np.isfinite(d['k_age_s']) else float('nan'), snap_t_last=d['k_snap_t_last'], cx=d['k_snap_cx'], cy=d['k_snap_cy'], z=d['k_snap_z_support'],
                   dev_mm=d['k_snap_dev_mm'], consistent=d['k_snap_consistent'])
        return rec, None

    def _forward(self, t_task, now_ros, rec, ik, lc, how):
        c = dict(ready=float(rec['ready']), age_s=now_ros - rec['src_latest'], snap_seq=float(rec['seq']), snap_t_last=rec['snap_t_last'] - self.T0, snap_cx=rec['cx'],
                 snap_cy=rec['cy'], snap_z_support=rec['z'], snap_ik=float(ik), snap_local=float(lc), snap_dev_mm=rec['dev_mm'], snap_consistent=rec['consistent'])
        self.log.append((float(t_task), how, rec['seq'], float(c['age_s']), rec['ready'], ik, lc))
        self.sink(float(t_task), c)

    def tick(self, t_task):
        q, self.q = self.q, []
        qc, self.qc = self.qc, []
        if self.T0 is None:
            for _ in q + qc:
                self._rej('no_T0')
            return
        now_ros = self.T0 + float(t_task)
        for label, data in q:
            d, why = _parse(label, data, REQ)
            if d is None:
                self._rej(why)
                continue
            rec, why = self._row(d, now_ros)
            if rec is None:
                self._rej(why)
                continue
            self.last_eval = rec['t_eval']                            # 全部驗證通過才更新接收進度
            if rec['seq'] == 0:
                self._rej('no_snapshot_yet')
                continue
            self.latest = rec
            ik, lc, how = rec['ik'], rec['local'], 'row'
            k = self.checks.get(rec['seq'])
            if ik == 1 and k is not None and self._same(k, rec):
                ik, lc, how = k['ik'], k['local'], 'row+check'
            self._forward(t_task, now_ros, rec, ik, lc, how)
            self.n_fwd += 1
        for label, data in qc:
            d, why = _parse(label, data, CHK_REQ)
            if d is None:
                self._rej('chk_' + why)
                continue
            kind, seq, ik, lc = _code(d['kind'], (0, 1)), _code(d['seq'], range(1, 10 ** 9)), _code(d['ik'], STATES), _code(d['local'], STATES)
            if None in (kind, seq, ik, lc) or not all(np.isfinite([d['t_publish'], d['cx'], d['cy'], d['z_support']])) or d['t_publish'] > now_ros + 1e-6:
                self._rej('chk_invalid')
                continue
            if kind != 1:
                continue                                              # 只用 known_type
            self.checks[seq] = dict(ik=ik, local=lc, cx=d['cx'], cy=d['cy'], z=d['z_support'])
            L = self.latest
            if L is not None and L['seq'] == seq and L['ik'] == 1 and seq not in self.merged:
                if self._same(self.checks[seq], L):
                    self.merged.add(seq)
                    self._forward(t_task, now_ros, L, ik, lc, 'check')      # 完成後沒有下一張影像也能採用；年齡此刻重算
                    self.n_merged += 1
                else:
                    self._rej('chk_target_mismatch')

    @staticmethod
    def _same(k, rec):
        return abs(k['cx'] - rec['cx']) < 1e-9 and abs(k['cy'] - rec['cy']) < 1e-9 and abs(k['z'] - rec['z']) < 1e-9

    def report(self):
        ages = np.array([x[3] for x in self.log if np.isfinite(x[3])], float)
        return dict(n_forwarded=self.n_fwd, n_merged_from_check=self.n_merged, rejected=self.rej, T0=self.T0, n_ready=sum(1 for x in self.log if x[4] == 1),
                    n_checks=len(self.checks), age_s=None if not len(ages) else dict(median=round(float(np.median(ages)), 3), max=round(float(ages.max()), 3)),
                    note='時間換成任務時刻（ROS − T0）；年齡在轉送當下重算（含轉送延遲）；完成通知以序號＋精確目標配對')
