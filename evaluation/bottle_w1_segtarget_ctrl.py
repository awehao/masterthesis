#!/usr/bin/env python3
"""學習模型提供抓取目標（known_type 路線；Codex 20261102_002139；版本化子類，基線＝完整組 bottle_w1_compactcarry3_ctrl）。
目標來源只有任務端候選節點的 known_type 不可變快照（bottle_seg3d_candidates_node schema 3）：
  量測＝學習遮罩＋深度的瓶中心 xy、支撐高度 z_support；先驗（明示）＝瓶半徑（中心擬合）、瓶高 0.20 m、抓持帶在支撐上 0.10 m。
  抓取目標 T_grasp＝grasp_target(xy, z_support＋0.10)；預抓、插入、抬升由同一 T_grasp 推出（同原流程）；仍以 window_branch 重算確認可行（不可行 ⇒ 受控停止）。
不接回舊估計器、不用固定桌高：vision_intake（V2 底盤相機）與 wrist_intake（W1 手腕估計）只記錄、不影響目標；_Tg 的 z 用採用快照的 z_support，不用 top_z。
採用（candidate_intake）：known_type ready、快照 ik 與局部點檢查皆 ok、目前觀測新鮮（收到時來源年齡 ≤ max_age）、與快照一致（偏差 ≤ 2 mm）、快照序號遞增；
  第一個合格快照 ⇒ 凍結（vfix）；之後更新的合格快照 ⇒ 以 W1 同一限速（5 mm/s）朝新目標移動，相對凍結值總偏差 > 5 mm ⇒ segtarget_inconsistent 受控停止。
鎖定：INSERT 開始後 0.9 s（同 W1），之後不再更新；閉爪閘＝已鎖定且施加目標已到達。
期限：漸變開始（x ≥ x_blend）仍未採用 ⇒ segtarget_no_fix；凍結到閉爪 > 15 s ⇒ segtarget_expired。採用後訊息中斷不另停（凍結語義，靜態瓶任務的候選期限內），以期限兜底。
注意：手臂備妥姿態（q_arm_ready）仍沿用名目分支（任務名目先驗），不是抓取目標來源；5 mm 淨空拒絕線與禁止接觸仍用模擬真值當安全網與評估。
「候選可達」只代表假設底盤位置下的抓持終點 IK 與局部點檢查；實際接近／插入／抬升路徑、自碰與環境淨空由閉迴路監看與事後核對。
"""
import math

import numpy as np

import bottle_branch as BB
import bottle_ref as BRF
import bottle_s0_ctrl as C0
import bottle_w1_compactcarry3_ctrl as C3

SEGREG = dict(max_age=0.5, dev_tol_mm=2.0, max_corr=0.005, rate=0.005, lock_after_insert_s=0.9, expire_s=15.0)


class W1PlaceController(C3.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET'

    def __init__(self, *a, seg=None, **kw):
        super().__init__(*a, **kw)
        self.seg = dict(SEGREG, **(seg or {}))
        self.seg_log, self.seg_rej, self.seg_seq, self.seg_z, self.seg_adopt = [], {}, 0, None, []
        self.ign_v2, self.ign_w1 = 0, 0

    # 舊估計器：只記錄，不影響目標
    def vision_intake(self, t_cap, t_recv, e):
        self.ign_v2 += 1
        self.vlog.append((float(t_cap), float(t_recv), bool(e.get('valid')), e.get('cx'), e.get('cy'), e.get('dist')))

    def wrist_intake(self, t_cap, t_recv, e):
        self.ign_w1 += 1

    def _srej(self, why):
        self.seg_rej[why] = self.seg_rej.get(why, 0) + 1

    def candidate_intake(self, t_recv, c):
        """c：known_type 欄位（任務時刻）：ready、age_s（目前觀測）、snap_seq、snap_t_last、snap_cx、snap_cy、snap_z_support、snap_ik、snap_local、snap_dev_mm、snap_consistent。"""
        R = self.seg
        self.seg_log.append((float(t_recv), {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in c.items()}))
        if self.judge.stop is not None or self.w_locked:
            return
        vals = [c.get(k) for k in ('age_s', 'snap_seq', 'snap_cx', 'snap_cy', 'snap_z_support')]
        if any(v is None or not np.isfinite(v) for v in vals):
            return self._srej('nonfinite')
        if c.get('ready', 0) < 0.5:
            return self._srej('not_ready')
        if int(c['snap_ik']) != 2 or int(c['snap_local']) != 2:
            return self._srej('check_not_ok')
        if c['age_s'] > R['max_age'] or c['age_s'] < 0:
            return self._srej('stale')
        if not (c.get('snap_consistent', 0) > 0.5 and c.get('snap_dev_mm', 1e9) <= R['dev_tol_mm']):
            return self._srej('inconsistent_with_snapshot')
        seq = int(c['snap_seq'])
        if seq <= self.seg_seq:
            return self._srej('seq_not_new')
        xy, zs = np.array([c['snap_cx'], c['snap_cy']], float), float(c['snap_z_support'])
        if self.vfix is None:
            if not self._seg_freeze(xy, zs, seq):
                return
        else:
            corr = float(np.linalg.norm(xy - self.vfix['xy']))
            if corr > R['max_corr'] or abs(zs - self.seg_z) > R['max_corr']:
                self.judge._stop(self.t, 'segtarget_inconsistent', corr=corr, dz=zs - self.seg_z, seq=seq)
                return
            self.w_goal = xy
        self.seg_seq = seq
        self.seg_adopt.append((self.t, seq, xy.tolist(), zs))

    def _seg_freeze(self, xy, zs, seq):
        self.seg_z = zs
        T_g = self._Tg(xy)
        T_p = T_g.copy()
        T_p[:3, 3] -= self.task.pre_m * T_g[:3, 2]
        br = BB.window_branch(T_p, T_g, BRF.lift_pose(T_g, self.task.lift_m), self.win[0], self.win[1])
        if br is None or not np.isfinite(br.get('margin', np.nan)) or br['margin'] <= 0.0:
            self.judge._stop(self.t, 'segtarget_unreachable', margin=None if br is None else br.get('margin'), seq=seq)
            self.seg_z = None
            return False
        self.task.T_grasp = T_g
        self.vfix = dict(t=self.t, xy=np.array(xy, float), spread=0.0, n=1, margin=float(br['margin']), seq=seq, z_support=zs)
        self.w_goal = np.array(xy, float)                        # W1 的施加／限速機制沿用：初值即目標
        self._ev('segtarget_freeze', xy=[round(float(v), 5) for v in xy], z_support=round(zs, 5), seq=seq, margin=float(br['margin']))
        return True

    def _Tg(self, xy):
        if self.seg_z is None:
            return np.full((4, 4), np.nan)                       # 未採用前沒有目標；不回填桌高
        return C0.grasp_target(float(xy[0]), float(xy[1]), self.seg_z + 0.10)

    def _w1_checks(self):
        R, st = self.seg, self.st
        if self.judge.stop is not None:
            return
        if not self.w_locked and st.phase == 'INSERT' and self.t - st.t_phase >= R['lock_after_insert_s'] - 1e-9:
            self.w_locked = True
            self._ev('segtarget_lock', goal=None if self.w_goal is None else [round(float(v), 5) for v in self.w_goal], seq=self.seg_seq)
            if self.vfix is None:
                self.judge._stop(self.t, 'segtarget_no_fix', rej=dict(self.seg_rej))

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None:
            if self.vfix is None and not self.blending and q[0] >= self.x_blend:
                self.judge._stop(self.t, 'segtarget_no_fix', rej=dict(self.seg_rej), n=len(self.seg_log))
                return
            if self.vfix is not None and self.t_close_req is None and self.t - self.vfix['t'] > self.seg['expire_s']:
                self.judge._stop(self.t, 'segtarget_expired', age=self.t - self.vfix['t'])
                return
        super()._advance_inner(T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['segtarget'] = dict(fix=None if self.vfix is None else dict(self.vfix, xy=self.vfix['xy'].tolist()), z_support=self.seg_z, seq=self.seg_seq, adopt=self.seg_adopt,
                              rejected=self.seg_rej, n_intake=len(self.seg_log), ignored_v2=self.ign_v2, ignored_w1=self.ign_w1, reg=self.seg,
                              priors=['bottle_radius_15mm', 'grasp_band_0.10m'], measured=['center_xy', 'z_support'],
                              note='目標只來自 known_type 快照；V2／W1 舊估計只記錄；備妥姿態沿用名目分支；淨空拒絕線與禁止接觸仍用模擬真值（安全網）')
        return r
