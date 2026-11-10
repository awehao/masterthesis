#!/usr/bin/env python3
"""底盤／手腕雙相機的候選選擇與交接（Howard 11/11「要善用兩顆鏡頭」；Codex 20261111_050821 (c)）。純邏輯，不訂閱 ROS；節點與離線重放共用。
來源與角色（依實錄重放 results/vision/seg3d_dualcam_tpa_r1 的實際能力，不是假設）：
  v1 底盤相機：高 0.31 m、在桌面下方 ⇒ 支撐面不可見；任務 2–4 s（距離 3–4 m）起 xy 有效（中心誤差中位數 0.6 mm、最大 2.1 mm）。
     ⇒ 形成「接近導引候選」guide（只有 xy）：可提早決定目標專屬車道、左右側與手腕觀看點；不能單獨給抓取高度。
  w1 手腕相機：抬高後支撐面可見 ⇒ 形成 known_type 抓取候選 grasp（xy＋支撐高度；瓶高、抓持帶為瓶型先驗）；收攏時也可能只有 xy。
每個來源各自一個 SourceTracker（沿用 bottle_seg3d_candidates 的規則：連續 ≥ K_MIN 列、跨幀 std、MAX_AGE、窗長；xy 候選只要求 xy_valid）。
DualSelect.evaluate(t_now)：
  1 各來源候選（帶 camera_id、擷取時刻 t_first／t_last、年齡＝t_now − t_last（擷取時刻，不因換來源刷新）、列數、std、版本號）。
  2 同物件核對：兩來源 xy 都 ready ⇒ |Δxy| ≤ TOL_SAME 一致；> TOL_OTHER 視為不同物件（只保留與目前採用同物件者；尚未採用 ⇒ 兩者都不採用並記 ambiguous）；
     介於兩者 ⇒ inconsistent：維持目前採用、不切換、不平均；直到出現一致比對前，未核對的單一來源候選也不採用；持續 > T_INCONS ⇒ reject（交控制端受控停止）。
     抓取候選（只來自手腕）與其他來源的 xy 交叉核對；未交叉核對者標 *_unchecked。
  3 guide＝grasp ready 時取 grasp 的 xy（手腕近距、含支撐）；否則取 v1 xy；都沒有 ⇒ None。grasp 只來自支撐可觀測的來源。
  4 採用遲滯：已採用來源仍 ready 且一致 ⇒ 不換；換來源記 handoff（t、from、to、Δxy）。不盲目平均兩來源。
輸出只是候選與交接紀錄；目標的限速更新（5 mm/s）、凍結、期限仍由控制端 segtarget 規則處理（切換不跳設定點）。
"""
import numpy as np

import bottle_seg3d_candidates as CT

TOL_SAME, TOL_OTHER, T_INCONS = 0.010, 0.10, 1.0


def _xyok(r):
    return r.get('valid', 0) > 0.5 and r.get('xy_valid', 0) > 0.5


class SourceTracker:
    def __init__(self, camera_id, k_min=CT.K_MIN, window_s=CT.WINDOW_S, max_age_s=CT.MAX_AGE_S):
        self.cam, self.k_min, self.window_s, self.max_age_s = camera_id, int(k_min), float(window_s), float(max_age_s)
        self.rows, self.n_rej, self.ver = [], dict(age=0, order=0, nonfinite=0), 0

    def add(self, r):
        if not (np.isfinite(r.get('stamp', np.nan)) and np.isfinite(r.get('recv_age_s', np.nan))):
            self.n_rej['nonfinite'] += 1
            return False
        if not 0.0 <= r['recv_age_s'] <= self.max_age_s:
            self.n_rej['age'] += 1                                          # 收到時已舊（舊影格）不收
            return False
        if self.rows and r['stamp'] <= self.rows[-1]['stamp']:
            self.n_rej['order'] += 1
            return False
        self.rows.append(dict(r, camera_id=self.cam))
        return True

    def evaluate(self, t_now):
        W = [r for r in self.rows if t_now - self.window_s <= r['stamp'] <= t_now + 1e-9]
        out = {}
        for kind, pred, need_sup in (('xy', _xyok, False), ('grasp', CT._known, True)):
            c = dict(kind=kind, camera_id=self.cam, status='not_ready', why=None)
            if len(W) < self.k_min:
                c['why'] = 'rows'
            elif t_now - W[-1]['stamp'] > self.max_age_s:
                c['why'] = 'stale'
            else:
                T = CT._tail(W, pred)
                if len(T) < self.k_min:
                    c['why'] = 'unobservable'
                else:
                    cx = np.array([r['cx_known_r'] for r in T])
                    cy = np.array([r['cy_known_r'] for r in T])
                    sd = dict(xy=float(max(cx.std(), cy.std())))
                    if need_sup:
                        zs = np.array([r['z_support'] for r in T])
                        sd['sup'] = float(zs.std())
                    c.update(n=len(T), std=sd, t_first=float(T[0]['stamp']), t_last=float(T[-1]['stamp']), age_s=float(t_now - T[-1]['stamp']),
                             center=[float(cx.mean()), float(cy.mean())])
                    if need_sup:
                        c.update(z_support=float(zs.mean()), z_grasp=float(zs.mean()) + CT.BAND_PRIOR)
                    ok = sd['xy'] <= CT.STD_XY and sd.get('sup', 0.0) <= CT.STD_SUP
                    c['status'] = 'ready' if ok else 'not_ready'
                    c['why'] = None if ok else 'inconsistent'
            out[kind] = c
        return out


class DualSelect:
    def __init__(self, v1=None, w1=None):
        self.src = dict(v1=v1 or SourceTracker('v1'), w1=w1 or SourceTracker('w1'))
        self.adopt = dict(guide=None, grasp=None)                           # 目前採用（來源、中心）
        self.t_incons, self.handoffs, self.log = None, [], []
        self.first = dict(guide=None, grasp=None)

    def add(self, cam, r):
        return self.src[cam].add(r)

    def _pick(self, role, cands, others, t_now):
        """cands：該角色可用（ready）候選；others：其他來源的 xy 候選（交叉核對用）。回 (採用候選或 None, 狀態)。
        不一致是物件層級：一旦兩來源不一致，直到出現一致比對前，任何角色的未核對候選都不採用；持續 > T_INCONS ⇒ reject。"""
        cur = self.adopt[role]

        def dist(a_, b_):
            return float(np.hypot(*(np.asarray(a_, float) - np.asarray(b_, float))))
        if not cands:
            return None, 'none'
        if cur is not None:
            cands = [c for c in cands if dist(c['center'], cur['center']) <= TOL_OTHER]   # 只保留與目前採用同物件者
            if not cands:
                return None, 'adopted_object_lost'
        keep = [c for c in cands if cur is not None and c['camera_id'] == cur['camera_id']]
        c = keep[0] if keep else sorted(cands, key=lambda z: ('w1', 'v1').index(z['camera_id']))[0]
        cmp_ = [o for o in cands + others if o['camera_id'] != c['camera_id']]
        ds = [dist(c['center'], o['center']) for o in cmp_]
        if cur is None and any(d > TOL_OTHER for d in ds):
            return None, 'ambiguous_two_objects'                             # 尚未採用時兩來源指向不同物件：都不採用
        same = [d for d in ds if d <= TOL_OTHER]
        if any(d > TOL_SAME for d in same):
            self.t_incons = t_now if self.t_incons is None else self.t_incons
            return None, ('reject' if t_now - self.t_incons > T_INCONS else 'inconsistent')   # 不採用新值、不切換、不平均（控制端維持既有採用）
        if same:
            self.t_incons = None                                             # 有一致比對才重設（物件層級，兩角色共用）
        elif self.t_incons is not None:
            return None, ('reject' if t_now - self.t_incons > T_INCONS else 'inconsistent')   # 曾不一致且未獲一致比對：未核對候選不採用
        return c, ('kept' if keep else 'selected') + ('' if same else '_unchecked')

    def evaluate(self, t_now):
        ev = {cam: tr.evaluate(t_now) for cam, tr in self.src.items()}
        xy_r = {c: ev[c]['xy'] for c in ('v1', 'w1') if ev[c]['xy']['status'] == 'ready'}
        grasp_c = [ev[c]['grasp'] for c in ('w1', 'v1') if ev[c]['grasp']['status'] == 'ready']
        g, gst = self._pick('grasp', grasp_c, [xy_r[c] for c in xy_r if all(c != gg['camera_id'] for gg in grasp_c)], t_now)
        guide_c = ([dict(g, kind='xy')] if g is not None else []) + [xy_r[c] for c in xy_r if g is None or c != g['camera_id']]
        u, ust = self._pick('guide', guide_c, [], t_now)
        res = dict(t=float(t_now), guide=u, guide_state=ust, grasp=g, grasp_state=gst, sources={c: {k: v['status'] if v['status'] == 'ready' else v['why'] for k, v in e.items()} for c, e in ev.items()})
        for role, c in (('guide', u), ('grasp', g)):
            cur = self.adopt[role]
            if c is not None:
                if cur is not None and cur['camera_id'] != c['camera_id']:
                    self.handoffs.append(dict(t=round(float(t_now), 3), role=role, frm=cur['camera_id'], to=c['camera_id'],
                                              dxy_mm=round(float(np.hypot(*(np.array(c['center']) - cur['center']))) * 1e3, 2), age_s=round(c['age_s'], 3)))
                self.adopt[role] = dict(camera_id=c['camera_id'], center=np.array(c['center'], float), t_last=c['t_last'])
                if self.first[role] is None:
                    self.first[role] = dict(t=round(float(t_now), 3), camera_id=c['camera_id'], age_s=round(c['age_s'], 3), state=(ust if role == 'guide' else gst))
        if 'reject' in (ust, gst):
            res['reject'] = True
        self.log.append(dict(t=round(float(t_now), 3), guide=None if u is None else u['camera_id'], guide_state=ust, grasp=None if g is None else g['camera_id'], grasp_state=gst))
        return res
