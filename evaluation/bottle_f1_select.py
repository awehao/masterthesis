#!/usr/bin/env python3
"""F1 多實例選物：給定任務子區域＋時序穩定＋鎖定（Codex 20261102 F1 裁定、225852）。純邏輯、不依賴 ROS，供 bottle_seg3d_f1_node 用；在建立候選快照之前完成，不同瓶的資料不會混進同一視窗。
輸入：一幀全部實例的 bottle_seg_depth3d.locate 結果（依信心排序）。輸出：要發布的 r（合格實例或 valid=False＋why）、選中索引、尺寸閘拒絕數，與逐實例紀錄。真值物件 ID 不進來。
每實例的位置：xy_valid ⇒ center_known_r；否則 xy_valid_silhouette ⇒ center_silhouette；否則 surface_centroid（只作分區判斷）。合格＝valid 且通過尺寸閘（同原節點）。
搜尋狀態：
  歧義（ambiguous，不任選最高信心）：區域內合格實例 ≥ 2；尺寸閘拒絕的實例落在區域外擴 M_BOUND 內（疑似合併）；合格實例落在分界線 ±M_BOUND 內（關聯不明）。
  區域內恰一個合格且 xy_valid：與上一幀差 ≤ R_LOCK 就累計，否則從 1 重算；累計到 K_LOCK ⇒ 鎖定（中心＝這幾幀平均）。累計期間發 why＝locking，不放行。
  其餘：why＝out_of_region（有合格實例但不在區域）或原因（無合格實例）；累計歸零。
鎖定後（鎖定中心固定，不跟隨）：
  距鎖定中心 ≤ R_LOCK 的合格實例恰一個 ⇒ 放行該實例；多於一個或該範圍內有尺寸閘拒絕的實例 ⇒ ambiguous；沒有 ⇒ lock_no_match。都不改鎖另一瓶。
R_LOCK 是身分關聯範圍，不是精度門檻：精度仍由既有候選（dev_tol 2 mm、supersede）管。依單瓶三趟接近段（bottle_rb1_gaze_1／online5_u3_1／rb2_gaze_1，起跑前）：對真值最大 1.8 mm、逐幀差最大 2.1 mm；
2 mm 當身分門檻會誤拒，所以取 50 mm（兩瓶間距 0.30 m 的 1/6）。首版不做重新鎖定：要重選須另外作廢候選與計畫。
"""
import math

import bottle_seg_depth3d as S3

K_LOCK, R_LOCK, M_BOUND = 3, 0.05, 0.05


def inst_pos(e):
    """(x, y, 來源)；沒有可用位置回傳 None。"""
    for key, flag in (('center_known_r', 'xy_valid'), ('center_silhouette', 'xy_valid_silhouette'), ('surface_centroid', None)):
        c = e.get(key)
        if c is not None and (flag is None or e.get(flag)) and all(math.isfinite(float(v)) for v in c[:2]):
            return float(c[0]), float(c[1]), key
    return None


def qualified(e):
    return bool(e.get('valid')) and e['r_silhouette'] <= S3.SIZE_GATE['r_max'] and e['z_visible_span'] <= S3.SIZE_GATE['h_max']


class F1Selector:
    def __init__(self, region, split_x=0.0, k_lock=K_LOCK, r_lock=R_LOCK, m_bound=M_BOUND):
        self.region, self.split_x, self.k_lock, self.r_lock, self.m_bound = tuple(float(v) for v in region), float(split_x), int(k_lock), float(r_lock), float(m_bound)
        self.lock, self.hist, self.t_lock = None, [], None

    def _in(self, p, m=0.0):
        x0, x1, y0, y1 = self.region
        return x0 - m <= p[0] <= x1 + m and y0 - m <= p[1] <= y1 + m

    def step(self, st, ests, confs=None):
        """回傳 (r, k_sel, n_rej, rec)。"""
        rec = []
        for k, e in enumerate(ests):
            p = inst_pos(e)
            rec.append(dict(k=k, conf=None if confs is None else float(confs[k]), valid=bool(e.get('valid')), qual=qualified(e), why=e.get('why'), xy=None if p is None else [round(p[0], 4), round(p[1], 4)], src=None if p is None else p[2],
                            r_sil=e.get('r_silhouette'), z_span=e.get('z_visible_span'), xy_valid=bool(e.get('xy_valid')), support=bool((e.get('support') or {}).get('observable')), z_support=(e.get('support') or {}).get('z'), n_pts=e.get('n_pts')))
        n_rej = sum(1 for x in rec if x['valid'] and not x['qual'])
        Q = [x for x in rec if x['qual'] and x['xy'] is not None]
        R = [x for x in rec if x['valid'] and not x['qual'] and x['xy'] is not None]
        if self.lock is not None:
            near = [x for x in Q if math.hypot(x['xy'][0] - self.lock[0], x['xy'][1] - self.lock[1]) <= self.r_lock]
            bad = [x for x in R if math.hypot(x['xy'][0] - self.lock[0], x['xy'][1] - self.lock[1]) <= self.r_lock]
            if len(near) > 1 or bad:
                return dict(valid=False, why='ambiguous'), None, n_rej, dict(state='locked', why='ambiguous', inst=rec, lock=self.lock)
            if not near:
                return dict(valid=False, why='lock_no_match'), None, n_rej, dict(state='locked', why='lock_no_match', inst=rec, lock=self.lock)
            k = near[0]['k']
            return ests[k], k, n_rej, dict(state='locked', why='ok', sel=k, inst=rec, lock=self.lock)
        inr = [x for x in Q if self._in(x['xy'])]
        amb = []
        if len(inr) >= 2:
            amb.append('n_in_region')
        if any(self._in(x['xy'], self.m_bound) for x in R):
            amb.append('size_rejected_near_region')
        if any(abs(x['xy'][0] - self.split_x) <= self.m_bound for x in Q):
            amb.append('near_split')
        if amb:
            self.hist = []
            return dict(valid=False, why='ambiguous'), None, n_rej, dict(state='search', why='ambiguous', amb=amb, inst=rec)
        if len(inr) == 1 and inr[0]['xy_valid']:
            x = inr[0]
            if self.hist and math.hypot(x['xy'][0] - self.hist[-1][0], x['xy'][1] - self.hist[-1][1]) > self.r_lock:
                self.hist = []
            self.hist.append(tuple(x['xy']))
            if len(self.hist) >= self.k_lock:
                n = len(self.hist)
                self.lock = (sum(h[0] for h in self.hist) / n, sum(h[1] for h in self.hist) / n)
                self.t_lock = float(st)
                return ests[x['k']], x['k'], n_rej, dict(state='locked', why='ok', sel=x['k'], just_locked=True, inst=rec, lock=self.lock)
            return dict(valid=False, why='locking'), None, n_rej, dict(state='search', why='locking', n_hist=len(self.hist), inst=rec)
        self.hist = []
        if inr or Q:
            why = 'out_of_region'
        elif R:
            why = 'size_gate_rejected_all'
        else:
            why = next((e.get('why') for e in ests if not e.get('valid')), 'no_instance')
        return dict(valid=False, why=why), None, n_rej, dict(state='search', why=why, inst=rec)
