#!/usr/bin/env python3
"""任務端的三維目標候選（Codex 20261101_232935 第 2 項、234651；只記錄、不採用、不閉爪，不回填桌高）。
輸入：手腕旁路輸出列（bottle_seg3d schema ≥ 3；stamp 為擷取時刻，recv_age_s 為收到時年齡）。兩類候選並列，各自有狀態 ready／not_ready（附原因）：
  complete     ＝最近連續 ≥ K_MIN 列「同一列完整」（valid、xy、瓶頂、支撐、高度同時成立），且跨幀一致（中心 std ≤ 3 mm、支撐 std ≤ 3 mm、瓶頂 std ≤ 5 mm）。
                 量測：中心（已知半徑圓擬合 ⇒ 先驗：瓶半徑）、支撐 z、瓶頂 z、高度。
  known_type   ＝最近連續 ≥ K_MIN 列 xy_valid 且 support_observed（瓶頂可缺），跨幀一致（中心、支撐）。
                 量測：中心、支撐 z；先驗：瓶半徑、瓶高 0.20 m（推得瓶頂）、抓持帶在瓶底上 0.10 m（同 W1 的 grasp_target 高度）。
  not_ready 原因：rows（列數不足）、stale（窗內最新列在查詢時刻已逾 MAX_AGE_S）、inconsistent、unobservable（沒有符合條件的連續列）。
三種狀態分開（Codex 20261102_001039）：
  1 觀測候選：上述 complete／known_type 的 ready／not_ready（穩定、新鮮）；
  2 ik_state：在**假設**的名目抓取底盤位置（瓶 y−0.42、x＝瓶 x＋−0.30／−0.10／+0.10）下，抓持終點 TCP＝grasp_target(中心, 支撐＋0.10) 任一點 IK 有解（關節餘裕 ≥ 0.05）；
  3 local_clear_state：該解下瓶心 0.10 m 內的非手指連桿點高於觀測支撐面 ≥ 0.02 m（局部點檢查值；0.10 m 是縮小的檢查範圍，不是消除碰撞問題，也不代表整個圓盤都已觀測）。
  都不包含：底盤能否抵達、預抓／插入／抬升路徑、自碰、完整環境碰撞。「候選可達」不是抓取成功。
  2、3 由 ReachWorker 單一背景執行緒計算（最新待算槽、不阻塞）；結果只對實際檢查的候選版本有效。
known_type 的瓶頂與抓持帶來自瓶型先驗，對同型瓶誤差為 0 不代表量測精度。
"""
import math
import threading

import numpy as np

K_MIN, MAX_AGE_S, WINDOW_S = 3, 0.5, 2.0
STD_XY, STD_SUP, STD_TOP = 0.003, 0.003, 0.005
H_PRIOR, BAND_PRIOR, R_PRIOR = 0.20, 0.10, 0.015
BASE_DX, BASE_DY = (-0.30, -0.10, 0.10), -0.42
LOCAL_R, CLEAR_SUP = 0.10, 0.02                         # 只核實際觀測到支撐面的範圍（locate 支撐搜尋半徑 0.10 m）；0.40 m 會把站在桌邊的底盤與手臂根部也圈進來
FINGERS = ('uflite_finger1', 'uflite_finger2')


def _complete(r):
    return all(r[k] > 0.5 for k in ('valid', 'xy_valid', 'top_observed', 'support_observed', 'height_valid'))


def _known(r):
    return r['valid'] > 0.5 and r['xy_valid'] > 0.5 and r['support_observed'] > 0.5


def _tail(rows, pred):
    """最近連續滿足 pred 的列（從最新往回）。"""
    out = []
    for r in reversed(rows):
        if not pred(r):
            break
        out.append(r)
    return out[::-1]


class CandidateTracker:
    def __init__(self):
        self.rows = []

    def add(self, r):
        if not (np.isfinite(r.get('stamp', np.nan)) and np.isfinite(r.get('recv_age_s', np.nan)) and 0.0 <= r['recv_age_s'] <= MAX_AGE_S):
            return False
        if self.rows and r['stamp'] <= self.rows[-1]['stamp']:
            return False
        self.rows.append(r)
        return True

    def evaluate(self, t_now):
        W = [r for r in self.rows if t_now - WINDOW_S <= r['stamp'] <= t_now + 1e-9]
        out = {}
        for kind, pred in (('complete', _complete), ('known_type', _known)):
            c = dict(kind=kind, status='not_ready', why=None)
            if len(W) < K_MIN:
                c['why'] = 'rows'
            elif t_now - W[-1]['stamp'] > MAX_AGE_S:
                c['why'] = 'stale'
            else:
                T = _tail(W, pred)
                if len(T) < K_MIN:
                    c['why'] = 'unobservable'
                else:
                    cx = np.array([r['cx_known_r'] for r in T])
                    cy = np.array([r['cy_known_r'] for r in T])
                    zs = np.array([r['z_support'] for r in T])
                    sd = dict(xy=float(max(cx.std(), cy.std())), sup=float(zs.std()))
                    if kind == 'complete':
                        zt = np.array([r['z_top'] for r in T])
                        sd['top'] = float(zt.std())
                    ok = sd['xy'] <= STD_XY and sd['sup'] <= STD_SUP and sd.get('top', 0.0) <= STD_TOP
                    c.update(n=len(T), std=sd, t_first=T[0]['stamp'], t_last=T[-1]['stamp'], age_s=t_now - T[-1]['stamp'],
                             center=[float(cx.mean()), float(cy.mean())], z_support=float(zs.mean()))
                    if kind == 'complete':
                        c.update(z_top=float(zt.mean()), height=float(zt.mean() - zs.mean()), measured=['center', 'z_support', 'z_top', 'height'], priors=['bottle_radius'])
                    else:
                        c.update(z_top=float(zs.mean()) + H_PRIOR, measured=['center', 'z_support'], priors=['bottle_radius', 'bottle_height_0.20', 'grasp_band_0.10'])
                    c['z_grasp'] = float(zs.mean()) + BAND_PRIOR
                    c['priors'] = c['priors'] + ['grasp_band_0.10'] if kind == 'complete' else c['priors']
                    if ok:
                        c['status'] = 'ready'
                    else:
                        c['why'] = 'inconsistent'
            out[kind] = c
        return out


def reach_check(cand, q_seed, iters=150):
    """兩項分開回報（純運動學；慢，給 ReachWorker 在背景跑）：
    ik＝在假設的名目抓取底盤位置（瓶 y−0.42、x＝瓶 x＋−0.30／−0.10／+0.10）下，抓持終點 TCP（grasp_target(中心, 支撐＋0.10)）IK 有解且關節餘裕 ≥ 0.05；
    local_clear＝該解下，瓶心 LOCAL_R 內的非手指連桿點高於觀測支撐面的最小值（局部點檢查值）。
    不包含：底盤能否抵達該位置、預抓／插入／抬升路徑、自碰、完整環境碰撞（桌緣、桌腳等未觀測部分）。"""
    import bottle_reach_scan as RS
    import bottle_s0_ctrl as C
    import bottle_s1_geom as G
    cx, cy = cand['center']
    side = cand.get('side', 'left')                                      # 右側整合：控制端回傳的抓取側（未給＝左，舊行為逐元素相同）
    if side == 'left':
        T = C.grasp_target(cx, cy, cand['z_grasp'])
    else:
        import bottle_side as BSD
        T = BSD.grasp_target(cx, cy, cand['z_grasp'], side)
    dy_base = BASE_DY if side == 'left' else -BASE_DY
    res = []
    for dx in BASE_DX:
        base = np.array([cx + dx, cy + dy_base, 0.0])
        seeds_ = [np.asarray(q_seed, float)]
        if side != 'left':
            sm_ = np.asarray(q_seed, float).copy()
            sm_[0] = -sm_[0]
            seeds_ = [sm_] + seeds_                                       # 右側：多一個 q1 取反的 IK 起點（只是種子，可行性仍由 IK 與餘裕判定）
        ok, _, qa, pe, re, marg = RS.ik(base, T, seeds_, iters=iters)
        if not ok or marg < 0.05:
            res.append(dict(dx=dx, ik=False, margin=float(marg), local_clear=None))
            continue
        W = G.link_world(np.r_[base, qa], G.Q_OPEN)
        P = np.vstack([v for k, v in W.items() if k not in FINGERS])
        near = np.hypot(P[:, 0] - cx, P[:, 1] - cy) <= LOCAL_R
        clr = float((P[near, 2] - cand['z_support']).min()) if near.any() else math.inf
        res.append(dict(dx=dx, ik=True, margin=float(marg), local_clear=clr))
    ik_ok = [x for x in res if x['ik']]
    loc_ok = [x for x in ik_ok if x['local_clear'] is not None and x['local_clear'] >= CLEAR_SUP]
    return dict(ik_state='ok' if ik_ok else 'fail', local_clear_state='ok' if loc_ok else ('fail' if ik_ok else 'n/a'),
                local_clear_min=max((x['local_clear'] for x in ik_ok), default=None), per_base=res,
                scope='假設底盤位置下的抓持終點 IK＋瓶心 0.10 m 內局部點檢查；不含底盤可達、預抓／插入／抬升路徑、自碰、完整環境碰撞', side=side)


def cand_id(c):
    """候選版本：類別、來源列戳（最早、最新）、列數與精確值（不取格）。"""
    return (c['kind'], round(c['t_first'], 6), round(c['t_last'], 6), int(c['n']), round(c['center'][0], 7), round(c['center'][1], 7), round(c['z_support'], 7))


class ReachWorker:
    """單一背景執行緒＋「最新待算」槽（Codex 20261102_001039）：query(c) 若 c 的版本已算過 ⇒ 回該結果；否則把 c 放進待算槽（取代較舊的待算者）並回 pending。
    結果只對「實際被檢查的那個版本」有效：目前候選版本不同時，ik_state／local_clear_state 一律 pending，另附上次實際檢查的版本、目標、結果（last_*）與差距（mm），
    由使用端自行判斷，不把它冒充目前版本的結果。
    被取代或過期的版本即使之後算完，也不會變成目前結果（只有版本完全相同才回報）。"""

    def __init__(self, q_seed):
        self.q_seed = np.asarray(q_seed, float)
        self.lock, self.ev = threading.Lock(), threading.Event()
        self.pending, self.done, self.n_run, self.n_replaced = None, {}, 0, 0
        self.last = None
        self.alive = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def query(self, c):
        if c.get('status') != 'ready':
            return dict(ik_state='n/a', local_clear_state='n/a')
        k = cand_id(c)
        with self.lock:
            if k in self.done:
                r = self.done[k]
                return dict(ik_state=r['ik_state'], local_clear_state=r['local_clear_state'], local_clear_min=r['local_clear_min'], checked_id=k, scope=r['scope'])
            if self.pending is not None and self.pending[0] != k:
                self.n_replaced += 1
            self.pending = (k, dict(c))
            self.ev.set()
            last = self.last
        out = dict(ik_state='pending', local_clear_state='pending')
        if last is not None:
            lk, lc = last
            lr = self.done.get(lk) or {}
            out.update(last_checked_id=lk, last_checked_target=[lc['center'][0], lc['center'][1], lc['z_grasp']],
                       last_ik_state=lr.get('ik_state', 'n/a'), last_local_clear_state=lr.get('local_clear_state', 'n/a'), last_local_clear_min=lr.get('local_clear_min'),
                       delta_mm=round(float(math.hypot(c['center'][0] - lc['center'][0], c['center'][1] - lc['center'][1], c['z_grasp'] - lc['z_grasp']) * 1e3), 3))
        return out

    def _loop(self):
        while self.alive:
            if not self.ev.wait(0.2):
                continue
            with self.lock:
                item, self.pending = self.pending, None
                self.ev.clear()
            if item is None:
                continue
            k, c = item
            try:
                r = reach_check(c, self.q_seed)
            except Exception as e:                                # noqa: BLE001
                r = dict(ik_state='fail', local_clear_state='n/a', local_clear_min=None, error=repr(e), scope='error')
            with self.lock:
                self.done[k] = r
                self.last = (k, c)
                self.n_run += 1
                while len(self.done) > 64:                        # 有界：只留最近 64 個版本的結果
                    del self.done[next(iter(self.done))]

    def stop(self):
        self.alive = False


SNAP_TOL, SNAP_MAX_AGE = 0.002, 3.0


class SnapshotChecker:
    """不可變快照的檢查（Codex 20261102_002139 第 1 項）：同一類候選 ready 時建立快照（序號、來源列戳、精確目標），交給單一背景工作者；
    檢查完成時呼叫 on_done(快照, 結果) 主動發布，不等下一張影像。快照在檢查中不被取代；只有目前觀測偏離快照 > SNAP_TOL（中心或支撐）
    或快照（以最新來源列計）超過 SNAP_MAX_AGE 時，才在下一個 ready 候選建立新快照。使用端採用時仍須自行核對快照年齡與目前觀測是否一致。"""

    def __init__(self, q_seed, on_done=None):
        self.q_seed, self.on_done = np.asarray(q_seed, float), on_done
        self.lock, self.ev = threading.Lock(), threading.Event()
        self.snap, self.seq, self.queue = {}, 0, {}                 # 待算槽每類一格（兩類同拍建快照時不互相覆蓋）
        self.n_run, self.alive = 0, True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    @staticmethod
    def deviation(c, s):
        return max(math.hypot(c['center'][0] - s['center'][0], c['center'][1] - s['center'][1]), abs(c['z_support'] - s['z_support']))

    def update(self, c, t_now):
        """c：目前評估的候選（同類）。回傳此類目前快照的摘要（含與目前觀測的偏差、是否一致、年齡）。"""
        kind = c['kind']
        with self.lock:
            s = self.snap.get(kind)
            if c.get('status') == 'ready':
                need = s is None or (s['result'] is not None and (self.deviation(c, s['cand']) > SNAP_TOL or t_now - s['cand']['t_last'] > SNAP_MAX_AGE))
                if need:
                    self.seq += 1
                    s = dict(seq=self.seq, cand={k: (list(v) if isinstance(v, list) else v) for k, v in c.items()}, t_created=t_now, result=None, t_done=None)
                    self.snap[kind] = s
                    self.queue[kind] = s['seq']
                    self.ev.set()
            if s is None:
                return dict(seq=0)
            dev = self.deviation(c, s['cand']) if c.get('status') == 'ready' else None
            r = s['result'] or {}
            return dict(seq=s['seq'], t_first=s['cand']['t_first'], t_last=s['cand']['t_last'], n=s['cand']['n'], center=s['cand']['center'], z_support=s['cand']['z_support'],
                        z_grasp=s['cand']['z_grasp'], ik_state=r.get('ik_state', 'pending'), local_clear_state=r.get('local_clear_state', 'pending'), local_clear_min=r.get('local_clear_min'),
                        age_s=t_now - s['cand']['t_last'], deviation_mm=None if dev is None else dev * 1e3, consistent=None if dev is None else bool(dev <= SNAP_TOL), t_done=s['t_done'])

    def _loop(self):
        while self.alive:
            if not self.ev.wait(0.2):
                continue
            with self.lock:
                item = self.queue.popitem() if self.queue else None
                if not self.queue:
                    self.ev.clear()
                s = None if item is None else self.snap.get(item[0])
                if s is not None and s['seq'] != item[1]:
                    s = None
                cand = None if s is None else dict(s['cand'])
            if s is None:
                continue
            try:
                r = reach_check(cand, self.q_seed)
            except Exception as e:                                # noqa: BLE001
                r = dict(ik_state='fail', local_clear_state='n/a', local_clear_min=None, error=repr(e))
            import time as _t
            with self.lock:
                s['result'], s['t_done'] = r, _t.perf_counter()
                self.n_run += 1
            if self.on_done is not None:
                try:
                    self.on_done(s, r)
                except Exception:                                 # noqa: BLE001  回呼出錯不得讓工作者執行緒結束
                    import traceback
                    traceback.print_exc()

    def stop(self):
        self.alive = False
