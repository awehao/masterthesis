#!/usr/bin/env python3
"""任務端的三維目標候選（Codex 20261101_232935 第 2 項、234651；只記錄、不採用、不閉爪，不回填桌高）。
輸入：手腕旁路輸出列（bottle_seg3d schema ≥ 3；stamp 為擷取時刻，recv_age_s 為收到時年齡）。兩類候選並列，各自有狀態 ready／not_ready（附原因）：
  complete     ＝最近連續 ≥ K_MIN 列「同一列完整」（valid、xy、瓶頂、支撐、高度同時成立），且跨幀一致（中心 std ≤ 3 mm、支撐 std ≤ 3 mm、瓶頂 std ≤ 5 mm）。
                 量測：中心（已知半徑圓擬合 ⇒ 先驗：瓶半徑）、支撐 z、瓶頂 z、高度。
  known_type   ＝最近連續 ≥ K_MIN 列 xy_valid 且 support_observed（瓶頂可缺），跨幀一致（中心、支撐）。
                 量測：中心、支撐 z；先驗：瓶半徑、瓶高 0.20 m（推得瓶頂）、抓持帶在瓶底上 0.10 m（同 W1 的 grasp_target 高度）。
  not_ready 原因：rows（列數不足）、stale（窗內最新列在查詢時刻已逾 MAX_AGE_S）、inconsistent、unobservable（沒有符合條件的連續列）。
可達與淨空（ReachWorker，背景執行緒，不阻塞呼叫端；未完成 ⇒ reach＝pending）：
  抓持 TCP 目標＝bottle_s0_ctrl.grasp_target(中心, 支撐 z＋0.10)；底盤取名目抓取相對位置（瓶 y−0.42、x 在瓶 x 加 −0.30／−0.10／+0.10），任一點 IK 有解
  （關節餘裕 ≥ 0.05）且瓶心 0.10 m 內（實際觀測到支撐面的範圍）的非手指連桿點高於觀測支撐面＋0.02 m ⇒ reach_ok。
  「候選可達」不是抓取成功；0.10 m 以外的桌面延伸（桌緣、桌腳）未觀測、未核對。known_type 的瓶頂與抓持帶來自瓶型先驗，對同型瓶誤差為 0 不代表量測精度。
"""
import math
import threading

import numpy as np

K_MIN, MAX_AGE_S, WINDOW_S = 3, 0.5, 2.0
STD_XY, STD_SUP, STD_TOP = 0.003, 0.003, 0.005
H_PRIOR, BAND_PRIOR, R_PRIOR = 0.20, 0.10, 0.015
BASE_DX, BASE_DY = (-0.30, -0.10, 0.10), -0.42
DISK_R, CLEAR_SUP = 0.10, 0.02                         # 只核實際觀測到支撐面的範圍（locate 支撐搜尋半徑 0.10 m）；0.40 m 會把站在桌邊的底盤與手臂根部也圈進來
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
    """名目抓取相對位置下的 IK 與觀測支撐面淨空（純運動學；慢，給 ReachWorker 在背景跑）。"""
    import bottle_reach_scan as RS
    import bottle_s0_ctrl as C
    import bottle_s1_geom as G
    cx, cy = cand['center']
    T = C.grasp_target(cx, cy, cand['z_grasp'])
    res = []
    for dx in BASE_DX:
        base = np.array([cx + dx, cy + BASE_DY, 0.0])
        ok, _, qa, pe, re, marg = RS.ik(base, T, [np.asarray(q_seed, float)], iters=iters)
        if not ok or marg < 0.05:
            res.append(dict(dx=dx, ik=bool(ok), margin=float(marg), clear=None))
            continue
        W = G.link_world(np.r_[base, qa], G.Q_OPEN)
        P = np.vstack([v for k, v in W.items() if k not in FINGERS])
        near = np.hypot(P[:, 0] - cx, P[:, 1] - cy) <= DISK_R
        clr = float((P[near, 2] - cand['z_support']).min()) if near.any() else math.inf
        res.append(dict(dx=dx, ik=True, margin=float(marg), clear=clr))
    good = [x for x in res if x['ik'] and x['clear'] is not None and x['clear'] >= CLEAR_SUP]
    return dict(reach_ok=bool(good), per_base=res)


class ReachWorker:
    """背景執行緒：同一個候選（類別＋中心 1 cm 格＋支撐 1 cm 格）只算一次；查詢時未完成 ⇒ pending。"""

    def __init__(self, q_seed):
        self.q_seed, self.cache, self.lock, self.busy = np.asarray(q_seed, float), {}, threading.Lock(), set()

    @staticmethod
    def key(c):
        return (c['kind'], round(c['center'][0], 2), round(c['center'][1], 2), round(c['z_support'], 2))

    def query(self, c):
        if c.get('status') != 'ready':
            return dict(reach='n/a')
        k = self.key(c)
        with self.lock:
            if k in self.cache:
                return dict(reach='ok' if self.cache[k]['reach_ok'] else 'fail', detail=self.cache[k])
            if k not in self.busy:
                self.busy.add(k)
                threading.Thread(target=self._run, args=(k, dict(c)), daemon=True).start()
        return dict(reach='pending')

    def _run(self, k, c):
        try:
            r = reach_check(c, self.q_seed)
        except Exception as e:                                    # noqa: BLE001
            r = dict(reach_ok=False, error=repr(e))
        with self.lock:
            self.cache[k] = r
            self.busy.discard(k)
