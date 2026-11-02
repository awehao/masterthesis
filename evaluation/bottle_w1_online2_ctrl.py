#!/usr/bin/env python3
"""區域導引在線 MotM 第二版（Codex 20261102_075357：觀測新鮮度改用來源擷取戳；版本化子類，基線＝bottle_w1_online_ctrl〔U1、U2 的版本〕，其餘全部相同）。
第一版的承接條件「最新合格觀測距今 ≤ 2.0 s」量的是 _seg_freeze 回呼時刻，不是影像擷取時刻，且 2 s 寬於原觀測條件 0.5 s——不能作為新鮮度證據。本版：
1 每筆候選進來時記來源擷取時刻 t_src＝收到時刻 − 轉送當下的觀測年齡 age_s（轉接器在轉送當下重算年齡；任務時間）。
  合格快照（_seg_freeze）存 t_src；同一筆舊觀測被重送，t_src 不變，不會變新鮮（入口本來就拒 age_s > max_age）。
2 抓取計畫結果到達時，承接條件改為：最新合格觀測的資料年齡 self.t − t_src ≤ seg.max_age（0.5 s，原觀測條件）且與計畫目標差 ≤ 2 mm／支撐高度差 ≤ 2 mm。
  不新鮮 ⇒ 不採用，結果保留為 ready（不重算），每拍等到新的合格觀測（年齡 ≤ 0.5 s 且一致）才承接；一直等不到 ⇒ 最晚處置點 grasp_plan_late。
  不一致 ⇒ 作廢重送（同第一版）。2 s 不再出現在任何條件裡。
3 承接紀錄加：承接當下的來源年齡、計畫所依快照的末幀年齡（snap_t_last）。
"""
import numpy as np

import bottle_s0_ctrl as C0
import bottle_w1_online_ctrl as ON

C3, C2 = ON.C3, ON.C2


class W1PlaceController(ON.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE2'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cand = None

    def candidate_intake(self, t_recv, c):
        age = c.get('age_s')
        self._cand = dict(t_recv=float(t_recv), age_s=None if age is None else float(age), snap_t_last=c.get('snap_t_last'))
        return super().candidate_intake(t_recv, c)

    def _seg_freeze(self, xy, zs, seq):
        out = super()._seg_freeze(xy, zs, seq)
        k = self._cand or {}
        if self.obs_last is not None and k.get('age_s') is not None and np.isfinite(k['age_s']):
            self.obs_last['t_src'] = k['t_recv'] - k['age_s']                     # 來源擷取時刻（任務時間）
            self.obs_last['snap_t_last'] = k.get('snap_t_last')
        if self.obs_first is not None and 't_src' not in self.obs_first and self.obs_last is not None and 't_src' in self.obs_last:
            self.obs_first['t_src'] = self.obs_last['t_src']
        return out

    def _obs_ok(self, g):
        o, tol = self.obs_last, self.seg['dev_tol_mm'] * 1e-3
        if o is None or 't_src' not in o:
            return 'obs_missing'
        if self.t - o['t_src'] > self.seg['max_age']:
            return 'obs_stale'
        if float(np.linalg.norm(o['xy'] - g['xy'])) > tol or abs(o['zs'] - g['zs']) > tol:
            return 'obs_inconsistent'
        return None

    def _gp_poll(self, q):
        g = self.gp
        if g is None or g['state'] not in ('pending', 'ready'):
            return
        if g['state'] == 'pending':
            m = self.pc.poll(g['id'])
            if m is None:
                if not self.pc.alive():
                    self.judge._stop(self.t, 'planner_dead', kind='grasp')
                elif q[0] >= g['x_latest']:
                    self.judge._stop(self.t, 'grasp_plan_late', x=float(q[0]), x_latest=g['x_latest'], waited_s=float(self.t - g['t_req']))
                return
            g.update(t_res=float(self.t), wall_recv=m['t_wall_recv'], compute_wall=m['wall']['t1'] - m['wall']['t0'], ok=m['ok'], error=m['error'], res=m['result'])
            if not m['ok']:
                g['state'] = 'infeasible'
                self.judge._stop(self.t, 'segtarget_unreachable', source='online_grasp_plan', error=m['error'], seq=g['seq'])
                return
            T_g = np.array(m['result']['T_g'], float)
            if float(np.abs(T_g - C0.grasp_target(float(g['xy'][0]), float(g['xy'][1]), g['zs'] + 0.10)).max()) > 1e-9:
                g['state'] = 'rejected:target_mismatch'
                self.judge._stop(self.t, 'grasp_plan_target_mismatch', id=g['id'])
                return
            g['state'] = 'ready'
        if q[0] >= g['x_latest']:
            g['state'] = 'rejected:late'
            self.judge._stop(self.t, 'grasp_plan_late', x=float(q[0]), x_latest=g['x_latest'], waited_s=float(self.t - g['t_req']), had_result=True)
            return
        why = self._obs_ok(g)
        if why == 'obs_inconsistent':
            g['state'] = 'rejected:' + why
            self._ev('grasp_plan_reject', id=g['id'], why=why)
            o = self.obs_last
            self.gp = None
            self._gp_submit(o['xy'], o['zs'], o['seq'])
            return
        if why is not None:                                                    # 不新鮮：不採用，結果保留，等新的合格觀測
            if not g.get('wait_logged'):
                g['wait_logged'] = True
                g['t_wait_obs'] = float(self.t)
                self._ev('grasp_plan_wait_obs', id=g['id'], why=why)
            return
        o = self.obs_last
        g.update(obs_age_at_accept=float(self.t - o['t_src']), snap_t_last=o.get('snap_t_last'),
                 snap_age_at_accept=None if o.get('snap_t_last') is None else float(self.t - float(o['snap_t_last'])))
        res = g.pop('res')
        self._gp_apply(g, res, q)

    def report(self):
        for g in self.gp_hist:
            g.pop('res', None)
        r = super().report()
        r['online']['version'] = 'online2：承接以來源擷取戳計觀測年齡 ≤ 0.5 s（seg.max_age）；不新鮮則保留結果等新觀測'
        return r
