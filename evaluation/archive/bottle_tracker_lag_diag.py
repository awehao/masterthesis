#!/usr/bin/env python3
"""第一案動靜判定延遲的來源分解（Codex 20261101_142349 (c1) 第 1 點；實際掃描回放、候選 2 追蹤器，不改參數）。
對每幀「移動中、2.5 m 內、≥2 點可見」且移動物未被發布為動態的幀，取真值圓心 半徑＋0.30 m 內的軌跡，並列：
  真值圓心速度；量測（該軌跡當幀關聯叢集質心）2 s 視窗淨速度；KF 位置（hist）2 s 淨速度；KF 瞬時速度；視窗內起步前（t < t_start）歷史比例；
  可見弧質心相對真值圓心的偏移（質心不是圓心：相對方位改變時，質心速度 ≠ 圓心速度）。
    python3 evaluation/bottle_tracker_lag_diag.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_tracker_scanlog_replay as TR              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402

CA = DC.CASES['w1_dyn_pass1']
WIN = float(NC.TRACKER.get('static_window_s', 2.0))
MEAS = {}


class LogKF(KC.make2(NC.ScanObstacleTracker)):
    """記錄每條軌跡每次 KF 更新的量測（叢集質心）與時刻；追蹤行為與候選 2 相同。"""

    def _update_tracks(self, clusters, t_ns):
        before = {tr.tid: tr.kf.step for tr in self._tracks}
        for tr in self._tracks:
            f0 = tr.kf.step

            def rec(t_ns_, y_xy, _f=f0, _tid=tr.tid):
                MEAS.setdefault(_tid, []).append((t_ns_ * 1e-9, float(y_xy[0]), float(y_xy[1])))
                return _f(t_ns=t_ns_, y_xy=y_xy)
            tr.kf.step = lambda t_ns=None, y_xy=None, _r=rec: _r(t_ns, y_xy)
        super()._update_tracks(clusters, t_ns)
        for tr in self._tracks:
            if tr.tid in before:
                tr.kf.step = before[tr.tid]
            elif not hasattr(tr, '_born_logged'):
                MEAS.setdefault(tr.tid, []).append((t_ns * 1e-9, float(tr.kf.position[0]), float(tr.kf.position[1])))
                tr._born_logged = True


def net_speed(seq, t, win):
    S = [s for s in seq if t - win - 1e-9 <= s[0] <= t + 1e-9]
    if len(S) < 2 or S[-1][0] - S[0][0] < 0.3:
        return None, 0.0
    return float(np.hypot(S[-1][1] - S[0][1], S[-1][2] - S[0][2]) / (S[-1][0] - S[0][0])), float(sum(1 for s in S if s[0] < CA['t_start']) / len(S))


def main():
    out = {}
    for run in ('bottle_w1_dyn1sr_np1', 'bottle_w1_dyn1sr_g1', 'bottle_w1_dyn1sr_trk1', 'bottle_w1_dyn1sr_trknp1'):
        MEAS.clear()
        rows = TR.replay(run, LogKF)
        hist = {}
        recs = []
        for r in rows:
            T = r['truth']
            for k in r['tracks']:
                hist.setdefault(k['tid'], []).append((r['t'], k['x'], k['y']))
            if T is None:
                continue
            d_surf = float(np.hypot(r['pose'][0] - T['x'], r['pose'][1] - T['y'])) - CA['radius']
            if float(np.hypot(T['vx'], T['vy'])) < 0.2 or d_surf > 2.5:
                continue
            if any(np.hypot(d['x'] - T['x'], d['y'] - T['y']) <= CA['radius'] + 0.30 for d in r['dyn']):
                continue
            trk = [k for k in r['tracks'] if np.hypot(k['x'] - T['x'], k['y'] - T['y']) <= CA['radius'] + 0.30]
            for k in trk:
                m_ns, m_pre = net_speed(MEAS.get(k['tid'], []), r['t'], WIN)
                k_ns, k_pre = net_speed(hist.get(k['tid'], []), r['t'], WIN)
                last = [m for m in MEAS.get(k['tid'], []) if m[0] <= r['t'] + 1e-9]
                off = float(np.hypot(last[-1][1] - T['x'], last[-1][2] - T['y'])) if last else None
                recs.append(dict(t=round(r['t'], 2), tid=k['tid'], age=k['age'], reject=k['reject'], v_true=round(float(np.hypot(T['vx'], T['vy'])), 3),
                                 v_kf=round(float(np.hypot(k['vx'], k['vy'])), 3), net_meas=None if m_ns is None else round(m_ns, 3), net_kf=None if k_ns is None else round(k_ns, 3),
                                 pre_frac_kf=round(k_pre, 2), centroid_off=None if off is None else round(off, 3)))
        def med(key):
            v = [x[key] for x in recs if x[key] is not None]
            return None if not v else float(np.median(v))
        o = dict(n=len(recs), v_true=med('v_true'), v_kf=med('v_kf'), net_meas=med('net_meas'), net_kf=med('net_kf'), pre_frac=med('pre_frac_kf'), centroid_off=med('centroid_off'),
                 n_pre_gt0=sum(1 for x in recs if x['pre_frac_kf'] > 0), n_meas_ge_thr=sum(1 for x in recs if (x['net_meas'] or 0) >= NC.TRACKER['min_net_speed']),
                 n_kfv_ge_thr=sum(1 for x in recs if x['v_kf'] >= NC.TRACKER['min_track_speed']), sample=recs[:3] + recs[len(recs) // 2:len(recs) // 2 + 2] + recs[-2:])
        out[run] = o
        print(f"[{run}] 未發布幀 {o['n']}：中位數 真值 {o['v_true']}、KF 瞬時 {o['v_kf']}、量測淨速 {o['net_meas']}、KF 位置淨速 {o['net_kf']} m/s；"
              f"視窗含起步前歷史的幀 {o['n_pre_gt0']}（中位比例 {o['pre_frac']}）；量測淨速 ≥ 0.25 的幀 {o['n_meas_ge_thr']}；KF 瞬時 ≥ 0.20 的幀 {o['n_kfv_ge_thr']}；質心離圓心中位 {o['centroid_off']} m")
        for x in o['sample']:
            print('     ', x)
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'tracker_lag_diag_case1.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
