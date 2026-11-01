#!/usr/bin/env python3
"""三維定位採集六個佈置的結果（只讀紀錄；Codex 20261101_220915）：回答「不輸入實際桌高時，哪些高度／視角能產生完整有效的三維目標，哪些必須換視角或拒絕」。
來源：各趟 seg3d_{ns}.npz（旁路節點即時輸出；欄位名取自訊息自帶的 layout 標籤）、{ns}_log.npz（擷取戳 ↔ 任務時刻、評估用真值 tb）、collect_report.json（佈置、時程、中止）。
階段：approach（底盤照紀錄接近）、look（固定俯視保持段）、transit（展臂／回收過渡）。
每個佈置 × 相機 × 階段報：輸出數、有效實例數、紀錄可見旗標（含遮擋）為真的幀數與其中有效數；xy_valid 比例與中心誤差（已知半徑／輪廓）；
  top_observed 比例與瓶頂誤差；support_observed 比例與支撐高度誤差（真值＝瓶底 z）；height_valid 比例與高度誤差；
  時序：proc_ms、age_ms（模擬時鐘）、pair_age_ms、self_dt_ms；丟幀（配對、最新槽、舊格晚配齊、拒舊）以最後一筆計數。
這是同一模型、同後處理（seg3d_v1f 版）、凍結的六趟；不重訓、不依結果調門檻。真值只用於評估。
    python3 evaluation/bottle_seg3d_collect_report.py [RUN ...]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ('seg3d_col_z045_p1', 'seg3d_col_z045_p2', 'seg3d_col_z060_p1c', 'seg3d_col_z060_p2', 'seg3d_col_z075_p1', 'seg3d_col_z075_p2')
H_BOTTLE = 0.20


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else dict(n=int(len(x)), median=round(float(np.median(x)), 2), p95=round(float(np.percentile(x, 95)), 2), max=round(float(x.max()), 2))


def one(run):
    D = os.path.join(HERE, 'runs', run)
    rep = json.load(open(os.path.join(D, 'collect_report.json')))
    t_obs = rep['t_obs']
    sched = {nm: (s0, s1) for nm, s0, s1 in rep['sched']}
    out = dict(run=run, layout=rep['layout'], abort=rep['abort'], min_clear_m=rep['min_clear_m'], bottle_move_mm=rep['bottle_move_m'] * 1e3, cams={})
    for ns in ('v1', 'w1'):
        Z = np.load(os.path.join(D, f'seg3d_{ns}.npz'))
        label = str(Z['label'])
        fields = ['recv_t'] + (label.split('fields=')[1].split(',') if 'fields=' in label else [])
        R = [dict(zip(fields, r)) for r in Z['rows']] if Z['rows'].shape[1] == len(fields) else []
        L = np.load(os.path.join(D, f'{ns}_log.npz'))
        lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
        by = {round(float(s), 4): k for k, s in enumerate(lc['stamp'])}
        groups = {}
        for r in R:
            k = by.get(round(r['stamp'], 4))
            if k is None:
                continue
            tt = float(lc['task_t'][k])
            ph = 'approach' if tt < t_obs else ('look' if sched['hold'][0] <= tt < sched['hold'][1] else 'transit')
            r.update(task_t=tt, expected=bool(lc['expected'][k] > 0.5), truth=[float(lc['tb_x'][k]), float(lc['tb_y'][k]), float(lc['tb_z'][k])])
            groups.setdefault(ph, []).append(r)
        cam = {}
        for ph, G_ in groups.items():
            V = [r for r in G_ if r['valid'] > 0.5]
            xy = lambda r, kx, ky: math.hypot(r[kx] - r['truth'][0], r[ky] - r['truth'][1]) * 1e3   # noqa: E731
            ok_obj = [r for r in V if xy(r, 'cx_known_r', 'cy_known_r') <= 50.0]
            cam[ph] = dict(
                n_out=len(G_), n_valid=len(V), expected=sum(r['expected'] for r in G_), valid_when_expected=sum(1 for r in G_ if r['expected'] and r['valid'] > 0.5),
                wrong_object=len(V) - len(ok_obj),
                xy_valid=sum(1 for r in ok_obj if r['xy_valid'] > 0.5), xy_err_known_r=stats([xy(r, 'cx_known_r', 'cy_known_r') for r in ok_obj if r['xy_valid'] > 0.5]),
                xy_valid_sil=sum(1 for r in ok_obj if r['xy_valid_silhouette'] > 0.5), xy_err_sil=stats([xy(r, 'cx_sil', 'cy_sil') for r in ok_obj if r['xy_valid_silhouette'] > 0.5]),
                top_observed=sum(1 for r in ok_obj if r['top_observed'] > 0.5), top_err=stats([abs(r['z_top'] - (r['truth'][2] + H_BOTTLE)) * 1e3 for r in ok_obj if r['top_observed'] > 0.5]),
                support_observed=sum(1 for r in ok_obj if r['support_observed'] > 0.5), support_err=stats([abs(r['z_support'] - r['truth'][2]) * 1e3 for r in ok_obj if r['support_observed'] > 0.5]),
                height_valid=sum(1 for r in ok_obj if r['height_valid'] > 0.5), height_err=stats([abs(r['height'] - H_BOTTLE) * 1e3 for r in ok_obj if r['height_valid'] > 0.5]),
                z_vis_lo_minus_truth=stats([(r['z_vis_lo'] - r['truth'][2]) * 1e3 for r in ok_obj]))
        last = R[-1] if R else {}
        cam['timing'] = dict(proc_ms=stats([r['proc_ms'] for r in R]), age_ms=stats([r['age_ms'] for r in R]), pair_age_ms=stats([r.get('pair_age_ms', float('nan')) for r in R]),
                             self_dt_ms=stats([r.get('self_dt_ms', float('nan')) for r in R]), self_filter_frac=float(np.mean([r['self_filter'] for r in R])) if R else None,
                             n_drop_pair=last.get('n_drop_pair'), n_drop_latest=last.get('n_drop_latest'), n_old_complete=last.get('n_old_complete'), n_stale=last.get('n_stale'), n_out=len(R),
                             monotonic=all(b['stamp'] > a_['stamp'] for a_, b in zip(R, R[1:])), label=label[:60])
        out['cams'][ns] = cam
    return out


def main():
    runs = sys.argv[1:] or RUNS
    res = {r: one(r) for r in runs if os.path.exists(os.path.join(HERE, 'runs', r, 'collect_report.json'))}
    json.dump(res, open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for run, o in res.items():
        print(f"== {run} 桌高 {o['layout']['table_z']} 瓶 {o['layout']['bottle']}；中止 {o['abort']}；最小淨空 {o['min_clear_m']:.3f} m；瓶位移 {o['bottle_move_mm']:.1f} mm")
        for ns in ('v1', 'w1'):
            c = o['cams'][ns]
            for ph in ('approach', 'look'):
                g = c.get(ph)
                if g is None:
                    continue
                print(f"   {ns} {ph:8s} 輸出 {g['n_out']} 有效 {g['n_valid']}（可見旗標 {g['expected']}／其中有效 {g['valid_when_expected']}）錯物體 {g['wrong_object']}｜xy_valid {g['xy_valid']} 誤差 {g['xy_err_known_r']}｜"
                      f"輪廓 {g['xy_valid_sil']} {g['xy_err_sil']}｜瓶頂 {g['top_observed']} {g['top_err']}｜支撐 {g['support_observed']} {g['support_err']}｜高度 {g['height_valid']} {g['height_err']}｜可見最低−瓶底 {g['z_vis_lo_minus_truth']}")
            t = c['timing']
            print(f"   {ns} 時序 proc {t['proc_ms']} age {t['age_ms']} pair_age {t['pair_age_ms']} self_dt {t['self_dt_ms']} 自身濾除 {t['self_filter_frac']}；丟 配對 {t['n_drop_pair']} 槽 {t['n_drop_latest']} 晚配齊 {t['n_old_complete']} 拒舊 {t['n_stale']}；單調 {t['monotonic']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
