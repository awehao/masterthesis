#!/usr/bin/env python3
"""候選 2 追蹤器（KF 時鐘＋預測位置馬氏關聯）實跑前的輕量回放核對（Codex 20261101_140323）：假動態減少不是因為障礙消失或大量拒收。非物理、同程序。
1 真移動物：每幀「可見」＝掃描點落在讀回真值圓（半徑＋0.03 m）上 ≥ 2 點（同 min_cluster_pts）且機器人到表面 ≤ surface_range 2.5 m（追蹤器只發布此範圍內表面點）且讀回速度 ≥ min_track_speed 0.2 m/s；「覆蓋」＝有發布的動態項在真值圓心 半徑＋0.30 m 內。
  起步後首次發布延遲（對案例 t_start）、首次發布後覆蓋率、最長漏失（連續可見未覆蓋，秒）；原版 vs 候選 2。
2 未知靜態物仍有位置保護：未入圖方塊 OBST（−2,0；0.4×0.4）與 w1c_1 的第二桌腳（漏地圖），可見且在 2.0 m 內的幀中，靜態障礙清單（或動態清單）有點在其表面 0.10 m 內的比例。
3 拒收：全程軌跡數、發布為動態的軌跡數、各拒收碼分布。
    python3 evaluation/test_bottle_tracker_kfcm.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_tracker_scanlog_replay as TR              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import test_bottle_place as TP                          # noqa: E402

ORIG = NC.ScanObstacleTracker
SURF_RANGE = float(NC.TRACKER['surface_range'])
MOVE_MIN = float(NC.TRACKER['min_track_speed'])
CAND = KC.make2(ORIG)
BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def box_d(p, c, sz):
    dx, dy = np.maximum(np.abs(p[:, 0] - c[0]) - sz[0] / 2, 0), np.maximum(np.abs(p[:, 1] - c[1]) - sz[1] / 2, 0)
    return np.hypot(dx, dy)


def hits(pose, rr):
    a = NC.scan_angles() + pose[2]
    ok = np.isfinite(rr) & (rr < NC.RANGE_MAX)
    return np.c_[pose[0] + rr[ok] * np.cos(a[ok]), pose[1] + rr[ok] * np.sin(a[ok])]


def recon(run, case):
    """dyn2sr_g1（無掃描紀錄）：重建掃描並保留 ranges 以便同一套可見度判定。"""
    CA = DC.CASES[case]
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    bx = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    ch = NC.NavChain(map_boxes=TR.map_boxes())
    rows, nxt = [], 0.0
    for i, t in enumerate(c['t']):
        if t + 1e-9 < nxt:
            continue
        nxt = t + 0.1
        q = (float(c['base_x'][i]), float(c['base_y'][i]), float(c['base_yaw'][i]))
        r = NC.ray_boxes(np.array(q[:2]), q[2], bx, G.LIDAR_Z)
        if np.isfinite(c['dyn_x'][i]):
            r = np.minimum(r, DC.ray_cyl(np.array(q[:2]), NC.scan_angles() + q[2], np.array([c['dyn_x'][i], c['dyn_y'][i]]), CA['radius'], NC.RANGE_MAX))
        ch.on_scan(float(t), r, q)
        truth = dict(x=float(c['dyn_x'][i]), y=float(c['dyn_y'][i]), vx=float(c['dyn_vx'][i]), vy=float(c['dyn_vy'][i])) if np.isfinite(c['dyn_x'][i]) else None
        rows.append(dict(t=float(t), pose=list(q), dyn=[dict(d) for d in ch.dyn], stat=[dict(d) for d in ch.stat], truth=truth, ranges=r,
                         tracks=[dict(tid=int(k.tid), mover=bool(k.is_mover), reject=int(k.reject), age=int(k.age)) for k in ch.tr._tracks]))
    return rows


def mover_metrics(rows, CA):
    vis, cov, ts = [], [], []
    for r in rows:
        T = r['truth']
        if T is None or r['t'] < CA['t_start'] - 1e-9:
            continue
        P = hits(r['pose'], np.asarray(r['ranges'], float))
        v = int((np.abs(np.hypot(P[:, 0] - T['x'], P[:, 1] - T['y']) - CA['radius']) <= 0.03).sum()) >= 2 if len(P) else False
        v = v and float(np.hypot(r['pose'][0] - T['x'], r['pose'][1] - T['y'])) - CA['radius'] <= SURF_RANGE   # 追蹤器只發布 surf_range 內的表面點（設計）
        v = v and float(np.hypot(T['vx'], T['vy'])) >= MOVE_MIN                                          # 只計真的在動的幀（到終點停住後不該當動態）
        c_ = any(np.hypot(d['x'] - T['x'], d['y'] - T['y']) <= CA['radius'] + 0.30 for d in r['dyn'])
        vis.append(v)
        cov.append(c_)
        ts.append(r['t'])
    vis, cov, ts = np.array(vis), np.array(cov), np.array(ts)
    first = ts[cov][0] if cov.any() else None
    m = vis & (ts >= (first if first is not None else 1e9))
    run_, best = 0, 0
    for v, c_ in zip(vis[ts >= (first or 1e9)], cov[ts >= (first or 1e9)]):
        run_ = run_ + 1 if (v and not c_) else 0
        best = max(best, run_)
    first_vis = ts[vis][0] if vis.any() else None
    return dict(n_visible=int(vis.sum()), latency_from_start=None if first is None else float(first - CA['t_start']), latency_from_visible=None if first is None or first_vis is None else float(first - first_vis),
                coverage_after_first=float(cov[m].mean()) if m.any() else None, longest_gap_s=best * 0.1)


def static_protect(rows, objs):
    """objs：[(名, 中心, 尺寸)]；可見（≥2 點落在表面 0.03 m 內）且機器人到表面 ≤ 2.0 m 的幀中，靜態或動態清單有點在表面 0.10 m 內的比例。"""
    out = {}
    for nm, c, sz in objs:
        n_v = n_p = 0
        for r in rows:
            P = hits(r['pose'], np.asarray(r['ranges'], float))
            if not len(P) or int((box_d(P, c, sz) <= 0.03).sum()) < 2:
                continue
            if float(box_d(np.array([r['pose'][:2]]), c, sz)[0]) > 2.0:
                continue
            n_v += 1
            Q = np.array([[d['x'], d['y']] for d in r['stat'] + r['dyn']]) if (r['stat'] or r['dyn']) else np.zeros((0, 2))
            n_p += int(len(Q) > 0 and float(box_d(Q, c, sz).min()) <= 0.10)
        out[nm] = dict(n_visible=n_v, protected=None if n_v == 0 else n_p / n_v)
    return out


def rejects(rows):
    seen = {}
    for r in rows:
        for k in r['tracks']:
            s = seen.setdefault(k['tid'], dict(mover=False, rej=set()))
            s['mover'] |= k['mover']
            s['rej'].add(k['reject'])
    codes = {}
    for s in seen.values():
        for c_ in s['rej']:
            codes[c_] = codes.get(c_, 0) + 1
    return dict(n_tracks=len(seen), n_mover_tracks=sum(1 for s in seen.values() if s['mover']), reject_codes={int(k): v for k, v in sorted(codes.items())})


def w1c1_rows(cls):
    """bottle_w1c_1：漏第二桌地圖（只放第一桌腳）、離線射線（同 bottle_tracker_replay）。"""
    Z = np.load(os.path.join(HERE, 'runs', 'bottle_w1c_1', 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    legs1 = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    legs2 = [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    scene = legs1 + [(SC.OBST['c'], SC.OBST['sz'])] + legs2
    NC.ScanObstacleTracker = cls
    try:
        ch = NC.NavChain(map_boxes=legs1)
    finally:
        NC.ScanObstacleTracker = ORIG
    rows, nxt = [], 0.0
    for i, t in enumerate(c['t']):
        if t + 1e-9 < nxt:
            continue
        nxt = t + 0.1
        q = (float(c['base_x'][i]), float(c['base_y'][i]), float(c['base_yaw'][i]))
        r = NC.ray_boxes(np.array(q[:2]), q[2], scene, G.LIDAR_Z)
        ch.on_scan(float(t), r, q)
        rows.append(dict(t=float(t), pose=list(q), dyn=[dict(d) for d in ch.dyn], stat=[dict(d) for d in ch.stat], ranges=r,
                         tracks=[dict(tid=int(k.tid), mover=bool(k.is_mover), reject=int(k.reject), age=int(k.age)) for k in ch.tr._tracks]))
    return rows, [(f'table2_{n}', c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']


def main():
    res = {}
    OBST = [('OBST', SC.OBST['c'][:2], SC.OBST['sz'][:2])]
    for run, case, rec in (('bottle_w1_dyn1sr_np1', 'w1_dyn_pass1', True), ('bottle_w1_dyn1sr_g1', 'w1_dyn_pass1', True), ('bottle_w1_dyn2sr_np1', 'w1_dyn_overtake1', True),
                           ('bottle_w1_dyn2sr_g1', 'w1_dyn_overtake1', False)):
        CA = DC.CASES[case]
        o = {}
        for nm, cls in (('orig', ORIG), ('cand2', CAND)):
            if rec:
                rows = TR.replay(run, cls)
            else:
                NC.ScanObstacleTracker = cls
                try:
                    rows = recon(run, case)
                finally:
                    NC.ScanObstacleTracker = ORIG
            o[nm] = dict(mover=mover_metrics(rows, CA), static=static_protect(rows, OBST), rejects=rejects(rows))
        res[run] = o
        a, b = o['orig'], o['cand2']
        print(f"   {run}：延遲（起步後／可見後）原 {a['mover']['latency_from_start']:.2f}/{a['mover']['latency_from_visible']:.2f}、候選 {b['mover']['latency_from_start']:.2f}/{b['mover']['latency_from_visible']:.2f} s；"
              f"覆蓋 原 {a['mover']['coverage_after_first']:.3f}／候選 {b['mover']['coverage_after_first']:.3f}；最長漏失 原 {a['mover']['longest_gap_s']:.1f}／候選 {b['mover']['longest_gap_s']:.1f} s；"
              f"OBST 保護 原 {a['static']['OBST']}／候選 {b['static']['OBST']}；軌跡 原 {a['rejects']}／候選 {b['rejects']}")
        chk(f'1 {run}：候選延遲不晚 0.2 s 以上、覆蓋不降 0.05 以上、最長漏失不增 0.2 s 以上', b['mover']['latency_from_start'] <= a['mover']['latency_from_start'] + 0.2 + 1e-9
            and b['mover']['coverage_after_first'] >= a['mover']['coverage_after_first'] - 0.05 and b['mover']['longest_gap_s'] <= a['mover']['longest_gap_s'] + 0.2 + 1e-9, (a['mover'], b['mover']))
        pa, pb = a['static']['OBST']['protected'], b['static']['OBST']['protected']
        chk(f'2 {run}：未入圖方塊仍有位置保護（候選比例不低於原版）', pa is None or (pb is not None and pb >= pa - 1e-9), (pa, pb))
    o = {}
    for nm, cls in (('orig', ORIG), ('cand2', CAND)):
        rows, objs = w1c1_rows(cls)
        o[nm] = dict(static=static_protect(rows, objs), rejects=rejects(rows), n_dyn_scans=sum(1 for r in rows if r['dyn']))
    res['bottle_w1c_1'] = o
    print(f"   bottle_w1c_1：第二桌腳保護 原 {o['orig']['static']}／候選 {o['cand2']['static']}；假動態掃描 原 {o['orig']['n_dyn_scans']}／候選 {o['cand2']['n_dyn_scans']}；軌跡 原 {o['orig']['rejects']}／候選 {o['cand2']['rejects']}")
    ok = all((v['protected'] is None) or (o['cand2']['static'][k]['protected'] is not None and o['cand2']['static'][k]['protected'] >= v['protected'] - 1e-9) for k, v in o['orig']['static'].items())
    print('   （bottle_w1c_1 全程離第二桌腳 > 2.0 m 靜態範圍：位置保護不適用，兩版都不發布靜態點；保護核對以各趟未入圖方塊 OBST 為準）' if all(v['n_visible'] == 0 for v in o['orig']['static'].values()) else '')
    chk('2 bottle_w1c_1：漏地圖的第二桌腳仍有位置保護（候選比例不低於原版；不在範圍內則不適用）', ok, o)
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'tracker_kfcm_coverage.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
