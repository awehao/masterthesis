#!/usr/bin/env python3
"""第八次第一包：一趟 S → 門前觀察區的評估（只讀紀錄；真值只供評估）。
時間一律是模擬時間；bag 中無 stamp 的話題以 /odom（接收時刻, stamp）插值換算。
內容：
  時程：計畫、巡航、對位、到達時刻與誤差；路徑長、重規劃。
  速度：實測底盤速度（odom 差分）平均／最大；巡航中 < 0.02 m/s 的累計時間與區段。
  距離（真值）：底盤圓盤（半徑 0.39）對行人表面、未知箱體、已知牆與家具、關閉門扇的最近距離與時刻。
  會遇：每個行人中心距機器人中心的最近距離、時刻、當時機器人速度與 CBF 狀態；距離 < 2.0 m 的時段（描述，不是事前判準）。
  CBF：/gmpc/diag_v2 的 cbf_active、eps0、min_h、accept_action。
  Shield：/cmd_vel_pre_shield 對 /cmd_vel 逐筆配對（取 /cmd_vel 之前最近一筆），差 > 1e-6 的筆數、最大差與時段。
  事件鏈：巡航中各行人 0.5 m 內有確認航跡的比例；未知箱體在 2.2 m 內時靜態 CBF 話題是否有點落在箱體 0.10 m 內。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_nav_report.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import mission_room_obstacles as MO  # noqa: E402

R_BASE = 0.39


def num(v, d=3):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def segments(mask, t):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if (not m or i == len(mask) - 1) and s is not None:
            e = i if m else i - 1
            out.append((num(t[s], 2), num(t[e], 2), num(t[e] - t[s], 2)))
            s = None
    return out


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    nav = json.load(open(os.path.join(D, 'm8nav.json')))
    tr, static, walkers = MB.load_truth(D)
    B = MB.read_bag(os.path.join(D, 'bag'))
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(t_rec):
        return float(np.interp(t_rec, rec, od[:, 0]))
    seg = {s['seg']: s['t'] for s in nav['segments']}
    t0, t1 = seg.get('CRUISE'), seg.get('ZERO')
    out = dict(run=run, result=nav['result'], arrive=nav.get('arrive'), t_plan=seg.get('PLAN'), t_cruise=t0, t_align=seg.get('ALIGN'), t_done=t1,
               plan=[{k: p.get(k) for k in ('id', 'kind', 't', 'ok', 'valid', 'L', 'wall_ms', 'cost_max', 'grid_age')} for p in nav['plans']], n_replan=len(nav['replans']))
    # 速度
    k = (od[:, 0] >= (t0 or 0)) & (od[:, 0] <= (t1 or od[-1, 0]))
    T, X, Y = od[k, 0], od[k, 1], od[k, 2]
    idx = np.searchsorted(T, np.arange(T[0], T[-1], 0.1))
    v = np.hypot(np.diff(X[idx]), np.diff(Y[idx])) / np.diff(T[idx])
    tv = T[idx][1:]
    kc = tv <= (seg.get('ALIGN') or tv[-1])
    out['speed'] = dict(mean_cruise=num(v[kc].mean()), max=num(v.max()), path_len_m=num(np.sum(np.hypot(np.diff(X[idx]), np.diff(Y[idx])))),
                        slow_lt_0p02_s=num(0.1 * int((v[kc] < 0.02).sum()), 2), slow_segments=[s_ for s_ in segments(v[kc] < 0.02, tv[kc]) if s_[2] >= 0.2])
    # 真值距離
    kt = (tr['t'] >= (t0 or 0)) & (tr['t'] <= (t1 or tr['t'][-1]))
    P = np.c_[tr['base_x'][kt], tr['base_y'][kt]]
    tt = tr['t'][kt]
    clr = {}
    for w in walkers:
        d = np.hypot(P[:, 0] - tr[f"{w['name']}_x"][kt], P[:, 1] - tr[f"{w['name']}_y"][kt])
        i = int(np.argmin(d))
        clr[w['name']] = dict(center_min=num(d[i]), disc_to_surface_min=num(d[i] - w['radius'] - R_BASE), t=num(tt[i], 2))
    for u in static:
        d = MB.poly_dist(P, MB.box_poly(u))
        i = int(np.argmin(d))
        clr[u['name']] = dict(disc_min=num(d[i] - R_BASE), t=num(tt[i], 2))
    kb = []
    for r in MO.known_boxes():
        kb.append(MB.poly_dist(P, np.array([(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])])))
    kb.append(MB.poly_dist(P, np.array([(0.0, 0.0), (0.95, 0.0), (0.95, 0.04), (0.0, 0.04)])))
    dk = np.min(kb, axis=0)
    i = int(np.argmin(dk))
    clr['known_walls_furniture_door'] = dict(disc_min=num(dk[i] - R_BASE), t=num(tt[i], 2))
    out['clearance_truth'] = clr
    # CBF
    dg = []
    for t_rec, m in B.get('/gmpc/diag_v2', []):
        try:
            r = json.loads(m.data)
        except ValueError:
            continue
        dg.append((float(r.get('t') or to_sim(t_rec)), r.get('cbf_active'), r.get('eps0'), r.get('min_h'), r.get('accept_action')))
    dg = [x for x in dg if (t0 or 0) <= x[0] <= (t1 or 1e9)]
    act = np.array([x[1] if x[1] is not None else np.nan for x in dg], float)
    eps = np.array([x[2] if x[2] is not None else np.nan for x in dg], float)
    mh = np.array([x[3] if x[3] is not None else np.nan for x in dg], float)
    acc = {}
    for x in dg:
        acc[str(x[4])] = acc.get(str(x[4]), 0) + 1
    tdg = np.array([x[0] for x in dg])
    out['cbf'] = dict(n=len(dg), frac_active=num(np.nanmean(act > 0)), active_max=num(np.nanmax(act)), eps0_max=num(np.nanmax(eps), 5), eps0_p95=num(np.nanpercentile(eps, 95), 5),
                      n_eps0_gt_1e3=int(np.nansum(eps > 1e-3)), min_h_min=num(np.nanmin(mh), 4), accept_action=acc,
                      active_segments=[s_ for s_ in segments(act > 0, tdg) if s_[2] >= 0.2])
    # 會遇
    enc = {}
    for w in walkers:
        d = np.hypot(P[:, 0] - tr[f"{w['name']}_x"][kt], P[:, 1] - tr[f"{w['name']}_y"][kt])
        i = int(np.argmin(d))
        ti = tt[i]
        j = int(np.argmin(np.abs(tdg - ti))) if len(tdg) else None
        vi = float(np.interp(ti, tv, v)) if len(v) else None
        enc[w['name']] = dict(center_min=num(d[i]), t=num(ti, 2), robot_speed=num(vi), cbf_active_at=None if j is None else act[j], min_h_at=None if j is None else num(mh[j], 4),
                              lt_2m=segments(d < 2.0, tt))
    out['encounters'] = enc
    # Shield
    pre = [(to_sim(x[0]), (x[1].linear.x, x[1].linear.y, x[1].angular.z)) for x in B.get('/cmd_vel_pre_shield', [])]
    post = [(to_sim(x[0]), (x[1].linear.x, x[1].linear.y, x[1].angular.z)) for x in B.get('/cmd_vel', [])]
    tp = np.array([x[0] for x in pre])
    diffs = []
    for tq, u in post:
        if not ((t0 or 0) <= tq <= (t1 or 1e9)) or not len(tp):
            continue
        j = int(np.searchsorted(tp, tq, side='right')) - 1
        if j < 0:
            continue
        diffs.append((tq, max(abs(a - b) for a, b in zip(u, pre[j][1]))))
    dd = np.array([x[1] for x in diffs]) if diffs else np.zeros(0)
    out['shield'] = dict(n=len(diffs), n_modified=int((dd > 1e-6).sum()), max_diff=num(dd.max() if len(dd) else np.nan, 4),
                         modified_segments=[s_ for s_ in segments(dd > 1e-6, np.array([x[0] for x in diffs])) if s_[2] >= 0.1] if len(dd) else [],
                         note='/cmd_vel 之前最近一筆 /cmd_vel_pre_shield 配對；配對時間差會帶來少量假差異')
    # 事件鏈：行人確認航跡覆蓋、靜態 CBF 對近距箱體
    cov = {w['name']: [0, 0] for w in walkers}
    for t_rec, m in B.get('/gmpc/tracks_debug', []):
        ts = to_sim(t_rec)
        if not ((t0 or 0) <= ts <= (t1 or 1e9)):
            continue
        Tk = MB.tracks(m)
        pr = MB.pose_at(od, ts)
        for w in walkers:
            c = MB.walker_at(tr, w['name'], ts)
            if math.hypot(c[0] - pr[1], c[1] - pr[2]) > 3.0:
                continue
            cov[w['name']][0] += 1
            if any(x['confirmed'] and math.hypot(x['x'] - c[0], x['y'] - c[1]) <= 0.5 for x in Tk):
                cov[w['name']][1] += 1
    st = {u['name']: [0, 0] for u in static}
    for t_rec, m in B.get('/gmpc/static_obstacles', []):
        ts = to_sim(t_rec)
        if not ((t0 or 0) <= ts <= (t1 or 1e9)):
            continue
        d_ = np.asarray(m.data, float)
        Pp = d_.reshape(-1, 5)[:, :2] if len(d_) and len(d_) % 5 == 0 else np.zeros((0, 2))
        pr = MB.pose_at(od, ts)
        for u in static:
            V = MB.box_poly(u)
            if MB.poly_dist(np.array([[pr[1], pr[2]]]), V)[0] > 2.2:
                continue
            st[u['name']][0] += 1
            if len(Pp) and (MB.poly_dist(Pp, V) <= 0.10).any():
                st[u['name']][1] += 1
    out['chain'] = dict(walker_confirmed_within_3m={k: dict(n=v[0], frac=num(v[1] / v[0]) if v[0] else None) for k, v in cov.items()},
                        static_cbf_box_within_2p2m={k: dict(n=v[0], frac=num(v[1] / v[0]) if v[0] else None) for k, v in st.items()})
    out['note'] = 'n＝1、seed1 r2 開發場景；有行人在場不等於有效會遇；只驗到 A 房'
    json.dump(out, open(os.path.join(D, 'm8_nav_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
