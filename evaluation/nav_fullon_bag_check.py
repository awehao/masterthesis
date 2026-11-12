#!/usr/bin/env python3
"""全開配置趟的紀錄核對（主 bag＋若有 bag_extra）。source ROS 後執行：python3 evaluation/nav_fullon_bag_check.py RUN_DIR OUT.json
逐筆核 GUARD 狀態（mode／action／faults／改動與數值差、輪速、輪加速度）、GUARD 目標與 Shield 輸出（/cmd_vel_smoothed）配對、
Shield 輸入輸出差（/cmd_vel_pre_shield 對 /cmd_vel_smoothed，依到達順序配對；配不到算未驗證）、重規劃（以 /clock 模擬時間）、min_h、定位誤差；
有 bag_extra 才有：/shield/diag、/gmpc/obstacles。沒有的欄位標 missing。"""
import json, os, sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

d, outp = sys.argv[1], sys.argv[2]
L, RW, W_MAX, A_MAX = 0.245, 0.05, 5.55, 125.0
W = np.array([[0, 1, L], [-1, 0, L], [0, -1, L], [1, 0, L]], float)


def read(uri, want):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=uri, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in rd.get_all_topics_and_types()}
    rd.set_filter(rosbag2_py.StorageFilter(topics=[w for w in want if w in types]))
    D = {w: [] for w in want}
    while rd.has_next():
        tp, raw, ts = rd.read_next()
        D[tp].append((ts * 1e-9, deserialize_message(raw, get_message(types[tp]))))
    return D, types


D, types = read(d + '/bag', ['/clock', '/plan', '/gmpc/min_h', '/cmd_vel_nav', '/cmd_vel_pre_shield', '/cmd_vel_smoothed', '/cmd_vel', '/wheel_guard/status', '/amcl_pose', '/model/omni_bot/pose', '/gmpc/heading', '/gmpc/diag_v2'])
tw = lambda m: np.array([m.linear.x, m.linear.y, m.angular.z])   # noqa: E731
ck = np.array([(t, m.clock.sec + m.clock.nanosec * 1e-9) for t, m in D['/clock']])
sim = lambda t: float(np.interp(t, ck[:, 0], ck[:, 1]))          # noqa: E731
res = dict(run=os.path.basename(d.rstrip('/')), counts={k: len(v) for k, v in D.items()})
# 重規劃（模擬時間）
tp = np.array([sim(t) for t, _ in D['/plan']])
res['replan'] = dict(n=len(tp), sim_period_median_s=round(float(np.median(np.diff(tp))), 3), sim_period_p95_s=round(float(np.percentile(np.diff(tp), 95)), 3), sim_span_s=round(float(tp[-1] - tp[0]), 1))
pl = [np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]) for _, m in D['/plan']]
sh = [float(np.linalg.norm(b[:min(len(b), 80), None, :] - a[None, :, :], axis=2).min(axis=1).max()) for a, b in zip(pl[:-1], pl[1:]) if len(a) and len(b)]
res['replan'].update(n_shift_gt_0p25m=int(sum(s > 0.25 for s in sh)), shift_max_m=round(max(sh), 2))
# GUARD 逐筆
st = [(t, json.loads(m.data)) for t, m in D['/wheel_guard/status']]
sm = np.array([tw(m) for _, m in D['/cmd_vel_smoothed']])
sm_set = {tuple(np.round(r, 12)) for r in sm}
g = dict(n=len(st), modes={}, actions={}, faults={}, wheel_speed_viol=0, wheel_acc_viol=0, acc_not_claimed=0, unpaired_nonzero_target=0, modified=0, modified_by={}, mod_max=0.0, lam_lt1=0, lam_min=1.0, dt_min=9, dt_max=0, omega_max=0.0)
prev = None
for t, s in st:
    o, tg = np.array(s['out'], float), np.array(s['target'], float)
    g['modes'][s['mode']] = g['modes'].get(s['mode'], 0) + 1
    g['actions'][s['action']] = g['actions'].get(s['action'], 0) + 1
    for f in s['faults']:
        g['faults'][f] = g['faults'].get(f, 0) + 1
    g['wheel_speed_viol'] += float(np.abs(W @ o).max()) > RW * W_MAX + 1e-6
    g['omega_max'] = max(g['omega_max'], float(s['omega_max']))
    if prev is not None and s['dt'] is not None:
        g['dt_min'], g['dt_max'] = min(g['dt_min'], s['dt']), max(g['dt_max'], s['dt'])
        if s['accel_guaranteed']:
            g['wheel_acc_viol'] += float(np.abs(W @ (o - prev)).max()) > RW * A_MAX * s['dt'] + 1e-6
        else:
            g['acc_not_claimed'] += 1
    stale = 'input_timeout' in s['faults']
    eff = np.zeros(3) if stale else tg
    if np.any(tg) and tuple(np.round(tg, 12)) not in sm_set:
        g['unpaired_nonzero_target'] += 1
    dd = float(np.abs(o - eff).max())
    if dd > 1e-9:
        g['modified'] += 1
        g['mod_max'] = max(g['mod_max'], dd)
        k = f"{s['action']}|{'input_timeout' if stale else 'fresh'}"
        g['modified_by'][k] = g['modified_by'].get(k, 0) + 1
    if s['lam'] is not None:
        g['lam_lt1'] += s['lam'] < 1 - 1e-9
        g['lam_min'] = min(g['lam_min'], float(s['lam']))
    prev = o
g['mod_max'], g['omega_max'], g['omega_limit'] = round(g['mod_max'], 6), round(g['omega_max'], 4), W_MAX
res['guard'] = g
# /cmd_vel 是否逐筆等於 GUARD 狀態的 out（唯一來源的間接證據）
outs = {tuple(np.round(np.array(s['out'], float), 12)) for _, s in st}
cv = np.array([tw(m) for _, m in D['/cmd_vel']])
res['cmd_vel'] = dict(n=len(cv), n_not_a_guard_output=int(sum(tuple(np.round(r, 12)) not in outs for r in cv)), v_max=round(float(np.hypot(cv[:, 0], cv[:, 1]).max()), 4), w_max=round(float(np.abs(cv[:, 2]).max()), 4),
                      note='每筆 /cmd_vel 都能在 GUARD 自報輸出中找到同值＝沒有第二個來源發出不同的值（間接；發布者清單這趟未錄到）')
# Shield 輸入輸出（主 bag）
pre = [(t, tw(m)) for t, m in D['/cmd_vel_pre_shield']]
smt = [(t, tw(m)) for t, m in D['/cmd_vel_smoothed']]
n_pair = min(len(pre), len(smt))
j, dv, unp = 0, [], 0
for t, u in pre:
    while j < len(smt) and smt[j][0] < t - 1e-4:
        j += 1
    if j < len(smt) and smt[j][0] - t < 0.03:
        dv.append((t, float(np.linalg.norm(smt[j][1] - u)), float(np.hypot(*u[:2])), float(np.hypot(*smt[j][1][:2]))))
        j += 1
    else:
        unp += 1
dv = np.array(dv)
res['shield_io'] = dict(n_in=len(pre), n_out=len(smt), paired=len(dv), unpaired=unp, n_changed=int((dv[:, 1] > 1e-4).sum()), change_max=round(float(dv[:, 1].max()), 4), frac_changed=round(float((dv[:, 1] > 1e-4).mean()), 4),
                        out_wheel_speed_viol=int(sum(float(np.abs(W @ r[1]).max()) > RW * W_MAX + 1e-6 for r in smt)), note='輸入輸出依到達順序在 30 ms 內配對；改動＝|輸出−輸入|>1e-4')
h = np.array([m.data for _, m in D['/gmpc/min_h']])
res['min_h'] = dict(n=len(h), min=round(float(h.min()), 3), n_below_0=int((h < 0).sum()), note='記錄到的屏障值；不代表約束貼合或最終命令未違反')
am = np.array([(t, m.pose.pose.position.x, m.pose.pose.position.y) for t, m in D['/amcl_pose']])
tr = np.array([(t, m.pose.position.x, m.pose.position.y) for t, m in D['/model/omni_bot/pose']])
e = [float(np.hypot(a[1] - tr[np.abs(tr[:, 0] - a[0]).argmin(), 1], a[2] - tr[np.abs(tr[:, 0] - a[0]).argmin(), 2])) for a in am]
res['amcl_vs_truth_m'] = dict(n=len(e), median=round(float(np.median(e)), 3), p95=round(float(np.percentile(e, 95)), 3), max=round(float(max(e)), 3))
res['heading_msgs'] = len(D['/gmpc/heading'])
# 真值結果
r = json.load(open(d + '/isaac_run.json'))
Lg = r['run']['log']
t = np.array([x['t'] for x in Lg]); cmd = np.array([x['cmd'] for x in Lg]); xy = np.array([[x['x'], x['y']] for x in Lg]); yaw = np.array([x['yaw'] for x in Lg])
mv = np.where(np.linalg.norm(cmd[:, :2], axis=1) > 1e-3)[0]
cd, cs = np.array([x['clr_dyn'] for x in Lg]), np.array([x['clr_static'] for x in Lg])
v = np.diff(xy, axis=0) / np.diff(t)[:, None]
sp = np.linalg.norm(v, axis=1)
m = sp > 0.05
ang = np.degrees(np.abs(np.arctan2(np.sin(np.arctan2(v[m, 1], v[m, 0]) - yaw[:-1][m]), np.cos(np.arctan2(v[m, 1], v[m, 0]) - yaw[:-1][m]))))
res['truth'] = dict(stop_reason=r['run']['stop_reason'], task_time_s=round(float(t[-1] - t[mv[0]]), 1), path_m=round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 2), min_clr_dyn_m=round(float(cd.min()), 3), min_clr_static_m=round(float(cs.min()), 3),
                    static_who=Lg[int(cs.argmin())]['clr_static_who'], end_dist_m=round(float(Lg[-1]['dist_goal']), 3), heading_vs_travel_deg_median=round(float(np.median(ang)), 1), clearance_note='真值表面間距（模擬器已扣機器人與障礙半徑）')
# bag_extra
if os.path.isdir(d + '/bag_extra'):
    X, _ = read(d + '/bag_extra', ['/shield/diag', '/gmpc/obstacles', '/gmpc/tracks_debug'])
    dg = np.array([list(m.data) for _, m in X['/shield/diag']])
    if len(dg):
        res['shield_diag'] = dict(n=len(dg), active=int((dg[:, 1] > 0.5).sum()), stale=int((dg[:, 12] > 0.5).sum()), fallback=int((dg[:, 16] > 0.5).sum()), unresolved=int((dg[:, 17] > 0.5).sum()), resid_after_max=round(float(dg[:, 14].max()), 6),
                                  d_min_m=round(float(dg[dg[:, 3] >= 0, 3].min()), 3), prev_source={str(int(v)): int((dg[:, 18] == v).sum()) for v in np.unique(dg[:, 18])}, wheel_fit_lt1=int(((dg[:, 19] < 1) & (dg[:, 19] >= 0)).sum()), wheel_fit_prev_infeasible=int((dg[:, 19] < 0).sum()),
                                  dv_max=round(float(dg[:, 4].max()), 4), scan_age_max_s=round(float(dg[:, 11].max()), 3), n_pts_median=int(np.median(dg[:, 2])))
    ob = X['/gmpc/obstacles']
    if ob:
        def nob(m):
            for a in ('obstacles', 'poses', 'markers', 'circles'):
                if hasattr(m, a):
                    return len(getattr(m, a))
            return -1
        n = np.array([nob(m) for _, m in ob])
        res['tracker_obstacles'] = dict(n_msg=len(n), n_max=int(n.max()), frac_nonempty=round(float((n > 0).mean()), 3), msg_type=str(type(ob[0][1]).__name__))
else:
    res['shield_diag'] = res['tracker_obstacles'] = 'missing（側錄未啟動；缺少不當零）'
side = d + '/side/topic_endpoints.txt'
res['publishers'] = open(side).read()[:1500] if os.path.exists(side) else 'missing（側錄未啟動）'
json.dump(res, open(outp, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'counts'}, ensure_ascii=False, indent=1))
