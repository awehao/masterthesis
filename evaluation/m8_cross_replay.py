#!/usr/bin/env python3
"""r7 候選 B 的離線比較（r2 口徑：Codex 20261110_223314 第 2 項——區分預測範圍內無衝突／超出 T_LONG（未知）／資料過期或紀錄結尾；到達＝逐步滾動重評未失效，不是整段已驗證）。
（r1：Codex 20261110_221334：先用既有紀錄，對照原路與一個近垂直穿越候選）。只讀紀錄、不改控制。
一 逐時刻（機器人照該趟實際位姿／速度／朝向，開迴路）：
   原路＝當時巡航路徑（/plan 終點＝門前觀察區）自投影點起；候選＝m8_cross.build_cross（方向取原路第一個衝突的確認航跡，stable 才建）。
   各報：是否立即可行（DT_MARGIN 時空核對）、第一個衝突（航跡、kind、覆蓋時刻、機器人時窗）、候選拒絕原因、
   接近等待點／進入幾何帶／離開幾何帶的時間界（t_lb、t_ub），原路另列舊時間式（切線轉角）以揭露模型修正的影響。
二 虛擬穿越試驗（每 TRIAL_DT 一次）：用「到該時刻為止」最後一條 stable 航跡中心線（線上可得，不用真值路線）；
   原路＝起始巡航路徑 plan#0，起點＝原路在近側側向 hb＋db 處；候選＝同交點的垂直穿越，起點＝W。兩者都假設靜止在起點。
   進入：整段（至終點）DT_MARGIN 無衝突才起步（同 WindowLayer2 CRUISE）；起步後每 STEP_DT 用當時錄到的航跡：
   機器人在帶內（first_arrival 有值）⇒ 零餘裕核對剩餘路徑到離開覆蓋為止（同 WindowLayer3 COMMIT 持續規則），失敗＝commit_invalid；
   不在帶內 ⇒ 整段 DT_MARGIN 核對，失敗＝wait（會在帶外停等，不算穿越失敗也不算到達）。到終點 align_r 0.25 內＝arrived。
   虛擬運動：ub＝時間界的慢剖面（T_DELAY、A_UB、V_UB_R）；nom＝該趟巡航實測速度中位數（A_UB 加速）。轉角不另減速（時間界已含）。
限制：錄到的航跡是該趟機器人視角下的感知結果（遮擋、距離會隨機器人位置不同）；虛擬機器人不經 GMPC／CBF／Shield；只驗決策邏輯與時窗，不代表安全。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_cross_replay.py RUN_ID [T_A T_B]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import m8_cross as X  # noqa: E402
import m8_window2 as W2  # noqa: E402

GOAL = (0.5, -1.543)
ALIGN_R = 0.25
TRIAL_DT, STEP_DT, T_TRIAL_MAX = 0.5, 0.2, 60.0


def yaw_of(o):
    return 2.0 * math.atan2(o.z, o.w)


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/tracks_debug', '/global_costmap/costmap', '/plan'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    nav = json.load(open(os.path.join(D, 'm8nav.json')))
    seg = {s_['seg']: s_['t'] for s_ in nav['segments']}
    ta = float(sys.argv[2]) if len(sys.argv) > 2 else float(seg['CRUISE'])
    tb = float(sys.argv[3]) if len(sys.argv) > 3 else float(nav['segments'][-1]['t'])
    plans = []
    for r, m in B['/plan']:
        if len(m.poses) < 10:
            continue
        P = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])
        if math.hypot(P[-1, 0] - GOAL[0], P[-1, 1] - GOAL[1]) < 0.05:
            plans.append((to_sim(r), P, np.array([yaw_of(p.pose.orientation) for p in m.poses])))
    cms = sorted(((MB.stamp(m), m) for _, m in B['/global_costmap/costmap']), key=lambda z: z[0])
    cur = {'m': None}

    def blocked(p):
        m = cur['m']
        if m is None:
            return True
        i = m.info
        ix, iy = int((p[0] - i.origin.position.x) / i.resolution), int((p[1] - i.origin.position.y) / i.resolution)
        if not (0 <= ix < i.width and 0 <= iy < i.height):
            return True
        c = m.data[iy * i.width + ix]
        return c >= 99 or c < 0

    def set_cm(t):
        cur['m'] = next((x[1] for x in reversed(cms) if x[0] <= t), None)
    drop = {int(v) for v in os.environ.get('DROP_TID', '').split(',') if v.strip()}   # 診斷用反事實（離線定位阻點，不是可用設定）
    cls = None
    if os.environ.get('STATIC_CLS'):                                    # r1 口徑＋分類候選（Codex 223314 第 3 項；離線，不接控制器）
        import m8_trackcls as TC
        cls = TC.StaticClassifier()
    H = W2.TrackHistory()
    TM = []                                                            # (t, M) 全程預先分類（線上歷史，只用過去資料）
    n_static_dropped = 0
    t_rec_end = to_sim(B['/gmpc/tracks_debug'][-1][0]) if B['/gmpc/tracks_debug'] else -1.0
    for r, m in B['/gmpc/tracks_debug']:
        t = to_sim(r)
        if t > tb + T_TRIAL_MAX:
            break
        T = [x for x in MB.tracks(m) if int(x['tid']) not in drop]
        if cls is not None:
            cls.update(t, T)
            keep = [x for x in T if not (x.get('confirmed') and cls.is_static(t, x['tid']))]
            n_static_dropped += sum(1 for x in T if x.get('confirmed')) - sum(1 for x in keep if x.get('confirmed'))
            T = keep
        H.update(t, T)
        TM.append((t, [(x, H.kind(t, x)) for x in T if x.get('confirmed')]))
    tt_tm = np.array([x[0] for x in TM])

    def M_at(t, with_status=False):
        """該時刻最新一筆航跡集合；with_status 時另回 ok｜stale（最新一筆比 TRACK_MAX_AGE 舊）｜record_end（超過紀錄結尾）。"""
        i = int(np.searchsorted(tt_tm, t, side='right')) - 1
        M = TM[i][1] if i >= 0 else []
        if not with_status:
            return M
        if t > t_rec_end + W2.TRACK_MAX_AGE or i >= len(TM) - 1 and t > tt_tm[-1] + W2.TRACK_MAX_AGE:
            return M, 'record_end'
        return M, ('ok' if i >= 0 and t - tt_tm[i] <= W2.TRACK_MAX_AGE else 'stale')
    mv = np.hypot(np.gradient(od[:, 1], od[:, 0]), np.gradient(od[:, 2], od[:, 0]))
    kc = (od[:, 0] >= ta) & (od[:, 0] <= tb) & (mv > 0.1)
    v_nom = float(np.median(mv[kc])) if kc.any() else W2.V_UB_R

    # ---- 一 逐時刻
    rows, last_line, P_cur, k_proj = [], None, None, 0
    for t, M in TM:
        if t < ta or t > tb:
            continue
        pp = [x for x in plans if x[0] <= t]
        if not pp:
            continue
        if P_cur is None or pp[-1][1] is not P_cur[0]:
            P_cur, k_proj = (pp[-1][1], pp[-1][2]), 0
        set_cm(t)
        pr = MB.pose_at(od, t)
        j = min(int(np.searchsorted(od[:, 0], t)), len(od) - 1)
        j0 = int(np.searchsorted(od[:, 0], t - 0.2))
        vel = np.array([(od[j, 1] - od[j0, 1]) / max(od[j, 0] - od[j0, 0], 1e-6), (od[j, 2] - od[j0, 2]) / max(od[j, 0] - od[j0, 0], 1e-6)])
        rp, ryaw = np.array([pr[1], pr[2]]), float(pr[3])
        d = np.linalg.norm(P_cur[0][k_proj:k_proj + 40] - rp, axis=1)
        if len(d):
            k_proj += int(np.argmin(d))
        P, Y = P_cur[0][k_proj:], P_cur[1][k_proj:]
        if len(P) < 2:
            continue
        s, lb, ub, parts = X.window_m(P, Y, rp, ryaw, vel)
        hd = P[min(len(P) - 1, 4)] - rp
        v_al = max(0.0, float(vel @ hd) / max(np.linalg.norm(hd), 1e-9))
        _, _, ub_old = W2.robot_window(P, rp, v_al)
        conf, cov, first = X.conflicts(P, lb, ub, M, W2.DT_MARGIN)
        conf_old, _, _ = X.conflicts(P, lb, ub_old, M, W2.DT_MARGIN)
        row = dict(t=round(t, 2), rp=rp.round(3).tolist(), orig=dict(ok=not conf.any(), status=X.status(conf, X.beyond(lb, ub, W2.DT_MARGIN)), ok_old_time=not conf_old.any(), first=first,
                                                                    rot_s=parts['rot_s'], corner_s=parts['corner_s']))
        threat = None
        if first is not None:
            threat = next(((x, k) for x, k in M if int(x.get('tid', -1)) == first['tid']), None)
        if threat is not None and threat[1] == 'stable' and math.hypot(threat[0]['vx'], threat[0]['vy']) >= W2.V_MOVER:
            last_line = (t, X.line_of(threat[0]), int(threat[0]['tid']))
            i0, i1 = X.band_span(P, last_line[1])
            row['orig'].update(band_in=X.times_at(s, lb, ub, i0), band_out=X.times_at(s, lb, ub, i1),
                               band_out_old_time=X.times_at(s, lb, ub_old, i1))
        if threat is None:
            row['cand'] = dict(built=False, why='no_conflict_on_original' if first is None else 'threat_not_confirmed')
        else:
            c, why = X.build_cross(rp, ryaw, P, Y, M, blocked, threat)
            if c is None:
                row['cand'] = dict(built=False, why=why)
            else:
                sc, lbc, ubc, pc = X.window_m(c['P'], c['yaw'], rp, ryaw, vel)
                cc, covc, fc = X.conflicts(c['P'], lbc, ubc, M, W2.DT_MARGIN)
                i0, i1 = X.band_span(c['P'], c['line'], start=c['i_W'])
                row['cand'] = dict(built=True, ok=not cc.any(), status=X.status(cc, X.beyond(lbc, ubc, W2.DT_MARGIN)), first=fc, approach_ok=not cc[:c['i_W'] + 1].any(), W=c['W'].round(3).tolist(), Wp=c['Wp'].round(3).tolist(),
                                   tid=c['tid'], len=c['len'], at_W=X.times_at(sc, lbc, ubc, c['i_W']), band_in=X.times_at(sc, lbc, ubc, i0),
                                   band_out=X.times_at(sc, lbc, ubc, i1), rot_s=pc['rot_s'], corner_s=pc['corner_s'])
        rows.append(row)

    # ---- 二 虛擬穿越試驗
    P0, Y0 = plans[0][1], plans[0][2]
    line_hist = []                                                      # 線上：到 t 為止最後一條 stable 航跡中心線（原路跨過者）
    for t, M in TM:
        for x, k in M:
            if k == 'stable' and math.hypot(x['vx'], x['vy']) >= W2.V_MOVER:
                p0, vh, n = X.line_of(x)
                lat = (P0 - p0) @ n
                if (lat > 0).any() and (lat < 0).any():                  # 原路跨過這條中心線
                    line_hist.append((t, x))
    lt = np.array([x[0] for x in line_hist]) if line_hist else np.zeros(0)
    hb, db = X.band_half(), W2.d_brake(W2.V_UB_R)

    def start_on_orig(line):
        p0, vh, n = X.line_of(line)
        lat = (P0 - p0) @ n
        side = math.copysign(1.0, lat[0])
        ii = np.flatnonzero(side * lat <= hb + db)
        return (int(ii[0]) - 1 if len(ii) and ii[0] > 0 else 0), side

    def simulate(P, Y, t_s, prof):
        """虛擬機器人自 P[0] 靜止起步沿 P；回傳結果字典。
        entry_status：進入時整段 clear_within_horizon｜clear_beyond_horizon（有點的時窗超出 T_LONG＝未知）。
        每步資料：最新航跡比 TRACK_MAX_AGE 舊或超過紀錄結尾 ⇒ data_insufficient（不算到達也不算失效）。
        到達＝arrived_rolling：每步滾動重評都沒失效；不是整段已驗證（n_steps_beyond＝核對範圍有點超出 T_LONG 的步數）。"""
        seglen = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
        S = np.cumsum(seglen)
        set_cm(t_s)
        M0, st0 = M_at(t_s, True)
        if st0 != 'ok':
            return dict(entry=False, data=st0)
        s_, lb_, ub_, _ = X.window_m(P[1:], Y[1:], P[0], Y[0], np.zeros(2))
        cf, _, f0 = X.conflicts(P[1:], lb_, ub_, M0, W2.DT_MARGIN)
        est = X.status(cf, X.beyond(lb_, ub_, W2.DT_MARGIN))
        if est == 'conflict':
            return dict(entry=False, first=f0)
        vmax = W2.V_UB_R if prof == 'ub' else v_nom
        delay = W2.T_DELAY if prof == 'ub' else W2.T_DELAY_LB
        pos_s, v, tau, n_bey = 0.0, 0.0, 0.0, 0
        base = dict(entry=True, entry_status=est)
        while tau < T_TRIAL_MAX:
            tau += STEP_DT
            if tau > delay:
                v = min(vmax, v + W2.A_UB * STEP_DT)
                pos_s += v * STEP_DT
            k = int(np.searchsorted(S, pos_s))
            if k >= len(P) - 1 or math.hypot(P[-1, 0] - np.interp(pos_s, S, P[:, 0]), P[-1, 1] - np.interp(pos_s, S, P[:, 1])) <= ALIGN_R:
                return dict(base, outcome='arrived_rolling', t=round(tau, 1), n_steps_beyond=n_bey)
            M, st = M_at(t_s + tau, True)
            if st != 'ok':
                return dict(base, outcome='data_insufficient', data=st, t=round(tau, 1), s=round(pos_s, 2), s_total=round(float(S[-1]), 2))
            q = np.array([np.interp(pos_s, S, P[:, 0]), np.interp(pos_s, S, P[:, 1])])
            dirv = P[k] - P[k - 1] if k > 0 else P[1] - P[0]
            vel = dirv / max(np.linalg.norm(dirv), 1e-9) * v
            R, RY = P[k:], Y[k:]
            s2, lb2, ub2, _ = X.window_m(R, RY, q, float(Y[max(k - 1, 0)]), vel)
            fa = W2.WindowLayer2._first_arrival(None, q, M)
            if fa is not None:                                          # 帶內：零餘裕持續核對到離開覆蓋
                _, cov2, _ = X.conflicts(R, lb2, ub2, M, 0.0)
                nc = np.flatnonzero(~cov2)
                i_out = int(nc[0]) if len(nc) else len(R) - 1
                cf2, _, f2 = X.conflicts(R[:i_out + 1], lb2[:i_out + 1], ub2[:i_out + 1], M, 0.0)
                n_bey += int((X.beyond(lb2[:i_out + 1], ub2[:i_out + 1], 0.0) > 0).any())
                if cf2.any():
                    return dict(base, outcome='commit_invalid', t=round(tau, 1), s=round(pos_s, 2), s_total=round(float(S[-1]), 2), first=f2, threat_kind=fa[2])
            else:
                cf2, _, f2 = X.conflicts(R, lb2, ub2, M, W2.DT_MARGIN)
                n_bey += int((X.beyond(lb2, ub2, W2.DT_MARGIN) > 0).any())
                if cf2.any():
                    return dict(base, outcome='wait', t=round(tau, 1), s=round(pos_s, 2), s_total=round(float(S[-1]), 2), first=f2)
        return dict(base, outcome='timeout', t=round(tau, 1))
    trials = []
    t_s = ta
    while t_s <= tb:
        i = int(np.searchsorted(lt, t_s, side='right')) - 1
        if i < 0:
            t_s += TRIAL_DT
            continue
        line = line_hist[i][1]
        tr = dict(t=round(t_s, 2), line_tid=int(line['tid']), line_age=round(t_s - line_hist[i][0], 2))
        i_a, side = start_on_orig(line)
        PA = np.vstack([P0[i_a][None, :], P0[i_a + 1:]])
        YA = Y0[i_a:]
        for prof in ('ub', 'nom'):
            tr[f'orig_{prof}'] = simulate(PA, YA, t_s, prof)
        set_cm(t_s)
        c, why = X.build_cross(P0[i_a], float(Y0[i_a]), P0[i_a:], Y0[i_a:], M_at(t_s), blocked, (line, 'stable'))
        if c is None:
            tr['cand_build'] = why
        else:
            PC = np.vstack([c['W'][None, :], c['P'][c['i_W'] + 1:]])
            YC = c['yaw'][c['i_W']:]
            tr['cand_start'] = c['W'].round(3).tolist()
            tr['cand_len'] = c['len']
            for prof in ('ub', 'nom'):
                tr[f'cand_{prof}'] = simulate(PC, YC, t_s, prof)
        trials.append(tr)
        t_s += TRIAL_DT

    # ---- 彙整
    def cnt(key, field, val):
        return sum(1 for r in rows if (r.get(key) or {}).get(field) == val)
    why_c = {}
    for r in rows:
        if not r['cand'].get('built'):
            w = r['cand']['why']
            w = w if isinstance(w, str) else w.get('why')
            why_c[w] = why_c.get(w, 0) + 1
    oc = {}
    for opt in ('orig_ub', 'orig_nom', 'cand_ub', 'cand_nom'):
        z = [tr[opt] for tr in trials if opt in tr]
        oc[opt] = dict(n=len(z), entry=sum(1 for q in z if q['entry']))
        for q in z:
            if q['entry']:
                oc[opt][q['outcome']] = oc[opt].get(q['outcome'], 0) + 1
        for q in z:
            if q['entry']:
                es = 'entry_' + q['entry_status']
                oc[opt][es] = oc[opt].get(es, 0) + 1
            elif q.get('data'):
                oc[opt]['no_entry_data_' + q['data']] = oc[opt].get('no_entry_data_' + q['data'], 0) + 1
        ar = [q['t'] for q in z if q.get('outcome') == 'arrived_rolling']
        oc[opt]['t_arrive_med'] = None if not ar else round(float(np.median(ar)), 1)
        inv = [q for q in z if q.get('outcome') == 'commit_invalid']
        oc[opt]['invalid_t_med'] = None if not inv else round(float(np.median([q['t'] for q in inv])), 1)
        oc[opt]['invalid_kinds'] = {kk: sum(1 for q in inv if q['first']['kind'] == kk) for kk in ('stable', 'unstable')}
    bi = [(r['orig'].get('band_in'), r['orig'].get('band_out'), (r['cand'] or {}).get('band_in'), (r['cand'] or {}).get('band_out')) for r in rows if r['cand'].get('built')]
    dur = dict(orig=[round(b[1]['t_ub'] - max(b[0]['t_lb'], 0.0), 2) for b in bi if b[0] and b[1]],
               cand=[round(b[3]['t_ub'] - b[2]['t_lb'], 2) for b in bi if b[2] and b[3]])
    S = dict(run=run, window=[round(ta, 2), round(tb, 2)], v_nom=round(v_nom, 3), hb=round(hb, 3), db=round(db, 3),
             per_instant=dict(n=len(rows), orig_ok=cnt('orig', 'ok', True), orig_ok_old_time=cnt('orig', 'ok_old_time', True),
                              cand_built=sum(1 for r in rows if r['cand'].get('built')), cand_ok=cnt('cand', 'ok', True), cand_approach_ok=cnt('cand', 'approach_ok', True),
                              cand_not_built=why_c,
                              band_dur_ub_minus_lb=dict(orig_med=None if not dur['orig'] else float(np.median(dur['orig'])), cand_med=None if not dur['cand'] else float(np.median(dur['cand'])))),
             per_instant_status=dict(orig={k_: sum(1 for r in rows if r['orig'].get('status') == k_) for k_ in ('conflict', 'clear_within_horizon', 'clear_beyond_horizon')},
                                     cand={k_: sum(1 for r in rows if (r['cand'] or {}).get('status') == k_) for k_ in ('conflict', 'clear_within_horizon', 'clear_beyond_horizon')}),
             trials=oc, record_end_t=round(t_rec_end, 2), static_cls=None if cls is None else dict(params=cls.params(), n_confirmed_dropped=n_static_dropped),
             note='開迴路逐時刻＋虛擬穿越試驗；錄到的航跡受該趟機器人視角影響；虛擬機器人不經 GMPC／CBF／Shield；只驗決策邏輯與時窗，不代表安全')
    if drop:
        S['counterfactual_drop_tid'] = sorted(drop)
        S['note'] += '；DROP_TID：診斷用反事實，移除指定航跡以定位阻點，不是可用設定、不代表通過'
    if cls is not None:
        S['note'] += '；STATIC_CLS：靜態分類候選（離線；判靜態的確認航跡不進時窗層，掃描、CBF、Shield 不受影響）'
    out = dict(summary=S, rows=rows, trials=trials)
    tag = ('_drop' + '_'.join(map(str, sorted(drop))) if drop else '') + ('_cls' if cls is not None else '')
    json.dump(out, open(os.path.join(D, 'm8_cross_replay_r2' + tag + '.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
