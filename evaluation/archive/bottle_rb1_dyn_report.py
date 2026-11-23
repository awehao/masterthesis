#!/usr/bin/env python3
"""R-B1＋迎面移動物實跑報告（Codex 20261102_201528 交付項；只讀紀錄）。對照無移動物基線 bottle_rb1_gaze_1。
時間鏈（各附當時操作相位）：
  障礙起動（讀回速度首次 > 0.05 m/s）；首次被追蹤為動態（tracker_log 動態項落在障礙真值 MATCH_M 內）；首次進入求解器（s1_cbf obs_json 非靜態項，區域＋O 後落在真值 MATCH_M 內）。
  動態項分「對應真障礙」與「其他」（不在真值 MATCH_M 內，可能是追蹤器假動態）分開計數；未進入與未記錄不混用（s1_cbf 每求解拍都有紀錄）。
距離三種口徑（對障礙表面、讀回位置）：底盤中心、0.39 m 圓盤外緣、連桿／持瓶（高度 ≤ 障礙高的點）。
安全層：Shield 介入與改寫量（nav_chain.shield）、第一拍列（paired_strategy.u0_rows）、CBF 鬆弛（s1_cbf eps0 > 1e-6 的拍、時刻、相位、當時是否有動態項）。
滑移：TCP→瓶相對平移（量測 FK＋讀回瓶位姿，對抓持成立時刻），列各事件時的值與相鄰事件之間的增量。
    python3 evaluation/bottle_rb1_dyn_report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

RUN, BASE = 'bottle_rb1_dyn_1', 'bottle_rb1_gaze_1'
OUT = os.path.join(HERE, 'results', 'bottle', 'rb1_dyn_report.json')
MATCH_M, R_DISC = 0.60, 0.39
SLIP_EV = ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_out_start', 'compact_returned')


def load(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    return D, r, {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}


def phase_at(r, tt):
    ph, pl = 'CRUISE', None
    for e in r['events']:
        if e['t'] > tt + 1e-9:
            break
        if e['what'] == 'phase':
            ph = e['to']
        elif e['what'] == 'place_phase':
            pl = e['to']
    return pl if ph == 'CARRY' and pl else ph


def slip_table(r, c):
    t = c['t']
    K = G.C.K
    tg = float(r['t_grasp'])
    idx = np.flatnonzero((t >= tg) & (np.mod(np.arange(len(t)), 5) == 0))
    d = []
    T0 = None
    for k in idx:
        q = np.array([c[n][k] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        Tt = K.fk(q, 'link_tcp')
        w, x, y, z = (c[n][k] for n in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        p = np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]])
        Rb = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = Rb, p
        Tr = np.linalg.inv(Tt) @ Tb
        if T0 is None:
            T0 = Tr
        d.append(float(np.linalg.norm(Tr[:3, 3] - T0[:3, 3])) * 1e3)
    ts, ds = t[idx], np.array(d)
    rel = next((e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'RELEASE'), ts[-1])
    marks = [('抓持成立', tg)] + [(n, next((e['t'] for e in r['events'] if e['what'] == n), None)) for n in SLIP_EV] + [('PARKED', next((e['t'] for e in r['events'] if e.get('to') == 'PARKED'), None)), ('RELEASE', rel)]
    out, prev = [], None
    for n, tt in marks:
        if tt is None:
            continue
        v = float(np.interp(tt, ts, ds))
        out.append(dict(ev=n, t=round(float(tt), 2), slip_mm=round(v, 2), delta_mm=None if prev is None else round(v - prev, 2)))
        prev = v
    m = ts <= rel
    return dict(table=out, max_mm=round(float(ds[m].max()), 2))


def main():
    D, r, c = load(RUN)
    _, rb, cb = load(BASE)
    O = np.array(r['region']['offset'], float)
    T0 = float(r['segtarget_link']['T0'])
    CA = r['dyn']['case']
    t = c['t']
    ox, oy = c['dyn_x'], c['dyn_y']
    vo = np.hypot(c['dyn_vx'], c['dyn_vy'])
    k_move = int(np.argmax(vo > 0.05))
    # 追蹤器
    TL = np.load(os.path.join(D, 'tracker_log.npz'), allow_pickle=True)
    tr, data = np.asarray(TL['t_recv'], float), np.asarray(TL['data'], float)
    first_trk, n_trk_true, n_trk_other = None, 0, 0
    trk_err, trk_verr = [], []
    for i in range(len(tr)):
        nd = int(data[i, 2]) // 5                                  # 欄位是「值數」，每條軌跡 5 值
        k = int(np.argmin(np.abs(t - tr[i])))
        for j in range(nd):
            px, py, vx, vy = data[i, 4 + 5 * j], data[i, 5 + 5 * j], data[i, 7 + 5 * j], data[i, 8 + 5 * j]   # (x, y, 半徑, vx, vy)
            e = float(np.hypot(px - ox[k], py - oy[k]))
            if e <= MATCH_M:
                n_trk_true += 1
                trk_err.append(e)
                trk_verr.append(float(np.hypot(vx - c['dyn_vx'][k], vy - c['dyn_vy'][k])))
                if first_trk is None:
                    first_trk = float(tr[i])
            else:
                n_trk_other += 1
    # 求解器
    C = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    ct = np.asarray(C['t'], float)
    obs = [json.loads(s) for s in C['obs_json']]
    first_sol, n_true, n_other, other_ticks = None, 0, 0, []
    true_ticks = []
    for i, x in enumerate(obs):
        k = int(np.argmin(np.abs(t - ct[i])))
        hit = False
        oth = False
        for o in x:
            if o.get('static'):
                continue
            if float(np.hypot(o['x'] + O[0] - ox[k], o['y'] + O[1] - oy[k])) <= MATCH_M:
                hit = True
            else:
                oth = True
        if hit:
            n_true += 1
            true_ticks.append(float(ct[i]))
            if first_sol is None:
                first_sol = float(ct[i])
        if oth:
            n_other += 1
            other_ticks.append(float(ct[i]))
    t_mv = float(t[k_move])
    trk_all = [float(tr[i]) for i in range(len(tr)) for j in range(int(data[i, 2]) // 5) if float(np.hypot(data[i, 4 + 5 * j] - ox[int(np.argmin(np.abs(t - tr[i])))], data[i, 5 + 5 * j] - oy[int(np.argmin(np.abs(t - tr[i])))])) <= MATCH_M]
    trk_pre, trk_post = [x for x in trk_all if x < t_mv], [x for x in trk_all if x >= t_mv]
    sol_pre, sol_post = [x for x in true_ticks if x < t_mv], [x for x in true_ticks if x >= t_mv]
    chain = dict(move=dict(t=round(t_mv, 2), phase=phase_at(r, t_mv)), scheduled_t_start=CA['t_start'],
                 before_move=dict(note='障礙仍靜止在起點，卻被追蹤器判成動態並進入求解器（誤判），與起動後分開', tracker_n=len(trk_pre), tracker_range=[round(min(trk_pre), 2), round(max(trk_pre), 2)] if trk_pre else None,
                                  solver_ticks=len(sol_pre), solver_range=[round(min(sol_pre), 2), round(max(sol_pre), 2)] if sol_pre else None, solver_phases=sorted({phase_at(r, x) for x in sol_pre})),
                 first_tracked_dynamic_after_move=None if not trk_post else dict(t=round(trk_post[0], 2), phase=phase_at(r, trk_post[0]), delay_s=round(trk_post[0] - t_mv, 2)),
                 first_in_solver_after_move=None if not sol_post else dict(t=round(sol_post[0], 2), phase=phase_at(r, sol_post[0]), delay_s=round(sol_post[0] - t_mv, 2)), solver_ticks_after_move=len(sol_post),
                 first_tracked_dynamic=None if first_trk is None else dict(t=round(first_trk, 2), phase=phase_at(r, first_trk)),
                 first_in_solver=None if first_sol is None else dict(t=round(first_sol, 2), phase=phase_at(r, first_sol)), last_in_solver=round(true_ticks[-1], 2) if true_ticks else None,
                 phases_in_solver=sorted({phase_at(r, x) for x in true_ticks}), solver_ticks_true=n_true, solver_ticks_other=n_other, other_range=[round(min(other_ticks), 2), round(max(other_ticks), 2)] if other_ticks else None, n_solve_ticks=len(obs),
                 tracker_true=n_trk_true, tracker_other=n_trk_other, tracker_pos_err=None if not trk_err else dict(p50=round(float(np.median(trk_err)), 3), max=round(max(trk_err), 3)), tracker_vel_err=None if not trk_verr else dict(p50=round(float(np.median(trk_verr)), 3), max=round(max(trk_verr), 3)))
    # 距離三種口徑
    dc = np.hypot(c['base_x'] - ox, c['base_y'] - oy) - float(CA['radius'])
    ki = int(np.argmin(dc))
    BP = G.bottle_points()
    best_l, best_b = (9.9, None), (9.9, None)
    for k in np.flatnonzero(dc < 2.0)[::5]:
        q = np.array([c[n][k] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        P = np.vstack(list(G.link_world(q, float(c['finger1_q'][k])).values()))
        P = P[P[:, 2] <= CA['height']]
        d = float((np.hypot(P[:, 0] - ox[k], P[:, 1] - oy[k]) - CA['radius']).min())
        if d < best_l[0]:
            best_l = (round(d, 4), round(float(t[k]), 2))
        if r['t_grasp'] <= t[k] <= next((e['t'] for e in r['events'] if e.get('to') == 'RELEASE'), 1e9):
            w, x, y, z = (c[n][k] for n in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
            Rb = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            Pb = (Rb @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]])
            Pb = Pb[Pb[:, 2] <= CA['height']]
            if len(Pb):
                d = float((np.hypot(Pb[:, 0] - ox[k], Pb[:, 1] - oy[k]) - CA['radius']).min())
                if d < best_b[0]:
                    best_b = (round(d, 4), round(float(t[k]), 2))
    dist = dict(base_center_to_surface=dict(m=round(float(dc[ki]), 4), t=round(float(t[ki]), 2), phase=phase_at(r, float(t[ki]))), base_disc_edge_to_surface=dict(m=round(float(dc[ki]) - R_DISC, 4), disc_radius=R_DISC),
                links_to_surface=dict(m=best_l[0], t=best_l[1]), held_bottle_to_surface=dict(m=best_b[0], t=best_b[1]))
    # 會遇窗速度
    k10 = np.flatnonzero(np.abs(t * 10 - np.round(t * 10)) < 1e-6)
    sp = np.hypot(np.diff(c['base_x'][k10]), np.diff(c['base_y'][k10])) / 0.1
    acc = np.abs(np.diff(sp)) / 0.1
    tn = t[k10][1:-1]
    win = (dc[k10][1:-1] - R_DISC) < 1.5
    enc = dict(window=[round(float(tn[win][0]), 2), round(float(tn[win][-1]), 2)] if win.any() else None, v_min=round(float(sp[1:][win].min()), 4) if win.any() else None, v_mean=round(float(sp[1:][win].mean()), 4) if win.any() else None,
               dv_max=round(float(acc[win].max()), 3) if win.any() else None, dv_p95=round(float(np.percentile(acc[win], 95)), 3) if win.any() else None)
    # 安全層
    sh = r['nav_chain']['shield']
    u0 = r['paired_strategy'].get('u0_rows') or {}
    eps, res_s = np.asarray(C['eps0'], float), np.asarray(C['resid_s'], float)
    m_e = eps > 1e-6
    has_dyn = np.array([any(not o.get('static') for o in x) for x in obs])
    tt_e = ct[m_e]
    safety = dict(shield=dict(n=sh['n'], active=sh['active'], stale=sh['stale'], fallback=sh['fallback'], unresolved=sh['unresolved'], dv_max=sh['dv_max'], d_min=round(sh['d_min'], 4)),
                  u0_rows=dict(n_solve=u0.get('n_solve'), n_bind=u0.get('n_bind'), rows_max=u0.get('rows_max'), violation=u0.get('violation')),
                  cbf_slack=dict(n_ticks=int(m_e.sum()), eps0_max=round(float(eps.max()), 4), t_at_max=round(float(ct[int(np.argmax(eps))]), 2), phase_at_max=phase_at(r, float(ct[int(np.argmax(eps))])),
                                 range=[round(float(tt_e[0]), 2), round(float(tt_e[-1]), 2)] if len(tt_e) else None, phases=sorted({phase_at(r, float(x)) for x in tt_e}), with_dyn_ticks=int((m_e & has_dyn).sum()), resid_s_min=float(np.nanmin(res_s))),
                  cbf_active_max=int(np.asarray(C['cbf_active']).max()), min_h_min=round(float(np.nanmin(np.asarray(C['min_h'], float))), 4))
    sm = np.asarray(r['solve_ms_all'], float)
    over = np.flatnonzero(sm > 50.0)
    safety['solve_over_50ms'] = [dict(t=round(float(ct[i]), 2), ms=round(float(sm[i]), 1), phase=phase_at(r, float(ct[i])), n_obs=len(obs[i]), has_dyn=bool(has_dyn[i])) for i in over if i < len(ct)]
    Cb = np.load(os.path.join(HERE, 'runs', BASE, 's1_cbf.npz'), allow_pickle=True)
    eb = np.asarray(Cb['eps0'], float)
    safety['cbf_slack_baseline'] = dict(n_ticks=int((eb > 1e-6).sum()), eps0_max=round(float(eb.max()), 4), t_at_max=round(float(np.asarray(Cb['t'], float)[int(np.argmax(eb))]), 2))
    # 對照
    def summ(rr):
        ev = {e['to']: e['t'] for e in rr['events'] if e['what'] in ('phase', 'place_phase')}
        pm = (rr.get('place') or {}).get('metrics') or {}
        g = [x for x in rr['online']['grasp_plans'] if x['state'] == 'accepted'][0]
        return dict(done=next((e['t'] for e in rr['events'] if e['what'] == 'mission_done'), None), success_orig=rr.get('success'), carry_nav=ev.get('CARRY_NAV'), place_approach=ev.get('PLACE_APPROACH'), parked=ev.get('PARKED'), release=ev.get('RELEASE'),
                    t_grasp=rr.get('t_grasp'), t_lift_ok=rr.get('t_lift_ok'), slip=rr['slip_max_mm_deg'], place_mm=round(pm['offset'] * 1e3, 2) if pm.get('offset') is not None else None, tilt_deg=round(pm['tilt'], 3) if pm.get('tilt') is not None else None,
                    forb={k: v['peak'] for k, v in rr['forbidden_pulses'].items() if v['peak'] > 0}, accept=round(g['t_acc'], 2), margin=round(g['x_latest'] - g['x_acc'], 4), timing={k: rr['timing'][k] for k in ('solve', 'advance')}, n_over_50ms=rr.get('n_over_50ms'),
                    events_done_end=[e['t'] for e in rr['events'] if e['what'] in ('done', 'end_tail')])
    out = dict(run=RUN, baseline=BASE, case=CA, region_offset=O.tolist(), T0=T0, chain=chain, distance=dist, encounter=enc, safety=safety, run_sum=summ(r), base_sum=summ(rb),
               carry_gov=dict(events=r['carry_gov']['events'], hold_s=r['carry_gov']['hold_s'], min_stop_margin=r['carry_gov']['min_stop_margin'], park_tol_n=len(r['carry_gov']['park_tol_log'])),
               slip_run=slip_table(r, c), slip_base=slip_table(rb, cb))
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('chain', 'distance', 'encounter', 'safety', 'run_sum', 'base_sum', 'carry_gov', 'slip_run', 'slip_base'):
        print(k, json.dumps(out[k], ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
