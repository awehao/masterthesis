#!/usr/bin/env python3
"""K3-B 整體判定器 r2（Codex 20261026_010158 必修 4：獨立判定與缺證據拒收；door_k3m_check_k3b.py〔r1，凍結 k3b r1〕不改，本檔重用其分項）。
核心改 door_k3b_release3（r3 帶、5 N 與終點保持新規則）。起跑前凍結。

r1 全部保留：舊 W1f2 鏈（只取代兩項明列 DONE 理由）、M1–M2、K1–K3、K6–K9；K4、K5 由下列 r2 版本取代。

r2 新增與取代：
  K0 room 覆蓋：自 mode 1 提交步至結束逐物理步連續；t_open 起每步 q_finger 為 2 個有限值、leaf_contacts 含兩指且 bar／ambiguous／panel 有限、
     wall_contacts 為 dict ⇒ 缺一即不通過（刪來源、刪奇數步都拒收）。
  K4' 5 N：
     - 恢復紀錄 ok、from 20、to 5、讀回兩指皆 5（不只 ok 旗標）；切換晚於 C_CLEAR 與 mode 3 提交、距 t_conf ≥ 0.5 s；
     - 之後定期讀回逐秒覆蓋到結束（筆數 ≥ ⌊(t_end−t_sw)/1⌋−1、時間遞增、每筆 ok 且讀回 5）。
  K5' 幾何逐步（FK，每物理步）：
     - |x| ≤ 3 mm、|z| ≤ Z(s)、轉角 ≤ 0.5°；
     - 門角 Δ ≥ −1° 且 Δ_pred＝Δ＋max(q̇,0)·0.2 ≤ 帶(s_meas)，q̇ 取 room 門角 0.1 s 後向差分；max s_meas ≥ 0.095。
  K10 退出重算：
     - 以任務端 C_CLEAR 事件步的 room 列重算 s_meas ≥ 0.02；
     - 以前 5 步與該步重算兩配對 d、d_pred（door_k3b_pair_geom 同一預測規則）≥ 5 mm。
  K11 追蹤落後：任務端 trace 每列的 state_step，比 room FK s_meas 與**上一列已發布** s_cmd，|差| ≤ 5 mm。
  K12 一般淨空重算：t_open 起每 5 步以幾何監看同一後端與預測規則（GeometryMonitor）重算 nonfinger／finger／self／base ⇒ 不得 stop。
  K13 終點保持重算：RETREAT_DONE～COMPLETE 間 room FK 逐步 s_meas ≥ 0.095、0.5 s 滑動窗位移 ≤ 1 mm、時長 ≥ 3 s、步連續。
  X1 固定登錄：
     - 任務端 monitor reg、執行端 release reg＝release3.REG；執行端 w2.config 參數＝W2c2 REG；
     - 求解端 args w2_retreat／w2_held True、w2_held_max_s 60；任務端 args w2_release True。
  X2 趟別：
     - task args run_id＝w2_mode_run_id＝RUN；執行端 config 三個 run_id＝RUN；求解端 args run_id／w2_mode_run_id＝RUN；
     - 執行端登記請求恰 3 筆（req_seq 1、2、3，皆 pending、內容為本趟 PAUSE_HELD→RELEASE_PREP_TEST→RELEASE_OPEN→RETREAT）、轉送恰 3 次。
  X3 事件時間（CLOCK_TOL 0.05 s、DELIVER_MAX 0.5 s）：
     - 每段：任務端請求 ≤ 執行端登記 ≤ 請求＋0.5；登記 ≤ 提交 ≤ 登記＋1.0；任務端得知提交於 [提交−tol, 提交＋1.5]；
     - 5 N：C_CLEAR ≤ 任務端請求 ≤ C_CLEAR＋0.1；執行端切換 ∈ [請求−tol, 請求＋0.5＋tol]；
     - COMPLETE ≤ room 結束；rel done ≥ COMPLETE。
  錨點證據（只記錄）：t_open 實測底盤、/park/gate 停位錨點、求解端暫停保持錨點，及兩兩差值。

    python3 evaluation/door_k3m_check_k3b2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_k3b2.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_release3 as KR                         # noqa: E402
import door_k3m_check_k3b as J1                        # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402
import door_k3m_check_w2c2 as W2C2                     # noqa: E402

R = KR.REG
K3B_REPLACED = J1.K3B_REPLACED
CLOCK_TOL, DELIVER_MAX = 0.05, 0.5
_fin = J1._fin
FINGERS = J1.FINGERS


def _fk_cache():
    import door_k2_check as KC
    return KC


def check_room_coverage(room, cs, why, st):
    S, I = J1._room(room)
    k1, k_open = cs[1]['step'], cs[2]['step']
    seg = [s for s in S if s[I['step']] >= k1]
    st_ = [s[I['step']] for s in seg]
    if not seg or any(b - a != 1 for a, b in zip(st_, st_[1:])):
        why.append('K0_room_steps_not_contiguous')
    bad = 0
    for s in seg:
        if s[I['step']] < k_open:
            continue
        fq, lc, wc = s[I['q_finger']], s[I['leaf_contacts']], s[I['wall_contacts']]
        ok = isinstance(fq, list) and len(fq) == 2 and all(_fin(x) for x in fq) and isinstance(lc, dict) and isinstance(wc, dict)
        if ok:
            for n in FINGERS:
                v = lc.get(n)
                if not isinstance(v, dict) or not all(_fin(v.get(k)) for k in ('bar', 'ambiguous', 'panel')):
                    ok = False
        bad += 0 if ok else 1
    if bad:
        why.append(f'K0_contact_or_finger_source_missing_{bad}')
    st['k3b2_coverage'] = {'n_rows': len(seg), 'n_bad_source': bad}


def check_effort2(room, rw, task, cs, t_conf, why, st):
    eff = rw.get('effort') or {}
    rr = eff.get('restore') or {}
    mon = ((task or {}).get('k3b') or {}).get('monitor') or {}
    t_clear = (mon.get('t') or {}).get('C_CLEAR')
    rb = rr.get('readback')
    if rr.get('ok') is not True or not (_fin(rr.get('from')) and abs(rr['from'] - 20.0) < 1e-6 and _fin(rr.get('to')) and abs(rr['to'] - 5.0) < 1e-6) \
            or not (isinstance(rb, list) and len(rb) == 2 and all(_fin(x) and abs(x - 5.0) < 1e-6 for x in rb)):
        why.append('K4_restore_record_or_readback_not_5')
        return
    t_sw = rr.get('t')
    if not _fin(t_clear) or not _fin(t_sw) or t_sw + 1e-9 < t_clear or rr.get('step', -1) < cs[3]['step']:
        why.append('K4_restore_before_clear_or_retreat')
    if t_conf is None or t_sw - t_conf < R['restore_quiet_s'] - 1e-6:
        why.append('K4_restore_quiet_window')
    S, I = J1._room(room)
    t_end = S[-1][I['sim_t']] if S else None
    per = [p for p in (((room or {}).get('finger_effort') or {}).get('pause') or {}).get('periodic') or [] if _fin(p.get('t')) and p['t'] > t_sw]
    need = int(math.floor((t_end - t_sw) / 1.0)) - 1 if _fin(t_end) else 1
    ts = [p['t'] for p in per]
    if len(per) < max(need, 1) or any(b <= a for a, b in zip(ts, ts[1:])) \
            or any(p.get('ok') is not True or not (isinstance(p.get('readback'), list) and all(_fin(x) and abs(x - 5.0) < 1e-6 for x in p['readback']))
                   for p in per):
        why.append(f'K4_periodic_coverage_or_readback_{len(per)}_need_{need}')
    st['k3b2_effort'] = {'t_switch': t_sw, 't_clear': t_clear, 'readback': rb, 'n_periodic_after': len(per), 'need': need}


def _smeas(KC, Hr, E_rel, s, I):
    E = np.linalg.inv(Hr) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
    sm = float((E[:3, 3] - E_rel[:3, 3]) @ KR.A_H)
    return E, sm


def check_geom_steps(room, cs, why, st):
    KC = _fk_cache()
    S, I = J1._room(room)
    k_open = cs[2]['step']
    idx0 = next((i for i, s in enumerate(S) if s[I['step']] == k_open), None)
    if idx0 is None:
        why.append('K5_no_open_row')
        return None
    s0 = S[idx0]
    q_rel = float(s0[I['door_q_rad']])
    Hr = KC._H_of(q_rel)
    E_rel = np.linalg.inv(Hr) @ KC._tcp(s0[I['base_xyth']], s0[I['q_arm_meas']])
    smap, bad = {}, []
    worst = {'x_mm': 0.0, 'z_mm': 0.0, 'rot_deg': 0.0, 'band_margin_deg': math.inf, 's_max': 0.0}
    qhist = []
    for s in S[idx0:]:
        E, sm = _smeas(KC, Hr, E_rel, s, I)
        smap[s[I['step']]] = (s[I['sim_t']], sm)
        d = E[:3, 3] - (E_rel[:3, 3] + sm * KR.A_H)
        Rm = E[:3, :3] @ E_rel[:3, :3].T
        rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2))))
        q = float(s[I['door_q_rad']])
        qhist.append((s[I['sim_t']], q))
        past = [x for x in qhist if x[0] <= s[I['sim_t']] - 0.1 + 1e-9]
        rate = (q - past[-1][1]) / (s[I['sim_t']] - past[-1][0]) if past else 0.0
        dq = math.degrees(q - q_rel)
        dq_pred = dq + math.degrees(max(rate, 0.0)) * R['t_pred_s']
        up = KR.band_upper_deg(sm)
        worst.update(x_mm=max(worst['x_mm'], abs(d[0]) * 1e3), z_mm=max(worst['z_mm'], abs(d[2]) * 1e3), rot_deg=max(worst['rot_deg'], rot),
                     band_margin_deg=min(worst['band_margin_deg'], up - dq_pred), s_max=max(worst['s_max'], sm))
        if abs(d[0]) > R['x_max_m'] or abs(d[2]) > KR.z_max_m(max(sm, 0.0)) or rot > R['rot_max_deg'] or dq < R['lower_deg'] or dq_pred > up:
            bad.append((s[I['step']], round(sm, 4), round(dq, 4), round(dq_pred, 4)))
    if bad:
        why.append(f'K5_geometry_or_band_pred_{len(bad)}')
    if worst['s_max'] < R['s_done_m'] - 1e-6:
        why.append('K5_retreat_not_reached')
    st['k3b2_geom'] = dict(worst, n_bad=len(bad), first_bad=bad[:5], q_rel_deg=math.degrees(q_rel), stride=1)
    return {'smap': smap, 'Hr': Hr, 'E_rel': E_rel, 'q_rel': q_rel}


def check_exit_recompute(room, task, g, why, st):
    import door_k3b_pair_geom as PG
    S, I = J1._room(room)
    mon = ((task or {}).get('k3b') or {}).get('monitor') or {}
    ev = next((e for e in (mon.get('events') or []) if e.get('to') == 'C_CLEAR'), None)
    k = None if ev is None else ev.get('step')
    by = {s[I['step']]: s for s in S}
    if type(k) is not int or k not in by or (k - 5) not in by or g is None:
        why.append('K10_exit_rows_missing')
        return
    pg = PG.PairGeom()
    out = None
    for kk in (k - 5, k):
        s = by[kk]
        qf = np.array(list(s[I['base_xyth']]) + list(s[I['q_arm_meas']]), float)
        out = pg.step(float(s[I['sim_t']]), qf, float(s[I['door_q_rad']]), s[I['q_finger']], 0.0)
    stt, d, dp, det = out
    sm = g['smap'].get(k, (None, None))[1]
    if stt != 'ok' or min(min(d), min(dp)) < R['pair_exit_m'] or not _fin(sm) or sm < R['near_s_m']:
        why.append('K10_exit_not_clear_recomputed')
    st['k3b2_exit'] = {'step': k, 'pair_d_mm': None if d is None else [x * 1e3 for x in d], 'pair_d_pred_mm': None if dp is None else [x * 1e3 for x in dp],
                       's_meas': sm}


def check_tracking(task, g, why, st):
    tr = [x for x in ((task or {}).get('trace') or []) if isinstance(x.get('k3b'), dict) and type(x.get('state_step')) is int]
    if g is None or not tr:
        why.append('K11_tracking_rows_missing')
        return
    worst, n, bad = 0.0, 0, 0
    for a, b in zip(tr, tr[1:]):
        s_pub = a['k3b'].get('s_cmd')
        hit = g['smap'].get(b['state_step'])
        if hit is None or not _fin(s_pub) or s_pub <= 0.0:
            continue
        e = abs(s_pub - hit[1])
        n += 1
        worst = max(worst, e)
        bad += int(e > R['y_track_max_m'] + 1e-9)
    if n == 0:
        why.append('K11_tracking_rows_missing')
    elif bad:
        why.append(f'K11_tracking_lag_{bad}')
    st['k3b2_tracking'] = {'n': n, 'worst_mm': worst * 1e3, 'n_bad': bad}


def check_general_clearance(room, cs, why, st, stride=5):
    import door_k3m_geom_monitor as GM
    S, I = J1._room(room)
    k_open = cs[2]['step']
    _gc = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))     # 與任務節點同一登錄設定
    gm = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(_gc['v_floor_registered']), ds_err=dict(_gc['downsampling']['ds_err_registered'])),
                            GM.default_backend())
    worst, stops = {}, []
    for s in S:
        if s[I['step']] < k_open or (s[I['step']] - k_open) % stride:
            continue
        qf = [float(x) for x in list(s[I['base_xyth']]) + list(s[I['q_arm_meas']])]
        fq = s[I['q_finger']]
        dec, det = gm.step(float(s[I['sim_t']]), qf, float(s[I['door_q_rad']]), float(sum(fq) / 2.0), 0.0)
        if dec == 'stop':
            stops.append((s[I['step']], det.get('why')))
        for c, v in (det.get('classes') or {}).items():
            worst[c] = min(worst.get(c, math.inf), v['d_pred'])
    if stops:
        why.append(f'K12_general_clearance_{len(stops)}')
    st['k3b2_clearance'] = {'worst_d_pred_mm': {k: v * 1e3 for k, v in worst.items()}, 'n_stop': len(stops), 'first_stop': stops[:3], 'stride': stride}


def check_end_hold(task, g, why, st):
    tt = ((((task or {}).get('k3b') or {}).get('monitor') or {}).get('t') or {})
    a, b = tt.get('RETREAT_DONE'), tt.get('COMPLETE')
    if g is None or not (_fin(a) and _fin(b)):
        why.append('K13_end_hold_missing')
        return
    pts = sorted((t, sm) for t, sm in g['smap'].values() if a - 1e-9 <= t <= b + 1e-9)
    if not pts or pts[-1][0] - pts[0][0] < R['complete_hold_s'] - 0.011 or any(y[0] - x[0] > 0.011 for x, y in zip(pts, pts[1:])):
        why.append('K13_end_hold_coverage')
        return
    if any(sm < R['s_done_m'] - 1e-9 for _, sm in pts):
        why.append('K13_end_hold_left_zone')
    worst = 0.0
    j = 0
    for i in range(len(pts)):
        while pts[i][0] - pts[j][0] > R['converge_win_s'] + 1e-9:
            j += 1
        win = [p[1] for p in pts[j:i + 1]]
        worst = max(worst, max(win) - min(win))
    if worst > R['converge_max_m'] + 1e-9:
        why.append('K13_end_hold_moved')
    st['k3b2_end_hold'] = {'t': [a, b], 'n': len(pts), 'worst_window_move_mm': worst * 1e3}


def check_reg_id_time(task, room, solver, cs, why, st):
    ta = (task or {}).get('args') or {}
    k = (task or {}).get('k3b') or {}
    mon, rel = k.get('monitor') or {}, k.get('rel') or {}
    rw = (room or {}).get('w2') or {}
    rc = rw.get('config') or {}
    sa = (solver or {}).get('args') or {}
    RUN = ta.get('run_id')
    reg_bad = []
    for name, reg in (('task_monitor', mon.get('reg')), ('exec_release', (rw.get('release') or {}).get('reg'))):
        if not isinstance(reg, dict) or any(not (_fin(reg.get(kk)) and abs(reg[kk] - v) < 1e-9) for kk, v in R.items() if isinstance(v, (int, float))):
            reg_bad.append(name)
    for kk in ('cand_timeout_s', 'cmd_max_age_s', 'post_commit_timeout_s', 'applied_timeout_s', 'dt'):
        if not (_fin(rc.get(kk)) and abs(rc[kk] - W2C2.REG[kk]) < 1e-9):
            reg_bad.append(f'room_config.{kk}')
    if sa.get('w2_retreat') is not True or sa.get('w2_held') is not True or not (_fin(sa.get('w2_held_max_s')) and abs(sa['w2_held_max_s'] - 60.0) < 1e-9):
        reg_bad.append('solver_args')
    if ta.get('w2_release') is not True:
        reg_bad.append('task_args.w2_release')
    if reg_bad:
        why.append('X1_registration:' + ','.join(reg_bad[:6]))
    id_bad = []
    if not isinstance(RUN, str) or ta.get('w2_mode_run_id') != RUN:
        id_bad.append('task_run_id')
    if not (rc.get('w2_mode_run_id') == rc.get('run_id') == rc.get('authority_run_id') == RUN):
        id_bad.append('room_config_run_id')
    if not (sa.get('run_id') == sa.get('w2_mode_run_id') == RUN):
        id_bad.append('solver_run_id')
    rt = rw.get('runtime') or {}
    reqs, rlay = rt.get('requests') or [], rt.get('relayed') or []
    names = ['PAUSE_HELD', 'RELEASE_PREP_TEST', 'RELEASE_OPEN', 'RETREAT']
    want = [{'run_id': RUN, 'req_seq': i, 'from_mode': names[i - 1], 'to_mode': names[i], 'mode_seq': i} for i in (1, 2, 3)]
    if [r.get('req') for r in reqs] != want or any(r.get('result') != 'pending' for r in reqs):
        id_bad.append('exec_requests')
    if [r.get('req_seq') for r in rlay] != [1, 2, 3]:
        id_bad.append('exec_relay')
    if id_bad:
        why.append('X2_identity:' + ','.join(id_bad[:6]))
    tm_bad = []
    pend = {e.get('mode_seq'): e.get('t') for e in ((rw.get('auth') or {}).get('events') or []) if e.get('kind') == 'pending'}
    w2t, rt_ = ((task or {}).get('w2') or {}).get('t') or {}, rel.get('t') or {}
    t_req = {1: w2t.get('requested'), 2: rt_.get('req_open'), 3: rt_.get('req_retreat')}
    t_seen = {1: w2t.get('committed'), 2: rt_.get('open_seen'), 3: rt_.get('retreat')}
    for m in (1, 2, 3):
        tr, tp, tc = t_req[m], pend.get(m), cs[m]['t']
        if not all(_fin(x) for x in (tr, tp, tc)):
            tm_bad.append(f'mode{m}_times_missing')
            continue
        if not (tr - CLOCK_TOL <= tp <= tr + DELIVER_MAX):
            tm_bad.append(f'mode{m}_register_window')
        if not (tp <= tc <= tp + W2C2.REG['cand_timeout_s'] + 1e-9):
            tm_bad.append(f'mode{m}_commit_window')
        ts = t_seen[m]
        if m != 2 and (not _fin(ts) or not (tc - CLOCK_TOL <= ts <= tc + W2C2.REG['state_stale_s'])):
            tm_bad.append(f'mode{m}_task_seen_window')
    if not (_fin(rt_.get('open')) and abs(rt_['open'] - cs[2]['t']) <= CLOCK_TOL):
        tm_bad.append('t_open_task_ne_exec')
    t_clear, t_er = (mon.get('t') or {}).get('C_CLEAR'), rt_.get('eff_req')
    t_sw = ((rw.get('effort') or {}).get('restore') or {}).get('t')
    if not all(_fin(x) for x in (t_clear, t_er, t_sw)) or not (t_clear - 1e-9 <= t_er <= t_clear + 0.1 + 1e-9) \
            or not (t_er - CLOCK_TOL <= t_sw <= t_er + 0.5 + CLOCK_TOL):
        tm_bad.append('effort_times')
    S, I = J1._room(room)
    t_end = S[-1][I['sim_t']] if S else None
    t_c = (mon.get('t') or {}).get('COMPLETE')
    if not (_fin(t_c) and _fin(t_end) and t_c <= t_end + 1e-9 and _fin(rt_.get('done')) and rt_['done'] >= t_c - 1e-9):
        tm_bad.append('complete_times')
    if tm_bad:
        why.append('X3_time:' + ','.join(tm_bad[:8]))
    st['k3b2_cross'] = {'reg_bad': reg_bad, 'id_bad': id_bad, 'time_bad': tm_bad, 'run_id': RUN, 'pending': pend, 't_req': t_req}


def anchors(room, gate, solver, cs, st):
    S, I = J1._room(room)
    s0 = next((s for s in S if s[I['step']] == cs[2]['step']), None)
    a_open = None if s0 is None else list(s0[I['base_xyth']])
    a_gate = ((gate or {}).get('gate') or {}).get('anchor')
    ev = (((solver or {}).get('stats') or {}).get('k3m_pause') or {}).get('events') or []
    a_hold = next((e.get('anchor') for e in ev if e.get('stage') == 'hold' and e.get('anchor') is not None), None)

    def dd(a, b):
        if a is None or b is None:
            return None
        return {'d_mm': math.hypot(a[0] - b[0], a[1] - b[1]) * 1e3, 'dyaw_deg': math.degrees(math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2])))}
    st['k3b2_anchors'] = {'t_open_base': a_open, 't_open_step': cs[2]['step'], 'park_gate_anchor': a_gate, 'solver_pause_hold_anchor': a_hold,
                          'open_vs_gate': dd(a_open, a_gate), 'open_vs_solver_hold': dd(a_open, a_hold),
                          'semantics': '釋放監看基準＝t_open 實測底盤；求解端保持錨點不重設；走廊／包絡基準不動'}


def check_k3b2(task, room, gate, solver, st):
    why = []
    rw = (room or {}).get('w2') or {}
    cs = J1.check_multi(rw, why)
    if cs is None:
        return why
    check_room_coverage(room, cs, why, st)
    t_conf = J1.check_exec_release(room, rw, cs, why, st)
    check_effort2(room, rw, task, cs, t_conf, why, st)
    g = check_geom_steps(room, cs, why, st)
    check_exit_recompute(room, task, g, why, st)
    check_tracking(task, g, why, st)
    check_general_clearance(room, cs, why, st)
    check_end_hold(task, g, why, st)
    J1.check_base_contacts(room, gate, cs, why, st)
    J1.check_task(task, why, st)
    J1.check_solver(solver, why, st)
    check_reg_id_time(task, room, solver, cs, why, st)
    anchors(room, gate, solver, cs, st)
    return why


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in K3B_REPLACED]
    why_old = [w for w in why_all if w not in K3B_REPLACED]
    st = dict(res.get('stats') or {})
    why_k = check_k3b2(task, room, gate, solver, st)
    why = why_old + why_k
    return dict(res, why=why, stats=st, judge='k3b2', replaced_old_reasons=replaced, k3b_replaced_names=list(K3B_REPLACED),
                old_pass_excluding_replaced=not why_old, k3b_ok=not why_k, **{'pass': not why},
                semantics='DONE＝解除、退出近區、恢復 5 N、退離終點並保持 ≥ 3 s；不是收臂')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_k3b2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3b_ok', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
