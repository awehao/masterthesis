#!/usr/bin/env python3
"""door_coopen_core_bench_3 分析 r2（door_coopen_core_bench_3_analysis.py 的版本化修正；Codex r25；r1 與其輸出保留）。
r1 缺口（Codex 核對）：相位只取含 'to' 的事件，漏掉中止事件 ⇒ 中止後 308 步被算入 OPEN／開門前；中止前最大門角應為 43.725°（43.968° 是中止後）。
r2：
  1. 相位邊界含 ABORTED（中止事件時刻）；所有開門統計只取 [t_OPEN, t_abort)。報中止前最大門角、底盤**淨位移**與**累積路程**（分開）、偏航範圍。
     出口距離 D 的下降沿用判定器 core2 欄位原意：登錄末段量測的降幅；78% 為有效下降步的輪次占比（不是距離或時間占比）。
  2. 漂移重算（實際夾持基準）：G_T_H＝task.json G_T_H；每物理步 H_pred＝T_tcp·G_T_H（T_tcp 由 room base_xyth＋q_arm_meas，door_k2_check._tcp），
     H_meas＝證據 T_WH；E＝H_meas⁻¹·H_pred ⇒ 把手座標的位置分量 (ex, ey, ez)、範數、姿態角。對齊門角、底盤偏航、q_ref − q（任務 trace，時間最近列）。
     依門角 5° 分箱報平均漂移分量；報漂移首次超過 3、4、5 mm 的時刻與門角；與任務端中止時記錄的 5.10 mm 對帳。
  3. 103.52 s 單步失接觸：前後 ±0.3 s 每步兩指手指—橫桿力；任務 trace 同期接觸窗口（finger_bar_min）；登錄 finger_loss_s 0.2 s／contact_min 0.5 N
     ⇒ 解釋為何任務端未以 finger_contact_lost 中止，而在 105.32 s 以 grasp_drift 中止（只描述，不預設保護失效）。
  4. 理由分類：真實（grasp_drift、grasp_finger_contact_below_min、geometry_evidence_missing＝96.40 s 時效 0.20378 s > 0.2 s）、
     凍結期（state_step_repeat_not_same_state_1：step 0 settle_sig why 由 short_window 變 stale）、中止連帶（未到 DONE 的缺證據）、
     中止後觀察期相位分類（door_angle_pre_open_over_1deg、finger_bar_outside_allowed_phase）；原判定保留不改。

    python3 evaluation/door_coopen_core_bench_3_analysis2.py   # 寫 runs/door_coopen_core_bench_3/analysis/door_coopen_core_bench_3_analysis2.json（不覆寫）
"""
import bisect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_3_analysis2.json')


def rot_angle(R):
    return math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    st = r['steps']
    ev = t['policy_events']
    t_abort = next(e['t'] for e in ev if e.get('abort'))
    ph = [(e['t'], e['to']) for e in ev if 'to' in e] + [(t_abort, 'ABORTED')]
    ph.sort()
    pt = [p[0] for p in ph]

    def phase_at(x):
        i = bisect.bisect_right(pt, x + 1e-9) - 1
        return ph[i][1] if i >= 0 else None
    T = {p: tt for tt, p in ph}
    op = [s for s in st if T['OPEN'] <= s[c['sim_t']] < t_abort]
    out = {'phases': ph}
    # 1. 門角／底盤
    qd = [math.degrees(s[c['door_q_rad']]) for s in op]
    B = np.array([s[c['base_xyth']] for s in op], float)
    path = float(np.sum(np.hypot(np.diff(B[:, 0]), np.diff(B[:, 1]))))
    yaw = np.unwrap(B[:, 2])
    out['open_pre_abort'] = {'t': [T['OPEN'], t_abort], 'q_door_max_deg': max(qd), 'q_door_end_deg': qd[-1],
                             'base_net_disp_m': float(math.hypot(B[-1, 0] - B[0, 0], B[-1, 1] - B[0, 1])), 'base_path_m': path,
                             'yaw_start_deg': math.degrees(yaw[0]), 'yaw_min_rel_deg': math.degrees(float(yaw.min() - yaw[0])),
                             'yaw_max_rel_deg': math.degrees(float(yaw.max() - yaw[0])), 'yaw_end_rel_deg': math.degrees(float(yaw[-1] - yaw[0]))}
    post = [s for s in st if s[c['sim_t']] >= t_abort]
    out['post_abort'] = {'n_steps': len(post), 'q_door_max_deg': max(math.degrees(s[c['door_q_rad']]) for s in post) if post else None}
    j = json.load(open(os.path.join(RUN, 'analysis', 'door_coopen_check_core11.json')))
    out['exit_progress_judge'] = dict((j.get('items') or {}).get('exit_progress') or {},
                                      interpretation='drop_m＝登錄末段量測的出口距離降幅；share＝有效下降步的輪次占比（非距離、非時間占比）')
    # 2. 漂移重算
    GTH = np.asarray(t['G_T_H'], float)
    evd = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('T_WH'), list) and len(e['T_WH']) == 16:
            evd[e['step']] = np.asarray(e['T_WH'], float).reshape(4, 4)
    trow = sorted(t['trace'], key=lambda x: x['t'])
    tts = [x['t'] for x in trow]
    rows = []
    for s in op:
        k = s[c['step']]
        if k not in evd:
            continue
        Tt = KC._tcp(list(s[c['base_xyth']]), list(s[c['q_arm_meas']]))
        E = np.linalg.inv(evd[k]) @ (Tt @ GTH)
        i = max(0, bisect.bisect_right(tts, s[c['sim_t']]) - 1)
        qr = trow[i].get('q_ref')
        rows.append({'t': s[c['sim_t']], 'q_deg': math.degrees(s[c['door_q_rad']]), 'yaw': float(s[c['base_xyth']][2]),
                     'e_mm': (E[:3, 3] * 1e3).tolist(), 'd_mm': float(np.linalg.norm(E[:3, 3])) * 1e3, 'rot_rad': rot_angle(E[:3, :3]),
                     'lag_deg': None if qr is None else math.degrees(qr) - math.degrees(s[c['door_q_rad']])})
    bins = []
    for lo in range(0, 45, 5):
        w = [x for x in rows if lo <= x['q_deg'] < lo + 5]
        if w:
            bins.append({'q_bin_deg': [lo, lo + 5], 'n': len(w), 'e_mean_mm': np.mean([x['e_mm'] for x in w], axis=0).round(3).tolist(),
                         'd_mean_mm': round(float(np.mean([x['d_mm'] for x in w])), 3), 'd_max_mm': round(max(x['d_mm'] for x in w), 3),
                         'rot_mean_rad': round(float(np.mean([x['rot_rad'] for x in w])), 5),
                         'lag_mean_deg': None if any(x['lag_deg'] is None for x in w) else round(float(np.mean([x['lag_deg'] for x in w])), 3)})
    first = {}
    for thr in (3.0, 4.0, 5.0):
        x = next((x for x in rows if x['d_mm'] > thr), None)
        first[f'{thr:.0f}mm'] = None if x is None else {'t': x['t'], 'q_deg': round(x['q_deg'], 3)}
    last = rows[-1] if rows else None
    out['drift_recompute'] = {'n': len(rows), 'bins_by_door_angle': bins, 'first_exceed': first, 'last_pre_abort': last,
                              'task_abort_record': next(e for e in ev if e.get('abort')),
                              'note': 'E＝H_meas⁻¹·(T_tcp·G_T_H)；位置分量在把手座標；任務端 grasp_drift 以位置範數＞5 mm 判'}
    # 3. 103.52 s 單步失接觸
    win = [(s[c['sim_t']], s[c['finger_contact_n']]) for s in st if 103.2 <= s[c['sim_t']] <= 103.85]
    tw = [{'t': x['t'], 'finger_bar_min_n': (x.get('contacts') or {}).get('finger_bar_min_n'), 'err_p': x.get('err_p')}
          for x in trow if 103.2 <= x['t'] <= 103.85]
    below = [x for x in st if T['HOLD'] <= x[c['sim_t']] < t_abort and isinstance(x[c['finger_contact_n']], list) and min(x[c['finger_contact_n']]) < 0.5]
    runs, cur = [], None
    for x in below:
        if cur and abs(x[c['sim_t']] - cur[1] - 0.01) < 1e-6:
            cur[1] = x[c['sim_t']]
        else:
            if cur:
                runs.append(cur)
            cur = [x[c['sim_t']], x[c['sim_t']]]
    if cur:
        runs.append(cur)
    cfg = t['config']
    out['contact_loss'] = {'room_per_step': win, 'task_windows': tw, 'below_0p5_runs': [{'t0': a, 't1': b, 'dur_s': round(b - a + 0.01, 3)} for a, b in runs],
                           'rule': {'finger_loss_s': cfg.get('finger_loss_s'), 'contact_min_n': cfg.get('contact_min_n'), 'drift_abort_m': cfg.get('drift_abort_m')},
                           'note': '單步低於 contact_min 未達 finger_loss_s 連續門檻 ⇒ 依登錄規則不觸發 finger_contact_lost；中止由 grasp_drift 觸發'}
    # 4. 理由分類
    why = j.get('why') or []
    cls = {'real': ['aborted:grasp_drift', 'grasp_drift_over_limit', 'grasp_finger_contact_below_min', 'geometry_evidence_missing'],
           'frozen_period': ['state_step_repeat_not_same_state_1'],
           'post_abort_phase_classification': ['door_angle_pre_open_over_1deg', 'finger_bar_outside_allowed_phase'],
           'termination_consequence': ['end_hold_too_short_or_missing', 'termination:ABORTED', 'record_does_not_cover_task_window', 'settle_unverifiable',
                                       'reference_not_reached', 'trace_unverifiable', 'settle_p2_unverifiable', 'pause_evidence_missing', 'finger_effort_pause_missing']}
    seen = {w for v in cls.values() for w in v}
    cls['phase_sequence'] = [w for w in why if w.startswith('phase_sequence')]
    cls['unclassified'] = [w for w in why if w not in seen and not w.startswith('phase_sequence')]
    out['reason_classes'] = cls
    out['reason_notes'] = {'geometry_evidence_missing': 'Codex 核對：OPEN 96.40 s 資料年齡 0.18068 s＋計算 0.02310 s＝0.20378 s > 0.2 s（真實時效不合格）',
                           'state_step_repeat_not_same_state_1': 'Codex 核對：凍結期 step 0，同來源步 settle_sig.why 由 short_window 變 stale',
                           'post_abort': f'中止後觀察期 {len(post)} 步（門 44°、夾爪依閂鎖保持閉合）被判為非開門／非允許相位；原判定保留'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('開門（中止前）', json.dumps(out['open_pre_abort'], ensure_ascii=False))
    print('漂移首次超過', first, '中止前末筆', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (last or {}).items() if k != 'e_mm'}, 'e_mm', [round(v, 2) for v in last['e_mm']])
    for b in bins:
        print('  門角', b['q_bin_deg'], 'n', b['n'], 'e_mean_mm', b['e_mean_mm'], 'd_mean', b['d_mean_mm'], 'd_max', b['d_max_mm'], 'rot', b['rot_mean_rad'], 'lag', b['lag_mean_deg'])
    print('低於 0.5 N 的連續段', out['contact_loss']['below_0p5_runs'][:10])
    print('103.2–103.85 每步', [(round(a, 2), b) for a, b in win if isinstance(b, list) and min(b) < 1.0][:12])
    print('任務窗口', tw[:6])
    print('未分類理由', cls['unclassified'])


if __name__ == '__main__':
    main()
