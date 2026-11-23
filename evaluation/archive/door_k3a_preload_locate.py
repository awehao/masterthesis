#!/usr/bin/env python3
"""HOLD 切向預載的來源定位（只讀既有取證資料；依 Codex 20261022_120507）。

逐物理步（ENGAGE 前 1 s → 首次單指 < 0.5 N）對齊：
  - 門角 q、q̇（room）；
  - 每指 Σ fn·n 與 Σ ft（把手座標：x 沿桿、y 門扇法向（+y 指向門扇內）、z 鉛直），兩指合計；
  - 抓取相對位姿 E＝H(q)⁻¹·T_tcp·G_T_H_grasp（平移，mm）；
  - 已發布目標（task trace，依來源時間）與求解器 T_cyc 相對 FK 的差（把手座標，mm；取 ≤ 該步的最後一筆）。
作用側核對（物理一致性，**推論而非 SDK 文件**）：
  (a) 夾合只能是壓力：若 fn·n 是「作用在門扇上的力」，finger1（橫桿上方）的 z 分量應 < 0、finger2（下方）應 > 0；
  (b) 開門（q 增大）時門扇須受朝機器人方向（把手 −y）的力；若 ft 也是作用在門扇上，Σft_y 由正轉負的時刻應與門開始轉動相近。
兩項皆只作一致性檢驗；不一致則保留「作用側未核」。
r2（Codex 20261022_121137）：
  - 合量只在**每指都讀取成功**時計算；任一指資料缺失 ⇒ 合量記 NA，並保存每指有效性（valid／no_contact／missing）。讀取成功但無接觸點＝確定無接觸（0）。
  - 14.96 s 類的「首次」若等於分析窗第一步，標為「分析窗起點已超過門檻，起始時刻未定位」；其餘稱「窗內首次超過」，不稱物理起始。
  - trace.t＝任務所用狀態的來源時間（非發布／接收時間）；snap_sim_t_exact＝求解快照時間；兩者不還原命令生效先後。
  - 法向壓力方向支持「作用在門扇上」；**切向作用側仍是推論**。不以 y 分量代替完整鉸鏈力矩。輸出 *_r2，原輸出保留。

    python3 evaluation/door_k3a_preload_locate.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_preload_locate_r2.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402

FINGERS = ('uflite_finger1', 'uflite_finger2')


def finger_sums(x, R):
    """每指 Σfn·n、Σft（把手座標）與有效性；合量只在每指都讀取成功時計算，否則 NA（不合成零）。"""
    out = {}
    con_ok = x.get('contacts') is not None and not x.get('contacts_err')
    fri_ok = x.get('friction') is not None and not x.get('friction_err')
    tot_n, tot_t, all_ok = np.zeros(3), np.zeros(3), True
    for b in FINGERS:
        d = {}
        if not con_ok:
            d['contacts_validity'], d['fn_n_handle'] = 'missing', None
        else:
            raw = x['contacts'].get(b, [])
            cl = [c for c in raw if c.get('fn') is not None and c.get('n') is not None and None not in c['n']]
            if len(cl) < len(raw):
                d['contacts_validity'], d['fn_n_handle'] = 'missing', None
            elif not cl:
                d['contacts_validity'], d['fn_n_handle'] = 'no_contact', [0.0, 0.0, 0.0]
            else:
                d['contacts_validity'] = 'valid'
                d['fn_n_handle'] = (R.T @ np.sum([c['fn'] * np.array(c['n']) for c in cl], axis=0)).tolist()
        if not fri_ok:
            d['friction_validity'], d['ft_handle'] = 'missing', None
        else:
            raw = x['friction'].get(b, [])
            fl = [f['ft'] for f in raw if f.get('ft') is not None and None not in f['ft']]
            if len(fl) < len(raw):
                d['friction_validity'], d['ft_handle'] = 'missing', None
            elif not fl:
                d['friction_validity'], d['ft_handle'] = 'no_contact', [0.0, 0.0, 0.0]
            else:
                d['friction_validity'] = 'valid'
                d['ft_handle'] = (R.T @ np.sum(np.array(fl), axis=0)).tolist()
        if d['fn_n_handle'] is None or d['ft_handle'] is None:
            all_ok = False
        else:
            tot_n += np.array(d['fn_n_handle'])
            tot_t += np.array(d['ft_handle'])
        out[b] = d
    out['sum_fn_n_handle'] = tot_n.tolist() if all_ok else None
    out['sum_ft_handle'] = tot_t.tolist() if all_ok else None
    out['sums_valid'] = all_ok
    return out


def main():
    run = sys.argv[1]
    out = os.path.join(run, 'analysis', 'door_k3a_preload_locate_r2.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    ph_t = {e['to']: e['t'] for e in ev}
    t_eng, t_hold, t_open = ph_t['ENGAGE'], ph_t['HOLD'], ph_t['OPEN']
    G = np.array(t['G_T_H'])
    room = {s[I['step']]: s for s in r['steps']}
    t_low = next(s[I['sim_t']] for s in r['steps'] if s[I['sim_t']] >= t_open
                 and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5)
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    sv = [x for x in a['log'] if x.get('solve_in')]
    sv_t = np.array([x['solve_in']['snap_sim_t_exact'] for x in sv])
    rows = []
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        x = json.loads(ln)
        tt = x['t']
        if tt is None or tt < t_eng - 1.0 or tt > t_low + 0.3:
            continue
        s = room.get(x['step'])
        if s is None:
            continue
        T_WH = np.array(x['T_WH']).reshape(4, 4)
        R = T_WH[:3, :3]
        row = {'t': tt, 'q_deg': math.degrees(s[I['door_q_rad']]), 'qdot': s[I['door_qdot_rad_s']]}
        row.update(finger_sums(x, R))
        H = KC._H_of(s[I['door_q_rad']])
        Tm = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        row['grasp_off_mm'] = ((np.linalg.inv(H) @ Tm @ G)[:3, 3] * 1e3).tolist()
        j = int(np.searchsorted(tr_t, tt + 1e-9, side='right') - 1)
        if j >= 0:
            Tg = np.array(tr[j]['target']).reshape(4, 4)
            row['trace_tgt_minus_fk_mm'] = (H[:3, :3].T @ (Tg[:3, 3] - Tm[:3, 3]) * 1e3).tolist()
        k = int(np.searchsorted(sv_t, tt + 1e-9, side='right') - 1)
        if k >= 0 and sv[k]['solve_in'].get('T_cyc') is not None:
            Tc = np.array(sv[k]['solve_in']['T_cyc']).reshape(4, 4)
            row['tcyc_minus_fk_mm'] = (H[:3, :3].T @ (Tc[:3, 3] - Tm[:3, 3]) * 1e3).tolist()
        row['phase'] = max([p for p, t0 in ph_t.items() if t0 <= tt], key=lambda p: ph_t[p], default=None)
        rows.append(row)
    # (a) 夾合壓力一致性
    f1z = [rw['uflite_finger1']['fn_n_handle'][2] for rw in rows if rw['uflite_finger1']['fn_n_handle'] is not None
           and abs(rw['uflite_finger1']['fn_n_handle'][2]) > 0.05]
    f2z = [rw['uflite_finger2']['fn_n_handle'][2] for rw in rows if rw['uflite_finger2']['fn_n_handle'] is not None
           and abs(rw['uflite_finger2']['fn_n_handle'][2]) > 0.05]
    check_a = {'finger1_z_negative_frac': (float(np.mean(np.array(f1z) < 0)) if f1z else None), 'n1': len(f1z),
               'finger2_z_positive_frac': (float(np.mean(np.array(f2z) > 0)) if f2z else None), 'n2': len(f2z)}
    # (b) Σft_y 由正轉負 vs 門開始轉動
    op = [rw for rw in rows if rw['t'] >= t_open]
    t_ft_cross = next((rw['t'] for rw in op if rw['sum_ft_handle'] is not None and rw['sum_ft_handle'][1] < 0), None)
    q0 = op[0]['q_deg'] if op else None
    t_move = next((rw['t'] for rw in op if rw['q_deg'] - q0 > 0.01), None)
    t_move_rate = next((rw['t'] for rw in op if rw['qdot'] > 0.005), None)
    # 預載建立時刻
    pre = [rw for rw in rows if rw['t'] < t_open]
    t_build = next((rw['t'] for rw in pre if rw['sum_ft_handle'] is not None and rw['sum_ft_handle'][1] > 2.0), None)
    t_tgt = next((rw['t'] for rw in pre if rw.get('trace_tgt_minus_fk_mm') and rw['trace_tgt_minus_fk_mm'][1] > 0.5), None)

    def win(lo, hi):
        w = [rw for rw in rows if lo <= rw['t'] < hi]
        if not w:
            return None

        def med(f):
            v = [f(rw) for rw in w]
            v = [x for x in v if x is not None]
            return None if not v else np.median(np.array(v, float), axis=0).tolist()
        return {'t0': w[0]['t'], 't1': w[-1]['t'], 'n': len(w), 'q_deg_med': med(lambda rw: rw['q_deg']),
                'sum_ft_handle_med': med(lambda rw: rw['sum_ft_handle']), 'sum_fn_n_handle_med': med(lambda rw: rw['sum_fn_n_handle']),
                'grasp_off_mm_med': med(lambda rw: rw['grasp_off_mm']),
                'trace_tgt_minus_fk_mm_med': med(lambda rw: rw.get('trace_tgt_minus_fk_mm')),
                'tcyc_minus_fk_mm_med': med(lambda rw: rw.get('tcyc_minus_fk_mm'))}
    res = {'run': os.path.basename(os.path.normpath(run)), 'phase_t': ph_t, 't_first_low_phys': t_low,
           'action_side_checks': {'a_clamp_compression': check_a,
                                  'b_pull_reversal': {'t_sum_ft_y_first_negative_after_open': t_ft_cross, 't_door_q_up_0.01deg': t_move,
                                                      't_door_qdot_over_0.005': t_move_rate,
                                                      'note': '時刻相近只表示相容；門角速率越門檻（早）與切向 y 轉負不分先後因果；切向作用側為推論，y 分量不代表完整鉸鏈力矩'}},
           'preload_build': {'t_sum_ft_y_over_2N_first_in_window': t_build,
                             't_trace_target_deeper_than_fk_0.5mm_first_in_window': t_tgt,
                             'window_start_t': rows[0]['t'] if rows else None,
                             'target_deeper_label': ('分析窗起點已超過門檻，起始時刻未定位' if (t_tgt is not None and rows and abs(t_tgt - rows[0]['t']) < 1e-9)
                                                     else '窗內首次超過（非物理起始時刻）'),
                             'n_rows_sums_invalid': sum(1 for rw in rows if not rw['sums_valid'])},
           'time_semantics': 'trace.t＝任務所用狀態的來源時間（非發布／接收時間）；snap_sim_t_exact＝求解快照時間；不還原命令生效先後。',
           'windows': {'pre_engage': win(t_eng - 1.0, t_eng), 'engage': win(t_eng, t_hold), 'hold': win(t_hold, t_open),
                       'open_first_0.5s': win(t_open, t_open + 0.5), 'open_to_low': win(t_open + 0.5, t_low)},
           'note': '把手座標 y：+y 指向門扇內（−y＝朝機器人＝開門方向）。fn·n、ft 的作用側由物理一致性推論，非 SDK 文件。只讀、不判定。',
           'rows_every_5': rows[::5]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('phase_t', 't_first_low_phys', 'action_side_checks', 'preload_build', 'windows')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
