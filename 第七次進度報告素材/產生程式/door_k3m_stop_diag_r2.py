#!/usr/bin/env python3
"""停止後實際行為核對 r2（只讀既有實錄；依 Codex 20261023_225057 第 1 節）。r1（door_k3m_stop_diag.py）與其輸出不改。

r2 的時間基準改為**執行端閂鎖步**（room events 的 k2_abort_latched.step），另保留策略中止時間；分兩段：
  - 段 A「策略決定→執行端閂鎖」：策略中止時間 ≤ t < 閂鎖步（r1 把這段的設定點變化誤歸為凍結後）；
  - 段 B「閂鎖後」：閂鎖步起至實錄末（最多 3.0 s；實錄在閂鎖後約 3.0–3.1 s 結束）。
逐步核對（段 B）：門角／門速、底盤位移與量測本體速度（meas_vb_body）、底盤命令（base_cmd_body）與手臂套用（applied_arm）、
實測關節、設定點（evidence arm_sp；無 evidence ⇒ 未驗證）、接觸（leaf_contacts 每剛體：手指—橫桿＝設計接觸，其餘＝非設計接觸）、
牆接觸、腕力、j6 反作用（evidence；無 ⇒ 未驗證）。
近似幾何沿同一條實測軌跡每 2 步重算，**最小值為取樣最小值**；手指開度取兩指平均（近似）。分類同 r1。
界線：固定底盤 K3-A 系列；不是 K3-M 底盤可動停止的證據；接觸量為模擬器接觸配對；不做根因主張；最終停止狀態未驗證。

    python3 evaluation/door_k3m_stop_diag_r2.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402

WIN_S = 3.0
EVERY = 2
NONDESIGN_N = 0.1
FINGERS = ('uflite_finger1', 'uflite_finger2')


def load(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    ev = {}
    p = os.path.join(run, 'evidence.jsonl')
    if os.path.exists(p):
        for ln in open(p):
            x = json.loads(ln)
            ev[x['step']] = x
    return t, r, ev


def seg_a(st, I, ev):
    """策略決定→執行端閂鎖：設定點、手臂套用、底盤命令的變化。"""
    if not st:
        return {'n_steps': 0}
    sp = [ev[s[I['step']]]['arm_sp'] for s in st if s[I['step']] in ev and ev[s[I['step']]].get('arm_sp') is not None]
    return {'n_steps': len(st), 't0': st[0][I['sim_t']], 't1': st[-1][I['sim_t']],
            'setpoint_change_max_rad': (float(np.abs(np.array(sp) - np.array(sp[0])).max()) if sp else 'unverified'),
            'applied_arm_absmax': float(max(np.abs(s[I['applied_arm']]).max() for s in st if s[I['applied_arm']] is not None)),
            'base_cmd_absmax': float(max(np.abs(s[I['base_cmd_body']]).max() for s in st if s[I['base_cmd_body']] is not None))}


def one(run, fn):
    t, r, ev = load(run)
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ab = next((e for e in t.get('policy_events', []) if 'abort' in e), None)
    L = [e['k2_abort_latched'] for e in r['events'] if 'k2_abort_latched' in e]
    if ab is None or len(L) != 1:
        return {'run': os.path.basename(os.path.normpath(run)), 'status': 'no_abort' if ab is None else f'latch_events={len(L)}'}
    lat = L[0]
    by_step = {int(s[I['step']]): s for s in r['steps']}
    s_lat = by_step[int(lat['step'])]
    t_lat = s_lat[I['sim_t']]
    A = [s for s in r['steps'] if ab['t'] - 1e-9 <= s[I['sim_t']] < t_lat - 1e-9]
    B = [s for s in r['steps'] if t_lat - 1e-9 <= s[I['sim_t']] <= t_lat + WIN_S + 1e-9]
    s0 = B[0]
    b0, q0 = np.array(s0[I['base_xyth']], float), np.array(s0[I['q_arm_meas']], float)
    sp0 = (ev.get(s0[I['step']]) or {}).get('arm_sp')
    cls = {'approx_negative_measured_contact': 0, 'approx_negative_no_measured_contact': 0, 'measured_nondesign_contact_only': 0}
    rows, nondesign = [], {}
    for i, s in enumerate(B):
        lc, wc = s[I['leaf_contacts']], s[I['wall_contacts']]
        nd = None
        if lc is not None:
            nd = {}
            for b, v in lc.items():
                f = sum(v.get(k, 0.0) for k in ('panel', 'ambiguous')) + (0.0 if b in FINGERS else v.get('bar', 0.0))
                nd[b] = f
                nondesign[b] = max(nondesign.get(b, 0.0), f)
        e = ev.get(s[I['step']]) or {}
        j6 = e.get('j6_reaction_wrench')
        vb = s[I['meas_vb_body']]
        row = {'dt': round(s[I['sim_t']] - t_lat, 4), 'step': int(s[I['step']]), 'q_door_deg': math.degrees(s[I['door_q_rad']]),
               'qdot': s[I['door_qdot_rad_s']], 'base_disp_mm': float(np.linalg.norm(np.array(s[I['base_xyth']][:2]) - b0[:2]) * 1e3),
               'base_dyaw_deg': math.degrees(s[I['base_xyth']][2] - b0[2]),
               'meas_v_lin': None if vb is None else math.hypot(vb[0], vb[1]), 'meas_w': None if vb is None else abs(vb[2]),
               'base_cmd_absmax': None if s[I['base_cmd_body']] is None else float(np.abs(s[I['base_cmd_body']]).max()),
               'applied_arm_absmax': None if s[I['applied_arm']] is None else float(np.abs(s[I['applied_arm']]).max()),
               'arm_dq_max_rad': float(np.abs(np.array(s[I['q_arm_meas']]) - q0).max()),
               'sp_change_max_rad': None if (sp0 is None or e.get('arm_sp') is None) else float(np.abs(np.array(e['arm_sp']) - np.array(sp0)).max()),
               'finger_bar': None if lc is None else [((lc.get(f) or {}).get('bar', 0.0)) for f in FINGERS],
               'nondesign_max_n': None if nd is None else max(nd.values() or [0.0]),
               'wall_max_n': None if wc is None else max(wc.values() or [0.0]), 'wrist_n': s[I['wrist_force_n']],
               'j6_force_n': None if j6 is None else float(np.linalg.norm(j6[:3]))}
        if i % EVERY == 0 and s[I['q_finger']] is not None:
            qf = np.array(list(s[I['base_xyth']]) + list(s[I['q_arm_meas']]), float)
            qd, qfg = float(s[I['door_q_rad']]), float(np.mean(s[I['q_finger']]))
            d = fn(qf, qd, qfg)
            parts, _ = GA.parts_clearance(qf, qd, qfg)
            row.update({f'g_{c}': d[c][0] for c in GM.CLASSES}, g_nonfinger_who=d['nonfinger'][1],
                       g_shell_bar=parts['uflite_gripper_link']['min_to_bar_m'], g_finger_bar=min(parts[f]['min_to_bar_m'] for f in FINGERS))
            neg = d['nonfinger'][0] < 0.0
            body = (d['nonfinger'][1] or '').split(':')[0]
            meas = None if nd is None else nd.get(body, 0.0) > NONDESIGN_N
            if neg and meas:
                cls['approx_negative_measured_contact'] += 1
            elif neg and meas is False:
                cls['approx_negative_no_measured_contact'] += 1
            elif not neg and nd is not None and max(nd.values() or [0.0]) > NONDESIGN_N:
                cls['measured_nondesign_contact_only'] += 1
        rows.append(row)

    def mx(k, f=max):
        v = [x[k] for x in rows if x.get(k) is not None]
        return None if not v else f(v)

    def tail(k, s_=0.5):
        a = min(rows, key=lambda x: abs(x['dt'] - (rows[-1]['dt'] - s_)))
        return rows[-1][k] - a[k]
    has_ev = bool(ev)
    return {'run': os.path.basename(os.path.normpath(run)), 'status': 'ok', 'abort': ab.get('abort'),
            'policy_abort_t': ab['t'], 'latch_step': int(lat['step']), 'latch_t': t_lat, 'policy_to_latch_s': t_lat - ab['t'],
            'has_evidence': has_ev, 'segment_A_policy_to_latch': seg_a(A, I, ev),
            'segment_B_after_latch': {
                'n_steps': len(rows), 'covered_s': rows[-1]['dt'],
                'door': {'q_latch_deg': rows[0]['q_door_deg'], 'q_max_deg': mx('q_door_deg'), 'q_end_deg': rows[-1]['q_door_deg'],
                         'qdot_absmax': max(abs(x['qdot']) for x in rows), 'qdot_end': rows[-1]['qdot'],
                         'last_0p5s_change_deg': tail('q_door_deg')},
                'base': {'disp_max_mm': mx('base_disp_mm'), 'disp_end_mm': rows[-1]['base_disp_mm'], 'dyaw_end_deg': rows[-1]['base_dyaw_deg'],
                         'last_0p5s_disp_mm': tail('base_disp_mm'), 'meas_v_lin_max': mx('meas_v_lin'), 'meas_v_lin_end': rows[-1]['meas_v_lin'],
                         'base_cmd_absmax': mx('base_cmd_absmax')},
                'arm': {'applied_absmax': mx('applied_arm_absmax'), 'meas_dq_max_rad': mx('arm_dq_max_rad'),
                        'setpoint_change_max_rad': mx('sp_change_max_rad') if has_ev else 'unverified'},
                'contact': {'finger_bar_max_n': [max(x['finger_bar'][k] for x in rows if x['finger_bar']) for k in range(2)],
                            'nondesign_max_n_by_body': nondesign, 'wall_max_n': mx('wall_max_n')},
                'wrist_max_n': mx('wrist_n'), 'j6_force_max_n': mx('j6_force_n') if has_ev else 'unverified',
                'geometry_sampled_min_m': {k: mx(k, min) for k in ('g_nonfinger', 'g_finger', 'g_base', 'g_self', 'g_shell_bar', 'g_finger_bar')},
                'geometry_sampled_min_who': min((x for x in rows if x.get('g_nonfinger') is not None), key=lambda x: x['g_nonfinger'])['g_nonfinger_who'],
                'classification_counts': cls},
            'rows': rows}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    fn = GM.default_backend()
    res = {'schema': 'door_k3m_stop_diag/2', 'time_base': 'executor_latch_step', 'window_s': WIN_S, 'geom_every_steps': EVERY,
           'nondesign_threshold_n': NONDESIGN_N, 'finger_opening': 'mean_of_two（近似）',
           'runs': [one(r, fn) for r in sys.argv[2:]],
           'note': ('時間基準＝執行端閂鎖步；段 A＝策略決定→閂鎖、段 B＝閂鎖後。幾何為每 2 步取樣最小值。固定底盤 K3-A 系列；'
                    '不是 K3-M 底盤可動停止的證據；不做根因主張；最終停止狀態未驗證。')}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    for R in res['runs']:
        if R['status'] != 'ok':
            print(R['run'], R['status'])
            continue
        A, B = R['segment_A_policy_to_latch'], R['segment_B_after_latch']
        g = B['geometry_sampled_min_m']
        print(f"== {R['run']}：策略 {R['policy_abort_t']:.3f} → 閂鎖 {R['latch_t']:.3f}（步 {R['latch_step']}，+{R['policy_to_latch_s']:.3f} s）；"
              f"段A 設定點 {A.get('setpoint_change_max_rad')}、套用手臂 {A.get('applied_arm_absmax')}")
        print(f"   段B {B['covered_s']:.2f} s：設定點 {B['arm']['setpoint_change_max_rad']}、套用手臂最大 {B['arm']['applied_absmax']}、底盤命令最大 {B['base']['base_cmd_absmax']}；"
              f"門 {B['door']['q_latch_deg']:.3f}→{B['door']['q_end_deg']:.3f}°（末 0.5 s {B['door']['last_0p5s_change_deg']:+.3f}°、末 q̇ {B['door']['qdot_end']:.4f}）；"
              f"底盤 末 {B['base']['disp_end_mm']:.2f} mm（末 0.5 s +{B['base']['last_0p5s_disp_mm']:.2f}、量測線速最大 {B['base']['meas_v_lin_max'] * 1e3:.2f} mm/s）")
        print(f"   接觸 指—桿 {[round(x, 1) for x in B['contact']['finger_bar_max_n']]} N、非設計 {({k: round(v, 3) for k, v in B['contact']['nondesign_max_n_by_body'].items() if v > 0})}、"
              f"牆 {B['contact']['wall_max_n']}、腕 {B['wrist_max_n']:.1f}、j6 {B['j6_force_max_n']}；取樣最小 非手指 {g['g_nonfinger'] * 1e3:.1f}／"
              f"殼—桿 {g['g_shell_bar'] * 1e3:.1f}／指—桿 {g['g_finger_bar'] * 1e3:.1f}／自碰 {g['g_self'] * 1e3:.1f}／底盤 {g['g_base'] * 1e3:.0f} mm；分類 {B['classification_counts']}")


if __name__ == '__main__':
    main()
