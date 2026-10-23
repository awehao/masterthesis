#!/usr/bin/env python3
"""停止後實際行為核對（只讀既有實錄；依 Codex 20261023_223218 第 3 節「最小停止診斷包」）。

對每趟中止的 K3-A 系列實錄（固定底盤、K2 閂鎖停止），以中止步為 0，對齊其後 3.0 s（物理步），同步核對：
  - 門角／門速（room door_q_rad、door_qdot_rad_s）；
  - 底盤位姿（room base_xyth）相對中止步；實測關節（room q_arm_meas）相對中止步；設定點（evidence arm_sp；無 evidence ⇒ 未驗證）是否凍結；
  - 接觸：room leaf_contacts 每剛體 bar／panel／ambiguous 法向力大小和、wall_contacts；手指—橫桿是設計接觸，其餘皆為非設計接觸；
  - 腕部：room wrist_force_n；evidence j6_reaction_wrench（反作用，非馬達輸出；無 evidence ⇒ 未驗證）。
沿**同一條實測軌跡**（實測底盤＋關節＋實測門角＋實測手指開度）每 2 步重算近似幾何（door_k3m_geom_monitor 後端：nonfinger、finger、base、self；
另列夾爪殼—橫桿與手指—橫桿），並分類：
  - approx_negative_measured_contact：近似距離 < 0 且該剛體實測有非設計接觸（> 0.1 N）；
  - approx_negative_no_measured_contact：近似距離 < 0 但實測未見該剛體的非設計接觸 ⇒ 「近似模型顯示不足」，不是實測碰撞；
  - measured_nondesign_contact_only：實測有非設計接觸但近似距離 ≥ 0。
缺資料（room 欄位為 None、無 evidence）⇒ 該項標未驗證，不補零。
界線：固定底盤實錄，停止時底盤不動；K3-M 的底盤剩餘位移不在這些資料內。接觸量測是模擬器接觸配對，不等於真實力感測。

    python3 evaluation/door_k3m_stop_diag.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
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


def one(run, fn):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ab = next((e for e in t.get('policy_events', []) if 'abort' in e), None)
    if ab is None:
        return {'run': os.path.basename(run), 'status': 'no_abort'}
    st = [s for s in r['steps'] if ab['t'] - 1e-9 <= s[I['sim_t']] <= ab['t'] + WIN_S + 1e-9]
    evp = os.path.join(run, 'evidence.jsonl')
    ev = {}
    if os.path.exists(evp):
        for ln in open(evp):
            x = json.loads(ln)
            ev[x['step']] = x
    s0 = st[0]
    b0, q0 = np.array(s0[I['base_xyth']], float), np.array(s0[I['q_arm_meas']], float)
    sp0 = (ev.get(s0[I['step']]) or {}).get('arm_sp')
    rows, cls = [], {'approx_negative_measured_contact': 0, 'approx_negative_no_measured_contact': 0, 'measured_nondesign_contact_only': 0}
    nondesign = {}
    for i, s in enumerate(st):
        lc = s[I['leaf_contacts']]
        wc = s[I['wall_contacts']]
        nd = None
        if lc is not None:
            nd = {}
            for b, v in lc.items():
                f = sum(v.get(k, 0.0) for k in ('panel', 'ambiguous')) + (0.0 if b in FINGERS else v.get('bar', 0.0))
                nd[b] = f
                nondesign[b] = max(nondesign.get(b, 0.0), f)
        e = ev.get(s[I['step']]) or {}
        j6 = e.get('j6_reaction_wrench')
        row = {'dt': round(s[I['sim_t']] - s0[I['sim_t']], 4), 'q_door_deg': math.degrees(s[I['door_q_rad']]), 'qdot': s[I['door_qdot_rad_s']],
               'base_disp_mm': float(np.linalg.norm(np.array(s[I['base_xyth']][:2]) - b0[:2]) * 1e3),
               'arm_dq_max_rad': float(np.abs(np.array(s[I['q_arm_meas']]) - q0).max()),
               'sp_change_max_rad': None if (sp0 is None or e.get('arm_sp') is None) else float(np.abs(np.array(e['arm_sp']) - np.array(sp0)).max()),
               'finger_bar': None if lc is None else [((lc.get(f) or {}).get('bar', 0.0)) for f in FINGERS],
               'nondesign_max_n': None if nd is None else max(nd.values() or [0.0]),
               'wall_max_n': None if wc is None else max(wc.values() or [0.0]), 'wrist_n': s[I['wrist_force_n']],
               'j6_force_n': None if j6 is None else float(np.linalg.norm(j6[:3]))}
        if i % EVERY == 0 and s[I['q_finger']] is not None:
            qf = np.array(list(s[I['base_xyth']]) + list(s[I['q_arm_meas']]), float)
            qd = float(s[I['door_q_rad']])
            qfg = float(np.mean(s[I['q_finger']]))
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
    return {'run': os.path.basename(os.path.normpath(run)), 'status': 'ok', 'abort': ab.get('abort'), 'abort_t': ab['t'],
            'has_evidence': bool(ev), 'n_steps': len(rows),
            'door': {'q_abort_deg': rows[0]['q_door_deg'], 'q_max_deg': mx('q_door_deg'), 'q_end_deg': rows[-1]['q_door_deg'],
                     'qdot_absmax': max(abs(x['qdot']) for x in rows)},
            'base_disp_max_mm': mx('base_disp_mm'), 'arm_dq_max_rad': mx('arm_dq_max_rad'),
            'setpoint_change_max_rad': mx('sp_change_max_rad') if ev else 'unverified',
            'finger_bar_max_n': None if mx('finger_bar') is None else [max(x['finger_bar'][k] for x in rows if x['finger_bar']) for k in range(2)],
            'nondesign_contact_max_n_by_body': nondesign, 'wall_max_n': mx('wall_max_n'), 'wrist_max_n': mx('wrist_n'),
            'j6_force_max_n': mx('j6_force_n') if ev else 'unverified',
            'geometry_min': {k: mx(k, min) for k in ('g_nonfinger', 'g_finger', 'g_base', 'g_self', 'g_shell_bar', 'g_finger_bar')},
            'geometry_min_who': min((x for x in rows if x.get('g_nonfinger') is not None), key=lambda x: x['g_nonfinger'])['g_nonfinger_who'],
            'classification_counts': cls, 'rows': rows}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    fn = GM.default_backend()
    res = {'schema': 'door_k3m_stop_diag/1', 'window_s': WIN_S, 'nondesign_threshold_n': NONDESIGN_N, 'runs': [one(r, fn) for r in sys.argv[2:]],
           'note': '固定底盤 K3-A 系列中止實錄（K2 閂鎖停止）的只讀核對；近似幾何與實測接觸分開分類；缺資料標未驗證。不是 K3-M 底盤可動停止的證據。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    for R in res['runs']:
        if R['status'] != 'ok':
            print(R['run'], R['status'])
            continue
        g = R['geometry_min']
        print(f"== {R['run']}（{R['abort']}，evidence {R['has_evidence']}）門 {R['door']['q_abort_deg']:.3f}→最大 {R['door']['q_max_deg']:.3f}°、|q̇| ≤ {R['door']['qdot_absmax']:.4f}；"
              f"底盤 {R['base_disp_max_mm']:.2f} mm；關節變化 {R['arm_dq_max_rad'] * 1e3:.2f} mrad；設定點變化 {R['setpoint_change_max_rad']}")
        print(f"   實測：指—桿最大 {R['finger_bar_max_n']} N、非設計接觸 {({k: round(v, 3) for k, v in R['nondesign_contact_max_n_by_body'].items()})}、牆 {R['wall_max_n']}、"
              f"腕 {R['wrist_max_n']:.1f} N、j6 {R['j6_force_max_n']}")
        print(f"   近似幾何最小：非手指 {g['g_nonfinger'] * 1e3:.1f}（{R['geometry_min_who']}）、手指對門件 {g['g_finger'] * 1e3:.1f}、底盤 {g['g_base'] * 1e3:.0f}、"
              f"自碰 {g['g_self'] * 1e3:.1f}、殼—桿 {g['g_shell_bar'] * 1e3:.1f}、指—桿 {g['g_finger_bar'] * 1e3:.1f} mm；分類 {R['classification_counts']}")


if __name__ == '__main__':
    main()
