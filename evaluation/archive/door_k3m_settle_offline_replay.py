#!/usr/bin/env python3
"""以既有一趟的紀錄離線重播 p2 SETTLE 完成訊號（Codex 20261024_101421 前置第 1 項；只讀，不開模擬）。

同步流：room_run.json 逐物理步的 (sim_t, door_q_rad)，只取任務節點實際收到的步號節奏（偶數步，50 Hz；
任務 trace 各列的步號全為偶數、且門角與房間逐步紀錄相差 ≤ 1e-8，見輸出 check_trace_vs_room）。
任務迴圈：task.json trace 中相位為 SETTLE 的每一列（實際的評估時刻與步號）。
每列重算：窗口值（SettleRateWindow；牆鐘取模擬時間，因此 stale 在離線無法重現，另註明）、
角度子條件、兩指接觸子條件（trace contacts 的 finger_bar_min_n）；抓取漂移子條件在未中止時恆成立
（漂移超過同一門檻即 grasp_drift 中止），不另重算。連續計時規則與 K3MP2Policy 相同。
輸出：每列窗口（兩端步號、門角、時間、rate、why）、子條件、連續時間、重設原因，以及「首次連續滿 1 s」的時刻。
這只描述候選觸發條件在該趟資料上的行為，不代表改後軌跡或暫停會成功。

    python3 evaluation/door_k3m_settle_offline_replay.py --run evaluation/runs/door_k3m_pause_1 --out <json>
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_settle_sig as SS          # noqa: E402
import door_k3m_policy_p as PP            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    room = json.load(open(os.path.join(a.run, 'room_run.json')))
    task = json.load(open(os.path.join(a.run, 'task.json')))
    cols = room['steps_cols']
    i_st, i_t, i_q = cols.index('step'), cols.index('sim_t'), cols.index('door_q_rad')
    steps = {int(r[i_st]): (float(r[i_t]), float(r[i_q])) for r in room['steps']}
    tr = task['trace']
    dq = [abs(steps[int(x['step'])][1] - x['q_door']) for x in tr if int(x['step']) in steps]
    parity = sorted({int(x['step']) % 2 for x in tr})
    cfg = PP.K3MPConfig()
    goal = math.radians(cfg.theta_goal_deg)
    rows = [x for x in tr if x['phase'] == 'SETTLE']
    out = {'run': os.path.relpath(a.run, os.path.dirname(HERE)), 'source': 'room_run.json steps（偶數步）＋task.json trace（SETTLE 列）',
           'check_trace_vs_room': {'n': len(dq), 'max_abs_dq': max(dq) if dq else None, 'trace_step_parity': parity},
           'config': {'settle_rate': cfg.settle_rate, 'settle_tol_deg': cfg.settle_tol_deg, 'settle_hold_s': cfg.settle_hold_s,
                      'contact_min_n': cfg.contact_min_n, 'window': vars(SS.SettleRateWindow()) | {'buf': None, 'anomalies': None}},
           'notes': ['stale（牆鐘停更）離線無法重現：牆鐘以模擬時間代入。',
                     '抓取漂移子條件：未中止期間恆成立（同一門檻即中止），不另重算。'],
           'rows': [], 'resets': [], 'first_complete_t': None}
    if not rows:
        out['why'] = 'no_settle_rows'
    else:
        w = SS.SettleRateWindow()
        s_end = int(rows[-1]['step'])
        feed = sorted(s for s in steps if s % 2 == 0 and s <= s_end)          # 以步號截止（房間與 trace 時間差 ~1e-6）
        k = 0
        ok_t0 = None
        for x in rows:
            st = int(x['step'])
            while k < len(feed) and feed[k] <= st:
                s = feed[k]
                w.on_sync(steps[s][0], s, steps[s][1], steps[s][0])
                k += 1
            sig = w.value(st, x['t'])
            fb = (x.get('contacts') or {}).get('finger_bar_min_n')
            angle_ok = abs(x['q_door'] - goal) <= math.radians(cfg.settle_tol_deg)
            rate_ok = bool(sig['valid'] and abs(sig['rate']) <= cfg.settle_rate)
            contact_ok = fb is not None and min(fb) >= cfg.contact_min_n
            ok = angle_ok and rate_ok and contact_ok
            if ok:
                ok_t0 = ok_t0 if ok_t0 is not None else x['t']
            else:
                if ok_t0 is not None:
                    why = ('sig_invalid:' + str(sig['why'])) if not sig['valid'] else 'rate' if not rate_ok else 'angle' if not angle_ok else 'contact'
                    out['resets'].append({'t': x['t'], 'why': why, 'held_s': x['t'] - ok_t0})
                ok_t0 = None
            held = 0.0 if ok_t0 is None else x['t'] - ok_t0
            if ok and held >= cfg.settle_hold_s and out['first_complete_t'] is None:
                out['first_complete_t'] = x['t']
                out['first_complete_ok_t0'] = ok_t0
            out['rows'].append({'t': x['t'], 'step': st, 'q_deg': math.degrees(x['q_door']), 'qdot_field': x['qdot_door'],
                                'sig': sig, 'angle_ok': angle_ok, 'rate_ok': rate_ok, 'contact_ok': contact_ok,
                                'finger_bar_min_n': fb, 'held_s': held})
        n = len(out['rows'])
        out['summary'] = {'n_rows': n, 't_settle': [rows[0]['t'], rows[-1]['t']],
                          'n_sig_valid': sum(r['sig']['valid'] for r in out['rows']),
                          'sig_why': {wy: sum(r['sig']['why'] == wy for r in out['rows']) for wy in sorted({r['sig']['why'] for r in out['rows']})},
                          'n_rate_ok': sum(r['rate_ok'] for r in out['rows']),
                          'n_field_le_rate': sum(abs(r['qdot_field']) <= cfg.settle_rate for r in out['rows']),
                          'n_angle_ok': sum(r['angle_ok'] for r in out['rows']), 'n_contact_ok': sum(r['contact_ok'] for r in out['rows']),
                          'n_resets': len(out['resets']), 'first_complete_t': out['first_complete_t'],
                          'window_anomalies': dict(w.anomalies)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out.get('summary', out.get('why')), ensure_ascii=False, indent=1))
    print('resets:', out['resets'][:10])
    return 0


if __name__ == '__main__':
    sys.exit(main())
