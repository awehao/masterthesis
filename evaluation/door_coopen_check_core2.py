#!/usr/bin/env python3
"""核心段判定器 r2（door_coopen_check_core.py 的版本化修正；Codex 20261027_002013 §1；v1 保留不改）。

**舊鏈判定**（chain_pass）＝v1 的 w1f2 全部判準＋J1 門速重建＋J2／J3 登錄核對，不變。
**核心證據**（core_evidence）與舊鏈判定分開：舊鏈通過不自動等於核心動作驗證通過；本檔不給核心動作「通過」布林，只報證據是否充分與量值。
A-2 量測修正：
  (M1) 門速：以門關節實測角（room_run door_q_rad、sim_t、step）按既有訊號規則（door_k3m_settle_sig.SettleRateWindow：0.1 s 因果窗、
       樣本間隔 > 0.03 s 或過長窗即無效）逐步重算；無效記 None，不當作零速，另計無效筆數。
  (M2) 殘差座標：把手座標位置＝(T_WH⁻¹·T_tcp) 的位置 − T_HG 的位置；姿態角＝((T_WH·T_HG)⁻¹·T_tcp) 的旋轉角。
  (M3) 分段：以任務 trace（state_step → phase）把執行端每步歸到相位；殘差分報 HOLD、OPEN、SETTLE＋PAUSE（接近段不併入）；
       出口同步進展只在 OPEN 區間：首個 OPEN 且門角 ≥ θ−10° 的有效 D 到最後一個 OPEN 有效 D 的變化，以及其間 D 下降步中門速有效且 > 0.05 °/s 的占比
       （分母只計門速有效的下降步）。
  (M4) 夾持狀態報告：進入 HOLD、OPEN 那一步的兩指開度（q_finger）與兩指接觸力（finger_contact_n），以及該步的把手座標殘差。
  (M5) 完整性：缺取證檔、OPEN 期間有效殘差配對為 0、OPEN 期間有效 D 為 0、報告例外 ⇒ core_evidence＝'insufficient'（列理由），不得稱量測可用。

    python3 evaluation/door_coopen_check_core2.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core2.json；不覆寫
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core as V1                    # noqa: E402
import door_k3m_settle_sig as SSIG                     # noqa: E402

SEG = {'HOLD': 'HOLD', 'OPEN': 'OPEN', 'SETTLE': 'SETTLE_PAUSE', 'PAUSE': 'SETTLE_PAUSE'}
RATE_MIN_DPS = 0.05


def phase_map(task):
    tr = [(r.get('state_step'), r.get('phase')) for r in ((task or {}).get('trace') or [])
          if isinstance(r.get('state_step'), int) and isinstance(r.get('phase'), str)]
    tr.sort(key=lambda x: x[0])
    steps = [s for s, _ in tr]

    def at(step):
        i = bisect.bisect_right(steps, step) - 1
        return tr[i][1] if i >= 0 else None
    return at


def door_rates(rows):
    """rows：[(t, step, q_rad)] 依步序。回傳 {step: rate_dps 或 None}、無效數。"""
    w = SSIG.SettleRateWindow()
    out, n_bad = {}, 0
    for t, step, q in rows:
        w.on_sync(t, step, q, t)
        v = w.value(step, t)
        if v['valid']:
            out[step] = math.degrees(v['rate'])
        else:
            out[step] = None
            n_bad += 1
    return out, n_bad


def report_a2(room, evidence_path, prm, task):
    import door_coopen_joint_v2 as J2
    import door_coopen_release_set_v1 as RS
    import door_coopen_wgmpc_v2 as WV
    import door_k2_check as KC
    why = []
    if not evidence_path or not os.path.exists(evidence_path):
        why.append('evidence_missing')
    cols = {c: i for i, c in enumerate(room['steps_cols'])}
    gb = json.load(open(os.path.join(WS, prm['grasp']['basis'])))['args']
    T_HG = np.asarray(gb['T_HG_full'], float)
    theta = float(prm['theta_goal_deg'])
    H0 = KC._H_of(0.0)
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    ev = {}
    if 'evidence_missing' not in why:
        for ln in open(evidence_path):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r.get('step'), int) and isinstance(r.get('T_WH'), list) and len(r['T_WH']) == 16:
                ev[r['step']] = np.asarray(r['T_WH'], float).reshape(4, 4)
    ph_at = phase_map(task)
    rows = []
    for s in room['steps']:
        b, qa, qd, t, st = s[cols['base_xyth']], s[cols['q_arm_meas']], s[cols['door_q_rad']], s[cols['sim_t']], s[cols['step']]
        if isinstance(b, list) and isinstance(qa, list) and V1._fin(qd) and V1._fin(t) and isinstance(st, int):
            rows.append((float(t), int(st), float(qd), b, qa, s))
    rates, n_rate_bad = door_rates([(t, st, qd) for t, st, qd, _, _, _ in rows])
    _, Dtrue = J2.d_map(RS.field(theta))
    seg_res, seg_dq, D_open, entry = {}, [], [], {}
    for t, st, qd, b, qa, s in rows:
        ph = ph_at(st)
        seg = SEG.get(ph)
        T_tcp = KC._tcp(list(b), list(qa))
        H_est = T_tcp @ np.linalg.inv(T_HG)
        d = math.atan2(H_est[1, 3], H_est[0, 3]) - a0
        q_tcp = -math.atan2(math.sin(d), math.cos(d))
        res = None
        if st in ev:
            p_h = (np.linalg.inv(ev[st]) @ T_tcp)[:3, 3] - T_HG[:3, 3]
            Rr = (np.linalg.inv(ev[st] @ T_HG) @ T_tcp)[:3, :3]
            res = (p_h, math.degrees(math.acos(float(np.clip((np.trace(Rr) - 1) / 2, -1, 1)))))
        if seg:
            seg_res.setdefault(seg, []).append(res)
            seg_dq.append((seg, math.degrees(q_tcp - qd)))
        if ph in ('HOLD', 'OPEN') and ph not in entry:
            qf, fc = s[cols['q_finger']], s[cols['finger_contact_n']]
            entry[ph] = {'step': st, 't': t, 'q_finger': qf, 'finger_contact_n': fc,
                         'residual_handle_mm': None if res is None else (res[0] * 1e3).tolist(), 'residual_rot_deg': None if res is None else res[1]}
        if ph == 'OPEN':
            D_open.append((st, math.degrees(qd), WV.D_bilinear(Dtrue, b), rates.get(st)))
    seg_out = {}
    for seg, lst in seg_res.items():
        ok = [r for r in lst if r is not None]
        P = np.array([r[0] for r in ok]) if ok else np.zeros((0, 3))
        seg_out[seg] = {'n_rows': len(lst), 'n_valid_pairs': len(ok),
                        'pos_absmax_mm_handle_xyz': (np.abs(P).max(axis=0) * 1e3).tolist() if len(ok) else None,
                        'pos_norm_max_mm': float(np.linalg.norm(P, axis=1).max() * 1e3) if len(ok) else None,
                        'rot_max_deg': float(max(r[1] for r in ok)) if ok else None,
                        'door_joint_vs_tcp_absmax_deg': float(max(abs(x) for sg, x in seg_dq if sg == seg))}
    validD = [x for x in D_open if x[2] is not None]
    i0 = next((i for i, x in enumerate(validD) if x[1] >= theta - 10), None)
    drops = [(validD[i][3], validD[i - 1][2] - validD[i][2]) for i in range(1, len(validD)) if validD[i - 1][2] - validD[i][2] > 1e-6]
    drops_rate_valid = [r for r, _ in drops if r is not None]
    exit_open = {'n_open_rows': len(D_open), 'n_open_valid_D': len(validD),
                 'D_first_open_ge_theta_minus_10_m': None if i0 is None else validD[i0][2], 'D_last_open_m': validD[-1][2] if validD else None,
                 'drop_m': (validD[i0][2] - validD[-1][2]) if (i0 is not None and validD) else None,
                 'q_door_max_open_deg': max((x[1] for x in D_open), default=None),
                 'n_drop_rows': len(drops), 'n_drop_rows_rate_invalid': len(drops) - len(drops_rate_valid),
                 'drop_rows_while_opening_share_of_rate_valid': (sum(1 for r in drops_rate_valid if r > RATE_MIN_DPS) / len(drops_rate_valid)) if drops_rate_valid else None,
                 'note': '輪次占比（分母只計門速有效的下降步），非進展量占比'}
    if (seg_out.get('OPEN') or {}).get('n_valid_pairs', 0) == 0:
        why.append('open_residual_pairs_zero')
    if not validD:
        why.append('open_exit_D_zero')
    return {'core_evidence': 'insufficient' if why else 'sufficient', 'core_evidence_why': why,
            'door_rate': {'rule': 'door_k3m_settle_sig.SettleRateWindow（0.1 s 因果窗、gap 0.03 s）', 'n_rows': len(rows), 'n_invalid': n_rate_bad},
            'residual_by_segment': seg_out, 'grasp_entry': entry, 'exit_progress_open': exit_open}


def judge(task, room, gate, solver, observer, evidence_path=None):
    chain = V1.judge(task, room, gate, solver, observer, None)
    chain.pop('a2_report', None)
    prm = None
    co = (task or {}).get('coopen') or {}
    if isinstance(co.get('params_path'), str) and os.path.exists(os.path.join(WS, co['params_path'])):
        prm = json.load(open(os.path.join(WS, co['params_path'])))
    try:
        core = report_a2(room, evidence_path, prm, task) if (room and prm) else {'core_evidence': 'insufficient', 'core_evidence_why': ['room_or_params_missing']}
    except Exception as e:
        core = {'core_evidence': 'insufficient', 'core_evidence_why': [f'report_exception:{e!r}']}
    return dict(chain, judge='coopen_core2', chain_pass=chain['pass'], core=core,
                note='chain_pass 只是舊鏈開門／停止判定；核心動作（邊開邊繞）證據見 core，不自動通過')


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'offset_hold_ok',
                                             'finger_effort_ok', 'trace_log_ok', 'coopen_registration_ok', 'why')}, ensure_ascii=False, default=float))
    print(json.dumps({'core': res.get('core')}, ensure_ascii=False, default=float)[:2500])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
