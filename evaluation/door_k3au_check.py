#!/usr/bin/env python3
"""K3-A-U 判定器：K3-A 判準全部照用（door_k3a_check.judge），另加接近軸參考卸載的獨立核對（規格修訂 1 §5）。

卸載核對（task.json 與 room_run.json）：
  - 恰一次擷取事件（unload_capture）；δ 有限且 −3 mm ≤ δ ≤ 0。
  - 重算：T_ref0＝H_frozen·T_HG（≤ 1e-9）；a＝T_ref0 工具 z 單位向量（≤ 1e-12）；δ＝a·(p_tcp − p_ref0)（≤ 1e-12）。
  - 快照：事件步號對應的物理步 FK 與事件 T_tcp 位置差 ≤ 5e-5 m（room 紀錄四捨五入至 1e-6 rad）。
  - 已發布目標（task trace）：擷取前＝T_ref0；擷取起＝p_ref0＋a·δ·σ((t−t_cap)/0.5)（≤ 1e-9 m），姿態＝T_ref0（≤ 1e-12），直到 OPEN。
  - 斜坡結束與保持結束事件存在；OPEN 時刻＝保持結束時刻且 ≥ t_cap＋1.0；HOLD ≥ 1.5 s。
  - 中止於保持結束之前 ⇒ 不得有 OPEN，中止後不得有已發布目標。
r2（Codex 20261022_122517 完整 judge 誤收）：
  - 時間契約：擷取事件外層／內層時間都有限且一致（≤ 1e-9）；擷取落在 HOLD 起 [0.5, 1.5] s（下界 −1e-9、上界＋1e-9：
    相位機在 τ > 1.5 時中止，故 1.5 為硬上界）；unload_ramp_end、unload_hold_end 各恰一次、時間有限；
    斜坡結束 − t_cap ∈ [0.5, 0.5＋Δ]、保持結束 − t_cap ∈ [1.0, 1.0＋Δ]（Δ＝一個控制週期 0.05 s＋1e-6，離散取樣誤差）；
    次序 t_cap < 斜坡結束 ≤ 保持結束＝OPEN。
  - 快照：擷取步號對應物理步的 sim_t 與 t_cap 差 ≤ 1e-4 s（room 的 sim_t 四捨五入至 1e-4）；FK 位置 ≤ 5e-5 m、
    旋轉（Frobenius 最大元素差）≤ 1e-4（room 的 q 四捨五入至 1e-6 rad）；該步門角速率 |q̇| ≤ 0.002 rad/s（＋1e-9）。

    python3 evaluation/door_k3au_check.py <run_dir>    # 寫 <run_dir>/analysis/door_k3au_check.json；結束碼 0＝通過
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_check as KC3                          # noqa: E402
import door_k3au_policy as U                          # noqa: E402

CFG = U.K3AUConfig()


def _M(x):
    return KC._rigid(x)


def check_unload_structure(task, why, st):
    caps = [e['unload_capture'] for e in task.get('policy_events', []) if 'unload_capture' in e]
    st['n_capture'] = len(caps)
    ev_to = {e['to']: e['t'] for e in task.get('policy_events', []) if 'to' in e}
    t_hold, t_open = ev_to.get('HOLD'), ev_to.get('OPEN')
    abort = next((e for e in task.get('policy_events', []) if 'abort' in e), None)
    if len(caps) != 1:
        why.append(f'unload_capture_count:{len(caps)}')
        if t_open is not None:
            why.append('open_without_single_capture')
        return
    c = caps[0]
    T0, Ttcp = _M(c.get('T_ref0')), _M(c.get('T_tcp'))
    try:
        a = np.asarray(c.get('a'), float).reshape(3)
        d = float(c.get('delta'))
    except (TypeError, ValueError):
        why.append('unload_event_invalid')
        return
    if T0 is None or Ttcp is None or not np.isfinite(a).all() or not math.isfinite(d):
        why.append('unload_event_invalid')
        return
    st['delta_m'] = d
    if not (CFG.unload_delta_min_m - 1e-12 <= d <= CFG.unload_delta_max_m + 1e-12):
        why.append('unload_delta_out_of_bounds')
    HF, THG = _M(task.get('H_frozen')), _M((task.get('grasp_basis') or {}).get('T_HG'))
    if HF is None or THG is None or float(np.abs(HF @ THG - T0).max()) > 1e-9:
        why.append('unload_T_ref0_not_hold_reference')
    z = T0[:3, 2] / np.linalg.norm(T0[:3, 2])
    if float(np.abs(z - a).max()) > 1e-12:
        why.append('unload_axis_not_tool_z')
    if abs(float(a @ (Ttcp[:3, 3] - T0[:3, 3])) - d) > 1e-12:
        why.append('unload_delta_not_projection')
    outer = next(e for e in task.get('policy_events', []) if 'unload_capture' in e)
    try:
        t_cap, t_out = float(c.get('t')), float(outer.get('t'))
    except (TypeError, ValueError):
        why.append('unload_capture_time_invalid')
        return
    if not (math.isfinite(t_cap) and math.isfinite(t_out)) or abs(t_cap - t_out) > 1e-9:
        why.append('unload_capture_time_invalid')
        return
    st['t_capture'] = t_cap
    if t_hold is None or not math.isfinite(float(t_hold)):
        why.append('unload_hold_time_missing')
        return
    tau = t_cap - t_hold
    st['capture_tau_s'] = tau
    if tau < CFG.unload_settle_s - 1e-9 or tau > CFG.unload_capture_deadline_s + 1e-9:
        why.append('unload_capture_outside_window')
    ramp_ev = [e for e in task.get('policy_events', []) if 'unload_ramp_end' in e]
    hold_ev = [e for e in task.get('policy_events', []) if 'unload_hold_end' in e]
    marks = ramp_ev + hold_ev
    DT = 0.05 + 1e-6
    t_ramp_end = t_hold_end = None
    for nm, lst in (('ramp', ramp_ev), ('hold', hold_ev)):
        if len(lst) > 1:
            why.append(f'unload_{nm}_end_count:{len(lst)}')
        for e in lst:
            try:
                ok = math.isfinite(float(e.get('t')))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                why.append(f'unload_{nm}_end_time_invalid')
    if len(ramp_ev) == 1 and math.isfinite(float(ramp_ev[0]['t'])):
        t_ramp_end = float(ramp_ev[0]['t'])
        if not (CFG.unload_ramp_s - 1e-9 <= t_ramp_end - t_cap <= CFG.unload_ramp_s + DT):
            why.append('unload_ramp_end_timing')
    if len(hold_ev) == 1 and math.isfinite(float(hold_ev[0]['t'])):
        t_hold_end = float(hold_ev[0]['t'])
        if not (CFG.unload_ramp_s + CFG.unload_hold_s - 1e-9 <= t_hold_end - t_cap <= CFG.unload_ramp_s + CFG.unload_hold_s + DT):
            why.append('unload_hold_end_timing')
    if t_ramp_end is not None and t_hold_end is not None and not (t_cap < t_ramp_end <= t_hold_end):
        why.append('unload_marks_order')
    st['t_ramp_end'], st['t_hold_end'] = t_ramp_end, t_hold_end
    # 已發布目標逐筆
    rows = [e for e in (task.get('trace') or []) if e.get('target') is not None and t_hold is not None and e['t'] >= t_hold - 1e-9
            and (t_open is None or e['t'] <= t_open + 1e-9) and (e.get('q_ref') is None)]
    worst_p, worst_r, n_ramp = 0.0, 0.0, 0
    for e in rows:
        T = np.asarray(e['target'], float).reshape(4, 4)
        if e['t'] < t_cap - 1e-9:
            exp = T0
        else:
            exp = T0.copy()
            exp[:3, 3] = T0[:3, 3] + a * d * U.smoothstep((e['t'] - t_cap) / CFG.unload_ramp_s)
            n_ramp += 1
        worst_p = max(worst_p, float(np.linalg.norm(T[:3, 3] - exp[:3, 3])))
        worst_r = max(worst_r, float(np.abs(T[:3, :3] - T0[:3, :3]).max()))
    st['unload_target_dev_m'], st['unload_target_rot_dev'], st['n_rows_after_capture'] = worst_p, worst_r, n_ramp
    if worst_p > 1e-9 or worst_r > 1e-12:
        why.append('unload_target_path_mismatch')
    if n_ramp == 0:
        why.append('unload_ramp_rows_missing')
    if abort is not None and (t_hold_end is None or abort['t'] < t_hold_end):
        if t_open is not None:
            why.append('open_after_abort_before_hold_end')
        if any(e.get('target') is not None and e['t'] > abort['t'] + 1e-9 for e in (task.get('trace') or [])):
            why.append('targets_after_abort')
        return
    if t_ramp_end is None or t_hold_end is None:
        why.append('unload_marks_missing')
    if t_open is None or t_hold_end is None or abs(t_open - t_hold_end) > 1e-9 or t_open - t_cap < CFG.unload_ramp_s + CFG.unload_hold_s - 1e-9:
        why.append('open_before_unload_complete')
    if t_hold is not None and t_open is not None and t_open - t_hold < CFG.unload_settle_s + CFG.unload_ramp_s + CFG.unload_hold_s - 1e-9:
        why.append('hold_shorter_than_unload')


def check_unload_snapshot(task, room, why, st, tol_p=5e-5, tol_r=1e-4, tol_t=1e-4):
    caps = [e['unload_capture'] for e in task.get('policy_events', []) if 'unload_capture' in e]
    if len(caps) != 1:
        return
    c = caps[0]
    cols = room.get('steps_cols') or []
    I = {k: i for i, k in enumerate(cols)}
    s = next((x for x in room.get('steps', []) if x[I['step']] == c.get('step')), None)
    if s is None:
        why.append('unload_capture_step_missing_in_room')
        return
    Te = KC._rigid(c.get('T_tcp'))
    if Te is None:
        why.append('unload_capture_not_from_snapshot')
        return
    T = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
    dev = float(np.linalg.norm(T[:3, 3] - Te[:3, 3]))
    rdev = float(np.abs(T[:3, :3] - Te[:3, :3]).max())
    st['capture_fk_dev_m'], st['capture_fk_rot_dev'] = dev, rdev
    if not math.isfinite(dev) or dev > tol_p or not math.isfinite(rdev) or rdev > tol_r:
        why.append('unload_capture_not_from_snapshot')
    try:
        dt = abs(float(s[I['sim_t']]) - float(c.get('t')))
    except (TypeError, ValueError, KeyError):
        dt = math.inf
    st['capture_snapshot_dt_s'] = dt
    if not math.isfinite(dt) or dt > tol_t:
        why.append('unload_capture_time_not_snapshot')
    qd = s[I['door_qdot_rad_s']] if 'door_qdot_rad_s' in I else None
    st['capture_qdot'] = qd
    if qd is None or not math.isfinite(float(qd)) or abs(float(qd)) > CFG.settle_rate + 1e-9:
        why.append('unload_capture_qdot_over')


def judge(task, room, gate):
    t2 = dict(task, schema='door_k3a_task/1') if task.get('schema') == 'door_k3au_task/1' else dict(task, schema='mismatch')
    res = KC3.judge(t2, room, gate)
    why = list(res['why'])
    if task.get('schema') != 'door_k3au_task/1':
        why.append('task_schema_k3au')
    st = dict(res.get('stats') or {})
    check_unload_structure(task, why, st)
    check_unload_snapshot(task, room, why, st)
    return {'pass': not why, 'why': why, 'stats': st, 'phase_sequence': res.get('phase_sequence')}


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    gp = os.path.join(run, 'park_gate.json')
    gate = json.load(open(gp)) if os.path.exists(gp) else None
    res = judge(task, room, gate)
    p = os.path.join(run, 'analysis', 'door_k3au_check.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'why', 'stats')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
