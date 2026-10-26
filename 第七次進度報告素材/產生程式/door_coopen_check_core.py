#!/usr/bin/env python3
"""核心段判定器（規格 results/specs/COOPEN_core_spec_draft2.md §4；Codex 20261026_235425）。
door_k3m_check_w1f2 全部判準照用（含 w1e2／w1e／w1b 鏈、trace_gap 0.25 s、紀錄完整性），只改兩處並加一組：
  (J1) 門角參考重建帶入登錄門速：w1b.target_constants 以 K3AConfig(θ, over_goal, **r_max_dps**) 建立（原版缺省 1.0 °/s，與 0.6 °/s 的跑次不符）；
  (J2) 登錄來源：task.json coopen.params_path 的 sha256＝coopen.params_sha256，且與 w1b.params_path 相同（同一登錄檔）；
  (J3) 抓取登錄（通過條件）：coopen.grasp_basis_has_full＝True；coopen.T_HG／a_H＝登錄檔 grasp.basis 檔（核 sha256）的 T_HG_full／a_H；
       coopen.r_max_dps＝config.r_max_dps＝登錄 r_max_dps。
A-2 量測報告（**只報告，不作通過條件**；規格 §1）：
  門關節實測角（room_run door_q_rad）與 TCP 推得門角（FK(base_xyth, q_arm_meas)·T_HG⁻¹ 的把手方位）之差；
  TCP—把手完整相對位姿殘差（取證 T_WH·T_HG 對 FK TCP：把手座標三軸位置與姿態角）；
  出口進展（實測底盤位姿在門角包絡 [θ−2, θ+5]° 淨空場上的雙線性測地距離：首次達 θ−10° 到門停的變化、下降輪次中門關節速率 > 0.05 °/s 的占比）；
  命令分層（執行端套用底盤命令 base_cmd_body 的合速度／每軸／偏航最大值、applied_src 分布；任務端引導開啟筆數）。

    python3 evaluation/door_coopen_check_core.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core.json；不覆寫
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3m_check_w1b as W1B                       # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


@contextlib.contextmanager
def rate_constants(r_max_dps):
    orig = W1B.target_constants

    @contextlib.contextmanager
    def tc(theta_deg, over_goal_deg, envelope):
        saved = (W1B.KC3.CFG, W1B.KC3.TCFG, W1B.KC3.THETA, W1B.KM.ENVELOPE)
        try:
            cfg = W1B.KP.K3AConfig(theta_goal_deg=float(theta_deg), over_goal_deg=float(over_goal_deg), r_max_dps=float(r_max_dps))
            W1B.KC3.CFG, W1B.KC3.TCFG = cfg, cfg.target_cfg()
            W1B.KC3.THETA = W1B.KC3.TCFG.theta_goal_rad
            W1B.KM.ENVELOPE = tuple(float(v) for v in envelope)
            yield
        finally:
            W1B.KC3.CFG, W1B.KC3.TCFG, W1B.KC3.THETA, W1B.KM.ENVELOPE = saved
    W1B.target_constants = tc
    try:
        yield
    finally:
        W1B.target_constants = orig


def check_coopen(task, why, st):
    co = (task or {}).get('coopen') or {}
    p = co.get('params_path')
    if not (isinstance(p, str) and os.path.exists(os.path.join(WS, p))):
        why.append('coopen_params_missing')
        return None
    if _sha(os.path.join(WS, p)) != co.get('params_sha256'):
        why.append('coopen_params_sha_mismatch')
        return None
    if ((task.get('w1b') or {}).get('params_path')) != p:
        why.append('coopen_w1b_params_not_same')
    prm = json.load(open(os.path.join(WS, p)))
    gb_p = os.path.join(WS, prm['grasp']['basis'])
    if not os.path.exists(gb_p) or _sha(gb_p) != prm['grasp']['basis_sha256']:
        why.append('coopen_grasp_basis_missing_or_sha_mismatch')
        return prm
    gb = json.load(open(gb_p))['args']
    if co.get('grasp_basis_has_full') is not True:
        why.append('coopen_grasp_basis_not_full')
    if not (isinstance(co.get('T_HG'), list) and np.allclose(np.asarray(co['T_HG'], float), np.asarray(gb['T_HG_full'], float), atol=1e-12)):
        why.append('coopen_T_HG_mismatch')
    if not (isinstance(co.get('a_H'), list) and np.allclose(np.asarray(co['a_H'], float), np.asarray(gb['a_H'], float), atol=1e-12)):
        why.append('coopen_a_H_mismatch')
    r_reg = float(prm['task_param_extension']['r_max_dps'])
    if not (_fin(co.get('r_max_dps')) and abs(co['r_max_dps'] - r_reg) < 1e-12 and _fin((task.get('config') or {}).get('r_max_dps'))
            and abs(task['config']['r_max_dps'] - r_reg) < 1e-12):
        why.append('coopen_r_max_mismatch')
    st['coopen_registration'] = {'params_path': p, 'r_max_dps': co.get('r_max_dps'), 'a_H': co.get('a_H')}
    return prm


def report_a2(room, evidence_path, prm):
    """A-2 量測報告（不作通過條件）。"""
    import door_coopen_joint_v2 as J2
    import door_coopen_release_set_v1 as RS
    import door_coopen_wgmpc_v2 as WV
    import door_k2_check as KC
    out = {}
    if not room or not prm:
        return {'available': False}
    cols = {c: i for i, c in enumerate(room['steps_cols'])}
    steps = room['steps']
    gb = json.load(open(os.path.join(WS, prm['grasp']['basis'])))['args']
    T_HG = np.asarray(gb['T_HG_full'], float)
    theta = float(prm['theta_goal_deg'])
    H0 = KC._H_of(0.0)
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    ev = {}
    if evidence_path and os.path.exists(evidence_path):
        for ln in open(evidence_path):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r.get('step'), int) and isinstance(r.get('T_WH'), list) and len(r['T_WH']) == 16:
                ev[r['step']] = np.asarray(r['T_WH'], float).reshape(4, 4)
    dq, res_p, res_r, rows = [], [], [], []
    for s in steps:
        b, qa, qd = s[cols['base_xyth']], s[cols['q_arm_meas']], s[cols['door_q_rad']]
        if not (isinstance(b, list) and isinstance(qa, list) and _fin(qd)):
            continue
        T_tcp = KC._tcp(list(b), list(qa))
        H_est = T_tcp @ np.linalg.inv(T_HG)
        d = math.atan2(H_est[1, 3], H_est[0, 3]) - a0
        q_tcp = -math.atan2(math.sin(d), math.cos(d))
        dq.append(math.degrees(q_tcp - qd))
        st_ = int(s[cols['step']])
        if st_ in ev:
            E = np.linalg.inv(ev[st_] @ T_HG) @ T_tcp
            res_p.append(E[:3, 3].tolist())
            res_r.append(math.degrees(math.acos(float(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))))
        rows.append((float(s[cols['sim_t']]), math.degrees(qd), math.degrees(float(s[cols['door_qdot_rad_s']]) if _fin(s[cols['door_qdot_rad_s']]) else 0.0), b))
    out['door_joint_vs_tcp_deg'] = {'n': len(dq), 'absmax': float(np.max(np.abs(dq))) if dq else None,
                                    'p95': float(np.percentile(np.abs(dq), 95)) if dq else None}
    rp = np.asarray(res_p) if res_p else np.zeros((0, 3))
    out['tcp_handle_residual'] = {'n': len(res_p), 'pos_absmax_mm_handle_xyz': (np.abs(rp).max(axis=0) * 1e3).tolist() if len(rp) else None,
                                  'rot_max_deg': float(max(res_r)) if res_r else None,
                                  'note': '含閉爪前接近段；只報告，不反推門檻'}
    _, Dtrue = J2.d_map(RS.field(theta))
    D = [(t, qdeg, qdot, WV.D_bilinear(Dtrue, b)) for t, qdeg, qdot, b in rows]
    i0 = next((i for i, x in enumerate(D) if x[1] >= theta - 10 and x[3] is not None), None)
    i1 = max((i for i, x in enumerate(D) if x[3] is not None), default=None)
    drops = [(D[i][2], D[i - 1][3] - D[i][3]) for i in range(1, len(D)) if D[i][3] is not None and D[i - 1][3] is not None and D[i - 1][3] - D[i][3] > 1e-6]
    out['exit_progress'] = {'D_first_ge_theta_minus_10_m': None if i0 is None else D[i0][3], 'D_last_valid_m': None if i1 is None else D[i1][3],
                            'drop_m': (D[i0][3] - D[i1][3]) if (i0 is not None and i1 is not None) else None,
                            'q_door_max_deg': max((x[1] for x in D), default=None),
                            'drop_rows_while_door_opening_share': (sum(1 for qd_, _ in drops if qd_ > 0.05) / len(drops)) if drops else None,
                            'note': '輪次占比，非進展量占比；D 為門角包絡場的雙線性取值'}
    cmd = [s[cols['base_cmd_body']] for s in steps if isinstance(s[cols['base_cmd_body']], list)]
    src = {}
    for s in steps:
        k = str(s[cols['applied_src']])
        src[k] = src.get(k, 0) + 1
    out['command_layers'] = {'base_cmd_lin_max': float(max(math.hypot(c[0], c[1]) for c in cmd)) if cmd else None,
                             'base_cmd_axis_max': float(max(max(abs(c[0]), abs(c[1])) for c in cmd)) if cmd else None,
                             'base_cmd_yaw_max': float(max(abs(c[2]) for c in cmd)) if cmd else None, 'applied_src_counts': src}
    out['available'] = True
    return out


def judge(task, room, gate, solver, observer, evidence_path=None):
    why_c, st_c = [], {}
    prm = check_coopen(task, why_c, st_c)
    if prm is None:
        return {'pass': False, 'why': why_c, 'stats': st_c, 'judge': 'coopen_core'}
    with rate_constants(prm['task_param_extension']['r_max_dps']):
        res = F2.judge(task, room, gate, solver, observer)
    why = list(res.get('why') or []) + why_c
    st = dict(res.get('stats') or {}, **st_c)
    try:
        a2 = report_a2(room, evidence_path, prm)
    except Exception as e:                                  # 報告失敗不影響判定，但照記
        a2 = {'available': False, 'error': repr(e)}
    return dict(res, why=why, stats=st, judge='coopen_core', a2_report=a2, **{'pass': not why, 'coopen_registration_ok': not why_c})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop', 'offset_hold_ok',
                                             'finger_effort_ok', 'trace_log_ok', 'coopen_registration_ok', 'why')}, ensure_ascii=False, default=float))
    print(json.dumps({'a2': {k: v for k, v in (res.get('a2_report') or {}).items()}}, ensure_ascii=False, default=float)[:1500])
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
