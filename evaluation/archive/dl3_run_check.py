#!/usr/bin/env python3
"""DL3 一趟（dl3v_M）跑後核對（事前登錄；規格 results/vision/DL3_vision_pregrasp_spec.md）。真值只在「評估」段。

    python3 evaluation/dl3_run_check.py runs/dl3v_M        # 輸出 runs/dl3v_M/analysis/dl3_run_check.json（已存在則拒絕覆寫）

核對（控制達成鎖定目標，與對真值偏差分開）：
 1 第一輪求解：solve_in.T_cyc 姿態＝鎖定目標姿態、target_src＝topic；求解端 dl3_first_target＝任務端 first_published_target。
 2 鎖定前零目標：求解端收到的第一個目標時刻不早於任務端鎖定事件。
 3 參數：任務、求解參數與來源趟 d1s4b_M 逐項相同，只允許登錄的差異；求解 offset_moving＝True。
 4 成功判準（對鎖定目標）：PREGRASP_COMPLETE、位置 ≤ 5 mm、姿態 ≤ 0.02 rad、保持 ≥ 0.5 s；停止後紀錄完整與否分列。
 5 評估（真值）：鎖定把手／目標對真值；停止時實測 TCP 對真值接觸前目標；GC2 以真值物體核對實測 TCP 位姿（評估用，不回饋）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import gripper_shell_check as G2                      # noqa: E402
import object_target_geometry as OT                   # noqa: E402

SRC = os.path.join(HERE, 'runs', 'd1s4b_M')
TASK_ALLOWED = {'handle_source', 'stop_at_pregrasp', 'pregrasp_rot_tol_rad', 'post_stop_record_s', 'post_stop_wall_cap_s', 'out'}
SOLVER_ALLOWED = {'require_topic_target', 'out'}


def diff_args(a, b, allowed):
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if k not in allowed and a.get(k) != b.get(k)}


def main():
    D = os.path.abspath(sys.argv[1])
    out_p = os.path.join(D, 'analysis', 'dl3_run_check.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    task = json.load(open(os.path.join(D, 'task.json')))
    sol = json.load(open(os.path.join(D, 'align_solver.json')))
    st0 = json.load(open(os.path.join(SRC, 'task.json')))
    ss0 = json.load(open(os.path.join(SRC, 'align_solver.json')))
    res = {'run': D, 'checks': {}}
    C = res['checks']
    vis = task.get('vision', {})
    lc = vis.get('latch_check') or {}
    # 1／2 第一輪與鎖定前零目標
    rows = [r for r in sol.get('log', []) if r.get('solve_in') and r['solve_in'].get('T_cyc') is not None]
    lock_s = np.array(vis['lock_target_s']) if vis.get('lock_target_s') else None
    if rows and lock_s is not None:
        T0 = np.array(rows[0]['solve_in']['T_cyc']).reshape(4, 4)
        C['first_solve_rot_equals_lock_rad'] = float(OT.geodesic(T0[:3, :3], lock_s[:3, :3]))
        C['first_solve_target_src'] = rows[0]['solve_in'].get('target_src')
        C['first_solve_ok'] = C['first_solve_rot_equals_lock_rad'] <= 1e-9 and C['first_solve_target_src'] == 'topic'
    ft = sol.get('dl3_first_target')
    fp = vis.get('first_published_target')
    if ft and fp:
        C['solver_first_rx_equals_task_first_pub'] = float(np.abs(np.array(ft['T']) - np.array(fp['T'])).max()) <= 1e-12
        lock_ev = next((e['sim_t'] for e in task['events'] if 'vision_latch' in e), None)
        C['no_target_before_lock'] = lock_ev is not None and ft['rx_sim_t'] is not None and ft['rx_sim_t'] >= lock_ev - 1e-9
    # 3 參數
    C['task_args_diff_vs_source'] = diff_args(st0['args'], task['args'], TASK_ALLOWED)
    C['solver_args_diff_vs_source'] = diff_args(ss0['args'], sol['args'], SOLVER_ALLOWED)
    C['solver_offset_moving'] = sol['args'].get('offset_moving')
    C['solver_n_published_after_stop_rx'] = (sol.get('stats') or {}).get('n_published_after_stop_rx')
    C['solver_stop_why'] = (sol.get('stats') or {}).get('stop_why')
    C['replay_and_physical_checks'] = '另跑 horizon_replay_check.py 與 motm_physical_check.py（analysis/），結果分列'
    # 4 成功判準
    pc = task.get('pregrasp_complete') or {}
    C['result'] = task.get('result')
    C['abort'] = task.get('abort')
    C['success_vs_lock'] = bool(task.get('result') == 'PREGRASP_COMPLETE' and pc.get('pos_err_m', 1) <= 0.005
                                and pc.get('rot_err_rad', 1) <= 0.02 and pc.get('held_s', 0) >= 0.5)
    C['pregrasp'] = pc
    C['post_stop_record_complete'] = task.get('post_stop_record_complete')
    C['post_stop_record_incomplete'] = task.get('post_stop_record_incomplete')
    tr = task.get('post_stop_trace') or []
    if tr and tr[0].get('tcp') and tr[-1].get('tcp'):
        C['post_stop_tcp_drift_mm'] = float(np.linalg.norm(np.array(tr[-1]['tcp']) - np.array(tr[0]['tcp'])) * 1e3)
    # 5 評估（真值）
    ev = {}
    truth = [json.loads(l) for l in open(os.path.join(D, 'wrist_live', 'truth.jsonl'))] if os.path.exists(
        os.path.join(D, 'wrist_live', 'truth.jsonl')) else []
    if lc.get('ok') and truth:
        t_lock = lc['latch']['t_latch']
        c_true = np.array(min(truth, key=lambda t: abs(t['t_cap'] - t_lock))['handle_center_world_at_capture'], float)
        H_true = np.eye(4)
        H_true[:3, 3] = c_true
        Hl = np.array(lc['lock_T_WH'])
        ev['lock_center_err_mm'] = float(np.linalg.norm(Hl[:3, 3] - c_true) * 1e3)
        ev['lock_rot_err_deg'] = float(np.degrees(OT.geodesic(Hl[:3, :3], np.eye(3))))
        # 真值接觸前目標（同一抓取參數，作用在真值把手）——用來比較鎖定目標與實測 TCP，不是生成控制目標
        Tl = np.array(lc['latch']['T_WE_s'])
        T_rel = np.linalg.inv(Hl) @ Tl                          # 抓取＋退讓在把手座標的相對位姿
        Tt = H_true @ T_rel
        ev['lock_target_vs_truth_target_mm'] = float(np.linalg.norm(Tl[:3, 3] - Tt[:3, 3]) * 1e3)
        ev['lock_target_vs_truth_target_deg'] = float(np.degrees(OT.geodesic(Tl[:3, :3], Tt[:3, :3])))
        if pc.get('tcp'):
            Tm = np.array(pc['tcp'])
            ev['tcp_at_complete_vs_truth_target_mm'] = float(np.linalg.norm(Tm[:3, 3] - Tt[:3, 3]) * 1e3)
            ev['tcp_at_complete_vs_truth_target_deg'] = float(np.degrees(OT.geodesic(Tm[:3, :3], Tt[:3, :3])))
            # GC2 以真值物體核對實測 TCP：取等效 T_HG 使 s = 0.03 的目標恰為實測 TCP（T_HG = Trans(−s·a_H)·H*⁻¹·Tm）；
            # check2 的 s 目標＝實測 TCP 的碰撞核對；s = 0 的抓取相容＝「由此位姿沿接近軸前進 30 mm 是否相容」
            C2 = G2.load_contract2()
            a_H = np.array([0.0, -1.0, 0.0])
            T_WO_true = H_true @ np.linalg.inv(OT.trans(C2['gc1']['bar_c']))
            T_HG_eff = OT.trans(-0.03 * a_H) @ np.linalg.inv(H_true) @ Tm
            g = G2.check2(C2, T_WO_true, 0.0, T_HG_eff, a_H, 0.03)
            def stat(tag):
                out = {}
                for part in ('shell', 'camera_assembly'):
                    out[part] = sorted({v['status'] for v in (g.get(part) or {}).get(tag, {}).values()})
                f = g.get('fingers') or {}
                out['fingers_split'] = sorted({v['status'] for v in (f.get('pairs') or {}).get(tag, {}).values()})
                out['fingers_gap_guard'] = sorted({v['status'] for v in (f.get('gap_guard') or {}).get(tag, {}).values()})
                return out
            ev['gc2_actual_pregrasp_pose_collision'] = {
                'statuses': stat('s'), 'note': '停止時實測 TCP 位姿對真值物體的靜態凸模型核對（評估用、不回饋）'}
            ev['gc2_hypothetical_advance30_grasp'] = {
                'statuses': stat('s0'), 'grasp_compat': (g.get('fingers') or {}).get('grasp_compat'),
                'overall_ok': g['ok'], 'why': g.get('why'),
                'note': '假想由實測位姿沿接近軸再前進 30 mm 的抓取相容與碰撞；與實際預抓取位姿碰撞**分開解讀**'}
    res['eval_truth_only'] = ev
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == '__main__':
    main()
