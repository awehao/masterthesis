#!/usr/bin/env python3
"""EB1：接近方向誤差預算（純離線；規格 results/vision/EB1_approach_error_budget_spec.md draft-2）。

恆等分解（同一時刻）：e_total = â*ᵀ(p̂_goal − p*_goal)［目標生成］＋ â*ᵀ(p_TCP − p̂_goal)［追蹤］。
p̂_goal＝求解節點當輪 solve_in.T_cyc；p*_goal＝同一抓取參數作用在真值把手；p_TCP＝room_run.json 逐物理步 base_xyth＋q_arm_meas 經 FK。
兩個時刻窗分開：預抓取窗（pre 判定成立前 0.5 s）、閉爪前窗（grip: close 前 0.5 s）。真值只在本評估檔。

    python3 evaluation/eb1_error_budget.py     # 輸出 results/vision/EB1_error_budget.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import dl0_autolabel as AL                                              # noqa: E402
import geometry_contract as GC                                          # noqa: E402
import object_target_geometry as OT                                     # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

RUNS = os.path.join(HERE, 'runs')
REF = 'd1s4b_M'
TASK_ALLOWED = {'out', 'open_m', 'handle_source', 'stop_at_pregrasp', 'pregrasp_rot_tol_rad', 'post_stop_record_s',
                'post_stop_wall_cap_s', 'inject_abort_at_s'}
TIER_B_EXTRA = {'motm_a_ref', 'motm_retreat_rate'}
SOLVER_ALLOWED = {'out', 'require_topic_target'}
WIN = 0.5
A_H = np.array([0.0, -1.0, 0.0])
NOMINAL_C = np.array([0.0, 1.165, 0.55])          # 資產把手中心（drawer-pose 預設；DL1 已核對實錄 T_cyc ≤ 2e-7）


def classify_runs():
    base = json.load(open(os.path.join(RUNS, REF, 'task.json')))['args']
    sb = json.load(open(os.path.join(RUNS, REF, 'align_solver.json')))['args']
    tiers, excluded = {'A': [], 'B': []}, {}
    for d in sorted(os.listdir(RUNS)):
        tp, sp = os.path.join(RUNS, d, 'task.json'), os.path.join(RUNS, d, 'align_solver.json')
        if not os.path.isfile(tp):
            continue
        try:
            t = json.load(open(tp))
        except Exception:                                   # noqa: BLE001
            excluded[d] = 'task_json_unreadable'
            continue
        a = t.get('args', {})
        if abs((a.get('grasp_depth_m') or 0) - 0.0068) > 1e-9:
            continue                                        # 非 bar26 現行抓取深度：不在母體
        if not os.path.isfile(sp):
            excluded[d] = 'no_solver_record（證據不足）'
            continue
        s = json.load(open(sp))['args']
        da = {k for k in set(a) | set(base) if a.get(k) != base.get(k)} - TASK_ALLOWED
        ds = {k for k in set(s) | set(sb) if s.get(k) != sb.get(k)} - SOLVER_ALLOWED
        if not da and not ds:
            tiers['A'].append(d)
        elif not ds and da <= TIER_B_EXTRA:
            tiers['B'].append(d)
        else:
            excluded[d] = 'config_differs:' + ','.join(sorted(da | ds))[:120]
    return tiers, excluded


def load_run(d, K):
    D = os.path.join(RUNS, d)
    t = json.load(open(os.path.join(D, 'task.json')))
    sol = json.load(open(os.path.join(D, 'align_solver.json')))
    rr = json.load(open(os.path.join(D, 'room_run.json')))
    cols = rr['steps_cols']
    ib, iq, it = cols.index('base_xyth'), cols.index('q_arm_meas'), cols.index('sim_t')
    steps = [(s[it], s[ib], s[iq]) for s in rr['steps'] if s[ib] is not None and s[iq] is not None]
    tp = os.path.join(D, 'wrist_live', 'truth.jsonl')
    if os.path.exists(tp):
        tr = [json.loads(x) for x in open(tp)]
        c = np.array(tr[-1]['handle_center_world_at_capture'], float)
        c_src = 'wrist_live 逐格真值（取最後一格；抽屜在預抓取與接近前靜止、開度 0）'
    else:
        c, c_src = NOMINAL_C.copy(), '名目參考：資產把手中心（非逐時刻真值；DL1 核對實錄 T_cyc 與名目 ≤ 2e-7）'
    H = np.eye(4)
    H[:3, 3] = c
    a = t['args']
    R_grasp = np.array(a['R_grasp'], float)
    T_HG = np.eye(4)
    T_HG[:3, :3] = R_grasp
    T_HG[:3, 3] = [0.0, float(a['grasp_depth_m']), 0.0]
    rows = [(r['sim_t'], np.array(r['solve_in']['T_cyc'], float).reshape(4, 4)) for r in sol['log']
            if r.get('solve_in') and r['solve_in'].get('T_cyc') is not None]
    ev = t['events']
    if t.get('pregrasp_complete'):
        t_pre = t['pregrasp_complete']['sim_t']
    else:
        t_pre = next((e['sim_t'] for e in ev if e.get('align') == 'approach'), None)
    t_close = next((e['sim_t'] for e in ev if e.get('grip') == 'close'), None)
    return {'run': d, 'H': H, 'c_src': c_src, 'T_HG': T_HG, 'rows': rows, 'steps': steps, 't_pre': t_pre, 't_close': t_close,
            'K': K, 'vision': 'latch_check' in (t.get('vision') or {})}


def tcp(K, base, q):
    return K.fk(np.array(list(base) + list(q), float), 'link_tcp')


STEP_TOL = 0.005
N_WIN = int(round(WIN / 0.01))


def window(R, t_end, s_goal):
    """事件步（|Δt| ≤ 5 ms 的最近物理步）往前 50 步（0.5 s）；每步配對**不晚於該步**的最近求解目標（含窗前那筆）。"""
    K, H, T_HG = R['K'], R['H'], R['T_HG']
    a_star = H[:3, :3] @ np.array([0.0, 1.0, 0.0])                    # 接近方向（真值）
    x_star = H[:3, :3] @ np.array([1.0, 0.0, 0.0])                    # 沿桿
    z_star = np.cross(x_star, a_star)
    T_star = OT.object_target(H, T_HG, A_H, s_goal)['T_WE']
    st = R['steps']
    ts = np.array([x[0] for x in st])
    ie = int(np.argmin(np.abs(ts - t_end)))
    if abs(ts[ie] - t_end) > STEP_TOL:
        return {'evidence': 'insufficient', 'why': f'no_physics_step_within_{STEP_TOL}s', 'dt': float(ts[ie] - t_end)}
    steps = st[max(0, ie - N_WIN):ie + 1]
    rt = np.array([r[0] for r in R['rows']])
    gen, trk, trk_rot, row_t = [], [], [], []
    for t, b, q in steps:
        k = int(np.searchsorted(rt, t + 1e-9, side='right')) - 1     # 最後一筆 rows[k].t ≤ t
        if k < 0:
            return {'evidence': 'insufficient', 'why': 'step_without_prior_target', 'step_t': t}
        Tg = R['rows'][k][1]
        row_t.append(float(rt[k]))
        Tm = tcp(K, b, q)
        d = Tm[:3, 3] - Tg[:3, 3]
        gen.append(float(a_star @ (Tg[:3, 3] - T_star[:3, 3])))
        trk.append((float(a_star @ d), float(x_star @ d), float(z_star @ d)))
        trk_rot.append(float(OT.geodesic(Tm[:3, :3], Tg[:3, :3])))
    trk = np.array(trk)
    Tm_end = tcp(K, steps[-1][1], steps[-1][2])
    tot_end = float(a_star @ (Tm_end[:3, 3] - T_star[:3, 3]))
    q3 = lambda v: {'median': float(np.median(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
    return {'event_t': t_end, 'end_step_t': float(ts[ie]), 'window_steps_t': [float(steps[0][0]), float(steps[-1][0])],
            'n_steps': len(steps), 'paired_target_t_range': [min(row_t), max(row_t)],
            'gen_approach_mm': {k: v * 1e3 for k, v in q3(gen).items()},
            'track_approach_mm': {k: v * 1e3 for k, v in q3(trk[:, 0]).items()},
            'track_along_bar_mm': {k: v * 1e3 for k, v in q3(trk[:, 1]).items()},
            'track_vertical_mm': {k: v * 1e3 for k, v in q3(trk[:, 2]).items()},
            'track_rot_deg_max': float(np.degrees(max(trk_rot))),
            'gen_at_end_mm': gen[-1] * 1e3, 'track_at_end_mm': float(trk[-1, 0]) * 1e3,
            'total_approach_at_end_mm': tot_end * 1e3,
            'T_tcp_end': Tm_end.tolist(), 'T_goal_star': T_star.tolist()}


def geometry(C1, R, Tm, s):
    """GC1 完整位姿核對：取等效 T_HG 使 s 目標恰為實測 TCP，回報 s0 的根部／淺側餘裕（評估用）。"""
    H = R['H']
    T_WO = H @ np.linalg.inv(OT.trans(C1['bar_c']))
    T_HG_eff = OT.trans(-s * A_H) @ np.linalg.inv(H) @ np.asarray(Tm, float)
    g = GC.check(C1, T_WO, 0.0, T_HG_eff, A_H, s)
    gc = g.get('grasp_compat') or {}
    out = {'ok': g['ok'], 'why': g.get('why'), 'root_gap_mm': gc.get('root_gap_along_approach_mm'),
           'bar_center_z_mm': gc.get('bar_center_z_finger_mm')}
    if gc.get('bar_center_z_finger_mm') is not None:
        out['shallow_margin_mm'] = gc['blade_z_range_mm'][1] - gc['bar_center_z_finger_mm']
    return out


def est_errors():
    """DL2 r3 開發資料沿真值接近方向（世界 +y）的有號誤差：把手中心（另報）與**工具目標生成誤差**（T_WE_s0 對真值抓取目標）。"""
    import dl2_eval as EV
    import dl2_obs_to_target as E2
    global T_HG_REF
    T_HG_REF, _ = E2.baseline_grasp(EV.R_grasp())
    d = json.load(open(os.path.join(HERE, 'results', 'vision', 'DL2_dev_eval_r3.json')))
    SRC = {'traj_wg4b_f02_P': 'd1_dev_f02P/wrist_v0', 'traj_mt_b1_02_P': 'd1_hold_mt02P/wrist_v0'}
    out = {}
    for rt in ('G0/N-obs', 'L1/N-obs'):
        per_bin = {}
        for g, cap in SRC.items():
            fr = {f['n']: f for f in AL.frames_of(cap)}
            for r in d['groups'][g]['rows']:
                x = r['routes'][rt]
                if not x['ok']:
                    continue
                b = next((f'{lo}-{hi}' for lo, hi in ((0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0)) if lo <= r['dist'] < hi), None)
                if b is None:
                    continue
                c = np.array(fr[r['n']]['c'], float)
                H = np.eye(4)
                H[:3, 3] = c
                Tt = OT.object_target(H, T_HG_REF, A_H, 0.0)['T_WE']          # 真值抓取目標（同一抓取參數）
                per_bin.setdefault(b, []).append({'center': (np.array(x['T_WH'])[1, 3] - c[1]) * 1e3,
                                                  'tool': (np.array(x['T_WE_s0'])[1, 3] - Tt[1, 3]) * 1e3})
        st = lambda v: {'n': len(v), 'median': float(np.median(v)), 'p5': float(np.percentile(v, 5)),
                        'p95': float(np.percentile(v, 95)), 'min': float(min(v)), 'max': float(max(v))}
        out[rt] = {b: {'handle_center': st([e['center'] for e in v]), 'tool_target_s0': st([e['tool'] for e in v])}
                   for b, v in sorted(per_bin.items())}
    return out


def main():
    out_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'vision', 'EB1_error_budget_r2.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    C1 = GC.load_contract()
    assert C1['ok'], C1.get('why')
    tiers, excluded = classify_runs()
    res = {'schema': 'eb1_error_budget/1', 'decomposition': 'e_total = gen（T_cyc 對真值目標）＋ track（TCP 對 T_cyc），沿真值接近軸有號投影；正＝更深入',
           'tiers': tiers, 'excluded': excluded, 'runs': {}, 'est_dev_signed_mm': est_errors(),
           'margins_nominal_mm': {'deep_root': 2.7, 'shallow_blade_tip': 4.3, 'pre_tol': 5.0}}
    for tier, names in tiers.items():
        for d in names:
            R = load_run(d, K)
            rec = {'tier': tier, 'c_src': R['c_src'], 'vision': R['vision'], 't_pre': R['t_pre'], 't_close': R['t_close']}
            if R['t_pre'] is not None:
                w = window(R, R['t_pre'], 0.03)
                rec['pregrasp'] = w
                if 'T_tcp_end' in w:
                    rec['pregrasp']['hypothetical_advance30'] = geometry(C1, R, w['T_tcp_end'], 0.03)
            if R['t_close'] is not None:
                w = window(R, R['t_close'], 0.0)
                rec['pre_close'] = w
                if 'T_tcp_end' in w:
                    rec['pre_close']['measured_tcp_static_model_margin'] = geometry(C1, R, w['T_tcp_end'], 0.0)
            res['runs'][d] = rec
            print(d, tier, 'pre end gen+track', round(rec.get('pregrasp', {}).get('gen_at_end_mm', float('nan')) + rec.get('pregrasp', {}).get('track_at_end_mm', float('nan')), 2),
                  'adv30 root', (rec.get('pregrasp', {}).get('hypothetical_advance30') or {}).get('root_gap_mm'),
                  '| close track', round((rec.get('pre_close') or {}).get('track_approach_mm', {}).get('median', float('nan')), 2),
                  'grasp root', ((rec.get('pre_close') or {}).get('measured_tcp_static_model_margin') or {}).get('root_gap_mm'))
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('寫出', out_p)


if __name__ == '__main__':
    main()
