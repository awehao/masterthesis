#!/usr/bin/env python3
"""K3-A 旋轉追蹤誤差來源的離線定位（只讀；依 Codex 20261022 door_k3a_adm_1 裁定）。窗口：OPEN → 首次單指 < 0.5 N（物理步）。

每個求解週期，以同一量測快照的門角 q 的模型把手 H(q) 與夾持成立時的 G_T_H，算四個位姿的「抓取旋轉」
（E＝H⁻¹·T·G 的旋轉向量，把手座標：x 沿橫桿、y 法向、z 鉛直；同時給平移）：
  tgt＝求解器實際採用的 T_cyc；pred＝FK(q_pred)（延遲預測）；sp＝FK(s_meas)（同步設定點）；meas＝FK(q_meas)（同步實測）。
差分：sp − tgt＝求解命令後的設定點相對目標；meas − sp＝設定點到實測（驅動落後、負載）；meas − pred＝延遲預測殘差。
命令：每個物理步取執行端實際套用的手臂速度（room applied_arm），與該步之前最後一次求解發布的手臂請求（request.v[3:9]，依
  publish_sim_t_exact 時間配對）比較；報差值與超過 1 mrad/s 的步比例（紀錄未逐筆配對，這是時間配對，不是同一筆命令）。
同步關節模型殘差：自快照 k（q_k、s_k）起，以兩快照之間各物理步實際套用速度積分設定點 s，並以模型 q⁺＝q＋α(s−q)＋α·d̂ 預測快照 k＋1 的 q；
  殘差＝實測 − 預測（各關節），並以 FK 換算為抓取旋轉／平移的影響。
同列雙指力差（物理步，取快照時刻）。先後與相關不等於因果。

r2：(a) 加入模型預期穩態位姿 exp＝FK(s_meas＋d̂)（設定點刻意含重力下垂補償，FK(s) 單獨不具意義）；分解改為
    exp − tgt（命令＋補償後模型預期相對目標）、meas − exp（模型未解釋）、meas − pred（延遲預測殘差）。
    (b) 命令比對改用 stages.endpoint_requested（執行端套用的是發布包絡終點；request 是包絡首項，與套用值本就不同）。
    (c) 與雙指力差的相關：窗口內各量皆隨時間單調，相關係數僅供參考，不具判別力。
r3：命令另做**逐位元**核對：每個物理步套用值是否等於某一次已發布包絡終點，以及套用值的變更次數（判定執行端有無限幅／改寫）；
    時間配對差異只反映發布時間戳與套用時刻未對齊，不作執行落差證據。

    python3 evaluation/door_k3a_rot_locate.py <run_dir> [...]   # 各寫 <run_dir>/analysis/door_k3a_rot_locate_r3.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
from door_k3a_diag2 import rotvec                     # noqa: E402

FINGERS = KC.FINGERS


def grasp_err(H, T, G):
    E = np.linalg.inv(H) @ T @ G
    return np.degrees(rotvec(E[:3, :3])), E[:3, 3] * 1e3


def fk(base, qa):
    return KC._tcp(list(base), list(qa))


def one(run):
    out = os.path.join(run, 'analysis', 'door_k3a_rot_locate_r3.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    alpha, dtp = np.asarray(ident['alpha'], float), float(ident['phys_dt_measured_s'])
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    t_open = next(e['t'] for e in t['policy_events'] if e.get('to') == 'OPEN')
    G = np.array(t['G_T_H'])
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])
    t_low = next(s[I['sim_t']] for s in st if s[I['sim_t']] >= t_open and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5)
    L = [x for x in a['log'] if x.get('solve_in') and t_open - 1e-9 <= x['solve_in']['snap_sim_t_exact'] <= t_low + 1e-9]
    pubs = sorted([(x.get('publish_sim_t_exact', x.get('publish_sim_t')), np.array(x['stages']['endpoint_requested']['v'][3:9]))
                   for x in a['log'] if x.get('published') and x.get('stages') and x['stages'].get('endpoint_requested')], key=lambda p: p[0])
    pub_t = np.array([p[0] for p in pubs])

    def step_at(ts):
        return int(np.searchsorted(st_t, ts + 1e-9, side='right') - 1)
    rows = []
    for i, x in enumerate(L):
        si = x['solve_in']
        ts = si['snap_sim_t_exact']
        k = step_at(ts)
        s_ = st[k]
        q = s_[I['door_q_rad']]
        H = KC._H_of(q)
        qm = np.array(si['q_meas'])
        base = qm[:3]
        E = {'tgt': grasp_err(H, np.array(si['T_cyc']).reshape(4, 4), G),
             'pred': grasp_err(H, fk(np.array(si['q_pred'])[:3], np.array(si['q_pred'])[3:9]), G),
             'sp': grasp_err(H, fk(base, si['s_meas']), G),
             'exp': grasp_err(H, fk(base, np.array(si['s_meas']) + np.array(si.get('offset_d_hat') or np.zeros(6))), G),
             'meas': grasp_err(H, fk(base, qm[3:9]), G)}
        lc = s_[I['leaf_contacts']] or {}
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        row = {'t': ts, 'q_deg': math.degrees(q), 'f_diff_n': fb[1] - fb[0], 'f_bar': fb,
               **{f'rot_{n}_deg': E[n][0].tolist() for n in E}, **{f'pos_{n}_mm': E[n][1].tolist() for n in E}}
        # 同步關節模型殘差：積分到下一快照
        if i + 1 < len(L):
            sn = L[i + 1]['solve_in']
            t1 = sn['snap_sim_t_exact']
            qa = qm[3:9].copy()
            sp = np.array(si['s_meas'], float)
            d = np.array(si.get('offset_d_hat') or np.zeros(6), float)
            k1 = step_at(t1)
            for kk in range(k + 1, k1 + 1):
                u = np.array(st[kk][I['applied_arm']], float)
                qa = qa + alpha * (sp - qa) + alpha * d
                sp = sp + dtp * u
            q_meas1 = np.array(sn['q_meas'])[3:9]
            res = q_meas1 - qa
            H1 = KC._H_of(st[k1][I['door_q_rad']])
            r_m, p_m = grasp_err(H1, fk(np.array(sn['q_meas'])[:3], q_meas1), G)
            r_p, p_p = grasp_err(H1, fk(np.array(sn['q_meas'])[:3], qa), G)
            row['model_res_rad'] = res.tolist()
            row['model_res_rot_deg'] = (r_m - r_p).tolist()
            row['model_res_pos_mm'] = (p_m - p_p).tolist()
            row['s_integrated_vs_snapshot_rad'] = float(np.abs(sp - np.array(sn['s_meas'])).max())
        rows.append(row)
    # 命令：物理步套用 vs 時間配對的最後一次發布
    cmd = []
    for s_ in st:
        ts = s_[I['sim_t']]
        if not (t_open <= ts <= t_low):
            continue
        j = int(np.searchsorted(pub_t, ts + 1e-9, side='right') - 1)
        if j < 0:
            continue
        u = np.array(s_[I['applied_arm']], float)
        cmd.append({'t': ts, 'diff': (u - pubs[j][1]).tolist(), 'req_abs_max': float(np.abs(pubs[j][1]).max())})
    dmax = np.max(np.abs([c['diff'] for c in cmd]), axis=0) if cmd else None
    endpoints = [p[1] for p in pubs]
    win_steps = [s_ for s_ in st if t_open <= s_[I['sim_t']] <= t_low]
    n_exact = sum(1 for s_ in win_steps if min(np.abs(np.array(s_[I['applied_arm']], float) - e).max() for e in endpoints) < 1e-6)   # room 紀錄四捨五入至 6 位小數
    n_change = sum(1 for a_, b_ in zip(win_steps, win_steps[1:])
                   if np.abs(np.array(a_[I['applied_arm']], float) - np.array(b_[I['applied_arm']], float)).max() > 1e-12)
    frac = float(np.mean([np.abs(c['diff']).max() > 1e-3 for c in cmd])) if cmd else None

    def ex(key, i):
        v = [rw[key][i] for rw in rows if key in rw]
        return None if not v else {'start': v[0], 'end': v[-1], 'min': min(v), 'max': max(v)}
    summ = {}
    for n in ('tgt', 'pred', 'sp', 'exp', 'meas'):
        for i, ax in enumerate('xyz'):
            summ[f'rot_{n}_{ax}_deg'] = ex(f'rot_{n}_deg', i)
    for i, ax in enumerate('xyz'):
        summ[f'model_res_rot_{ax}_deg'] = ex('model_res_rot_deg', i)
    last = rows[-1]
    decomp = {ax: {'tgt': last['rot_tgt_deg'][i], 'exp_minus_tgt': last['rot_exp_deg'][i] - last['rot_tgt_deg'][i],
                   'meas_minus_exp': last['rot_meas_deg'][i] - last['rot_exp_deg'][i], 'meas_minus_pred': last['rot_meas_deg'][i] - last['rot_pred_deg'][i],
                   'meas': last['rot_meas_deg'][i], 'pos_exp_minus_tgt_mm': last['pos_exp_mm'][i] - last['pos_tgt_mm'][i],
                   'pos_meas_minus_exp_mm': last['pos_meas_mm'][i] - last['pos_exp_mm'][i]} for i, ax in enumerate('xyz')}
    first = rows[0]
    decomp_open = {ax: {'tgt': first['rot_tgt_deg'][i], 'exp_minus_tgt': first['rot_exp_deg'][i] - first['rot_tgt_deg'][i],
                        'meas_minus_exp': first['rot_meas_deg'][i] - first['rot_exp_deg'][i], 'meas': first['rot_meas_deg'][i]} for i, ax in enumerate('xyz')}
    cum_res = np.sum([rw['model_res_rot_deg'] for rw in rows if 'model_res_rot_deg' in rw], axis=0)
    corr = {}
    fd = np.array([rw['f_diff_n'] for rw in rows])
    for n in ('tgt', 'exp', 'meas'):
        for i, ax in enumerate('xyz'):
            v = np.array([rw[f'rot_{n}_deg'][i] for rw in rows])
            corr[f'{n}_{ax}'] = float(np.corrcoef(v, fd)[0, 1]) if np.std(v) > 0 and np.std(fd) > 0 else None
    res = {'run': os.path.basename(os.path.normpath(run)), 't_open': t_open, 't_first_low': t_low, 'n_cycles': len(rows),
           'at_open_decomposition_deg': decomp_open, 'at_first_low_decomposition_deg': decomp, 'window_summary': summ,
           'model_residual': {'cum_rot_deg': cum_res.tolist(),
                              'max_abs_res_rad': np.max(np.abs([rw['model_res_rad'] for rw in rows if 'model_res_rad' in rw]), axis=0).tolist(),
                              'max_s_integration_mismatch_rad': max(rw.get('s_integrated_vs_snapshot_rad', 0.0) for rw in rows)},
           'commands': {'n_steps': len(cmd), 'applied_minus_lastpub_absmax_rad_s': None if dmax is None else dmax.tolist(),
                        'frac_steps_diff_over_1mrad_s': frac,
                        'n_steps_applied_equals_some_published_endpoint': n_exact, 'n_window_steps': len(win_steps),
                        'n_applied_changes': n_change, 'n_solver_cycles_in_window': len(L), 'req_abs_max_rad_s': max(c['req_abs_max'] for c in cmd) if cmd else None,
                        'note': '時間配對（物理步 vs 其前最後一次發布的包絡終點 endpoint_requested），非逐筆配對'},
           'corr_with_finger_force_diff': corr,
           'note': '只讀定位。tgt＝T_cyc；pred＝FK(q_pred)；sp＝FK(s_meas)；meas＝FK(q_meas)；皆以快照門角的模型把手與 G_T_H_grasp 計。相關不等於因果。',
           'rows': rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    for run in sys.argv[1:]:
        res = one(run)
        print(json.dumps({k: res[k] for k in ('run', 't_open', 't_first_low', 'n_cycles', 'at_open_decomposition_deg', 'at_first_low_decomposition_deg', 'model_residual', 'commands',
                                              'corr_with_finger_force_diff')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
