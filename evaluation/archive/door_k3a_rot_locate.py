#!/usr/bin/env python3
"""K3-A 旋轉追蹤誤差來源的離線定位（只讀）。r4：依 Codex 20261022 r3 審查的三項必修重寫；r1–r3 輸出保留（程式見 git 歷史）。

主證據＝**時間一致的相鄰快照動態殘差**：自求解快照 k（q_k、s_k、d̂_k）起，以兩快照之間各物理步實際套用的手臂速度積分設定點，
以模型 q⁺＝q＋α(s−q)＋α·d̂ 預測快照 k＋1 的 q；殘差＝同時刻實測 − 預測（各關節），再以 FK 換算為抓取旋轉／平移的影響
（把手座標：x 沿橫桿、y 法向、z 鉛直）。每區間都由實測重新初始化 ⇒ **只報逐區間分布，不作累積量解讀**。
分窗比較：HOLD（夾持保持）與 OPEN→首次單指 < 0.5 N。
次要（描述性）：
  - 抓取旋轉：tgt＝T_cyc；static_ref＝FK(s＋d̂)（**靜態平衡參考**，含模型本就允許的一階追蹤落後，不稱「模型未解釋」）；meas＝FK(q_meas)。
  - 延遲預測：q_pred 是快照後 d_state·dt（1.6×0.05＝0.08 s）的預測 ⇒ 與該時刻的實測物理步配對後才比較。
  - 取樣點明列：「首次單指低接觸前最後一個求解快照」的時刻，與物理步首次低接觸時刻分欄。
命令：每個物理步套用值與發布包絡終點在 1e-6 rad/s 容差內的相符情形，分成「匹配到該步之前已發布者」與「只能匹配到較晚記錄的發布」
  （後者可能是記錄時序，不能當成用了未來命令）。**未完成逐筆因果配對，不能排除命令鏈影響。**

    python3 evaluation/door_k3a_rot_locate.py <run_dir> [...]   # 各寫 <run_dir>/analysis/door_k3a_rot_locate_r4.json（已存在則拒絕覆寫）
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
TOL_CMD = 1e-6        # room 紀錄四捨五入至 6 位小數


def predict_next(q, s, d, applied, alpha, dtp):
    """模型逐物理步：q⁺ = q + α(s − q) + α·d，s⁺ = s + dt_p·u。applied＝區間內各物理步套用速度（依序）。回傳 (q, s)。"""
    q = np.array(q, float)
    s = np.array(s, float)
    d = np.array(d, float)
    a = np.asarray(alpha, float)
    for u in applied:
        q = q + a * (s - q) + a * d
        s = s + dtp * np.asarray(u, float)
    return q, s


def grasp_err(H, T, G):
    E = np.linalg.inv(H) @ T @ G
    return np.degrees(rotvec(E[:3, :3])), E[:3, 3] * 1e3


def fk(base, qa):
    return KC._tcp(list(base), list(qa))


def dist(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None
    return {'n': int(v.shape[0]), 'mean': v.mean(axis=0).tolist(), 'p5': np.percentile(v, 5, axis=0).tolist(),
            'p95': np.percentile(v, 95, axis=0).tolist(), 'absmax': np.abs(v).max(axis=0).tolist(),
            'frac_negative': (v < 0).mean(axis=0).tolist()}


def one(run):
    out = os.path.join(run, 'analysis', 'door_k3a_rot_locate_r4.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    alpha, dtp = np.asarray(ident['alpha'], float), float(ident['phys_dt_measured_s'])
    dt_cyc = 1.0 / float(a['args']['rate'])
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    G = np.array(t['G_T_H'])
    st = r['steps']
    st_t = np.array([s[I['sim_t']] for s in st])
    t_low_phys = next(s[I['sim_t']] for s in st if s[I['sim_t']] >= t_open
                      and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5)

    def step_at(ts):
        return int(np.searchsorted(st_t, ts + 1e-9, side='right') - 1)
    snaps = [x for x in a['log'] if x.get('solve_in') and t_hold - 1e-9 <= x['solve_in']['snap_sim_t_exact'] <= t_low_phys + 1e-9]
    d_state = float((snaps[0].get('delay_comp') or {}).get('d_state_cycles', 0.0)) if snaps else 0.0
    rows = []
    for i in range(len(snaps) - 1):
        si, sn = snaps[i]['solve_in'], snaps[i + 1]['solve_in']
        t0, t1 = si['snap_sim_t_exact'], sn['snap_sim_t_exact']
        k0, k1 = step_at(t0), step_at(t1)
        applied = [st[kk][I['applied_arm']] for kk in range(k0 + 1, k1 + 1)]
        qa0 = np.array(si['q_meas'])[3:9]
        qp, sp = predict_next(qa0, si['s_meas'], si.get('offset_d_hat') or np.zeros(6), applied, alpha, dtp)
        q1 = np.array(sn['q_meas'])
        res = q1[3:9] - qp
        H1 = KC._H_of(st[k1][I['door_q_rad']])
        r_m, p_m = grasp_err(H1, fk(q1[:3], q1[3:9]), G)
        r_p, p_p = grasp_err(H1, fk(q1[:3], qp), G)
        lc = st[k1][I['leaf_contacts']] or {}
        fb = [lc.get(f, {}).get('bar', 0.0) for f in FINGERS]
        rows.append({'t0': t0, 't1': t1, 'phase': 'HOLD' if t0 < t_open else 'OPEN', 'n_phys': len(applied),
                     'res_rad': res.tolist(), 'res_rot_deg': (r_m - r_p).tolist(), 'res_pos_mm': (p_m - p_p).tolist(),
                     's_integration_mismatch_rad': float(np.abs(sp - np.array(sn['s_meas'])).max()),
                     'q_door_deg': math.degrees(st[k1][I['door_q_rad']]), 'f_diff_n': fb[1] - fb[0]})
    win = {}
    for ph in ('HOLD', 'OPEN'):
        w = [x for x in rows if x['phase'] == ph]
        win[ph] = {'n_intervals': len(w), 'res_rad': dist([x['res_rad'] for x in w]), 'res_rot_deg': dist([x['res_rot_deg'] for x in w]),
                   'res_pos_mm': dist([x['res_pos_mm'] for x in w])}
    # ---- 描述性：首次單指低接觸之前最後一個求解快照
    last = [x for x in a['log'] if x.get('solve_in') and t_open <= x['solve_in']['snap_sim_t_exact'] < t_low_phys]
    desc = None
    if last:
        si = last[-1]['solve_in']
        ts = si['snap_sim_t_exact']
        k = step_at(ts)
        H = KC._H_of(st[k][I['door_q_rad']])
        qm = np.array(si['q_meas'])
        dh = np.array(si.get('offset_d_hat') or np.zeros(6))
        rt, _ = grasp_err(H, np.array(si['T_cyc']).reshape(4, 4), G)
        rs, _ = grasp_err(H, fk(qm[:3], np.array(si['s_meas']) + dh), G)
        rm, _ = grasp_err(H, fk(qm[:3], qm[3:9]), G)
        tp = ts + d_state * dt_cyc                                   # 延遲預測終點
        kp = step_at(tp)
        qpred = np.array(si['q_pred'])
        Hp = KC._H_of(st[kp][I['door_q_rad']])
        rpm, _ = grasp_err(Hp, fk(st[kp][I['base_xyth']], st[kp][I['q_arm_meas']]), G)
        rpp, _ = grasp_err(Hp, fk(qpred[:3], qpred[3:9]), G)
        desc = {'snapshot_t': ts, 'phys_first_low_t': t_low_phys, 'label': '首次單指低接觸之前最後一個求解快照（非物理首次低接觸時刻）',
                'rot_deg': {ax: {'tgt': rt[i], 'static_ref': rs[i], 'meas': rm[i], 'static_ref_minus_tgt': rs[i] - rt[i],
                                 'meas_minus_static_ref': rm[i] - rs[i]} for i, ax in enumerate('xyz')},
                'delay_pred': {'pred_end_t': tp, 'paired_phys_t': st[kp][I['sim_t']],
                               'meas_minus_pred_rot_deg': (rpm - rpp).tolist(),
                               'meas_minus_pred_joint_rad': (np.array(st[kp][I['q_arm_meas']]) - qpred[3:9]).tolist()},
                'note': 'static_ref＝FK(s＋d̂) 為靜態平衡參考；meas − static_ref 含模型允許的一階追蹤落後，只報為「相對靜態平衡參考的差」。'}
    # ---- 命令：容差相符，區分因果
    pubs = sorted([(x.get('publish_sim_t_exact', x.get('publish_sim_t')), np.array(x['stages']['endpoint_requested']['v'][3:9]))
                   for x in a['log'] if x.get('published') and x.get('stages') and x['stages'].get('endpoint_requested')], key=lambda p: p[0])
    cmd = {'n_steps': 0, 'n_match_any': 0, 'n_match_prior': 0, 'n_match_later_only': 0, 'max_later_lead_s': 0.0}
    for s_ in st:
        ts = s_[I['sim_t']]
        if not (t_open <= ts <= t_low_phys):
            continue
        cmd['n_steps'] += 1
        u = np.array(s_[I['applied_arm']], float)
        m = [p[0] for p in pubs if np.abs(u - p[1]).max() < TOL_CMD]
        if m:
            cmd['n_match_any'] += 1
            if any(tp_ <= ts + 1e-9 for tp_ in m):
                cmd['n_match_prior'] += 1
            else:
                cmd['n_match_later_only'] += 1
                cmd['max_later_lead_s'] = max(cmd['max_later_lead_s'], min(m) - ts)
    cmd['statement'] = ('本窗套用命令均與某筆發布包絡終點在 1e-6 rad/s 容差內相符' if cmd['n_match_any'] == cmd['n_steps']
                        else '部分套用命令找不到相符的發布包絡終點') + '，但尚未完成逐筆因果配對，不能排除命令鏈影響。'
    res = {'run': os.path.basename(os.path.normpath(run)), 'version': 4, 't_hold': t_hold, 't_open': t_open, 't_first_low_phys': t_low_phys,
           'model': {'alpha': alpha.tolist(), 'phys_dt': dtp, 'd_state_cycles': d_state},
           'adjacent_residual_by_phase': win, 'max_s_integration_mismatch_rad': max(x['s_integration_mismatch_rad'] for x in rows),
           'descriptive_last_snapshot_before_low': desc, 'commands': cmd,
           'note': '主證據＝相鄰快照動態殘差（逐區間分布，不作累積）。先後與相關不等於因果；不宣告負載、估計器或命令鏈為卸載根因。',
           'rows': rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    for run in sys.argv[1:]:
        res = one(run)
        print(json.dumps({k: res[k] for k in ('run', 't_open', 't_first_low_phys', 'adjacent_residual_by_phase', 'max_s_integration_mismatch_rad',
                                              'descriptive_last_snapshot_before_low', 'commands')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
