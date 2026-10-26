#!/usr/bin/env python3
"""door_coopen_core_g_1 底盤偏差 d̂_b 的**因果重播**（只讀；Codex 20261027_010258 §3-1）。
資料：room_run 逐物理步 base_xyth（該步套用前位姿）與 base_cmd_body（該步套用的本體命令）；owner＝1（wholebody_applied）視為 exec_mode 0、api 已套用。
  位姿取偶數步號（50 Hz，與線上 /odom 同頻；線上用 /odom stamp，本重播用 room_run 同義欄位）。
0. 配對核對：Δp_k＝p_{k+1}−p_k 轉本體／dt_p 與 cmd_k、cmd_{k−1} 的差 RMS；並列先前 resid_check 用的 meas_vb_body[k]−cmd[k]（後向差分配同步命令）。
1. 因果重播：估計器 door_base_bias_est（登錄常數，不調）於 ALIGN_HOLD 起（9.40 s）至中止（25.64 s）啟用；每一合格區間先以 estimate(t_a)
   取**僅用此前區間**的 d̂_b，再把本區間加入。一步底盤速度殘差：無補償＝b_obs，有補償＝b_obs − d̂_b（d̂_b 為 None ⇒ 與無補償同）。
2. TCP 投影：殘差位移（世界 R(θ_a)·殘差·Δt）經位置雅可比投影到「實測 TCP 對當時目標的垂直誤差方向」，第二次暫停 [21.48, 25.64) 每 0.25 s 一窗合計。
界線：一步殘差只說遞推一致性；不據此宣稱閉迴路接近段會收斂。

    python3 evaluation/door_coopen_core_g_1_base_bias_replay.py   # 寫 runs/door_coopen_core_g_1/analysis/…_base_bias_replay.json（不覆寫）
"""
import bisect
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_base_bias_est as E                                            # noqa: E402
import door_k2_check as KC                                                # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_g_1_base_bias_replay.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
TCP = 'link_tcp'
DTP = 0.01
T_EN = 9.399999789893627          # task_trace ALIGN_HOLD 起
T_PAUSE2 = 21.48


def R3(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def body_rate(pa, pb, dt):
    d = np.asarray(pb, float) - np.asarray(pa, float)
    d[2] = (d[2] + math.pi) % (2 * math.pi) - math.pi
    return (R3(float(pa[2])).T @ d) / dt


def stats(a):
    a = np.asarray(a, float)
    return {'n': int(len(a)), 'mean': a.mean(axis=0).tolist(), 'rms': np.sqrt((a ** 2).mean(axis=0)).tolist()} if len(a) else {'n': 0}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    t = json.load(open(os.path.join(RUN, 'task.json')))
    t_abort = next(e['t'] for e in t['policy_events'] if 'abort' in e)
    st = sorted(r['steps'], key=lambda s: s[c['step']])
    by = {s[c['step']]: s for s in st}
    own = c['owner(0=nav,1=wb,2=glide)']

    # 0. 配對核對（啟用區間內、逐物理步）
    d_now, d_prev, d_old = [], [], []
    for s in st:
        k = s[c['step']]
        if not (T_EN <= s[c['sim_t']] < t_abort) or (k + 1) not in by or (k - 1) not in by:
            continue
        v = body_rate(s[c['base_xyth']], by[k + 1][c['base_xyth']], DTP)
        d_now.append(v - np.asarray(s[c['base_cmd_body']], float))
        d_prev.append(v - np.asarray(by[k - 1][c['base_cmd_body']], float))
        d_old.append(np.asarray(s[c['meas_vb_body']], float) - np.asarray(s[c['base_cmd_body']], float))
    pairing = {'fwd_minus_cmd_k': stats(d_now), 'fwd_minus_cmd_k_minus_1': stats(d_prev), 'meas_vb_body_k_minus_cmd_k（resid_check 所用）': stats(d_old)}

    # 1. 因果重播
    est = E.BaseBiasEstimator(DTP)
    K = WholeBodyKinematics.from_urdf_file(URDF)
    a_H = np.asarray(json.load(open(GB))['args']['a_H'], float)
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    logs = sorted([x for x in sv['log'] if (x.get('solve_in') or {}).get('T_cyc') is not None], key=lambda x: x['solve_in']['snap_sim_t_exact'])
    lt = [x['solve_in']['snap_sim_t_exact'] for x in logs]
    rows, est_why = [], {}
    for s in st:
        k, tk = s[c['step']], s[c['sim_t']]
        if tk >= t_abort:
            break
        est.set_enabled(tk >= T_EN - 1e-9, tk)
        if k % 2 == 0:
            pa = est.pose_prev
            b_hat, why, _ = est.estimate(tk)
            est_why[str(why)] = est_why.get(str(why), 0) + 1
            rec = est.on_pose(tk, s[c['base_xyth']])
            if rec.get('used'):
                b_obs = np.asarray(rec['b_obs'], float)
                res_w = b_obs - (b_hat if b_hat is not None else 0.0)
                row = {'t0': rec['t0'], 't1': rec['t1'], 'b_obs': b_obs.tolist(), 'b_hat': None if b_hat is None else b_hat.tolist(), 'res_with': res_w.tolist()}
                i = bisect.bisect_right(lt, rec['t0']) - 1
                sa = by.get(k - rec['m'])
                if i >= 0 and sa is not None and pa is not None:
                    T = np.asarray(logs[i]['solve_in']['T_cyc'], float).reshape(4, 4)
                    qf = np.concatenate([np.asarray(sa[c['base_xyth']], float), np.asarray(sa[c['q_arm_meas']], float)])
                    J = np.asarray(K.jacobian(qf, TCP), float)[:3, 0:3]
                    ax = KC._H_of(sa[c['door_q_rad']])[:3, :3] @ a_H
                    pv = K.fk(qf, TCP)[:3, 3] - T[:3, 3]
                    pv = pv - float(pv @ ax) * ax
                    n = pv / np.linalg.norm(pv) if np.linalg.norm(pv) > 1e-9 else np.zeros(3)
                    dtw = rec['t1'] - rec['t0']
                    Rw = R3(float(qf[2]))
                    row['perp_mm'] = float(np.linalg.norm(pv)) * 1e3
                    row['proj_no_um'] = float(n @ (J @ (Rw @ b_obs) * dtw)) * 1e6
                    row['proj_with_um'] = float(n @ (J @ (Rw @ res_w) * dtw)) * 1e6
                rows.append(row)
        est.on_cmd(k, tk, s[c['base_cmd_body']], 0 if s[own] == 1 else 4, 1 if s[c['kind']] == 'wholebody_applied' else 0)

    def seg(lo, hi):
        w = [x for x in rows if lo <= x['t0'] < hi]
        wh = [x for x in w if x['b_hat'] is not None]
        return {'interval_s': [lo, hi], 'n_intervals': len(w), 'n_with_estimate': len(wh),
                'no_comp': stats([x['b_obs'] for x in w]), 'with_comp': stats([x['res_with'] for x in w]),
                'no_comp_where_estimate': stats([x['b_obs'] for x in wh]), 'with_comp_where_estimate': stats([x['res_with'] for x in wh]),
                'proj_no_sum_mm': sum(x.get('proj_no_um', 0.0) for x in w) / 1e3, 'proj_with_sum_mm': sum(x.get('proj_with_um', 0.0) for x in w) / 1e3}
    wins = []
    w0 = T_PAUSE2
    while w0 < t_abort:
        w = [x for x in rows if w0 <= x['t0'] < min(w0 + 0.25, t_abort) and 'proj_no_um' in x]
        if w:
            wins.append({'t0': w0, 'n': len(w), 'proj_no_mm': sum(x['proj_no_um'] for x in w) / 1e3, 'proj_with_mm': sum(x['proj_with_um'] for x in w) / 1e3})
        w0 += 0.25
    out = {'schema': 'door_coopen_core_g_1_base_bias_replay/1', 'estimator': {'WIN_S': E.WIN_S, 'N_MIN': E.N_MIN, 'MAX_AGE_S': E.MAX_AGE_S, 'CAP': list(E.CAP)},
           'enable_s': [T_EN, t_abort], 'pairing_check': pairing, 'estimate_why_at_pose': est_why,
           'rejects': dict(est.rejects), 'n_used': est.n_used,
           'enable_all': seg(T_EN, t_abort), 'pause2': seg(T_PAUSE2, t_abort), 'pause2_windows': wins,
           'note': '一步殘差：d̂_b 只用 t_a 之前的合格區間（因果）；room_run 位姿取偶數步；僅說遞推一致性'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    f = lambda v: '(' + ', '.join(f'{x * 1e3:+.3f}' for x in v) + ')'                       # noqa: E731
    for k_, v in pairing.items():
        print(f'配對 {k_}：mean {f(v["mean"])} rms {f(v["rms"])}（mm/s, mm/s, mrad/s）')
    for nm in ('enable_all', 'pause2'):
        g = out[nm]
        print(f"{nm} {g['interval_s'][0]:.2f}–{g['interval_s'][1]:.2f}：區間 {g['n_intervals']}（有估計 {g['n_with_estimate']}）；"
              f"無補償 mean {f(g['no_comp']['mean'])} rms {f(g['no_comp']['rms'])}；有補償 mean {f(g['with_comp']['mean'])} rms {f(g['with_comp']['rms'])}；"
              f"垂直投影合計 無 {g['proj_no_sum_mm']:+.3f} → 有 {g['proj_with_sum_mm']:+.3f} mm")
    print('拒用', dict(est.rejects), '估計狀態', est_why)


if __name__ == '__main__':
    main()
