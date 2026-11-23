#!/usr/bin/env python3
"""長窗診斷後的只讀契約核對（Codex 20261023_232730 第 2 節；不開模擬、不調參、不改執行端）。

一、門角與門速一致性（door_q_qd：門角取門扇 RigidPrim 姿態、門速取同一 view 的 −ω_z；兩者在本步 API 寫入後、world.step 前同時讀取）：
  逐窗（靜止、OPEN、閂鎖後 0–1 s、1–3 s、3 s–窗末）比較
    - 角度斜率（窗口首末門角差／時間）、
    - 紀錄門速的窗口平均與標準差、
    - 紀錄門速與「本步差分 dq_k＝(q_k − q_{k−1})/dt」及「下一步差分 dq_{k+1}」的相關。
  不套比例修正；只報一致或不一致與其時序特徵。
二、零命令至物理狀態的執行鏈（每物理步：讀位姿 → ex.step → 以本步 yaw 轉世界座標 → set_linear_velocity／set_angular_velocity → 手臂 apply_action → world.step）：
  三層：
    (1) 紀錄命令 base_cmd_body（res.base_cmd，本體座標）——有紀錄；
    (2) API 寫入參數（世界座標向量）——**沒有直接紀錄**；程式中由 (1) 與同步讀到的 yaw 決定性轉換（零命令 ⇒ 精確零向量）；
    (3) 物理步後的實際速度——**沒有紀錄**；只有相鄰步位姿差分 meas_vb_body（＝前一個物理步的位移／dt）與 base_xyth。
  逐段（HOLD、中止前 1 s、閂鎖後 1 s–窗末）比較 (1) 的平均與 (3) 的位姿差分平均。
  其他寫入來源：程式掃描結果另列（見 NOTE）；park_mode 欄為 HoldMonitor 監看狀態，不是位置保持控制。
不做根因主張。

    python3 evaluation/door_k3ab_lw_contract_check.py <out_json> <run_dir>:<policy_abort_t>:<latch_t>[:<open_t>] [...]
"""
import json
import os
import sys

import numpy as np


def load(run):
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    S = r['steps']
    st = np.array([s[I['step']] for s in S])
    return r, I, S, st


def door_windows(S, I, wins):
    t = np.array([s[I['sim_t']] for s in S])
    q = np.array([s[I['door_q_rad']] for s in S])
    qd = np.array([s[I['door_qdot_rad_s']] for s in S])
    dq = np.diff(q) / np.diff(t)
    out = {}
    for nm, (a, b) in wins.items():
        m = (t[1:] >= a) & (t[1:] < b)
        n = int(m.sum())
        if n < 5:
            out[nm] = {'n': n}
            continue
        idx = np.nonzero(m)[0]
        slope = (q[idx[-1] + 1] - q[idx[0]]) / (t[idx[-1] + 1] - t[idx[0]])
        rec = qd[1:][m]
        d_now = dq[m]

        def corr(x, y):
            return None if np.std(x) < 1e-12 or np.std(y) < 1e-12 else float(np.corrcoef(x, y)[0, 1])
        nxt = idx[idx + 1 < len(dq)]
        out[nm] = {'n': n, 'angle_slope': float(slope), 'qdot_mean': float(rec.mean()), 'qdot_std': float(rec.std()),
                   'dq_step_std': float(d_now.std()), 'ratio_qdot_mean_to_slope': None if abs(slope) < 1e-9 else float(rec.mean() / slope),
                   # 列 k＝idx+1：紀錄門速 qd[k]；本步差分 dq[k−1]＝dq[idx]；下一步差分 dq[k]＝dq[idx+1]
                   'corr_qdot_k_vs_dq_k': corr(rec, d_now), 'corr_qdot_k_vs_dq_k1': corr(qd[nxt + 1], dq[nxt + 1])}
    return out


def base_windows(S, I, wins):
    out = {}
    for nm, (a, b) in wins.items():
        ss = [s for s in S if a <= s[I['sim_t']] < b]
        if len(ss) < 5:
            out[nm] = {'n': len(ss)}
            continue
        c = np.mean([s[I['base_cmd_body']] for s in ss], axis=0)
        mv = np.mean([s[I['meas_vb_body']] for s in ss if s[I['meas_vb_body']] is not None], axis=0)
        b0, b1 = np.array(ss[0][I['base_xyth']]), np.array(ss[-1][I['base_xyth']])
        dt = ss[-1][I['sim_t']] - ss[0][I['sim_t']]
        n_zero = sum(1 for s in ss if not np.any(np.asarray(s[I['base_cmd_body']], float)))
        out[nm] = {'n': len(ss), 'cmd_body_mean': c.tolist(), 'meas_vb_body_mean': mv.tolist(), 'world_rate': ((b1 - b0) / dt).tolist(),
                   'n_cmd_exact_zero': n_zero, 'park_mode': sorted({str(s[I['park_mode']]) for s in ss}) if 'park_mode' in I else None,
                   'kinds': sorted({str(s[I['kind']]) for s in ss})}
        if np.linalg.norm(c[:2]) > 1e-9 and np.linalg.norm(mv[:2]) > 1e-9:
            out[nm]['cos_cmd_vs_meas_lin'] = float(np.dot(c[:2], mv[:2]) / np.linalg.norm(c[:2]) / np.linalg.norm(mv[:2]))
    return out


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'schema': 'door_k3ab_lw_contract_check/1', 'runs': []}
    for spec in sys.argv[2:]:
        parts = spec.split(':')
        run, ab, lat = parts[0], float(parts[1]), float(parts[2])
        t_open = float(parts[3]) if len(parts) > 3 else None
        r, I, S, st = load(run)
        assert np.all(np.diff(st) == 1), '步號不連續'
        t_end = S[-1][I['sim_t']]
        dw = {'static_0_15s': (0.0, 15.0), 'post_0_1s': (lat, lat + 1.0), 'post_1_3s': (lat + 1.0, lat + 3.0), 'post_3s_end': (lat + 3.0, t_end + 1)}
        if t_open is not None:
            dw['OPEN_to_abort'] = (t_open, ab)
        bw = {'HOLD_18_19.5': (18.0, 19.5), 'pre_abort_1s': (ab - 1.0, ab), 'post_1s_end': (lat + 1.0, t_end + 1)}
        R = {'run': os.path.basename(os.path.normpath(run)), 'policy_abort_t': ab, 'latch_t': lat, 'open_t': t_open, 'door': door_windows(S, I, dw),
             'base': base_windows(S, I, bw)}
        res['runs'].append(R)
        print('==', R['run'])
        for k, v in R['door'].items():
            if 'angle_slope' in v:
                print(f"  門 {k:14s} 斜率 {v['angle_slope']:+.5f}  紀錄門速平均 {v['qdot_mean']:+.5f}（比 {v['ratio_qdot_mean_to_slope']}）"
                      f"  corr(k,k) {v['corr_qdot_k_vs_dq_k']}  corr(k,k+1) {v['corr_qdot_k_vs_dq_k1']}")
            else:
                print(f"  門 {k:14s} {v}")
        for k, v in R['base'].items():
            print(f"  底盤 {k:14s} 命令 {np.round(np.array(v['cmd_body_mean']) * 1e3, 2).tolist()}  位姿差分 {np.round(np.array(v['meas_vb_body_mean']) * 1e3, 2).tolist()}"
                  f"  cos {v.get('cos_cmd_vs_meas_lin')}  零命令步 {v['n_cmd_exact_zero']}/{v['n']}")
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
