#!/usr/bin/env python3
"""時域消融（H1／H5）的共同指標：只看 **W-GMPC 控制窗口**，按相位分。

窗口 = 求解節點第一輪 → 求解節點停止（退開完成、交給關節空間收臂）。
導航、展開、收臂、回程不在窗口內 —— 那些段落不能拿來說明預測時域的效果。

指標（每相位與整窗）：
  追蹤   求解輪的 err_p／err_r（以量測狀態對當輪目標）p50／p95／max
  限制   設定點 s（求解輪）與實測角 q（逐物理步）到「硬限位 ± joint_margin」的最小餘裕與關節
  平順   每軸命令增量（相鄰求解輪）p95／max；手臂命令翻號數（|u| > 0.05）；
         TCP 實測速度峰值（逐物理步 FK）
  計算   核心 total、整輪牆鐘、發布時輸入年齡 p50／p95／p99／max；漏時槽、
         暖啟動丟棄、求解失敗、SQP 停止理由分布
  任務   最終相位、abort、夾持漂移、非預期接觸 —— 取自 motm_metrics.json（若有）

    python3 evaluation/horizon_metrics.py runs/<RUN> [--out x.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

PHASES = ['ALIGN', 'ENGAGE_WAIT', 'OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD',
          'RELEASE_WAIT', 'RETREAT']


def pct(v, ps=(50, 95, 99)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    out = {f'p{p}': round(float(np.percentile(v, p)), 6) for p in ps}
    out['max'] = round(float(v.max()), 6)
    out['n'] = int(len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = a.run_dir
    sol = json.load(open(os.path.join(D, 'align_solver.json')))
    task = json.load(open(os.path.join(D, 'task.json')))
    room = json.load(open(os.path.join(D, 'room_run.json')))
    args = sol['args']
    log = sol['log']
    K = WholeBodyKinematics.from_urdf_file(args['urdf'])
    lo, hi = (np.array(x, float)[3:] for x in K.joint_limits())
    margin = 0.05            # WGMPCConfig.joint_margin（兩組共同）
    elo, ehi = lo + margin, hi - margin

    # ---- 相位時間（任務事件）----
    ev = task['events']
    ph_t = {}
    for e in ev:
        if 'phase' in e and e['phase'] not in ph_t:
            ph_t[e['phase']] = float(e['sim_t'])
    t_win0 = float(log[0]['sim_t'])
    t_stop = next((float(e['sim_t']) for e in ev if e.get('solver_stop')),
                  float(log[-1]['sim_t']))
    bounds = [(p, ph_t[p]) for p in PHASES if p in ph_t]

    def phase_of(t):
        cur = 'PRE_ALIGN'
        for p, t0 in bounds:
            if t >= t0:
                cur = p
        return cur

    # ---- 求解輪 ----
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    prev_u = None
    flips = collections.Counter()
    prev_sign = {}
    stops = collections.Counter()
    n_fail = 0
    for L in log:
        t = float(L['sim_t'])
        if t > t_stop:
            break
        ph = phase_of(t)
        for g in (ph, 'WINDOW'):
            b = by[g]
            b['err_p'].append(L.get('err_p'))
            b['err_r'].append(L.get('err_r'))
            tm = L.get('timing_ms') or {}
            b['core_ms'].append(tm.get('total'))
            b['cycle_wall_ms'].append(L.get('cycle_wall_ms'))
            b['age_out_s'].append(L.get('age_out'))
            s = np.array(L['solve_in']['s_meas'], float)
            m = np.minimum(s - elo, ehi - s)
            b['sp_margin'].append(float(m.min()))
            b['sp_margin_joint'].append(int(m.argmin()) + 1)
        stops[L.get('sqp_stop')] += 1
        if not L.get('ok'):
            n_fail += 1
        u = L.get('request_body')
        if u is not None:
            u = np.array(u, float)
            if prev_u is not None:
                for g in (ph, 'WINDOW'):
                    by[g]['du_base'].append(float(np.abs(u[:3] - prev_u[:3]).max()))
                    by[g]['du_arm'].append(float(np.abs(u[3:] - prev_u[3:]).max()))
            for j in range(3, 9):
                if abs(u[j]) > 0.05:
                    sg = np.sign(u[j])
                    if j in prev_sign and prev_sign[j] != sg:
                        flips[ph] += 1
                        flips['WINDOW'] += 1
                    prev_sign[j] = sg
            prev_u = u

    # ---- 逐物理步真值（實測角餘裕、TCP 速度）----
    c = room['steps_cols']
    i = c.index
    S = [s for s in room['steps'] if t_win0 <= s[1] <= t_stop]
    tcp_prev = None
    for k, s in enumerate(S):
        t = float(s[1])
        ph = phase_of(t)
        q = np.array(s[i('q_arm_meas')], float)
        m = np.minimum(q - elo, ehi - q)
        for g in (ph, 'WINDOW'):
            by[g]['meas_margin'].append(float(m.min()))
            by[g]['meas_margin_joint'].append(int(m.argmin()) + 1)
        if k % 5 == 0:                 # 每 50 ms 一點算 FK
            p = K.fk(np.r_[s[i('base_xyth')], q], args['tcp'])[:3, 3]
            if tcp_prev is not None:
                v = float(np.linalg.norm(p - tcp_prev[1]) / (t - tcp_prev[0]))
                for g in (ph, 'WINDOW'):
                    by[g]['tcp_speed'].append(v)
            tcp_prev = (t, p)

    def summarise(b, g):
        def mn(key, jkey):
            v = b.get(key)
            if not v:
                return None
            k = int(np.argmin(v))
            return {'min_rad': round(float(v[k]), 5), 'joint': int(b[jkey][k])}
        return {
            'tracking': {'err_p_m': pct(b['err_p']), 'err_r_rad': pct(b['err_r'])},
            'limits': {'setpoint': mn('sp_margin', 'sp_margin_joint'),
                       'measured': mn('meas_margin', 'meas_margin_joint')},
            'smoothness': {'du_base': pct(b['du_base'], (95,)),
                           'du_arm': pct(b['du_arm'], (95,)),
                           'arm_sign_flips': int(flips[g]),
                           'tcp_speed_mps': pct(b['tcp_speed'], (50, 95))},
            'compute': {'core_ms': pct(b['core_ms']),
                        'cycle_wall_ms': pct(b['cycle_wall_ms']),
                        'age_out_s': pct(b['age_out_s'])},
        }

    out = {
        'run': os.path.basename(D.rstrip('/')),
        'N': args['N'],
        'window': {'from_sim_t': round(t_win0, 3), 'to_sim_t': round(t_stop, 3),
                   'definition': '求解節點第一輪 → 求解節點停止（退開完成）'},
        'phase_start': {p: round(t, 3) for p, t in bounds},
        'solver': {'n_cycles': len([L for L in log if float(L['sim_t']) <= t_stop]),
                   'n_failed': n_fail, 'sqp_stop': dict(stops),
                   'n_missed_slot': sol['stats'].get('n_missed_slot'),
                   'n_warm_discard': sol['stats'].get('n_warm_discard'),
                   'n_shaped': sol['stats'].get('n_shaped')},
        'by_phase': {g: summarise(by[g], g) for g in ['WINDOW'] + PHASES
                     if g in by},
        'limits_note': f'有效限位 = 硬限位 ± joint_margin {margin} rad；負值 = 超出有效限位',
    }
    mm = os.path.join(D, 'motm_metrics.json')
    if os.path.exists(mm):
        m = json.load(open(mm))
        out['task'] = {k: m.get(k) for k in ('final_phase', 'abort',
                                             'grasp_drift_max_mm',
                                             'contact_to_pull_s',
                                             'undesignated_contact')}
    print(json.dumps({'run': out['run'], 'N': out['N'], 'window': out['window'],
                      'solver': out['solver'],
                      'WINDOW': out['by_phase']['WINDOW']},
                     ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
