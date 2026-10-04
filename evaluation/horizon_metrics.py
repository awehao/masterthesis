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

sys.path.insert(0, HERE)
from horizon_replay_check import NON_SOLVE_REASONS                # noqa: E402

FLIP_THR = 0.05          # rad/s：低於此視為近零，不參與翻號
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


def count_reversals(useq, thr=0.05, joints=range(3, 9)):
    """手臂命令反轉計數。useq = [(相位, 九維命令), …]，依求解輪順序。

    回傳 {相位或 'WINDOW': Counter(adjacent, gapped, adj_pairs, boundary)}：
      adjacent   相鄰兩輪都 |u| > thr 且反號
      gapped     同相位內、中間隔著近零命令的反號
      adj_pairs  相鄰兩輪都 |u| > thr 的比較次數（adjacent 的分母）
      boundary   跨相位切換的反號（只記在 WINDOW）
    各關節分開比。
    """
    flip = collections.defaultdict(collections.Counter)
    last = {}                 # j -> (sign, 上一輪是否 > thr)
    prev_ph = None
    for ph, u in useq:
        u = np.asarray(u, float)
        boundary = None
        if prev_ph is not None and ph != prev_ph:
            boundary = {j: v[0] for j, v in last.items()}
            last = {}
        for j in joints:
            if abs(u[j]) > thr:
                sg = float(np.sign(u[j]))
                if boundary is not None and j in boundary and boundary[j] != sg:
                    flip['WINDOW']['boundary'] += 1
                if j in last:
                    psg, adj = last[j]
                    if adj:
                        for g in (ph, 'WINDOW'):
                            flip[g]['adj_pairs'] += 1
                    if psg != sg:
                        for g in (ph, 'WINDOW'):
                            flip[g]['adjacent' if adj else 'gapped'] += 1
                last[j] = (sg, True)
            elif j in last:
                last[j] = (last[j][0], False)
        prev_ph = ph
    return flip


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
    prev_ph = None
    # 翻號分三類（各關節分開比，閾值 FLIP_THR rad/s）：
    #   adjacent   相鄰兩個求解輪都 > 閾值且符號相反（真正的週期間反轉）
    #   gapped     中間隔著近零命令的反轉（相位內）
    #   boundary   跨相位切換的反轉（另列，不算進任一相位）
    # adj_pairs = 相鄰兩輪都 > 閾值的有效比較次數（分母）
    useq = []                # (相位, 九維命令)，依求解輪順序
    stops = collections.Counter()
    n_fail = 0
    n_non_solve = collections.Counter()
    for L in log:
        t = float(L['sim_t'])
        if t > t_stop:
            break
        # 非求解列：只有**節點既定的合法等待理由**（horizon_replay_check.NON_SOLVE_REASONS）
        # 才跳過——不計入追蹤／計算／餘裕／命令增量，另計數。其他缺 solve_in 的列一律報錯，
        # 不泛化成「沒有 solve_in 就忽略」。（2026-10-05 修：先前遇到合法等待列即 KeyError）
        if 'solve_in' not in L:
            if L.get('reason') not in NON_SOLVE_REASONS:
                raise ValueError(f'sim_t {t}：缺 solve_in 且理由 {L.get("reason")!r} '
                                 '不是合法等待列')
            n_non_solve[L.get('reason')] += 1
            continue
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
            useq.append((ph, u))
            prev_u = u
            prev_ph = ph

    # ---- 逐物理步真值（實測角餘裕、TCP 速度）----
    c = room['steps_cols']
    i = c.index
    S = [s for s in room['steps'] if t_win0 <= s[1] <= t_stop]
    tcp_prev = None
    tcp_hist = []
    for k, s in enumerate(S):
        t = float(s[1])
        ph = phase_of(t)
        q = np.array(s[i('q_arm_meas')], float)
        m = np.minimum(q - elo, ehi - q)
        for g in (ph, 'WINDOW'):
            by[g]['meas_margin'].append(float(m.min()))
            by[g]['meas_margin_joint'].append(int(m.argmin()) + 1)
        # 逐物理步（10 ms）FK；另以 5 步（50 ms）區間平均速度作對照
        p = K.fk(np.r_[s[i('base_xyth')], q], args['tcp'])[:3, 3]
        tcp_hist.append((t, p))
        if tcp_prev is not None and t > tcp_prev[0]:
            v = float(np.linalg.norm(p - tcp_prev[1]) / (t - tcp_prev[0]))
            for g in (ph, 'WINDOW'):
                by[g]['tcp_speed'].append(v)
        if len(tcp_hist) > 5:
            t5, p5 = tcp_hist[-6]
            for g in (ph, 'WINDOW'):
                by[g]['tcp_speed_50ms'].append(
                    float(np.linalg.norm(p - p5) / (t - t5)))
        tcp_prev = (t, p)

    flip = count_reversals(useq, FLIP_THR)
    dur = {'WINDOW': t_stop - t_win0}
    for k_, (p_, t0_) in enumerate(bounds):
        t1_ = bounds[k_ + 1][1] if k_ + 1 < len(bounds) else t_stop
        dur[p_] = max(0.0, min(t1_, t_stop) - t0_)

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
                           'arm_reversals': {
                               'threshold_rad_s': FLIP_THR,
                               'adjacent': int(flip[g]['adjacent']),
                               'adjacent_valid_pairs': int(flip[g]['adj_pairs']),
                               'adjacent_rate_per_s': (
                                   round(flip[g]['adjacent'] / dur[g], 3)
                                   if dur.get(g) else None),
                               'gapped': int(flip[g]['gapped']),
                               'boundary': (int(flip[g]['boundary'])
                                            if g == 'WINDOW' else None),
                               'note': '各關節分開計；adjacent = 相鄰兩求解輪都 > 閾值且反號'},
                           'tcp_speed_10ms_mps': pct(b['tcp_speed'], (50, 95)),
                           'tcp_speed_50ms_avg_mps': pct(b['tcp_speed_50ms'], (50, 95))},
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
                   'n_non_solve_rows': dict(n_non_solve),
                   # 節點統計缺（例如中止路徑未寫出）⇒ 這三項標缺、不補造；其他指標照算
                   'n_missed_slot': (sol.get('stats') or {}).get('n_missed_slot'),
                   'n_warm_discard': (sol.get('stats') or {}).get('n_warm_discard'),
                   'n_shaped': (sol.get('stats') or {}).get('n_shaped'),
                   'stats_missing': not bool(sol.get('stats')),
                   'stats_missing_note': ('證據不足：align_solver.json 沒有節點統計，'
                                          'n_missed_slot／n_warm_discard／n_shaped 標缺'
                                          if not sol.get('stats') else None)},
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
