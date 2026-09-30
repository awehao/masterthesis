"""單趟分析：分類趟次狀態、按相位統計、只在已連接的 PULL 區間算同動。

**不覆寫原始資料、不新增正式通過標準。** 判定門檻一律讀適用規格；
讀不到就標示無法判定，不硬編新值。

用法
----
    python3 evaluation/analysis/coman_run_analyze.py \\
        --run evaluation/runs/<RUN_ID> \\
        --spec evaluation/results/specs/wb_coman_drawer20_criteria_v1.yaml \\
        --out evaluation/analysis/out/<RUN_ID>

輸出
----
`<out>/analysis.json`（機器可讀）與 `<out>/summary.md`（人讀）。
兩者都記錄輸入來源、sha、評估窗、有效樣本數與缺失原因。

分類
----
`not_started`      未進入任務（沒有趟次資料）
`insufficient_data` 有資料但必要欄位／時間基準不明，**拒絕計算**
`task_incomplete`   進了任務但未走完流程
`complete_data`     具備完整分析資料

這四類**都不是**正式通過判定 —— 正式資格由 v1 狀態機在趟次中判定，
本檔只描述「能不能分析」與「分析到什麼」。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
WS = os.path.dirname(EVAL)
sys.path.insert(0, EVAL)

NOT_STARTED = 'not_started'
INSUFFICIENT = 'insufficient_data'
INCOMPLETE = 'task_incomplete'
COMPLETE = 'complete_data'

# 同動需要的實測量。**命令與設定點不得代替。**
SIM_NEEDS = ('底盤實測線速度或可定向的位置序列', '手臂實測關節角或關節速度')


def sha16(path):
    try:
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
    except OSError:
        return None


def load_json(path):
    try:
        return json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        return None


class Missing(list):
    """缺失原因集合；**不補零**，只記錄。"""

    def add(self, code, detail):
        rec = {'code': code, 'detail': detail}
        if rec not in self:          # 同一原因只記一次
            self.append(rec)
        return None


def phase_windows(ts, phases, max_gap_s):
    """連續相同相位的區間 [(phase, t0, t1, n, gaps)]。

    **跨越資料空隙不合併** —— 空隙大於 max_gap_s 就切開，並記錄。
    """
    out = []
    if not ts:
        return out
    cur = {'phase': phases[0], 't0': ts[0], 't1': ts[0], 'n': 1, 'gaps': 0}
    for i in range(1, len(ts)):
        gap = ts[i] - ts[i - 1]
        if phases[i] != cur['phase'] or gap > max_gap_s:
            cur['gaps'] += int(gap > max_gap_s and phases[i] == cur['phase'])
            out.append(cur)
            cur = {'phase': phases[i], 't0': ts[i], 't1': ts[i], 'n': 1,
                   'gaps': 0}
        else:
            cur['t1'] = ts[i]
            cur['n'] += 1
    out.append(cur)
    return out


def longest_run(ts, flags, max_gap_s):
    """最長連續為真的時間長度。**空隙處中斷，不跨越累積。**

    回傳 (最長秒數, 為真的總秒數, 被空隙中斷次數)。
    """
    best = cur = total = 0.0
    breaks = 0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt > max_gap_s:
            breaks += 1
            cur = 0.0
            continue
        if flags[i] and flags[i - 1]:
            cur += dt
            total += dt
            best = max(best, cur)
        else:
            cur = 0.0
    return best, total, breaks


def stats(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return None
    v2 = sorted(v)
    n = len(v2)
    q = lambda p: v2[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {'n': n, 'min': v2[0], 'p50': q(0.5), 'p95': q(0.95), 'max': v2[-1],
            'rms': math.sqrt(sum(x * x for x in v) / n)}


# --------------------------------------------------------------- 主分析
def analyze(run_dir, spec_path, out_dir):
    miss = Missing()
    rep = {'schema': 'coman_run_analysis/1',
           'inputs': {'run_dir': run_dir, 'spec': spec_path,
                      'spec_sha256_16': sha16(spec_path)},
           'time_bases': {'task': 'sim', 'cost': 'wall',
                          'note': '模擬時間與牆鐘分開；不相加、不互換'},
           'not_a_pass_criterion': ('本檔的分類與統計**不是**正式通過判定；'
                                    '正式資格由 v1 狀態機於趟次中判定')}

    sim = load_json(os.path.join(run_dir, 'sim', 'drawer_run.json'))
    pre = load_json(os.path.join(run_dir, 'preflight.json'))
    outcome = os.path.join(run_dir, 'OUTCOME.md')
    rep['inputs']['preflight'] = pre
    rep['inputs']['has_outcome_note'] = os.path.exists(outcome)
    rep['inputs']['drawer_run_sha256_16'] = sha16(
        os.path.join(run_dir, 'sim', 'drawer_run.json'))

    if sim is None:
        rep['classification'] = NOT_STARTED
        miss.add('no_run_data', 'sim/drawer_run.json 不存在或無法解析 ⇒ 未進入任務')
        rep['missing'] = list(miss)
        rep['三題'] = {k: '無證據' for k in
                       ('完整操作', '拉動期間同動', '執行期時效')}
        return rep

    rep['run'] = {k: sim.get(k) for k in
                  ('grasp_model', 'stop_reason', 'sim_time_s', 'wall_s',
                   'target_opening_used_m', 'target_opening_case_m',
                   'temp_max_c', 'cpu_limit_c', 'monitor_failure')}
    bf = sim.get('base_fixation') or {}
    rep['run']['base_mode'] = bf.get('mode')
    rep['run']['asset_sha'] = sim.get('sha256_16')
    if sim.get('sim_time_s') and sim.get('wall_s'):
        rep['run']['rtf'] = round(float(sim['sim_time_s'])
                                  / max(float(sim['wall_s']), 1e-9), 4)

    # ---- 規格：只讀，不新增門檻 ----
    spec = None
    try:
        import yaml
        spec = yaml.safe_load(open(spec_path, encoding='utf-8'))
    except Exception as exc:                                  # noqa: BLE001
        miss.add('spec_unreadable', f'{spec_path}：{exc}')
    thr = {}
    if spec:
        thr['target_stroke_m'] = (spec.get('profile') or {}).get('target_stroke_m')
        pp = spec.get('P_pull') or {}
        thr['P1_final_opening_err_m_max'] = pp.get('P1_final_opening_err_m_max')
        thr['P1_hold_s'] = pp.get('P1_hold_s')
        thr['P6_retreat_clear_m'] = pp.get('P6_retreat_clear_m')
        sm = (spec.get('simultaneity') or spec.get('S_sim') or {})
        thr['sim_base_lin_min_mps'] = sm.get('base_lin_min_mps')
        thr['sim_joint_rate_min_rps'] = sm.get('joint_rate_min_rps')
        thr['sim_min_continuous_s'] = sm.get('min_continuous_s')
        cont = spec.get('continuity') or {}
        thr['max_gap_s'] = cont.get('max_gap_s')
    rep['thresholds_from_spec'] = thr
    max_gap = thr.get('max_gap_s')
    if max_gap is None:
        max_gap = 0.05
        miss.add('max_gap_default',
                 'spec 無 continuity.max_gap_s；分析用 0.05 s 切開資料空隙'
                 '（**僅用於切窗，不是判定門檻**）')

    log = sim.get('log') or []
    cols = sim.get('log_cols') or []
    if not log or not cols:
        rep['classification'] = INSUFFICIENT
        miss.add('no_log', 'log 或 log_cols 為空 ⇒ 拒絕計算')
        rep['missing'] = list(miss)
        return rep
    ix = {c: i for i, c in enumerate(cols)}
    for need in ('t', 'phase', 'opening'):
        if need not in ix:
            miss.add('missing_column', f'log_cols 缺 {need} ⇒ 拒絕計算')
    if any(m['code'] == 'missing_column' for m in miss):
        rep['classification'] = INSUFFICIENT
        rep['missing'] = list(miss)
        return rep

    get = lambda r, c: (float(r[ix[c]]) if c in ix and r[ix[c]] is not None
                        and isinstance(r[ix[c]], (int, float)) else None)
    ts = [float(r[ix['t']]) for r in log]
    phs = [str(r[ix['phase']]) for r in log]
    rep['samples'] = {'n_rows': len(log), 'sim_t_first': ts[0],
                      'sim_t_last': ts[-1],
                      'median_dt_s': (sorted(ts[i] - ts[i - 1]
                                             for i in range(1, len(ts)))
                                      [max(0, (len(ts) - 1) // 2)]
                                      if len(ts) > 1 else None)}

    # ---- 相位窗 ----
    wins = phase_windows(ts, phs, max_gap)
    rep['phase_windows'] = [{'phase': w['phase'], 't0': round(w['t0'], 4),
                             't1': round(w['t1'], 4),
                             'dur_s': round(w['t1'] - w['t0'], 4),
                             'n': w['n'], 'gap_splits': w['gaps']}
                            for w in wins]
    seen = [w['phase'] for w in wins]
    rep['phases_seen'] = sorted(set(seen))

    # ---- 按相位統計 ----
    per = {}
    for w in wins:
        rows = [r for r, t in zip(log, ts) if w['t0'] <= t <= w['t1']]
        d = {'dur_s': round(w['t1'] - w['t0'], 4), 'n': len(rows)}
        for c, label in (('e_par', 'tcp_err_parallel_m'),
                         ('e_perp', 'tcp_err_perp_m'),
                         ('rot_conv_err_deg', 'rot_conv_err_deg'),
                         ('limit_margin', 'joint_limit_margin_rad'),
                         ('opening', 'opening_m'),
                         ('f_norm', 'wrist_force_n')):
            st = stats([get(r, c) for r in rows]) if c in ix else None
            if st:
                d[label] = {k: round(v, 6) for k, v in st.items()}
            elif c in ix:
                miss.add('all_nan', f'相位 {w["phase"]} 的 {c} 全為無效值')
        # 逐關節餘裕的最小值與對應關節
        jn = [f'joint{i}' for i in range(1, 7)]
        if all(j in ix for j in jn):
            lo = hi = None
            try:
                sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
                from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as LIM
                lo, hi = list(LIM.lower), list(LIM.upper)
            except Exception as exc:                          # noqa: BLE001
                miss.add('no_joint_limits',
                         f'取不到關節限位（{type(exc).__name__}）⇒ 不算逐關節餘裕')
            if lo:
                worst, wj = None, None
                for r in rows:
                    for k, j in enumerate(jn):
                        q = get(r, j)
                        if q is None:
                            continue
                        m = min(q - lo[k], hi[k] - q)
                        if worst is None or m < worst:
                            worst, wj = m, j
                if worst is not None:
                    d['joint_limit_margin_min'] = {'rad': round(worst, 6),
                                                   'joint': wj}
        per.setdefault(w['phase'], []).append(d)
    rep['per_phase'] = per

    # ---- 同動：**只看已確認連接且處於 PULL 的區間** ----
    sim_rep = {'window_definition':
               '已確認連接（attached）且相位為 pull 的交集；接近／保持／退出另列',
               'measured_only': '使用實測運動；命令與設定點不得代替'}
    # **要成對才算數**：只有 base_vy（或只有 base_x）算不出線速度。
    # 先前用 any(...) 太寬鬆，會讓缺一半欄位的趟次被當成可判定。
    have_base_vel = (('base_vx' in ix and 'base_vy' in ix)
                     or ('base_x' in ix and 'base_y' in ix))
    have_joint = all(f'joint{i}' in ix for i in range(1, 7))
    thr_ok = (thr.get('sim_base_lin_min_mps') is not None
              and thr.get('sim_joint_rate_min_rps') is not None)
    if not thr_ok:
        sim_rep['verdict'] = 'cannot_determine'
        sim_rep['reason'] = (
            f'適用規格 {os.path.basename(spec_path)} 未提供同動門檻'
            '（simultaneity.base_lin_min_mps／joint_rate_min_rps／'
            'min_continuous_s）⇒ **無法判定，不硬編新值**。'
            '見 FIELD_INVENTORY G7：同動門檻目前不在任何已核准規格中。')
        miss.add('no_sim_thresholds', sim_rep['reason'])
    if not have_base_vel:
        sim_rep.setdefault('verdict', 'cannot_determine')
        sim_rep['missing_base_motion'] = (
            '趟次無底盤實測速度或可定向位置欄位（只有 base_drift 距離純量與'
            'base_dyaw_deg 絕對值）⇒ **無法判定同動**，見 FIELD_INVENTORY G3')
        miss.add('no_base_velocity', sim_rep['missing_base_motion'])
    if not have_joint:
        sim_rep.setdefault('verdict', 'cannot_determine')
        miss.add('no_joint_positions', '無 joint1…joint6 ⇒ 無法得手臂速率')
    if sim_rep.get('verdict') != 'cannot_determine':
        pull = [(t, r) for t, p, r in zip(ts, phs, log) if p == 'pull']
        att = ('attached' in ix)
        if att:
            pull = [(t, r) for t, r in pull if get(r, 'attached')]
        else:
            sim_rep['attached_source'] = (
                'log 無 attached 欄；改以 events 的 engage 時間界定')
            ev_t = None
            for e in (sim.get('events') or []):
                if isinstance(e, dict) and 'engage' in str(e.get('event', '')):
                    ev_t = float(e.get('sim_t'))
                    break
            if ev_t is None:
                sim_rep['verdict'] = 'cannot_determine'
                sim_rep['reason'] = '找不到連接確認時間 ⇒ 無法界定拉動同動窗'
                miss.add('no_attach_time', sim_rep['reason'])
            else:
                pull = [(t, r) for t, r in pull if t >= ev_t]
                sim_rep['attach_sim_t'] = ev_t
        if sim_rep.get('verdict') != 'cannot_determine' and pull:
            pt = [t for t, _ in pull]
            jn = [f'joint{i}' for i in range(1, 7)]
            rates, blin = [], []
            for i in range(1, len(pull)):
                dt = pt[i] - pt[i - 1]
                if dt <= 0 or dt > max_gap:
                    rates.append(None)
                    blin.append(None)
                    continue
                qs0 = [get(pull[i - 1][1], j) for j in jn]
                qs1 = [get(pull[i][1], j) for j in jn]
                rates.append(None if None in qs0 or None in qs1
                             else max(abs(b - a) / dt
                                      for a, b in zip(qs0, qs1)))
                bx0, bx1 = get(pull[i - 1][1], 'base_x'), get(pull[i][1], 'base_x')
                by0, by1 = get(pull[i - 1][1], 'base_y'), get(pull[i][1], 'base_y')
                vx = get(pull[i][1], 'base_vx')
                vy = get(pull[i][1], 'base_vy')
                if vx is not None and vy is not None:
                    blin.append(math.hypot(vx, vy))
                elif None not in (bx0, bx1, by0, by1):
                    blin.append(math.hypot(bx1 - bx0, by1 - by0) / dt)
                else:
                    blin.append(None)
            bt = float(thr['sim_base_lin_min_mps'])
            jt = float(thr['sim_joint_rate_min_rps'])
            flags = [False] + [(b is not None and r is not None
                                and b > bt and r > jt)
                               for b, r in zip(blin, rates)]
            best, total, breaks = longest_run(pt, flags, max_gap)
            span = pt[-1] - pt[0] if len(pt) > 1 else 0.0
            sim_rep.update({
                'verdict': 'computed',
                'window_sim_t': [round(pt[0], 4), round(pt[-1], 4)],
                'window_dur_s': round(span, 4),
                'n_samples': len(pt),
                'n_valid_intervals': sum(1 for b, r in zip(blin, rates)
                                         if b is not None and r is not None),
                'n_skipped_gaps': breaks,
                'thresholds': {'base_lin_min_mps': bt,
                               'joint_rate_min_rps': jt},
                'longest_continuous_s': round(best, 4),
                'total_simultaneous_s': round(total, 4),
                'fraction_of_window': (round(total / span, 4) if span > 0
                                       else None),
                'joint_rate_source': ('關節角差分（無實測關節速度，'
                                      'FIELD_INVENTORY G4）'),
            })
            if thr.get('sim_min_continuous_s') is not None:
                sim_rep['meets_spec_min_continuous'] = (
                    best >= float(thr['sim_min_continuous_s']))
        elif sim_rep.get('verdict') != 'cannot_determine':
            sim_rep['verdict'] = 'no_pull_window'
            miss.add('no_pull_window', '沒有「已連接且 pull」的樣本')
    rep['simultaneity_pull'] = sim_rep

    # 接近／保持／退出另列（不併入拉動占比）
    rep['other_phase_motion'] = {
        p: {'dur_s': round(sum(w['dur_s'] for w in v), 4)}
        for p, v in ((p, per.get(p, [])) for p in
                     ('approach', 'hold', 'release', 'retreat'))
        if v}

    # ---- 耗時（wall）----
    cost = {}
    dr = load_json(os.path.join(run_dir, 'diag_record.json'))
    if dr and isinstance(dr.get('cols'), dict):
        for tag in ('dist', 'safety'):
            c = dr['cols'].get(tag)
            rows = (dr.get('data') or {}).get(tag) or []
            if c is None:
                miss.add('diag_cols_missing',
                         f'{tag} 的欄名名單未收到（上游 ~/diag_fields）⇒ '
                         f'**拒絕按位置猜**')
                continue
            cix = {n: i for i, n in enumerate(c)}
            for nm in ('node_cycle_ms', 'node_ms', 'filter_ms'):
                if nm in cix:
                    st = stats([float(r[cix[nm]]) for r in rows
                                if cix[nm] < len(r)])
                    if st:
                        cost[f'{tag}.{nm}'] = {k: round(v, 3)
                                               for k, v in st.items()}
    else:
        miss.add('no_diag_record', 'diag_record.json 不存在 ⇒ 無節點耗時')
    sol = load_json(os.path.join(run_dir, 'solver_out.json'))
    if sol and sol.get('log'):
        st = stats([float(r.get('solve_ms')) for r in sol['log']
                    if r.get('solve_ms') is not None])
        if st:
            cost['solver.solve_ms'] = {k: round(v, 3) for k, v in st.items()}
        cost['solver.cycles'] = len(sol['log'])
        cost['solver.completed'] = sol.get('completed')
    else:
        miss.add('no_solver_out', 'solver_out.json 不存在或無 log ⇒ 無求解耗時')
    ch = sim.get('coman_chain9') or {}
    if ch:
        cost['chain9'] = {k: ch.get(k) for k in
                          ('received', 'rejected', 'frozen_steps',
                           'last_reject')}
        wl = ch.get('wheel_limit') or {}
        cost['chain9_wheel'] = {k: wl.get(k) for k in
                                ('modified_steps', 'timeout_decel_steps',
                                 'stop_unverified_steps')}
    rep['cost_wall'] = cost

    # ---- 依值配對的延遲估計 ----
    e2e = sim.get('coman_e2e_log') or []
    ec = sim.get('coman_e2e_cols') or []
    if e2e and ec:
        eix = {n: i for i, n in enumerate(ec)}
        key = 'e2e_age_s_value_paired_estimate'
        ages = [float(r[eix[key]]) for r in e2e
                if key in eix and eix[key] < len(r)
                and isinstance(r[eix[key]], (int, float))
                and math.isfinite(float(r[eix[key]]))]
        npair = sum(1 for r in e2e if 'pairing' in eix
                    and str(r[eix['pairing']]) == 'value_paired_estimate')
        nun = sum(1 for r in e2e if 'pairing' in eix
                  and str(r[eix['pairing']]) == 'unpaired')
        cand = stats([float(r[eix['n_candidates']]) for r in e2e
                      if 'n_candidates' in eix and eix['n_candidates'] < len(r)])
        rep['latency_value_paired_estimate'] = {
            'label': '**依值配對的延遲估計**，不是已證明同一筆命令',
            'n_rows': len(e2e), 'n_paired': npair, 'n_unpaired': nun,
            'age_s': ({k: round(v, 5) for k, v in stats(ages).items()}
                      if stats(ages) else None),
            'n_candidates': cand,
            'not_allowed': '不得單獨用它宣告端到端時效通過'}
    else:
        miss.add('no_e2e_log', 'coman_e2e_log 不存在 ⇒ 無延遲估計')

    # ---- 分類 ----
    need_phases = {'approach', 'pull', 'hold', 'release', 'retreat'}
    got = set(rep['phases_seen'])
    if sim_rep.get('verdict') == 'cannot_determine' or any(
            m['code'] in ('no_base_velocity', 'no_sim_thresholds',
                          'diag_cols_missing') for m in miss):
        rep['classification'] = INSUFFICIENT
    elif not need_phases <= got:
        rep['classification'] = INCOMPLETE
        rep['incomplete_detail'] = {'missing_phases': sorted(need_phases - got)}
    else:
        rep['classification'] = COMPLETE

    rep['三題'] = {
        '完整操作': ('見 classification 與 phase_windows；'
                     '**正式通過由 v1 狀態機判定，本檔不替代**'),
        '拉動期間同動': sim_rep.get('verdict'),
        '執行期時效': ('cost_wall 與 latency_value_paired_estimate；'
                       '各節點分開，不相加當作端到端驗收'),
    }
    rep['missing'] = list(miss)
    return rep


def to_md(rep):
    L = []
    w = L.append
    w(f"# 單趟分析：{os.path.basename(rep['inputs']['run_dir'])}\n")
    w(f"**分類：`{rep['classification']}`**  ")
    w(f"（分類不是正式通過判定 —— {rep['not_a_pass_criterion']}）\n")
    w('## 輸入與版本\n')
    w('| 項目 | 值 |')
    w('|---|---|')
    for k, v in rep['inputs'].items():
        w(f'| `{k}` | {v} |')
    if 'run' in rep:
        w('\n## 趟次\n')
        w('| 項目 | 值 |')
        w('|---|---|')
        for k, v in rep['run'].items():
            w(f'| `{k}` | {v} |')
    if rep.get('phase_windows'):
        w('\n## 相位窗（sim）\n')
        w('| 相位 | t0 | t1 | 長度 s | 樣本 | 空隙切分 |')
        w('|---|---|---|---|---|---|')
        for x in rep['phase_windows']:
            w(f"| {x['phase']} | {x['t0']} | {x['t1']} | {x['dur_s']} "
              f"| {x['n']} | {x['gap_splits']} |")
    s = rep.get('simultaneity_pull') or {}
    w('\n## 拉動期間同動（**只含已連接且 pull**）\n')
    w(f"判定：**{s.get('verdict')}**  ")
    if s.get('reason'):
        w(f"原因：{s['reason']}  ")
    if s.get('missing_base_motion'):
        w(f"缺項：{s['missing_base_motion']}  ")
    if s.get('verdict') == 'computed':
        w('')
        w('| 量 | 值 |')
        w('|---|---|')
        for k in ('window_sim_t', 'window_dur_s', 'n_samples',
                  'n_valid_intervals', 'n_skipped_gaps', 'thresholds',
                  'longest_continuous_s', 'total_simultaneous_s',
                  'fraction_of_window', 'joint_rate_source'):
            if k in s:
                w(f'| `{k}` | {s[k]} |')
    if rep.get('cost_wall'):
        w('\n## 耗時（**wall**，各節點分開，不相加）\n')
        w('| 項目 | 值 |')
        w('|---|---|')
        for k, v in rep['cost_wall'].items():
            w(f'| `{k}` | {v} |')
    if rep.get('latency_value_paired_estimate'):
        e = rep['latency_value_paired_estimate']
        w('\n## 延遲估計\n')
        w(f"{e['label']}。{e['not_allowed']}\n")
        w('| 量 | 值 |')
        w('|---|---|')
        for k in ('n_rows', 'n_paired', 'n_unpaired', 'age_s', 'n_candidates'):
            w(f'| `{k}` | {e.get(k)} |')
    w('\n## 缺失原因（**不補零**）\n')
    if rep.get('missing'):
        w('| 代號 | 說明 |')
        w('|---|---|')
        for m in rep['missing']:
            w(f"| `{m['code']}` | {m['detail']} |")
    else:
        w('無')
    w('\n## 三題\n')
    w('| 問題 | 本趟 |')
    w('|---|---|')
    for k, v in rep['三題'].items():
        w(f'| {k} | {v} |')
    return '\n'.join(L) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='趟次目錄（唯讀）')
    ap.add_argument('--spec', default=os.path.join(
        EVAL, 'results/specs/wb_coman_drawer20_criteria_v1.yaml'))
    ap.add_argument('--out', required=True, help='**獨立**分析輸出目錄')
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath(a.run) + os.sep):
        print('拒絕：輸出目錄在趟次目錄內，會污染原始資料', file=sys.stderr)
        return 2
    rep = analyze(a.run, a.spec, a.out)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, 'analysis.json'), 'w'),
              ensure_ascii=False, indent=1)
    open(os.path.join(a.out, 'summary.md'), 'w',
         encoding='utf-8').write(to_md(rep))
    print(f"分類 {rep['classification']}；同動 "
          f"{(rep.get('simultaneity_pull') or {}).get('verdict')}；"
          f"缺失 {len(rep.get('missing') or [])} 項 → {a.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
