"""分析框架的離線測試（合成資料，不開模擬器）。

涵蓋工作單「驗證與交付」要求的六個情境。
"""
from __future__ import annotations
import json, math, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from coman_run_analyze import (COMPLETE, INCOMPLETE, INSUFFICIENT,             # noqa
                               NOT_STARTED, analyze, longest_run, phase_windows)

SPEC_V1 = os.path.join(EVAL, 'results/specs/wb_coman_drawer20_criteria_v1.yaml')
COLS = ['t', 'phase', 'opening', 'e_par', 'e_perp', 'limit_margin',
        'base_vx', 'base_vy', 'attached'] + [f'joint{i}' for i in range(1, 7)]


def row(t, ph, op=0.0, bv=0.0, jr=0.0, att=1.0, j0=0.0):
    return [round(t, 4), ph, op, 0.0, 0.0, 1.0, bv, 0.0, att] + [j0] * 6


def make_run(tmp, log, cols=COLS, extra=None, spec_sim=None):
    d = os.path.join(tmp, 'run'); os.makedirs(os.path.join(d, 'sim'), exist_ok=True)
    sim = {'log_cols': cols, 'log': log, 'grasp_model': 'fixed_attachment',
           'stop_reason': 'sim_limit', 'sim_time_s': 10.0, 'wall_s': 12.0,
           'target_opening_used_m': 0.020,
           'base_fixation': {'mode': 'free_base'}, 'events': []}
    sim.update(extra or {})
    json.dump(sim, open(os.path.join(d, 'sim', 'drawer_run.json'), 'w'))
    return d


def spec_with_sim(tmp, base=0.005, jr=0.005, minc=None):
    """在暫存目錄造一份**加了同動門檻**的規格（不改動 v1 原檔）。"""
    import yaml
    s = yaml.safe_load(open(SPEC_V1, encoding='utf-8'))
    s['simultaneity'] = {'base_lin_min_mps': base, 'joint_rate_min_rps': jr}
    if minc is not None:
        s['simultaneity']['min_continuous_s'] = minc
    p = os.path.join(tmp, 'spec_sim.yaml')
    yaml.safe_dump(s, open(p, 'w', encoding='utf-8'), allow_unicode=True)
    return p


def ramp(t0, t1, ph, dt=0.01, bv=0.0, jstep=0.0, op=0.0, att=1.0, skip=()):
    out, t, j = [], t0, 0.0
    k = 0
    while t < t1 - 1e-9:
        if not any(a <= t < b for a, b in skip):
            out.append(row(t, ph, op, bv, att=att, j0=j))
        j += jstep * dt
        t += dt
        k += 1
    return out


def main() -> int:
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    tmp = tempfile.mkdtemp(prefix='coman_an_')
    try:
        out = os.path.join(tmp, 'out')

        # ---- 1 未進入任務／空資料 ----
        d = os.path.join(tmp, 'empty'); os.makedirs(d)
        r = analyze(d, SPEC_V1, out)
        check('1 未進入任務 → not_started', r['classification'] == NOT_STARTED)
        check('1 三題皆標「無證據」',
              all(v == '無證據' for v in r['三題'].values()))
        d2 = make_run(tmp, [])
        r = analyze(d2, SPEC_V1, out)
        check('1 空 log → insufficient_data 並拒絕計算',
              r['classification'] == INSUFFICIENT
              and any(m['code'] == 'no_log' for m in r['missing']))

        # ---- 6 必要欄位不明時明確拒絕 ----
        d3 = make_run(tmp, [[0.0, 'pull']], cols=['t', 'phase'])
        r = analyze(d3, SPEC_V1, out)
        check('6 缺 opening 欄 → 拒絕計算並具名',
              r['classification'] == INSUFFICIENT
              and any('opening' in m['detail'] for m in r['missing']))
        sp = spec_with_sim(tmp)
        log = ramp(0, 1, 'pull', bv=0.01, jstep=0.05, att=1.0)
        d4 = make_run(tmp, log, cols=[c for c in COLS if c != 'base_vx'])
        r = analyze(d4, sp, out)
        check('6 無底盤速度欄 → 同動 cannot_determine 且具名缺項',
              (r['simultaneity_pull'].get('verdict') == 'cannot_determine'
               and any(m['code'] == 'no_base_velocity' for m in r['missing'])))

        # ---- 同動門檻不在規格時 ----
        d5 = make_run(tmp, log)
        r = analyze(d5, SPEC_V1, out)
        check('門檻不在規格 → cannot_determine，不硬編',
              r['simultaneity_pull']['verdict'] == 'cannot_determine'
              and any(m['code'] == 'no_sim_thresholds' for m in r['missing']))

        # ---- 2 缺測造成同動中斷 ----
        full = ramp(0, 2.0, 'pull', bv=0.01, jstep=0.05)
        gap = ramp(0, 2.0, 'pull', bv=0.01, jstep=0.05, skip=((0.9, 1.3),))
        r_full = analyze(make_run(tmp, full), sp, out)
        r_gap = analyze(make_run(tmp, gap), sp, out)
        sf, sg = r_full['simultaneity_pull'], r_gap['simultaneity_pull']
        check('2 有缺測時最長連續同動**變短**',
              sg['longest_continuous_s'] < sf['longest_continuous_s'],
              f"  {sg['longest_continuous_s']} < {sf['longest_continuous_s']}")
        check('2 缺測被記為中斷，未跨越累積', sg['n_skipped_gaps'] >= 1,
              f"  n_skipped_gaps={sg['n_skipped_gaps']}")

        # ---- 3 命令非零但實測不動 ----
        still = ramp(0, 2.0, 'pull', bv=0.0, jstep=0.0)
        r = analyze(make_run(tmp, still), sp, out)
        s3 = r['simultaneity_pull']
        check('3 實測不動 → 同動時間為 0（命令不列入計算）',
              s3['total_simultaneous_s'] == 0.0
              and s3['longest_continuous_s'] == 0.0)
        check('3 仍有有效樣本（不是資料不足）', s3['n_valid_intervals'] > 0)

        # ---- 4 接近時同動、拉動時不同動 ----
        mixed = (ramp(0, 1.0, 'approach', bv=0.02, jstep=0.1)
                 + ramp(1.0, 1.2, 'engage', bv=0.0, jstep=0.0)
                 + ramp(1.2, 3.0, 'pull', bv=0.0, jstep=0.0))
        r = analyze(make_run(tmp, mixed), sp, out)
        s4 = r['simultaneity_pull']
        check('4 拉動段同動為 0（接近段的同動**未被計入**）',
              s4['total_simultaneous_s'] == 0.0,
              f"  拉動窗 {s4['window_dur_s']} s")
        check('4 接近段另列', 'approach' in (r.get('other_phase_motion') or {}))

        # ---- 5 任務失敗仍保留在比較表 ----
        incomplete = ramp(0, 1.0, 'approach') + ramp(1.0, 2.0, 'pull')
        r_inc = analyze(make_run(tmp, incomplete), sp, out)
        check('5 缺相位 → task_incomplete（**不是**丟棄）',
              r_inc['classification'] in (INCOMPLETE, INSUFFICIENT))
        sys.path.insert(0, HERE)
        from coman_run_plots import fig_compare
        fd = os.path.join(tmp, 'fig'); os.makedirs(fd, exist_ok=True)
        r_ok = analyze(make_run(tmp, full), sp, out)
        p = fig_compare([r_ok, r_inc,
                         {'classification': NOT_STARTED,
                          'inputs': {'run_dir': 'x'},
                          'run': {'base_mode': 'importer_fix_base'}}], fd)
        check('5 比較圖含失敗趟次仍能產生', os.path.exists(p))

        # ---- 完整資料的分類 ----
        comp = (ramp(0, .5, 'approach') + ramp(.5, .7, 'engage')
                + ramp(.7, 2.7, 'pull', bv=0.01, jstep=0.05)
                + ramp(2.7, 4.7, 'hold') + ramp(4.7, 5.0, 'release')
                + ramp(5.0, 6.0, 'retreat'))
        r = analyze(make_run(tmp, comp), sp, out)
        check('完整相位 → complete_data', r['classification'] == COMPLETE,
              f"  {r['classification']}")
        check('同動只算拉動窗（2.0 s）',
              abs(r['simultaneity_pull']['window_dur_s'] - 2.0) < 0.02,
              f"  {r['simultaneity_pull']['window_dur_s']}")

        # ---- 輸出不得寫進趟次目錄 ----
        rd = make_run(tmp, comp)
        check('輸出目錄在趟次目錄內會被拒絕（主程式檢查）',
              os.path.abspath(os.path.join(rd, 'a')).startswith(
                  os.path.abspath(rd) + os.sep))

        # ---- 輔助函式 ----
        ts = [0.0, 0.01, 0.02, 0.5, 0.51]
        fl = [True] * 5
        best, tot, br = longest_run(ts, fl, 0.15)
        check('longest_run：空隙處中斷不跨越', abs(best - 0.02) < 1e-9 and br == 1,
              f'  best={best:.3f} breaks={br}')
        w = phase_windows([0, .01, .02, .5, .51], ['a', 'a', 'a', 'a', 'a'], 0.15)
        check('phase_windows：同相位跨空隙也切開', len(w) == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('分析框架離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
