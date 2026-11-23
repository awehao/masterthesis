#!/usr/bin/env python3
"""c1_summarize／c0_summarize.analyse 的守門測試（判定工具以替身取代，不重播、不開模擬器）。

要守的事：
  - 協定問題或證據不足的趟次不得進配對差（各種條件分開測，避免一種掩蓋另一種）
  - 資料完整的任務 FAIL **必須保留**在配對差中
  - 缺開帶 ⇒ 逐趟記為證據問題、指標填空，不拋例外
  - 工具失敗 ⇒ 記為工具異常，不讀舊分析檔冒充本次結果
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c0_summarize as C0                                        # noqa: E402
import c1_summarize as C1                                        # noqa: E402


def good(N=5, physical='PASS'):
    return {'status': 'executed', 'tool_errors': [], 'evidence_issues': [], 'N': N,
            'replay': 'PASS', 'physical': physical, 'physical_insufficient': [],
            'physical_failed': [] if physical == 'PASS' else ['S3_grasp'], 'm': {'x': 1.0}}


# ---------------------------------------------------------------- 配對可用性
def test_clean_pair_usable():
    assert C1.pair_problems(good(5), good(1), 5, 1, 0) == []


def test_task_fail_with_complete_data_is_kept():
    """任務 FAIL 但紀錄完整 ⇒ 仍可用（不能只比成功趟次）。"""
    assert C1.pair_problems(good(5), good(1, physical='FAIL'), 5, 1, 0) == []


@pytest.mark.parametrize('mut,expect', [
    (lambda r: r.update(N=9), 'N=9 與排程'),
    (lambda r: r.update(N=None), 'N 缺失'),
    (lambda r: r.update(physical='INSUFFICIENT', physical_insufficient=['S1_open_hold']), '證據不足'),
    (lambda r: r.update(physical='FAIL', physical_failed=['S3_grasp'],
                        physical_insufficient=['S6_contact']), '證據不足'),
    (lambda r: r.update(tool_errors=['horizon_metrics.py 離開碼 1']), '工具異常'),
    (lambda r: r.update(replay='FAIL'), '重播 FAIL'),
    (lambda r: r.update(status='startup_failure'), '批次狀態 startup_failure'),
    (lambda r: r.update(physical=None), '物理判定 None'),
    (lambda r: r.update(evidence_issues=['指標缺值：window_s']), '指標缺值'),
])
def test_each_problem_excludes_pair(mut, expect):
    r1 = good(1)
    mut(r1)
    why = C1.pair_problems(good(5), r1, 5, 1, 0)
    assert why and any(expect in w for w in why), why


def test_problems_not_masked():
    """兩種問題同時存在 ⇒ 兩個都要列出。"""
    r1 = good(9)
    r1.update(physical='INSUFFICIENT', physical_insufficient=['S1_open_hold'])
    why = C1.pair_problems(good(5), r1, 5, 1, 0)
    assert any('N=9' in w for w in why) and any('證據不足' in w for w in why)


def test_config_mismatch_excludes():
    assert any('設定核對' in w for w in C1.pair_problems(good(5), good(1), 5, 1, 1))


def test_missing_run_excludes():
    assert C1.pair_problems(good(5), None, 5, 1, 0)


# ---------------------------------------------------------------- analyse 的穩健性
def _fake_run(tmp_path, rid):
    D = tmp_path / rid
    D.mkdir()
    (D / 'align_solver.json').write_text('{}')
    (D / 'task.json').write_text(json.dumps({'events': [{'phase': 'ALIGN', 'sim_t': 1.0},
                                                        {'phase': 'DONE', 'sim_t': 50.0}]}))
    return D


def _fake_sh(behaviour):
    def sh(args):
        tool = args[0]
        out = args[args.index('--out') + 1] if '--out' in args else None
        b = behaviour.get(tool, 'ok')
        if b == 'crash':
            return 1, ''
        if tool == 'horizon_replay_check.py':
            json.dump({'verdict': 'PASS', 'N_args': 5}, open(out, 'w'))
            return 0, ''
        if tool == 'motm_physical_check.py':
            if b == 'noband':
                json.dump({'verdict': 'INSUFFICIENT', 'failed': [],
                           'insufficient': ['S1_open_hold', 'S2_close_hold'],
                           'S1_open_hold': {'verdict': '證據不足：task.json 沒有有效的 open_band_m'},
                           'S2_close_hold': '沒有開保持區段，無從判關保持',
                           'S3_grasp': {'longest_both_fingers_s': 10.0},
                           'S4_drift': {'max_mm': 0.1, 'grip_fraction_in_window': 1.0},
                           'S5_home': {'dist_m': 0.15}}, open(out, 'w'))
                return 2, ''
        if tool == 'horizon_metrics.py':
            json.dump({'N': 5, 'window': {'from_sim_t': 1.0, 'to_sim_t': 31.0},
                       'by_phase': {'WINDOW': {}}, 'solver': {}}, open(out, 'w'))
            return 0, ''
        if tool == 'motm_metrics.py':
            return 0, json.dumps({'final_phase': 'DONE', 'abort': None})
        return 0, ''
    return sh


def test_missing_band_recorded_not_crash(tmp_path, monkeypatch):
    _fake_run(tmp_path, 'r_noband')
    monkeypatch.setattr(C0, 'RUNS', str(tmp_path))
    monkeypatch.setattr(C0, 'sh', _fake_sh({'motm_physical_check.py': 'noband'}))
    r = C0.analyse('r_noband', expect_open_m=0.1)
    assert r['physical'] == 'INSUFFICIENT' and r['m']['open_hold_s'] is None
    assert any('證據不足' in e for e in r['evidence_issues'])
    r['status'] = 'executed'
    assert any('證據不足' in w for w in C1.run_problems(r, 5))


def test_tool_failure_never_reads_stale_file(tmp_path, monkeypatch):
    D = _fake_run(tmp_path, 'r_stale')
    (D / 'analysis').mkdir()
    stale = {'N': 5, 'window': {'from_sim_t': 0, 'to_sim_t': 99}, 'by_phase': {'WINDOW': {}},
             'solver': {'n_cycles': 12345}}
    json.dump(stale, open(D / 'analysis' / 'horizon_metrics.json', 'w'))
    monkeypatch.setattr(C0, 'RUNS', str(tmp_path))
    monkeypatch.setattr(C0, 'sh', _fake_sh({'horizon_metrics.py': 'crash'}))
    r = C0.analyse('r_stale', expect_open_m=0.1)
    assert r['m']['n_cycles'] is None and r['m']['window_s'] is None
    assert any('horizon_metrics.py' in e for e in r['tool_errors'])
    assert not (D / 'analysis' / 'horizon_metrics.json').exists()
