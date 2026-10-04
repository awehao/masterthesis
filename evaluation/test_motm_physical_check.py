#!/usr/bin/env python3
"""motm_physical_check 的守門測試：合成 room_run／task／align_solver，逐一注入反例。

時間軸（dt 0.01 s）：0–1 接近、1 夾上（attach）、1–3 拉開到 200 mm、3–6 開保持、
6–8 關回、8–11 關保持、11 RELEASE_WAIT、11–12 退開。底盤隨開度後退 ⇒ TCP 相對把手不動。
"""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motm_physical_check as PC                                 # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
DT = 0.01
X0, Y0, TH = -0.1, 0.9, 1.5708
Q = [0.0, 0.5, 1.2, 0.0, 0.8, 0.0]
NAMES = ['base_link', 'left_finger', 'right_finger', 'link6']


def opening(t):
    if t < 1.0:
        return 0.0
    if t < 3.0:
        return 0.2 * (t - 1.0) / 2.0
    if t < 6.0:
        return 0.2
    if t < 8.0:
        return 0.2 * (8.0 - t) / 2.0
    return 0.0


def build():
    cols = ['step', 'sim_t', 'owner', 'kind', 'base_cmd_body', 'meas_vb_body',
            'applied_arm', 'base_xyth', 'q_arm_meas', 'opening_m', 'q_finger',
            'finger_contact_n', 'contact_held_s', 'drawer_pairs', 'drawer_net_n']
    steps = []
    held = 0.0
    for k in range(1201):
        t = round((k + 1) * DT, 6)
        op = opening(t)
        fc = [1.2, 1.1] if 0.9 <= t <= 11.0 else ([0.0, 0.0] if t > 0.5 else None)
        held = held + DT if fc and min(fc) >= 0.5 else 0.0
        pairs = ([[1, 2, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0], [2, 2, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0]]
                 if fc and min(fc) >= 0.5 else [])
        steps.append([k, t, 1, 'wb', [0, 0, 0], [0, 0, 0], None,
                      [X0, Y0 - op, TH], list(Q), op, [0.01, 0.01], fc,
                      round(held, 6), pairs, 0.0])
    room = {'steps_cols': cols, 'steps': steps,
            'drawer_pairs': {'bodies': NAMES},
            'config': {'physics_dt': DT, 'start_pose': f'{X0},{Y0},{TH}'}}
    task = {'events': [{'sim_t': 0.5, 'phase': 'ALIGN'}, {'sim_t': 1.0, 'attached': True},
                       {'sim_t': 1.0, 'phase': 'OPEN'}, {'sim_t': 3.0, 'phase': 'OPEN_HOLD'},
                       {'sim_t': 6.0, 'phase': 'CLOSE'}, {'sim_t': 8.0, 'phase': 'CLOSE_HOLD'},
                       {'sim_t': 11.0, 'phase': 'RELEASE_WAIT'},
                       {'sim_t': 11.1, 'phase': 'RETREAT'}]}
    sol = {'args': {'urdf': URDF, 'tcp': 'link_tcp'}}
    return room, task, sol


@pytest.fixture(scope='module')
def base():
    return build()


def run(tmp_path, name, room, task, sol):
    d = tmp_path / name
    d.mkdir()
    for fn, obj in (('room_run.json', room), ('task.json', task),
                    ('align_solver.json', sol)):
        json.dump(obj, open(d / fn, 'w'))
    return PC.check(str(d))


def col(room, name):
    return room['steps_cols'].index(name)


# ---------------------------------------------------------------- 正例
def test_clean_passes(tmp_path, base):
    out, rc = run(tmp_path, 'ok', *copy.deepcopy(base))
    assert rc == 0 and out['verdict'] == 'PASS', out['results']
    assert out['S4_drift']['max_mm'] < 1e-6
    assert out['data']['n_gaps'] == 0
    # 早段雙指力未取得要明列，但不判失敗
    assert out['data']['missing_or_nonfinite']['finger_contact_n']['n'] == 50


# ---------------------------------------------------------------- 時間缺口
def _two_short_holds(room, gap_kind):
    """開保持只留兩段各 1 s，中間 5 s 沒資料。"""
    ko = col(room, 'opening_m')
    S = room['steps']
    keep = []
    for s in S:
        t = s[1]
        if 3.0 <= t < 6.0:
            continue                         # 原保持段整段拿掉，下面重建
        keep.append(s)
    seg = []
    proto = next(s for s in S if abs(s[1] - 3.5) < 1e-9)
    for j in range(100):                     # 第一段 1 s：t 3.00–3.99
        r = copy.deepcopy(proto); r[1] = round(3.0 + j * DT, 6); r[ko] = 0.2
        seg.append(r)
    for j in range(100):                     # 第二段 1 s：t 9.00–9.99（中間 5 s 無資料）
        r = copy.deepcopy(proto); r[1] = round(9.0 + j * DT, 6); r[ko] = 0.2
        seg.append(r)
    before = [s for s in keep if s[1] < 3.0]
    after = [copy.deepcopy(s) for s in keep if s[1] >= 6.0]
    for s in after:
        s[1] = round(s[1] + 4.0, 6)           # 後段時間整體後移，與第二段銜接
    new = before + seg + after
    for k, s in enumerate(new):
        s[0] = k if gap_kind == 'time_only' else s[0]
    if gap_kind == 'both':
        for k, s in enumerate(new):
            s[0] = int(round(s[1] / DT)) - 1
    room['steps'] = new
    return room


@pytest.mark.parametrize('gap_kind', ['time_only', 'both'])
def test_gap_breaks_hold(tmp_path, base, gap_kind):
    room, task, sol = copy.deepcopy(base)
    room = _two_short_holds(room, gap_kind)
    for e in task['events']:
        if e['sim_t'] >= 6.0:
            e['sim_t'] += 4.0
    out, rc = run(tmp_path, f'gap_{gap_kind}', room, task, sol)
    assert out['results']['S1_open_hold'] is False, out['S1_open_hold']
    # 第一段含拉開末端 2.95–3.00 s 已入帶的 5 步 ⇒ 1.05 s；兩段都 < 2 s 才對
    assert out['S1_open_hold']['longest_s'] == pytest.approx(1.05, abs=0.011)
    assert out['data']['n_gaps'] >= 1 and rc == 1


def test_step_number_gap_alone_breaks(tmp_path, base):
    """時間連續但步號跳號（中間步遺失後被重新蓋時間）也算缺口。"""
    room, task, sol = copy.deepcopy(base)
    for s in room['steps']:
        if s[1] >= 4.5:
            s[0] += 7
    out, rc = run(tmp_path, 'stepgap', room, task, sol)
    # 2.95–4.49 s（含入帶 5 步）= 1.55 s；沒有缺口時是 3.1 s
    assert out['S1_open_hold']['longest_s'] == pytest.approx(1.55, abs=0.011)
    assert out['results']['S1_open_hold'] is False


def test_missing_opening_breaks_hold(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    ko = col(room, 'opening_m')
    for s in room['steps']:
        if abs(s[1] - 4.5) < 0.005:
            s[ko] = None
    out, _ = run(tmp_path, 'opnone', room, task, sol)
    assert out['S1_open_hold']['longest_s'] < 2.0
    assert out['results']['S1_open_hold'] is False


# ---------------------------------------------------------------- 接觸證據
def test_contact_all_none_insufficient(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    k = col(room, 'drawer_pairs')
    for s in room['steps']:
        s[k] = None
    out, rc = run(tmp_path, 'cnone', room, task, sol)
    assert out['results']['S6_contact'] is None and rc == 2
    assert out['verdict'] == 'INSUFFICIENT'


def test_contact_one_step_none_insufficient(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    room['steps'][700][col(room, 'drawer_pairs')] = None
    out, rc = run(tmp_path, 'c1none', room, task, sol)
    assert out['results']['S6_contact'] is None and rc == 2


@pytest.mark.parametrize('bad', [math.nan, math.inf])
def test_contact_nonfinite_insufficient(tmp_path, base, bad):
    room, task, sol = copy.deepcopy(base)
    k = col(room, 'drawer_pairs')
    room['steps'][400][k] = [[0, 1, bad, 0.0, 0.0, 0.0, 0.0, 0.0]]
    out, rc = run(tmp_path, f'cnan{bad}', room, task, sol)
    assert out['results']['S6_contact'] is None and rc == 2
    assert out['S6_contact']['n_rows_invalid_or_nonfinite'] == 1


def test_non_finger_contact_fails(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    k = col(room, 'drawer_pairs')
    room['steps'][400][k] = [[0, 1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]]
    out, rc = run(tmp_path, 'cbase', room, task, sol)
    assert out['results']['S6_contact'] is False and rc == 1
    assert 'base_link' in out['S6_contact']['non_finger_over_0p01n']


def test_contact_with_gap_insufficient(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    room['steps'] = [s for s in room['steps'] if not (11.5 <= s[1] < 11.8)]
    out, rc = run(tmp_path, 'cgap', room, task, sol)
    assert out['results']['S6_contact'] is None


# ---------------------------------------------------------------- 其他
def test_drift_detected(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    kb = col(room, 'base_xyth')
    for s in room['steps']:
        if 4.0 <= s[1] <= 11.0:
            s[kb] = [s[kb][0] + 0.02, s[kb][1], s[kb][2]]
    out, rc = run(tmp_path, 'drift', room, task, sol)
    assert out['results']['S4_drift'] is False and out['S4_drift']['max_mm'] > 10


def test_drift_window_gap_insufficient(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    kq = col(room, 'q_arm_meas')
    room['steps'][500][kq] = [math.nan] * 6
    out, rc = run(tmp_path, 'driftnan', room, task, sol)
    assert out['results']['S4_drift'] is None


def test_home_fail(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    room['config']['start_pose'] = f'{X0 + 0.5},{Y0},{TH}'
    out, rc = run(tmp_path, 'home', room, task, sol)
    assert out['results']['S5_home'] is False and rc == 1


def test_finger_none_in_hold_breaks_grasp(tmp_path, base):
    room, task, sol = copy.deepcopy(base)
    kf = col(room, 'finger_contact_n')
    for s in room['steps']:
        if 0.9 <= s[1] <= 11.0 and int(round(s[1] / DT)) % 150 == 0:
            s[kf] = None                      # 每 1.5 s 缺一筆
    out, _ = run(tmp_path, 'fnone', room, task, sol)
    assert out['S3_grasp']['longest_both_fingers_s'] < 2.0
    assert out['results']['S3_grasp'] is False
