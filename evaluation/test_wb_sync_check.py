"""wb_sync_check.py 的離線測試：不開模擬器，用合成趟次驗證 sync 判準。

重點驗兩件事：
  S 同動**由實測運動判定** —— 命令兩分量同時非零但手臂實際沒動時必須未通過
  C 底盤比的是**命令預期軌跡 vs 實測**，不是靜止上限 ——
    走完命令的 120 mm 必須通過（arm 的 ≤20 mm 判準會誤殺）
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
CHECK = os.path.join(HERE, 'wb_sync_check.py')
SPEC = os.path.join(HERE, 'results', 'specs', 'wb_sync_criteria_v1.yaml')
REAL = os.path.join(HERE, 'runs', 'wb_arm_220513', 'sim', 'wb_run.json')
COLS = json.load(open(REAL))['log_cols']
IDX = {c: k for k, c in enumerate(COLS)}

DT = 0.01
T0 = 20.0
T_LEAD, T_RAMP, T_HOLD, T_TAIL = 2.0, 1.0, 3.0, 2.0
VX, RATE = 0.03, 0.05


def scale_of(el):
    if el < T_LEAD:
        return 0.0, 'zero_lead'
    if el < T_LEAD + T_RAMP:
        return (el - T_LEAD) / T_RAMP, 'ramp_up'
    if el < T_LEAD + T_RAMP + T_HOLD:
        return 1.0, 'hold'
    if el < T_LEAD + 2 * T_RAMP + T_HOLD:
        return 1.0 - (el - T_LEAD - T_RAMP - T_HOLD) / T_RAMP, 'ramp_down'
    if el < T_LEAD + 2 * T_RAMP + T_HOLD + T_TAIL:
        return 0.0, 'zero_tail'
    return None, 'stopped'


def make(d, *, arm_moves=True, base_moves=True, arm_shift=0.0,
         single_sided=False, t_end=60.0):
    """arm_shift：把手臂運動在時間上平移，用來造出「不重疊」的情形。"""
    os.makedirs(os.path.join(d, 'sim'), exist_ok=True)
    sent, log = [], []
    sp2 = q2 = bx = 0.0
    for k in range(int(t_end / DT)):
        t = round((k + 1) * DT, 4)
        el = t - T0
        sc, seg = scale_of(el) if el >= 0 else (0.0, 'pre')
        if el >= 0 and sc is not None and k % 5 == 0:
            v = [0.0] * 9
            v[0] = VX * sc
            v[4] = 0.0 if single_sided else RATE * sc
            sent.append([t, round(el, 4), seg, sc] + v)
        vx = VX * sc if (el >= 0 and sc is not None and base_moves) else 0.0
        sa, _ = scale_of(el - arm_shift) if el - arm_shift >= 0 else (0.0, '')
        sa = sa or 0.0
        r = RATE * sa if (el >= 0 and arm_moves) else 0.0
        if el >= 0 and sc is not None:
            sp2 += (RATE * sc) * DT
        q2 += r * DT
        bx += vx * DT
        row = [0.0] * len(COLS)
        row[IDX['t']] = t
        row[IDX['recv_seq']] = k
        row[IDX['cmd_age']] = 0.03
        row[IDX['sent_prev_vwx']] = vx if base_moves else VX * (sc or 0.0)
        row[IDX['sent_prev_vwy']] = 0.0
        row[IDX['base_x']] = bx
        row[IDX['base_lin_meas']] = abs(vx)
        row[IDX['joint2_sp']] = sp2
        row[IDX['joint2_act']] = q2
        log.append(row)
    json.dump({'log_cols': COLS, 'log': log, 'stop_reason': 'sim_limit',
               'cpu_limit_c': 92.0, 'cpu_temp_max_c': 80.0,
               'wheel_level_limiting_implemented': False,
               'cmd_chain': {'rejected': 0, 'fail': None}},
              open(os.path.join(d, 'sim', 'wb_run.json'), 'w'))
    json.dump({'cols': ['sim_t', 'elapsed', 'segment', 'scale']
                       + [f'v{i}' for i in range(9)], 'sent': sent},
              open(os.path.join(d, 'cmd_source.json'), 'w'))
    return d


def run_check(d, spec=SPEC):
    p = subprocess.run([sys.executable, CHECK, '--run', d, '--spec', spec],
                       capture_output=True, text=True, cwd=WS)
    return p.returncode, p.stdout + p.stderr


def load(d):
    return json.load(open(os.path.join(d, 'sync_check.json')))


def key(r, name):
    return next(c for c in r['checks'] if c['key'].startswith(name))


fails = []
def expect(name, cond, detail=''):
    print(f"  [{'ok' if cond else '**FAIL**'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp(prefix='wb_sync_test_')
try:
    print('A 規格載入與驗證')
    d = make(os.path.join(tmp, 'nominal'))
    rc, out = run_check(d)
    r = load(d)
    expect('A1 規格版本與雜湊寫入輸出',
           r['spec_version'] == 'v1' and len(r['spec_sha256']) == 64)
    expect('A2 規格全文複製進趟次目錄',
           os.path.exists(os.path.join(d, 'criteria_used.yaml')))
    S = yaml.safe_load(open(SPEC).read())
    for nm, mut, want in [
            ('schema', lambda b: b.update({'schema': 'x/9'}), 3),
            ('prospective', lambda b: b.update({'prospective': False}), 3),
            ('禁止扣漂移',
             lambda b: b['C_base_tracking'].update(
                 {'drift_subtraction_forbidden': False}), 3)]:
        b = copy.deepcopy(S); mut(b)
        f = os.path.join(tmp, f'bad_{nm}.yaml')
        yaml.safe_dump(b, open(f, 'w'))
        rcx, _ = run_check(d, f)
        expect(f'A3 {nm} 不符即拒絕判定', rcx == want, f'rc={rcx}')
    b = copy.deepcopy(S); b['S_simultaneity']['S2_longest_continuous_s_min'] = 99.0
    f = os.path.join(tmp, 'strict.yaml'); yaml.safe_dump(b, open(f, 'w'))
    rcx, _ = run_check(d, f)
    expect('A4 改 YAML 門檻會改變判定（證明真的在讀）',
           key(load(d), 'S2')['pass'] is False, f'rc={rcx}')
    run_check(d)   # 還原輸出

    print('S 同動由實測運動判定')
    r = load(d)
    expect('S1 名目趟次同動通過', key(r, 'S1')['pass'] and key(r, 'S2')['pass'],
           f"累計 {r['measured']['simultaneous_cumulative_s']}s、"
           f"最長 {r['measured']['simultaneous_longest_s']}s")
    exp = T_HOLD + 2 * (1 - 1 / 6) * T_RAMP
    expect('S2 最長連續接近剖面推算值（約 4.6s）',
           abs(r['measured']['simultaneous_longest_s'] - exp) < 0.15,
           f"實得 {r['measured']['simultaneous_longest_s']}s vs 推算 {exp:.2f}s")

    d2 = make(os.path.join(tmp, 'arm_dead'), arm_moves=False)
    run_check(d2); r2 = load(d2)
    src2 = json.load(open(os.path.join(d2, 'cmd_source.json')))
    sc2 = {c: k for k, c in enumerate(src2['cols'])}
    nboth = sum(1 for x in src2['sent']
                if abs(x[sc2['v0']]) > 1e-12 and abs(x[sc2['v4']]) > 1e-12)
    expect('S3 命令兩分量同時非零但手臂沒動 → 同動未通過',
           (not key(r2, 'S1')['pass']) and (not key(r2, 'S2')['pass']),
           f'命令同時非零 {nboth} 則，實測同動 '
           f"{r2['measured']['simultaneous_cumulative_s']}s")

    d3 = make(os.path.join(tmp, 'staggered'), arm_shift=6.0)
    run_check(d3); r3 = load(d3)
    expect('S4 兩者都動但時間錯開 → 同動未通過',
           not key(r3, 'S2')['pass'],
           f"最長連續 {r3['measured']['simultaneous_longest_s']}s")

    print('C 底盤比命令預期軌跡，不是靜止上限')
    r = load(d)
    mm = r['measured']
    expect('C1 走完命令約 120 mm 仍通過（非靜止上限）',
           key(r, 'C1')['pass'] and math.hypot(*mm['base_measured_m']) > 0.10,
           f"實測 {math.hypot(*mm['base_measured_m'])*1000:.2f} mm、"
           f"誤差 {mm['base_pos_err_norm_m']*1000:.3f} mm")
    expect('C2 arm 的 20 mm 靜止上限會誤殺此趟（證明兩判準不同）',
           math.hypot(*mm['base_measured_m']) > 0.020)
    d4 = make(os.path.join(tmp, 'base_dead'), base_moves=False)
    run_check(d4); r4 = load(d4)
    expect('C3 底盤沒按命令走 → C1 未通過',
           not key(r4, 'C1')['pass'],
           f"誤差 {r4['measured']['base_pos_err_norm_m']*1000:.1f} mm")
    expect('C4 沿軌／橫向另列回報',
           'base_along_track_m' in mm and 'base_cross_track_m' in mm)

    print('E 同一則訊息')
    d5 = make(os.path.join(tmp, 'single_sided'), single_sided=True)
    run_check(d5); r5 = load(d5)
    expect('E1 單邊非零訊息會被抓出', not key(r5, 'E1')['pass'],
           key(r5, 'E1')['detail'])
    expect('E2 名目趟次 E1 通過', key(r, 'E1')['pass'])
    expect('E3 輸出標明「同一份快照」是結構保證非量測',
           '結構保證' in r['endpoint_same_snapshot'])
    expect('D1 輸出標明未扣漂移', r['drift_subtraction_applied'] is False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
