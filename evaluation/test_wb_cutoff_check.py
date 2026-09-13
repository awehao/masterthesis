"""wb_cutoff_check.py 的離線測試：不開模擬器，用合成趟次驗證斷訊判準。

重點：
  * 缺少輸出 → 判定**不成立**（缺少 ≠ 通過）
  * 話題其實沒停 → F3 未通過；觀測程序整趟沒收到 → F4 未通過（不得靜默放行）
  * 逾時未觸發／凍結太慢／凍結後仍下底盤命令 → T 未通過
  * 底盤沒停下 → G1 未通過；**只有漂移量級的位移仍通過**（不要求恰為零）
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
CHECK = os.path.join(HERE, 'wb_cutoff_check.py')
SPEC = os.path.join(HERE, 'results', 'specs', 'wb_cutoff_criteria_v1.yaml')
COLS = json.load(open(os.path.join(HERE, 'runs', 'wb_sync_221734',
                                   'sim', 'wb_run.json')))['log_cols']
IDX = {c: k for k, c in enumerate(COLS)}

DT = 0.01
T0, T_CUT = 20.0, 24.5
DRIFT = 0.0000885            # m/s，已知殘留漂移


def make(d, *, freeze_delay=0.2, base_stops=True, arm_stops=True,
         zero_base_cmd_after_freeze=True, msgs_after_cut=0,
         msgs_before_cut=90, safety_alive=True, cut_aborted=None,
         omit=(), t_end=60.0, extra_disp_mps=0.0):
    os.makedirs(os.path.join(d, 'sim'), exist_ok=True)
    t_fr = T_CUT + freeze_delay
    log, bx, q2, sp2 = [], 0.0, 0.0, 0.0
    for k in range(int(t_end / DT)):
        t = round((k + 1) * DT, 4)
        moving = T0 + 2.0 <= t < t_fr
        bvel = 0.03 if moving else (
            0.03 if (not base_stops and t >= t_fr) else DRIFT + extra_disp_mps)
        arate = 0.05 if moving else (
            0.05 if (not arm_stops and t >= t_fr) else 0.0)
        integ = 1.0 if t < t_fr else 0.0
        if t < t_fr:
            sp2 += (0.05 if moving else 0.0) * DT
        bx += bvel * DT
        q2 += arate * DT
        row = [0.0] * len(COLS)
        row[IDX['t']] = t
        row[IDX['recv_seq']] = k if t < T_CUT else int(T_CUT / DT)
        row[IDX['cmd_age']] = 0.03 if t < T_CUT else (t - T_CUT)
        row[IDX['base_x']] = bx
        row[IDX['base_lin_meas']] = bvel
        row[IDX['joint2_act']] = q2
        row[IDX['joint2_sp']] = sp2
        row[IDX['integrating']] = integ
        if integ or not zero_base_cmd_after_freeze:
            row[IDX['vx_cmd']] = 0.03 if (moving or not integ) else 0.0
        log.append(row)
    json.dump({'log_cols': COLS, 'log': log, 'stop_reason': 'sim_limit',
               'cpu_limit_c': 92.0, 'cpu_temp_max_c': 80.0,
               'wheel_level_limiting_implemented': False,
               'cmd_chain': {'rejected': 0, 'fail': None,
                             'frozen_steps': int((t_end - t_fr) / DT)
                             if freeze_delay < 900 else 0}},
              open(os.path.join(d, 'sim', 'wb_run.json'), 'w'))
    cut = {'schema': 'wb_cut_adapter/1', 'pid': 12345, 'cut_sim_t': T_CUT,
           'planned_cut_sim_t': T_CUT, 'exited': True, 'exit_wait_s': 0.3,
           'signal': 'SIGTERM->pgid', 'cmdline_verified': 'arm_vel_adapter.py'}
    if cut_aborted:
        cut['aborted'] = cut_aborted
    if 'cut' not in omit:
        json.dump(cut, open(os.path.join(d, 'cut_adapter.json'), 'w'))
    msgs = [[round(T_CUT - (msgs_before_cut - j) * 0.05, 4), 0.0, [0.0] * 9]
            for j in range(msgs_before_cut)]
    msgs += [[round(T_CUT + 0.5 + j * 0.05, 4), 0.0, [0.0] * 9]
             for j in range(msgs_after_cut)]
    if 'rec' not in omit:
        json.dump({'schema': 'wb_topic_recorder/1', 'topic': '/wb_vel_cmd',
                   'n': len(msgs), 'msgs': msgs},
                  open(os.path.join(d, 'wb_vel_cmd_record.json'), 'w'))
    if 'live' not in omit:
        json.dump({'isaac': True, 'rsp': True, 'dist': True,
                   'safety': safety_alive, 'adapter': False},
                  open(os.path.join(d, 'post_cut_liveness.json'), 'w'))
    return d


def run_check(d, spec=SPEC):
    p = subprocess.run([sys.executable, CHECK, '--run', d, '--spec', spec],
                       capture_output=True, text=True, cwd=WS)
    return p.returncode, p.stdout + p.stderr


def load(d):
    return json.load(open(os.path.join(d, 'cutoff_check.json')))


def key(r, n):
    return next(c for c in r['checks'] if c['key'].startswith(n))


fails = []
def expect(name, cond, detail=''):
    print(f"  [{'ok' if cond else '**FAIL**'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp(prefix='wb_cut_test_')
try:
    print('A 規格與輸出完整性')
    d = make(os.path.join(tmp, 'nominal'))
    rc, out = run_check(d)
    expect('A1 名目趟次全數通過', rc == 0, out.strip().splitlines()[-2] if out else '')
    r = load(d)
    expect('A2 規格版本與雜湊寫入輸出',
           r['spec_version'] == 'v1' and len(r['spec_sha256']) == 64)
    S = yaml.safe_load(open(SPEC).read())
    for nm, mut in [('schema', lambda b: b.update({'schema': 'x/9'})),
                    ('prospective', lambda b: b.update({'prospective': False})),
                    ('禁止扣漂移', lambda b: b['G_actual_motion'].update(
                        {'drift_subtraction_forbidden': False}))]:
        b = copy.deepcopy(S); mut(b)
        f = os.path.join(tmp, f'bad_{nm}.yaml'); yaml.safe_dump(b, open(f, 'w'))
        expect(f'A3 {nm} 不符即拒絕', run_check(d, f)[0] == 3)
    for om in ('cut', 'rec', 'live'):
        d0 = make(os.path.join(tmp, f'omit_{om}'), omit=(om,))
        rcx, outx = run_check(d0)
        expect(f'A4 缺少 {om} → 判定不成立（非通過）',
               rcx == 3 and '缺少' in outx)
    d0 = make(os.path.join(tmp, 'aborted'), cut_aborted='PID 身分不符')
    expect('A5 切斷器中止 → 判定不成立', run_check(d0)[0] == 3)

    print('F 話題確實停止（獨立觀測）')
    d1 = make(os.path.join(tmp, 'still_publishing'), msgs_after_cut=40)
    run_check(d1); r1 = load(d1)
    expect('F1 話題其實沒停 → F3 未通過', not key(r1, 'F3')['pass'],
           key(r1, 'F3')['detail'])
    d2 = make(os.path.join(tmp, 'rec_dead'), msgs_before_cut=0)
    run_check(d2); r2 = load(d2)
    expect('F2 觀測程序整趟沒收到 → F4 未通過（不靜默放行）',
           not key(r2, 'F4')['pass'], key(r2, 'F4')['detail'])
    expect('F3 名目趟次 F2/F3/F4 通過',
           all(key(r, k)['pass'] for k in ('F2', 'F3', 'F4')))

    print('Q 其他程序繼續運作')
    d3 = make(os.path.join(tmp, 'safety_dead'), safety_alive=False)
    run_check(d3); r3 = load(d3)
    expect('Q1 安全層已死 → Q2 未通過', not key(r3, 'Q2')['pass'])

    print('T 逾時觸發與處置')
    d4 = make(os.path.join(tmp, 'slow_freeze'), freeze_delay=1.5)
    run_check(d4); r4 = load(d4)
    expect('T1 凍結太慢 → T2 未通過', not key(r4, 'T2')['pass'],
           key(r4, 'T2')['detail'])
    d5 = make(os.path.join(tmp, 'cmd_after_freeze'),
              zero_base_cmd_after_freeze=False)
    run_check(d5); r5 = load(d5)
    expect('T2 凍結後仍下底盤命令 → T3 未通過', not key(r5, 'T3')['pass'])
    expect('T3 名目趟次 T1–T4 通過',
           all(key(r, k)['pass'] for k in ('T1', 'T2', 'T3', 'T4')))

    print('G 實際運動：停止但不要求恰為零')
    d6 = make(os.path.join(tmp, 'base_never_stops'), base_stops=False)
    run_check(d6); r6 = load(d6)
    expect('G1 底盤沒停 → G1 未通過', not key(r6, 'G1')['pass'])
    d7 = make(os.path.join(tmp, 'arm_never_stops'), arm_stops=False)
    run_check(d7); r7 = load(d7)
    expect('G2 手臂沒停 → G2 未通過', not key(r7, 'G2')['pass'])
    expect('G3 保持段位移符合設計預期（0.03×0.2=6.0 mm）',
           key(r, 'G3a')['pass']
           and abs(r['measured']['base_disp_cut_to_freeze_m'] - 0.006) < 0.001,
           f"保持段 {r['measured']['base_disp_cut_to_freeze_m']*1000:.3f} mm")
    expect('G4 凍結後只有漂移量級位移仍通過（不要求恰為零）',
           key(r, 'G3b')['pass']
           and r['measured']['base_disp_after_freeze_m_raw'] > 0,
           f"凍結後 {r['measured']['base_disp_after_freeze_m_raw']*1000:.3f} mm > 0")
    expect('G5 兩段分開回報，不合成單一數字',
           'base_disp_cut_to_freeze_m' in r['measured']
           and 'base_disp_after_freeze_m_raw' in r['measured'])
    expect('G6 手臂保持段符合設計預期（0.05×0.2=10.0 mrad）',
           key(r, 'G4a')['pass']
           and abs(r['measured']['arm_disp_cut_to_freeze_rad'] - 0.010) < 0.002,
           f"保持段 {r['measured']['arm_disp_cut_to_freeze_rad']*1000:.3f} mrad")
    d8 = make(os.path.join(tmp, 'creep'), extra_disp_mps=0.0009)  # 約 7 mm/8s
    run_check(d8); r8 = load(d8)
    expect('G7 凍結後明顯超出漂移量級的潛移 → G3b 未通過',
           not key(r8, 'G3b')['pass'],
           f"凍結後 {r8['measured']['base_disp_after_freeze_m_raw']*1000:.2f} mm")
    expect('G8 輸出標明未扣漂移且不要求恰為零',
           r['drift_subtraction_applied'] is False
           and r['exact_zero_not_required'] is True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
