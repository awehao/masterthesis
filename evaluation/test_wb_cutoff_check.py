"""wb_cutoff_check.py（v2）的離線測試：不開模擬器，用合成趟次驗證斷訊判準。

v2 的重點，逐項用合成資料驗：

* **「最後收斂但中途走太遠」仍必須未通過** —— 位移判準取窗內最大值，
  收斂判準不得替代它。合成一段速率始終低於停穩門檻（0.004 < 0.005 rad/s）、
  外擺 10 mrad 再回到設定點附近的運動：S 通過、C1 通過、**M2 未通過**
  （自凍結值起算的窗內最大位移 14.7 mrad > 上限 10.0）。
* 停穩要求**持續**低於門檻，碰一下不算。
* 缺少輸出、切斷器中止、規格版本不符即拒絕判定（缺少 ≠ 通過）。
* 不得覆蓋以不同規格版本做成的既有判定。
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
CHECK = os.path.join(HERE, 'wb_cutoff_check.py')
SPEC = os.path.join(HERE, 'results', 'specs', 'wb_cutoff_criteria_v2.yaml')
SPEC_V1 = os.path.join(HERE, 'results', 'specs', 'wb_cutoff_criteria_v1.yaml')
COLS = json.load(open(os.path.join(HERE, 'runs', 'wb_sync_221734',
                                   'sim', 'wb_run.json')))['log_cols']
IDX = {c: k for k, c in enumerate(COLS)}

DT = 0.01
T0, T_CUT = 20.0, 24.5
DRIFT = 0.0000885
LAG = 0.0041          # 凍結當下的追隨滯後（實測量級）
RESID = 0.0006        # 收斂後殘差
TAU = 0.15


def make(d, *, freeze_delay=0.2, base_stops=True,
         bump_rate=0.0, bump_out_s=3.0, bump_hold_s=1.0, bump_start_s=0.5,
         settle_tau=TAU, arm_never_settles=False,
         msgs_after_cut=0, msgs_before_cut=90, safety_alive=True,
         cut_aborted=None, omit=(), t_end=60.0, spec_for_used=None):
    """bump_*：凍結後的外擺（梯形）。速率保持在停穩門檻以下，
    但位移可超過預算 —— 用來驗「最後收斂但中途走太遠」。"""
    os.makedirs(os.path.join(d, 'sim'), exist_ok=True)
    t_fr = T_CUT + freeze_delay

    def bump(dt_):
        if bump_rate == 0.0 or dt_ < bump_start_s:
            return 0.0
        x = dt_ - bump_start_s
        peak = bump_rate * bump_out_s
        if x < bump_out_s:
            return bump_rate * x
        if x < bump_out_s + bump_hold_s:
            return peak
        y = x - bump_out_s - bump_hold_s
        return max(0.0, peak - bump_rate * y)

    log, bx, sp2 = [], 0.0, 0.0
    for k in range(int(t_end / DT)):
        t = round((k + 1) * DT, 4)
        moving = T0 + 2.0 <= t < t_fr
        if t < t_fr:
            sp2 += (0.05 if moving else 0.0) * DT
            q2 = sp2 - (LAG if moving else 0.0)
        else:
            if arm_never_settles:
                q2 = sp2 - LAG + 0.05 * (t - t_fr)
            else:
                e = math.exp(-(t - t_fr) / settle_tau)
                q2 = sp2 + RESID + (-LAG - RESID) * e + bump(t - t_fr)
        bvel = 0.03 if moving else (0.03 if (not base_stops and t >= t_fr)
                                    else DRIFT)
        bx += bvel * DT
        row = [0.0] * len(COLS)
        row[IDX['t']] = t
        row[IDX['recv_seq']] = k if t < T_CUT else int(T_CUT / DT)
        row[IDX['cmd_age']] = 0.03 if t < T_CUT else (t - T_CUT)
        row[IDX['base_x']] = bx
        row[IDX['base_lin_meas']] = bvel
        row[IDX['joint2_act']] = q2
        row[IDX['joint2_sp']] = sp2
        row[IDX['integrating']] = 1.0 if t < t_fr else 0.0
        if t < t_fr:
            row[IDX['vx_cmd']] = 0.03 if moving else 0.0
        log.append(row)
    json.dump({'log_cols': COLS, 'log': log, 'stop_reason': 'sim_limit',
               'cpu_limit_c': 92.0, 'cpu_temp_max_c': 80.0,
               'wheel_level_limiting_implemented': False,
               'cmd_chain': {'rejected': 0, 'fail': None,
                             'frozen_steps': int((t_end - t_fr) / DT)}},
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
                   'safety': safety_alive, 'adapter': False, 'recorder': True},
                  open(os.path.join(d, 'post_cut_liveness.json'), 'w'))
    if spec_for_used:
        shutil.copyfile(spec_for_used, os.path.join(d, 'criteria_used.yaml'))
    return d


def run_check(d, spec=SPEC, extra=()):
    p = subprocess.run([sys.executable, CHECK, '--run', d, '--spec', spec,
                        *extra], capture_output=True, text=True, cwd=WS)
    return p.returncode, p.stdout + p.stderr


def load(d, name='cutoff_check.json'):
    return json.load(open(os.path.join(d, name)))


def key(r, n):
    return next(c for c in r['checks'] if c['key'].startswith(n))


fails = []
def expect(name, cond, detail=''):
    print(f"  [{'ok' if cond else '**FAIL**'}] {name}"
          f"{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp(prefix='wb_cut2_test_')
try:
    print('A 規格載入、完整性與覆寫防護')
    d = make(os.path.join(tmp, 'nominal'))
    rc, out = run_check(d)
    expect('A1 名目趟次全數通過', rc == 0,
           out.strip().splitlines()[-1] if out else '')
    r = load(d)
    expect('A2 規格版本、雜湊與預算來源寫入輸出',
           r['spec_version'] == 'v2' and len(r['spec_sha256']) == 64
           and 'stop_budget' in r['travel_budget_source'])
    expect('A3 不追溯條款寫入輸出', 'not_retroactive' in r
           and 'wb_cutoff_223153' in r['not_retroactive'])
    expect('A4 適用範圍限制寫入輸出',
           '接觸操作' in r['scope_limit'])
    S = yaml.safe_load(open(SPEC).read())
    for nm, mut in [
            ('schema', lambda b: b.update({'schema': 'x/9'})),
            ('prospective', lambda b: b.update({'prospective': False})),
            ('not_retroactive', lambda b: b.pop('not_retroactive')),
            ('禁止扣漂移', lambda b: b['M_travel_budget'].update(
                {'drift_subtraction_forbidden': False})),
            ('須用最大值', lambda b: b['M_travel_budget'].update(
                {'use_max_not_endpoint': False}))]:
        b = copy.deepcopy(S); mut(b)
        f = os.path.join(tmp, f'bad_{nm}.yaml'); yaml.safe_dump(b, open(f, 'w'))
        expect(f'A5 {nm} 不符即拒絕判定', run_check(d, f)[0] == 3)
    for om in ('cut', 'rec', 'live'):
        d0 = make(os.path.join(tmp, f'omit_{om}'), omit=(om,))
        rcx, outx = run_check(d0)
        expect(f'A6 缺少 {om} → 判定不成立（非通過）',
               rcx == 3 and '缺少' in outx)
    d0 = make(os.path.join(tmp, 'aborted'), cut_aborted='PID 身分不符')
    expect('A7 切斷器中止 → 判定不成立', run_check(d0)[0] == 3)
    dv1 = make(os.path.join(tmp, 'is_v1'), spec_for_used=SPEC_V1)
    rcx, outx = run_check(dv1)
    expect('A8 既有規格版本不符 → 拒絕覆蓋', rcx == 4 and '拒絕覆蓋' in outx)
    rcx, _ = run_check(dv1, extra=('--retrospective',))
    expect('A9 --retrospective 寫到獨立檔名、不動原規格副本',
           os.path.exists(os.path.join(dv1, 'cutoff_check_v2_retrospective.json'))
           and not os.path.exists(os.path.join(dv1, 'cutoff_check.json'))
           and yaml.safe_load(open(os.path.join(dv1, 'criteria_used.yaml'))
                              ).get('version') == 'v1')

    print('M 位移上限：最後收斂但中途走太遠**仍須未通過**')
    # 外擺 10 mrad，速率固定 0.004 rad/s（低於停穩門檻 0.005），走完再回來。
    # 起點延後到 1.5 s，避開滯後收斂的尾巴（否則兩者相加會短暫超過門檻，
    # S 就會先擋下來，驗不到 M 的獨立作用）。
    d1 = make(os.path.join(tmp, 'excursion'), bump_rate=0.004,
              bump_start_s=1.5, bump_out_s=2.5, bump_hold_s=0.5)
    run_check(d1); r1 = load(d1)
    mm1 = r1['measured']
    expect('M1 速率全程低於停穩門檻 → S 通過',
           key(r1, 'S1')['pass'] and key(r1, 'S2')['pass'])
    expect('M2 末值仍收斂到凍結設定點附近 → C1 通過',
           key(r1, 'C1')['pass'],
           f"末值誤差 {mm1['final_err_to_frozen_sp_rad']*1000:+.3f} mrad")
    expect('**M3 中途走太遠 → M2 未通過**', not key(r1, 'M2')['pass'],
           f"窗內最大 {mm1['arm_max_after_freeze_rad']*1000:.3f} mrad（上限 10.0）")
    expect('M4 若改看終點淨值就會漏掉（證明兩者不等價）',
           abs(mm1['final_err_to_frozen_sp_rad']) < 0.010
           < mm1['arm_max_after_freeze_rad'])
    expect('M5 名目趟次 M1–M5 通過',
           all(key(r, k)['pass'] for k in ('M1', 'M2', 'M3', 'M4', 'M5')),
           f"凍結後最大 {r['measured']['arm_max_after_freeze_rad']*1000:.3f} mrad")
    expect('M6 三段位移都有回報',
           all(k in r['measured'] for k in
               ('arm_max_cut_to_freeze_rad', 'arm_max_after_freeze_rad',
                'arm_max_whole_post_cut_rad')))

    print('S 停穩：需持續低於門檻')
    d2 = make(os.path.join(tmp, 'never_settles'), arm_never_settles=True)
    run_check(d2); r2 = load(d2)
    expect('S1 手臂一直在動 → S2 未通過', not key(r2, 'S2')['pass'])
    # 先降到門檻下，之後又被推上去：碰一下不算停
    d3 = make(os.path.join(tmp, 'touch_and_go'), bump_rate=0.02,
              bump_out_s=0.6, bump_hold_s=0.2, bump_start_s=3.0)
    run_check(d3); r3 = load(d3)
    expect('S2 降下後又動起來 → S2 未通過（碰一下不算停）',
           not key(r3, 'S2')['pass'])
    d4 = make(os.path.join(tmp, 'base_never_stops'), base_stops=False)
    run_check(d4); r4 = load(d4)
    expect('S3 底盤沒停 → S1 未通過', not key(r4, 'S1')['pass'])

    print('F/Q/T 其餘分項')
    d5 = make(os.path.join(tmp, 'still_publishing'), msgs_after_cut=40)
    run_check(d5); expect('F1 話題其實沒停 → F3 未通過',
                          not key(load(d5), 'F3')['pass'])
    d6 = make(os.path.join(tmp, 'rec_dead'), msgs_before_cut=0)
    run_check(d6); expect('F2 觀測程序整趟沒收到 → F4 未通過',
                          not key(load(d6), 'F4')['pass'])
    d7 = make(os.path.join(tmp, 'safety_dead'), safety_alive=False)
    run_check(d7); expect('Q1 安全層已死 → Q2 未通過',
                          not key(load(d7), 'Q2')['pass'])
    d8 = make(os.path.join(tmp, 'slow_freeze'), freeze_delay=1.5)
    run_check(d8); expect('T1 凍結太慢 → T2 未通過',
                          not key(load(d8), 'T2')['pass'])
    expect('T2 名目趟次 T1–T4 通過',
           all(key(r, k)['pass'] for k in ('T1', 'T2', 'T3', 'T4')))
    expect('D1 輸出標明未扣漂移', r['drift_subtraction_applied'] is False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
