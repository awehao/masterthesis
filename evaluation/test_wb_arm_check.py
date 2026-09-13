"""wb_arm_check.py 的離線測試：不開模擬器，用合成趟次驗證三項修正。

A 規格真的被載入（schema／prospective／禁止扣漂移都會擋下判定）
B B2 速度分母用**樣本時刻平均差**，不是窗端點差
C C 類用**窗內最大原始值**判定，中途偏出再回來不得通過
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import copy

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
CHECK = os.path.join(HERE, 'wb_arm_check.py')
SPEC = os.path.join(HERE, 'results', 'specs', 'wb_arm_criteria_v2.yaml')
REAL = os.path.join(HERE, 'runs', 'wb_base_diag_160757', 'sim', 'wb_run.json')
COLS = json.load(open(REAL))['log_cols']
IDX = {c: k for k, c in enumerate(COLS)}

DT = 0.01
T_LEAD, T_RAMP, T_HOLD, T_TAIL = 2.0, 1.0, 3.0, 2.0
T0 = 20.0                                   # 剖面起點（模擬時間）
RATE = 0.05


def profile_rate(el):
    if el < T_LEAD:
        return 0.0, 'zero_lead'
    if el < T_LEAD + T_RAMP:
        return RATE * (el - T_LEAD) / T_RAMP, 'ramp_up'
    if el < T_LEAD + T_RAMP + T_HOLD:
        return RATE, 'hold'
    if el < T_LEAD + 2 * T_RAMP + T_HOLD:
        return RATE * (1 - (el - T_LEAD - T_RAMP - T_HOLD) / T_RAMP), 'ramp_down'
    if el < T_LEAD + 2 * T_RAMP + T_HOLD + T_TAIL:
        return 0.0, 'zero_tail'
    return None, 'stopped'


def make_run(d, *, base_xy=lambda t: (0.0, 0.0), yaw=lambda t: 0.0,
             act_gain=1.0, t_end=60.0):
    """合成一趟 arm 模式資料。act_gain=1 表示實測完全跟上設定點。"""
    os.makedirs(os.path.join(d, 'sim'), exist_ok=True)
    sent, log = [], []
    sp2 = 0.0
    n = int(t_end / DT)
    for k in range(n):
        t = round((k + 1) * DT, 4)
        el = t - T0
        r, seg = profile_rate(el) if el >= 0 else (0.0, 'pre')
        if el >= 0 and r is not None and abs((k % 5)) == 0:      # 20 Hz 發布
            v = [0.0] * 9
            v[4] = r
            sent.append([t, round(el, 4), seg, r / RATE if RATE else 0.0] + v)
        if el >= 0 and r is not None:
            sp2 += r * DT
        bx, by = base_xy(t)
        row = [0.0] * len(COLS)
        row[IDX['t']] = t
        row[IDX['recv_seq']] = k
        row[IDX['cmd_age']] = 0.03
        row[IDX['base_x']] = bx
        row[IDX['base_y']] = by
        row[IDX['usd_x']] = bx
        row[IDX['usd_y']] = by
        row[IDX['base_yaw']] = yaw(t)
        row[IDX['joint2_sp']] = sp2
        row[IDX['joint2_act']] = sp2 * act_gain
        log.append(row)
    json.dump({'log_cols': COLS, 'log': log, 'stop_reason': 'sim_limit',
               'sim_time_s': t_end, 'cpu_limit_c': 92.0, 'cpu_temp_max_c': 80.0,
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
    return json.load(open(os.path.join(d, 'arm_check.json')))


def key(r, name):
    return next(c for c in r['checks'] if c['key'].startswith(name))


fails = []
def expect(name, cond, detail=''):
    print(f"  [{'ok' if cond else '**FAIL**'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp(prefix='wb_arm_test_')
try:
    # ---------- A 規格真的被載入 ----------
    print('A 規格載入與驗證')
    d = make_run(os.path.join(tmp, 'nominal'))
    rc, out = run_check(d)
    r = load(d)
    expect('A1 規格雜湊與版本寫入輸出',
           r['spec_version'] == 'v2' and len(r['spec_sha256']) == 64,
           f"version={r['spec_version']}")
    expect('A2 規格全文複製進趟次目錄',
           os.path.exists(os.path.join(d, 'criteria_used.yaml')))

    S = yaml.safe_load(open(SPEC).read())
    bad = os.path.join(tmp, 'bad_schema.yaml')
    b = copy.deepcopy(S); b['schema'] = 'wb_arm_criteria/999'
    yaml.safe_dump(b, open(bad, 'w'))
    rc2, out2 = run_check(d, bad)
    expect('A3 schema 不符即拒絕判定', rc2 == 3 and 'schema 不符' in out2, f'rc={rc2}')

    bad2 = os.path.join(tmp, 'no_forbid.yaml')
    b2 = copy.deepcopy(S)
    b2['C_base_incidental']['drift_subtraction_forbidden'] = False
    yaml.safe_dump(b2, open(bad2, 'w'))
    rc3, out3 = run_check(d, bad2)
    expect('A4 未禁止扣漂移即拒絕判定',
           rc3 == 3 and '拒絕判定' in out3, f'rc={rc3}')

    bad3 = os.path.join(tmp, 'not_prosp.yaml')
    b3 = copy.deepcopy(S); b3['prospective'] = False
    yaml.safe_dump(b3, open(bad3, 'w'))
    rc4, out4 = run_check(d, bad3)
    expect('A5 未宣告 prospective 即拒絕判定', rc4 == 3, f'rc={rc4}')

    loose = os.path.join(tmp, 'loose.yaml')
    b4 = copy.deepcopy(S); b4['B_tracking']['B1_abs_err_rad_max'] = 0.5
    yaml.safe_dump(b4, open(loose, 'w'))
    d2 = make_run(os.path.join(tmp, 'sloppy'), act_gain=0.5)   # 差 100 mrad
    rc5, _ = run_check(d2)
    r5 = load(d2)
    rc6, _ = run_check(d2, loose)
    r6 = load(d2)
    expect('A6 門檻改 YAML 會改變判定結果（證明真的在讀）',
           (not key(r5, 'B1')['pass']) and key(r6, 'B1')['pass'],
           f"嚴格 {key(r5,'B1')['pass']} → 放寬 {key(r6,'B1')['pass']}")

    # ---------- B B2 分母 ----------
    print('B 保持段速度分母')
    r = load(d)
    den = r['measured']['hold_rate_denominator_s']
    src = json.load(open(os.path.join(d, 'cmd_source.json')))
    sc = {c: k for k, c in enumerate(src['cols'])}
    hv = [x[sc['sim_t']] for x in src['sent'] if x[sc['segment']] == 'hold']
    span = max(hv) - min(hv)            # v1 用的「整段首末時間差」
    expect('B1 分母不等於整段首末差（v1 的分母）',
           abs(den - span) > 1e-6 and den < span,
           f'修正後 {den:.4f}s vs v1 {span:.4f}s')
    bias = (r['measured']['hold_mean_rate_rps'] * den / span) - RATE
    expect('B1b v1 分母確實會低估速度', bias < -1e-6,
           f'用 v1 分母會低估 {bias/RATE*100:+.3f}%')
    expect('B2 標稱速度可判定為通過（act 完全跟上）',
           key(r, 'B2')['pass'],
           f"{r['measured']['hold_mean_rate_rps']:.6f} rad/s")
    ideal = RATE
    err = abs(r['measured']['hold_mean_rate_rps'] - ideal)
    expect('B3 速度估計誤差 < 0.5 %（分母修正後不低估）',
           err / ideal < 0.005, f'誤差 {err/ideal*100:.4f}%')

    # ---------- C 窗內最大 vs 終點淨值 ----------
    print('C 底盤判定用窗內最大值')
    def excursion(t):
        """命令期間偏出 50 mm，之後回到 1 mm —— 終值會過、最大值不該過。"""
        c = T0 + T_LEAD + T_RAMP + T_HOLD / 2
        return (0.050 * math.exp(-((t - c) / 1.2) ** 2) + 0.001, 0.0)
    d3 = make_run(os.path.join(tmp, 'excursion'), base_xy=excursion)
    rc7, _ = run_check(d3)
    r7 = load(d3)
    mx = r7['measured']['base_max_disp_m_raw']
    fn = r7['measured']['base_final_disp_m_raw']
    expect('C1 窗內最大值判定為未通過', not key(r7, 'C1')['pass'],
           f'最大 {mx*1000:.2f} mm')
    expect('C2 終點淨值本身在門檻內（證明兩者不等價）', fn <= 0.020,
           f'終值 {fn*1000:.2f} mm')
    expect('C3 終點淨值有另外回報', 'base_final_disp_m_raw' in r7['measured'])

    def yaw_exc(t):
        c = T0 + T_LEAD + T_RAMP + T_HOLD / 2
        return math.radians(2.0) * math.exp(-((t - c) / 1.2) ** 2)
    d4 = make_run(os.path.join(tmp, 'yaw_exc'), yaw=yaw_exc)
    rc8, _ = run_check(d4)
    r8 = load(d4)
    expect('C4 偏航同樣用窗內最大值判定', not key(r8, 'C2')['pass'],
           f"最大 {r8['measured']['base_max_yaw_deg_raw']:+.3f}°")
    expect('C5 偏航終點淨值在門檻內', abs(r8['measured']['base_final_yaw_deg_raw']) <= 1.0,
           f"終值 {r8['measured']['base_final_yaw_deg_raw']:+.4f}°")

    # ---------- C6 評估窗不隨 log 長度改變 ----------
    d5 = make_run(os.path.join(tmp, 'shortlog'), base_xy=excursion, t_end=40.0)
    rc9, _ = run_check(d5)
    r9 = load(d5)
    expect('C6 評估窗與 log 長度無關',
           r9['eval_window_s'] == r7['eval_window_s'],
           f"{r9['eval_window_s']} vs {r7['eval_window_s']}")
    d6 = make_run(os.path.join(tmp, 'tooshort'), t_end=28.0)
    rc10, _ = run_check(d6)
    r10 = load(d6)
    expect('C7 log 未涵蓋評估窗即不算通過',
           (not r10['log_covers_window']) and (not key(r10, 'W0')['pass']))

    expect('D1 輸出標明未扣漂移',
           load(d)['drift_subtraction_applied'] is False)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
