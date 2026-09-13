"""有界 arm 單動測試的**離線判定**：只讀趟次輸出，不開模擬器。

**事前定版**：本檔與 evaluation/results/specs/wb_arm_criteria_v1.yaml
在任何 arm 趟次執行之前寫定。判定結果不得回頭改門檻。

主判準是 **實測 vs 設定點**（B1）。設定點本身等於「實際套用命令的積分」
（命令鏈以 sp += dq × dt 積分），兩者相等是恆等式，**不列為成果**。
跨程序的鏈路保真度另計為 B4（設定點總變化 vs 上游名目積分）。

底盤附帶運動（C）一律用**原始值**判定；同趟零命令段漂移率只作對照記錄，
**不相減**、不得據以宣稱底盤靜止。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

SPEC = 'evaluation/results/specs/wb_arm_criteria_v1.yaml'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='趟次目錄')
a = ap.parse_args()

run = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
src = json.load(open(os.path.join(a.run, 'cmd_source.json')))
cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]

# ---- 由命令源的段標籤取得時間窗（模擬時間）----
scols = {c: k for k, c in enumerate(src['cols'])}
sent = src['sent']
def seg_times(name):
    v = [r[scols['sim_t']] for r in sent if r[scols['segment']] == name]
    return (min(v), max(v)) if v else (None, None)

t_lead0, t_lead1 = seg_times('zero_lead')
t_up0, _ = seg_times('ramp_up')
t_hold0, t_hold1 = seg_times('hold')
_, t_tail1 = seg_times('zero_tail')
if t_up0 is None or t_hold0 is None:
    print('**命令源沒有 ramp_up/hold 段 —— 無法判定**')
    sys.exit(3)

def win(a_, b_):
    return (t >= a_) & (t <= b_)

def mean_at(col, a_, b_):
    m = win(a_, b_)
    v = L[m, i[col]]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float('nan')

# 參考點：斜升開始前 0.5 s；終值：log 最後 1.0 s
q2_ref = mean_at('joint2_act', t_up0 - 0.5, t_up0)
q2_end = mean_at('joint2_act', t[-1] - 1.0, t[-1])
sp_ref = mean_at('joint2_sp', t_up0 - 0.5, t_up0)
sp_end = mean_at('joint2_sp', t[-1] - 1.0, t[-1])
d_act = q2_end - q2_ref
d_sp = sp_end - sp_ref

res = []
def chk(key, ok, detail):
    res.append({'key': key, 'pass': bool(ok), 'detail': detail})

# ---------------- A 命令鏈 ----------------
cc = run['cmd_chain']
chk('A1_rejected==0', cc.get('rejected') == 0, f"rejected={cc.get('rejected')}")
chk('A2_fail_is_none', cc.get('fail') is None, f"fail={cc.get('fail')}")
base_nz = [r for r in sent
           if any(abs(r[scols[f'v{k}']]) > 1e-12 for k in (0, 1, 2))]
chk('A3_base_cmd_all_zero', not base_nz,
    f'底盤分量非零的發布筆數 {len(base_nz)}')
age = L[:, i['cmd_age']]
age_max = float(np.nanmax(age)) if np.isfinite(age).any() else float('nan')
chk('A4_cmd_age_max<=0.10', age_max <= 0.10, f'max {age_max:.4f}s')
sr = run.get('stop_reason')
chk('A5_no_low_speed_abort', cc.get('fail') is None and sr in ('sim_limit',),
    f'stop_reason={sr}')

# ---------------- B 手臂追蹤 ----------------
chk('B1_|act-sp|<=5mrad', abs(d_act - d_sp) <= 0.005,
    f'Δact {d_act:+.6f} rad, Δsp {d_sp:+.6f} rad, 差 {(d_act-d_sp)*1000:+.3f} mrad')
hold_rate = ((mean_at('joint2_act', t_hold1 - 0.05, t_hold1)
              - mean_at('joint2_act', t_hold0, t_hold0 + 0.05))
             / (t_hold1 - t_hold0))
chk('B2_hold_rate 0.05±10%', abs(hold_rate - 0.05) <= 0.005,
    f'{hold_rate:.6f} rad/s（區間 {t_hold0:.2f}–{t_hold1:.2f}s）')
cross = {}
for j in (1, 3, 4, 5, 6):
    r0 = mean_at(f'joint{j}_act', t_up0 - 0.5, t_up0)
    r1 = mean_at(f'joint{j}_act', t[-1] - 1.0, t[-1])
    cross[f'joint{j}'] = r1 - r0
worst = max(cross, key=lambda k: abs(cross[k]))
chk('B3_串音<=10mrad', abs(cross[worst]) <= 0.010,
    f'最大 {worst} {cross[worst]*1000:+.3f} mrad')
nominal = 0.200
chk('B4_|Δsp-名目|<=4mrad', abs(d_sp - nominal) <= 0.004,
    f'Δsp {d_sp:+.6f} vs 名目 {nominal:.3f}，差 {(d_sp-nominal)*1000:+.3f} mrad')

# ---------------- C 底盤附帶運動（原始值，不扣漂移）----------------
bx0 = mean_at('base_x', t_up0 - 0.5, t_up0)
by0 = mean_at('base_y', t_up0 - 0.5, t_up0)
yw0 = mean_at('base_yaw', t_up0 - 0.5, t_up0)
bx1 = mean_at('base_x', t[-1] - 1.0, t[-1])
by1 = mean_at('base_y', t[-1] - 1.0, t[-1])
yw1 = mean_at('base_yaw', t[-1] - 1.0, t[-1])
disp = math.hypot(bx1 - bx0, by1 - by0)
dyaw = math.degrees(yw1 - yw0)
chk('C1_底盤位移<=20mm(原始)', disp <= 0.020,
    f'{disp*1000:.3f} mm（Δx {(bx1-bx0)*1000:+.3f}, Δy {(by1-by0)*1000:+.3f}）')
chk('C2_底盤偏航<=1.0deg(原始)', abs(dyaw) <= 1.0, f'{dyaw:+.4f}°')

# 同趟零命令段漂移率（**只作對照，不相減**）
drift = {}
if t_lead0 is not None and t_lead0 - t[0] > 6.0:
    da, db = t[0] + 3.0, t_lead0 - 1.0
    ddisp = math.hypot(mean_at('base_x', db - 0.5, db) - mean_at('base_x', da, da + 0.5),
                       mean_at('base_y', db - 0.5, db) - mean_at('base_y', da, da + 0.5))
    dyw = math.degrees(mean_at('base_yaw', db - 0.5, db)
                       - mean_at('base_yaw', da, da + 0.5))
    drift = {'window_s': [round(da, 2), round(db, 2)],
             'disp_mm_per_s': round(ddisp / (db - da) * 1000, 4),
             'yaw_deg_per_s': round(dyw / (db - da), 5),
             'note': '**僅作對照，未從 C1/C2 扣除**'}

# ---------------- D 保護 ----------------
chk('D1_pregrasp仍禁止', run.get('wheel_level_limiting_implemented') is False,
    f"wheel_level_limiting_implemented={run.get('wheel_level_limiting_implemented')}")
chk('D2_熱中止=92.0', float(run.get('cpu_limit_c', -1)) == 92.0,
    f"cpu_limit_c={run.get('cpu_limit_c')}, 峰值 {run.get('cpu_temp_max_c')}")

# ---------------- 輸出 ----------------
npass = sum(r['pass'] for r in res)
print(f'\n=== arm 單動判定（規格 {SPEC}，事前定版）===')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:26} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
if drift:
    print(f"  同趟零命令段漂移（對照，未扣除）：{drift['disp_mm_per_s']} mm/s、"
          f"{drift['yaw_deg_per_s']} °/s，窗 {drift['window_s']}")
ok = npass == len(res)
print(f"  結論：{'可進 sync' if ok else '**不進 sync**；門檻不下修'}")

json.dump({'schema': 'wb_arm_check/1', 'spec': SPEC, 'run': a.run,
           'prospective': True, 'checks': res,
           'passed': npass, 'total': len(res), 'all_pass': ok,
           'measured': {'d_q2_act_rad': round(d_act, 6),
                        'd_q2_sp_rad': round(d_sp, 6),
                        'hold_mean_rate_rps': round(hold_rate, 6),
                        'cross_talk_rad': {k: round(v, 6) for k, v in cross.items()},
                        'base_disp_m_raw': round(disp, 6),
                        'base_dyaw_deg_raw': round(dyaw, 5),
                        'cmd_age_max_s': round(age_max, 4)},
           'zero_command_drift_same_run': drift,
           'drift_subtraction': '**未扣除**；不得相減後宣稱底盤靜止',
           'identity_not_evidence': ('Δsp 等於套用命令積分是演算法恆等式；'
                                     '主判準為 B1（實測 vs 設定點）')},
          open(os.path.join(a.run, 'arm_check.json'), 'w'),
          ensure_ascii=False, indent=1)
print(f"  -> {os.path.join(a.run, 'arm_check.json')}")
sys.exit(0 if ok else 1)
