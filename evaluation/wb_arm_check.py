"""有界 arm 單動測試的**離線判定**：只讀趟次輸出，不開模擬器。

門檻**全部由 YAML 規格載入**，不寫死在本檔。本檔會驗證 schema/version，
並把所用規格的完整內容與 sha256 複製進趟次目錄，
避免「規格與實作各說各話」。

主判準是 **實測 vs 設定點**（B1）。命令鏈以 `sp += dq × dt` 積分，
所以「Δ設定點 = 套用命令積分」是**恆等式**，不列為成果。
B4 測的是**設定點總增量對名目剖面的符合程度**，
**不是**逐筆跨程序保真度 —— 總增量相同可由互相抵銷的逐筆差異造成。

C 類用**事前固定評估窗內的最大原始值**判定（中途偏出再回來不算通過）；
終點淨值另列回報，不互相替代。同趟零命令段漂移率只作對照，
`drift_subtraction_forbidden` 由本檔實際讀取並強制，**不做相減**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
import yaml

DEFAULT_SPEC = 'evaluation/results/specs/wb_arm_criteria_v2.yaml'
WANT_SCHEMA = 'wb_arm_criteria/2'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--spec', default=DEFAULT_SPEC)
a = ap.parse_args()

# ---------------- 載入並驗證規格 ----------------
raw = open(a.spec, 'rb').read()
spec_sha = hashlib.sha256(raw).hexdigest()
S = yaml.safe_load(raw.decode('utf-8'))
if S.get('schema') != WANT_SCHEMA:
    print(f"**規格 schema 不符**：要 {WANT_SCHEMA}，讀到 {S.get('schema')}")
    sys.exit(3)
if not S.get('prospective', False):
    print('**規格未宣告 prospective —— 拒絕判定**')
    sys.exit(3)
CB = S['C_base_incidental']
if not CB.get('drift_subtraction_forbidden', False):
    print('**規格未禁止扣除漂移 —— 拒絕判定**（本檔不提供相減路徑）')
    sys.exit(3)
AC, BT, W, D = S['A_chain'], S['B_tracking'], S['windows'], S['D_protections']

# 規格留存到趟次目錄
shutil.copyfile(a.spec, os.path.join(a.run, 'criteria_used.yaml'))

run = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
src = json.load(open(os.path.join(a.run, 'cmd_source.json')))
cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]

scols = {c: k for k, c in enumerate(src['cols'])}
sent = src['sent']
def seg_times(name):
    v = [r[scols['sim_t']] for r in sent if r[scols['segment']] == name]
    return (min(v), max(v)) if v else (None, None)

t_lead0, _ = seg_times('zero_lead')
t_up0, _ = seg_times('ramp_up')
t_hold0, t_hold1 = seg_times('hold')
t_pub_end = max(r[scols['sim_t']] for r in sent)
if t_up0 is None or t_hold0 is None:
    print('**命令源沒有 ramp_up/hold 段 —— 無法判定**')
    sys.exit(3)

# ---- 事前固定的評估窗（不隨 log 長度改變）----
W_C0 = t_up0
W_C1 = t_pub_end + float(W['post_profile_s'])
covers = t[-1] >= W_C1

def sel(a_, b_):
    return (t >= a_) & (t <= b_)

def mean_t(col, a_, b_):
    """回傳 (值平均, **樣本時刻平均**)；分母要用實際樣本時刻，不用窗端點。"""
    m = sel(a_, b_)
    v, tt = L[m, i[col]], t[m]
    ok = np.isfinite(v)
    if not ok.any():
        return float('nan'), float('nan')
    return float(v[ok].mean()), float(tt[ok].mean())

def mean_at(col, a_, b_):
    return mean_t(col, a_, b_)[0]

REF = float(W['ref_before_ramp_s'])
TAIL = float(W['settle_tail_s'])
# 終點一律取**評估窗末**，不取 log 末
q2_ref = mean_at('joint2_act', t_up0 - REF, t_up0)
q2_end = mean_at('joint2_act', W_C1 - TAIL, W_C1)
sp_ref = mean_at('joint2_sp', t_up0 - REF, t_up0)
sp_end = mean_at('joint2_sp', W_C1 - TAIL, W_C1)
d_act, d_sp = q2_end - q2_ref, sp_end - sp_ref

res = []
def chk(key, ok, detail):
    res.append({'key': key, 'pass': bool(ok), 'detail': detail})

chk('W0_log涵蓋評估窗', covers or not W.get('require_log_covers_window', True),
    f'評估窗至 {W_C1:.2f}s，log 至 {t[-1]:.2f}s')

# ---------------- A 命令鏈 ----------------
cc = run['cmd_chain']
chk('A1_rejected', cc.get('rejected') == AC['A1_rejected_eq'],
    f"rejected={cc.get('rejected')}（要 {AC['A1_rejected_eq']}）")
chk('A2_fail_is_none', (cc.get('fail') is None) == AC['A2_fail_is_none'],
    f"fail={cc.get('fail')}")
base_nz = [r for r in sent
           if any(abs(r[scols[f'v{k}']]) > 1e-12 for k in (0, 1, 2))]
chk('A3_底盤分量恆零', (not base_nz) == AC['A3_base_components_all_zero'],
    f'非零發布筆數 {len(base_nz)}')
age = L[:, i['cmd_age']]
age_max = float(np.nanmax(age)) if np.isfinite(age).any() else float('nan')
chk('A4_cmd_age_max', age_max <= AC['A4_cmd_age_max_s'],
    f"max {age_max:.4f}s（≤ {AC['A4_cmd_age_max_s']}）")
sr = run.get('stop_reason')
chk('A5_stop_reason', sr in AC['A5_expected_stop_reasons'],
    f"{sr}（允許 {AC['A5_expected_stop_reasons']}）")

# ---------------- B 手臂追蹤 ----------------
chk('B1_|實測−設定點|', abs(d_act - d_sp) <= BT['B1_abs_err_rad_max'],
    f'Δ實測 {d_act:+.6f} rad、Δ設定點 {d_sp:+.6f} rad、'
    f"差 {(d_act-d_sp)*1000:+.3f} mrad（≤ {BT['B1_abs_err_rad_max']*1000:.0f}）")

EW = float(BT['B2_edge_window_s'])
q_a, ta_ = mean_t('joint2_act', t_hold0, t_hold0 + EW)
q_b, tb_ = mean_t('joint2_act', t_hold1 - EW, t_hold1)
hold_rate = (q_b - q_a) / (tb_ - ta_)      # 分母＝**樣本時刻平均差**
tol = BT['B2_hold_mean_rate_rps'] * BT['B2_hold_mean_rate_tol_frac']
chk('B2_保持段平均速度',
    abs(hold_rate - BT['B2_hold_mean_rate_rps']) <= tol,
    f"{hold_rate:.6f} rad/s（{BT['B2_hold_mean_rate_rps']}±"
    f"{BT['B2_hold_mean_rate_tol_frac']*100:.0f}%）"
    f'；分母 t̄ {tb_:.4f}−{ta_:.4f}={tb_-ta_:.4f}s')

cross = {}
for j in (1, 3, 4, 5, 6):
    cross[f'joint{j}'] = (mean_at(f'joint{j}_act', W_C1 - TAIL, W_C1)
                          - mean_at(f'joint{j}_act', t_up0 - REF, t_up0))
worst = max(cross, key=lambda k: abs(cross[k]))
chk('B3_其他五軸串音', abs(cross[worst]) <= BT['B3_other_joints_abs_rad_max'],
    f"最大 {worst} {cross[worst]*1000:+.3f} mrad"
    f"（≤ {BT['B3_other_joints_abs_rad_max']*1000:.0f}）")

nominal = float(S['profile']['nominal_integral_rad'])
chk('B4_設定點總增量對名目',
    abs(d_sp - nominal) <= BT['B4_sp_vs_nominal_rad_max'],
    f'Δ設定點 {d_sp:+.6f} vs 名目 {nominal:.3f}，差 {(d_sp-nominal)*1000:+.3f} mrad'
    f"（≤ {BT['B4_sp_vs_nominal_rad_max']*1000:.0f}；**總增量符合度，非逐筆**）")

# ---------------- C 底盤附帶運動（窗內最大原始值）----------------
bx0 = mean_at('base_x', t_up0 - REF, t_up0)
by0 = mean_at('base_y', t_up0 - REF, t_up0)
yw0 = mean_at('base_yaw', t_up0 - REF, t_up0)
m = sel(W_C0, W_C1)
dx, dy = L[m, i['base_x']] - bx0, L[m, i['base_y']] - by0
dyaw_deg = np.degrees(L[m, i['base_yaw']] - yw0)
disp_series = np.hypot(dx, dy)
k_max = int(np.argmax(disp_series))
max_disp = float(disp_series[k_max])
k_yaw = int(np.argmax(np.abs(dyaw_deg)))
max_yaw = float(dyaw_deg[k_yaw])
chk('C1_窗內最大平移(原始)', max_disp <= CB['C1_max_abs_disp_m_max'],
    f"{max_disp*1000:.3f} mm @ t={t[m][k_max]:.2f}s"
    f"（≤ {CB['C1_max_abs_disp_m_max']*1000:.0f}）")
chk('C2_窗內最大偏航(原始)', abs(max_yaw) <= CB['C2_max_abs_yaw_deg_max'],
    f"{max_yaw:+.4f}° @ t={t[m][k_yaw]:.2f}s"
    f"（≤ {CB['C2_max_abs_yaw_deg_max']}）")
# 終點淨值：**另列回報，不替代最大值判定**
fx = mean_at('base_x', W_C1 - TAIL, W_C1) - bx0
fy = mean_at('base_y', W_C1 - TAIL, W_C1) - by0
final_disp = math.hypot(fx, fy)
final_yaw = math.degrees(mean_at('base_yaw', W_C1 - TAIL, W_C1) - yw0)

drift = {}
if t_lead0 is not None and t_lead0 - t[0] > 6.0:
    da, db = t[0] + 3.0, t_lead0 - 1.0
    d0x, dta = mean_t('base_x', da, da + 0.5)
    d1x, dtb = mean_t('base_x', db - 0.5, db)
    d0y = mean_at('base_y', da, da + 0.5)
    d1y = mean_at('base_y', db - 0.5, db)
    d0w = mean_at('base_yaw', da, da + 0.5)
    d1w = mean_at('base_yaw', db - 0.5, db)
    span = dtb - dta
    drift = {'window_s': [round(da, 2), round(db, 2)],
             'disp_mm_per_s': round(math.hypot(d1x-d0x, d1y-d0y)/span*1000, 4),
             'yaw_deg_per_s': round(math.degrees(d1w-d0w)/span, 5),
             'note': '**僅作對照，未從 C1/C2 扣除**'}

# ---------------- D 保護 ----------------
chk('D1_pregrasp仍禁止',
    (run.get('wheel_level_limiting_implemented') is False)
    == D['pregrasp_forbidden_unchanged'],
    f"wheel_level_limiting_implemented={run.get('wheel_level_limiting_implemented')}")
chk('D2_熱中止門檻', float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
    f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")

# ---------------- 輸出 ----------------
npass = sum(r['pass'] for r in res)
ok = npass == len(res)
print(f'\n=== arm 單動判定 ===')
print(f'  規格 {a.spec}')
print(f'  schema {S["schema"]} version {S["version"]} sha256 {spec_sha[:16]}')
print(f'  評估窗（事前固定）{W_C0:.2f} – {W_C1:.2f}s'
      f'（＝斜升開始 → 剖面結束 +{W["post_profile_s"]}s）')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:20} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
print(f'  底盤終點淨值（**另列，不替代最大值判定**）：'
      f'{final_disp*1000:.3f} mm、{final_yaw:+.4f}°')
if drift:
    print(f"  同趟零命令段漂移（對照，**未扣除**）：{drift['disp_mm_per_s']} mm/s、"
          f"{drift['yaw_deg_per_s']} °/s，窗 {drift['window_s']}")
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}"
      f"（sync 需另行放行）")

json.dump({'schema': 'wb_arm_check/2', 'run': a.run,
           'spec_path': a.spec, 'spec_schema': S['schema'],
           'spec_version': S['version'], 'spec_sha256': spec_sha,
           'spec_copied_to': 'criteria_used.yaml',
           'eval_window_s': [round(W_C0, 3), round(W_C1, 3)],
           'eval_window_def': '斜升開始 → 命令剖面結束 + post_profile_s（事前固定）',
           'log_covers_window': bool(covers),
           'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
           'measured': {'d_q2_act_rad': round(d_act, 6),
                        'd_q2_sp_rad': round(d_sp, 6),
                        'hold_mean_rate_rps': round(hold_rate, 6),
                        'hold_rate_denominator_s': round(tb_ - ta_, 6),
                        'cross_talk_rad': {k: round(v, 6) for k, v in cross.items()},
                        'base_max_disp_m_raw': round(max_disp, 6),
                        'base_max_yaw_deg_raw': round(max_yaw, 5),
                        'base_final_disp_m_raw': round(final_disp, 6),
                        'base_final_yaw_deg_raw': round(final_yaw, 5),
                        'cmd_age_max_s': round(age_max, 4)},
           'zero_command_drift_same_run': drift,
           'drift_subtraction_applied': False,
           'drift_subtraction_forbidden_by_spec': True,
           'B4_semantics': ('設定點總增量對名目剖面的符合程度；'
                            '非逐筆跨程序保真度，無法排除互相抵銷的逐筆差異'),
           'identity_not_evidence': 'Δ設定點＝套用命令積分為恆等式，不列為成果'},
          open(os.path.join(a.run, 'arm_check.json'), 'w'),
          ensure_ascii=False, indent=1)
print(f"  -> {os.path.join(a.run, 'arm_check.json')}")
sys.exit(0 if ok else 1)
