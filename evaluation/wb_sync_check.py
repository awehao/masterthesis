"""有界 sync 同動測試的**離線判定**：只讀趟次輸出，不開模擬器。

門檻全部由 YAML 規格載入（schema／version／prospective／禁止扣漂移都驗），
規格全文複製進趟次目錄並記錄 sha256。

與 arm 判定的兩處關鍵差異：

* **底盤不用靜止上限**。底盤本來就該主動移動，所以比的是
  「實際送進物理 API 的速度積分」與「實測軌跡」，並分列沿軌／橫向分量。
* **同動由實測運動判定**。以底盤實測速度與 joint2 實測速率各自的「運動中」
  門檻取交集，算累計時間與最長連續時間。
  **不以「兩個命令分量同時非零」代替** —— 那只證明命令同時下達。
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

DEFAULT_SPEC = 'evaluation/results/specs/wb_sync_criteria_v1.yaml'
WANT_SCHEMA = 'wb_sync_criteria/1'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--spec', default=DEFAULT_SPEC)
a = ap.parse_args()

raw = open(a.spec, 'rb').read()
spec_sha = hashlib.sha256(raw).hexdigest()
S = yaml.safe_load(raw.decode('utf-8'))
if S.get('schema') != WANT_SCHEMA:
    print(f"**規格 schema 不符**：要 {WANT_SCHEMA}，讀到 {S.get('schema')}")
    sys.exit(3)
if not S.get('prospective', False):
    print('**規格未宣告 prospective —— 拒絕判定**')
    sys.exit(3)
CT = S['C_base_tracking']
if not CT.get('drift_subtraction_forbidden', False):
    print('**規格未禁止扣除漂移 —— 拒絕判定**')
    sys.exit(3)
AC, BT, SM, W, D, E = (S['A_chain'], S['B_arm_tracking'], S['S_simultaneity'],
                       S['windows'], S['D_protections'], S['E_single_message'])
shutil.copyfile(a.spec, os.path.join(a.run, 'criteria_used.yaml'))

run = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
src = json.load(open(os.path.join(a.run, 'cmd_source.json')))
cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]
DT = 0.01

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

W_C0, W_C1 = t_up0, t_pub_end + float(W['post_profile_s'])
covers = t[-1] >= W_C1
REF, TAIL = float(W['ref_before_ramp_s']), float(W['settle_tail_s'])

def sel(a_, b_):
    return (t >= a_) & (t <= b_)

def mean_t(col, a_, b_):
    m = sel(a_, b_)
    v, tt = L[m, i[col]], t[m]
    ok = np.isfinite(v)
    if not ok.any():
        return float('nan'), float('nan')
    return float(v[ok].mean()), float(tt[ok].mean())

def mean_at(col, a_, b_):
    return mean_t(col, a_, b_)[0]

res = []
def chk(key, ok, detail):
    res.append({'key': key, 'pass': bool(ok), 'detail': detail})

chk('W0_log涵蓋評估窗', covers or not W.get('require_log_covers_window', True),
    f'評估窗至 {W_C1:.2f}s，log 至 {t[-1]:.2f}s')

# ---------------- A 命令鏈 ----------------
cc = run['cmd_chain']
chk('A1_rejected', cc.get('rejected') == AC['A1_rejected_eq'],
    f"rejected={cc.get('rejected')}")
chk('A2_fail_is_none', (cc.get('fail') is None) == AC['A2_fail_is_none'],
    f"fail={cc.get('fail')}")
age = L[:, i['cmd_age']]
age_max = float(np.nanmax(age)) if np.isfinite(age).any() else float('nan')
chk('A4_cmd_age_max', age_max <= AC['A4_cmd_age_max_s'],
    f"max {age_max:.4f}s（≤ {AC['A4_cmd_age_max_s']}）")
sr = run.get('stop_reason')
chk('A5_stop_reason', sr in AC['A5_expected_stop_reasons'], f'{sr}')

# ---------------- E 同一則訊息 ----------------
single = [r for r in sent
          if (abs(r[scols['v0']]) > 1e-12) != (abs(r[scols['v4']]) > 1e-12)]
chk('E1_無單邊非零訊息', (not single) == E['E1_no_single_sided_message'],
    f'單邊非零筆數 {len(single)} / 共 {len(sent)} 則')
rs = L[:, i['recv_seq']]
rs = rs[rs >= 0]
chk('E2_recv_seq 單調不減', bool(np.all(np.diff(rs) >= 0)) == E['E2_recv_seq_monotonic'],
    f'最小差 {int(np.min(np.diff(rs))) if rs.size > 1 else 0}')

# ---------------- B 手臂追蹤 ----------------
q2_ref = mean_at('joint2_act', t_up0 - REF, t_up0)
q2_end = mean_at('joint2_act', W_C1 - TAIL, W_C1)
sp_ref = mean_at('joint2_sp', t_up0 - REF, t_up0)
sp_end = mean_at('joint2_sp', W_C1 - TAIL, W_C1)
d_act, d_sp = q2_end - q2_ref, sp_end - sp_ref
chk('B1_|實測−設定點|', abs(d_act - d_sp) <= BT['B1_abs_err_rad_max'],
    f'Δ實測 {d_act:+.6f}、Δ設定點 {d_sp:+.6f}，差 {(d_act-d_sp)*1000:+.3f} mrad'
    f"（≤ {BT['B1_abs_err_rad_max']*1000:.0f}）")
EW = float(BT['B2_edge_window_s'])
q_a, ta_ = mean_t('joint2_act', t_hold0, t_hold0 + EW)
q_b, tb_ = mean_t('joint2_act', t_hold1 - EW, t_hold1)
hold_rate = (q_b - q_a) / (tb_ - ta_)
tol = BT['B2_hold_mean_rate_rps'] * BT['B2_hold_mean_rate_tol_frac']
chk('B2_保持段平均速度', abs(hold_rate - BT['B2_hold_mean_rate_rps']) <= tol,
    f"{hold_rate:.6f} rad/s；分母 {tb_-ta_:.4f}s")
cross = {f'joint{j}': (mean_at(f'joint{j}_act', W_C1 - TAIL, W_C1)
                       - mean_at(f'joint{j}_act', t_up0 - REF, t_up0))
         for j in (1, 3, 4, 5, 6)}
worst = max(cross, key=lambda k: abs(cross[k]))
chk('B3_其他五軸串音', abs(cross[worst]) <= BT['B3_other_joints_abs_rad_max'],
    f'最大 {worst} {cross[worst]*1000:+.3f} mrad')
nom = float(S['profile']['arm_nominal_integral_rad'])
chk('B4_設定點總增量對名目', abs(d_sp - nom) <= BT['B4_sp_vs_nominal_rad_max'],
    f'Δ設定點 {d_sp:+.6f} vs {nom:.3f}，差 {(d_sp-nom)*1000:+.3f} mrad'
    '（**總增量符合度，非逐筆**）')

# ---------------- C 底盤：命令預期軌跡 vs 實測 ----------------
m = sel(W_C0, W_C1)
ex = float(np.nansum(np.where(np.isfinite(L[m, i['sent_prev_vwx']]),
                              L[m, i['sent_prev_vwx']], 0.0)) * DT)
ey = float(np.nansum(np.where(np.isfinite(L[m, i['sent_prev_vwy']]),
                              L[m, i['sent_prev_vwy']], 0.0)) * DT)
bx0 = mean_at('base_x', t_up0 - REF, t_up0)
by0 = mean_at('base_y', t_up0 - REF, t_up0)
yw0 = mean_at('base_yaw', t_up0 - REF, t_up0)
mx = mean_at('base_x', W_C1 - TAIL, W_C1) - bx0
my = mean_at('base_y', W_C1 - TAIL, W_C1) - by0
err = math.hypot(mx - ex, my - ey)
n_ex = math.hypot(ex, ey)
if n_ex > 1e-9:                      # 沿軌／橫向分解（命令方向為軸）
    ux, uy = ex / n_ex, ey / n_ex
    along, crossp = mx * ux + my * uy, -mx * uy + my * ux
else:
    along, crossp = float('nan'), float('nan')
chk('C1_軌跡誤差範數', err <= CT['C1_pos_err_norm_m_max'],
    f'預期 {n_ex*1000:.3f} mm、實測 {math.hypot(mx,my)*1000:.3f} mm、'
    f"誤差 {err*1000:.3f} mm（≤ {CT['C1_pos_err_norm_m_max']*1000:.0f}）")
dyaw = math.degrees(mean_at('base_yaw', W_C1 - TAIL, W_C1) - yw0)
chk('C2_偏航(原始)', abs(dyaw) <= CT['C2_abs_yaw_deg_max'],
    f"{dyaw:+.4f}°（≤ {CT['C2_abs_yaw_deg_max']}）")

# ---------------- S 同動：由實測運動判定 ----------------
bv = L[:, i['base_lin_meas']]
q2 = L[:, i['joint2_act']]
ar = np.abs(np.gradient(q2, t))
mov_b = bv > float(SM['base_moving_mps'])
mov_a = ar > float(SM['arm_moving_rps'])
both = mov_b & mov_a & m
cum = float(both.sum() * DT)
# 最長連續區間
best = cur = 0
for v in both:
    cur = cur + 1 if v else 0
    best = max(best, cur)
longest = best * DT
chk('S1_同時運動累計時間', cum >= SM['S1_cumulative_s_min'],
    f"{cum:.2f}s（≥ {SM['S1_cumulative_s_min']}）")
chk('S2_最長連續同時運動', longest >= SM['S2_longest_continuous_s_min'],
    f"{longest:.2f}s（≥ {SM['S2_longest_continuous_s_min']}）")

# ---------------- D 保護 ----------------
chk('D1_pregrasp仍禁止',
    (run.get('wheel_level_limiting_implemented') is False)
    == D['pregrasp_forbidden_unchanged'],
    f"wheel_level_limiting_implemented={run.get('wheel_level_limiting_implemented')}")
chk('D2_熱中止門檻', float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
    f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")

npass = sum(r['pass'] for r in res)
ok = npass == len(res)
print('\n=== sync 同動判定 ===')
print(f'  規格 {a.spec}')
print(f'  schema {S["schema"]} version {S["version"]} sha256 {spec_sha[:16]}')
print(f'  評估窗（事前固定）{W_C0:.2f} – {W_C1:.2f}s')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:20} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
print(f'  底盤沿軌 {along*1000:+.3f} mm、橫向 {crossp*1000:+.3f} mm'
      f'（命令方向為軸；**原始值，未扣漂移**）')
print(f"  同動門檻：底盤 >{SM['base_moving_mps']} m/s、joint2 >{SM['arm_moving_rps']} rad/s"
      f'（**由實測運動判定，非命令分量**）')
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}"
      '（pregrasp 未解除；本趟不含斷訊試驗）')

json.dump({'schema': 'wb_sync_check/1', 'run': a.run, 'spec_path': a.spec,
           'spec_schema': S['schema'], 'spec_version': S['version'],
           'spec_sha256': spec_sha, 'spec_copied_to': 'criteria_used.yaml',
           'eval_window_s': [round(W_C0, 3), round(W_C1, 3)],
           'log_covers_window': bool(covers),
           'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
           'measured': {
               'd_q2_act_rad': round(d_act, 6), 'd_q2_sp_rad': round(d_sp, 6),
               'hold_mean_rate_rps': round(hold_rate, 6),
               'cross_talk_rad': {k: round(v, 6) for k, v in cross.items()},
               'base_expected_m': [round(ex, 6), round(ey, 6)],
               'base_measured_m': [round(mx, 6), round(my, 6)],
               'base_pos_err_norm_m': round(err, 6),
               'base_along_track_m': round(along, 6),
               'base_cross_track_m': round(crossp, 6),
               'base_dyaw_deg_raw': round(dyaw, 5),
               'simultaneous_cumulative_s': round(cum, 3),
               'simultaneous_longest_s': round(longest, 3),
               'cmd_age_max_s': round(age_max, 4)},
           'simultaneity_basis': ('由**實測**底盤速度與 joint2 速率的門檻交集判定；'
                                  '未以命令分量同時非零代替'),
           'base_criterion_note': ('底盤比的是「實際送進物理 API 的速度積分」'
                                   '與實測軌跡，**不是靜止上限**'),
           'drift_subtraction_applied': False,
           'endpoint_same_snapshot': ('CmdChain.step() 單一回傳的結構保證，'
                                      '非獨立量測；上游訊息同時性見 E1'),
           'excluded': ['接收端斷訊試驗', 'pregrasp']},
          open(os.path.join(a.run, 'sync_check.json'), 'w'),
          ensure_ascii=False, indent=1)
print(f"  -> {os.path.join(a.run, 'sync_check.json')}")
sys.exit(0 if ok else 1)
