"""E2 接收端斷訊的離線判定：只讀趟次輸出，不開模擬器。

門檻由 YAML 載入並驗證版本；規格全文複製進趟次目錄。
防護沿用：不同規格版本需 --retrospective、同版本需 --allow-recheck。

重點不在「再驗一次話題會停」（E1 已驗過），而在
**E2 的停止分支是否在實際執行端被呼叫、語意是否正確**：
逾時計數 > 0、`limit_mode` 轉為 timeout、手臂速率立即為零、
底盤每一步輪速與輪加速度合規。

**單步歸零不算失敗**：本階段輪緣速度約 0.03 m/s，
單步預算 r·α_max·dt = 0.0625 m/s 已足夠。反向階躍會觸發限制，
不代表歸零也必須觸發。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_wheel_limit import wheel_matrix                            # noqa: E402

DEFAULT_SPEC = 'evaluation/results/specs/wb_e2_cutoff_criteria_v1.yaml'
WANT = 'wb_e2_cutoff_criteria/1'
MODE_TIMEOUT = 1.0          # log 的 limit_mode 編碼

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--spec', default=DEFAULT_SPEC)
ap.add_argument('--allow-recheck', action='store_true')
ap.add_argument('--retrospective', action='store_true')
a = ap.parse_args()

raw = open(a.spec, 'rb').read()
spec_sha = hashlib.sha256(raw).hexdigest()
S = yaml.safe_load(raw.decode('utf-8'))
if S.get('schema') != WANT:
    print(f"**規格 schema 不符**：要 {WANT}，讀到 {S.get('schema')}"); sys.exit(3)
if not S.get('prospective') or not S.get('not_retroactive'):
    print('**規格未宣告 prospective／不追溯 —— 拒絕判定**'); sys.exit(3)
SS = S['S_stop']
if not SS.get('drift_subtraction_forbidden') or not SS.get('use_max_not_endpoint'):
    print('**規格未禁止扣漂移／未要求取最大值 —— 拒絕判定**'); sys.exit(3)
P, F, Q, E2T, AA, LL, W_, D = (S['P_premise'], S['F_topic_stopped'],
                               S['Q_others_alive'], S['E2T_timeout_branch'],
                               S['A_arm_stop'], S['L_base_stop_compliance'],
                               S['windows'], S['D_protections'])

OUT = os.path.join(a.run, 'e2_cutoff_check.json')
USED = os.path.join(a.run, 'criteria_used.yaml')
prev_ver = None
if os.path.exists(USED):
    try:
        prev_ver = yaml.safe_load(open(USED).read()).get('version')
        prev_schema = yaml.safe_load(open(USED).read()).get('schema')
    except Exception:
        prev_ver, prev_schema = '讀不出', None
    if prev_schema != S['schema']:
        prev_ver = f'{prev_schema}:{prev_ver}'
if prev_ver is not None and prev_ver != S['version']:
    if not a.retrospective:
        print(f'**拒絕覆蓋**：{a.run} 由規格 {prev_ver} 治理，'
              f'本次是 {S["schema"]}:{S["version"]}。加 --retrospective')
        sys.exit(4)
    OUT = os.path.join(a.run, f'e2_cutoff_check_{S["version"]}_retrospective.json')
else:
    if os.path.exists(OUT) and not a.allow_recheck:
        print(f'**拒絕覆寫**：{OUT} 已存在。加 --allow-recheck'); sys.exit(5)
    shutil.copyfile(a.spec, USED)


def rd(n):
    p = os.path.join(a.run, n)
    return json.load(open(p)) if os.path.exists(p) else None


run = rd(os.path.join('sim', 'wb_run.json'))
cut = rd('cut_adapter.json')
rec = rd('wb_vel_cmd_record.json')
live = rd('post_cut_liveness.json')
missing = [n for n, v in [('sim/wb_run.json', run), ('cut_adapter.json', cut),
                          ('wb_vel_cmd_record.json', rec),
                          ('post_cut_liveness.json', live)] if v is None]
if missing:
    print(f'**缺少輸出：{missing} —— 判定不成立**（缺少 ≠ 通過）'); sys.exit(3)
if cut.get('aborted'):
    print(f"**切斷器中止：{cut['aborted']} —— 本趟不成立**"); sys.exit(3)
if run.get('execution_version') != 'E2':
    print('**本趟不是 E2 —— 判定不成立**'); sys.exit(3)

cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]
DT = 0.01
T_CUT = float(cut['cut_sim_t'])
W_END = T_CUT + float(W_['observe_after_cut_s'])
covers = t[-1] >= W_END
Wm = wheel_matrix(float(run['cmd_chain']['wheel_limit']['wheel_base_L']))

res = []
def chk(k, ok, d):
    res.append({'key': k, 'pass': bool(ok), 'detail': d})


chk('W0_log涵蓋觀察窗', covers or not W_.get('require_log_covers_window', True),
    f'觀察窗至 {W_END:.2f}s，log 至 {t[-1]:.2f}s')

bv = L[:, i['base_lin_meas']]
q2 = L[:, i['joint2_act']]
sp = L[:, i['joint2_sp']]
ar = np.abs(np.gradient(q2, t))
kcut = int(np.argmin(np.abs(t - T_CUT)))

pre = (t >= T_CUT - 0.2) & (t <= T_CUT)
chk('P1_切斷時兩者都在運動',
    bool((bv[pre] > P['base_moving_mps']).all()
         and (ar[pre] > P['arm_moving_rps']).all()),
    f'底盤 {bv[pre].min():.5f}–{bv[pre].max():.5f} m/s、'
    f'joint2 {ar[pre].min():.5f}–{ar[pre].max():.5f} rad/s')

chk('F1_adapter 確實結束', cut.get('exited') is True,
    f"exited={cut.get('exited')}，等待 {cut.get('exit_wait_s')}s")
msgs = [m for m in rec['msgs'] if m[0] is not None]
before = [m for m in msgs if m[0] <= T_CUT]
after = [m for m in msgs if m[0] > T_CUT + F['F2_last_msg_within_s']]
last = max((m[0] for m in msgs), default=None)
chk('F2_最後一則距切斷',
    last is not None and (last - T_CUT) <= F['F2_last_msg_within_s'],
    f'最後 sim {last}，切斷 {T_CUT:.3f}')
chk('F3_切斷後無新訊息', len(after) == F['F3_msgs_after_cut_eq'],
    f'{len(after)} 則')
chk('F4_切斷前觀測正常', len(before) >= F['F4_min_msgs_before_cut'],
    f"{len(before)} 則（≥ {F['F4_min_msgs_before_cut']}）")
chk('Q1_執行端跑到 sim_limit', run.get('stop_reason') == 'sim_limit',
    f"stop_reason={run.get('stop_reason')}")
chk('Q2_安全層仍存活', live.get('safety') is True,
    ', '.join(f'{k}={v}' for k, v in live.items()))

# ---------------- E2T：E2 的逾時分支確實被呼叫 ----------------
cc = run['cmd_chain']
wl = cc.get('wheel_limit', {})
nfroz = cc.get('frozen_steps', 0)
ndec = wl.get('timeout_decel_steps', 0)
integ = L[:, i['integrating']]
post = np.where((t > T_CUT) & (integ < 0.5))[0]
t_freeze = float(t[post[0]]) if post.size else float('nan')
chk('E2T1_凍結步數 > 0', nfroz > E2T['E1_frozen_steps_gt'],
    f'frozen_steps={nfroz}')
chk('E2T2_**E2 逾時減速分支被呼叫**', ndec > E2T['E2_timeout_decel_steps_gt'],
    f'timeout_decel_steps={ndec}（E1 的鏈沒有這個分支）')
chk('E2T3_凍結時刻在期限內',
    post.size > 0 and (t_freeze - T_CUT) <= E2T['E4_freeze_within_s'],
    f'凍結 sim {t_freeze:.3f}，切斷後 {t_freeze - T_CUT:+.3f}s')
lm = L[:, i['limit_mode']]
after_fr = (t >= t_freeze) & (t <= W_END) & np.isfinite(lm)
chk('E2T4_凍結後 limit_mode 轉為 timeout',
    bool(after_fr.any()) and bool((lm[after_fr] == MODE_TIMEOUT).all()),
    f'凍結後 limit_mode 取值 {sorted(set(lm[after_fr].tolist()))}（1=timeout）')

# ---------------- A 手臂立即停止 ----------------
kfr = int(np.argmin(np.abs(t - t_freeze)))
we = int(np.argmin(np.abs(t - W_END)))
sp_fr = sp[kfr]
sp_dev = float(np.max(np.abs(sp[kfr:we + 1] - sp_fr)))
chk('A1_設定點停止積分', sp_dev <= AA['A1_setpoint_frozen_tol_rad'],
    f'觀察窗內設定點最大變動 {sp_dev*1e6:.3f} µrad')
rows = run.get('limit_rows') or []
to_rows = [r for r in rows if r.get('mode') == 'timeout']
arm_nz = [r for r in to_rows if max(abs(x) for x in r['u_out'][3:]) > 0.0]
chk('A2_逾時分支手臂輸出速率恰為零',
    bool(to_rows) and not arm_nz,
    f'逾時記錄 {len(to_rows)} 筆，手臂非零者 {len(arm_nz)} 筆')
chk('A3_逾時明記不保留耦合方向',
    bool(to_rows) and all(r.get('coupling_preserved') is False for r in to_rows),
    f"coupling_preserved 取值 {sorted({r.get('coupling_preserved') for r in to_rows})}")

# ---------------- L 底盤每一步合規 ----------------
tolr = float(LL['L3_tolerance'])
seg = (t >= T_CUT) & (t <= W_END)
bcmd = np.column_stack([L[seg, i['vx_cmd']], L[seg, i['vy_cmd']],
                        L[seg, i['wz_cmd']]])
bcmd = np.nan_to_num(bcmd)
ws = np.max(np.abs(bcmd @ Wm.T), axis=1)
wa = np.max(np.abs(np.diff(bcmd, axis=0) @ Wm.T), axis=1) / DT
chk('L1_停止過程輪速逐步合規',
    float(ws.max()) <= LL['L1_wheel_speed_max_mps'] + tolr,
    f"max {ws.max():.6f} ≤ {LL['L1_wheel_speed_max_mps']}")
chk('L2_停止過程輪加速度逐步合規',
    float(wa.max()) <= LL['L2_wheel_accel_max_mps2'] + tolr,
    f"max {wa.max():.6f} ≤ {LL['L2_wheel_accel_max_mps2']}")
# 歸零步數：**不要求多步**
nz = int(np.sum(ws[np.where(t[seg] >= t_freeze)[0]] > 1e-9))
single = nz <= 1
chk('L3_停止過程確實到零', bool((ws[-5:] <= 1e-9).all()),
    f'觀察窗末輪速 {ws[-1]:.9f}；凍結後非零步數 {nz}'
    f'（{"單步歸零" if single else "多步減速"}；'
    f"多步非必要：{LL['multi_step_not_required']}）")

# ---------------- S 停止時間與行程 ----------------
def below_held(series, thr, deadline, min_cont):
    m = np.where((t >= t_freeze) & (series <= thr))[0]
    if not m.size:
        return float('nan'), False, 0.0
    k = m[0]
    s_ = (t >= t[k]) & (t <= W_END)
    return (float(t[k] - t_freeze),
            bool((series[s_] <= thr).all()) and (t[k] - t_freeze) <= deadline
            and float(s_.sum() * DT) >= min_cont,
            float(s_.sum() * DT))

d_b, ok_b, c_b = below_held(bv, P['base_moving_mps'],
                            SS['S1_base_rate_below_within_s'],
                            SS['S3_min_continuous_below_s'])
d_a, ok_a, c_a = below_held(ar, P['arm_moving_rps'],
                            SS['S2_arm_rate_below_within_s'],
                            SS['S3_min_continuous_below_s'])
chk('S1_底盤停穩', ok_b, f'凍結後 {d_b:+.3f}s 低於門檻，持續 {c_b:.2f}s')
chk('S2_手臂停穩', ok_a, f'凍結後 {d_a:+.3f}s 低於門檻，持續 {c_a:.2f}s')
bx, by = L[:, i['base_x']], L[:, i['base_y']]
bmax = float(np.max(np.hypot(bx[kfr:we + 1] - bx[kfr], by[kfr:we + 1] - by[kfr])))
amax = float(np.max(np.abs(q2[kfr:we + 1] - q2[kfr])))
chk('S4_凍結後底盤最大位移(原始)', bmax <= SS['S4_base_after_freeze_max_m'],
    f"{bmax*1000:.3f} mm（≤ {SS['S4_base_after_freeze_max_m']*1000:.0f}；未扣漂移）")
chk('S5_凍結後 joint2 最大位移(原始)', amax <= SS['S5_arm_after_freeze_max_rad'],
    f"{amax*1000:.4f} mrad（≤ {SS['S5_arm_after_freeze_max_rad']*1000:.0f}）")

chk('D1_pregrasp 由獨立條件禁止',
    run.get('pregrasp_preconditions_met') is False
    and run.get('wheel_level_limiting_implemented') is True,
    f"wheel_limit={run.get('wheel_level_limiting_implemented')}, "
    f"preconditions={run.get('pregrasp_preconditions_met')}")
chk('D2_熱中止門檻', float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
    f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")
lsb = run.get('low_speed_interface_bound', {})
chk('D3_未放寬界限或加速度上限',
    float(lsb.get('lin_mps', -1)) == 0.05 and float(lsb.get('ang_rps', -1)) == 0.2
    and float(wl.get('a_max_rad_s2', -1)) == 125.0,
    f"界限 {lsb.get('lin_mps')}/{lsb.get('ang_rps')}、α_max {wl.get('a_max_rad_s2')}")

npass = sum(r['pass'] for r in res)
ok = npass == len(res)
label = (LL['label_if_single_step'] if single
         else '逾時分支接線與**多步**減速已驗證')
print('\n=== E2 接收端斷訊驗收 ===')
print(f'  規格 {a.spec}  {S["schema"]} {S["version"]} sha256 {spec_sha[:16]}')
print(f'  切斷 sim {T_CUT:.3f}；觀察窗 {T_CUT:.2f} – {W_END:.2f}s（事前固定）')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:26} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
print(f'  **結果標籤**：{label}')
print('  停止不要求數值恰為零；未扣漂移。多步減速非通過條件。')
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}")
json.dump({'schema': 'wb_e2_cutoff_check/1', 'run': a.run, 'spec_path': a.spec,
           'spec_version': S['version'], 'spec_sha256': spec_sha,
           'cut_sim_t': T_CUT, 'observe_window_s': [T_CUT, round(W_END, 3)],
           'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
           'result_label': label, 'single_step_zeroing': bool(single),
           'measured': {'freeze_sim_t': round(t_freeze, 4),
                        'freeze_delay_s': round(t_freeze - T_CUT, 4),
                        'frozen_steps': nfroz, 'timeout_decel_steps': ndec,
                        'nonzero_steps_after_freeze': nz,
                        'wheel_speed_max': float(ws.max()),
                        'wheel_accel_max': float(wa.max()),
                        'setpoint_drift_rad': sp_dev,
                        'base_max_after_freeze_m': bmax,
                        'arm_max_after_freeze_rad': amax,
                        'base_settle_delay_s': d_b, 'arm_settle_delay_s': d_a},
           'multi_step_evidence': ('多步減速**僅有離線證據**'
                                   if single else '本趟觀察到多步減速'),
           'drift_subtraction_applied': False,
           'scope_limit': S['scope_limit'],
           'not_retroactive': S['not_retroactive'],
           'not_claimed': ('上游全身安全性在命令被修改後**未重新論證**；'
                           '停止掃掠範圍**無避碰保證**；pregrasp 仍禁止')},
          open(OUT, 'w'), ensure_ascii=False, indent=1)
print(f'  -> {OUT}')
sys.exit(0 if ok else 1)
