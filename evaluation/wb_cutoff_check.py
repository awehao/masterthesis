"""接收端斷訊趟的**離線判定**：只讀趟次輸出，不開模擬器。

門檻由 YAML 載入（schema／prospective／禁止扣漂移都驗），規格全文複製進趟次目錄。

判定用到四份獨立產出：
  sim/wb_run.json          執行端自身的逐步 log
  cut_adapter.json         切斷器記錄的切斷時刻與 adapter 是否真的結束
  wb_vel_cmd_record.json   **另一個程序**觀測到的 /wb_vel_cmd 實際訊息
  post_cut_liveness.json   觀察窗結束時各程序是否仍存活

「話題停止更新」以獨立觀測程序的記錄為準，**不以執行端自己的 cmd_age 推論**。
「停止」不要求數值恰為零 —— 本系統零命令下本有殘留漂移；也不扣除漂移。
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

DEFAULT_SPEC = 'evaluation/results/specs/wb_cutoff_criteria_v1.yaml'
WANT = 'wb_cutoff_criteria/1'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--spec', default=DEFAULT_SPEC)
a = ap.parse_args()

raw = open(a.spec, 'rb').read()
spec_sha = hashlib.sha256(raw).hexdigest()
S = yaml.safe_load(raw.decode('utf-8'))
if S.get('schema') != WANT:
    print(f"**規格 schema 不符**：要 {WANT}，讀到 {S.get('schema')}")
    sys.exit(3)
if not S.get('prospective', False):
    print('**規格未宣告 prospective —— 拒絕判定**')
    sys.exit(3)
G = S['G_actual_motion']
if not G.get('drift_subtraction_forbidden', False):
    print('**規格未禁止扣除漂移 —— 拒絕判定**')
    sys.exit(3)
P, F, Q, T, W, D = (S['P_premise'], S['F_topic_stopped'], S['Q_others_alive'],
                    S['T_timeout_handling'], S['windows'], S['D_protections'])
shutil.copyfile(a.spec, os.path.join(a.run, 'criteria_used.yaml'))

def rd(name):
    p = os.path.join(a.run, name)
    return json.load(open(p)) if os.path.exists(p) else None

run = rd(os.path.join('sim', 'wb_run.json'))
cut = rd('cut_adapter.json')
rec = rd('wb_vel_cmd_record.json')
live = rd('post_cut_liveness.json')
missing = [n for n, v in [('sim/wb_run.json', run), ('cut_adapter.json', cut),
                          ('wb_vel_cmd_record.json', rec),
                          ('post_cut_liveness.json', live)] if v is None]
if missing:
    print(f"**缺少輸出：{missing} —— 判定不成立**（缺少 ≠ 通過）")
    sys.exit(3)
if cut.get('aborted'):
    print(f"**切斷器中止：{cut['aborted']} —— 本趟不成立**")
    sys.exit(3)

cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]
DT = 0.01
T_CUT = float(cut['cut_sim_t'])
OBS = float(W['observe_after_cut_s'])
W_END = T_CUT + OBS
covers = t[-1] >= W_END

res = []
def chk(key, ok, detail):
    res.append({'key': key, 'pass': bool(ok), 'detail': detail})

chk('W0_log涵蓋觀察窗', covers or not W.get('require_log_covers_window', True),
    f'觀察窗至 {W_END:.2f}s，log 至 {t[-1]:.2f}s')

bv = L[:, i['base_lin_meas']]
q2 = L[:, i['joint2_act']]
ar = np.abs(np.gradient(q2, t))
kcut = int(np.argmin(np.abs(t - T_CUT)))

# ---------------- P 前提 ----------------
pre = (t >= T_CUT - 0.2) & (t <= T_CUT)
chk('P1_切斷時兩者都在運動',
    bool((bv[pre] > P['base_moving_mps']).all()
         and (ar[pre] > P['arm_moving_rps']).all()),
    f'切斷前 0.2s：底盤 {bv[pre].min():.5f}–{bv[pre].max():.5f} m/s、'
    f'joint2 {ar[pre].min():.5f}–{ar[pre].max():.5f} rad/s')

# ---------------- F 話題確實停止（獨立觀測）----------------
chk('F1_adapter 確實結束', cut.get('exited') is True,
    f"exited={cut.get('exited')}，等待 {cut.get('exit_wait_s')}s，"
    f"訊號 {cut.get('signal')}")
msgs = [m for m in rec['msgs'] if m[0] is not None]
before = [m for m in msgs if m[0] <= T_CUT]
after = [m for m in msgs if m[0] > T_CUT + F['F2_last_msg_within_s']]
last = max((m[0] for m in msgs), default=None)
chk('F2_最後一則距切斷',
    last is not None and (last - T_CUT) <= F['F2_last_msg_within_s'],
    f'最後 sim {last}，切斷 {T_CUT:.3f}，差 '
    f"{(last - T_CUT) if last else float('nan'):+.4f}s"
    f"（≤ {F['F2_last_msg_within_s']}）")
chk('F3_切斷後無新訊息', len(after) == F['F3_msgs_after_cut_eq'],
    f'切斷 +{F["F2_last_msg_within_s"]}s 之後 {len(after)} 則')
chk('F4_切斷前觀測正常', len(before) >= F['F4_min_msgs_before_cut'],
    f"切斷前 {len(before)} 則（≥ {F['F4_min_msgs_before_cut']}）")

# ---------------- Q 其他程序繼續運作 ----------------
chk('Q1_執行端跑到 sim_limit',
    (run.get('stop_reason') == 'sim_limit') == Q['Q1_endpoint_ran_to_sim_limit'],
    f"stop_reason={run.get('stop_reason')}")
chk('Q2_安全層在觀察窗後仍存活',
    live.get('safety') is True, f"安全層存活={live.get('safety')}；"
    f"其餘 {', '.join(f'{k}={v}' for k, v in live.items() if k != 'safety')}")
chk('Q3_切斷非 cleanup 所為',
    cut.get('signal', '').startswith('SIGTERM') and 'cmdline_verified' in cut,
    f"由 wb_cut_adapter 對已核對身分的單一 PID {cut.get('pid')} 送 "
    f"{cut.get('signal')}")

# ---------------- T 逾時觸發與處置 ----------------
cc = run['cmd_chain']
nfroz = cc.get('frozen_steps', 0)
chk('T1_凍結步數 > 0', nfroz > T['T1_frozen_steps_gt'],
    f'frozen_steps={nfroz}（上游歸零那趟為 0）')
integ = L[:, i['integrating']]
post = np.where((t > T_CUT) & (integ < 0.5))[0]
t_freeze = float(t[post[0]]) if post.size else float('nan')
chk('T2_凍結時刻在期限內',
    post.size > 0 and (t_freeze - T_CUT) <= T['T2_freeze_within_s'],
    f'凍結 sim {t_freeze:.3f}，切斷後 {t_freeze - T_CUT:+.3f}s'
    f"（≤ {T['T2_freeze_within_s']}）")
mpost = t >= t_freeze
zb = np.nanmax(np.abs(np.column_stack([L[mpost, i['vx_cmd']],
                                       L[mpost, i['vy_cmd']],
                                       L[mpost, i['wz_cmd']]])))
chk('T3_凍結後底盤命令為零', zb == 0.0,
    f'凍結後套用底盤命令 |max| {zb:.9f}')
sp = L[:, i['joint2_sp']]
sp_f = sp[np.argmax(mpost)]
win = mpost & (t <= W_END)
chk('T4_設定點停止積分',
    float(np.nanmax(np.abs(sp[win] - sp_f))) <= T['T4_arm_setpoint_frozen_tol_rad'],
    f'觀察窗內設定點最大變動 {np.nanmax(np.abs(sp[win] - sp_f))*1e6:.3f} µrad')

# ---------------- G 實際運動 ----------------
# 期限自**凍結時刻**起算：切斷後命令保持到過期才凍結，那段是設計行為
def first_below(series, thr, deadline):
    m = np.where((t >= t_freeze) & (series <= thr))[0]
    if not m.size:
        return float('nan'), False
    k = m[0]      # 需**持續**在門檻下，不是碰一下
    held = bool((series[k:][(t[k:] <= W_END)] <= thr).all())
    return float(t[k] - t_freeze), held and (t[k] - t_freeze) <= deadline

d_b, ok_b = first_below(bv, P['base_moving_mps'],
                        G['G1_base_below_threshold_within_s'])
d_a, ok_a = first_below(ar, P['arm_moving_rps'],
                        G['G2_arm_below_threshold_within_s'])
chk('G1_底盤降到門檻下並保持', ok_b,
    f"凍結後 {d_b:+.3f}s 降到 {P['base_moving_mps']} m/s 以下並維持至觀察窗結束"
    f"（期限 {G['G1_base_below_threshold_within_s']}s）")
chk('G2_手臂降到門檻下並保持', ok_a,
    f"凍結後 {d_a:+.3f}s 降到 {P['arm_moving_rps']} rad/s 以下並維持"
    f"（期限 {G['G2_arm_below_threshold_within_s']}s）")

kfr = int(np.argmin(np.abs(t - t_freeze)))
we = int(np.argmin(np.abs(t - W_END)))
def bdisp(k0, k1):
    return math.hypot(L[k1, i['base_x']] - L[k0, i['base_x']],
                      L[k1, i['base_y']] - L[k0, i['base_y']])
d_hold = bdisp(kcut, kfr)
d_after = bdisp(kfr, we)
chk('G3a_保持段位移(設計行為)', d_hold <= G['G3a_base_disp_cut_to_freeze_m_max'],
    f"{d_hold*1000:.3f} mm（≤ {G['G3a_base_disp_cut_to_freeze_m_max']*1000:.0f}；"
    '設計預期 0.03×0.2=6.0 mm，命令保持到過期）')
chk('G3b_凍結後位移(原始)', d_after <= G['G3b_base_disp_after_freeze_m_max'],
    f"{d_after*1000:.3f} mm（≤ {G['G3b_base_disp_after_freeze_m_max']*1000:.0f}；"
    '**未扣漂移**，已知漂移 8s 約 0.7 mm；不要求恰為零）')
q_hold = abs(q2[kfr] - q2[kcut])
q_after = abs(q2[we] - q2[kfr])
chk('G4a_保持段 joint2 位移(設計行為)',
    q_hold <= G['G4a_arm_disp_cut_to_freeze_rad_max'],
    f"{q_hold*1000:.4f} mrad（≤ {G['G4a_arm_disp_cut_to_freeze_rad_max']*1000:.0f}；"
    '設計預期 0.05×0.2=10.0 mrad）')
chk('G4b_凍結後 joint2 位移(原始)',
    q_after <= G['G4b_arm_disp_after_freeze_rad_max'],
    f"{q_after*1000:.4f} mrad（≤ {G['G4b_arm_disp_after_freeze_rad_max']*1000:.0f}）")
disp, dq = d_after, q_after

# ---------------- D 保護 ----------------
chk('D1_pregrasp仍禁止',
    (run.get('wheel_level_limiting_implemented') is False)
    == D['pregrasp_forbidden_unchanged'],
    f"wheel_level_limiting_implemented={run.get('wheel_level_limiting_implemented')}")
chk('D2_熱中止門檻', float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
    f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")

npass = sum(r['pass'] for r in res)
ok = npass == len(res)
print('\n=== 接收端斷訊判定 ===')
print(f'  規格 {a.spec}')
print(f'  schema {S["schema"]} version {S["version"]} sha256 {spec_sha[:16]}')
print(f'  切斷 sim {T_CUT:.3f}（預定 {cut.get("planned_cut_sim_t")}）；'
      f'觀察窗 {T_CUT:.2f} – {W_END:.2f}s（事前固定）')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:22} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
print('  **停止不要求數值恰為零**：本系統零命令下本有殘留漂移，未扣除')
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}")

json.dump({'schema': 'wb_cutoff_check/1', 'run': a.run, 'spec_path': a.spec,
           'spec_schema': S['schema'], 'spec_version': S['version'],
           'spec_sha256': spec_sha, 'spec_copied_to': 'criteria_used.yaml',
           'cut_sim_t': T_CUT, 'observe_window_s': [T_CUT, round(W_END, 3)],
           'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
           'measured': {
               'freeze_sim_t': round(t_freeze, 4),
               'freeze_delay_s': round(t_freeze - T_CUT, 4),
               'frozen_steps': nfroz,
               'wb_vel_cmd_msgs_before_cut': len(before),
               'wb_vel_cmd_msgs_after_cut': len(after),
               'wb_vel_cmd_last_sim_t': last,
               'base_settle_delay_s': round(d_b, 4),
               'arm_settle_delay_s': round(d_a, 4),
               'base_disp_cut_to_freeze_m': round(d_hold, 6),
               'base_disp_after_freeze_m_raw': round(d_after, 6),
               'arm_disp_cut_to_freeze_rad': round(q_hold, 6),
               'arm_disp_after_freeze_rad_raw': round(q_after, 6)},
           'topic_stop_evidence': ('由獨立觀測程序 wb_topic_recorder 記錄的 '
                                   '/wb_vel_cmd 實際訊息，非執行端內部 cmd_age 推論'),
           'drift_subtraction_applied': False,
           'exact_zero_not_required': True,
           'displacement_split_note': ('位移分「保持段（切斷→凍結，設計上命令'
                                       '保持到過期）」與「凍結後」兩段；'
                                       '合成單一數字會讓是否真的停下無法判讀'),
           'differs_from_upstream_zero_test': ('上游歸零趟次 frozen_steps=0，'
                                               '接收端一路收到新鮮零命令；'
                                               '本趟話題實際停止更新')},
          open(os.path.join(a.run, 'cutoff_check.json'), 'w'),
          ensure_ascii=False, indent=1)
print(f"  -> {os.path.join(a.run, 'cutoff_check.json')}")
sys.exit(0 if ok else 1)
