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

DEFAULT_SPEC = 'evaluation/results/specs/wb_cutoff_criteria_v2.yaml'
WANT = 'wb_cutoff_criteria/2'

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--spec', default=DEFAULT_SPEC)
ap.add_argument('--allow-recheck', action='store_true',
                help='允許以**同一規格版本**覆寫既有判定。'
                     '預設拒絕，避免既有結果被靜默蓋掉')
ap.add_argument('--retrospective', action='store_true',
                help='對既有判定為**不同規格版本**的趟次重算。'
                     '結果寫到 cutoff_check_<version>_retrospective.json，'
                     '**不覆蓋**原判定，也不追認為通過')
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
M = S['M_travel_budget']
if not M.get('drift_subtraction_forbidden', False):
    print('**規格未禁止扣除漂移 —— 拒絕判定**')
    sys.exit(3)
if not M.get('use_max_not_endpoint', False):
    print('**規格未要求以窗內最大值判定 —— 拒絕判定**')
    sys.exit(3)
if not S.get('not_retroactive'):
    print('**規格未標註不追溯 —— 拒絕判定**')
    sys.exit(3)
P, F, Q, T, W, D = (S['P_premise'], S['F_topic_stopped'], S['Q_others_alive'],
                    S['T_timeout_handling'], S['windows'], S['D_protections'])
ST, CV = S['S_settled'], S['C_convergence']
# **防護**：不得覆蓋以不同規格版本做成的既有判定。
# 判斷依據是趟次目錄裡的 criteria_used.yaml —— 那是**當時治理該趟的規格**，
# 比判定輸出耐用（判定檔可能被搬走或改名，規格副本不會）。
# 這道防護是因為我曾經誤覆蓋過一次 v1 趟次的判定而加上的。
OUT = os.path.join(a.run, 'cutoff_check.json')
USED = os.path.join(a.run, 'criteria_used.yaml')
prev_ver = None
if os.path.exists(USED):
    try:
        prev_ver = yaml.safe_load(open(USED).read()).get('version')
    except Exception:
        prev_ver = '讀不出'
if prev_ver is not None and prev_ver != S['version']:
    if not a.retrospective:
        print(f'**拒絕覆蓋**：{a.run} 由規格 {prev_ver} 治理，本次是 '
              f'{S["version"]}。要重算請加 --retrospective'
              f'（寫到獨立檔名，原判定與原規格副本都不動，且不追認為通過）')
        sys.exit(4)
    OUT = os.path.join(a.run, f'cutoff_check_{S["version"]}_retrospective.json')
    print(f'[retro] 回溯重算 → {os.path.basename(OUT)}；'
          f'原規格 {prev_ver} 與原判定保持不動', flush=True)
else:
    # **同版本也不得靜默覆寫**：既有結果要被蓋掉必須明講
    if os.path.exists(OUT) and not a.allow_recheck:
        print(f'**拒絕覆寫**：{OUT} 已存在（同規格版本 {S["version"]}）。'
              f'要重算請加 --allow-recheck')
        sys.exit(5)
    shutil.copyfile(a.spec, USED)

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
# ---------------- S 是否確實停穩 ----------------
def below_and_held(series, thr, deadline, min_cont):
    """回傳 (首次低於門檻的延遲, 是否通過)。
    需**持續**低於門檻到觀察窗結束，且持續長度 >= min_cont —— 碰一下不算停。"""
    m = np.where((t >= t_freeze) & (series <= thr))[0]
    if not m.size:
        return float('nan'), False, 0.0
    k = m[0]
    seg = (t >= t[k]) & (t <= W_END)
    held = bool((series[seg] <= thr).all())
    cont = float(seg.sum() * DT)
    delay = float(t[k] - t_freeze)
    return delay, (held and delay <= deadline and cont >= min_cont), cont

d_b, ok_b, c_b = below_and_held(bv, P['base_moving_mps'],
                                ST['S1_base_rate_below_within_s'],
                                ST['S3_min_continuous_below_s'])
d_a, ok_a, c_a = below_and_held(ar, P['arm_moving_rps'],
                                ST['S2_arm_rate_below_within_s'],
                                ST['S3_min_continuous_below_s'])
chk('S1_底盤停穩', ok_b,
    f"凍結後 {d_b:+.3f}s 低於 {P['base_moving_mps']} m/s，持續 {c_b:.2f}s 至窗末"
    f"（期限 {ST['S1_base_rate_below_within_s']}s、需持續 ≥ {ST['S3_min_continuous_below_s']}s）")
chk('S2_手臂停穩', ok_a,
    f"凍結後 {d_a:+.3f}s 低於 {P['arm_moving_rps']} rad/s，持續 {c_a:.2f}s 至窗末")

# ---------------- M 停穩前最多走多遠（**窗內最大**，非終點淨值）----------------
kfr = int(np.argmin(np.abs(t - t_freeze)))
we = int(np.argmin(np.abs(t - W_END)))
def qmax(k0, k1):
    seg = q2[k0:k1 + 1]
    return float(np.max(np.abs(seg - q2[k0])))
def bmax(k0, k1):
    dx = L[k0:k1 + 1, i['base_x']] - L[k0, i['base_x']]
    dy = L[k0:k1 + 1, i['base_y']] - L[k0, i['base_y']]
    return float(np.max(np.hypot(dx, dy)))

m1 = qmax(kcut, kfr)
m2 = qmax(kfr, we)
m3 = qmax(kcut, we)
b4 = bmax(kcut, kfr)
b5 = bmax(kfr, we)
chk('M1_切斷→凍結最大位移', m1 <= M['M1_cut_to_freeze_max_rad'],
    f"{m1*1000:.4f} mrad（≤ {M['M1_cut_to_freeze_max_rad']*1000:.0f}；"
    '命令依設計保持到過期，名目 0.05×0.2=10.0 mrad）')
chk('M2_凍結後最大位移', m2 <= M['M2_after_freeze_max_rad'],
    f"{m2*1000:.4f} mrad（≤ {M['M2_after_freeze_max_rad']*1000:.1f}；"
    '**窗內最大、原始值、未扣漂移**；上限出自離線幾何推導）')
chk('M3_切斷後全段最大位移', m3 <= M['M3_whole_post_cut_max_rad'],
    f"{m3*1000:.4f} mrad（≤ {M['M3_whole_post_cut_max_rad']*1000:.0f}；"
    'M1+M2 導出的上界，非獨立約束）')
chk('M4_底盤切斷→凍結最大位移', b4 <= M['M4_base_cut_to_freeze_max_m'],
    f"{b4*1000:.3f} mm（≤ {M['M4_base_cut_to_freeze_max_m']*1000:.0f}）")
chk('M5_底盤凍結後最大位移', b5 <= M['M5_base_after_freeze_max_m'],
    f"{b5*1000:.3f} mm（≤ {M['M5_base_after_freeze_max_m']*1000:.0f}；未扣漂移）")

# ---------------- C 是否在追蹤凍結目標（另列，不替代 M）----------------
sp_fr = float(sp_f)
err = q2 - sp_fr
final_err = float(err[we])
sgn = 1.0 if (sp_fr - q2[kfr]) >= 0 else -1.0
over = float(max(0.0, np.max((err[kfr:we + 1]) * sgn)))
chk('C1_末值距凍結設定點', abs(final_err) <= CV['C1_final_err_to_frozen_sp_rad_max'],
    f"{final_err*1000:+.4f} mrad（≤ {CV['C1_final_err_to_frozen_sp_rad_max']*1000:.0f}；"
    '**停在設定點附近，不是恰好停在設定點上**）')
chk('C2_超越量', over <= CV['C2_overshoot_rad_max'],
    f"{over*1000:.4f} mrad（≤ {CV['C2_overshoot_rad_max']*1000:.0f}；另列回報，不替代 M）")

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
print(f'  位移上限出自 {M["budget_source"]}（離線推導，與任何趟次量值無關）')
print('  **不追溯**：v1 趟次一律保留原判定')
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}")

json.dump({'schema': 'wb_cutoff_check/2', 'run': a.run, 'spec_path': a.spec,
           'spec_schema': S['schema'], 'spec_version': S['version'],
           'spec_sha256': spec_sha, 'spec_copied_to': 'criteria_used.yaml',
           'cut_sim_t': T_CUT, 'observe_window_s': [T_CUT, round(W_END, 3)],
           'retrospective': bool(a.retrospective),
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
               'base_settle_delay_s': round(d_b, 4),
               'arm_settle_delay_s': round(d_a, 4),
               'arm_max_cut_to_freeze_rad': round(m1, 6),
               'arm_max_after_freeze_rad': round(m2, 6),
               'arm_max_whole_post_cut_rad': round(m3, 6),
               'base_max_cut_to_freeze_m': round(b4, 6),
               'base_max_after_freeze_m': round(b5, 6),
               'final_err_to_frozen_sp_rad': round(final_err, 6),
               'overshoot_rad': round(over, 6)},
           'topic_stop_evidence': ('由獨立觀測程序 wb_topic_recorder 記錄的 '
                                   '/wb_vel_cmd 實際訊息，非執行端內部 cmd_age 推論'),
           'drift_subtraction_applied': False,
           'exact_zero_not_required': True,
           'travel_budget_source': M['budget_source'],
           'not_retroactive': S['not_retroactive'],
           'scope_limit': S['scope_limit'],
           'displacement_split_note': ('位移分「切斷→凍結（命令依設計保持到過期）」、'
                                       '「凍結後（非命令運動）」與「切斷後全段」三段，'
                                       '一律取**窗內最大原始值**，非終點淨值；'
                                       '收斂判準另列，**不替代**位移上限'),
           'differs_from_upstream_zero_test': ('上游歸零趟次 frozen_steps=0，'
                                               '接收端一路收到新鮮零命令；'
                                               '本趟話題實際停止更新')},
          open(OUT, 'w'), ensure_ascii=False, indent=1)
print(f'  -> {OUT}')
sys.exit(0 if ok else 1)
