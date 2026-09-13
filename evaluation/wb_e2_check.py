"""E2 接入驗收的離線判定：只讀趟次輸出，不開模擬器。

門檻由 YAML 載入並驗證版本；規格全文複製進趟次目錄。
沿用既有防護：不同規格版本需 --retrospective、同版本需 --allow-recheck。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import yaml

DEFAULT_SPEC = 'evaluation/results/specs/wb_e2_criteria_v1.yaml'
WANT = 'wb_e2_criteria/1'

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
if not S['profile'].get('bounds_unchanged'):
    print('**規格未宣告界限未動 —— 拒絕判定**'); sys.exit(3)
T, L, B, SA, D = (S['T_triggered'], S['L_compliance'], S['B_bounds'],
                  S['S_arm'], S['D_protections'])

OUT = os.path.join(a.run, 'e2_check.json')
USED = os.path.join(a.run, 'criteria_used.yaml')
prev_ver = None
if os.path.exists(USED):
    try:
        prev_ver = yaml.safe_load(open(USED).read()).get('version')
    except Exception:
        prev_ver = '讀不出'
if prev_ver is not None and prev_ver != S['version']:
    if not a.retrospective:
        print(f'**拒絕覆蓋**：{a.run} 由規格 {prev_ver} 治理。加 --retrospective')
        sys.exit(4)
    OUT = os.path.join(a.run, f'e2_check_{S["version"]}_retrospective.json')
else:
    if os.path.exists(OUT) and not a.allow_recheck:
        print(f'**拒絕覆寫**：{OUT} 已存在。加 --allow-recheck'); sys.exit(5)
    shutil.copyfile(a.spec, USED)

run = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
cols = run['log_cols']
Lg = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
wl = run.get('cmd_chain', {}).get('wheel_limit', {})

res = []
def chk(k, ok, d):
    res.append({'key': k, 'pass': bool(ok), 'detail': d})

if run.get('execution_version') != 'E2':
    print('**本趟不是 E2 —— 判定不成立**'); sys.exit(3)

lam = Lg[:, i['lam']]
mod = Lg[:, i['modified']]
ws = Lg[:, i['wheel_speed_max']]
wa = Lg[:, i['wheel_accel_max']]
fin = np.isfinite(lam)
n_mod = int(np.nansum(mod[np.isfinite(mod)]))
lam_min = float(np.nanmin(lam[fin])) if fin.any() else float('nan')

chk('T1_限制被觸發的步數', n_mod >= T['T1_min_modified_steps'],
    f"{n_mod} 步（≥ {T['T1_min_modified_steps']}）；"
    f"摘要 modified_steps={wl.get('modified_steps')}")
chk('T2_最小 λ 明顯小於 1', lam_min < T['T2_min_lam_below'],
    f"λ_min = {lam_min:.6f}（< {T['T2_min_lam_below']}）")
rows = run.get('limit_rows') or []
reasons = sorted({r.get('reason') for r in rows if r.get('modified')})
chk('T3_被限制時有記錄原因',
    bool(rows) and (n_mod == 0 or bool(reasons)),
    f'逐步記錄 {len(rows)} 筆；被限制步的原因 {reasons or "無"}')

tol = float(L['L3_tolerance'])
ws_max = float(np.nanmax(ws[np.isfinite(ws)])) if np.isfinite(ws).any() else float('nan')
wa_max = float(np.nanmax(wa[np.isfinite(wa)])) if np.isfinite(wa).any() else float('nan')
chk('L1_輪速逐步合規', ws_max <= L['L1_wheel_speed_max_mps'] + tol,
    f"max {ws_max:.6f} ≤ {L['L1_wheel_speed_max_mps']}")
chk('L2_輪加速度逐步合規', wa_max <= L['L2_wheel_accel_max_mps2'] + tol,
    f"max {wa_max:.6f} ≤ {L['L2_wheel_accel_max_mps2']}")

vx, wz = Lg[:, i['vx_cmd']], Lg[:, i['wz_cmd']]
vmax = float(np.nanmax(np.abs(vx[np.isfinite(vx)])))
wmax = float(np.nanmax(np.abs(wz[np.isfinite(wz)])))
chk('B1_線速度未超界', vmax <= B['B1_lin_mps_max'] + 1e-9,
    f"max |vx| {vmax:.6f} ≤ {B['B1_lin_mps_max']}")
chk('B2_角速度未超界', wmax <= B['B2_ang_rps_max'] + 1e-9,
    f"max |wz| {wmax:.6f} ≤ {B['B2_ang_rps_max']}")
lsb = run.get('low_speed_interface_bound', {})
chk('B3_界限設定未被放寬',
    float(lsb.get('lin_mps', -1)) == B['B1_lin_mps_max']
    and float(lsb.get('ang_rps', -1)) == B['B2_ang_rps_max'],
    f"執行端界限 {lsb.get('lin_mps')}/{lsb.get('ang_rps')}")

cc = run['cmd_chain']
chk('S1_手臂門檻不變',
    float(cc['config']['arm_rate_max']) == SA['S1_arm_rate_max'],
    f"arm_rate_max={cc['config']['arm_rate_max']}")
chk('S2_rejected', cc.get('rejected') == SA['S2_rejected_eq'],
    f"rejected={cc.get('rejected')}")
chk('S3_fail_is_none', (cc.get('fail') is None) == SA['S3_fail_is_none'],
    f"fail={cc.get('fail')}")

chk('D1_pregrasp 由獨立條件禁止',
    run.get('pregrasp_preconditions_met') is False
    and run.get('wheel_level_limiting_implemented') is True,
    f"wheel_limit_implemented={run.get('wheel_level_limiting_implemented')}, "
    f"pregrasp_preconditions_met={run.get('pregrasp_preconditions_met')}")
chk('D2_熱中止門檻', float(run.get('cpu_limit_c', -1)) == float(D['thermal_abort_c']),
    f"cpu_limit_c={run.get('cpu_limit_c')}，峰值 {run.get('cpu_temp_max_c')} °C")

npass = sum(r['pass'] for r in res)
ok = npass == len(res)
print('\n=== E2 接入驗收 ===')
print(f'  規格 {a.spec}  schema {S["schema"]} version {S["version"]} '
      f'sha256 {spec_sha[:16]}')
for r in res:
    print(f"  [{'通過' if r['pass'] else '**未通過**'}] {r['key']:24} {r['detail']}")
print(f'\n  {npass}/{len(res)} 項通過')
print(f"  適用範圍：{S['scope_limit'].strip().splitlines()[0]}")
print(f"  結論：{'判定通過' if ok else '**判定未通過**；門檻不下修'}")
json.dump({'schema': 'wb_e2_check/1', 'run': a.run, 'spec_path': a.spec,
           'spec_version': S['version'], 'spec_sha256': spec_sha,
           'checks': res, 'passed': npass, 'total': len(res), 'all_pass': ok,
           'measured': {'modified_steps': n_mod, 'lam_min': lam_min,
                        'wheel_speed_max': ws_max, 'wheel_accel_max': wa_max,
                        'vx_max': vmax, 'wz_max': wmax},
           'scope_limit': S['scope_limit'], 'not_retroactive': S['not_retroactive'],
           'not_claimed': ('上游全身安全性在命令被修改後**未重新論證**；'
                           '停止掃掠範圍**無避碰保證**；pregrasp 仍禁止')},
          open(OUT, 'w'), ensure_ascii=False, indent=1)
print(f'  -> {OUT}')
sys.exit(0 if ok else 1)
