#!/usr/bin/env python3
"""抽屜在階段 A 有沒有被動到：**位移與接觸分開判定**。

兩個軸是獨立的，不是一條階梯
----------------------------
* **位移**：開度（抽屜剛體世界 y 相對零點的位移）超過靜止基線的門檻。
  這只能支持一句話 —— 「抽屜位移超過靜止基線」。
  它**不能**證明夾爪碰到抽屜，也不能證明是機器人推的：同一個數值也可能來自
  求解器數值漂移、接觸以外的擾動，或基線本身的配置差異。
* **接觸**：PhysX 接觸力合力（`drawer_contact_fmag`）超過基線的噪訊底線，
  或獨立的真值最短距離曾經 <= 接觸門檻。接觸要看**這一組**，不是看開度。

兩者分開報告，判定字串同時說出兩邊的狀態，不把其中一個當成另一個的證據。

門檻不給預設值
--------------
開度漂移與接觸力的噪訊底線都取決於質量、阻尼、求解器設定與步長，
**沒有量過就不該猜**。門檻要嘛由基線趟次依**事前寫好的取值規則**產生
（`--baseline` ＋ `--rule`），要嘛顯式給（`--tol-m` / `--contact-force-tol-n`）。
兩者都沒有就拒絕下判定。

單趟基線給的是**這次模擬配置的工程偵測門檻**，不是所有情況的漂移上界。
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

# 配置指紋的欄位。門檻只對**同一配置**有效，所以每一項都要逐一比對。
#   asset_sha256  資產**內容**雜湊 —— 檔名相同但幾何改過時，只比檔名會悄悄
#                 沿用另一個場景的門檻
#   drawer_pose_xy 擺放位置。把手離機器人的距離變了，接觸與位移的意義就變了
#   physics_dt_s  積分步長。漂移量取決於它
#   opening_formula 開度讀法。換了讀法（例如改讀關節座標）數值口徑就不同
#   drawer_schema / drawer_name 資產版本與名稱（schema 升版代表語意改過）
FP_FIELDS = ('asset_sha256', 'drawer_pose_xy', 'physics_dt_s',
             'opening_formula', 'drawer_schema', 'drawer_name')


def config_fingerprint(rec):
    """由趟次紀錄取出配置指紋。缺任何一項就留 None —— **缺項視為不符**，
    不以「大概一樣」通融。"""
    dw = (rec or {}).get('drawer') or {}
    return {
        'asset_sha256': dw.get('asset_sha256'),
        'drawer_pose_xy': (None if dw.get('pose_xy') is None
                           else [float(v) for v in dw['pose_xy']]),
        'physics_dt_s': (None if dw.get('physics_dt_s') is None
                         else float(dw['physics_dt_s'])),
        'opening_formula': dw.get('opening_formula'),
        'drawer_schema': dw.get('schema'),
        'drawer_name': dw.get('name'),
        'asset_path': dw.get('asset'),      # 僅供人閱讀，**不參與比對**
    }


def compare_fingerprint(want, got, pose_tol_m=1e-9, dt_tol_s=1e-12):
    """逐項比對配置指紋。回傳不符項目的串列（空 = 相符）。"""
    bad = []
    for k in FP_FIELDS:
        a_, b_ = (want or {}).get(k), (got or {}).get(k)
        if a_ is None or b_ is None:
            bad.append(f'{k}：門檻 {a_!r} / 本趟 {b_!r} —— **缺項視為不符**')
            continue
        if k == 'drawer_pose_xy':
            if (len(a_) != len(b_)
                    or max(abs(float(x) - float(y))
                           for x, y in zip(a_, b_)) > pose_tol_m):
                bad.append(f'drawer_pose_xy：門檻 {a_} / 本趟 {b_}')
        elif k == 'physics_dt_s':
            if abs(float(a_) - float(b_)) > dt_tol_s:
                bad.append(f'physics_dt_s：門檻 {a_} / 本趟 {b_}')
        elif a_ != b_:
            _sa = str(a_); _sb = str(b_)
            bad.append(f'{k}：門檻 {_sa[:24]}… / 本趟 {_sb[:24]}…'
                       if len(_sa) > 24 else f'{k}：門檻 {_sa} / 本趟 {_sb}')
    return bad


# 事前定版的取值規則（見 evaluation/results/wgmpc_stage_a_baseline_rule.yaml）
RULE_MAXABS_PLUS_RES = 'max_abs_plus_resolution'
RULES = (RULE_MAXABS_PLUS_RES,)


def derive_threshold(series, rule=RULE_MAXABS_PLUS_RES, resolution=None,
                     label=''):
    """依**事前寫好的規則**由基線序列產生門檻。回傳 (門檻, 依據字典)。

    `max_abs_plus_resolution`：基線**全窗**最大絕對值 ＋ 紀錄解析度的預留量。
    全窗而不是末值：漂移不保證單調，取末值會低估。
    預留量用紀錄解析度（落盤的四捨五入位數），因為低於它的差異讀不出來。
    """
    if rule not in RULES:
        raise ValueError(f'未知取值規則 {rule!r}；可用 {RULES}')
    x = np.asarray(series, float)
    fin = np.isfinite(x)
    if not fin.any():
        raise ValueError(f'基線序列（{label}）沒有有效值：無法產生門檻')
    peak = float(np.abs(x[fin]).max())
    if resolution is None or not math.isfinite(float(resolution)):
        raise ValueError('必須給紀錄解析度（resolution）作為預留量，'
                         '否則門檻會落在讀不出來的差異之下')
    tol = peak + float(resolution)
    return tol, {
        'rule': rule,
        'label': label,
        'baseline_peak_abs': peak,
        'resolution_allowance': float(resolution),
        'threshold': tol,
        'n_samples': int(fin.sum()),
        'n_nonfinite': int((~fin).sum()),
        'window': '全窗（不是末值）—— 漂移不保證單調，取末值會低估',
        'scope': '**這次模擬配置的工程偵測門檻**，不是所有情況的漂移上界',
    }


def classify(opening, tol_m, contact_fmag=None, contact_force_tol_n=None,
             d_min=None, contact_d_tol_m=0.0, t=None):
    """回傳判定字典。位移與接觸**分開**。"""
    if tol_m is None or not math.isfinite(float(tol_m)):
        raise ValueError('tol_m 必須給定：開度漂移的噪訊底線沒有量過不該猜')
    q = np.asarray(opening, float)
    fin = np.isfinite(q)
    if not fin.any():
        raise ValueError('開度全為非有限值：無法判定（缺資料不等於沒有位移）')
    t = None if t is None else np.asarray(t, float)
    k = int(np.argmax(np.where(fin, np.abs(q), -np.inf)))
    peak = float(np.abs(q[fin]).max())
    out = {
        'n_samples': int(len(q)),
        'n_finite_opening': int(fin.sum()),
        # ---- 軸一：位移 ----
        'displacement': {
            'opening_max_abs_m': peak,
            'opening_at_max_m': float(q[k]),
            't_at_max': (None if t is None else float(t[k])),
            'tol_m': float(tol_m),
            'n_samples_beyond_tol': int((np.abs(q[fin]) > float(tol_m)).sum()),
            'over_baseline': bool(peak > float(tol_m)),
            'claim_supported': ('抽屜位移超過靜止基線' if peak > float(tol_m)
                                else '抽屜位移未超過靜止基線'),
            'claim_NOT_supported': ('**不能**單憑開度證明夾爪碰到或推了抽屜；'
                                    '接觸看獨立的接觸力／距離證據'),
            'threshold_is': ('紀錄變化偵測門檻（量的是紀錄解析度）'
                             '，不是物理漂移上界'),
        },
    }
    # ---- 軸二：接觸（獨立證據）----
    con = {'force': None, 'distance': None}
    if contact_fmag is None:
        con['force'] = {'available': False,
                        'note': '**缺接觸力紀錄** ⇒ 接觸與否不判定；'
                                '缺少不等於沒有發生'}
    elif contact_force_tol_n is None or not math.isfinite(
            float(contact_force_tol_n)):
        con['force'] = {'available': False,
                        'note': '有接觸力紀錄但**沒有門檻** ⇒ 不判定；'
                                '接觸力的噪訊底線沒有量過不該猜'}
    else:
        f = np.asarray(contact_fmag, float)
        ffin = np.isfinite(f)
        if not ffin.any():
            con['force'] = {'available': False,
                            'note': '接觸力全為非有限值 ⇒ 不判定'}
        else:
            j = int(np.argmax(np.where(ffin, f, -np.inf)))
            con['force'] = {
                'available': True,
                'fmag_max_n': float(f[ffin].max()),
                't_at_max': (None if t is None else float(t[j])),
                'tol_n': float(contact_force_tol_n),
                'n_samples_beyond_tol':
                    int((f[ffin] > float(contact_force_tol_n)).sum()),
                'contact': bool(f[ffin].max() > float(contact_force_tol_n)),
            }
    if d_min is None:
        con['distance'] = {'available': False,
                           'note': '**缺真值最短距離紀錄** ⇒ 這一路證據不判定'}
    else:
        d = np.asarray(d_min, float)
        dfin = np.isfinite(d)
        if not dfin.any():
            con['distance'] = {'available': False,
                               'note': '最短距離全為非有限值 ⇒ 不判定'}
        else:
            i = int(np.argmin(np.where(dfin, d, np.inf)))
            con['distance'] = {
                'available': True,
                'd_min_m': float(d[dfin].min()),
                't_at_min': (None if t is None else float(t[i])),
                'tol_m': float(contact_d_tol_m),
                'contact': bool(d[dfin].min() <= float(contact_d_tol_m)),
            }
    # **接觸偵測的能力限制**，與判定一起輸出（不是只寫在規則檔裡）
    con['sensing_limits'] = {
        'zero_does_not_prove_no_contact':
            '淨合力為零**不證明沒有接觸**（合力可相互抵銷；未取得接觸點與法向）',
        'zero_does_not_prove_sensor_would_report':
            '淨合力為零**不證明有接觸時感測器一定會回報**',
        'needs': '已知接觸的**正向對照**（刻意接觸並確認讀數非零）',
    }
    avail = [c for c in (con['force'], con['distance'])
             if c and c.get('available')]
    if not avail:
        con['verdict'] = 'undetermined'
        con['verdict_meaning'] = '接觸與否**無法判定**（沒有可用的獨立證據）'
    elif any(c['contact'] for c in avail):
        con['verdict'] = 'contact_evidence'
        con['verdict_meaning'] = '獨立證據顯示有接觸'
    else:
        con['verdict'] = 'no_contact_evidence'
        con['verdict_meaning'] = '可用的獨立證據**未**顯示接觸'
    out['contact'] = con
    # ---- 合併判定：兩邊都說出來，不互相代替 ----
    d_lbl = ('位移超過靜止基線' if out['displacement']['over_baseline']
             else '位移未超過靜止基線')
    out['verdict'] = (f'{"displacement_over_baseline" if out["displacement"]["over_baseline"] else "displacement_within_baseline"}'
                      f'|{con["verdict"]}')
    out['verdict_meaning'] = f'{d_lbl}；接觸：{con["verdict_meaning"]}'
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('record', help='趟次紀錄 JSON（wb_run.json）')
    ap.add_argument('--tol-m', type=float, default=None)
    ap.add_argument('--contact-force-tol-n', type=float, default=None)
    ap.add_argument('--contact-d-tol-m', type=float, default=0.0)
    ap.add_argument('--baseline', default='',
                    help='基線趟次的門檻檔（wgmpc_stage_a_baseline_derive.py 產出）')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rec = json.load(open(a.record))
    tol, ftol, src = a.tol_m, a.contact_force_tol_n, {}
    if a.baseline:
        b = json.load(open(a.baseline))
        if b.get('verdict') == 'rejected':
            print(f'**門檻檔是被拒絕的基線**（{a.baseline}）：'
                  f'{b.get("reasons")}', file=sys.stderr)
            return 78
        # ---- 門檻與配置綁定：不符就**拒用**，不套用另一趟的門檻 ----
        want = b.get('config_fingerprint')
        if want is None:
            print(f'**門檻檔沒有配置指紋**（{a.baseline}，schema '
                  f'{b.get("schema")}）：無法確認它來自同一配置，拒用。'
                  f'請以 wgmpc_stage_a_baseline_derive.py 重新產生',
                  file=sys.stderr)
            return 78
        mism = compare_fingerprint(want, config_fingerprint(rec))
        if mism:
            print('**門檻與本趟配置不符，拒用**（不套用另一趟的門檻）：',
                  file=sys.stderr)
            for m in mism:
                print(f'  - {m}', file=sys.stderr)
            return 78
        if tol is None:
            tol = b['opening_threshold']['threshold']
        if ftol is None and 'contact_force_threshold' in b:
            ftol = b['contact_force_threshold']['threshold']
        src = {'from': a.baseline, 'opening': b.get('opening_threshold'),
               'contact_force': b.get('contact_force_threshold'),
               'baseline_run': b.get('baseline_run'),
               'config_fingerprint_matched': want}
    if tol is None:
        print('**沒有門檻**：開度漂移的噪訊底線沒有量過，不以猜測的門檻下判定。'
              '請給 --baseline 或 --tol-m', file=sys.stderr)
        return 73
    cols = rec.get('log_cols')
    log = rec.get('log')
    if not cols or not log:
        print('**紀錄缺 log_cols／log**，不下判定', file=sys.stderr)
        return 74
    ix = {c: i for i, c in enumerate(cols)}
    if 'drawer_opening' not in ix:
        print('**紀錄沒有 drawer_opening 欄**：這趟可能不是 solver_drawer 模式。'
              '缺少不等於沒有位移，不下判定', file=sys.stderr)
        return 74
    g = lambda c: ([r[ix[c]] for r in log] if c in ix else None)
    res = classify(g('drawer_opening'), tol,
                   contact_fmag=g('drawer_contact_fmag'),
                   contact_force_tol_n=ftol,
                   d_min=g('drawer_d_min'),
                   contact_d_tol_m=a.contact_d_tol_m,
                   t=g('t'))
    res['threshold_source'] = src or {'from': '命令列', 'tol_m': float(tol),
                                      'contact_force_tol_n': ftol}
    dd, cc = res['displacement'], res['contact']
    print(f'抽屜判定：{res["verdict"]}')
    print(f'  {res["verdict_meaning"]}')
    print(f'  位移：|開度| 最大 {dd["opening_max_abs_m"]*1e3:.4f} mm'
          f'（門檻 {dd["tol_m"]*1e3:.4f} mm，超出門檻的取樣 '
          f'{dd["n_samples_beyond_tol"]}/{dd["n_finite_opening"] if "n_finite_opening" in dd else res["n_finite_opening"]}）')
    print(f'    {dd["claim_supported"]}')
    print(f'    {dd["claim_NOT_supported"]}')
    for k in ('force', 'distance'):
        c = cc[k]
        if c and c.get('available'):
            if k == 'force':
                print(f'  接觸力：最大 {c["fmag_max_n"]:.6f} N'
                      f'（門檻 {c["tol_n"]:.6f} N）⇒ '
                      f'{"有接觸" if c["contact"] else "未顯示接觸"}')
            else:
                print(f'  真值最短距離：最小 {c["d_min_m"]*1e3:.4f} mm'
                      f'（門檻 {c["tol_m"]*1e3:.4f} mm）⇒ '
                      f'{"有接觸" if c["contact"] else "未顯示接觸"}')
        elif c:
            print(f'  {k}：{c["note"]}')
    if a.out:
        json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    # 位移超過基線或有接觸證據 ⇒ 非零；接觸無法判定時**也**非零（不當成通過）
    if dd['over_baseline'] or cc['verdict'] != 'no_contact_evidence':
        return 75
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
