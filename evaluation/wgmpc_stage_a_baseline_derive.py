#!/usr/bin/env python3
"""由基線趟次依**事前定版的規則**產生階段 A 的門檻。

規則在 evaluation/results/wgmpc_stage_a_baseline_rule.yaml，**在跑基線之前**
就已定版。本程式只套用它 —— 沒有任何「依結果調整」的選項。

為什麼「有紀錄」不等於「可用來產生門檻」
----------------------------------------
提早中止但已落盤的趟次會產生**偏低**的門檻（觀察窗短 ⇒ 全窗最大值小），
而偏低的門檻會讓階段 A 的任何微小數值都看起來超標。所以產生門檻之前要核
四件事：完整觀察、正常收尾、關鍵讀值完整、機器人確實未受命令驅動。
任一項不成立就**拒絕產生門檻**，不以「數字看起來合理」通融。

門檻與配置綁定
--------------
門檻只對「同一份資產、同一個擺放、同一個物理步長、同一套開度讀法」的趟次有效。
輸出檔帶一份**配置指紋**（含資產內容雜湊），階段 A 的判定器要重算並比對 ——
檔名相同但幾何改過的資產會被抓出來。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import re
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wgmpc_stage_a_drawer_check import (derive_threshold,              # noqa: E402
                                        RULE_MAXABS_PLUS_RES,
                                        config_fingerprint)

RULE_KEY_EXTRACT = '取值規則'
RULE_KEY_ACCEPT = '基線趟次的驗收條件'


def _rule_section(rule, prefix):
    k = [k for k in rule if k.startswith(prefix)]
    if len(k) != 1:
        raise KeyError(f'規則檔缺少（或重複）以 {prefix!r} 開頭的段落：{k}')
    return rule[k[0]]


def check_timestamps(t, observe_sim_s, dt, max_gap_steps,
                     start_allow_steps=5.0):
    """時間戳必須**涵蓋全窗**。回傳 (證據, 失敗理由串列)。

    為什麼只核筆數不夠：筆數可以達到九成而資料集中在前段 ——
    例如 sim_time_s 寫 60、收尾正常，但 log 只到 53.99 s。
    那樣門檻仍然系統性偏低，而且從筆數看不出來。
    """
    ev, bad = {}, []
    x = np.asarray(t, float)
    if not len(x):
        return {'n': 0}, ['時間戳為空']
    fin = np.isfinite(x)
    ev['n_nonfinite'] = int((~fin).sum())
    if ev['n_nonfinite']:
        bad.append(f'時間戳有 {ev["n_nonfinite"]} 筆非有限值')
        return ev, bad
    ev['t_first'] = float(x[0])
    ev['t_last'] = float(x[-1])
    ev['required_last'] = float(observe_sim_s) - 2.0 * float(dt)
    ev['start_allow_steps'] = float(start_allow_steps)
    d = np.diff(x)
    ev['max_gap_s'] = float(d.max()) if len(d) else 0.0
    ev['max_gap_steps'] = (ev['max_gap_s'] / float(dt)) if dt else float('inf')
    ev['min_gap_s'] = float(d.min()) if len(d) else 0.0
    ev['gap_limit_steps'] = float(max_gap_steps)
    if len(d) and d.min() < 0.0:
        bad.append(f'時間戳遞減（最小間隔 {ev["min_gap_s"]:.6g} s）')
    if ev['t_first'] > float(start_allow_steps) * float(dt):
        bad.append(f'起點未涵蓋：第一筆時間戳 {ev["t_first"]:.4f} s > '
                   f'{start_allow_steps:g} 個物理步 '
                   f'{start_allow_steps*float(dt):.4f} s')
    if ev['t_last'] < ev['required_last']:
        bad.append(f'終點未涵蓋：最後一筆時間戳 {ev["t_last"]:.4f} s < '
                   f'{ev["required_last"]:.4f} s'
                   f'（要求 {observe_sim_s} s 減 2 個物理步）'
                   f' ⇒ 資料集中在前段，門檻會系統性偏低')
    if ev['max_gap_steps'] > float(max_gap_steps) + 1e-9:
        bad.append(f'中間有缺口：最大相鄰間隔 {ev["max_gap_s"]:.6g} s '
                   f'= {ev["max_gap_steps"]:.1f} 個物理步 > '
                   f'上限 {max_gap_steps}')
    return ev, bad


def check_acceptance(rec, accept, observe_sim_s, physics_dt_s, cols, log,
                     max_gap_steps=5, start_allow_steps=5.0,
                     base_disp_limit_m=None):
    """基線趟次的驗收。回傳 (是否通過, 證據字典, 失敗理由串列)。"""
    ix = {c: i for i, c in enumerate(cols)}
    ev, bad = {}, []

    # ---- 完整觀察 ----
    sim_t = rec.get('sim_time_s')
    ev['sim_time_s'] = sim_t
    ev['observe_sim_s_required'] = float(observe_sim_s)
    if sim_t is None:
        bad.append('紀錄沒有 sim_time_s ⇒ 無法確認觀察是否完整')
    elif float(sim_t) < float(observe_sim_s) - float(physics_dt_s):
        bad.append(f'觀察不完整：sim_time_s {float(sim_t):.3f} s < '
                   f'要求 {float(observe_sim_s):.3f} s'
                   f'（容差一個物理步 {physics_dt_s}）⇒ '
                   f'全窗最大值會系統性偏低')

    # ---- 正常收尾 ----
    sr = rec.get('stop_reason')
    ev['stop_reason'] = sr
    if sr != 'sim_limit':
        bad.append(f'收尾不正常：stop_reason = {sr!r}，要求 sim_limit'
                   f'（提早中止的趟次不得用來產生門檻）')

    # ---- 關鍵讀值完整 ----
    for k in ('drawer_read_fail_steps', 'drawer_contact_read_fail_steps'):
        v = rec.get(k)
        ev[k] = v
        if v is None:
            bad.append(f'紀錄沒有 {k} ⇒ 無法確認讀值完整')
        elif int(v) != 0:
            _e1 = rec.get('drawer_contact_read_first_error'
                          if 'contact' in k else 'drawer_read_first_error')
            bad.append(f'{k} = {v} ⇒ 有步數讀不到抽屜狀態'
                       + (f'；首筆例外 {_e1}' if _e1 else
                          '；**紀錄裡沒有例外內容**，無法診斷'))
    n_expect = int(round(float(observe_sim_s) / float(physics_dt_s)))
    ev['n_samples'] = len(log)
    ev['n_samples_expected'] = n_expect
    if len(log) < 0.9 * n_expect:
        bad.append(f'取樣筆數 {len(log)} 低於預期 {n_expect} 的九成')
    # **時間戳涵蓋全窗**：筆數達標但資料集中在前段的情形，只看筆數抓不到
    if 't' not in ix:
        bad.append('紀錄沒有 t 欄 ⇒ 無法確認時間戳涵蓋全窗')
    else:
        tev, tbad = check_timestamps([r[ix['t']] for r in log],
                                     observe_sim_s, physics_dt_s,
                                     max_gap_steps, start_allow_steps)
        ev['timestamps'] = tev
        bad += tbad
    for c in ('drawer_opening', 'drawer_contact_fmag'):
        if c not in ix:
            bad.append(f'紀錄沒有 {c} 欄 ⇒ 這趟不是 solver_drawer 模式')
            continue
        x = np.asarray([r[ix[c]] for r in log], float)
        nbad = int((~np.isfinite(x)).sum())
        ev[f'{c}_nonfinite'] = nbad
        if nbad:
            bad.append(f'{c} 有 {nbad} 筆非有限值 ⇒ 關鍵讀值不完整'
                       f'（NaN 表示沒量到，不是沒有發生）')

    # ---- 機器人確實未受命令驅動 ----
    cc = rec.get('cmd_chain') or {}
    ev['cmd_chain_received'] = cc.get('received')
    if cc.get('received') is None:
        bad.append('紀錄沒有 cmd_chain.received ⇒ 無法確認未受命令')
    elif int(cc['received']) != 0:
        bad.append(f'命令鏈收到 {cc["received"]} 筆命令 ⇒ '
                   f'這趟不是「不發移動命令」的靜止基線')
    if 'integrating' in ix:
        itg = np.asarray([r[ix['integrating']] for r in log], float)
        ev['n_integrating'] = int(np.nansum(itg))
        if ev['n_integrating'] > 0:
            bad.append(f'有 {ev["n_integrating"]} 步在積分命令')
    else:
        bad.append('紀錄沒有 integrating 欄 ⇒ 無法確認未受命令')
    # ---- 底盤：**全窗相對起點的最大位移**，不是首末差 ----
    # 首末差排除不了中途移動與返回。
    if 'base_x' in ix and 'base_y' in ix:
        bx = np.asarray([r[ix['base_x']] for r in log], float)
        by = np.asarray([r[ix['base_y']] for r in log], float)
        ok_xy = np.isfinite(bx) & np.isfinite(by)
        if not ok_xy.any():
            bad.append('base_x／base_y 全為非有限值 ⇒ 無法確認底盤未移動')
        else:
            bxf, byf = bx[ok_xy], by[ok_xy]
            disp = np.hypot(bxf - bxf[0], byf - byf[0])
            k = int(np.argmax(disp))
            ev['base_disp_max_m'] = float(disp.max())
            ev['base_disp_max_at_index'] = k
            ev['base_disp_first_to_last_m'] = float(
                np.hypot(bxf[-1] - bxf[0], byf[-1] - byf[0]))
            ev['base_path_m'] = float(
                np.hypot(np.diff(bxf), np.diff(byf)).sum())
            ev['base_disp_criterion'] = '全窗相對起點的最大位移（非首末差）'
            if base_disp_limit_m is None:
                bad.append('底盤位移門檻**尚未核准定版**：'
                           f'本趟全窗相對起點的最大位移為 '
                           f'{ev["base_disp_max_m"]*1e3:.4f} mm'
                           f'（首末差 {ev["base_disp_first_to_last_m"]*1e3:.4f} mm、'
                           f'路徑長 {ev["base_path_m"]*1e3:.4f} mm）。'
                           f'請以 --base-disp-limit-mm 顯式給值')
            else:
                ev['base_disp_limit_m'] = float(base_disp_limit_m)
                if ev['base_disp_max_m'] > float(base_disp_limit_m):
                    bad.append(
                        f'底盤全窗最大位移 {ev["base_disp_max_m"]*1e3:.4f} mm > '
                        f'{float(base_disp_limit_m)*1e3:.4f} mm ⇒ 底盤並非靜止')
    else:
        bad.append('紀錄沒有 base_x／base_y 欄 ⇒ 無法確認底盤未移動')

    # ---- twist 與位移的矛盾：**留在驗收裡，不悄悄移除** ----
    # 判準：若 ∫|v| dt 與全窗最大位移相差超過 10 倍，必須同時出現「停更簽章」
    #（速度與位置自**同一步**起皆完全相同）。沒有簽章 ⇒ 未解釋的異常 ⇒ 拒絕。
    # 這不是把異常當成通過 —— 簽章與量到的數字都寫進證據與門檻檔。
    if ('base_lin_meas' in ix and 't' in ix
            and ev.get('base_disp_max_m') is not None):
        bl = np.asarray([r[ix['base_lin_meas']] for r in log], float)
        tt = np.asarray([r[ix['t']] for r in log], float)
        bfin = np.isfinite(bl) & np.isfinite(tt)
        if not bfin.any():
            bad.append('base_lin_meas 全為非有限值 ⇒ twist 異常無法刻畫')
        else:
            blf, ttf = bl[bfin], tt[bfin]
            integ = float(np.sum(np.abs(blf[:-1]) * np.diff(ttf)))
            ev['base_twist_integral_m'] = integ
            ev['base_lin_meas_max'] = float(np.abs(blf).max())
            dmax = max(ev['base_disp_max_m'], 1e-12)
            ratio = integ / dmax
            ev['base_twist_vs_disp_ratio'] = ratio
            if ratio > 10.0:
                # 停更簽章：速度與位置自同一步起都不再變化
                def _freeze_idx(a_):
                    ch = np.nonzero(np.diff(a_) != 0)[0]
                    return int(ch[-1] + 1) if len(ch) else 0
                iv = _freeze_idx(blf)
                ipx = _freeze_idx(bx[ok_xy])
                ev['freeze_idx_twist'] = iv
                ev['freeze_idx_pose'] = ipx
                ev['freeze_tail_samples'] = int(len(blf) - iv - 1)
                same_step = abs(iv - ipx) <= 1
                tail_const = (len(blf) - iv - 1) > 0 and bool(
                    np.all(blf[iv:] == blf[iv]))
                ev['freeze_signature'] = bool(same_step and tail_const)
                ev['freeze_note'] = (
                    'twist 與位置自同一步起皆不再變化 ⇒ 讀數為停更後的殘值，'
                    '既不能當移動的證據也不能當靜止的證據。'
                    '**機制未證實**（符合 PhysX 剛體睡眠的簽章，但未做對照實驗）')
                if not ev['freeze_signature']:
                    bad.append(
                        f'twist 與位移相差 {ratio:.1f} 倍（∫|v|dt '
                        f'{integ*1e3:.2f} mm vs 最大位移 '
                        f'{ev["base_disp_max_m"]*1e3:.4f} mm），'
                        f'且**沒有停更簽章**（twist 凍結於第 {iv} 筆、'
                        f'位置凍結於第 {ipx} 筆，尾段常數={tail_const}）'
                        f' ⇒ 未解釋的量測異常，拒絕')
    elif 'base_lin_meas' in ix:
        bad.append('無法刻畫 twist 異常（缺 t 欄或位移未算出）')

    # ---- 抽屜為被動件 ----
    dw = rec.get('drawer') or {}
    au = dw.get('authored') or {}
    ev['drawer_built'] = bool(dw)
    if not dw:
        bad.append('紀錄裡沒有抽屜資訊 ⇒ 這趟沒有抽屜，門檻無意義')
    for k in ('drive_stiffness', 'drive_damping', 'drive_max_force'):
        if k not in au:
            bad.append(f'抽屜紀錄缺 {k} ⇒ 無法確認它是被動件')
        elif float(au[k]) != 0.0:
            bad.append(f'抽屜 {k} 讀回 {au[k]} 不為 0 ⇒ 它不是被動件，'
                       f'漂移不是自由漂移')
    return (not bad), ev, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('record', help='基線趟次的 sim/wb_run.json')
    ap.add_argument('--rule', default=os.path.join(
        HERE, 'results/wgmpc_stage_a_baseline_rule.yaml'))
    ap.add_argument('--observe-sim-s', type=float, default=None,
                    help='要求的觀察時長；不給則取規則的下界')
    ap.add_argument('--base-disp-limit-mm', type=float, default=None,
                    help='底盤全窗最大位移的上限（mm）。不給則取規則檔已核定的值')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    rule = yaml.safe_load(open(a.rule))
    rr = _rule_section(rule, RULE_KEY_EXTRACT)
    if rr['規則名'] != RULE_MAXABS_PLUS_RES:
        print(f'**規則名 {rr["規則名"]!r} 與程式支援的不符**', file=sys.stderr)
        return 76
    res_m = float(rr['開度門檻']['解析度預留量_m'])
    res_n = float(rr['接觸力門檻']['解析度預留量_n'])
    cons = rule['可覆寫參數的約束']
    accept = _rule_section(rule, RULE_KEY_ACCEPT)
    need_obs = (float(a.observe_sim_s) if a.observe_sim_s is not None
                else float(cons['observe_sim_s']['min']))

    rec = json.load(open(a.record))
    cols, log = rec.get('log_cols'), rec.get('log')
    if not cols or not log:
        print('**基線紀錄缺 log_cols／log**', file=sys.stderr)
        return 74
    dw = rec.get('drawer') or {}
    dt_s = dw.get('physics_dt_s')
    if dt_s is None:
        print('**基線紀錄沒有 drawer.physics_dt_s** ⇒ 無法核對步長，'
              '不產生門檻', file=sys.stderr)
        return 74
    if abs(float(dt_s) - float(cons['physics_dt_s']['eq'])) > 1e-12:
        print(f'**物理步長 {dt_s} 不等於規則要求的 '
              f'{cons["physics_dt_s"]["eq"]}**', file=sys.stderr)
        return 77

    _tsreq = accept.get('時間戳涵蓋全窗') or {}
    _gap = float(_tsreq.get('max_gap_steps', 5))
    _sa = _tsreq.get('起點涵蓋', '')
    _m = re.search(r'<=\s*([0-9.]+)\s*個物理步', str(_sa))
    _start_allow = float(_m.group(1)) if _m else 5.0
    # **底盤位移門檻已核定為 1.0 mm**（Howard 2026-10-03）⇒ 自規則檔讀取。
    # 命令列仍可覆寫，覆寫值會寫進門檻檔的證據，事後看得出用的是哪一個。
    _bd = accept.get('機器人確實未受命令驅動') or []
    _m2 = re.search(r'<=\s*([0-9.]+)\s*mm',
                    ' '.join(str(x) for x in _bd))
    _disp_rule = (float(_m2.group(1)) * 1e-3) if _m2 else None
    _disp = (float(a.base_disp_limit_mm) * 1e-3
             if a.base_disp_limit_mm is not None else _disp_rule)
    _disp_src = ('命令列 --base-disp-limit-mm'
                 if a.base_disp_limit_mm is not None else '規則檔（已核定）')
    ok, ev, bad = check_acceptance(rec, accept, need_obs, float(dt_s),
                                   cols, log, max_gap_steps=_gap,
                                   start_allow_steps=_start_allow,
                                   base_disp_limit_m=_disp)
    if not ok:
        print('**基線驗收未通過，不產生門檻**：', file=sys.stderr)
        for b in bad:
            print(f'  - {b}', file=sys.stderr)
        if a.out:
            json.dump({'verdict': 'rejected', 'reasons': bad,
                       'evidence': ev, 'record': os.path.relpath(a.record)},
                      open(a.out, 'w'), ensure_ascii=False, indent=1)
        return 77

    ix = {c: i for i, c in enumerate(cols)}
    op = [r[ix['drawer_opening']] for r in log]
    cf = [r[ix['drawer_contact_fmag']] for r in log]
    t = [r[ix['t']] for r in log] if 't' in ix else None

    tol_m, meta_m = derive_threshold(op, RULE_MAXABS_PLUS_RES, res_m, '開度')
    tol_n, meta_n = derive_threshold(cf, RULE_MAXABS_PLUS_RES, res_n, '接觸力')
    fp = config_fingerprint(rec)
    out = {
        'schema': 'wgmpc_stage_a_baseline_threshold/2',
        'rule_file': os.path.relpath(a.rule),
        'rule_name': RULE_MAXABS_PLUS_RES,
        'derived_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        # **配置指紋**：階段 A 必須重算並逐項比對，不符即拒用
        'config_fingerprint': fp,
        'baseline_run': {
            'record': os.path.relpath(a.record),
            'n_steps': len(log),
            'sim_t_span_s': (None if t is None else
                             [float(min(t)), float(max(t))]),
            'drawer': dw,
            'acceptance': {'verdict': 'pass', 'evidence': ev,
                           'required_observe_sim_s': need_obs},
        },
        'opening_threshold': meta_m,
        'contact_force_threshold': meta_n,
        'scope': ('**這次模擬配置的工程偵測門檻**，不是所有情況的漂移上界；'
                  '單趟基線不提供統計區間'),
        'base_disp_limit_source': _disp_src,
        'wording': {
            'threshold_name': ('**紀錄變化偵測門檻** —— 量的是紀錄解析度，'
                               '不是物理漂移上界。基線全窗峰值為 0，'
                               '所以門檻等於解析度預留量本身'),
            'over_threshold_means': '讀值超過靜止基線',
            'over_threshold_does_NOT_mean': [
                '確定有接觸',
                '確定抽屜被推動',
            ],
            'contact_sensing': {
                'proven': '接觸力 API 能成功讀取（本趟 0 步失敗）',
                'NOT_proven': [
                    '淨合力為零**不證明沒有接觸**（合力可相互抵銷；'
                    '也未取得接觸點與法向）',
                    '淨合力為零**不證明有接觸時感測器一定會回報**',
                ],
                'needed': '已知接觸的**正向對照**（刻意接觸並確認讀數非零）',
            },
            'reproducibility': {
                'may_say': '同配置下**所測趟次可重現**（位元相同）',
                'may_NOT_say': ['模擬在所有條件下都是決定性的',
                                '位移不是隨機變數'],
                'why': '只測了同一份配置、同一台機器、同一個 Isaac 版本的數趟',
            },
        },
        'not_validated_by_this_baseline': [
            '交棒流程（前置調姿 → W-GMPC）—— 尚未實跑',
            '協同接近、到達與保持',
            '屏障在運動中的行為（機器人靜止）',
        ],
    }
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'[基線] 驗收通過：觀察 {ev["sim_time_s"]:.3f} s'
          f'（要求 {need_obs:.1f}）、收尾 {ev["stop_reason"]}、'
          f'命令 {ev["cmd_chain_received"]} 筆、'
          f'底盤全窗最大位移 {ev["base_disp_max_m"]*1e3:.4f} mm'
          f'（上限 {float(_disp)*1e3:.4f} mm）')
    if ev.get('freeze_signature') is not None:
        print(f'  twist 異常已刻畫：∫|v|dt／最大位移 = '
              f'{ev["base_twist_vs_disp_ratio"]:.1f} 倍，停更簽章 '
              f'{ev["freeze_signature"]}（twist 第 {ev["freeze_idx_twist"]} 筆、'
              f'位置第 {ev["freeze_idx_pose"]} 筆起不再變化，'
              f'尾段 {ev["freeze_tail_samples"]} 筆）')
        print(f'  {ev["freeze_note"]}')
    print(f'[基線] 規則 {RULE_MAXABS_PLUS_RES}（執行前定版）')
    print(f'  開度：全窗最大 |opening| = {meta_m["baseline_peak_abs"]*1e3:.6f} mm'
          f' ＋ 解析度預留 {res_m*1e3:.6f} mm'
          f' ⇒ **tol_m = {tol_m*1e3:.6f} mm**')
    print(f'  接觸力：全窗最大 |fmag| = {meta_n["baseline_peak_abs"]:.9f} N'
          f' ＋ 解析度預留 {res_n:.9f} N'
          f' ⇒ **tol_n = {tol_n:.9f} N**')
    print(f'  取樣 {meta_m["n_samples"]} 筆（非有限 {meta_m["n_nonfinite"]} 筆）')
    print(f'  配置指紋：資產 {fp["asset_sha256"][:12]}…、'
          f'擺放 {fp["drawer_pose_xy"]}、步長 {fp["physics_dt_s"]}')
    print(f'  門檻檔 {a.out} —— **固定供階段 A 使用，不逐趟重算**')
    print(f'  範圍：{out["scope"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
