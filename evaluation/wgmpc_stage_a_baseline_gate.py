#!/usr/bin/env python3
"""基線趟次的**啟動前閘門**：可覆寫參數必須符合事前定版的規則。

為什麼要這道閘門
----------------
覆寫環境變數本身不是問題 —— 問題是覆寫之後這趟就不再是「對齊階段 A 的
靜止基線」，而門檻卻照樣產生。最危險的一項是起跑溫度：把 42 °C 用環境變數
調高，等於把熱中止線往上推，而那是明文不得放寬的。

規則檔給每一項的允許範圍（`max` / `min` / `eq` / `source`），本程式逐項核對。
不符即回傳非零碼，運行器據此不啟動。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import yaml


def _finite_num(v):
    """是否為有限數值。**NaN 與 inf 一律不算**。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def check(cons, got, stage_cfg=None, asset_path=None):
    """逐項核對。回傳不符項目的串列（空 = 全部通過）。"""
    bad = []
    for key, spec in cons.items():
        if key == '說明' or not isinstance(spec, dict):
            continue
        # **已解除的約束**：規則檔明寫 enforced: false ⇒ 跳過。
        # 解除要在事前規則裡有授權與日期，不是靠環境變數繞過
        #（環境覆寫本來只允許調低）。
        if spec.get('enforced') is False:
            continue
        if key not in got:
            bad.append(f'{key}：規則有約束但本趟沒有提供值')
            continue
        v = got[key]
        # **數值必須先是有限值**。NaN 與任何上下界的比較都是 False，
        # 所以不先擋住的話 NaN 會「零違規」通過 —— 實測 observe_sim_s=NaN
        # 與 start_temp_c=NaN 都曾通過閘門。
        if any(k in spec for k in ('max', 'min')) or isinstance(
                spec.get('eq'), (int, float)) and not isinstance(
                spec.get('eq'), bool):
            if not _finite_num(v):
                bad.append(f'{key} = {v!r} 不是有限數值'
                           f'（NaN／inf 與任何界的比較都是 False，'
                           f'不先擋住會零違規通過）')
                continue
            if float(v) <= 0.0 and key in ('observe_sim_s', 'physics_dt_s'):
                bad.append(f'{key} = {v} 必須為正')
                continue
        if 'max' in spec:
            if float(v) > float(spec['max']):
                bad.append(f'{key} = {v} 超過上限 {spec["max"]}'
                           f'（{spec.get("rule", "")}）'
                           + (f'；{spec["why"]}' if 'why' in spec else ''))
        if 'min' in spec:
            if float(v) < float(spec['min']):
                bad.append(f'{key} = {v} 低於下限 {spec["min"]}'
                           f'（{spec.get("rule", "")}）'
                           + (f'；{spec["why"]}' if 'why' in spec else ''))
        if 'eq' in spec:
            want = spec['eq']
            same = (abs(float(v) - float(want)) <= 1e-12
                    if isinstance(want, (int, float))
                    and not isinstance(want, bool)
                    else str(v) == str(want))
            if not same:
                bad.append(f'{key} = {v!r} 必須等於 {want!r}'
                           f'（{spec.get("rule", "")}）'
                           + (f'；{spec["why"]}' if 'why' in spec else ''))
    # ---- 櫃體位姿：必須與階段 A 的設定相同 ----
    if 'drawer_pose_xy' in cons:
        if stage_cfg is None:
            bad.append('drawer_pose_xy 需要階段 A 設定檔才能核對，但沒有提供')
        else:
            sc = yaml.safe_load(open(stage_cfg))
            want = [float(x) for x in sc['scene']['pose_xy']]
            v = [float(x) for x in got.get('drawer_pose_xy', [])]
            if not all(math.isfinite(x) for x in v):
                bad.append(f'drawer_pose_xy {v} 含非有限值')
            elif len(v) != len(want) or max(
                    abs(x - y) for x, y in zip(v, want)) > 1e-9:
                bad.append(f'drawer_pose_xy {v} 與階段 A 的 {want} 不同'
                           f'（{cons["drawer_pose_xy"].get("rule", "")}）')
    # ---- 資產：必須與階段 A 同一份（以路徑規範化比對；內容雜湊另記） ----
    if 'drawer_asset' in cons:
        src = cons['drawer_asset'].get('source', '')
        if asset_path is None:
            bad.append('drawer_asset 需要資產路徑才能核對，但沒有提供')
        elif not os.path.exists(asset_path):
            bad.append(f'資產檔不存在：{asset_path}')
        elif src and os.path.normpath(src) not in os.path.normpath(asset_path):
            bad.append(f'資產 {asset_path} 不是規則指定的 {src}')
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', required=True)
    ap.add_argument('--stage-config', default='')
    ap.add_argument('--start-temp-c', type=float, required=True)
    ap.add_argument('--observe-sim-s', type=float, required=True)
    ap.add_argument('--physics-dt-s', type=float, required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--joint-margin', type=float, required=True)
    ap.add_argument('--drawer-pose', required=True)
    ap.add_argument('--drawer-asset', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rule = yaml.safe_load(open(a.rule))
    cons = rule.get('可覆寫參數的約束')
    if not cons:
        print('**規則檔沒有「可覆寫參數的約束」段落** ⇒ 無法核對覆寫，不啟動',
              file=sys.stderr)
        return 80
    got = {
        'start_temp_c': a.start_temp_c,
        'observe_sim_s': a.observe_sim_s,
        'physics_dt_s': a.physics_dt_s,
        'mode': a.mode,
        'joint_margin': a.joint_margin,
        'drawer_pose_xy': [float(v) for v in a.drawer_pose.split(',')],
        'drawer_asset': a.drawer_asset,
    }
    bad = check(cons, got, a.stage_config or None, a.drawer_asset)
    rep = {'rule': os.path.relpath(a.rule), 'values': got,
           'verdict': 'pass' if not bad else 'fail', 'violations': bad,
           'asset_sha256': (hashlib.sha256(open(a.drawer_asset, 'rb').read())
                            .hexdigest() if os.path.exists(a.drawer_asset)
                            else None)}
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
    if bad:
        print('**基線參數不符事前規則，不啟動**：', file=sys.stderr)
        for b in bad:
            print(f'  - {b}', file=sys.stderr)
        return 80
    _st = cons.get('start_temp_c', {})
    _stmsg = ('起跑線**已解除**（規則檔 enforced: false）'
              if _st.get('enforced') is False
              else f'起跑線 {a.start_temp_c}°C <= {_st.get("max")}')
    print(f'[基線] 參數核對通過（規則 {os.path.basename(a.rule)}）：'
          f'{_stmsg}、'
          f'觀察 {a.observe_sim_s}s >= {cons["observe_sim_s"]["min"]}、'
          f'步長 {a.physics_dt_s}、模式 {a.mode}、'
          f'餘裕 {a.joint_margin}、擺放 {got["drawer_pose_xy"]}')
    print(f'[基線] 資產雜湊 {rep["asset_sha256"][:12]}…（門檻將以它綁定配置）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
