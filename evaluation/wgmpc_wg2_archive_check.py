#!/usr/bin/env python3
"""趟次封存的**內容核對**（可重跑）。

為什麼不能用「目錄內有任意 JSON」判定
------------------------------------
原 runner 用 `ls "$DIR/sim/"*.json` 判封存完成。加入錄影後，目錄裡可能
出現錄影相關或其他中途產物，**存在不等於執行結果已落盤**。
本腳本改為核對內容：`sim/wb_run.json` 能解析、必要紀錄齊備、
錄影索引與磁碟上的影格**數量一致**。

退出碼
------
    0 封存完整    70 wb_run.json 缺少或無法解析    71 必要紀錄不齊
    72 錄影不完整    73 節點輸出不齊
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REQUIRED_SIM = ['schema', 'mode', 'stop_reason', 'sim_time_s', 'wall_s',
                'log', 'log_cols', 'cmd_chain', 'cpu_temp_max_c',
                'cpu_limit_c', 'command_interface', 'frame_wiring']
REQUIRED_NODE = ['args', 'started', 'log']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--expect-recording', action='store_true')
    ap.add_argument('--expect-arm-model', default=None)
    ap.add_argument('--node-out', default='wg2_out.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = a.run_dir
    rep = {'run_dir': D, 'checks': []}
    bad = []

    def ck(name, ok, detail='', code=None):
        print(f'  {name:48s} {"ok" if ok else "**錯**"}  {detail}', flush=True)
        rep['checks'].append({'name': name, 'ok': bool(ok),
                              'detail': str(detail)})
        if not ok:
            bad.append(code or 71)

    print('=== 封存內容核對（不以任意 JSON 存在判定）===')
    p = os.path.join(D, 'sim', 'wb_run.json')
    if not os.path.exists(p):
        ck('sim/wb_run.json 存在', False, '**缺少**', 70)
        return _fin(rep, bad, a)
    try:
        sim = json.load(open(p))
    except (ValueError, OSError) as e:
        ck('sim/wb_run.json 可解析', False, f'**{e}**', 70)
        return _fin(rep, bad, a)
    ck('sim/wb_run.json 可解析', True,
       f'{os.path.getsize(p)/1e6:.2f} MB')

    miss = [k for k in REQUIRED_SIM if k not in sim]
    ck('執行端必要紀錄齊備', not miss, f'缺 {miss}' if miss else
       f'{len(REQUIRED_SIM)} 項')
    ck('執行端有步進紀錄', bool(sim.get('log')),
       f"{len(sim.get('log') or [])} 列 × {len(sim.get('log_cols') or [])} 欄")
    ck('停止原因已記錄', bool(sim.get('stop_reason')),
       str(sim.get('stop_reason')))
    ck('CPU 溫度已記錄且未超限',
       (sim.get('cpu_temp_max_c') is not None
        and sim.get('cpu_limit_c') is not None
        and sim['cpu_temp_max_c'] <= sim['cpu_limit_c']),
       f"峰 {sim.get('cpu_temp_max_c')} / 限 {sim.get('cpu_limit_c')}")

    # ---- 錄影：索引與磁碟影格**數量一致** ----
    rf = sim.get('record_frames')
    if a.expect_recording:
        if not rf:
            ck('錄影紀錄存在', False, '**本趟要求錄影但 record_frames 為空**', 72)
        else:
            n_idx = int(rf.get('n') or 0)
            rdir = rf.get('dir') or ''
            if not os.path.isabs(rdir):
                rdir = os.path.join(os.getcwd(), rdir)
            pngs = sorted(glob.glob(os.path.join(rdir, 'f*.png')))
            ck('錄影影格數 > 0', n_idx > 0, f'索引 {n_idx} 格', 72)
            ck('索引與磁碟影格數一致', n_idx == len(pngs),
               f'索引 {n_idx}／磁碟 {len(pngs)}', 72)
            idx = rf.get('index') or []
            mono = all(idx[i][1] <= idx[i + 1][1]
                       for i in range(len(idx) - 1)) if len(idx) > 1 else True
            ck('索引的模擬時間單調不減', mono, f'{len(idx)} 筆', 72)
            nz = [q for q in pngs if os.path.getsize(q) > 0]
            ck('所有影格檔非空', len(nz) == len(pngs),
               f'非空 {len(nz)}／共 {len(pngs)}', 72)
            if pngs:
                tot = sum(os.path.getsize(z) for z in pngs)
                span = ((idx[-1][1] - idx[0][1]) if len(idx) > 1 else 0.0)
                eff = (round(n_idx / span, 2) if span > 0 else None)
                rep['recording'] = {
                    'n': n_idx, 'dir': rdir,
                    'total_mb': round(tot / 1e6, 1),
                    'sim_span_s': round(span, 3), 'eff_fps': eff}
                print(f'    影片：{n_idx} 格、{tot/1e6:.1f} MB、'
                      f'涵蓋模擬 {span:.2f} s'
                      + (f'（有效 {eff} fps，以模擬時間計）' if eff else ''),
                      flush=True)
    else:
        ck('本趟未要求錄影', rf is None or not rf, '')

    # ---- 節點輸出 ----
    q = os.path.join(D, a.node_out)
    if not os.path.exists(q):
        ck(f'{a.node_out} 存在', False, '**缺少**', 73)
    else:
        try:
            nd = json.load(open(q))
        except (ValueError, OSError) as e:
            ck(f'{a.node_out} 可解析', False, f'**{e}**', 73)
            nd = None
        if nd is not None:
            ck(f'{a.node_out} 可解析', True,
               f'{os.path.getsize(q)/1e6:.2f} MB')
            # **legacy 判別**：`started`／`stats`／`args.arm_model` 是本輪
            # 才加的欄位。舊趟次（如 free4）沒有它們**不是封存缺陷**，
            # 但新趟次必須有 —— 以 `--expect-arm-model` 區分這兩種用法。
            legacy = 'started' not in nd
            rep['node_format'] = 'legacy' if legacy else 'current'
            if legacy and not a.expect_arm_model:
                print('    節點輸出為 **legacy 格式**（本輪新增欄位之前的趟次）'
                      '⇒ 以下新欄位不列為缺陷', flush=True)
                ck('節點有步進紀錄（legacy 路徑）', bool(nd.get('log')),
                   f"{len(nd.get('log') or [])} 輪")
                ck('legacy 趟次有發布週期',
                   any(x.get('published') for x in (nd.get('log') or [])),
                   f"{sum(1 for x in (nd.get('log') or []) if x.get('published'))} 輪")
                return _fin(rep, bad, a)
            mn = [k for k in REQUIRED_NODE if k not in nd]
            ck('節點必要紀錄齊備', not mn, f'缺 {mn}' if mn else '')
            am = (nd.get('args') or {}).get('arm_model')
            rep['arm_model'] = am
            if a.expect_arm_model:
                ck(f'實際模型選擇 == {a.expect_arm_model}',
                   am == a.expect_arm_model, f'arm_model={am}')
            ck('節點確實啟動（非拒絕啟動）', nd.get('started') is True,
               f"started={nd.get('started')}　"
               f"{str(nd.get('refuse_reason') or '')[:60]}")
            st = nd.get('stats') or {}
            rep['node_stats'] = {k: st.get(k) for k in
                                 ('published', 'dropped_stale', 'no_solution',
                                  'reached_held', 'stop_why',
                                  'task_sim_span_s', 'n_step_mismatch',
                                  'stopped_on_chain_fail')}
            ck('有發布週期', (st.get('published') or 0) > 0,
               f"published={st.get('published')}")

    return _fin(rep, bad, a)


def _fin(rep, bad, a):
    code = 0 if not bad else max(bad)
    rep['exit_code'] = code
    rep['archive_complete'] = (code == 0)
    print()
    print('**封存完整**' if code == 0 else f'**封存不完整（exit {code}）**')
    print('這是封存內容核對，**不是**到達與保持的判定。')
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return code


if __name__ == '__main__':
    sys.exit(main())
