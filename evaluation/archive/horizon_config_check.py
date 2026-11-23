#!/usr/bin/env python3
"""H1／H5 時域消融的設定核對：一組趟次之間，除了 N 之外是否完全相同。

N 一律以求解節點**落盤**的 `align_solver.json → args.N` 為準，不信 shell 傳參。
比對三個來源：求解節點 args、任務節點 args、模擬器 room_run.json 的 config
（手指碰撞、抽屜資產），另加辨識檔的 sha256。

    python3 evaluation/horizon_config_check.py runs/A runs/B [runs/C …] [--allow N]
    同 N 的兩趟核「控制設定一致」：加 `--allow`（不給值）＝ 不允許任何差異。

離開碼：0 = 只有允許的差異；1 = 有其他差異（列出每一項）；2 = 讀不到必要檔案。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# 每趟本來就不同、與設定無關的欄位（輸出路徑、趟次 ID 等）
VOLATILE = {
    'solver': {'out', 'run_id'},
    'task': {'out'},
    'sim': {'out'},
}


def sha16(path):
    try:
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
    except OSError:
        return None


def resolved(run_dir):
    """回傳 {'solver.x': v, 'task.y': v, 'sim.z': v, 'hash.*': v}。"""
    out = {}
    sp = os.path.join(run_dir, 'align_solver.json')
    tp = os.path.join(run_dir, 'task.json')
    rp = os.path.join(run_dir, 'room_run.json')
    for p in (sp, tp):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    sa = json.load(open(sp))['args']
    for k, v in sa.items():
        if k not in VOLATILE['solver']:
            out[f'solver.{k}'] = v
    ta = json.load(open(tp))['args']
    for k, v in ta.items():
        if k not in VOLATILE['task']:
            out[f'task.{k}'] = v
    if os.path.exists(rp):
        cfg = json.load(open(rp)).get('config', {})
        for k in ('finger_collision', 'drawer_asset', 'drawer_pose',
                  'start_pose', 'physics_dt', 'world'):
            if k in cfg:
                v = cfg[k]
                if k == 'finger_collision' and isinstance(v, dict):
                    v = v.get('mode')
                out[f'sim.{k}'] = v
        # 優先用模擬器載入當下記錄的雜湊；舊趟次沒記，只能以核對當下的檔案代替，
        # 並另列來源（兩者來源不同時會顯示為差異，要能解釋）
        if isinstance(cfg.get('drawer_asset_sha256'), str):
            out['hash.drawer_asset'] = cfg['drawer_asset_sha256'][:16]
            out['hash.drawer_asset_source'] = 'recorded_at_load'
        elif isinstance(cfg.get('drawer_asset'), str):
            out['hash.drawer_asset'] = sha16(cfg['drawer_asset'])
            out['hash.drawer_asset_source'] = 'file_at_check_time'
    else:
        out['sim.room_run'] = 'MISSING'
    wp = os.path.join(run_dir, 'wholebody.json')
    wa = json.load(open(wp)).get('args') if os.path.exists(wp) else None
    if isinstance(wa, dict):
        for k, v in wa.items():
            if k != 'out':
                out[f'wholebody.{k}'] = v
    else:
        # 展開節點目前不落盤參數 ⇒ 它的 --motm 等設定**無法由檔案核對**
        out['wholebody.args'] = 'NOT_RECORDED'
    if isinstance(sa.get('arm_ident'), str):
        out['hash.arm_ident'] = sha16(sa['arm_ident'])
    if isinstance(sa.get('urdf'), str):
        out['hash.urdf'] = sha16(sa['urdf'])
    return out


def compare(runs, allow):
    cfgs = {r: resolved(r) for r in runs}
    keys = sorted(set().union(*[set(c) for c in cfgs.values()]))
    diffs = {}
    for k in keys:
        vals = {r: cfgs[r].get(k, '<absent>') for r in runs}
        if len({json.dumps(v, sort_keys=True, default=str)
                for v in vals.values()}) > 1:
            diffs[k] = vals
    allowed = {k: v for k, v in diffs.items() if k in allow}
    other = {k: v for k, v in diffs.items() if k not in allow}
    return cfgs, allowed, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--allow', nargs='*', default=['solver.N'],
                    help='允許不同的欄位（預設只有 solver.N）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    try:
        cfgs, allowed, other = compare(a.runs, set(a.allow))
    except FileNotFoundError as e:
        print(f'**讀不到必要檔案**：{e}')
        return 2
    names = [os.path.basename(r.rstrip('/')) for r in a.runs]
    print('N（落盤值）：' + '、'.join(
        f'{n}={cfgs[r].get("solver.N")}' for n, r in zip(names, a.runs)))
    print(f'允許的差異：{len(allowed)} 項 {sorted(allowed)}')
    if other:
        print(f'**其他差異 {len(other)} 項**（非 N 的差異都要能解釋）：')
        for k, vals in other.items():
            print(f'  {k}: ' + '；'.join(
                f'{n}={json.dumps(vals[r], ensure_ascii=False, default=str)[:80]}'
                for n, r in zip(names, a.runs)))
    else:
        print('其他差異：0 項')
    if any(c.get('wholebody.args') == 'NOT_RECORDED' for c in cfgs.values()):
        print('**注意**：展開節點（drawer_wholebody_node）未落盤參數，其設定無法由檔案核對')
    if a.out:
        json.dump({'runs': names,
                   'N': {n: cfgs[r].get('solver.N') for n, r in zip(names, a.runs)},
                   'allowed': allowed, 'other': other},
                  open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    return 1 if other else 0


if __name__ == '__main__':
    sys.exit(main())
