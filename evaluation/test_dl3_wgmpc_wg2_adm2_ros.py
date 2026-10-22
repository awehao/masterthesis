#!/usr/bin/env python3
"""ADM2（--offset-observer interval）的 **ROS 假世界**接線測試（無 Isaac、無 GPU）。
替身執行端 wgmpc_sp_stub_endpoint.py 餵 /clock、/joint_states、/odom、/coman/arm_setpoint、/coman/applied_cmd（每物理步）；
跑真正的 dl3_wgmpc_wg2_adm2_node.py。**替身沒有物理**（其模型與求解器的索引約定不同、且無偏移），本檔只核接線與紀錄，
不判估計值是否正確、不判到達或保持。

    python3 evaluation/test_dl3_wgmpc_wg2_adm2_ros.py      # 需 ROS 2 環境；只清理本趟程序
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('ADM2R_TMP', '/tmp/adm2_ros')
DOM0 = int(os.environ.get('ADM2R_DOMAIN', '196'))
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:60s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, stub_args, node_args, run_s=14.0, dom=0):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    if os.path.exists(out):
        os.remove(out)                                            # 本測試自己的暫存輸出
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s)] + stub_args,
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        node = subprocess.run([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm2_node.py'),
                               '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                               '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                               '--offset-free', '--out', out] + node_args,
                              cwd=WS, env=env, capture_output=True, text=True, timeout=run_s + 60)
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()
    if not os.path.exists(out):
        print(node.stdout[-1500:])
        print(node.stderr[-1500:])
        return None
    return json.load(open(out))


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    print('R1  預設（--offset-observer 未給）⇒ observer none、紀錄無 offset_interval')
    w = scenario('r1', ['--ready-after', '1.0'], [], dom=0)
    if w is None:
        ck('R1 有產出', False)
    else:
        of = w.get('offset_free') or {}
        ck('observer＝none', of.get('observer') == 'none', str(of.get('observer')))
        ck('interval 報告為 None', of.get('interval') is None)
        ck('逐輪紀錄沒有 offset_interval', not any('offset_interval' in x for x in w.get('log') or []))
        ck('有求解發布（介面正常）', any(x.get('published') for x in w.get('log') or []))

    print('\nR2  interval ⇒ 逐區間紀錄、更新、m 合理')
    w = scenario('r2', ['--ready-after', '1.0'], ['--offset-observer', 'interval'], dom=1)
    if w is None:
        ck('R2 有產出', False)
    else:
        of = w.get('offset_free') or {}
        iv = of.get('interval') or {}
        recs = [x['offset_interval'] for x in w.get('log') or [] if x.get('offset_interval')]
        upd = [r for r in recs if r.get('used') == 'updated']
        ck('observer＝interval', of.get('observer') == 'interval')
        ck('有更新', (iv.get('n_updated') or 0) > 0 and len(upd) > 0, f"n_updated={iv.get('n_updated')} held={iv.get('held_reasons')}")
        ck('報告計數＝紀錄計數', iv.get('n_updated') == len(upd), f'{iv.get("n_updated")} vs {len(upd)}')
        ck('區間步數在 [1, m_max]', all(1 <= (r.get('m') or 0) <= iv.get('m_max', 10) for r in upd),
           str(sorted({r.get('m') for r in upd})))
        ck('t0 < t1、step0 < step1', all(r['t0'] < r['t1'] and r['step0'] < r['step1'] for r in upd))
        ck('每區間至多提交一次', len({(r['step0'], r['step1']) for r in upd}) == len(upd))
        ck('紀錄 d_hat_after＝報告末值', not upd or [round(x, 6) for x in upd[-1]['d_hat_after']] == of.get('d_hat_rad'),
           f"{upd[-1]['d_hat_after'] if upd else None} vs {of.get('d_hat_rad')}")
        ck('緩衝無衝突', ((iv.get('buffer') or {}).get('n_conflict') or 0) == 0, str(iv.get('buffer')))

    print('\nR3  interval＋失效閂鎖（exec_mode 3）⇒ 失效之後不再提交')
    w = scenario('r3', ['--ready-after', '1.0', '--fail-at', '6.0'], ['--offset-observer', 'interval'], dom=2)
    if w is None:
        ck('R3 有產出', False)
    else:
        recs = [(x.get('sim_t'), x['offset_interval']) for x in w.get('log') or [] if x.get('offset_interval')]
        after = [r for t, r in recs if t is not None and t >= 6.0 and r.get('used') == 'updated']
        ck('失效（6.0 s）之後無更新', len(after) == 0, f'{len(after)} 筆')
        ck('失效前有更新', any(r.get('used') == 'updated' for t, r in recs if t is not None and t < 6.0))

    print('\nR4  interval＋api_applied＝False ⇒ 保持（api_not_applied 或被擋）')
    w = scenario('r4', ['--ready-after', '1.0', '--api-false-at', '6.0'], ['--offset-observer', 'interval'], dom=3)
    if w is None:
        ck('R4 有產出', False)
    else:
        recs = [(x.get('sim_t'), x['offset_interval']) for x in w.get('log') or [] if x.get('offset_interval')]
        after = [r for t, r in recs if t is not None and t >= 6.06]
        ck('之後無更新', not any(r.get('used') == 'updated' for r in after), f'{[r.get("why") for r in after][:3]}')
        ck('停發前有更新', any(r.get('used') == 'updated' for t, r in recs if t is not None and t < 6.0))

    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
