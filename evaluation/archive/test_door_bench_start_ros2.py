#!/usr/bin/env python3
"""ADM17r2（先等台架錨點）的 ROS 接線測試（真求解節點＋ door_bench_stub_endpoint2；無 Isaac）。重現並修正 door_coopen_core_bench_1 的啟動缺口。
  R4 重現：ADM17（r1）＋替身延後 1.5 s（牆鐘）才發 /park/gate ⇒ 求解端 stop_why＝park_mode_refused、無首解宣告、替身未釋放（逾時）。
  R5 修正：ADM17r2 ＋ 同一延後 ⇒ 等到錨點（bench.anchor_wait_s > 0）後才預熱與求解；宣告首解、替身以同 seq 釋放；求解無失敗。
  R6 逾時：ADM17r2 ＋ 替身不發 /park/gate（延後 999 s）＋ --bench-anchor-wait-s 2 ⇒ 拒跑（stop 未進迴圈；無宣告）。
  R7 一致（無延後）：ADM17r2 ⇒ 同 test_door_bench_start_ros R1 的核心項（宣告、同 seq 釋放、d0、首列 bias）。

    python3 evaluation/test_door_bench_start_ros2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('BENCH2_TMP', '/tmp/claude-1000/door_bench_ros2')
DOM0 = int(os.environ.get('BENCH2_DOMAIN', '190'))
PRM9 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')
REG = json.load(open(PRM9))['bench']
ALPHA = np.asarray(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))['alpha'], float) * np.ones(6)
FAIL = []
RUN = 'bench_ros2'


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, dom, node_file, gate_delay, wait_s=None, stub_timeout=20.0):
    os.makedirs(TMP, exist_ok=True)
    so, to = os.path.join(TMP, f'{tag}_solver.json'), os.path.join(TMP, f'{tag}_stub.json')
    for f in (so, to):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    sc = [sys.executable, '-u', os.path.join(HERE, 'door_bench_stub_endpoint2.py'), '--params', PRM9, '--run-id', RUN, '--run-s', '3',
          '--timeout-wall-s', str(stub_timeout), '--gate-delay-wall-s', str(gate_delay), '--out', to]
    stub = subprocess.Popen(sc, cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    time.sleep(1.0)
    c = [sys.executable, '-u', os.path.join(HERE, node_file), '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0', '0', '0',
         '--w-s', '0.001', '--w-a', '0.05', '--w-s-arm', '0.05', '--duration-s', '30', '--u-prev-policy', 'strict', '--assume-initial-rest',
         '--continuous', '--cmd-env', '--run-id', RUN, '--offset-observer', 'interval', '--offset-free', '--park-hold',
         '--bench-start', '--bench-params', PRM9, '--out', so] + ([] if wait_s is None else ['--bench-anchor-wait-s', str(wait_s)])
    node = subprocess.Popen(c, cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stub.wait(timeout=stub_timeout + 40)
    except subprocess.TimeoutExpired:
        stub.terminate()
    time.sleep(1.0)
    if node.poll() is None:
        node.terminate()
        try:
            node.wait(timeout=15)
        except subprocess.TimeoutExpired:
            node.kill()
    if stub.poll() is None:
        stub.kill()
    out = node.stdout.read() if node.stdout else ''
    sv = json.load(open(so)) if os.path.exists(so) else None
    st = json.load(open(to)) if os.path.exists(to) else None
    return sv, st, out


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}\n[R4 重現：ADM17 r1 ＋ 延後 1.5 s]')
    sv, st, out = scenario('r4', 0, 'dl3_wgmpc_wg2_adm17_node.py', 1.5, stub_timeout=8.0)
    ck('R4 求解端 park_mode_refused', sv is not None and 'park_mode_refused' in str((sv.get('stats') or {}).get('stop_why')), str((sv or {}).get('stats', {}).get('stop_why'))[:90])
    ck('R4 替身未釋放', st is not None and st['release']['state'] != 'released', '' if st is None else st['release']['state'])
    print('[R5 修正：ADM17r2 ＋ 延後 1.5 s]')
    sv, st, out = scenario('r5', 1, 'dl3_wgmpc_wg2_adm17r2_node.py', 1.5)
    b = (sv or {}).get('bench') or {}
    ck('R5 等到錨點（anchor_wait_s > 0）', isinstance(b.get('anchor_wait_s'), float) and b['anchor_wait_s'] > 0.0, str(b.get('anchor_wait_s')))
    ann = b.get('announced') or {}
    ck('R5 宣告首解、替身同 seq 釋放', st is not None and st['release']['state'] == 'released' and st['first_applied_seq'] == ann.get('source_seq'),
       '' if st is None else json.dumps(st['release'], default=float)[:120])
    ck('R5 求解無失敗', sv is not None and all(x.get('ok', True) for x in sv.get('log', [])), '')
    print('[R6 逾時：ADM17r2 ＋ 不發 /park/gate、等待 2 s]')
    sv, st, out = scenario('r6', 2, 'dl3_wgmpc_wg2_adm17r2_node.py', 999.0, wait_s=2.0, stub_timeout=8.0)
    ck('R6 拒跑（未收到 bench_anchor）', '未收到 bench_anchor' in out and (sv is None or not (sv.get('bench') or {}).get('announced')), out.strip().splitlines()[-1][:100] if out.strip() else '')
    print('[R7 一致：ADM17r2 無延後]')
    sv, st, out = scenario('r7', 3, 'dl3_wgmpc_wg2_adm17r2_node.py', 0.0)
    b = (sv or {}).get('bench') or {}
    ann = b.get('announced') or {}
    ck('R7 宣告且同 seq 釋放', st is not None and st['release']['state'] == 'released' and st['first_applied_seq'] == ann.get('source_seq'), '')
    ck('R7 d0＝d_init、n_upd 0', np.allclose(b.get('d0_set', [9] * 6), REG['d_init'], atol=1e-15) and b.get('n_upd_at_init') == 0, '')
    rows = [x for x in (sv or {}).get('log', []) if (x.get('solve_in') or {}).get('arm_bias') is not None]
    ck('R7 首列 arm_bias＝α⊙d_init', bool(rows) and np.allclose(rows[0]['solve_in']['arm_bias'], ALPHA * np.asarray(REG['d_init']), atol=1e-12), '')
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
