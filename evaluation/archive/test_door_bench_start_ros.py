#!/usr/bin/env python3
"""台架初態啟動的 ROS 接線測試（真求解節點 ADM17 ＋ door_bench_stub_endpoint 替身執行端；無 Isaac、無物理）。
  R1 一致（受控體 d_true＝d_init）：求解端在凍結的 t0 求出首解並宣告；替身以同一 source_seq 釋放；握手零命令（meta 未到時）不被當首解；預熱已做未發布；
     求解紀錄 bench.d0_set＝d_init、n_upd_at_init 0；首列 solve_in.arm_bias＝α⊙d_init；釋放後 1 s 內 |q − q_nominal| 最大 < 0.003 rad（記錄實值）。
  R2 壓力（d_true＝d_init ＋ 登錄不確定度）：同樣釋放、求解無失敗；記錄 |q − q_nominal| 最大（受控體偏差不等於登錄初值的反例，只報量級）。
  R3 旗標關閉：ADM17 不帶 --bench-*（亦不帶 --park-hold；原替身不發停車閘門）、配原替身 wgmpc_sp_stub_endpoint ⇒ 求解紀錄無 bench、偏移觀測由 0 起（首列 arm_bias 為辨識值 α⊙0 以外不檢）、有發布。

    python3 evaluation/test_door_bench_start_ros.py
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
TMP = os.environ.get('BENCH_TMP', '/tmp/claude-1000/door_bench_ros')
DOM0 = int(os.environ.get('BENCH_DOMAIN', '185'))
PRM9 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')
REG = json.load(open(PRM9))['bench']
ALPHA = np.asarray(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))['alpha'], float) * np.ones(6)
FAIL = []
RUN = 'bench_ros'


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def solver_cmd(out, bench):
    c = [sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm17_node.py'), '--arm-model', 'setpoint', '--N', '5', '--rate', '20',
         '--target-offset', '0', '0', '0', '--w-s', '0.001', '--w-a', '0.05', '--w-s-arm', '0.05', '--duration-s', '25',
         '--u-prev-policy', 'strict', '--assume-initial-rest', '--continuous', '--cmd-env', '--run-id', RUN,
         '--offset-observer', 'interval', '--offset-free', '--out', out]
    if bench:
        c += ['--park-hold', '--bench-start', '--bench-params', PRM9]
    return c


def scenario(tag, dom, bench=True, d_true=None, run_s=4.0):
    os.makedirs(TMP, exist_ok=True)
    so, to = os.path.join(TMP, f'{tag}_solver.json'), os.path.join(TMP, f'{tag}_stub.json')
    for f in (so, to):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    if bench:
        sc = [sys.executable, '-u', os.path.join(HERE, 'door_bench_stub_endpoint.py'), '--params', PRM9, '--run-id', RUN,
              '--run-s', str(run_s), '--out', to] + ([] if d_true is None else ['--d-true', ','.join(f'{v:.9f}' for v in d_true)])
    else:
        sc = [sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s + 4.0), '--ready-after', '0.5']
    stub = subprocess.Popen(sc, cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    time.sleep(1.0)
    node = subprocess.Popen(solver_cmd(so, bench), cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stub.wait(timeout=run_s + 90)
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
    sv = json.load(open(so)) if os.path.exists(so) else None
    st = json.load(open(to)) if os.path.exists(to) else None
    if sv is None:
        print((node.stdout.read() if node.stdout else '')[-2500:])
    return sv, st


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}\n[R1 一致]')
    sv, st = scenario('r1', 0)
    if sv is None or st is None:
        ck('R1 有產出', False)
    else:
        ann = (sv.get('bench') or {}).get('announced') or {}
        ck('R1 求解端宣告首解（kind solution、sim_t＝t0）', ann.get('kind') == 'solution' and abs(float(ann.get('sim_t', 9)) - 0.0) < 1e-6, str(ann)[:120])
        ck('R1 替身以宣告的 source_seq 釋放', st['release']['state'] == 'released' and st['first_applied_seq'] == ann.get('source_seq'),
           json.dumps(st['release'], default=float)[:160])
        _ni = int((sv.get('handshake_startup') or {}).get('n_init_cmd', -1))
        ck('R1 握手零命令不被當首解（宣告 seq > 握手筆數、釋放 seq＝宣告）', _ni >= 0 and int(ann.get('source_seq', 0)) > _ni
           and st['first_applied_seq'] == ann.get('source_seq'), f'握手 {_ni} 筆')
        ck('R1 預熱求解已做、未發布', ((sv.get('bench') or {}).get('warmup') or {}).get('published') is False, str((sv.get('bench') or {}).get('warmup')))
        b = sv.get('bench') or {}
        ck('R1 d0_set＝d_init、n_upd_at_init 0', np.allclose(b.get('d0_set', [9] * 6), REG['d_init'], atol=1e-15) and b.get('n_upd_at_init') == 0, '')
        rows = [x for x in sv.get('log', []) if (x.get('solve_in') or {}).get('arm_bias') is not None]
        ck('R1 首列 arm_bias＝α⊙d_init', bool(rows) and np.allclose(rows[0]['solve_in']['arm_bias'], ALPHA * np.asarray(REG['d_init']), atol=1e-12),
           '' if not rows else str(np.round(rows[0]['solve_in']['arm_bias'], 6)))
        ck('R1 釋放後 1 s 內 |q − q_nominal| < 0.003 rad', st['dev_max_1s_rad'] < 0.003, f"{st['dev_max_1s_rad']:.5f} rad")
        ck('R1 求解無失敗', all(x.get('ok', True) for x in sv.get('log', [])), str(sum(1 for x in sv.get('log', []) if not x.get('ok', True))))
    print('[R2 壓力：d_true＝d_init＋不確定度]')
    sv2, st2 = scenario('r2', 1, d_true=np.asarray(REG['d_init']) + np.asarray(REG['uncertainty']))
    if sv2 is None or st2 is None:
        ck('R2 有產出', False)
    else:
        ck('R2 釋放', st2['release']['state'] == 'released', '')
        ck('R2 求解無失敗', all(x.get('ok', True) for x in sv2.get('log', [])), '')
        print(f"    R2 釋放後 1 s 內 |q − q_nominal| 最大 {st2['dev_max_1s_rad']:.5f} rad（記錄；受控體偏差≠登錄初值）")
    print('[R3 旗標關閉回歸]')
    sv3, _ = scenario('r3', 2, bench=False)
    if sv3 is None:
        ck('R3 有產出', False)
    else:
        ck('R3 無 bench 紀錄', 'bench' not in sv3, '')
        ck('R3 有發布', int((sv3.get('stats') or {}).get('n_published', sum(1 for x in sv3.get('log', []) if x.get('published')))) > 0, '')
        rows = [x for x in sv3.get('log', []) if (x.get('offset_interval') or {}).get('d_before') is not None]
        ck('R3 偏移觀測由 0 起', bool(rows) and np.allclose(rows[0]['offset_interval']['d_before'], 0.0, atol=0), '')
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
