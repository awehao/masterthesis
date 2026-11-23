#!/usr/bin/env python3
"""台架啟動 r4 的 ROS 接線測試（test_door_bench_start_ros3.py 的版本化複本；Codex r27：只換求解節點 ADM17r4＋--offset-hold-on-free --offset-update-in-open、登錄 r10；判準同 r3）。
以下為 r3 說明。
台架啟動 r3 的 ROS 接線測試（Codex r24）：**真求解節點 ADM17r3 ＋ 真轉接器 arm_vel_adapter_k3m2 ＋ 替身執行端 r3**（替身只接轉接器段封裝；無 Isaac）。
正例須在替身（執行端角色）看到求解端宣告的同一 source_seq，並以真 EnvIntake／E2／DualSourceExecutor 通過原首步核對。
  C1 延後訂閱：轉接器晚求解端 3 s 啟動 ⇒ 求解端等待轉接器就緒（adapter_ready.wait_s ≥ 2）後才首解；替身同 seq 釋放、first_intake／first_step ok。
  C2 延後 odom：替身前 5 s（牆鐘）不發 /odom ⇒ 轉接器 yaw 缺、未就緒；odom 出現後就緒並同 seq 釋放；釋放時刻 ≥ 5 s。
  C3 yaw 過期／恢復：/odom 只在 [0, 1) s 發、[1, 7) s 停、之後恢復（求解端約 1.5 s 啟動）⇒ 7 s 前不得釋放；之後同 seq 釋放、首步核對 ok。
  C4 缺就緒逾時：不啟動轉接器、--bench-adapter-ready-wait-s 4 ⇒ 求解端拒跑（轉接器未就緒）、無首解宣告、替身未釋放。
  C5 轉接器旗標關閉：k3m2 不帶 --bench-ready-run-id 時不建立就緒回報（靜態：k3m2 對 k3m 只有新增，且就緒計時器只在旗標時建立）。

    python3 evaluation/test_door_bench_start_ros3.py
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('BENCH4_TMP', '/tmp/claude-1000/door_bench_ros4')
DOM0 = int(os.environ.get('BENCH4_DOMAIN', '205'))
PRM10 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r10.json')
FAIL = []
RUN = 'bench_ros4'


def ck(n, ok, d=''):
    print(f'  {n:66s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, dom, adapter_delay=0.0, adapter=True, odom_off='', wait_s=None, stub_timeout=40.0):
    os.makedirs(TMP, exist_ok=True)
    so, to = os.path.join(TMP, f'{tag}_solver.json'), os.path.join(TMP, f'{tag}_stub.json')
    for f in (so, to):
        if os.path.exists(f):
            os.remove(f)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    procs = []
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_bench_stub_endpoint3.py'), '--params', PRM10, '--run-id', RUN,
                             '--odom-off', odom_off, '--timeout-wall-s', str(stub_timeout), '--out', to],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    time.sleep(1.5)
    c = [sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r4_node.py'), '--arm-model', 'setpoint', '--N', '5', '--rate', '20',
         '--target-offset', '0', '0', '0', '--w-s', '0.001', '--w-a', '0.05', '--w-s-arm', '0.05', '--duration-s', '60', '--u-prev-policy', 'strict',
         '--assume-initial-rest', '--continuous', '--cmd-env', '--run-id', RUN, '--offset-observer', 'interval', '--offset-free', '--park-hold',
         '--cmd-topic', '/wgmpc/cmd_world', '--bench-start', '--bench-params', PRM10, '--mode-switch-run-id', RUN, '--offset-hold-on-free', '--offset-update-in-open', '--out', so] + ([] if wait_s is None else ['--bench-adapter-ready-wait-s', str(wait_s)])
    node = subprocess.Popen(c, cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    procs.append(node)
    if adapter:
        time.sleep(adapter_delay)
        ad = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'arm_vel_adapter_k3m2.py'), '--in-topic', '/wgmpc/cmd_world', '--consumer-node',
                               '/door_k2_sim', '--cmd-env', '--env-in-topic', '/wgmpc/cmd_env', '--bench-ready-run-id', RUN],
                              cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        procs.append(ad)
    try:
        stub.wait(timeout=stub_timeout + 30)
    except subprocess.TimeoutExpired:
        stub.terminate()
    time.sleep(1.0)
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
    out = node.stdout.read() if node.stdout else ''
    sv = json.load(open(so)) if os.path.exists(so) else None
    st = json.load(open(to)) if os.path.exists(to) else None
    return sv, st, out


def positive(tag, sv, st, min_release=None):
    ann = ((sv or {}).get('bench') or {}).get('announced') or {}
    ar = ((sv or {}).get('bench') or {}).get('adapter_ready') or {}
    ck(f'{tag} 轉接器就緒後才首解', ar.get('ok') is True, f"wait {ar.get('wait_s')} s")
    ck(f'{tag} 替身以宣告的同一 seq 釋放', st is not None and st['release']['state'] == 'released' and (st['release'].get('released') or {}).get('source_seq') == ann.get('source_seq'),
       '' if st is None else json.dumps(st['release'], default=float)[:110])
    ck(f'{tag} first_intake ok（真 EnvIntake／E2）', ((st or {}).get('first_intake') or {}).get('ok') is True, str((st or {}).get('first_intake'))[:90])
    ck(f'{tag} first_step ok（原首步核對）', ((st or {}).get('first_step') or {}).get('ok') is True, str(((st or {}).get('first_step') or {}).get('why')))
    if min_release is not None:
        ck(f'{tag} 釋放時刻 ≥ {min_release} s', st is not None and float(st.get('release_wall_s', 0)) >= min_release, f"{(st or {}).get('release_wall_s')}")


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}\n[C1 延後訂閱：轉接器晚 3 s]')
    sv, st, _ = scenario('c1', 0, adapter_delay=3.0)
    positive('C1', sv, st)
    ck('C1 等待 ≥ 2 s', (((sv or {}).get('bench') or {}).get('adapter_ready') or {}).get('wait_s', 0) >= 2.0, '')
    print('[C2 延後 odom：前 5 s 不發]')
    sv, st, _ = scenario('c2', 1, odom_off='0:5')
    positive('C2', sv, st, min_release=5.0)
    print('[C3 yaw 過期／恢復：odom 1–7 s 停]')
    sv, st, _ = scenario('c3', 2, odom_off='1:7')
    positive('C3', sv, st, min_release=7.0)
    print('[C4 缺就緒逾時：不啟動轉接器、等待 4 s]')
    sv, st, out = scenario('c4', 3, adapter=False, wait_s=4.0, stub_timeout=12.0)
    ck('C4 求解端拒跑（轉接器未就緒）', '轉接器未就緒' in out, out.strip().splitlines()[-1][:100] if out.strip() else '')
    ck('C4 無首解宣告、替身未釋放', st is not None and st['release']['state'] != 'released' and not st['release'].get('announced'), '' if st is None else st['release']['state'])
    print('[C5 轉接器旗標關閉（靜態）]')
    A = open(os.path.join(HERE, 'arm_vel_adapter_k3m.py')).read().splitlines()
    Bv = open(os.path.join(HERE, 'arm_vel_adapter_k3m2.py')).read().splitlines()
    rm = [ln[1:] for ln in difflib.unified_diff(A, Bv, lineterm='', n=0) if ln.startswith('-') and not ln.startswith('---')]
    rm = [ln for ln in rm if ln.strip() and not ln.lstrip().startswith('#')]
    src = '\n'.join(Bv)
    ck('C5 k3m2 對 k3m 刪改限於 docstring 首行、匯入、建構子簽名、parse／Adapter 呼叫',
       all(ln.startswith(('"""Thin adapter', 'from std_msgs.msg import', '                 in_topic=', '    a, _ = ap.parse_known_args()',
                          '                 in_topic=a.in_topic')) for ln in rm), str(rm)[:160])
    ck('C5 就緒計時器只在旗標且封裝模式時建立', 'if self.bench_run and self.env_on:' in src, '')
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
