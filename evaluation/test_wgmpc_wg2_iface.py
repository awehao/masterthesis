"""WG2 介面測試（**不開 Isaac、不用 GPU**）。

驗四件事：
  A 座標：本節點發到 cmd_in 的是**世界**速度，且 adapter 轉回**本體**後
    與核心的原始輸出一致 —— **不得有雙重轉換**。
  B 單一 executor 所有權：節點只加入一個 executor，不用 TF spin_thread。
  C 新鮮度：狀態停止發布後，過期解**不發布**。
  D 命令三階段（請求／修改後／實際套用）可分辨，且 u_prev 取實際套用者。

用最小假世界（持續發布 /clock、/joint_states、/odom）＋ 實際的
安全節點與 adapter；**不含 Isaac、不含抽屜、不含接觸或視覺**。
"""
from __future__ import annotations
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wgmpc_core import body_to_world            # noqa
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa

SC = os.environ.get('WG2_TMP', f'/tmp/wg2_{os.getpid()}')
DOMAIN = os.environ.get('WG2_DOMAIN', '161')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
_bad = 0


def ck(name, cond, extra=''):
    global _bad
    print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
    _bad += not cond


def run_case(name, stop_js_at=-1.0, duration=12.0, with_safety=True):
    os.makedirs(SC, exist_ok=True)
    out = os.path.join(SC, f'{name}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    procs = []
    try:
        w = subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', str(duration + 20), '--stop-js-at', str(stop_js_at)],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(w)
        time.sleep(3.0)
        if with_safety:
            # **最小消費端樁**：只為了讓 adapter 的 joints 順序核對能通過
            # （F4 的守衛）。它不積分、不模擬物理 ⇒ 不得取代 Isaac 驗證。
            sk = subprocess.Popen(
                [sys.executable, os.path.join(HERE, 'wgmpc_wg2_sink.py'),
                 '--node-name', 'wgmpc_wg2_sink'],
                cwd=WS, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(sk)
            time.sleep(2.0)
            s = subprocess.Popen(
                ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
                 '--ros-args', '-p', 'use_sim_time:=true',
                 '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
                 '-p', f'wholebody_urdf:={URDF}',
                 '-p', 'vmax_base_lin:=0.035255',
                 '-p', 'vmax_base_ang:=0.199900', '-p', 'vmax_arm:=0.999900'],
                cwd=WS, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(s)
            ad = subprocess.Popen(
                [sys.executable, os.path.join(HERE, 'arm_vel_adapter.py'),
                 '--consumer-node', '/wgmpc_wg2_sink'],
                cwd=WS, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(ad)
            time.sleep(4.0)
        n = subprocess.Popen(
            [sys.executable, '-u', os.path.join(HERE, 'wgmpc_wg2_node.py'),
             '--N', '5', '--rate', '20',
             '--target', '10.80', '8.40', '0.60',
             '--duration-s', str(duration), '--out', out],
            cwd=WS, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        procs.append(n)
        log = n.communicate(timeout=duration + 60)[0]
    finally:
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.5)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    d = json.load(open(out)) if os.path.exists(out) else None
    return d, (log or '')


def main() -> int:
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')

    # ---------- B 單一 executor 所有權（靜態核對）----------
    print('B 單一 executor 所有權')
    src = open(os.path.join(HERE, 'wgmpc_wg2_node.py'), encoding='utf-8').read()
    code = '\n'.join(l for l in src.split('\n')
                     if not l.lstrip().startswith(('#', '*'))
                     and '`' not in l)
    ck('不用 TransformListener(spin_thread=True)',
       'spin_thread' not in code)
    ck('不用 MultiThreadedExecutor／ReentrantCallbackGroup',
       'MultiThreadedExecutor' not in code
       and 'ReentrantCallbackGroup' not in code)
    ck('add_node 只出現一次', code.count('add_node(') == 1)
    ck('狀態快照為 frozen dataclass（不可變）',
       '@dataclass(frozen=True)' in src)
    ck('回呼只組新快照並原子指派（無就地改）',
       'self._snap = Snap(' in src and '_snap[' not in src)

    # ---------- A 座標：無雙重轉換（離線核對算術）----------
    print('\nA 座標（本體 → 世界 → adapter 轉回本體）')
    w2b_src = open(os.path.join(HERE, 'arm_vel_adapter.py'),
                   encoding='utf-8').read()
    i, j = w2b_src.index('def world_to_body9'), w2b_src.index('class ')
    ns = {'math': math}
    exec(w2b_src[i:j], ns)
    w2b = ns['world_to_body9']
    mx = 0.0
    for th in (0.0, 0.7, math.pi / 2, -1.9):
        u_body = np.array([0.03, -0.02, 0.15, 0.4, -0.3, 0.2, -0.1, 0.5, -0.4])
        u_world = body_to_world(th) @ u_body
        back = np.asarray(w2b(list(u_world), th), float)
        mx = max(mx, float(np.abs(back - u_body).max()))
    ck('本體→世界→(adapter)→本體 可往返', mx < 1e-12, f'  max|Δ| {mx:.2e}')
    ck('節點在發布前做 body→world',
       'body_to_world(float(q0[2])) @ r.u0' in src)
    ck('**反例**：少做 body→world 會與往返結果不同',
       float(np.abs(np.asarray(
           w2b(list(np.array([0.03, -0.02, 0.15, 0, 0, 0, 0, 0, 0])), 0.7),
           float)[:2] - np.array([0.03, -0.02])).max()) > 1e-6)

    # ---------- D 三階段 ＋ 正常閉迴路 ----------
    print('\nD 完整鏈：請求／修改後／實際套用（含安全層與 adapter）')
    d, log = run_case('chain', duration=12.0)
    ck('節點有產出紀錄', d is not None)
    if d is None:
        print(log[-2000:])
        return 1
    lg = d['log']
    pub = [r for r in lg if r.get('published')]
    ck('有發布的週期', len(pub) >= 10, f'  {len(pub)} / {len(lg)} 輪')
    ck('每筆發布都同時記了 request_world 與 request_body',
       all('request_world' in r and 'request_body' in r for r in pub))
    srcs = {r['u_prev_src'] for r in lg}
    ck('u_prev 來源有被記錄', bool(srcs), f'  {sorted(srcs)}')
    ck('**u_prev 取到實際套用者**（applied）', 'applied' in srcs,
       f'  出現 {sum(1 for r in lg if r["u_prev_src"]=="applied")} 次')
    ck('所有週期都有完整耗時（含 total）',
       all('total' in r['timing_ms'] for r in lg))
    tt = sorted(r['timing_ms']['total'] for r in pub)
    if tt:
        p50 = tt[len(tt) // 2]
        p95 = tt[min(len(tt) - 1, int(round(0.95 * (len(tt) - 1))))]
        print(f'    核心 total（已發布輪）：p50 {p50:.2f} ms、'
              f'p95 {p95:.2f} ms、max {tt[-1]:.2f} ms')
    ck('殘差全部合格', bool(pub) and all(r['residual'] <= 1e-6 for r in pub),
       f'  max {max((r["residual"] for r in pub), default=float("nan")):.2e}')
    cw = sorted(r['cycle_wall_ms'] for r in lg
                if r.get('cycle_wall_ms') is not None)
    ck('逐輪**牆鐘**間隔有記錄（核心 total 不含介面開銷）', len(cw) >= 10,
       f'  {len(cw)} 筆')
    if cw:
        c50 = cw[len(cw) // 2]
        c95 = cw[min(len(cw) - 1, int(round(0.95 * (len(cw) - 1))))]
        print(f'    迴圈頂到頂：p50 {c50:.2f} ms、p95 {c95:.2f} ms、'
              f'max {cw[-1]:.2f} ms（設定週期 50 ms）')
        # **不可**用「牆鐘 − 核心」當介面開銷：迴圈會等到 deadline，
        # 所以牆鐘 p50 接近週期代表**跟得上**，差額主要是刻意的等待。
        ck('牆鐘間隔 p95 不超過週期的 1.1 倍（跟得上，無系統性超時）',
           c95 <= 55.0, f'  p95 {c95:.2f} ms')
        print(f'    超過週期的輪次：'
              f'{sum(1 for x in cw if x > 50.0)} / {len(cw)}'
              f'（**不可**用牆鐘減核心當介面開銷 —— 差額主要是等待 deadline）')
    nb = [r for r in lg if not r.get('published')]
    print(f'    未發布 {len(nb)} 輪；原因 '
          f'{sorted({r.get("dropped") or r.get("reason") or "?" for r in nb})}')

    # ---------- C 新鮮度：停止發布狀態 ----------
    print('\nC 新鮮度：到 sim 8 s 停止發布 /joint_states')
    d2, log2 = run_case('stale', stop_js_at=8.0, duration=16.0)
    ck('節點有產出紀錄', d2 is not None)
    if d2 is not None:
        lg2 = d2['log']
        late = [r for r in lg2 if r['sim_t'] > 7.5]
        stale_pub = [r for r in late
                     if r.get('published') and r['age_out'] > 0.2]
        ck('**過期解一筆都沒有發布**', not stale_pub,
           f'  違規 {len(stale_pub)} 筆')
        drops = [r for r in lg2 if not r.get('published')]
        ck('停發後有週期被擋下', len(drops) >= 1, f'  {len(drops)} 輪未發布')
        ages = [r['age_in'] for r in lg2 if r.get('published')]
        ck('已發布者的輸入年齡皆 <= 0.2 s',
           not ages or max(ages) <= 0.2 + 1e-9,
           f'  max {max(ages):.4f} s' if ages else '')

    print()
    print('WG2 介面測試：' + ('全部通過' if _bad == 0 else f'**{_bad} 項失敗**'))
    print('本測試只驗**介面**，不含 Isaac 物理回授 ⇒ '
          '**不得**據此宣稱 W-GMPC 自由空間核心已驗證。')
    return 1 if _bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
