"""O7 的直接反例：注入連續 300／260 ms 求解延遲，驗排程修正。

**不開 Isaac、不用 GPU。** 用持續發布狀態的最小假世界
（coman_solver_sched_world.py）加上實際的求解端節點
（以 coman_solver_delay_wrap.py 注入延遲）。

四項要確認的事（依裁定）：
  A 不再因錯過 deadline 而連續跳過回呼
  B 真正停止發布仍會觸發**原有**新鮮度守門
  C 過期結果不會被重新蓋上新時間後發布
  D 不出現追趕式密集發布

同時比對**修正前**的行為：以 --legacy 執行時把每輪的 _pump 拿掉
（還原成只在 deadline 尚有剩餘時間時才處理回呼），確認舊寫法會重現
main5 的守門中止 —— 沒有這個對照，就無法證明修的是真的缺陷。
"""
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SC = os.environ.get('COMAN_SCHED_TMP',
                    os.path.join('/tmp', f'coman_sched_{os.getpid()}'))
DOMAIN = os.environ.get('COMAN_SCHED_DOMAIN', '151')


def run_case(name, delay_ms, duration_s, stop_js_at=-1.0, legacy=False,
             max_input_age=0.2, timeout_s=90):
    """起假世界 ＋ 求解端，回傳 (solver_out, 求解端 stdout)。"""
    os.makedirs(SC, exist_ok=True)
    out = os.path.join(SC, f'{name}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN,
               COMAN_SOLVE_DELAY_MS=str(delay_ms))
    if legacy:
        env['COMAN_SCHED_LEGACY'] = '1'
    procs = []
    try:
        w = subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', str(duration_s + 25),
             '--stop-js-at', str(stop_js_at)],
            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        procs.append(w)
        time.sleep(3.0)
        s = subprocess.Popen(
            [sys.executable, '-u',
             os.path.join(HERE, 'coman_solver_delay_wrap.py'),
             '--out', out, '--stroke-m', '0.020', '--pull-duration-s', '4.0',
             '--vmax-base-lin', '0.035255', '--vmax-base-ang', '0.199900',
             '--vmax-arm', '0.999900',
             '--e2-base-lin', '0.05', '--e2-base-ang', '0.2',
             '--e2-arm-rate', '1.0',
             '--max-input-age', str(max_input_age),
             '--timeout-s', str(duration_s)],
            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        procs.append(s)
        try:
            log = s.communicate(timeout=timeout_s)[0]
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(s.pid), signal.SIGTERM)
            log = s.communicate(timeout=20)[0]
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
    bad = 0

    def check(name, cond, extra=''):
        nonlocal bad
        print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
        bad += not cond

    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')

    # ---------- A/C：連續 300／260 ms，狀態持續發布 ----------
    print('A+C 注入連續 300／260 ms（狀態持續發布）')
    d, log = run_case('long', '300,260', duration_s=20)
    check('求解端有產出紀錄', d is not None)
    if d is None:
        print(log[-2500:])
        return 1
    lg = d['log']
    guard = 'JointState 已' in log and '中止（guard）' in log
    check('**不再**因回呼飢餓觸發 JointState 守門', not guard,
          '' if not guard else '  **仍觸發**')
    ms = [r['solve_ms'] for r in lg]
    check('確實注入了長求解', bool(ms) and max(ms) > 250,
          f'  max {max(ms):.1f} ms' if ms else '')
    dropped = [r for r in lg if r.get('dropped')]
    published = [r for r in lg if r.get('seq', -1) > 0]
    check('有過期結果被丟棄', len(dropped) > 0,
          f'  丟棄 {len(dropped)}、發布 {len(published)}')
    # 已發布者的**輸入年齡**必須都在界內 —— 這是「不把舊解當新解」的實質判準，
    # 不是「一筆都不能發」。
    # `in_age`（模擬時鐘之差）與 `solve_ms`（牆鐘耗時）是不同的量，**不相比**。
    _ages = [r['in_age'] for r in published if r.get('in_age') == r.get('in_age')]
    check('**已發布者的輸入年齡全部 <= 0.2 s**',
          bool(_ages) and max(_ages) <= 0.2 + 1e-9,
          f'  max in_age {max(_ages):.4f} s、n={len(_ages)}' if _ages else '  無資料')
    _dages = [r['in_age'] for r in dropped if r.get('in_age') == r.get('in_age')]
    check('被丟棄者的輸入年齡全部 > 0.2 s',
          bool(_dages) and min(_dages) > 0.2,
          f'  min in_age {min(_dages):.4f} s' if _dages else '  無資料')
    if dropped:
        check('丟棄原因是輸入過期（不是別的）',
              all('過期' in r['dropped'] for r in dropped),
              f'  例：{dropped[0]["dropped"]}')
        check('被丟棄的列 cmd 為 None（沒有被蓋新時間再發）',
              all(r.get('cmd') is None for r in dropped))
    check('solve_ms 已分段（cons／qp／kin）',
          bool(lg) and all(k in lg[0] for k in ('cons_ms', 'qp_ms', 'kin_ms')))
    check('迴圈持續前進（不是解一次就卡住）', len(lg) >= 8, f'  {len(lg)} 輪')
    print(f'    分段例：cons {lg[0]["cons_ms"]:.2f} / qp {lg[0]["qp_ms"]:.2f} '
          f'/ kin {lg[0]["kin_ms"]:.2f} ms')
    print()

    # ---------- D：中等延遲，結果有效 ⇒ 驗發布間隔不追趕 ----------
    print('D 注入 120 ms（超過 50 ms 週期但輸入仍新鮮）')
    d2, log2 = run_case('mid', '120', duration_s=20)
    check('求解端有產出紀錄', d2 is not None)
    if d2 is not None:
        lg2 = d2['log']
        pub = [r for r in lg2 if r.get('seq', -1) > 0]
        check('結果**有**發布（沒有過度丟棄）', len(pub) >= 5, f'  {len(pub)} 筆')
        check('不再觸發 JointState 守門',
              not ('JointState 已' in log2 and '中止（guard）' in log2))
        # 追趕式密集發布是**牆鐘**現象（已落後卻立刻連發），
        # 所以用 pub_mono 量；來源模擬時間另外看它是否單調前進。
        wg = [pub[i + 1]['pub_mono'] - pub[i]['pub_mono']
              for i in range(len(pub) - 1)]
        burst = [g for g in wg if g < 0.020]
        check('**無追趕式密集發布**（牆鐘相鄰發布間隔皆 >= 20 ms）',
              not burst, f'  最小 {min(wg)*1e3:.1f} ms、'
                         f'{len(burst)} 筆過密' if wg else '')
        sg = [pub[i + 1]['t'] - pub[i]['t'] for i in range(len(pub) - 1)]
        check('來源模擬時間嚴格前進（不重複用同一狀態發兩次）',
              all(g > 0 for g in sg),
              f'  最小 {min(sg):.4f} s' if sg else '')
        if wg:
            print(f'    牆鐘發布間隔 中位 {sorted(wg)[len(wg)//2]*1e3:.1f} ms、'
                  f'最小 {min(wg)*1e3:.1f} ms、最大 {max(wg)*1e3:.1f} ms')
    print()

    # ---------- B：真的停發 ⇒ 原有守門仍要觸發 ----------
    print('B 到 sim 12 s 停止發布 /joint_states（其餘照發）')
    d3, log3 = run_case('stopjs', '0', duration_s=30, stop_js_at=12.0)
    fired = 'JointState 已' in log3 and '中止（guard）' in log3
    check('**原有新鮮度守門仍會觸發**', fired,
          '' if fired else '  **未觸發 —— 保護被弱化**')
    if fired:
        ln = [x for x in log3.splitlines() if 'JointState 已' in x]
        print(f'    {ln[-1].strip()}')
    print()

    # ---------- 對照：舊寫法必須重現 main5 的中止 ----------
    print('E 診斷（**非通過條件**）：舊排程是否重現 main5 的守門中止')
    d4, log4 = run_case('legacy', '300,260', duration_s=20, legacy=True)
    fired4 = 'JointState 已' in log4 and '中止（guard）' in log4
    print(f'    舊排程重現守門中止：{"是" if fired4 else "**否**"}')
    if d4 is not None:
        print(f'    舊排程 {len(d4["log"])} 輪（新排程 {len(lg)} 輪）')
    if not fired4:
        print('    ⇒ **本測試環境重現不出 main4／main5 的守門中止。**')
        print('       `TransformListener(spin_thread=True)` 把求解端節點加進一個')
        print('       專用 SingleThreadedExecutor 並在背景執行緒 spin()，')
        print('       所以「主迴圈沒 spin」並**不等於**回呼沒被處理。')
        print('       純 Python 忙迴圈每 5 ms（sys.getswitchinterval）讓出 GIL，')
        print('       也重現不出 GIL 獨佔。')
        print('       ⇒ **O7 的成因仍未確立**；本次修的是**可由程式碼直接確認**')
        print('         的排程缺陷（主迴圈唯一的回呼處理點在超時時不會執行），')
        print('         但**沒有證明**它就是 main5 中止的原因。')

    print()
    print('求解端排程反例：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    print('本測試只驗**排程與有效性**，**不涵蓋** O6（一次 solve 為何 301 ms）。')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
