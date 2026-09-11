"""最終命令的輪層檢查，以及 guard 內部狀態（只有錄了才報）。

分兩類輸出，因為兩趟的錄製設定不同：

  共同指標   `/cmd_vel` 的輪速合規。兩趟都有，可以配對比較。
  ON 專屬    `/wheel_guard/status` 的 mode／action／lam／dt／faults。
             v2_off_093455 沒有錄這些，**缺少不等於零次**，所以缺的時候
             印出「未錄製」而不是 0。

輪加速度分開處理。guard 的 `dt` 是它自己兩次輸出之間的實際間隔，是唯一
能拿來判定加速度是否合規的時間基準。沒有 guard 狀態時，只能用 bag 的
接收時間戳做估計 —— 那個間隔已知會被接收端擠壓，因此標為估計值，不當作
與輪速同等的驗收。

用法：
    python3 evaluation/analyze_wheel_chain.py RUN_DIR [RUN_DIR ...]
"""
import json
import os
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

from geometry_msgs.msg import Twist
from std_msgs.msg import String

L, R, W_MAX, A_MAX = 0.245, 0.05, 5.55, 125.0
W = np.array([[0, 1, L], [-1, 0, L], [0, -1, L], [1, 0, L]], float)
V_LIM = R * W_MAX                      # 0.2775 m/s，輪面速度上限


def read(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    cmd, smoothed, nav, guard, diag2 = [], [], [], [], []
    while r.has_next():
        tn, data, ts = r.read_next()
        t = ts * 1e-9
        if tn in ('/cmd_vel', '/cmd_vel_smoothed', '/cmd_vel_nav'):
            m = deserialize_message(data, Twist)
            row = (t, m.linear.x, m.linear.y, m.angular.z)
            {'/cmd_vel': cmd, '/cmd_vel_smoothed': smoothed,
             '/cmd_vel_nav': nav}[tn].append(row)
        elif tn == '/wheel_guard/status':
            guard.append(json.loads(deserialize_message(data, String).data))
        elif tn == '/gmpc/diag_v2':
            diag2.append(json.loads(deserialize_message(data, String).data))
    return (np.array(cmd), np.array(smoothed) if smoothed else None,
            np.array(nav), guard, diag2)


def wheel_speed(A):
    return np.abs(A[:, 1:4] @ W.T).max(axis=1)


def report(run_dir):
    bag = os.path.join(run_dir, 'bag')
    cmd, smoothed, nav, guard, diag2 = read(bag)
    print(f'== {os.path.basename(run_dir)} ==')

    # ---- 共同指標：最終命令的輪速 -------------------------------------
    mx = wheel_speed(cmd)
    over = mx > V_LIM + 1e-9
    print(f'  最終命令 /cmd_vel  n={len(cmd)}')
    print(f'    max|W·u| = {mx.max():.6f} m/s   限值 r·ω_max = {V_LIM:.4f}'
          f'   超限 {over.sum()}/{len(cmd)}')
    print(f'    對應輪速 {mx.max()/R:.4f} rad/s   限值 {W_MAX:.2f} rad/s')

    # ---- ON 專屬：guard 內部狀態 ---------------------------------------
    print('  guard 內部狀態：', end='')
    if not guard:
        print('**未錄製**（本趟 bag 沒有 /wheel_guard/status）')
        print('    → 修改次數、縮放量、模式切換在本趟不可報告，'
              '也不可當作「零次」')
    else:
        print(f'{len(guard)} 筆')
        schemas = {g.get('schema') for g in guard}
        print(f'    schema {schemas}')
        modes = [g['mode'] for g in guard]
        acts = [g['action'] for g in guard]
        lam = np.array([g['lam'] if g['lam'] is not None else np.nan
                        for g in guard], float)
        dt = np.array([g['dt'] if g['dt'] is not None else np.nan
                       for g in guard], float)
        for k in sorted(set(modes)):
            print(f'    mode  {k:<18} {modes.count(k):5d} 筆')
        for k in sorted(set(acts)):
            print(f'    action {k:<17} {acts.count(k):5d} 筆')
        sw = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
        print(f'    模式切換次數 {sw}')
        clipped = np.isfinite(lam) & (lam < 1.0 - 1e-9)
        print(f'    實際縮放（lam < 1）{clipped.sum()}/{np.isfinite(lam).sum()} 筆')
        if clipped.any():
            c = lam[clipped]
            print(f'      lam 最小 {c.min():.6f}  中位 {np.median(c):.6f}')
        faults = [f for g in guard for f in g.get('faults', [])]
        print(f'    faults {sorted(set(faults)) if faults else "無"}')
        ag = [g.get('accel_guaranteed') for g in guard]
        print(f'    accel_guaranteed True {ag.count(True)} / False {ag.count(False)}')
        # guard 自報的 dt 是唯一能判定加速度合規的時間基準
        d = dt[np.isfinite(dt) & (dt > 0)]
        if len(d):
            print(f'    guard dt 中位 {np.median(d)*1000:.2f} ms  '
                  f'p95 {np.percentile(d,95)*1000:.2f} ms  最大 {d.max()*1000:.2f} ms')

    # ---- 輪加速度：兩種來源分開講 --------------------------------------
    if guard:
        w = np.array([np.abs(np.array(g['out']) @ W.T).max() / R for g in guard])
        d = dt.copy()
        ok = np.isfinite(d) & (d > 0)
        aw = np.abs(np.diff(w))[ok[1:]] / d[1:][ok[1:]]
        n_over = int((aw > A_MAX + 1e-6).sum())
        print(f'  輪加速度（guard 逐筆 dt，可判定）'
              f' p95 {np.percentile(aw,95):.2f}  最大 {aw.max():.2f}'
              f'  限值 {A_MAX}  超限 {n_over}/{len(aw)}')
    else:
        t = cmd[:, 0]
        w = wheel_speed(cmd) / R
        d = np.diff(t)
        ok = d > 0
        aw = np.abs(np.diff(w))[ok] / d[ok]
        print(f'  輪加速度（**估計值**，來源為 bag 接收時間戳，'
              f'已知會被接收端擠壓）')
        print(f'    p95 {np.percentile(aw,95):.2f}  最大 {aw.max():.2f}'
              f'  限值 {A_MAX}（不作為驗收）')

    # ---- 逐筆來源對應：需要中間輸出才做得到 ----------------------------
    print('  逐筆來源對應：', end='')
    if smoothed is None:
        print('**無法做**（沒有 /cmd_vel_smoothed）')
        navset = set(map(tuple, np.round(nav[:, 1:4], 12)))
        hit = sum(1 for row in np.round(cmd[:, 1:4], 12)
                  if tuple(row) in navset)
        print(f'    只能做數值吻合：最終命令值出現在控制器輸出值集合中 '
              f'{hit}/{len(cmd)}')
        print('    這不是逐筆來源對應 —— 重複命令與延遲都可能讓它成立')
    else:
        print(f'可做（/cmd_vel_smoothed {len(smoothed)} 筆）')
        if guard:
            # 兩件事必須分開：
            #  * input_timeout 時 guard 刻意把「有效目標」換成零，但狀態裡的
            #    `target` 欄位記的是最後收到的輸入，不是那個零。拿它跟 out 比
            #    會把失效處置誤讀成輪級截斷。
            #  * lam == 1 時 out = prev + 1.0*(target-prev)，浮點往返不會精確
            #    等於 target。用位元相等去數「有改動」會數到 ~1e-9 的殘差。
            # 判定截斷一律看 lam，不看 out 與 target 的差。
            to = [g for g in guard if 'input_timeout' in g.get('faults', [])]
            ok = [g for g in guard if 'input_timeout' not in g.get('faults', [])]
            lam_ok = np.array([g['lam'] for g in ok
                               if g['lam'] is not None], float)
            n_clip = int((lam_ok < 1.0 - 1e-9).sum()) if len(lam_ok) else 0
            print(f'    正常樣本 {len(ok)}，其中輪級截斷（lam < 1）{n_clip} 筆')
            if len(lam_ok):
                print(f'      lam 最小 {lam_ok.min():.6f}  平均 {lam_ok.mean():.6f}')
            print(f'    input_timeout（有效目標改為零）{len(to)} 筆')
            resid = max((np.max(np.abs(np.array(g['out'])
                                       - np.array(g['target']))) for g in ok),
                        default=0.0)
            print(f'    正常樣本 |out - target| 最大 {resid:.3e}'
                  f'（lam=1 的浮點往返殘差，非截斷）')
    print(f'  /gmpc/diag_v2：{len(diag2)} 筆'
          + (f'（schema {sorted({d.get("schema") for d in diag2})}）'
             if diag2 else '（**未錄製**）'))
    print()


for rd in sys.argv[1:]:
    report(rd)
