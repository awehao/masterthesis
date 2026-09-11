#!/usr/bin/env python3
"""步驟 4 整合驗收分析：故障注入 -> 逾時觸發 -> 真值停止。

停止判定（事前固定）：真值平移速度 <= 0.01 m/s 且持續 >= 1.0 s 模擬時間。
"""
import argparse, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
a = ap.parse_args()
R, L, A_MAX, W_MAX = 0.05, 0.245, 125.0, 5.55
W = np.array([[0., 1., L], [-1., 0., L], [0., -1., L], [1., 0., L]])
om = lambda u: (W @ np.asarray(u, float)) / R      # noqa: E731

C = json.load(open(os.path.join(a.run_dir, 'chain.json')))
INJ = json.load(open(os.path.join(a.run_dir, 'inject.json')))
run = json.load(open(os.path.join(a.run_dir, 'isaac_run.json'))).get('run', {})
t_inj = INJ['inject_wall']
print(f'== {os.path.basename(a.run_dir)}  故障 = {INJ["fault"]} ==')
print(f'  停止判定：真值速度 <= {C["stop_speed_thresh"]} m/s 且持續 '
      f'>= {C["stop_hold_s"]} s（模擬時間）')

# --- 牆鐘 -> 模擬時間的映射（位姿同時有兩者）---
P = np.array([[r[0], r[1], r[2], r[3]] for r in C['pose']])   # wall, sim, x, y
# 最近鄰的解析度受限於位姿取樣間隔（50 ms）；用線性擬合，並回報殘差。
_FIT = np.polyfit(P[:, 0], P[:, 1], 1)
_RES = float(np.max(np.abs(np.polyval(_FIT, P[:, 0]) - P[:, 1])))
def w2s(tw):
    return float(np.polyval(_FIT, tw)), _RES
sim_inj, err = w2s(t_inj)
print(f'  注入：牆鐘 {t_inj:.3f} -> 模擬 {sim_inj:.3f} s（映射誤差 {err*1000:.1f} ms）')

# --- 各序列在注入後的最後一筆 ---
for nm, key in (('/cmd_vel_nav', 'nav'), ('/cmd_vel_smoothed', 'smoothed'),
                ('/cmd_vel', 'out')):
    S = np.array(C[key]) if C[key] else np.zeros((0, 4))
    if not len(S):
        print(f'  {nm:20s} 無資料'); continue
    after = S[S[:, 0] > t_inj]
    last = S[-1]
    print(f'  {nm:20s} 共 {len(S)} 則，注入後 {len(after)} 則，'
          f'最後一筆 t+{last[0]-t_inj:+.3f}s u=({last[1]:+.4f},{last[2]:+.4f},{last[3]:+.4f})')

# --- guard 狀態：逾時觸發 ---
ST = [json.loads(s) for _, s in C['status']]
STW = [t for t, _ in C['status']]
to = [(STW[i], s) for i, s in enumerate(ST) if 'input_timeout' in s.get('faults', [])]
print(f'\n  guard 狀態 {len(ST)} 則')
if to:
    print(f'    input_timeout 首次觸發：牆鐘 t+{to[0][0]-t_inj:+.3f}s'
          f'（模擬 {w2s(to[0][0])[0]:.3f}），共 {len(to)} 則')
    print(f'    該筆 action={to[0][1]["action"]} '
          f'accel_guaranteed={to[0][1]["accel_guaranteed"]} λ={to[0][1]["lam"]}')
else:
    print('    input_timeout 未觸發')
modes = {}
for s in ST: modes[s['mode']] = modes.get(s['mode'], 0) + 1
print(f'    模式分布 {modes}')

# --- 獨立驗算：由 guard 輸出序列（seq 配對）檢查減速是否守限 ---
S2 = sorted(ST, key=lambda s: s['seq'])
rows = []
for i in range(1, len(S2)):
    p, c = S2[i-1], S2[i]
    if c.get('dt') is None: continue
    dw = float(np.max(np.abs(om(c['out']) - om(p['out']))))
    rows.append((c['seq'], c['dt'], dw, dw / c['dt'], c['accel_guaranteed']))
Rr = np.array([[r[1], r[2], r[3]] for r in rows])
gu = np.array([r[4] for r in rows], dtype=bool)
print(f'\n  === 由輸出序列獨立驗算（不只看 accel_guaranteed）===')
print(f'    全部 {len(Rr)} 筆：輪加速 max {Rr[:,2].max():.4f} / 上限 {A_MAX}'
      f'  超出 {int((Rr[:,2] > A_MAX + 1e-6).sum())}')
if gu.any():
    print(f'    accel_guaranteed=True 的 {int(gu.sum())} 筆：max {Rr[gu,2].max():.4f}'
          f'  超出 {int((Rr[gu,2] > A_MAX + 1e-6).sum())}')
if (~gu).any():
    print(f'    accel_guaranteed=False 的 {int((~gu).sum())} 筆：max {Rr[~gu,2].max():.4f}'
          f'（失效處置，不宣稱守限）')
W2 = np.array([float(np.max(np.abs(om(s['out'])))) for s in S2])
print(f'    輪速 max {W2.max():.4f} / 上限 {W_MAX}  超出 {int((W2 > W_MAX+1e-6).sum())}')

# --- 接收端逾時事件 ---
ev = run.get('cmd_timeout_events') or []
print(f'\n  === Isaac 接收端命令逾時（--cmd-timeout {run.get("cmd_timeout")}）===')
if ev:
    for e in ev[:5]:
        print(f'    觸發 模擬 {e[0]:.3f} s（注入後 {e[0]-sim_inj:+.3f} s）'
              f'，歸零前最後命令 {[round(v,4) for v in e[1]]}，逾時 {e[2]:.3f} s')
    print(f'    共 {len(ev)} 次；緊急歸零為**失效處置**，不宣稱滿足正常加速度限制')
else:
    print('    未觸發')

# --- 真值停止時刻 ---
sim = P[:, 1]; xy = P[:, 2:4]
d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
dt = np.diff(sim)
v = d / np.maximum(dt, 1e-9)
# 兩個不同的量，分開報：
#   first_below  真值速度首次低於門檻的時刻
#   stop_t       通過「持續 >= stop_hold_s」判定的時刻（= 該段的起點）
first_below = None
stop_t = None
hold = 0.0
seg_start = None
for i in range(len(v)):
    if sim[i] < sim_inj: continue
    if v[i] <= C['stop_speed_thresh']:
        if seg_start is None:
            seg_start = sim[i]
            if first_below is None:
                first_below = sim[i]
        hold += dt[i]
        if hold >= C['stop_hold_s'] and stop_t is None:
            stop_t = seg_start
            break
    else:
        hold = 0.0
        seg_start = None
print(f'\n  === 真值停止 ===')
vi = v[(sim[:-1] >= sim_inj - 0.5) & (sim[:-1] <= sim_inj)]
print(f'    注入前速度 {vi.mean():.4f} m/s（前 0.5 s 平均）')
if first_below is not None:
    print(f'    首次低於門檻：模擬 {first_below:.3f} s，注入後 {first_below - sim_inj:+.3f} s')
if stop_t is not None:
    print(f'    通過持續判定（>= {C["stop_hold_s"]} s）：該段起點 模擬 {stop_t:.3f} s，'
          f'注入後 {stop_t - sim_inj:+.3f} s')
else:
    print(f'    **在記錄區間內未達停止判定**；最後速度 {v[-1]:.4f} m/s')
