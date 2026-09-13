"""base 介面測試的三項證據。

1. 完整鏈路是否傳到執行端 —— 端點身分、收發與**實際套用**紀錄
2. 底盤是否按命令運動 —— 對**最終套用命令**積分，與實際位移比較；另報手臂漂移
3. 上游停止後是否安全收尾 —— 最終命令與**實際速度**時間序列

名目位移只是參考：若安全層完全不修改命令，本剖面的名目位移是 0.12 m
（0.03 m/s × 4.0 s 等效）。**驗算以套用命令為準**，不拿 0.12 m 當唯一答案。
"""
import json, os, sys
import numpy as np

run = sys.argv[1]
J = json.load(open(os.path.join(run, 'sim', 'wb_run.json')))
C = json.load(open(os.path.join(run, 'cmd_source.json')))
P = json.load(open(os.path.join(run, 'preflight.json')))
ci = {c: k for k, c in enumerate(J['log_cols'])}
L = J['log']
t = np.array([r[ci['t']] for r in L])
seq = np.array([r[ci['recv_seq']] for r in L])
vx = np.array([r[ci['vx_cmd']] for r in L])
vy = np.array([r[ci['vy_cmd']] for r in L])
wz = np.array([r[ci['wz_cmd']] for r in L])
bx = np.array([r[ci['base_x']] for r in L])
by = np.array([r[ci['base_y']] for r in L])
blin = np.array([r[ci['base_lin_meas']] for r in L])
bang = np.array([r[ci['base_ang_meas']] for r in L])
rates = np.array([[r[ci[f'{j}_rate_meas']] for j in
                   [f'joint{i}' for i in range(1, 7)]] for r in L])
acts = np.array([[r[ci[f'{j}_act']] for j in
                  [f'joint{i}' for i in range(1, 7)]] for r in L])

print('=== 1 完整鏈路是否傳到執行端 ===')
print(f"  起動前檢查：{'通過' if P['passed'] else '**未通過** ' + str(P['failed'])}")
for k in ('/joint_states 端點身分', '/odom 端點身分',
          '距離資料 端點身分', '/wb_vel_cmd 端點身分'):
    r = P['report'].get(k)
    if r:
        print(f"    {k}：{r['detail']}")
cc = J['cmd_chain']
print(f"  命令源發出 {C['n_sent']} 則 → 執行端 callback {J['callbacks']} 則")
print(f"  接收 {cc['received']}、拒收 {cc['rejected']}、失效 {cc['fail']}")
applied = seq > 0
print(f"  **實際套用** {int(applied.sum())} / {len(L)} 個物理步；"
      f"recv_seq {int(seq[applied].min()) if applied.any() else '—'}"
      f"–{int(seq[applied].max()) if applied.any() else '—'}")

print('\n=== 2 底盤是否按命令運動 ===')
dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
m = np.isfinite(vx)
sx = float(np.nansum(np.where(m, vx, 0) * dt))
sy = float(np.nansum(np.where(m, vy, 0) * dt))
print(f'  對**最終套用命令**積分：Δx {sx:+.4f} m、Δy {sy:+.4f} m'
      f'（合成 {np.hypot(sx, sy):.4f} m）')
print(f'  實際位移：Δx {bx[-1]-bx[0]:+.4f} m、Δy {by[-1]-by[0]:+.4f} m'
      f'（合成 {np.hypot(bx[-1]-bx[0], by[-1]-by[0]):.4f} m）')
print(f'  差：{np.hypot(bx[-1]-bx[0]-sx, by[-1]-by[0]-sy)*1000:.2f} mm')
print(f'  （名目參考 0.12 m —— 僅在安全層完全不修改命令時成立，不作唯一答案）')
print(f'  底盤實測速度峰值 {blin.max():.4f} m/s；角速度峰值 {abs(bang).max():.4f} rad/s')
print(f'  **手臂漂移**（base 模式零速度命令下）：')
for i in range(6):
    d = acts[-1, i] - acts[0, i]
    print(f'    joint{i+1}  位置變化 {d*1000:+8.3f} mrad；'
          f'速度峰值 {np.abs(rates[:, i]).max():.6f} rad/s')

print('\n=== 3 上游停止後是否安全收尾 ===')
t_last = C['sent'][-1][0] if C['sent'] else None
print(f"  命令源最後一則 sim_t = {t_last}")
print(f"  執行端停止原因 {J['stop_reason']}；凍結步數 {cc['frozen_steps']}")
print(f'  {"t":>7}{"vx_cmd":>10}{"底盤實測":>10}{"手臂速度max":>12}')
for tq in np.arange(t_last or t[0], t[-1] + 0.01, 1.0):
    k = int(np.argmin(np.abs(t - tq)))
    print(f'  {t[k]:7.2f}{vx[k]:10.4f}{blin[k]:10.5f}'
          f'{np.abs(rates[k]).max():12.6f}')
print('  **這趟不宣稱接收端逾時已驗證** —— 命令源停止發布後，安全層仍可能持續')
print('  輸出零、adapter 繼續發；上表只能說明上游失效處置與實際停止行為。')
print('  命令源先歸零再停止發布，**也不算「運動中突然斷訊」測試**。')
