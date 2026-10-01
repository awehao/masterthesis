#!/usr/bin/env python3
"""手臂設定點追蹤模型的**離線辨識與驗證**（可重跑）。

為什麼需要這支腳本
------------------
W-GMPC 核心的預測模型是 `q_{k+1} = q_k + dt·B(θ)u`，
即**命令速度在 dt 內完全達成**。第四趟（wgmpc_wg2_free4_182234）實測
顯示手臂不是這樣動的：執行端把命令速率**積分成位置設定點**，
再由 Isaac 的 articulation 位置控制器去追，於是多了一個
核心狀態向量裡沒有的隱藏狀態。

時間對齊（依執行端原始碼，不是猜的）
------------------------------------
`isaac_wholebody_sim_e2.py` 主迴圈每個物理步的順序是：

    world.step()                    # 物理前進一個 physics_dt
    qa = 讀實測關節位置               # ← log 的 joint*_act
    chain.step(t, dt, qa)           # 產生新設定點 sp，dt = **物理步長**
    apply_action(sp)                # 寫入位置目標 ← log 的 joint*_sp
    log.append(t, ..., sp, qa, ...)

因此 log 第 i 列的 `act_i` 是**寫入前**讀到的量測，`sp_i` 是**該步寫入**
並支配 i→i+1 這一段物理的位置目標。待辨識的一步關係式是

    act_{i+1} = act_i + α·(sp_i − act_i) + b                      … (1)

`sp` 的積分在 `wb_cmd_chain_e2.py::_apply` 是
`nxt = [p + r*dt ...]`，`dt` 為**物理步長**。上游命令是 20 Hz，
但設定點積分是**每物理步**（100 Hz）—— 兩者不可混稱。

兩個模型的 50 ms 預測比較
-------------------------
* **理想速度積分模型**（現行核心）：`q(+50ms) = q(0) + u_a · 0.05`
* **設定點追蹤模型**：把 (1) 連乘 5 個物理步

兩者用**同一段實際套用的命令／設定點歷史**。另外附一個更嚴格的版本：
設定點不取 log 實錄，而是由命令在視界內自行遞推
（這才是 MPC 實際會用的形式）。

輸出的 α 與 b 是**候選模型參數**，不是已定版。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ARM_N = 6
PHYS_DT_NOMINAL = 0.01
CTRL_DT_NOMINAL = 0.05


def load(run_dir: str):
    sim = json.load(open(os.path.join(run_dir, 'sim', 'wb_run.json')))
    wg2p = os.path.join(run_dir, 'wg2_out.json')
    wg2 = json.load(open(wg2p)) if os.path.exists(wg2p) else None
    ci = {c: i for i, c in enumerate(sim['log_cols'])}
    L = np.array(sim['log'], float)
    return sim, wg2, ci, L


def task_window(wg2, L, ci):
    """任務窗 = W-GMPC 節點**有發布**的週期範圍。沒有節點紀錄時退回全段。"""
    t = L[:, ci['t']]
    if wg2 is None:
        return t.min(), t.max(), '（無節點紀錄，取全段）'
    pub = [x for x in wg2['log'] if x.get('published')]
    if not pub:
        return t.min(), t.max(), '（節點未發布任何命令，取全段）'
    return pub[0]['sim_t'], pub[-1]['sim_t'], f'（節點發布 {len(pub)} 週期）'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sim, wg2, ci, L = load(a.run_dir)
    t = L[:, ci['t']]
    SP = np.column_stack([L[:, ci[f'joint{i}_sp']] for i in range(1, ARM_N + 1)])
    AC = np.column_stack([L[:, ci[f'joint{i}_act']] for i in range(1, ARM_N + 1)])

    t0, t1, how = task_window(wg2, L, ci)
    print('=== 手臂設定點追蹤模型：離線辨識與驗證 ===')
    print(f'趟次 {os.path.basename(a.run_dir.rstrip("/"))}')
    print(f'任務窗 {t0:.2f}–{t1:.2f} s {how}')

    # ---- 物理步長：由實測時間戳判定，不用名目值 ----
    w = (t >= t0) & (t <= t1)
    dts = np.diff(t[w])
    phys_dt = float(np.median(dts))
    print(f'物理步長（實測）p50 {phys_dt:.5f} s　'
          f'min {dts.min():.5f} max {dts.max():.5f}　名目 {PHYS_DT_NOMINAL}')
    if abs(phys_dt - PHYS_DT_NOMINAL) > 1e-4:
        print('  **實測步長與名目不符** ⇒ 後續一律用實測值')

    # ---- 設定點是否真的每物理步都在變（核對「不是 20 Hz」）----
    idx = np.where(w)[0]
    i0, i1 = idx[0], idx[-1]
    ok = np.isfinite(SP[i0:i1 + 1]).all(axis=1) & np.isfinite(AC[i0:i1 + 1]).all(axis=1)
    dsp = np.abs(np.diff(SP[i0:i1 + 1], axis=0)).max(axis=1)
    valid_pair = ok[:-1] & ok[1:]
    frac = float(np.mean(dsp[valid_pair] > 0))
    print(f'設定點在相鄰物理步有變化的比例 **{100*frac:.1f}%**'
          f'（n={int(valid_pair.sum())} 個 {phys_dt*1e3:.0f} ms 區間）')
    print(f'  ⇒ 設定點積分頻率 ≈ {1/phys_dt:.0f} Hz，**不是**上游命令的 20 Hz')

    # ---- 建一步資料集：(sp_i − act_i) → act_{i+1} − act_i ----
    S = SP[i0:i1 + 1]
    Q = AC[i0:i1 + 1]
    m = valid_pair
    X = (S[:-1] - Q[:-1])[m]                 # 誤差
    Y = (Q[1:] - Q[:-1])[m]                  # 實際位移
    n = X.shape[0]
    # ---- 前半估參數、後半驗證（**不同時擬合與驗證**）----
    half = n // 2
    tr = slice(0, half)
    print(f'\n切分：前半估參數 n={half}（t {t[i0]:.2f}–{t[i0+half]:.2f} s）、'
          f'後半驗證 n={n-half}（t {t[i0+half]:.2f}–{t[i1]:.2f} s）')

    print('\n=== 逐軸一步辨識（前半） ===')
    print(f"{'關節':>7} {'α':>9} {'b (rad)':>11} {'R²':>7} "
          f"{'等效 τ (s)':>11}")
    alpha = np.zeros(ARM_N); bias = np.zeros(ARM_N); tau = np.zeros(ARM_N)
    for j in range(ARM_N):
        A = np.column_stack([X[tr, j], np.ones(half)])
        sol, *_ = np.linalg.lstsq(A, Y[tr, j], rcond=None)
        alpha[j], bias[j] = sol
        pred = A @ sol
        ss = np.sum((Y[tr, j] - Y[tr, j].mean()) ** 2)
        r2 = 1.0 - np.sum((Y[tr, j] - pred) ** 2) / ss if ss > 0 else float('nan')
        # α = 1 − exp(−Δt/τ)  ⇒  τ = −Δt / ln(1 − α)
        tau[j] = (-phys_dt / math.log(1 - alpha[j])
                  if 0 < alpha[j] < 1 else float('nan'))
        print(f'  joint{j+1} {alpha[j]:9.5f} {bias[j]:+11.3e} {r2:7.4f} '
              f'{tau[j]:11.5f}')
    print(f'  六軸 α p50 **{np.median(alpha):.5f}**　'
          f'等效 τ p50 **{np.median(tau):.5f} s**')

    # ---- 50 ms 預測比較（後半驗證段）----
    k_ctrl = int(round(CTRL_DT_NOMINAL / phys_dt))
    print(f'\n=== {CTRL_DT_NOMINAL*1e3:.0f} ms 預測比較（後半驗證段，'
          f'{k_ctrl} 個物理步） ===')
    # 起點：驗證段中每個可用的物理步 i，比較 act_{i+k} 的預測與實際
    base = i0 + half
    starts = []
    # 起點需要 SP[i−1]（命令區間的左端）⇒ 下界 +1
    # 需要 SP[i−1]（命令區間左端）與 SP[i+kp+1]（逐筆重播的末增量）
    for i in range(max(base, i0 + 1), i1 - k_ctrl):
        r = slice(i - 1, i + k_ctrl + 2)
        if np.isfinite(SP[r]).all() and np.isfinite(AC[r]).all():
            starts.append(i)
    starts = np.array(starts)
    print(f'  起點數 {len(starts)}')

    def cmd_window(i):
        """i → i+kp 這段**實際作用**的命令積分。

        設定點的遞推是 `sp_i = sp_{i−1} + u_i·dt_p`（`_apply`），而 i→i+1 的
        物理由 **sp_i** 支配。因此 i→i+kp 作用的設定點是 sp_i … sp_{i+kp−1}，
        對應命令 u_i … u_{i+kp−1}，積分為 **sp_{i+kp−1} − sp_{i−1}**。
        先前寫成 sp_{i+kp} − sp_i，**偏了一個物理步**（Howard 指出）。
        """
        return SP[i + k_ctrl - 1] - SP[i - 1]

    def pred_ideal(i):
        """理想速度積分（現行核心的假設），命令取實際作用區間。"""
        return AC[i] + cmd_window(i)

    def pred_sp_logged(i):
        """設定點追蹤：連乘 kp 步，設定點**取 log 實錄**（模型品質上界）。"""
        q = AC[i].copy()
        for k in range(k_ctrl):
            q = q + alpha * (SP[i + k] - q) + bias
        return q

    def pred_sp_cmd_held(i):
        """設定點追蹤 ＋ 設定點由**單一保持命令**遞推（MPC 實際採用的形式）。

        命令取該視窗實際作用的**平均**速率。這是近似：20 Hz 命令在
        對齊控制週期的視窗內本來就固定，但逐物理步滑動的視窗會跨越更新。
        """
        u = cmd_window(i) / (k_ctrl * phys_dt)
        q = AC[i].copy(); s = SP[i].copy()
        for _ in range(k_ctrl):
            q = q + alpha * (s - q) + bias
            s = s + u * phys_dt
        return q

    def pred_sp_cmd_replay(i):
        """設定點追蹤 ＋ **逐筆重播實際命令**。

        s_0 = sp_i 已含 u_i，故第 k 次的增量是 u_{i+k+1}·dt_p =
        sp_{i+k+1} − sp_{i+k}。於是遞推出的設定點**逐步等於 log 實錄**
        —— 因為設定點本來就是命令的積分。所以這一項與
        `pred_sp_logged` 在數值上恆等，保留它是為了把這個恆等**明示**，
        不是當成第三個獨立數字。
        """
        q = AC[i].copy(); s = SP[i].copy()
        for k in range(k_ctrl):
            q = q + alpha * (s - q) + bias
            s = s + (SP[i + k + 1] - SP[i + k])     # = u_{i+k+1}·dt_p
        return q

    names = ['理想速度積分（現行核心）',
             '設定點追蹤｜設定點取實錄',
             '設定點追蹤｜單一保持命令（視窗平均）',
             '設定點追蹤｜逐筆重播實際命令（應與取實錄恆等）']
    fns = [pred_ideal, pred_sp_logged, pred_sp_cmd_held, pred_sp_cmd_replay]

    # **兩種起點集合**。逐物理步滑動的視窗會跨越 20 Hz 命令更新；
    # 對齊控制週期的才是 MPC 實際面對的情形。兩者都報，不挑一個。
    aligned = None
    if wg2 is not None:
        cand = []
        for x in wg2['log']:
            if not x.get('published'):
                continue
            i = int(np.argmin(np.abs(t - x['sim_t'])))
            if (i >= max(base, i0 + 1) and i + k_ctrl < i1
                    and abs(t[i] - x['sim_t']) <= phys_dt
                    and np.isfinite(SP[i - 1:i + k_ctrl + 2]).all()
                    and np.isfinite(AC[i - 1:i + k_ctrl + 1]).all()):
                cand.append(i)
        aligned = np.array(sorted(set(cand)))

    res = {}
    for tag, ss in (('逐物理步滑動', starts), ('對齊控制週期', aligned)):
        if ss is None or len(ss) == 0:
            print(f'\n  -- {tag}：無可用起點，略過')
            continue
        print(f'\n  -- 起點集合「{tag}」 n={len(ss)} --')
        act = np.array([AC[i + k_ctrl] for i in ss])
        for nm, fn in zip(names, fns):
            P = np.array([fn(i) for i in ss])
            rmse = np.sqrt((((P - act) * 1e3) ** 2).mean(axis=0))
            res[f'{tag}｜{nm}'] = rmse
            print(f'    {nm}')
            print('      逐軸 RMSE (mrad)：' +
                  '  '.join(f'j{j+1} {rmse[j]:.3f}' for j in range(ARM_N)))
            print(f'      **最差關節 {rmse.max():.3f} mrad**　'
                  f'六軸 p50 {np.median(rmse):.3f}')
        ri = res[f'{tag}｜{names[0]}'].max()
        print('    最差關節相對理想模型的改善：' + '　'.join(
            f'{nm.split("｜")[1]} {ri/res[f"{tag}｜{nm}"].max():.1f}×'
            for nm in names[1:]))

    # 恆等核對：逐筆重播 == 取實錄（設定點即命令的積分）
    for tag in ('逐物理步滑動', '對齊控制週期'):
        kl = f'{tag}｜{names[1]}'; kr = f'{tag}｜{names[3]}'
        if kl in res and kr in res:
            d = float(np.abs(res[kl] - res[kr]).max())
            print(f'\n  恆等核對（{tag}）：逐筆重播 vs 取實錄 RMSE 最大差 '
                  f'{d:.3e} mrad　{"**恆等成立**" if d < 1e-9 else "**不成立**"}')

    print('\n=== 界線 ===')
    print('  * α 與 b 是**候選模型參數**，由單趟（n=1）資料辨識，尚未定版。')
    print('  * 前半估、後半驗證只排除「同時擬合與驗證」，**不**等於跨趟泛化；')
    print('    其他負載、姿態與接觸情境未涵蓋。')
    print('  * 本核對從**執行端設定點**起算，故與安全層、E2 的影響分離；')
    print('    但這不表示安全層未影響 free4 的閉迴路行為。')
    print('  * 預測改善**不等於**閉迴路會到達並保持 —— 那要另行物理對照。')

    if a.out:
        json.dump({
            'run_dir': a.run_dir,
            'task_window_s': [t0, t1],
            'phys_dt_measured_s': phys_dt,
            'ctrl_steps_per_cycle': k_ctrl,
            'setpoint_change_fraction': frac,
            'n_pairs': int(n), 'n_train': int(half), 'n_valid': int(n - half),
            'alpha': alpha.tolist(), 'bias_rad': bias.tolist(),
            'tau_s': tau.tolist(),
            'alpha_p50': float(np.median(alpha)),
            'tau_p50_s': float(np.median(tau)),
            'n_pred_starts_sliding': int(len(starts)),
            'n_pred_starts_cycle_aligned': (int(len(aligned))
                                            if aligned is not None else 0),
            'rmse_mrad': {k: v.tolist() for k, v in res.items()},
            'status': '候選模型參數，未定版',
        }, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'\n-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
