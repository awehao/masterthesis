# Jerk Metric Issue — Deferred

## 觀察 (2026-06-03, batch10 跑完後)

`/cmd_vel` 衍生加速度的 std,在 GMPC+CBF 上爆掉:

| metric (m/s² 或 rad/s²) | RPP | MPPI | GMPC | **GMPC+CBF** |
|---|---|---|---|---|
| `jerk_vx` | 0.61 ± 1.13 | 0.13 ± 0.06 | 1.08 ± 0.15 | **18.5 ± 30.3** |
| `jerk_vy` | 0.00 ± 0.00 | 0.29 ± 0.03 | 0.42 ± 0.12 | **6.33 ± 10.4** |
| `jerk_wz` | 5.16 ± 12.8 | 0.23 ± 0.07 | 1.21 ± 0.20 | **36.3 ± 60.9** |

關鍵訊號:**std > mean** → 是少數巨大 spike 拉爆 std,不是整路在抖。

## 根因 (real, not artifact)

1. `analyze.py` 的 jerk 是 `np.diff(/cmd_vel) / np.diff(t)`,直接讀 controller **輸出**端,**與 AMCL 漂移無關** — 所以這是 controller 真實行為。
2. GMPC+CBF 的 spike 來源(由實際 log 觀察):
   - **緊急煞車**:OSQP `infeasible` / `max iterations reached` → `u = 0`,從巡航 v 瞬間到 0 → 巨大 Δu/Δt
   - **CBF 進出 danger 區的跳階**:`min_h < cbf_danger_thresh = 0.4` 時 `Q_xy *= 0.20`、`slack_weight *= 20`,QP 解答跳階
   - **OSQP polish=False**:每步求解未拋光,同 state 解略不同

## 為什麼這要解(不是純為圖好看)

- 真機跑會「打踉蹌」,影片給 advisor 看一眼就知道
- 論文後段 **whole-body MPC + 接觸操作**:底盤 jerk → 懸臂手臂末端執行器抖動放大 → 接觸力控制失準
- p95 jerk 而非 mean jerk 才是接觸操作真正在乎的(spike 才會打飛物件)

## 解決方案 (三選一或組合)

### A. 加 `nav2_velocity_smoother` 在 `/cmd_vel` 後 ★ 推薦
- 工作量:~30 min
- 效果:預期 jerk 直接降 5–10×,p95 也壓住
- 注意:Nav2 stack 已經有 `velocity_smoother` node 在 `nav2_omni_mppi.launch.py` 裡(remap `cmd_vel → cmd_vel_smoothed`)。**但 GMPC stack 沒接**,直接把 `gmpc_controller` 寫 `/cmd_vel` 出去
- 改法:
  1. `gmpc_node` 改成 publish `/cmd_vel_raw`
  2. 加一個 `nav2_velocity_smoother` 訂 `/cmd_vel_raw` → 發 `/cmd_vel`
  3. 或自寫 EWMA / one-pole low-pass node(簡單版)
- 代價:控制延遲 ~50–100 ms(smoother 的時間常數),但對 0.18 m/s 的 robot 可接受

### B. 換 metric:用 jerk 的 **p95**(95-percentile abs)而非 std ★ 5 分鐘
- 工作量:`analyze.py` 改 1 行 `np.std` → `np.percentile(abs, 95)`
- 效果:統計上更公平(std 對 outlier 太敏感),但**不改善真實 spike 行為**
- 用途:只是讓圖好看,不解決底層問題;報告寫法仍要誠實提 spike 存在

### C. 報告誠實 + 列 future work ★ 0 分鐘(最低成本退路)
- 在 P12 / P14 加一段:「動態場景下偶發 OSQP infeasible 觸發緊急煞車,造成 cmd_vel 短暫 spike (p95 jerk_vx ≈ N m/s²);Phase 2 將透過 velocity smoother + cheaper slack 解決」
- 缺點:advisor 一定會問,但 framing 對的話可接受

## 建議行動順序

1. **這次報告期**:走 **C**,plot.py 已暫時拿掉 jerk 三個 panel([plot.py:67-79](evaluation/plot.py#L67-L79) 已加註)
2. **Phase 1.5(暑假)**:做 **A + B**,重跑 batch
3. 同時順手做 **OSQP polish=True**(可能解決部分 spike,但會增加 solve_time,要驗 trade-off)

## 相關檔案

- [analyze.py:338-351](evaluation/analyze.py#L338-L351) — 目前 jerk 計算 (std-based)
- [plot.py:67-79](evaluation/plot.py#L67-L79) — 已將 jerk_{vx,vy,wz} 從 METRICS 移除,註解寫明原因
- [gmpc.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py) — 緊急煞車邏輯 + CBF gain-scheduling
- [gmpc_nav2_cbf.launch.py](src/ammr_wholebody_mpc/launch/gmpc_nav2_cbf.launch.py) — 目前 stack,沒接 velocity_smoother

## 不要在沒解之前

- 不要把 jerk 數字放進報告/簡報(已從 plot.py 摘除)
- 不要對 advisor 宣稱「GMPC+CBF 是 smooth controller」 — 真機上會被看出來
- 不要把 jerk 跟 baseline 一起比 — 解釋成本超過 metric 價值
