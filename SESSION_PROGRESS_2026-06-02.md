# Session 進度紀錄 — 2026-06-02

> 目標:準備第二次進度報告。本 session 從整理 params 檔開始,做到 GMPC + CBF 動態避障 stack 完整 + 評估框架建立。

---

## 目錄
1. [本 session 完成清單](#完成清單)
2. [檔案結構變更](#檔案結構變更)
3. [Sprint 1 — Params 清理](#sprint-1--params-清理)
4. [Sprint 3 — GMPC 離線原型](#sprint-3--gmpc-離線原型)
5. [GMPC ROS2 整合(Phase 3 Step 2)](#gmpc-ros2-整合phase-3-step-2)
6. [評估框架](#評估框架)
7. [Baseline 對齊 + Weight Tuning](#baseline-對齊--weight-tuning)
8. [CBF 動態避障 stack](#cbf-動態避障-stack)
9. [驗證 + 已知數據](#驗證--已知數據)
10. [下次工作清單](#下次工作清單)
11. [論文章節對應](#論文章節對應)

---

## 完成清單

### ✅ 已完成(這 session)

- [x] **Sprint 1 任務 1** — 刪除 DiffDrive 舊壞檔(3 個),把 Baseline A RPP 改成正規對照
- [x] **Sprint 3 — GMPC 離線原型** — 完整 SE(2) 數學 + 5 軌跡 benchmark + DoD 全達標
- [x] **MATH.md** — 每公式來源、推導、citation、程式碼行號(13 節 + 13 條 self-test 覆蓋表)
- [x] **Phase 3 Step 2 — GMPC ROS2 node 整合** — `gmpc_node`、`goal_to_plan_relay`、整合 Nav2 SmacPlanner
- [x] **Continuous replan** — `goal_to_plan_relay` 加 1Hz timer 重發 plan 請求
- [x] **「Goal reached」綠字 log** — `gmpc_node` 到達 tolerance 時印通知
- [x] **評估框架** — `analyze.py`(bag→CSV)+ `plot.py`(CSV→bar chart)+ `record.sh`(規範化錄製)
- [x] **MCAP 自動偵測** — `analyze.py` 同時支援 .mcap(Jazzy 預設)和 .db3
- [x] **Baseline 對齊** — RPP 的 vx_max/wz_max/ax_max 拉成跟 MPPI/GMPC 相同(公平對比)
- [x] **GMPC v3 調參** — R 拉 6x、a_max 收緊(jerk_vx 從 1.25 → 0.82)
- [x] **CBF 數學** — `gmpc.py` 加 single-step CBF 線性不等式 + slack 變數 + safer fallback
- [x] **CBF self-test** — 5 case 驗證(遠/近/邊界 + 急煞)
- [x] **GMPC diagnostic topics** — `/gmpc/solve_time_ms`、`/gmpc/min_h`、`/gmpc/cbf_zones` MarkerArray
- [x] **`obstacle_aggregator`** — 從 Gazebo ground-truth pose 收 → 重發 `/gmpc/obstacles`
- [x] **CBF-aware launch** — `gmpc_nav2_cbf.launch.py`(全套 nav2 + relay + aggregator + GMPC CBF on)
- [x] **`analyze.py` 加安全指標** — `min_clearance_m`、`collision_count`、`solve_time_{mean,p95,max}_ms`
- [x] **`plot.py` 支援 4 method** — RPP / MPPI / GMPC / GMPC+CBF(13 個 metrics)
- [x] **`record.sh` 加錄 GMPC 診斷 topics** + Ctrl+C 修復

### ⏸ Pending(下次)
詳見 [下次工作清單](#下次工作清單)。

---

## 檔案結構變更

### 新增

| 路徑 | 說明 |
|------|------|
| [src/ammr_wholebody_mpc/offline_prototype/se2.py](src/ammr_wholebody_mpc/offline_prototype/se2.py) | SE(2) Lie 工具(self-test 6 組) |
| [.../offline_prototype/trajectories.py](src/ammr_wholebody_mpc/offline_prototype/trajectories.py) | 5 條參考軌跡 |
| [.../offline_prototype/kinematics.py](src/ammr_wholebody_mpc/offline_prototype/kinematics.py) | Omni forward 模擬 |
| [.../offline_prototype/gmpc.py](src/ammr_wholebody_mpc/offline_prototype/gmpc.py) | 離線版 QP + OSQP |
| [.../offline_prototype/run.py](src/ammr_wholebody_mpc/offline_prototype/run.py) | 閉迴路 driver |
| [.../offline_prototype/plot.py](src/ammr_wholebody_mpc/offline_prototype/plot.py) | 視覺化 |
| [.../offline_prototype/MATH.md](src/ammr_wholebody_mpc/offline_prototype/MATH.md) | 完整數學文件 + citation |
| [.../offline_prototype/logs/*.npz](src/ammr_wholebody_mpc/offline_prototype/logs/) | 5 軌跡實驗結果 |
| [.../offline_prototype/plots/*.png](src/ammr_wholebody_mpc/offline_prototype/plots/) | 6 張視覺化 |
| [src/ammr_wholebody_mpc/ammr_wholebody_mpc/se2.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/se2.py) | 安裝版 SE(2) |
| [.../ammr_wholebody_mpc/gmpc.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py) | **核心:GMPC + CBF + slack + safer fallback** |
| [.../path_processor.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/path_processor.py) | `/plan` → (X_ref_win, ξ_ref_win) |
| [.../gmpc_node.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py) | ROS2 GMPC node + 診斷 topics + CBF |
| [.../test_path_publisher.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/test_path_publisher.py) | 手動發測試 `/plan`(line/arc/square) |
| [.../goal_to_plan_relay.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/goal_to_plan_relay.py) | **`/goal_pose` → ComputePathToPose + continuous replan** |
| [.../obstacle_aggregator.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/obstacle_aggregator.py) | Gazebo pose → `/gmpc/obstacles` |
| [src/ammr_wholebody_mpc/config/gmpc_params.yaml](src/ammr_wholebody_mpc/config/gmpc_params.yaml) | GMPC 所有 param 預設值 |
| [src/ammr_wholebody_mpc/launch/gmpc_test.launch.py](src/ammr_wholebody_mpc/launch/gmpc_test.launch.py) | 獨立測試(無 nav2) |
| [.../launch/gmpc_nav2.launch.py](src/ammr_wholebody_mpc/launch/gmpc_nav2.launch.py) | nav2 + GMPC(無 CBF) |
| [.../launch/gmpc_nav2_cbf.launch.py](src/ammr_wholebody_mpc/launch/gmpc_nav2_cbf.launch.py) | **nav2 + GMPC + CBF + aggregator** |
| [evaluation/analyze.py](evaluation/analyze.py) | rosbag → CSV + 22 個指標 |
| [evaluation/plot.py](evaluation/plot.py) | CSV → 4 method × 13 metrics bar chart |
| [evaluation/record.sh](evaluation/record.sh) | 規範化錄 bag + Ctrl+C 反應快 |
| [evaluation/README.md](evaluation/README.md) | 評估框架說明 |
| [evaluation/BENCHMARK_WORKFLOW.md](evaluation/BENCHMARK_WORKFLOW.md) | 跑 benchmark 完整 SOP |

### 刪除(地雷檔)

| 路徑 | 原因 |
|------|------|
| ~~src/ammr_bringup/launch/nav2.launch.py~~ | DiffDrive params 配 Omni 底盤(矛盾) |
| ~~src/ammr_bringup/launch/nav2_static.launch.py~~ | 同上 |
| ~~src/ammr_bringup/config/nav2_params.yaml~~ | DiffDrive + MPPI 矛盾組合 |

### 修改

- [src/ammr_navigation/config/nav2_params.yaml](src/ammr_navigation/config/nav2_params.yaml) — Baseline A RPP 加檔頭註解、velocity_smoother 對齊 MPPI/GMPC
- [src/ammr_navigation/config/nav2_params_mppi.yaml](src/ammr_navigation/config/nav2_params_mppi.yaml) — Baseline B,維持(已修好)
- [src/ammr_wholebody_mpc/setup.py](src/ammr_wholebody_mpc/setup.py) + setup.cfg + package.xml — entry points + deps
- [src/ammr_wholebody_mpc/ammr_wholebody_mpc/test_path_publisher.py](src/ammr_wholebody_mpc/ammr_wholebody_mpc/test_path_publisher.py) — QoS 從 TRANSIENT_LOCAL → VOLATILE(避免殭屍殘留)

---

## Sprint 1 — Params 清理

### 問題
- 3 個 params 檔互相矛盾:`ammr_bringup/config/nav2_params.yaml`(DiffDrive)+ 2 個 launch 引用它,而底盤已是 Omni
- RPP 的 velocity_smoother 跟 MPPI/GMPC 數值不同(0.4 vs 0.35),不公平

### 解法
1. 刪除 3 個地雷檔(Diff Drive launch + 壞 params)
2. [nav2_params.yaml](src/ammr_navigation/config/nav2_params.yaml) 加檔頭明確標示「Baseline A」+ 「RPP 僅輸出 vx+wz」
3. RPP 的 chassis limit 對齊到 MPPI/GMPC:`vx_max=0.35`、`wz_max=0.8`、`ax_max=1.5`

### 對應的對齊表

| Baseline | T2 launch | Params | Planner | Controller |
|----------|-----------|--------|---------|-----------|
| A: RPP | `ammr_navigation/nav2.launch.py` | `nav2_params.yaml` | NavFn | RPP |
| B: MPPI | `ammr_navigation/nav2_omni_mppi.launch.py` | `nav2_params_mppi.yaml` | SmacPlanner2D | MPPI Omni |
| C: GMPC | `ammr_wholebody_mpc/gmpc_nav2.launch.py` | 上述 + `gmpc_params.yaml` | SmacPlanner2D | **GMPC node** |
| D: GMPC+CBF | `ammr_wholebody_mpc/gmpc_nav2_cbf.launch.py` | 上述 + CBF overrides | SmacPlanner2D | **GMPC + CBF + aggregator** |

---

## Sprint 3 — GMPC 離線原型

### 範圍
純 Python(不接 ROS),把 SE(2) Lie group + linear MPC + OSQP 從零實作 + 驗證。

### 數學鏈
1. SE(2) 操作:hat/vee、exp/log、inv、Ad、ad、`geodesic_error`
2. 連續誤差動態:`ė = -ad(ξ_ref)·e + δξ`(從定義推導,標準 Lie-group MPC pattern)
3. 離散化:Forward Euler `A_d = I - dt·ad(ξ_ref)` — **沿參考軌跡多點線性化**(修復 4/14 舊版單點線性化問題)
4. Condensed QP:`P = 2(Γᵀ Q̄ Γ + R̄)`、`q = 2Γᵀ Q̄ Φ e₀`
5. 約束:速度 box + 加速度有限差分

### DoD(全達標)

| 軌跡 | RMSE_xy | RMSE_yaw | solve mean | infeas |
|------|---------|----------|------------|--------|
| 01_straight | 0.036 m | 0.074 rad | 0.11 ms | 0 |
| 02_lateral | 0.072 m | 0.074 rad | 0.26 ms | 0 |
| 03_diagonal | 0.048 m | 0.074 rad | 0.15 ms | 0 |
| 04_s_curve | 0.025 m | 0.049 rad | 0.13 ms | 0 |
| **05_yaw_wrap** | **0.020 m** | **0.027 rad** | **0.09 ms** | **0** |

`yaw_wrap`(連續旋轉 25 秒、跨 ±π)在舊版必爆,**這版反而 RMSE 最小**。

### Prior art 對照
- 跟 **Tang et al. 2024**(*GMPC: Geometric Model Predictive Control for Wheeled Mobile Robot Trajectory Tracking*)**數學完全相同**(ADJ scheme = eq 17)
- **差異化**:Omni 底盤(B=I,paper 是 nonholonomic B=C(0))、Nav2 整合、CBF、whole-body 接口

---

## GMPC ROS2 整合(Phase 3 Step 2)

### 架構
```
/goal_pose ──► goal_to_plan_relay ──action──► planner_server (SmacPlanner2D)
                                                       │
                                                       ▼
                                                  /plan (Path)
                                                       │
                                                       ▼
[/model/dyn_obs_*/pose] ──► obstacle_aggregator ──► /gmpc/obstacles
                                                       │
                                                       ▼
                                                gmpc_controller ──► /cmd_vel
                                                       │
                                                       ▼
                            /gmpc/solve_time_ms, /gmpc/min_h, /gmpc/cbf_zones
```

### 關鍵設計決策
1. **不用 nav2 BT** — 自己寫 `goal_to_plan_relay` 把 `/goal_pose` 轉成 `ComputePathToPose` action,planner 出 `/plan`,GMPC 自己消費
2. **continuous replan** — relay 加 1Hz timer,沒到 goal 就持續重發 plan 請求,跟 nav2 BT 行為一致
3. **goal arrival edge-trigger log** — 綠字告知到達(預設 silent 會誤以為沒動)
4. **velocity_smoother 不需要** — GMPC 自己有 acceleration 限制

---

## 評估框架

### `analyze.py` 計算的指標(22 個)

| 類別 | 指標 |
|------|------|
| **任務完成** | `success`、`arrival_time_s`、`total_time_s`、`path_length_m` |
| **追蹤精度** | `tracking_rmse_m`(每個 /odom 對最近 /plan 點距離) |
| **平滑度** | `smooth_vx/vy/wz`(std of /cmd_vel) |
| **Jerk** | `jerk_vx/vy/wz`(std of accel = finite diff) |
| **安全** | `min_clearance_m`、`collision_count`、`collided` |
| **求解效能** | `solve_time_mean_ms`、`solve_time_p95_ms`、`solve_time_max_ms` |
| **計數** | `n_odom`、`n_cmd`、`n_plan`、`n_obstacles`、`n_solve_time` |

### `plot.py` 視覺化
- 4 method × 13 metrics summary bar chart(mean ± std)
- 每個 metric 獨立圖(報告用)
- 終端 print mean ± std + 成功率對比表

### `record.sh` 錄製
- 預設 60s 上限(可調)
- `timeout --signal=INT --kill-after=5` → Ctrl+C 立刻 propagate 進去 rosbag2
- 錄製 topics:`/odom /cmd_vel /cmd_vel_nav /plan /goal_pose /tf /tf_static /gmpc/solve_time_ms /gmpc/obstacles /gmpc/min_h`

---

## Baseline 對齊 + Weight Tuning

### 三個版本演進

**v1**:RPP 用自己原始 chassis 限制(vx_max=0.4)
**v2**:RPP chassis 對齊到 MPPI/GMPC(vx_max=0.35),GMPC 加 continuous replan
**v3**:GMPC R 拉 6 倍、a_max 收緊(降 jerk)

### corner_goal (17,17) 三方對比(v3 / aligned)

| Metric | RPP | MPPI | GMPC(v3) | 誰贏 |
|--------|-----|------|----------|------|
| arrival_time_s | 96.2 | **79.8** | 91.2 | MPPI |
| tracking_rmse_m | **0.075** | 0.165 | 0.143 | RPP |
| path_length_m | 28.5 | 27.3 | **27.0** | GMPC |
| **smooth_vy** | 0.000 | 0.015 | **0.044** | **GMPC ← Omni 賣點** |
| smooth_wz | 0.194 | **0.141** | 0.206 | MPPI |
| jerk_vx | **0.080** | 0.286 | 0.823 | RPP |
| jerk_wz | 0.343 | **0.146** | 1.010 | MPPI |
| n_plan(replan 數) | 98 | 79 | **92** | continuous replan OK |

**結論**:在靜態長距離追蹤場景,GMPC ≈ MPPI(略慢、RMSE 略大),**唯一明確贏的是 `smooth_vy`**(Omni 側移利用)。RPP 在 tracking/jerk 上是專家。

### v1 → v3 GMPC 改善

| Metric | v1 | v2 | v3 | 改善 |
|--------|-----|-----|-----|------|
| jerk_vx | 1.29 | 1.25 | **0.82** | 36% ↓ |
| arrival_time | 97.8 | 98.3 | **91.2** | 7s ↓(continuous replan 補) |
| n_plan | 2 | 1 | **92** | continuous replan 確認運作 |
| tracking_rmse | 0.17 | 0.13 | 0.14 | 微差 |

**結論**:R 拉 6x + a_max 收緊把 jerk 從 1.25 砍到 0.82,但**沒到預期的 0.3**。再拉 R 會犧牲追蹤精度。**真正要靠 CBF + 場景化指標**才能體現 GMPC 優勢。

---

## CBF 動態避障 stack

### 設計
- **單步 CBF**(Ames et al. 2017 標準 safety filter):h(p) = ||p - o||² - (r + d_safe)²,約束 ḣ + α·h ≥ 0 只塞在 δξ_0
- **Slack 變數**:加 ε ≥ 0 + 高罰 ρ·ε²,讓 CBF 在跟 acc 衝突時變「best-effort」,QP 永遠可解
- **Safer fallback**:OSQP infeasible 時送 u=0(emergency brake),不再 fallback 到 ξ_ref(會撞)

### CBF self-test 驗證(5 case)

| Case | 描述 | 結果 | 驗證 |
|------|------|------|------|
| E | 距離遠 h=1.89 | vx=0.300(全速) | ✓ 不影響 |
| F | h<0(已在禁區) | vx=0.225(acc 上限減速) | ✓ |
| G | 中距離 h=0.28 | vx=0.225(主動減速) | ✓ |
| H | 偏軸 | vx=0.225, vy=-0.05 | ⚠️ Q 拉回路徑,CBF 不強制避(可接受) |
| I | h=0 邊界 + 高速接近 | **vx=0 急煞** | ✓ Slack + safer fallback |

### Diagnostic topics
- `/gmpc/solve_time_ms`(Float32):per-step OSQP wall time → 算 mean/p95/max
- `/gmpc/min_h`(Float32):smallest CBF barrier value → 算 min_clearance
- `/gmpc/cbf_zones`(MarkerArray):橘色半透明圓柱在 Foxglove 顯示 CBF 安全圈

### 對應的 launch
```
T1: ros2 launch ammr_bringup gazebo_dynamic.launch.py
    └─ world: random_room_dynamic.sdf (含 3 個動態圓柱)
    └─ 啟 obstacle_driver

T2: ros2 launch ammr_wholebody_mpc gmpc_nav2_cbf.launch.py
    ├─ nav2 servers: map_server + amcl + planner_server + lifecycle_mgr
    ├─ goal_to_plan_relay
    ├─ obstacle_aggregator (從 /model/dyn_obs_*/pose → /gmpc/obstacles)
    └─ gmpc_controller (cbf_enable=True, α=2.0, margin=0.30m)
```

---

## 驗證 + 已知數據

### 離線 Sprint 3(完整跑過,DoD 達標)
- 5 軌跡 × 100-480 步 × 0 infeasibility
- solve time mean 0.09-0.26 ms

### ROS2 integration(基本跑過)
- corner_goal 3 baseline 全部成功到達 (17, 17)
- v3 數據完整,9 個指標都有(見上方 corner_goal 對比表)

### CBF stack(smoke-tested,**還沒跑完整 benchmark**)
- 4 個 entry points 都 ros2 pkg executables 可見
- launch syntax check 過(7 nodes)
- obstacle_aggregator 獨立啟動 OK(log: "tracking 3 obstacles @ 20Hz")
- analyze.py 25 個 CSV 欄位、plot.py 13 個指標 × 4 方法
- **尚未在 Gazebo 動態場景做 4-way benchmark**(下次工作第一個)

---

## 下次工作清單

### 🥇 P0 — 第一輪 dynamic benchmark(必做)

跑 [evaluation/BENCHMARK_WORKFLOW.md](evaluation/BENCHMARK_WORKFLOW.md) 的流程,改用 `gazebo_dynamic.launch.py`:

```bash
# 3 個 baseline × 動態場景 × goal (17, 17)
ros2 launch ammr_bringup gazebo_dynamic.launch.py     # T1 共用
# A: ros2 launch ammr_navigation nav2.launch.py + record mppi
# B: ros2 launch ammr_navigation nav2_omni_mppi.launch.py
# C: ros2 launch ammr_wholebody_mpc gmpc_nav2.launch.py        ← 無 CBF
# D: ros2 launch ammr_wholebody_mpc gmpc_nav2_cbf.launch.py    ← 有 CBF
```

**預期結果**(報告主圖):
- 碰撞數:RPP/MPPI 0-1 次 / GMPC(無 CBF)≥ 1 次撞 / **GMPC+CBF = 0**
- min_clearance:GMPC+CBF **永遠 ≥ d_safe=0.30m**
- solve_time:GMPC < 1ms / MPPI 5-15ms

### 🥈 P1 — **全 horizon CBF + 動態障礙物**(本次延後)

> **這是 b. 階段 2 的主要工作**。預估 1-2 天工程 + 1-2 天 debug。

**目前限制**(單步 CBF):
- 只看當下 obstacle 位置,不預測未來
- 動態障礙物從旁邊衝來時,**CBF 沒提前反應**,直到撞前一瞬間才急煞 → 仍可能撞

**升級內容**:
1. **取得障礙物速度** — `obstacle_aggregator` 加 dt 差分 / 從 driver yaml 拉
2. **對每個 horizon step k 計算預測**:
   - `o_i(k) = o_i(0) + v_obs · k·dt`(constant velocity prediction)
   - `X_k+1 = X_k · exp((ξ_ref+δξ_k)·dt)`(需要 forward integrate)
3. **CBF rows per (obstacle × step)**:
   - 20 step × 3 obstacle = **60 條約束**
   - QP 從 ~60 rows 變 ~120 rows
4. **forward integration 對 δξ 非線性**:
   - 用 nominal trajectory(ξ_ref 對應的 X_ref)當近似 → 變保守
   - 或用迭代 SQP(再次線性化) → 解兩次 QP
5. **可能要加 KF 預測軌跡**(constant velocity 假設爛時)

**參考文獻**:
- Zeng, Zhang, Sreenath (2021) *"Safety-Critical MPC with Discrete-time CBF"* ACC
- Singletary, Ahmadi, Burdick (2022) *"Multi-Step Discrete-Time CBF"*

**預期增益**:動態場景下避障路徑更平滑,提前繞行 vs 急煞。

### 🥉 P2 — Monte Carlo 自動化

**目前手動**:每個 baseline 要手動 T1/T2/T3/T4 一次。

**升級**:
1. `generate_map.py` 接受 SEED arg → 多 random seeds 場景
2. `dynamic_trajectories.yaml` 也 seed-parameterise(隨機起終點 + 速度)
3. `benchmark_mc.sh`:loop over seeds × methods,自動 launch / send goal / record / kill
4. analyze.py 一次處理所有 bag
5. plot.py 改畫 box plot 或 error bar(σ across seeds)

**最小可發表**:5 seeds × 4 methods = 20 runs,跑半天。

### 🏅 P3 — `/scan` 整合 LiDAR detection

**目前**:obstacle 位置從 Gazebo ground truth 拿(`/model/dyn_obs_*/pose`)
**升級**:從 `/scan` 抓 cluster(DBSCAN)→ centroids → 軌跡追蹤 → 傳入 CBF
**意義**:真實機器人沒有 ground truth,要驗證 stack 真的能 deploy

### 🏆 P4 — Whole-body 整合(Phase 2 大跨步)

把底盤 GMPC 擴展成 SE(2) + 7-DOF 手臂的 whole-body QP:
- end-effector twist:`V_ee = Ad_G · ξ_base + J_arm · q_dot`
- 變數從 3 維 → 10 維
- 加 arm joint limit、velocity limit
- 對應 thesis Phase 2

### 🎓 P5 — 接觸操作(Phase 3 thesis 主體)

把 CBF 從「避障(不等式)」擴展到「受約束接觸(等式)」:
- 開門:手把 ∈ 弧線(等式約束)
- 拉抽屜:手把 ∈ 直線(等式約束)
- 加 force/torque CBF(力極限)
- 對應 thesis Phase 3

---

## 論文章節對應

### 第二次進度報告(2 週內)

```
1. 引言 / 動機(沿用第一次)

2. 相關工作 (NEW)
   2.1 Lie-group MPC: Sola 18, Lynch & Park 17, Tang 24
   2.2 Nav2 controllers: NavFn+RPP, SmacPlanner2D+MPPI
   2.3 CBF safety: Ames 17, Zeng 21

3. 方法 (NEW, 引用 MATH.md)
   3.1 SE(2) error dynamics(eq 7, 8, 9 from Tang 24)
   3.2 Linearisation scheme (eq 17 from Tang 24)
   3.3 Discrete QP construction (Phi, Gamma, P, q)
   3.4 Constraint handling (velocity / acceleration)
   3.5 CBF single-step safety filter (NEW: Ames 17 形式)

4. 系統整合 (NEW)
   4.1 ROS2 Nav2 整合架構圖
   4.2 goal_to_plan_relay + continuous replan
   4.3 obstacle_aggregator
   4.4 Foxglove 視覺化 + diagnostic topics

5. 實驗 (NEW)
   5.1 離線軌跡追蹤驗證 (Sprint 3 五軌跡 RMSE 表)
   5.2 ROS2 整合 baseline 對比 (corner_goal 三方表)
   5.3 動態場景對比 (NEW: 4-way 含 GMPC+CBF)
        ★ collision count 表
        ★ min_clearance 圖
        ★ solve_time 對比(GMPC < 1ms vs MPPI 5-15ms)
        ★ tracking RMSE / arrival_time

6. 失敗教訓與修復
   6.1 4/14 (x,y,θ)-state MPC failure (wrap-around)
   6.2 修復:SE(2) log + 多點線性化

7. 結論 + 後續工作
   ★ 全 horizon CBF(P1)
   ★ /scan 整合(P3)
   ★ Monte Carlo 框架(P2)
   ★ Whole-body / 接觸操作(P4 / P5)
```

### Thesis 主體(暑假/下學期)

```
Phase 1: dynamic nav (本次報告)
Phase 2: whole-body 底盤+手臂協調
Phase 3: 接觸操作(開門/拉抽屜)
Phase 4: CBF 安全保證 + 動態場景擴展
Phase 5: 與真實 hardware 整合
```

---

## Quick reference — 常用指令

### 開發循環
```bash
cd ~/masterthesis
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ammr_wholebody_mpc
source install/setup.bash
```

### 跑 GMPC + CBF 動態場景
```bash
# T1: Gazebo + 動態圓柱
ros2 launch ammr_bringup gazebo_dynamic.launch.py
# T2: nav2 + GMPC + CBF + aggregator
ros2 launch ammr_wholebody_mpc gmpc_nav2_cbf.launch.py
# T3: 錄
cd ~/masterthesis/evaluation && ./record.sh gmpc_cbf dyn_corner 150
# T4: 送 goal
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" --once
# 分析
python3 analyze.py bags/gmpc_cbf__dyn_corner --method gmpc_cbf --run dyn_corner
python3 plot.py
xdg-open results/figs/summary.png
```

### 離線 Sprint 3 重跑
```bash
cd src/ammr_wholebody_mpc/offline_prototype
python3 run.py
python3 plot.py
xdg-open plots/00_summary.png
```

### Foxglove 連線
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
# 連 ws://localhost:8765
```

---

*Last updated: 2026-06-02 session end. 下次接續從「P0 第一輪 dynamic benchmark」開始。*
