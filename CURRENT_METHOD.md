# 目前方法完整整理 — GMPC + Horizon CBF + Gain Scheduling

> 截至 2026-06-02 工程結束的所有技術細節,包含數學、架構、實作、參數。
> 主要供:第二次進度報告寫作、論文方法章節撰寫。

---

## 目錄
1. [系統架構總覽](#1-系統架構總覽)
2. [數學:SE(2) GMPC](#2-數學se2-gmpc)
3. [CBF:Full-Horizon Control Barrier Function](#3-cbffull-horizon-control-barrier-function)
4. [Gain Scheduling 動態權重調整](#4-gain-scheduling-動態權重調整)
5. [Reference Path 處理](#5-reference-path-處理)
6. [Obstacle 感知 (Ground Truth)](#6-obstacle-感知-ground-truth)
7. [ROS2 節點與訊息流](#7-ros2-節點與訊息流)
8. [QP 矩陣總組裝](#8-qp-矩陣總組裝)
9. [所有可調參數一覽](#9-所有可調參數一覽)
10. [程式碼對應索引](#10-程式碼對應索引)
11. [已知局限](#11-已知局限)
12. [Cite 文獻](#12-cite-文獻)

---

## 1. 系統架構總覽

```
                  ┌──────────────────────────┐
                  │  Nav2 Stack              │
                  │  ┌────────────────────┐  │
                  │  │ map_server         │  │
                  │  │ amcl               │  │
                  │  │ planner_server     │  │
                  │  │ (SmacPlanner2D)    │  │
                  │  │ lifecycle_manager  │  │
                  │  └────────────────────┘  │
                  └────────────┬─────────────┘
                               │ /plan (nav_msgs/Path)
                               ▼
   /goal_pose ──► goal_to_plan_relay ──► /compute_path_to_pose action
                  (1 Hz continuous replan)
                               │
                               ▼ /plan
   ┌───────────────────────────────────────────────────────────┐
   │  GMPC Node (20 Hz control loop)                           │
   │                                                           │
   │  /plan ─────► path_processor ──┐                          │
   │                                ▼                          │
   │  TF map→base ──► X_now ───► GMPC.solve() ──► /cmd_vel    │
   │                                ▲                          │
   │  /gmpc/obstacles ──────────────┘                          │
   │  (x, y, r, vx, vy) per obs                                │
   └───────────────────────────────────────────────────────────┘
                               │
                               ▼ /cmd_vel
                   ┌──────────────────────────┐
                   │  omni_drive_controller   │
                   │  (4-wheel kinematics)    │
                   └──────────────────────────┘

   Side channels:
     Gazebo: /model/dyn_obs_*/{pose, cmd_vel}
                       │
                       ▼
                obstacle_aggregator
                       │
                       ▼ /gmpc/obstacles (Float32MultiArray)
                       
   Diagnostics out of gmpc_node:
     /gmpc/solve_time_ms  (per-step OSQP wall-time)
     /gmpc/min_h          (smallest CBF barrier)
     /gmpc/cbf_zones      (MarkerArray for Foxglove)
```

**控制頻率**:20 Hz(dt = 50 ms)
**預測 horizon**:N = 20 steps = 1.0 秒
**求解器**:OSQP 1.1.1

---

## 2. 數學:SE(2) GMPC

### 2.1 表示

機器人姿態 $X \in SE(2)$,3×3 同質矩陣:

$$X = \begin{bmatrix} R(\theta) & p \\ 0 & 1 \end{bmatrix}, \quad p = (x, y)^\top \in \mathbb{R}^2$$

Body twist $\xi = (v_x, v_y, \omega)^\top \in \mathfrak{se}(2)$,**body frame**。

Body twist 運動學:
$$\dot X = X \cdot \hat\xi$$

### 2.2 SE(2) Lie 工具(全部用 closed form,沒近似)

| 運算 | 公式 | 程式 |
|------|------|------|
| hat | $\hat\xi = \begin{bmatrix}0 & -\omega & v_x \\ \omega & 0 & v_y \\ 0 & 0 & 0\end{bmatrix}$ | `se2.hat` |
| exp | $\exp(\hat\xi) = \begin{bmatrix}R(\omega) & V(\omega)v \\ 0 & 1\end{bmatrix}$ where $V(\omega) = \frac{1}{\omega}\begin{bmatrix}\sin\omega & -(1-\cos\omega) \\ 1-\cos\omega & \sin\omega\end{bmatrix}$ | `se2.exp_` |
| log | $\theta = \mathrm{atan2}(R_{21}, R_{11})$, $v = V^{-1}(\theta)\cdot p$ | `se2.log_` |
| Ad | $\mathrm{Ad}_X = \begin{bmatrix}R & -Jp \\ 0 & 1\end{bmatrix}$, $J = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$ | `se2.Ad` |
| ad | $\mathrm{ad}(\xi) = \begin{bmatrix}0 & -\omega & v_y \\ \omega & 0 & -v_x \\ 0 & 0 & 0\end{bmatrix}$ | `se2.ad` |
| 幾何誤差 | $e = \log(X_{\mathrm{ref}}^{-1} X)^\vee$ | `se2.geodesic_error` |

### 2.3 誤差動態

$X_{\mathrm{ref}}(t), X(t)$ 都按 body twist 演化。定義 $E = X_{\mathrm{ref}}^{-1} X$,代入運動學:

$$\dot E = E \hat\xi - \hat\xi_{\mathrm{ref}} E$$

小誤差 $E \approx I + \hat e$ 展開到一階:

$$\boxed{\dot e = -\mathrm{ad}(\xi_{\mathrm{ref}}) \cdot e + (\xi - \xi_{\mathrm{ref}})}$$

定義 $\delta\xi = \xi - \xi_{\mathrm{ref}}$,離散化(Forward Euler):

$$\boxed{e_{k+1} = A_d(k) \cdot e_k + \Delta t \cdot \delta\xi_k}, \quad A_d(k) = I - \Delta t \cdot \mathrm{ad}(\xi_{\mathrm{ref}}(k))$$

**注意**:$A_d$ 隨 $k$ 變化 — **沿參考軌跡的多點線性化**。修復了 (x,y,θ) 狀態 MPC 在大角度誤差時的失效。

### 2.4 Prediction matrices (condensed form)

$$E = \Phi \cdot e_0 + \Gamma \cdot z, \quad z = [\delta\xi_0; \delta\xi_1; \ldots; \delta\xi_{N-1}]^\top \in \mathbb{R}^{3N}$$

其中 $\Phi \in \mathbb{R}^{3N \times 3}$ 由 $\prod A_d(k)$ 組成,$\Gamma$ 是下三角 block matrix。
程式碼:[`gmpc._build_prediction`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py)

### 2.5 Cost function

$$J = \underbrace{\sum_{k=1}^{N-1} e_k^\top Q\, e_k + e_N^\top Q_f\, e_N}_{\text{tracking}} + \underbrace{\sum_{k=0}^{N-1} \delta\xi_k^\top R\, \delta\xi_k}_{\text{input}} + \underbrace{\sum_{k=1}^{N-1} \rho \cdot \varepsilon_k^2}_{\text{slack}}$$

寫成 OSQP 標準形式 $\frac{1}{2} z^\top P z + q^\top z$:

$$P = 2 \cdot (\Gamma^\top \bar Q \Gamma + \bar R), \quad q = 2 \cdot \Gamma^\top \bar Q \Phi \cdot e_0$$

(slack 列加在 $P, q$ 末端 — 詳見 §8)

### 2.6 約束

**速度 box**(逐 step):
$$u_{\min} \le \xi_{\mathrm{ref}}(k) + \delta\xi_k \le u_{\max}, \quad k = 0, \ldots, N-1$$

**加速度 box**(相鄰差分):
$$|u_k - u_{k-1}| \le a_{\max} \cdot \Delta t, \quad u_{-1} := \xi_{\mathrm{prev}}$$

---

## 3. CBF:Full-Horizon Control Barrier Function

### 3.1 Barrier function

對每個障礙物 $i$,每個 horizon step $k$:

$$h_i(k) = \|p(k) - o_i(k)\|^2 - (r_i + d_{\mathrm{safe}})^2$$

- $p(k)$:機器人在 step $k$ 的位置(linearization point,見下)
- $o_i(k) = o_i(0) + v_i \cdot k\Delta t$:**等速度預測**未來位置
- $r_i$:obstacle 半徑
- $d_{\mathrm{safe}}$:額外 safety margin(現在 = 0.40 m)

**Linearization point 選擇**:
| step | $p(k)$ |
|------|--------|
| $k = 0$ | $X_{\mathrm{now}}$ 實際當前位置 |
| $k \ge 1$ | $X_{\mathrm{ref}}(k)$ 參考軌跡 nominal 位置 |

### 3.2 CBF 條件

連續時間 velocity-layer CBF:
$$\dot h_i(k) + \alpha \cdot h_i(k) \ge -\varepsilon_k$$

其中:
$$\dot h_i(k) = 2 (p(k) - o_i(k))^\top \big( R(\theta(k)) \cdot v_{\mathrm{body}}(k) - v_{\mathrm{obs}}(k) \big)$$

代入 $v_{\mathrm{body}}(k) = \xi_{\mathrm{ref}}(k) + \delta\xi_k$,整理成 $\delta\xi_k$ 的**線性不等式**:

$$\boxed{\nabla_{\mathrm{body}}(k) \cdot \delta\xi_k + \varepsilon_k \ge b_i(k)}$$

其中:
- $\nabla_{\mathrm{body}}(k) = 2 (p(k) - o_i(k))^\top R(\theta(k)) \in \mathbb{R}^{1 \times 3}$
- $b_i(k) = 2(p(k) - o_i(k))^\top v_{\mathrm{obs}}(k) - \alpha h_i(k) - \nabla_{\mathrm{body}}(k) \xi_{\mathrm{ref}}(k)$

### 3.3 Slack 變數

$$\varepsilon_0 = 0 \text{(hardcoded — 當下絕對安全)}$$
$$\varepsilon_k \ge 0, \quad k = 1, \ldots, N-1 \text{(soft,有 L2 罰款)}$$

**為什麼這樣設計**:
- $k=0$ 是「現在這一瞬間」,允許 slack 就等於允許「現在撞」 — 不可接受
- $k \ge 1$ 是預測未來,允許 slack 是為了在 receding horizon 內**有時間化解危機**,避免 QP 因預測太早而 infeasible

**Cost 中的 slack**:$\sum_k \rho \cdot \varepsilon_k^2$,$\rho = $ `cbf_slack_weight` × `slack_scale`(gain scheduling 動態調)

### 3.4 程式對應
- `gmpc._build_cbf_horizon`:組 CBF rows
- `gmpc.GMPC.solve`:整合 + augment QP

---

## 4. Gain Scheduling 動態權重調整

### 4.1 動機

固定的 $Q, R, \rho$ 會在 robot 靠近 obstacle 時造成兩個衝突:
- Tracking 想拉 robot 回 path
- CBF 想推 robot 離 obstacle

導致機器人在邊界震盪或卡死(Freezing Robot Problem)。

### 4.2 Danger level

每個 control step 開始,**pre-pass** 計算當下最小 barrier:

$$h_{\min} = \min_i \big( \|p_{\mathrm{now}} - o_i(0)\|^2 - (r_i + d_{\mathrm{safe}})^2 \big)$$

然後計算 danger level $\in [0, 1]$:

$$\boxed{\mathrm{danger} = \max\Big(0, \; 1 - \frac{\max(0, h_{\min})}{\eta_{\mathrm{thresh}}}\Big)}$$

- $h_{\min} \ge \eta_{\mathrm{thresh}}$ → danger = 0(安全區,正常運作)
- $h_{\min} \to 0$ → danger → 1(boundary,全力閃)
- $h_{\min} < 0$ → danger = 1(已踩線,emergency)

### 4.3 Scale 公式

$$Q_{\mathrm{eff}} = Q \cdot \big(1 - (1 - q_{\min}) \cdot \mathrm{danger}\big)$$

$$\rho_{\mathrm{eff}} = \rho \cdot \big(1 + (\rho_{\max} - 1) \cdot \mathrm{danger}\big)$$

| Danger | Q scale | Slack scale | 意義 |
|--------|---------|-------------|------|
| 0 (安全) | 1.0 | 1.0 | 正常追蹤 + 正常 slack |
| 0.5 | 0.85 | 3.0 | 略微讓步 |
| 1.0 (boundary) | $q_{\min}$ = 0.7 | $\rho_{\max}$ = 5.0 | 保留 70% tracking + 5x 硬 slack |

### 4.4 程式對應
`gmpc.GMPC.solve` 第 4 步,計算 `danger`、`Q_scale`、`slack_weight_eff`。

---

## 5. Reference Path 處理

### 5.1 訊號流

Nav2 `planner_server` 出 `nav_msgs/Path`(map frame),
`goal_to_plan_relay` 連續 replan(1 Hz)維持 fresh path,
`gmpc_node.path_processor` 把它轉成 GMPC 需要的 reference window。

### 5.2 Arclength 採樣

`path_processor.build_reference_window`:
1. 將 path 轉成 cumulative arclength $s$
2. 找 robot 投影點 $s_{\mathrm{start}}$
3. 從 $s_{\mathrm{start}}$ 開始,以間距 $v_{\mathrm{nominal}} \cdot \Delta t$ 採 N+1 個點
4. 每個樣本 yaw 用**路徑切線方向**(`atan2(dy, dx)`)— **不**用 planner 給的 quaternion(NavFn 會給 identity)

### 5.3 Reference twist 計算

每對相鄰樣本 $(X_k, X_{k+1})$:

$$\xi_{\mathrm{ref}}(k) = \frac{\log(X_k^{-1} X_{k+1})^\vee}{\Delta t}$$

這保證 reference 與 SE(2) 運動學自洽。

---

## 6. Obstacle 感知 (Ground Truth)

### 6.1 來源

`obstacle_aggregator` 對每個 `dyn_obs_X` 同時訂閱兩個 Gazebo topic:

| Topic | 用途 |
|-------|------|
| `/model/dyn_obs_X/pose` | 障礙物實際位置 (x, y) |
| `/model/dyn_obs_X/cmd_vel` | 障礙物**真實速度** (vx, vy) |

### 6.2 為什麼 cmd_vel 是 GT velocity

`random_room_dynamic.sdf` 中圓柱設定 `<kinematic>true</kinematic>` + `<gravity>false</gravity>`:
- 不受物理動力學影響
- `dynamic_obstacle_driver` 發布的 `cmd_vel` **就是當下實際速度**(沒延遲)
- 在端點反向時,cmd_vel 立刻切換 → 0 延遲

### 6.3 Fallback

如果 `cmd_vel` 還沒到(剛啟動),用 5-sample LSQ 從 pose 歷史估算速度。`cmd_vel` 一旦到了就切換為 GT。

### 6.4 對外訊息

`/gmpc/obstacles` (`std_msgs/Float32MultiArray`),flat 格式:

```
[x1, y1, r1, vx1, vy1,
 x2, y2, r2, vx2, vy2,
 ...]
```

長度 = 5 × n_obstacles。

---

## 7. ROS2 節點與訊息流

### 7.1 GMPC Node (`gmpc_node.py`)

| 接 | 來自 | 用途 |
|----|------|------|
| `/plan` | planner_server(透過 relay) | reference path |
| `/gmpc/obstacles` | obstacle_aggregator | CBF 用 |
| TF map→base_footprint | AMCL + omni_drive | X_now |

| 發 | 訊息 | 用途 |
|----|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 命令底盤 |
| `/gmpc/solve_time_ms` | `Float32` | 診斷:OSQP solve time |
| `/gmpc/min_h` | `Float32` | 診斷:CBF barrier |
| `/gmpc/cbf_zones` | `MarkerArray` | Foxglove 視覺化 CBF 安全圈 |

### 7.2 Goal-to-Plan Relay

訂閱 `/goal_pose` → 呼叫 `compute_path_to_pose` action → 取回 path → 重發 `/plan`。
**continuous replan**:1 Hz 重複請求,持續刷新 path。
goal_tolerance 內停止 replan。

### 7.3 Obstacle Aggregator

對 yaml 中每個 obstacle:
- 訂 `/model/<name>/pose` 維護最新位置
- 訂 `/model/<name>/cmd_vel` 取 GT velocity
- 20 Hz 重發 `/gmpc/obstacles`

---

## 8. QP 矩陣總組裝

最終 QP(送給 OSQP):

$$\min_z \; \frac{1}{2} z^\top P z + q^\top z$$
$$\text{s.t.} \quad l \le A z \le u$$

### 8.1 決策變數

$$z = \underbrace{[\delta\xi_0; \delta\xi_1; \ldots; \delta\xi_{N-1}]}_{3N} \cup \underbrace{[\varepsilon_1; \varepsilon_2; \ldots; \varepsilon_{N-1}]}_{N-1}$$

維度:$3N + (N-1) = 3 \cdot 20 + 19 = 79$

### 8.2 P 矩陣結構

$$P = \begin{bmatrix} P_{\delta\xi} & 0 \\ 0 & 2 \rho_{\mathrm{eff}} I \end{bmatrix}$$

其中:
- $P_{\delta\xi} = 2(\Gamma^\top \bar Q_{\mathrm{eff}} \Gamma + \bar R) \in \mathbb{R}^{3N \times 3N}$
- $\bar Q_{\mathrm{eff}}$ 是 Q_scale × Q 的 block diagonal,terminal 用 Q_f scale
- 對角的 $\rho_{\mathrm{eff}}$ 是 gain scheduling 後的 slack penalty

### 8.3 約束矩陣 $A$ 結構

$$A = \begin{bmatrix} A_{\mathrm{vel}} & 0 \\ A_{\mathrm{acc}} & 0 \\ A_{\mathrm{cbf}} & C_{\mathrm{slack}} \\ 0 & I_{N-1} \end{bmatrix}$$

| Block | 行數 | 描述 |
|-------|------|------|
| Velocity box | $3N = 60$ | 速度限制(逐 step) |
| Acceleration box | $3N = 60$ | 加速度限制(相鄰差分) |
| CBF rows | $n_{\mathrm{obs}} \cdot N = $ 60 (3 obs × 20 steps) | 每 obstacle 每 step 一條;step 0 沒 slack |
| Slack non-neg | $N - 1 = 19$ | $\varepsilon_k \ge 0$ |

總約束行數:60 + 60 + 60 + 19 = **199**

### 8.4 OSQP 設定

```python
prob.setup(P, q, A, l, u,
           verbose=False,
           eps_abs=1e-6, eps_rel=1e-6,
           polish=False, max_iter=4000)
```

每次 control step 從零 setup(目前沒做 warm start)。

### 8.5 Fallback

OSQP 回傳 `status != 'solved'` 或 `'solved inaccurate'` → **safer emergency brake**:
強制 $u_0 = 0$(不是 fallback 到 $\xi_{\mathrm{ref}}$,否則會撞)。

---

## 9. 所有可調參數一覽

### 9.1 Sample timing

| 參數 | 值 | 說明 |
|------|------|------|
| `control_frequency` | 20 Hz | dt = 0.05s |
| `horizon` (N) | 20 steps | 1.0s lookahead |
| `v_nominal` | **0.18** m/s | reference 巡航速度 |

### 9.2 Chassis limits (對齊 MPPI 公平比較)

| | min | max | a_max |
|---|---|---|---|
| vx | -0.20 | 0.35 | 0.8 |
| vy | -0.25 | 0.25 | 0.6 |
| wz | -0.80 | 0.80 | 1.2 |

### 9.3 GMPC cost weights

| | vx | vy | yaw |
|---|---|---|---|
| Q (running state) | 10 | 10 | 5 |
| R (input deviation) | 3.0 | 3.0 | 1.5 |
| Qf multiplier | 5 × Q | | |

### 9.4 CBF

| 參數 | 值 | 說明 |
|------|------|------|
| `cbf_alpha` | 5.0 | $\alpha$ in $\dot h + \alpha h \ge 0$ |
| `cbf_safe_margin` | 0.40 m | $d_{\mathrm{safe}}$ |
| `cbf_slack_weight` | 1.0e4 | $\rho$ 基準值 |

### 9.5 Gain scheduling

| 參數 | 值 | 說明 |
|------|------|------|
| `cbf_danger_thresh` | 0.2 | $\eta_{\mathrm{thresh}}$ — h 低於此才縮 |
| `cbf_Q_min_scale` | 0.7 | $q_{\min}$ — 危險時 Q 最低係數 |
| `cbf_slack_max_scale` | 5.0 | $\rho_{\max}$ — 危險時 slack 最大係數 |

### 9.6 Goal / replan

| 參數 | 值 | 說明 |
|------|------|------|
| `goal_tolerance_xy` | 0.20 m | gmpc_node 認到達 |
| `replan_period` | 1.0 s | relay 重 plan |
| `replan_goal_tolerance` | 0.30 m | relay 停止 replan |
| `tf_timeout` | 0.10 s | TF lookup 上限 |

---

## 10. 程式碼對應索引

| 主題 | 檔案 |
|------|------|
| SE(2) Lie 工具 + self-test | [`se2.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/se2.py) |
| GMPC QP 建構 + horizon CBF + gain scheduling | [`gmpc.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py) |
| Path → reference window | [`path_processor.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/path_processor.py) |
| ROS2 控制節點 | [`gmpc_node.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py) |
| Goal → plan bridge + continuous replan | [`goal_to_plan_relay.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/goal_to_plan_relay.py) |
| Obstacle position + GT velocity 聚合 | [`obstacle_aggregator.py`](src/ammr_wholebody_mpc/ammr_wholebody_mpc/obstacle_aggregator.py) |
| GMPC 預設參數 | [`gmpc_params.yaml`](src/ammr_wholebody_mpc/config/gmpc_params.yaml) |
| Nav2 整合 (CBF + 正常 planner) | [`gmpc_nav2_cbf.launch.py`](src/ammr_wholebody_mpc/launch/gmpc_nav2_cbf.launch.py) |
| Nav2 整合 (CBF + blind planner) | [`gmpc_nav2_cbf_blind.launch.py`](src/ammr_wholebody_mpc/launch/gmpc_nav2_cbf_blind.launch.py) |
| Gazebo 動態場景 | [`gazebo_dynamic.launch.py`](src/ammr_bringup/launch/gazebo_dynamic.launch.py) |
| 動態障礙物軌跡 | [`dynamic_trajectories.yaml`](src/ammr_bringup/config/dynamic_trajectories.yaml) |
| Bag → CSV 分析 | [`evaluation/analyze.py`](evaluation/analyze.py) |
| CSV → 對比圖 | [`evaluation/plot.py`](evaluation/plot.py) |

---

## 11. 已知局限

### 11.1 等速度預測 失準
- CBF 假設 $o_i(k) = o_i(0) + v_i \cdot k\Delta t$
- 但 ping-pong obstacle 在端點瞬間反向
- 即使 GT velocity 0 lag,**反向那一刻的預測**仍然是「往舊方向繼續」
- 結果:robot 朝錯方向閃 → 撞

**緩解**:Kalman Filter / IMM 預測(下次工作)。

### 11.2 多 obstacle 衝突
- 密集 obstacle 場景,多條 CBF 行同時 binding
- QP 找最小違反妥協 → robot 卡在邊界
- Gain scheduling 緩解但沒徹底解決

### 11.3 Single-Pass linearization
- 在 step $k \ge 1$,我們用 $X_{\mathrm{ref}}(k)$ 當 linearization point
- 實際 robot 軌跡 ≠ $X_{\mathrm{ref}}$(因為 CBF 偏移),預測有誤
- **SQP 迭代**可改善,但計算成本翻倍

### 11.4 No warm start
- 每 step OSQP 從零 setup,沒利用上 step 的解
- solve_time 比 warm start 慢 ~3-5x
- 工程優化空間

### 11.5 動態自由空間表現劣於 MPPI
- MPPI 2000 條 rollouts 隱含遍歷未來不確定性
- 我們 1 條預測,model error 直接傳到避障
- **這是 GMPC + CBF 的本質適用域問題,不是 bug**

---

## 12. Cite 文獻

寫論文方法章節時必引:

| 主題 | 文獻 |
|------|------|
| SE(2) Lie 工具(hat/vee/exp/log/Ad/ad) | **Solà, Deray, Atchuthan 2018** *"A micro Lie theory for state estimation in robotics"* arXiv:1812.01537 |
| Body twist 約定 + Adjoint | **Lynch & Park 2017** *Modern Robotics*, §3.3, §8.2 |
| SE(2) GMPC 數學框架(eq 17 改良線性化) | **Tang, Wu, Lan, Dong, Jin, Tian, Zhang, Shi 2024** *"GMPC: Geometric Model Predictive Control for Wheeled Mobile Robot Trajectory Tracking"* IEEE RA-L |
| Lie-group error dynamics 模板 | **Bullo & Lewis 2005** *Geometric Control of Mechanical Systems* |
| Condensed-form linear MPC | **Borrelli, Bemporad, Morari 2017** *Predictive Control for Linear and Hybrid Systems* §11.3 |
| CBF 基礎 + safety filter | **Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada 2017** *"Control Barrier Function Based Quadratic Programs for Safety Critical Systems"* IEEE TAC |
| CBF + slack (soft safety) | **Ames 2017** 同上 §V.B |
| Discrete-time horizon CBF | **Zeng, Zhang, Sreenath 2021** *"Safety-Critical MPC with Discrete-time CBF"* ACC |
| OSQP 求解器 | **Stellato, Banjac, Goulart, Bemporad, Boyd 2020** *"OSQP: An Operator Splitting Solver for Quadratic Programs"* Mathematical Programming Computation |

---

## 摘要句(口頭 / Abstract 用)

> 「我們將 SE(2) Geometric MPC 整合進 ROS2 Nav2 stack,以全向移動底盤的 holonomic 控制為核心,加入 **full-horizon Control Barrier Function** 作為安全濾波層。CBF 對每個障礙物在預測時域內逐步 enforce 安全裕量(`step 0` 為硬約束,`k ≥ 1` 為軟約束加 slack),並透過 **danger-aware gain scheduling** 在靠近邊界時動態切換『偏好跟蹤』與『偏好避障』的權重。Obstacle 速度由 Gazebo 真值取得以排除估測延遲。整體 QP 經條件式形式由 OSQP 求解,當前實作在靜態 + 平滑可預測動態場景驗證有效;在 ping-pong 反轉動態場景受常速度預測模型限制,效能不及 MPPI,此即下一步全 Kalman Filter / Neural Potential Field 的工作動機。」

---

*更新於 2026-06-02 工程進度結束。下次更新請在實作 Phase 3 受約束接觸操作或加入 EKF 預測模型後。*
