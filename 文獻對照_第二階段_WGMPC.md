# 第二階段：W-GMPC 實作 ↔ 文獻對照表

> **範圍**：第二階段（手臂／全身）中，第六次進度報告實際報告的 W-GMPC 實作與三類比較：
> 增廣狀態模型、多步成本與 SQP／QP 求解、延遲與偏移補償、完整移動中抽屜操作（MotM）、
> 嚴格停車對照、H1/H5 時域消融、B1/P 架構比較。附錄另列不在第六次簡報展開的 D1 腕部視覺。
>
> **與其他文件的分工**：
> - 第一階段（導航、SE(2) GMPC、CBF、Shield）→ [文獻對照.md](文獻對照.md)。
> - 舊第二階段草稿（全身安全濾波、連桿幾何、手臂力矩、自我濾除）與 RL 殘差 →
>   [文獻對照_第二階段.md](文獻對照_第二階段.md)。**那些模組不在第六次正式比較的抽屜管線內**
>   （見下方「先確認」），本文件不重複。
>
> **依據**：`src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core.py`、`wgmpc_core_sp.py`、
> `evaluation/wgmpc_wg2_node.py`、`evaluation/offset_moving.py`、`evaluation/wg4b_b1_core.py`、
> `evaluation/results/specs/wgmpc_wg0_problem_spec.yaml`，以及各批次正式結果檔與 `align_solver.json` 的 `args`。
> 整理日期 2026-10-05；程式版本以 commit `cc43b8c88` 的工作樹為準。
>
> 每節格式：**實際公式與參數** → **文獻** → **關係** → **可主張／不可主張**。
> 判準沿用 [文獻對照.md](文獻對照.md)：參數對照不等於方法等價；機制不同不借用方法名稱；工程防護不寫成已證明的保證。

## 驗證標記

- **✓ 書目已核對**：2026-10-05 以網路查核題名、作者、出處與卷期頁（來源列於文末）。**不表示已逐條比對原文公式**。
- **○ 背景**：標準教科書或經典文獻，書目為常用引用格式，本次未逐頁核對；只支持方法背景，不作特定實作或效果的證據。
- **第一階段已核對**：沿用 [文獻對照.md](文獻對照.md) BibTeX 與查核報告的結果。
- 本文件**沒有任何一節**宣稱「公式已對原文逐條核對」；寫進論文前，核心方法引用（§2、§4、§5、§8、§9、§12、§14）應各自補一次原文核對。

---

## 先確認：各正式批次實際開了什麼

依各批次代表趟的 `align_solver.json` → `args`（C0：`c0b1_p1_*`；WG4-B：`wg4b_f01_B`／`wg4b_f02_P`；
嚴格停車：`phf_01_M`／`phf_02_H`）。

| 功能 | 參數 | C0（H1/H5） | WG4-B P | WG4-B B1 | 嚴格停車 M／H |
|---|---|---|---|---|---|
| 手臂設定點增廣模型 | `arm_model` | setpoint | setpoint | **不用**（單步運動學） | setpoint |
| 預測步數 | `N` | 1／5 | 5 | 單步（`N` 參數不作用） | 5 |
| 偏移估計（靜態閘門） | `offset_free`、`offset_init_static` | 開 | 開 | **節點跳過**（`not self._b1`） | 開 |
| 運動中偏移觀測（v2.1） | `offset_moving` | 未開 | 未開 | — | **開（兩組）** |
| 延遲補償 | `delay_comp_state/cmd_cycles` | 1.6／1.0 | 1.6／1.0 | **節點跳過** | 1.6／1.0 |
| 以實際套用值預測 | `use_applied_for_predict` | 關 | 關 | — | 關 |
| 近目標輸出整形 | `near_target_gamma` | 1.0 | 1.0 | 1.0 | M 1.0；**H 停車保持期停用** |
| 列縮放 | `no_row_scaling` | 否（開縮放） | 否 | — | 否 |
| 停車模式 | `park_hold` | — | — | — | H：開 |

**程式裡有、但第六次正式比較的抽屜管線沒有用的**（寫論文時不要寫成系統的一部分）：

- 全身速度安全濾波、連桿級 barrier 列（`wholebody_safety_filter.py`、`arm_link_distance.py`）：
  WG0 規格明訂「v1 **完全沒有**避碰列」；運行器紀錄「**CBF 關閉**」。抽屜管線的保護是命令鏈三層
  （低速介面界限 → 輪級 λ → 底盤變化率上限），**不是碰撞安全保證**。
- RL 殘差：第四次進度工作，未接入。
- 腕部 RGB-D：第六次簡報不展開（附錄 D1）。控制目標一律使用模擬真值。

---

# 第一部分：模型

## 1. 全身運動學與單一雅可比

**實作**（`wholebody_kinematics.py`；第六次簡報第 5 頁）

```
q = [x, y, θ, q1..q6] ∈ SE(2)×R⁶（9 個廣義座標）
J(q) = ∂(TCP)/∂q ∈ R^{6×9}；底盤三個平面自由度當成同一運動鏈上的關節
u = [v_x^B, v_y^B, ω, q̇1..q̇6]（底盤為本體座標），J_b = J_w·B(θ)
```

**文獻**
- ○ Bayle, Fourquet, Renaud, "Manipulability of wheeled mobile manipulators: Application to motion generation", *IJRR* 22(7–8):565–581, 2003 — 輪式行動機械臂把底盤與手臂合成單一雅可比。
- ○ Siciliano, Sciavicco, Villani, Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009, ch. 3 — 幾何雅可比推導。

**關係**：合成雅可比是既有做法，**不是貢獻**。本系統的工程特點是 URDF 為唯一幾何來源，並有獨立驗證程式。

**可主張**：模型與執行端使用同一份 URDF 幾何，且已交叉驗證。**不可主張**：新的全身運動學方法。

---

## 2. 手臂設定點增廣狀態與控制步組合 ★

**實作**（`wgmpc_core_sp.py` 文件字串；第 5 頁、備用 A）

```
每物理步（dt_p = 10 ms）：  x⁺ = x + α⊙(s − x) + b,   s⁺ = s + u_a·dt_p
連乘 kp = dt_c/dt_p = 5 次（dt_c = 50 ms）：
  x_next = P⊙x + Q⊙s + G⊙u_a + h,   s_next = s + u_a·dt_c
  β = 1 − α, P = β^kp, Q = 1 − β^kp, G = α·dt_p·Σ_{j=0}^{kp−1} j·β^{kp−1−j}, h = b(1 − β^kp)/α
z = [q(9); s(6)] ∈ R¹⁵，控制輸入仍為 9 維
```

離線辨識（free4，前半估、後半驗）：每物理步 α ≈ 0.09501（六軸一致），等效時間常數約 0.100 s；
50 ms 預測最差關節 RMSE：理想速度積分 27.9 mrad、設定點追蹤模型 0.44 mrad。
α = 0.09501、kp = 5 時 G = 0.00864，與 dt_c = 0.05 相差 5.8 倍。

**文獻**
- ○ Maciejowski, *Predictive Control with Constraints*, Prentice Hall, 2002 — 以狀態增廣納入致動器或輸入積分動態、Δu 形式的預測模型。
- ○ Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill, 2017 — 離散時間預測模型與取樣。
- ○ Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999 — 估計／驗證資料切分、一階模型辨識。

**關係**：「把致動器內部狀態增廣進預測模型」是標準建模手法。本系統的具體內容是：辨識到的每物理步係數；控制步閉式組合（不可直接把 α 當成 50 ms 係數）；以及量化比較，說明理想速度積分會高估命令效果。

**可主張**：
- 本平台的手臂命令鏈（速率 → 設定點積分 → 位置控制器追蹤）在預測中必須表示；增廣後 50 ms 預測誤差降約兩個數量級（單一辨識資料集）。
- 手臂區塊對 (x, s, u) 嚴格線性，因此在仿射模型中沒有線性化誤差；底盤區塊仍需線性化。

**不可主張**：
- 剛體動力學或接觸力預測模型；這只是一階設定點追蹤模型。
- 辨識範圍外的泛化。
- 「設定點是額外自由度」：設定點只由 u 驅動，沒有自己的輸入。

---

## 3. 任務誤差與 SO(3) 對數映射的微分修正

**實作**（`wgmpc_core.py: task_error / task_error_jacobian`）

```
e = [p_des − p(q) ;  log(R(q)ᵀ R_des)^∨]          位置在世界座標、姿態在本體座標
H = ∂e/∂q = [ −J_p ;  −J_l(e_r)^{-1} Rᵀ J_ω ]
```

註解記錄：少了 `J_l^{-1}` 時，30° 誤差下約有 20% 的偏差（WG0 實測）。

**文獻**
- ○ Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics", arXiv:1812.01537, 2018 — SO(3) 的 log、左右雅可比與其逆。
- 第一階段已核對：Tang et al., "GMPC: Geometric Model Predictive Control for Wheeled Mobile Robot Trajectory Tracking", arXiv:2403.07317, 2024；Teng et al., "An Error-State MPC on Connected Matrix Lie Groups for Legged Robot Control", IROS 2022 — 李群誤差狀態 MPC。

**關係**：只有 TCP 姿態誤差使用 SO(3) 對數與其微分修正，**不是整個系統的李群誤差狀態 MPC**；底盤在本模型中是平面關節座標。

**可主張**：姿態誤差的線性化包含 log 微分修正，並有量化理由。**不可主張**：繼承 Teng 或 Tang 的誤差狀態 MPC 架構或其理論性質。

---

# 第二部分：最佳化與求解

## 4. 多步成本：追蹤、終端權重、命令與增量、整機協同 ★

**實作**（`wgmpc_core.py: nonlinear_cost`、`wgmpc_core_sp.py: coord_cost`；第 6 頁）

```
J = Σ_{k=1}^{N} e_kᵀ W_k e_k  +  Σ_{k=0}^{N−1} ( u_kᵀ R u_k + (u_k − u_{k−1})ᵀ S (u_k − u_{k−1}) )  +  J_coord
W_k = Q（k < N），W_N = Qf_scale·Q（Qf_scale = 5）；u_{−1} = u_prev（上一輪實際命令）
J_coord = w_vref Σ_k ‖(u_k,base − v_ref) ⊘ v_max,base‖²  +  w_qn Σ_k ‖q_arm,k − q_nom‖²
```

協同兩項只在增廣核心中實作；原核心遇到非零權重時會拒絕求解。MotM 任務參數為 `--motm-w-qn 0.8 --motm-w-vref-pre 0.3`。
**每輪預測窗內使用同一個 TCP 目標**，不是未來目標軌跡序列。

**文獻**
- ○ Rawlings, Mayne, Diehl 2017；○ Maciejowski 2002 — 追蹤成本、終端成本、Δu 懲罰的標準形式。
- ○ Mayne, Rawlings, Rao, Scokaert, "Constrained model predictive control: Stability and optimality", *Automatica* 36(6):789–814, 2000 — 終端成本、終端集合與穩定性條件。
- ✓ Minniti, Farshidian, Grandia, Hutter, "Whole-Body MPC for a Dynamically Stable Mobile Manipulator", *IEEE RA-L* 4(4):3687–3694, 2019 — 行動機械臂的全身 MPC。

**關係**：
- 成本結構是標準 MPC 形式。終端權重 ×5 是**工程設定**，不是由終端集合或 Lyapunov 條件推得。
- 協同兩項是本系統的任務層設計：底盤照外部速度剖面移動，手臂吸收短期差異，名目姿態避免手臂一路伸到關節餘量。
- Minniti 2019 處理的是含動力學與平衡限制的全身 MPC，與本系統（運動學加一階設定點模型、無平衡限制）機制不同，只能作為相關工作。

**可主張**：在同一最佳化問題中同時決定底盤與手臂命令，並以協同項分擔任務。**不可主張**：
- 閉迴路穩定性或遞迴可行性保證（沒有終端集合，也沒有證明）。
- 「最佳分工」。
- 預測式接觸力控制。

---

## 5. 序列二次規劃（SQP）與信賴域

**實作**（`wgmpc_wg0_problem_spec.yaml: solve_flow`；`WGMPCConfig`）

```
每輪：單一狀態快照 → u^nom = shift(上輪解) → 非線性 rollout → 每個預測步重算 A_k、B_k、c_k、H_k
     → QP（加逐軸 |u_k − u^nom_k| ≤ Δ）→ 以非線性 rollout 的真實成本 J_nl 判定接受
接受：ΔJ = J_nl(cand) − J_nl(nom) ≤ tol_accept（含等號）⇒ Δ ← min(γ_up·Δ, Δ_max)
拒絕：Δ ← γ_dn·Δ 後重試；至多 n_sqp = 10 次
Δ_0 = 0.2, Δ_max = 1.0, Δ_min_conv = 0.01, γ_up = 2, γ_dn = 0.5
五種結果分開回報：converged_step／converged_cost／no_progress／trust_region_exhausted／iter_limit（＋qp_failed）
```

**文獻**
- ○ Nocedal & Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, ch. 4（信賴域）、ch. 18（SQP）。
- ✓ Diehl, Bock, Schlöder, "A real-time iteration scheme for nonlinear optimization in optimal feedback control", *SIAM J. Control Optim.* 43(5):1714–1736, 2005 — 即時迭代（RTI）與 shift 初始化。

**關係**：
- 屬於 SQP 加信賴域的一種簡化實作。**接受規則不是標準的實際／預測下降比率 ρ 檢定**，而是「非線性成本不增加即接受」；這是為了讓已在最佳點的零更新不被誤判為失敗。
- 以上輪解 shift 當初始猜測，與 RTI 的初始化相同。但本系統每輪最多迭代 10 次，並不是 RTI 的「每取樣一次迭代」，**不可稱為 RTI**。

**可主張**：在真實非線性成本下做步長接受，並分開回報收斂與非收斂狀態（不把 iter_limit 當成收斂）。**不可主張**：SQP 收斂保證、RTI 的收縮性結果。

---

## 6. OSQP 與等價列縮放

**實作**（`wgmpc_core_sp.py` 文件字串）

```
每列 l_i ≤ A_i U ≤ h_i 同乘正數 d_i（精確算術下可行集合不變）
接受判定與殘差核對一律使用未縮放的 A、lo、hi；只接受 status = solved
```

動機：`measured_position` 列的係數 G ≈ 0.00864，欄正規化後係數跨度達 115.8 倍，OSQP 在限位附近會打到迭代上限。

**文獻**
- 第一階段已核對：Stellato, Banjac, Goulart, Bemporad, Boyd, "OSQP: an operator splitting solver for quadratic programs", *Math. Program. Comput.* 12(4):637–672, 2020（含其內建的 Ruiz 型等化縮放）。
- ○ Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices", Tech. Rep. RAL-TR-2001-034, Rutherford Appleton Laboratory, 2001。

**關係**：是求解前的對角預條件，與 OSQP 內部的 Ruiz 型縮放同屬等化概念，但是另外加在外層。

**可主張**：縮放不改限制內容，且最終以原單位核對殘差。**不可主張**：浮點運算下完全等價（只在精確算術下成立）。

---

## 7. 限制：速度／加速度框、雙重關節限位、輪級

**實作**：逐軸速度框（底盤線速度 0.035255 m/s、角速度 0.1999 rad/s；手臂 0.9999 rad/s）；
加速度框（底盤 0.5 m/s²、2.0 rad/s² 為**開發值**，無已核准來源；手臂 19.984 rad/s²）；
關節限位同時約束**預測設定點**與**預測實測角**，各留 `joint_margin` 0.05 rad；輪速與輪加速度在 QP 之內。

兩組限位都保留，理由是反例：含偏置 b 時，穩態偏移為 b/α 而不是 b。
以 joint2 為例（α = 0.09504、b = 2.715e-4）：設定點界成立，但五個物理步後實測角超出限位 0.623 mrad。

**文獻**：
- 輪級限制沿用第一階段（[文獻對照.md](文獻對照.md) §1–2）。
- 手臂限值來源見舊稿 [§21](文獻對照_第二階段.md)：UFACTORY Lite 6 手冊，部分數值為本專案實驗測定。

**可主張**：執行端真正會硬失效的量（設定點）與實際關節都納入限制，並有反例說明兩者不可互相取代。
**不可主張**：底盤加速度上限有原廠依據；碰撞安全。

---

# 第三部分：執行、延遲與偏移

## 8. 延遲補償：把量測狀態推到命令生效時刻

**實作**（`wgmpc_wg2_node.py: _predict_delay`）

```
量測 ──D_pub──> 發布 ──D_cmd──> 生效；  D_state = D_pub + D_cmd
以增廣模型逐物理步前推 n = round(D_state/dt_p) 步；時刻 τ 作用的命令取 τ − D_cmd 時已發布的那一筆
rec9 逐筆量測：D_pub p50 0.60、D_cmd p50 1.00 週期 ⇒ 設定 d_state = 1.6、d_cmd = 1.0
```

延遲只改變求解的**起點狀態**，不改權重、預測窗或限制；B1 不做延遲補償。

**文獻**
- ✓ Smith, "Closer control of loops with dead time", *Chem. Eng. Prog.* 53(5):217–219, 1957 — 以模型預估補償回授延遲（Smith 預估器）。
- ✓ Findeisen & Allgöwer, "Computational delay in nonlinear model predictive control", *IFAC Proc. Vol.* 37:427–432, 2004（ADCHEM 2004）— 在 NMPC 中以預測處理計算延遲，並給出穩定性條件。

**關係**：同屬「以模型預測補償延遲」的概念。本系統做的是前推初始狀態，**不是 Smith 預估器的回授結構**；也沒有驗證 Findeisen & Allgöwer 的穩定性條件。具體特點在於延遲分成兩段**實測**，並拆開使用。

**可主張**：延遲是逐筆量測後設定，且拆分兩段的必要性有離線重跑佐證。**不可主張**：Smith 預估器、延遲系統穩定性保證。

---

## 9. 偏移估計（手臂穩態下垂）

**實作**（`wgmpc_wg2_node.py` 約 L1145–1185）

```
只在設定點近乎靜止時更新（max|s − s_prev| < offset_gate 0.002 rad）：
  d̂ ← d̂ + (dt/τ)·((q_arm − s) − d̂)，τ = 0.5 s，限幅 |d̂| ≤ 0.05 rad
模型偏置 b = α ⊙ d̂ ⇒ 模型穩態 q = s + d̂
--offset-init-static：首輪以 (q − s) 作初值（啟動時設定點靜止）
離線重現：有下垂而模型不知 ⇒ 平台 10.88 mm；加估計 ⇒ 0.02 mm
```

**文獻**
- ✓ Pannocchia & Rawlings, "Disturbance models for offset-free model-predictive control", *AIChE J.* 49(2):426–437, 2003。
- ✓ Muske & Badgwell, "Disturbance modeling for offset-free linear model predictive control", *J. Process Control* 12(5):617–632, 2002。

**關係**：目標相同，都是以擾動狀態吸收模型與實體的落差、消除穩態偏移。但形式不同：本系統是**有閘門的一階低通估計**，不是將積分擾動增廣進觀測器（例如 Kalman 或 Luenberger）的設計，也沒有檢查文獻中的偵測性與 offset-free 條件。

**可主張**：「以擾動估計吸收手臂穩態下垂」的做法與 offset-free MPC 的思路一致，並有離線量化效果。**不可主張**：offset-free 定理的零偏移保證；不可稱為「disturbance observer」或「offset-free MPC」而不加限定。

---

## 10. 運動中偏移觀測（v2.1）

**實作**（`offset_moving.py`；只在嚴格停車批次開啟，兩組都開）

```
等速穩態下 q − s = d − (dt_p/α)·ṡ ⇒ d_obs = (q − s) + (dt_p/α)·ṡ（係數約 0.105 s）
ṡ 由同步量測的設定點與實際時間差計算；窗口 0.30 s 內各軸 |ṡ − mean| ≤ 0.01 rad/s 且樣本 ≥ 5 才給觀測
```

**文獻**：無直接文獻。這是由 §2 模型推得的觀測式；有合成資料測試（`test_offset_moving.py` 12/12）。

**可主張**：模型一致的觀測式推導，以及它在功能確認中恢復雙指承載（n = 1）。**不可主張**：偏移估計就是單指承載的根因（沒有做排除其他介入的實驗）。

---

## 11. 時槽與命令鏈（工程，不另引文獻）

- 求解時槽以**模擬時鐘**排程（`slot_basis = simulation_clock`、名目 0.05 s），記錄漏時槽、重複跳過與暖啟動丟棄。
  RTF 不足時控制週期仍對齊模擬時間，**不代表牆鐘即時性**。
- 執行端命令鏈三層：低速介面界限（軟體保守值、閂鎖）→ 輪級 λ 限制 → 底盤變化率上限。
- 近目標輸出整形（γ、deadband）：工程處理，不以文獻方法命名。

這些放在論文的 implementation 小節，寫成實作細節，不寫成方法貢獻。

---

# 第四部分：任務與比較

## 12. 移動中操作（MotM）★

**實作**（第 7 頁）：導航 → 減速段（glide 節點）→ 滾動交棒（切換前預核全身命令，切換當步核對速度框）→
展開（軌跡節點）→ ALIGN 起由 W-GMPC 控制接近、夾持、開關、放開、退開 → 關節空間收臂 → 交還導航返回。
接近與夾持時底盤持續低速移動，開關時由協同項分擔。

**文獻**
- ✓ Burgess-Limerick, Lehnert, Leitner, Corke, "An Architecture for Reactive Mobile Manipulation On-The-Move", ICRA 2023, pp. 1623–1629 — 底盤不停的移動中抓取架構，回報任務時間最多減少 48%。
- ✓ Haviland, Sünderhauf, Corke, "A Holistic Approach to Reactive Mobile Manipulation", *IEEE RA-L* 7(2):3122–3129, 2022 — 把底盤與手臂視為一體的反應式 QP 控制。
- ✓ Minniti et al. 2019（§4）— 全身 MPC。

**關係**：與 Burgess-Limerick 2023、Haviland 2022 **動機相同**（不先停車、整機協同可縮短時間），但**機制不同**：
- 他們用反應式 QP 控制器處理抓取或取放。
- 本系統用多步 W-GMPC 加協同項，處理接觸式抽屜開關（摩擦夾持，被動滑軌）。

不可把他們的時間收益或成功率數字當作本系統的證據。

**可主張**：在模擬中完成含摩擦夾持的移動中抽屜開關與返回。**不可主張**：
- 移動中操作的普遍時間收益（本系統只有一個任務、一個配置）。
- 「永不停車」：保持與反轉階段允許近零速度。
- 實機或視覺控制成果。

---

## 13. 抽屜／鉸接物件操作

**文獻**
- ✓ Jain & Kemp, "Pulling open doors and drawers: Coordinating an omni-directional base and a compliant arm with Equilibrium Point control", ICRA 2010, pp. 1807–1814 — 全向底盤加柔順手臂開門與抽屜，不需事先的機構模型。

**關係**：同為全向底盤加手臂開抽屜，但條件不同：
- Jain & Kemp 2010 是實機、未知機構、平衡點控制（柔順）。
- 本系統是模擬、**已知滑軌方向與把手真值**、W-GMPC 位置／速度控制，沒有力控。

**不可主張**：未知機構估計、柔順或力控能力、真實環境的穩健性。

---

## 14. 單步 QP 基線 B1（WG4-B）

**實作**（`wg4b_b1_core.py`；第 10 頁）

```
min_u ‖J_b u − e‖² + μ²‖u_arm − v_post‖² + λ²‖u ⊘ w‖² + w_vref‖(u_base − v_ref) ⊘ v_max‖² + w_qn‖q_arm + dt·u_arm − q_nom‖²
限制全部由目前狀態建立（速度框、加速度框、實測與設定點限位、輪速、輪加速度）；OSQP 每輪冷啟動
沒有增廣動態模型（只有 q⁺ = q + dt·u）、SQP、終端權重、延遲補償與偏移估計
改編自凍結原型 wholebody_pregrasp.py --solver qp；協同權重數值沿用 P，只在 3 趟預算內調 kp（依登錄順序選 1.0）
```

**文獻**
- ✓ Whitney, "Resolved motion rate control of manipulators and human prostheses", *IEEE Trans. Man-Machine Systems* 10(2):47–53, 1969 — resolved-rate 速度控制。
- ✓ Nakamura & Hanafusa, "Inverse kinematic solutions with singularity robustness for robot manipulator control", *ASME J. Dyn. Syst. Meas. Control* 108:163–171, 1986 — 阻尼最小平方。
- ○ Wampler, "Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods", *IEEE Trans. SMC* 16(1):93–101, 1986。
- ✓ Haviland et al. 2022（§12）— 現代帶限制的 QP 型 resolved-rate 全身控制。

**關係**：B1 是「帶阻尼與限制的 QP 型 resolved-rate」基線，由本專案原型改編，**不是任何特定論文方法的重現**。

**可主張**：在本任務與事前定義的配置下，P 三趟完成，B1 三趟未通過開啟保持；B1 每輪計算較省、相鄰命令增量較小。
**不可主張**：
- 單步 QP 方法的最佳性能。
- 差異單獨來自缺少多步預測（兩者在模型、補償與成本語意上都不同）。
- 「Haviland 式控制器做不到」。

---

## 15. 時域長度消融（H1 vs H5）

**實作**：同一增廣核心，只改 `solver.N`（1 或 5）；控制步 0.05 s。
N = 1 時，唯一的任務步就是終端步（權重 ×5），協同項隨 N 累加，因此**有效成本不只是預測窗長度不同**
（`H1_H5_resolved_config.yaml` D1–D6）。

**文獻**
- ○ Mayne et al. 2000（§4）；○ Grüne & Pannek, *Nonlinear Model Predictive Control: Theory and Algorithms*, 2nd ed., Springer, 2017 — 預測窗長度與閉迴路性質的理論分析。

**關係**：文獻談的是預測窗長度對穩定性與次最佳性的理論影響，本系統是**經驗消融**，兩者沒有直接對應。

**可主張**：在此任務、此配置下的三對比較，H5 在整窗追蹤、命令平順、停頓與夾持穩定指標上一致較好，H1 核心計算較省。
**不可主張**：
- 「純前瞻機制」。
- 文獻中預測窗長度的理論結論在本系統成立。
- 統計顯著（n = 3，順序與熱狀態未平衡）。

---

## 16. 嚴格停車對照與實驗設計（工程定義，不另引文獻）

- **停車**的定義是全窗實測停住（線速度 ≤ 1 mm/s、相對錨點 ≤ 1 mm、偏航 ≤ 0.5°），以停車伺服保持（k = 2、合速度 ≤ 5 mm/s）；不是「命令設為零」。
- 計時從共同 GO 到獨立終端完成；三對交錯先後；n = 3，不做顯著性檢定。
- 這是**兩個完整策略**的比較（停車伺服、整形、分擔方式都不同）。入場差距中約 6 s 屬於停車入場剖面的剩餘收斂／等待區段。

**可引用作動機**：Burgess-Limerick 2023（§12），說明「不停車」在文獻中被視為省時途徑。
**不可**拿它的 48% 與本系統的 13–14% 比較，兩者任務、對照組與計時定義都不同。

---

# 附錄：D1 腕部 RGB-D 把手辨識（第二階段，但不在第六次簡報）

**實作**（`d1_handle_detect.py`）：深度點雲 → 多平面 RANSAC（由觀測決定前板）→ 前方 20–60 mm 帶 → 分群 →
固定半徑 13 mm 圓柱 RANSAC → 端點可辨識性三條件。保留集一次評估：0.4–1.0 m 距離內，L2 19/20、誤差中位 2.28 mm。

**文獻**
- ○ Fischler & Bolles, "Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography", *Commun. ACM* 24(6):381–395, 1981。
- ✓ Schnabel, Wahl, Klein, "Efficient RANSAC for Point-Cloud Shape Detection", *Computer Graphics Forum* 26(2):214–226, 2007 — 點雲中的平面、圓柱等基本形狀偵測。

**關係**：本系統是使用已知尺寸先驗（半徑、長度）的 RANSAC 幾何辨識，屬於標準做法的應用。

**可主張**：同場景、另一條接近序列上的有界保留評估。**不可主張**：DL 成果、跨物件泛化、實機 D435、可抓取性。

---

# BibTeX

```bibtex
@book{rawlings2017mpc,
  title     = {Model Predictive Control: Theory, Computation, and Design},
  author    = {Rawlings, James B. and Mayne, David Q. and Diehl, Moritz M.},
  edition   = {2nd}, publisher = {Nob Hill Publishing}, address = {Madison, WI}, year = {2017}
}
@article{mayne2000constrained,
  title   = {Constrained model predictive control: Stability and optimality},
  author  = {Mayne, David Q. and Rawlings, James B. and Rao, Christopher V. and Scokaert, Pierre O. M.},
  journal = {Automatica}, volume = {36}, number = {6}, pages = {789--814}, year = {2000}
}
@book{maciejowski2002predictive,
  title = {Predictive Control with Constraints}, author = {Maciejowski, Jan M.},
  publisher = {Prentice Hall}, year = {2002}
}
@book{grune2017nmpc,
  title = {Nonlinear Model Predictive Control: Theory and Algorithms}, author = {Gr{\"u}ne, Lars and Pannek, J{\"u}rgen},
  edition = {2nd}, publisher = {Springer}, year = {2017}
}
@book{nocedal2006numerical,
  title = {Numerical Optimization}, author = {Nocedal, Jorge and Wright, Stephen J.},
  edition = {2nd}, publisher = {Springer}, year = {2006}
}
@article{diehl2005rti,
  title   = {A real-time iteration scheme for nonlinear optimization in optimal feedback control},
  author  = {Diehl, Moritz and Bock, Hans Georg and Schl{\"o}der, Johannes P.},
  journal = {SIAM Journal on Control and Optimization}, volume = {43}, number = {5}, pages = {1714--1736}, year = {2005}
}
@techreport{ruiz2001scaling,
  title       = {A scaling algorithm to equilibrate both rows and columns norms in matrices},
  author      = {Ruiz, Daniel}, institution = {Rutherford Appleton Laboratory},
  number      = {RAL-TR-2001-034}, year = {2001}
}
@book{ljung1999system,
  title = {System Identification: Theory for the User}, author = {Ljung, Lennart},
  edition = {2nd}, publisher = {Prentice Hall}, year = {1999}
}
@article{sola2018micro,
  title   = {A micro {L}ie theory for state estimation in robotics},
  author  = {Sol{\`a}, Joan and Deray, Jeremie and Atchuthan, Dinesh},
  journal = {arXiv preprint arXiv:1812.01537}, year = {2018}
}
@article{smith1957closer,
  title   = {Closer control of loops with dead time}, author = {Smith, Otto J. M.},
  journal = {Chemical Engineering Progress}, volume = {53}, number = {5}, pages = {217--219}, year = {1957}
}
@inproceedings{findeisen2004computational,
  title     = {Computational delay in nonlinear model predictive control},
  author    = {Findeisen, Rolf and Allg{\"o}wer, Frank},
  booktitle = {IFAC Proceedings Volumes (ADCHEM 2004)}, volume = {37}, pages = {427--432}, year = {2004}
}
@article{pannocchia2003disturbance,
  title   = {Disturbance models for offset-free model-predictive control},
  author  = {Pannocchia, Gabriele and Rawlings, James B.},
  journal = {AIChE Journal}, volume = {49}, number = {2}, pages = {426--437}, year = {2003},
  doi     = {10.1002/aic.690490213}
}
@article{muske2002disturbance,
  title   = {Disturbance modeling for offset-free linear model predictive control},
  author  = {Muske, Kenneth R. and Badgwell, Thomas A.},
  journal = {Journal of Process Control}, volume = {12}, number = {5}, pages = {617--632}, year = {2002}
}
@article{minniti2019wholebody,
  title   = {Whole-Body {MPC} for a Dynamically Stable Mobile Manipulator},
  author  = {Minniti, Maria Vittoria and Farshidian, Farbod and Grandia, Ruben and Hutter, Marco},
  journal = {IEEE Robotics and Automation Letters}, volume = {4}, number = {4}, pages = {3687--3694}, year = {2019},
  doi     = {10.1109/LRA.2019.2927955}
}
@article{haviland2022holistic,
  title   = {A Holistic Approach to Reactive Mobile Manipulation},
  author  = {Haviland, Jesse and S{\"u}nderhauf, Niko and Corke, Peter},
  journal = {IEEE Robotics and Automation Letters}, volume = {7}, number = {2}, pages = {3122--3129}, year = {2022}
}
@inproceedings{burgesslimerick2023onthemove,
  title     = {An Architecture for Reactive Mobile Manipulation On-The-Move},
  author    = {Burgess-Limerick, Ben and Lehnert, Chris and Leitner, J{\"u}rgen and Corke, Peter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}, pages = {1623--1629}, year = {2023},
  doi       = {10.1109/ICRA48891.2023.10161021}
}
@inproceedings{jain2010pulling,
  title     = {Pulling open doors and drawers: Coordinating an omni-directional base and a compliant arm with Equilibrium Point control},
  author    = {Jain, Advait and Kemp, Charles C.},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}, pages = {1807--1814}, year = {2010},
  doi       = {10.1109/ROBOT.2010.5509445}
}
@article{whitney1969resolved,
  title   = {Resolved motion rate control of manipulators and human prostheses}, author = {Whitney, Daniel E.},
  journal = {IEEE Transactions on Man-Machine Systems}, volume = {10}, number = {2}, pages = {47--53}, year = {1969}
}
@article{nakamura1986inverse,
  title   = {Inverse kinematic solutions with singularity robustness for robot manipulator control},
  author  = {Nakamura, Yoshihiko and Hanafusa, Hideo},
  journal = {ASME Journal of Dynamic Systems, Measurement, and Control}, volume = {108}, pages = {163--171}, year = {1986}
}
@article{wampler1986manipulator,
  title   = {Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods},
  author  = {Wampler, Charles W.},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics}, volume = {16}, number = {1}, pages = {93--101}, year = {1986}
}
@article{bayle2003manipulability,
  title   = {Manipulability of wheeled mobile manipulators: Application to motion generation},
  author  = {Bayle, Bernard and Fourquet, Jean-Yves and Renaud, Marc},
  journal = {The International Journal of Robotics Research}, volume = {22}, number = {7--8}, pages = {565--581}, year = {2003}
}
@article{fischler1981ransac,
  title   = {Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography},
  author  = {Fischler, Martin A. and Bolles, Robert C.},
  journal = {Communications of the ACM}, volume = {24}, number = {6}, pages = {381--395}, year = {1981}
}
@article{schnabel2007efficient,
  title   = {Efficient {RANSAC} for Point-Cloud Shape Detection},
  author  = {Schnabel, Ruwen and Wahl, Roland and Klein, Reinhard},
  journal = {Computer Graphics Forum}, volume = {26}, number = {2}, pages = {214--226}, year = {2007}
}
```

第一階段已有的 `stellato2020osqp`、`tang2024gmpc`、`teng2022errorstate` 直接沿用 [文獻對照.md](文獻對照.md)。

---

# 論文使用清單

## A. 核心方法引用（寫進方法章，建議先補原文核對）

| 方法 | 引用 | 本文件 |
|---|---|---|
| 增廣致動器狀態的預測模型 | maciejowski2002predictive、rawlings2017mpc；辨識 ljung1999system | §2 |
| 多步成本、終端權重、Δu | rawlings2017mpc、maciejowski2002predictive | §4 |
| SQP 與信賴域 | nocedal2006numerical | §5 |
| QP 求解器 | stellato2020osqp（第一階段） | §6 |
| SO(3) log 微分 | sola2018micro | §3 |
| 延遲補償概念 | smith1957closer、findeisen2004computational | §8 |
| 擾動估計概念 | pannocchia2003disturbance、muske2002disturbance | §9 |
| 單步基線 | whitney1969resolved、nakamura1986inverse | §14 |

## B. 相關工作（說明異同，不承接其結論或數字）

minniti2019wholebody（全身 MPC）、haviland2022holistic（整機反應式 QP）、burgesslimerick2023onthemove（移動中操作）、
jain2010pulling（全向底盤開抽屜）、diehl2005rti（RTI；本系統不是 RTI）、mayne2000constrained、grune2017nmpc（預測窗長度理論；本系統未驗證）。

## C. 正文必須保留的界線

- 模擬、已知場景、真值目標；n = 3 的有界比較，不做顯著性檢定。
- 沒有穩定性、遞迴可行性或碰撞安全保證；命令鏈限制不等於幾何安全。
- 嚴格停車 vs MotM、H1/H5、B1/P 是三個不同問題的批次，不跨批次合併。

---

**書目查核來源（2026-10-05）**：
[Minniti 2019（arXiv 1902.10415）](https://arxiv.org/pdf/1902.10415)、
[Haviland 2022（QUT ePrints）](https://eprints.qut.edu.au/229003/)、
[Burgess-Limerick 2023（Monash）](https://research.monash.edu/en/publications/an-architecture-for-reactive-mobile-manipulation-on-the-move/)、
[Jain & Kemp 2010（dblp）](https://dblp.dagstuhl.de/rec/conf/icra/JainK10.html)、
[Pannocchia & Rawlings 2003（Wiley）](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690490213)、
[Diehl et al. 2005（Semantic Scholar）](https://www.semanticscholar.org/paper/A-Real-Time-Iteration-Scheme-for-Nonlinear-in-Diehl-Bock/05644bba98565e0c493b88eb80c35743e3dbd670)、
[Findeisen & Allgöwer 2004（ScienceDirect）](https://www.sciencedirect.com/science/article/pii/S1474667017387694)、
[Muske & Badgwell 2002（ScienceDirect）](https://www.sciencedirect.com/science/article/abs/pii/S0959152401000518)、
[Smith 1957（CiNii）](https://cir.nii.ac.jp/crid/1574231875295965056)、
[Schnabel et al. 2007（Uni Bonn）](https://cg.cs.uni-bonn.de/publication/schnabel-2007-efficient)；
Whitney 1969 與 Nakamura & Hanafusa 1986 的卷期頁取自多篇引用文獻的一致記載。
