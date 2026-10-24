# B0：無門底盤表觀速度殘差短測——小規格＋最小實作＋離線測試＋凍結 b0 r1（**未開跑，待核可**）

依 Codex 20261025_020934：
- 同意準備「無門底盤短測」套件，一次送齊；尚未授權六條件實跑。
- 先做這組，再回到 W1c 暫停期相對運動。
- 即使無門仍出現同類殘差，也不等於已解釋 W1c 的滑移與淨空下降。

## 1. 問題
門趟（p4b_1、w1b_30_1、w1c_30_1、w1d_30_1）的「實測位移 − 同區間命令積分」呈方向性殘差，本體座標約 (−0.8, +0.6)：
- 已確認無接觸力的 UNFOLD 約 1.25 mm/s，APPROACH 約 2.57 mm/s；
- 夾持中的 HOLD 約 4.0 mm/s。

本短測只回答一件事：**沒有門時，同停位、同執行鏈、固定手臂姿態下，零命令與 PARK_HOLD 是否仍出現同類殘差、是否隨姿態改變。**

## 2. 條件（6＝3 姿態 × 2 模式；各 10 s 觀察＋1 s 暖機；各自獨立初始化＝各自一個 Isaac 程序與 RUN_ID）
**姿態**
| 姿態 | 關節值 | 來源 |
|---|---|---|
| k2_ready | (0, 0, 0.035, 0, −1.5707963, 0) | K2_READY（K2 規格 draft-3；已核對，不退回原 STOW） |
| q_pre | (−0.9471, 0.0758, 1.6024, −1.539, −0.9479, −0.0544) | run_door_wide_w1d.sh Q_PRE（展開端 --q-unfold） |
| grasp | (−0.687744, 0.487065, 2.114029, −1.638494, −0.688516, 0.087497) | door_wide_w1c_30_1 room_run：OPEN 前最後一步（20.97 s）q_arm_meas |

**模式**
- zero：底盤命令恆 0。
- park_hold：錨點＝第 0 步實測位姿；每 5 物理步（20 Hz）以 `park_fixed.park_hold_cmd`（HoldServo 預設 k_p 2.0、v_max 5 mm/s、w_max 0.02，與門趟同）更新，其間零階保持。

## 3. 固定項與差異（揭露）
**與 isaac_door_k2_sim_ev4 相同**
- physics_dt 0.01、GroundPlane、同 URDF、同停位 (0.6, −0.5332, 90°)。
- 手指碰撞 split；支撐球＋地面零摩擦；手指 μs 1.0／μd 0.8。
- 手臂 kp 600／kd 60；手指 kp 1e4／kd 1e3。
- 初始化順序：set_joint_positions → set_gains → apply_action。
- 底盤寫入：每步 set_linear_velocity(R(yaw)·cmd, vz＝0)、set_angular_velocity(0, 0, wz)，在 apply_action 與 world.step 之前。

**差異**
- 不建門與牆。
- 手指開度全條件固定 0.0089（張開）：grasp 姿態沒有橫桿可夾。
- 手臂直接置於條件姿態，不經展開過程。
- 底盤命令不經 ROS 鏈、adapter、命令鏈與輪級限制器：
  - 零命令與 ≤ 5 mm/s 保持命令下，這些環節預期不改值，但未在本短測驗證；
  - 也沒有求解端的 1–2 週期延遲。
- **不做**偏差補償、不調伺服、不改停止門檻。

## 4. 量測與預先登錄的判讀
**逐步紀錄** `probe.json` steps，欄位：
- step、sim_t；步進前／後位姿；
- 本體命令、寫入的世界速度、寫入後讀回、寫入前根速度、步進後根速度與角速度；
- 手臂 q、q̇，手指 q；
- 輪框關節（base_link_rim_*_joint）速度，滾子關節 |q̇| 最大值；讀不到記 None（NA），不另開取證支線；
- 相位（warmup／observe）。

**彙整**（core.summarize，與門趟 r2 同一時間語意）
- 殘差：觀察段同一物理區間「實際位移 − Σ R(yaw_k)·cmd_k·dt」÷T，轉本體座標。
- 0.1 s 窗口中位數。
- 逐步「步進後根速度 − 寫入值」中位數。
- API 寫入讀回最大差。
- 與門趟方向的夾角。
- park_hold 另報：末 2 s 錨點誤差與命令。「錨點差＝命令／k_p」只是伺服律一致性，不作來源證據。

**類別**（只描述，不作通過／失敗）
- |殘差| < 0.5 mm/s ⇒ small；
- 否則與門趟方向 (−0.79, +0.61) 夾角 ≤ 30° ⇒ same_kind，> 30° ⇒ different_direction。

**對照數值**：門趟 UNFOLD 1.24–1.29、APPROACH 2.57–2.58、HOLD 3.99–4.02 mm/s。只並列，不配成等價條件，因為門趟手臂在移動或處於夾持中。

**可回答**
- 無門時是否仍有同類殘差；
- 殘差是否隨姿態改變；
- 寫入值是否被讀回一致；
- 殘差是否在步進中出現（步進後根速度 ≠ 寫入值）。

**不能回答**
- 殘差的物理來源；
- W1c 的滑移與淨空下降。

## 5. 實作與離線測試
| 檔案 | 內容 |
|---|---|
| `base_residual_probe_core.py` | 條件表、CommandSource（zero／park_hold ZOH）、summarize、classify、park_steady |
| `isaac_base_residual_probe.py` | Isaac 端：一程序一條件，只做場景與逐步讀寫，邏輯全在 core |
| `run_base_residual_probe.sh` | 單條件運行器：凍結核對、條件名核對、記「起 sim PID」供熱守護 |
| `run_base_residual_probe_batch.sh` | 六條件依序，各自 run_guarded_v2.sh，任一非 0 即停批 |
| `test_base_residual_probe_core.py` 9 項 | 見下 |

`test_base_residual_probe_core.py` 的 9 項（全過）：
- 條件表與不退回原 STOW。
- 合成偏差在零命令下被還原。
- **命令在區間中途步階時，同區間積分仍逐窗還原偏差**（r1 混用時間語意的反例）。
- park_hold 的 20 Hz ZOH 與穩態錨點誤差＝|偏差|／k_p。
- 類別邊界：small、反向、38°、11°。
- yaw 座標轉換。
- 寫入讀回差。
- 觀察段不連續 ⇒ 拒收。
- 不接受補償模式。

Isaac 端無法離線執行；只核語法，並確認呼叫順序與 ev4 一致。

## 6. 申請
- 六條件各一次（`TAG=1 bash evaluation/run_base_residual_probe_batch.sh`）。
- 乾淨環境、92 °C 守護；失敗保留不補跑。
- 短測無門、無接觸，只交數據與彙整；不交影片，因為不是任務實驗。若需要影片請指示。
- 完成後回到 W1c 暫停期相對運動，再回到 30° 保持與大角度主線。不延伸成全面底盤重整。
