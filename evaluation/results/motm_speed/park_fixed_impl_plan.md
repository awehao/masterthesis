# PARK_FIXED 接線計畫（草案，待 Codex 審）

日期：2026-10-05。依 `park_fixed_vs_motm_spec.md`。**尚未改程式、未跑模擬。**

## 0. 已完成：固定底盤離線可行性（`evaluation/park_fixed_feasibility.py` → `park_fixed_feasibility.json`）

底盤固定在現有停位 (−0.136412, 0.560, 1.297349)，TCP 姿態固定為實錄抓取姿態（兩組實錄 ENGAGE_WAIT 目標相同：
p = (0, 1.1718, 0.55)），逐段核：

| 段 | 路徑模型 | IK 殘差 | 有效限位最小餘裕 | 自碰最小（點雲） | 手臂—櫃體／抽屜 | 夾爪—抽屜本體 |
|---|---|---|---|---|---|---|
| 展開／收臂 | 收攏↔接觸前姿態，關節空間直線 | — | 0.011 rad（j3，**收攏姿態本身**） | 2.8 mm（link1–link5） | 93.8 mm | 65.7 mm |
| 對準／退開 | TCP 沿 y 30 mm，每 2 mm IK | ≤ 2e-9 m／3e-8 rad | 0.881 rad | 12.9 mm | 63.8 mm | 35.7 mm |
| 開 0→210 mm（關＝反向） | TCP 沿 −y，每 2 mm IK，抽屜同步移動 | ≤ 2e-9 m／3e-8 rad；相鄰步最大 0.016 rad | 0.486 rad（j5） | 8.5 mm（link4–link6） | 63.8 mm | 35.7 mm |

判讀：此路徑模型下，**固定底盤在現有停位能完成完整 200 mm 開關的幾何路徑**，不需要先換停位。展開／收臂的 2.8 mm 與
0.011 rad 來自共用的收攏姿態，兩組相同（與既有紀錄的點雲法 2.64 mm 一致，該法已知高報）。限制：點雲距離近似、關節空間
內插 ≠ 實際展開軌跡、無動力學與接觸力；通過 ≠ 物理可行證明。物理可行性由功能確認趟判定。

## 1. 現有流程（MOTM=0）為何底盤會動

導航 →（mission 依距離）減速段 glide 以 30 mm/s 滾動 →（mission 依速度框）控制權轉給全身節點 → 全身節點
WARMUP 延續導航底盤速度、UNFOLD 一邊展開一邊以 P 控制把底盤推到停位 → 求解節點接手後底盤自由度未鎖。

## 2. 接線（新旗標 `PARK_FIXED=1`，隱含 MOTM=0；預設不帶 = 既有行為一位元不變）

| 元件 | 改動 | 為何在這裡 |
|---|---|---|
| `drawer_glide_node.py` `--park-stop` | 剖面改成在停位歸零：v(d) = min(v_cap, √(2·decel·max(0, d)))（無 v_roll 維持段）；橫向與偏航照舊 P 控制；進入入場容差（≤ 10 mm、≤ 0.05 rad）且命令已降到零後**鎖存輸出零** | 減速段本來就擁有停車前的控制權 |
| `isaac_drawer_room_sim.py` `--park-fixed` | **靜止閘門**：控制者為 glide 時，逐物理步以真值位姿差分算線速度／yaw 速率，並讀實際套用的底盤命令；連續 max(0.5 s, max_cmd_age_s) 皆 ≤ 1 mm/s、≤ 0.01 rad/s、套用命令 ≤ 1e-6、位姿在入場容差內、手臂仍收攏 ⇒ 通過，**錨點＝通過當步實測位姿**；發 `/park/gate`、寫 `park_gate.json`。**保持監看**：從轉移給全身（UNFOLD 起）到 RESTOW 完成交還導航，逐物理步核速率、套用命令、相對錨點 ≤ 1 mm／0.5°；第一次違規記錄（步、相位、原因）並發停止請求走既有受控收尾，**不靜默改寫命令**；紀錄缺口／非有限 ⇒ 證據不足。`room_run.json` 每步加 base_mode | 只有執行端同時擁有每個物理步的真值位姿與實際套用命令 |
| `drawer_mission_node.py` `--park-fixed` | glide→wholebody 的轉移請求改為**只在 `/park/gate` 通過後**發出（取代速度框條件） | 轉移請求本來就由 mission 發 |
| `drawer_wholebody_node.py` `--park-fixed` | WARMUP 底盤輸出零（不承接導航速度——閘門已保證套用為零）；UNFOLD 只動手臂，底盤三軸恆為零、不做停位 P 修正；`done_base` 改為「閘門已通過」；收臂（若由本節點）同樣底盤零 | 舊停車組「邊動邊展開」的來源 |
| `wgmpc_wg2_node.py`／`wgmpc_core_sp.py` `--base-fixed` | 整個時域 `u_base[k] = 0`：底盤三軸的速度框列 l = h = 0（等式），vmax 不變（成本 1/vmax² 不受影響）；**不是**求完後剪掉；u_prev 底盤非零 ⇒ 不可行 ⇒ 既有失敗收尾。逐輪記 base_fixed | 規格要求在求解層鎖自由度 |
| `drawer_task_node.py` | **已讀碼確認不需改底盤邏輯**：非 MotM 時 RESTOW、HANDBACK_WAIT 的底盤三軸本來就是零（只有 `a.motm` 才填 vr_now），RETREAT 由求解節點負責（`--base-fixed` 已鎖）。只加 `--park-fixed` 旗標寫入 task.json 供核對；協同話題不開（MOTM=0） | 2026-10-05 讀 drawer_task_node.py 829–890 行 |
| `run_nav_handover.sh`／`run_plan_batch.sh` | 新方法 `PARK_FIXED`：MOTM=0、各節點帶 `--park-fixed`／`--base-fixed`、任務參數與 MOTM 組共同部分相同 | 批次工具已有 status／凍結／中斷處理 |

不做：把底盤改成靜態剛體、鎖世界位姿、瞬移或只給停車組的虛構煞車；不改速度／加速度／輪級／關節限制。

## 3. 測試（只做規格列的必要反例）

- 閘門與保持監看的純函式化（可離線測）：零命令但移動、非零命令但靜止、中途出去又回來、原地轉向、yaw 跨 ±π、
  缺步、非有限值、時間戳停更／倒退 ⇒ 各自應有的判定。沿用 `parked_operation_audit.py` 的度量定義，不另造一套。
- 求解核心 `base_fixed`：可行時 u_base 全時域 = 0；u_prev 底盤非零 ⇒ 不可行；預設路徑與既有輸出逐位元相同。
- 既有測試（stop_paths、shutdown、cycle_record、replay_check、run_plan_batch）全過。

## 4. 執行順序

1. 實作＋離線測試 → 送審。
2. PARK_FIXED 一趟功能確認、同版本 MOTM 一趟功能確認（不算正式）；核模式合規（parked_operation_audit＋park_gate）與 S1–S6。
3. 凍結程式、解析後兩組配置、判定器、排程與門檻。
4. 正式 3 對／6 趟：M→F、F→M、M→F。計時從共同任務 GO 到獨立終端完成；停車、閘門、等待全部計入。
5. 失敗照列；模式違規另列；工程問題停批。

## 5. 想請 Codex 確認

(a) 靜止閘門與保持監看放在執行端（模擬器）是否恰當；違規時發哪個停止請求最合適（既有任務中止路徑）。
(b) 「任務 GO」的共同定義：建議用 mission 發出第一筆導航目標的模擬時間（兩組同一事件）。
(c) 入場容差 10 mm／0.05 rad 若未達成（glide 停偏），是中止該趟還是允許 glide 再修正一次（仍在閘門前）。
(d) 功能確認若 PARK_FIXED 物理上做不到（例如夾持或拉力不足），是否先停下回報、不調參。

---

## 6. 實作狀態（2026-10-05，依 Codex reviews/20261005_113349_reply.md；尚未開模擬）

| 元件 | 實作 | 預設路徑 |
|---|---|---|
| `park_fixed.py`（新） | StaticGate／HoldMonitor 純邏輯；同一物理步真值位姿差分＋實際寫入底盤命令；證據不足另判 | — |
| `dual_source_executor.py` | 新增可選 `switch_guard(to, step, t)`：提交切換當步多核一道條件；不取代就緒與預核，不成立照窗口順延／取消 | `None` ⇒ 不變 |
| `isaac_drawer_room_sim.py` `--park-fixed` | 閘門（轉給全身前）、保持監看（實際轉給全身 → 實際交還導航，含 UNFOLD）、守門（已通過＋當前成立＋前一步新鮮＋未違規）、**滾動下界明確設 0**；違規閂鎖：`ex.cancel_handover`＋發 `/park/violation`；`room_run.json` 加 `park_mode` 欄、存 `park_gate.json` | 不帶 ⇒ 不變（v_min_lin 仍 0.010） |
| `drawer_glide_node.py` `--park-stop` | 停在停位：v = sign(d)·min(v_cap, √(2·a·|d|), 1.0·|d|)，越位回正 ≤ 10 mm/s（有界）；誤差 ≤ 1.5 mm／0.005 rad 且套用速率 ≤ 2 mm/s ⇒ 鎖存精確零；牆鐘 30 s 逾時 ⇒ 發 `/park/violation` 停止。離散模擬：0.6 m 起 6.1 s 進鎖存容差、無越位 | 不帶 ⇒ 原滾動剖面 |
| `drawer_mission_node.py` `--park-fixed` | 判斷函式的滾動下界設 0；**閘門已通過＋當前成立＋訊息新鮮**才請求轉給全身；收到違規停止；記共同 GO `go_sim_t`（首次發計畫的模擬時間，兩組都記） | 只多記 `go_sim_t` |
| `drawer_wholebody_node.py` `--park-fixed` | WARMUP 底盤零（不承接導航速度）；UNFOLD 只動手臂、底盤恆零、不做停位修正；若生成非零底盤 ⇒ 記錄並停止（不剪）；收到違規停止；與 `--motm` 同時 ⇒ 拒絕啟動 | 不帶 ⇒ 不變 |
| `drawer_task_node.py` `--park-fixed` | 收到 `/park/violation` 或本節點生成非零底盤命令 ⇒ 走既有中止收尾（A0 語意：停求解、不發目標、夾爪維持）；與 `--motm` 同時 ⇒ 拒絕啟動。RESTOW／HANDBACK 的零底盤生成方式不變 | 不帶 ⇒ 不變 |
| `wgmpc_core.py`／`wgmpc_core_sp.py` | `base_fixed`：`vbox()` 底盤 0（速度框 l = h = 0 的整時域等式；名目序列同夾）；`vmax()` 不變；理想核心見 True 拒絕 | False ⇒ 逐位元不變 |
| `wgmpc_wg2_node.py` `--base-fixed` | **模式閘門**：u_prev 底盤 > 1e-6 ⇒ 拒絕並停止（不宣稱 QP 無解）；**輸出核對**：整形後發布命令底盤 > 1e-6 ⇒ 不發布並停止；逐輪 `base_fixed`、stats `base_fixed` | 不帶 ⇒ 不變 |
| `horizon_replay.py`／`horizon_replay_check.py` | 重播依 args 還原 `base_fixed`（舊趟次 False）；`park_mode_refused` 列入合法非求解理由 | 舊趟次不變 |
| `run_nav_handover.sh`／`run_plan_batch.sh` | `PARK_FIXED=1` 帶齊各節點旗標；與 `MOTM=1` 同時 ⇒ 拒跑（碼 87）；批次方法 `PARK_FIXED` | 不設 ⇒ 不變 |

### 離線測試

- `test_park_fixed.py` 24/24：閘門（靜止通過、保持不足、零命令但移動、非零命令但靜止、原地轉、停偏、手臂未收、中途移動重起、缺步／時間不前進／非有限重起）、保持監看（靜止 PASS、出去又回來、慢漂累積、yaw 偏離、非零命令、yaw 跨 ±π、缺步 INSUFFICIENT、未交還 INSUFFICIENT、第一次違規閂鎖）、執行端守門（順延到可行步才切換、一直不行則取消留在導航、None 不變、MOTM 滾動下界仍擋停住）。
- `test_wgmpc_base_fixed.py` 8/8：vmax 不變／vbox 底盤 0；**預設路徑與改動前核心逐位元相同**（mt_b1_02_P 實錄 10 輪；改動前核心存 `results/motm_speed/pre_park_core/`）；base_fixed 解出整時域 |U_base| ≤ 1e-12（實測約 1e-24；OSQP 不給精確 0.0）；u_prev 底盤 1e-3 m/s 仍可行並降到 0；理想核心拒絕。
- 既有測試全過：stop_paths、shutdown_classify、cycle_record、horizon_replay_check、horizon_metrics、parked_operation_audit、control_authority、dual_source_executor、run_plan_batch 15/15；mt_b1_01_M 重播仍 PASS；求解節點建構對照只多 `_base_fixed`（False）。

### 可行性腳本修正後（`park_fixed_feasibility.json`）

措辭改為「**找到目前取樣路徑上的固定底盤 IK 解**」。設計接觸只排除夾持相位的手指—橫桿：
- 開 0→210 mm：夾爪殼—橫桿最小 7.19 mm；手臂—櫃體／抽屜 63.8 mm；自碰 8.48 mm；限位餘裕 0.486 rad。
- 對準／退開（距抓取點 > 2 mm）：**finger2—橫桿 −1.8 mm（點雲重疊）**。FK 中手指為 URDF 預設位置、未建模實際張開角，可能是原因；此段 TCP 相對把手的路徑兩組相同（MotM 與舊停車組實錄都完成接觸前對準）。列為未解限制，不追加掃描、不稱碰撞餘裕已驗收。
- 展開／收臂：2.8 mm 自碰與 0.011 rad 照舊保留為未解限制（實際 UNFOLD 先動 j3，非表內直線）。

### 尚未驗證（需功能確認趟）

靜止閘門與保持監看在真實物理中的表現（展開時手臂反作用是否推動底盤）、減速段實際停車時間、`parked_operation_audit` 的整窗需改以「實際轉給全身 → 實際交還導航」為準（模擬器 `park_gate.json` 為主判定，audit 交叉核對）。
