# DL3：限定視覺預抓取接入——小規格（draft-1，送審；核可前不實作、不開跑）

2026-10-06。接續 DL2 離線串接原型封存（`freeze_dl2_offline.sha256`）。**只到把手前 30 mm 的接觸前位姿，不閉爪、不開抽屜**；
不新增模型、把手資產或任務；沿用既有控制配置與限制；真值只作評估，不能補回被拒絕的目標。

## 0　現況（要替換的真值依賴）

- `drawer_task_node` 每步由模擬器 `/drawer/sync_state` 取得**真值**把手位姿 `self.H`；NAVIGATE／UNFOLD／ALIGN 都用 `target_at(standoff=0.030, H)`。
- ALIGN 分兩段：`pre`（到 s = 30 mm 接觸前位姿，誤差 < `pre_tol` 連續 `pre_settle_s`）→ `approach`（收退讓量）。**DL3 只做到 `pre` 完成即停。**
- S4（d1s4b_M）：凍結 D1 線上 shadow 可運作；ALIGN 前有 44 格 L2；render-on-demand 影格固定晚約 0.2 s，姿態以有界歷史延遲匹配。

## 1　視覺目標來源（新節點版本，不改凍結的 S4 節點）

- 新節點 `dl3_vision_target_node.py`：沿用 S4 的擷取契約、配對、LatestSlot、失敗隔離（呼叫凍結 `d1_shadow_core`），
  每格跑凍結 D1（G0）＋DL2 估計端（N-obs；**N-obs 失敗不回退 N-prior**）＋DL1 目標，發布 `/dl3/handle_est`（JSON）：
  `T_WH`、`T_WE_s`（s = 0.03）、`t_obs`（影格 rendering_time）、`t_pub`、`prior_used`、`why`、`geometry_contract`。
- 只用 G0（幾何路徑）：L1 需 GPU 線上推論，本包不納入。

## 2　時間語意與鎖定（latch）

- 物體在預抓取期間**靜止**是明示先驗（本任務抽屜在夾持前不動；真值只在評估端核對此先驗）。
- **鎖定時刻**＝任務進入 ALIGN 時。取 `[t_latch − W, t_latch]` 內 `ok` 的估計（`t_obs` 在窗內；W＝**待定**，見 §6）；
  需 ≥ k 筆，且中心散佈（相對中位的最大偏差）≤ d_tol、軸向角散佈 ≤ a_tol；合格 ⇒ 以中位中心、平均軸重組姿態（同 DL2 規則）並產生 s = 0.03 目標，**之後目標固定**。
- 不合格（筆數不足、散佈過大、全為拒絕、時間戳非有限或在未來）⇒ **中止預抓取**（安全停止，記 `vision_latch_failed:<原因>`），不改用真值或地圖先驗。
- 鎖定後不再更新（不做視覺伺服）；新鮮度只表示「鎖定時用的觀測在窗內」，不宣稱物體移動時可追蹤。

## 3　鎖定前的相位（NAVIGATE／UNFOLD）

鎖定前沒有視覺目標，但手臂展開需要接近方向。提案：用**地圖先驗把手位姿**（設定的櫃體擺放，非逐步真值），標 `map_prior`。
**注意**：本模擬中地圖先驗＝真值擺放，因此鎖定前的目標與真值等價；本包的視覺貢獻只在 ALIGN 目標。是否在地圖先驗加入已知偏移（例如 x 方向 30 mm）以顯示視覺修正效果，見 §6。

## 4　鎖定時的估計候選檢查（只用估計，不讀真值）

1. **幾何**：以估計把手位姿反推估計物體位姿 `T_WO_est = T_WH_est · Trans(−c_bar)`（開度 0：夾持前關閉＝明示先驗），
   呼叫 GC1／GC2 凍結的核對（手指＋殼＋相機凸包對**估計物體**的模型）；不通過 ⇒ 中止 `vision_target_geometry_reject`。不讀資產真值擺放。
2. **可達性**：以既有 `arm_pregrasp.solve_ik` 在任務當下的底盤位姿解 s = 0.03 目標；關節須在既有 `LITE6_SAFE` 內；不可解 ⇒ 中止 `vision_target_unreachable`。
   不新增門檻；全身控制器仍以現行配置追蹤（底盤可動），此檢查只是「固定底盤可解」的保守前檢。
3. 通過後目標標 `geometry_contract: checked_estimated_object_static`（對估計物體，不是對真值）。

## 5　執行與評估（核可後才跑；一趟）

- 控制配置：沿用正式 MotM 配置（C0／H5 等已凍結設定，逐項列入 run 規格），唯一差異＝ALIGN 目標來源＝視覺鎖定目標；`pre` 完成後停（不收退讓量、不閉爪）。
- 評估（真值只在此）：鎖定時估計把手位姿誤差；最終 TCP 對**真值**接觸前目標的位置／旋轉誤差；手指／殼對真值物體的 GC 核對（評估用，不回饋）；
  鎖定窗內觀測筆數、拒絕原因、年齡分佈；任務時序與漏時槽；溫度。
- 結果分「控制達成鎖定目標」與「鎖定目標對真值的偏差」兩部分；不宣稱夾持、開抽屜或泛化。

## 6　待審
1. 鎖定窗 W、最少筆數 k、散佈容差 d_tol／a_tol 的取值依據（候選：S4 實測 ALIGN 前 L2 筆數與時間分佈；或以 DL2 開發資料的單格誤差分佈決定，登錄後凍結）。
2. 鎖定前用地圖先驗：是否加入已知偏移以檢驗視覺修正（會改情境，需另登錄），或維持等價真值並明寫限制。
3. 新節點沿用 S4 契約是否足夠；或直接在 S4 節點新版本內加 DL2 估計。
4. 可達性前檢只做固定底盤 IK 是否足夠。

---

# draft-2（依 Codex DL3 規格審查四項；以本節為準。名稱：**幾何視覺預抓取接入**）

### 1　鎖定窗與鎖定規則（登錄的工程候選；一致性篩選，不是定位精度保證）
- `W = 5 s`、`k = 3`、中心最大散佈 `d_tol = 5 mm`（各筆到融合中心）、軸線與法向角散佈各 `3°`（各筆到融合方向）。不足即拒絕，**不自動擴窗**。
- 只用鎖定時刻 `t_latch` 前**已收到**（任務節點以模擬時鐘記錄的收到時刻 `t_recv ≤ t_latch`）的**唯一影格**，且 `t_obs ∈ [t_latch − W, t_latch]`；較晚完成的結果不事後納入。
- 融合：中心取逐軸中位；軸號以第一筆合格估計對齊後平均；內側法向（y 軸）另行平均並檢查散佈；以 DL1 組姿、工具目標旋轉對設計抓取參考選號（同 DL2）。
- 保存：入選影格、每筆排除原因、各筆年齡、散佈、融合姿態、最終目標。
- 「物體靜止」＝數秒舊觀測仍可使用的明示假設，不代表通過動態目標新鮮度驗收。
- **離線鎖定演練**（`dl3_latch_rehearsal.py`，既有 d1s4b_M 紀錄，不新增擷取、不用真值選參數；`results/vision/DL3_latch_rehearsal.json`）：
  t_latch 31.66 s；窗附近 35 格中入選 7（年齡 3.25–4.85 s）、估計拒絕 17、鎖定後才收到 6、窗外 5；散佈 1.58 mm／軸 0.65°／法向 0.35° ⇒ 鎖定成功。
  （評估端才看真值：融合中心偏 2.20 mm、s = 0.03 目標偏 2.25 mm／0.25°。）注意年齡最大 4.85 s，接近窗界。

### 2　真值替換點與停止邊界（`drawer_task_node.py` 新旗標 `--handle-source vision`；預設 `truth` 行為完全不變）
| 位置 | 現況（真值） | vision 模式 |
|---|---|---|
| L447 等待量測 | 等 `nd.H`（真值）到齊 | 照舊等模擬器狀態（開度為抽屜關節量測，DL3 不用於目標）；另記錄 |
| L452 `_fk_off` | 真值把手 y 對設計 FK | 只作診斷欄位、標 `truth_diagnostic_only`，不參與控制 |
| L630 求解前目標（NAVIGATE／UNFOLD 及上線前） | `target_at(standoff)` 用真值 H | 鎖定前用**地圖先驗把手位姿**（`--map-handle-pose`，＝設定擺放＋資產把手偏移；本模擬中等於真值擺放，已揭露），只用於展開／接近方向 |
| L640–648 ALIGN 進入判定 | `st − nd.H[0] < 0.2` 視為把手讀值有效 | 沿用底盤／手臂到位條件；把手讀值有效改為「鎖定流程可執行」 |
| L661 `solver_start` | 進 ALIGN 即發 | **鎖定＋前檢全部通過後才發**；任一失敗 ⇒ A0 受控停止 |
| L674 `T_ref` | `target_at(0)`（真值） | 鎖定目標 s = 0 |
| L818／824／832／843 ALIGN 目標、斜坡終點、到位誤差 | `target_at(standoff)` | 全部指向**同一個**鎖定目標 s = 0.03 |
| L763 `handle_reading_valid` | 真值讀值年齡 | 鎖定有效（鎖定後固定） |
- 鎖定失敗、前檢失敗 ⇒ **A0 受控停止**（既有中止收尾：發 `/wgmpc/stop`、不再發目標、夾爪維持張開、不追加恢復），記 `vision_latch_failed:<原因>`／`vision_precheck_failed:<原因>`；不改發地圖或真值目標。
- `--stop-at-pregrasp`：`pre` 判定成立時，在轉 `approach`、收退讓量或閉爪**之前**走同一受控停止，結果記獨立的 **`PREGRASP_COMPLETE`**（不是 DONE）；停止後再記錄 3 s 的命令與 TCP 殘餘運動。

### 3　GC 與 IK 前檢的能力邊界
- 估計物體：`T_WO_est = T_WH_est · T_OH(0)⁻¹`，`T_OH(0) = Trans(c_bar)` 取自 GC1 鎖定契約（bar26 資產）；呼叫 GC2 `check2(T_WO_est, q = 0, …)`（手指＋殼＋相機）。
  **把工具、把手、櫃體一起剛體變換時局部距離不變**，因此通過只表示「指定抓取轉換與估計物體模型在目標處相容」，**不是觀測準確性閘門，也不驗證運動路徑**。
- 固定底盤 IK 前檢：`arm_pregrasp.solve_ik`（種子＝當下手臂角）於**當下底盤狀態**解 s = 0.03 目標；解後核對控制器有效限位 `LITE6_SAFE ± joint_margin 0.05 rad`；保存 FK 位置／姿態殘差與底盤狀態。
  失敗稱「**固定底盤 IK 前檢未通過**」，不稱全身不可達。端點 IK／GC 通過不寫成軌跡安全。
- 評估端（真值只在此）：把**估計目標**與**實測 TCP**直接對獨立的真值物體核對（GC2 以真值 T_WO 檢查實測 TCP 位姿的手指／殼／相機；目標對真值接觸前目標的偏差）；不從真值重新生成工具目標冒充核對估計。

### 4　控制版本與成功判準
- 控制配置＝**d1s4b_M 的解析配置**（批次腳本＋排程 `d1_s4_schedule_r2.tsv`＋`freeze_d1_s4_r5.sha256`；任務參數逐項見 `task.json args`：`standoff_m 0.03`、`pre_tol_m 0.005`、`pre_settle_s 0.5`、`pre_ramp_mps 0.03`、`grasp_depth_m 0.0068`、`approach_rate 0.02`、`motm_w_qn 0.8`、`motm_w_vref_pre 0.3`、`restow_mode sync`、MotM；
  求解節點 `align_solver.json args` 逐項，含 `reach_pos_m 0.005`、`reach_rot_rad 0.02`）。實作時由紀錄自動匯出 `DL3_control_config.yaml` 並凍結；唯一差異＝把手來源、視覺節點、`--stop-at-pregrasp`。
- **成功判準（對鎖定目標）**：TCP 位置誤差 ≤ 5 mm 且姿態誤差 ≤ 0.02 rad，連續 ≥ 0.5 s（三者皆為既有配置值）。
- **另報（不混入成功判準）**：鎖定目標對真值接觸前目標的位置／姿態偏差；停止時實測 TCP 對真值目標的偏差；真值物體 GC 核對；鎖定細節；停止後殘餘運動；漏時槽；溫度。
- 一趟；不宣稱夾持、開抽屜、學習式控制或泛化。

### 5　實作順序（核可後）
1. `dl3_latch.py`（純函式，已寫）＋測試；2. `dl3_vision_target_node.py`（沿用 S4 契約與凍結核心，S4 節點不動）＋故障注入測試；
3. `drawer_task_node.py` vision 模式（預設行為不變的回歸測試）；4. 匯出並凍結控制配置；5. 送審後才開跑。

---

# 實作（draft-2 核可範圍內的最小實作＋離線測試；**未開跑**）

| 檔案 | 內容 |
|---|---|
| `dl3_latch.py` | 鎖定純函式（W 5 s、k 3、5 mm、3°；已收到才算、唯一影格、軸號對齊後平均、法向融合與散佈） |
| `dl3_latch_rehearsal.py` → `results/vision/DL3_latch_rehearsal.json` | d1s4b_M 離線演練：7 筆入選、散佈 1.58 mm／0.65°／0.35°（評估端：目標偏 2.25 mm／0.25°） |
| `dl3_vision_target_node.py` | S4 節點複本（S4 原檔不動）；每格 `process_frame()`＝凍結 D1＋DL2 N-obs，發 `/dl3/handle_est`；不讀真值 |
| `dl3_task_vision.py` | 任務掛鉤純函式：地圖先驗、DL1 目標、鎖定＋GC2（估計物體）＋固定底盤 IK 前檢（`LITE6_SAFE ± 0.05 rad`、FK 殘差、底盤狀態） |
| `dl3_drawer_task_node.py` | `drawer_task_node.py` 複本（原檔在 DL1 凍結內，不動）＋§2 表列替換點；與原檔只改 3 行且皆受旗標保護 |
| `run_nav_handover_dl3.sh` | 啟動腳本複本（原檔在 S4 凍結內，不動）：換視覺節點與任務節點 |
| `results/vision/DL3_control_config.yaml`、`dl3_schedule.tsv` | 由 d1s4b_M 紀錄匯出的解析配置（任務 54 項、求解 62 項參數）與一趟排程 |

測試：`test_dl3_latch.py` 18/18；`test_dl3_vision_target_node.py` 12/12（ROS，含故障注入與 process_frame 離線一致）；
`test_dl3_task_integration.py` 16/16（ROS、不開 Isaac；真值刻意與估計差 30 mm：A 無估計 ⇒ A0 停止、從未 solver_start、鎖定前目標＝地圖先驗；
B 一致估計 ⇒ 鎖定與前檢通過、啟動後目標全等於鎖定目標、到位 0.5 s ⇒ PREGRASP_COMPLETE、停止後殘餘記錄、未轉 approach；C 缺旗標拒絕啟動）；
truth 模式回歸：`test_park_abort_integration.py` 指向 DL3 複本（--park-fixed／--park-hold）7/7 ×2；S4 節點故障測試、DL2 測試照舊通過。
凍結：DL1／GC1／GC2／DL2 清單全部核對一致；S4 r5 清單中只有 `d1_s4_audit.py`／其測試為 S4 跑完後依審查補強（22:41 > 22:30），控制相關檔一致。

---

# 實作 r2（依 Codex 實作審查三項必修；以本節為準）

1. **第一輪求解用鎖定後目標**：vision 模式**鎖定前不產生、不發布任何 `/drawer/tcp_target`**（因此不再使用地圖先驗，原 §3／§2 表中「地圖先驗」一列作廢；
   `/drawer/tcp_target` 的唯一消費者是求解節點）。鎖定成功當輪先發鎖定來源目標（斜坡起點＝當下 TCP、姿態與斜坡終點＝鎖定結果）再發 `solver_start`。
   求解節點另建複本 `dl3_wgmpc_wg2_node.py`（原檔在 S4 凍結內不動），唯一改動：`--require-topic-target` ⇒ 待命結束還須已收到合格話題目標；
   並落盤 `dl3_first_target`（收到時刻與矩陣）。因鎖定前沒有任何目標，收到的第一個目標必為鎖定來源 ⇒ 第一輪必用鎖定後目標。
   跑後以 `dl3_run_check.py` 核第一輪 `solve_in.T_cyc` 姿態＝鎖定目標、`target_src = topic`、求解端首收目標＝任務端首發目標、首收時刻不早於鎖定。
2. **停止後紀錄牆鐘上限**：`--post-stop-wall-cap-s 15`（事前固定）；逾時保存部分紀錄並標 `post_stop_record_incomplete`；`post_stop_record_complete` 分列，
   「預抓取到位成立」與「停止後證據完整」分開。
3. **環境明設**：`run_nav_handover_dl3.sh` 明設 `OFFSET_MOVING=1` 及其餘 9 個批次未設、啟動腳本會讀的值（與 d1s4b_M 落盤一致；WHEEL_FIT 未落盤採預設 project），
   不依賴父 shell；跑後逐項比對求解與任務落盤參數（只允許登錄差異）。
- 文件更正：任務節點複本相對原檔＝**新增 vision 掛鉤，並替換三處既有判斷**（ALIGN solver_start 條件、started 條件、handle_reading_valid），另加鎖定前不發目標的條件分支與 pre 完成停止段。

測試 r2：`test_dl3_task_integration.py` 20/20（A 無估計 ⇒ 從未發任何目標；B 鎖定前零目標、首個目標先於 solver_start 且即鎖定目標、全部目標皆鎖定來源、停止後紀錄完整；
C 缺旗標拒絕；D PREGRASP_COMPLETE 後時鐘停更 ⇒ 牆鐘上限內結束、標不完整、結果仍為 PREGRASP_COMPLETE）；`test_dl3_solver_start_gate.py` 4/4（無目標不離開待命、收到即離開、
不帶旗標與原節點同、複本只改待命條件一行）；其餘 DL3 測試照舊。
