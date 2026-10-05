# DL2：視覺觀測 → 物體局部座標目標的離線串接——小規格（draft-1，送審；核可前不實作）

2026-10-06。第二階段計畫 §9.13.4。接續 DL0（凍結偵測／分割）、DL1（`object_target`，封存）、GC1／GC2（封存、暫停）。
**不接控制、不重新訓練、不跑新模擬、不開新盲測。** 已開封的 DL0 測試（lateral／view）是歷史測試，本包不使用。

## 0　輸入資料與凍結路徑

| 項目 | 來源 | 說明 |
|---|---|---|
| 開發資料 | `DL0_dev_eval_v3.json` 所評的兩個開發群組：`traj_wg4b_f02_P`（87 格）、`traj_mt_b1_02_P`（99 格） | 既有擷取（RGB-D、讀回內參、擷取時相機位姿）；dev_near 用於訓練，不納入 |
| G0 幾何基線 | 凍結 D1 `d1_handle_detect.detect`（`freeze_d1.sha256`） | 輸出 L1（p0、axis）、L2（center）或拒絕原因 |
| L1 學習式遮罩 | 凍結模型 `freeze_dl0_model_v2.sha256`（ep11）＋`dl0_g1_check.g1_detect` | 同上欄位 |
| 目標生成 | DL1 `object_target_geometry`（`freeze_dl1_offline.sha256`） | `handle_pose_from_partial`、`object_target`、`choose_candidate` |

既有逐格偵測結果（`DL0_dev_eval_v3.json` 的 rows）可直接重用；重算時須與既有結果逐格一致（否則停止並回報），不重新調參。

## 1　觀測 → 把手位姿（新檔 `evaluation/dl2_obs_to_target.py`，純函式＋離線腳本）

- **中心**：只用 L2 中心。L2 缺（`center_unobservable`）⇒ 拒絕 `center_missing`；**不以 L1 軸線上的點或已知長度補中心**。
- **軸線**：L1 axis。偵測 `ambiguous`／`no_candidate`／其他拒絕 ⇒ 原因原樣傳遞。
- **前板法向**（凍結偵測器不輸出法向）——兩種，皆明記於每筆 `prior_used`：
  - **N-obs（主）**：以凍結 D1 的 `backproject`、`ransac_plane` 在 L2 中心 0.15 m 內、扣除橫桿內點後的深度點上擬合平面；取法向近水平（|n_z| ≤ 0.2，與 D1 前板條件同）且到橫桿軸距離 20–60 mm（D1 `BAND`）的平面；
    外側方向＝朝向相機的一側。無合格平面 ⇒ 拒絕 `normal_unobservable`。參數全部沿用 D1 既有值，不新調。
  - **N-prior（對照）**：先驗「前板為鉛直面」：n ∝ axis × ẑ，外側取朝相機側。`prior_used` 記 `vertical_front_prior`。
- **軸號**：橫桿兩端對稱，號無法由觀測決定 ⇒ `handle_pose_from_partial(return_candidates=True)` 取兩個候選，再以**設計抓取旋轉**為參考經 `choose_candidate` 選定；`prior_used` 記 `design_grasp_reference`。
- **時間**：每筆記觀測時刻（擷取 sim_t）與目標生成查詢時刻；年齡 > `max_age` ⇒ 拒絕 `observation_stale`。離線重播的年齡＝0（同一時刻），此條只以單元測試核對；`max_age` 由審查決定（見 §5）。
- 位姿約定＝DL1 資產 prim 約定（x 沿桿、y 指向內側＝−外側法向）。

## 2　目標候選

以 DL1 `object_target`（基準 `T_HG`＝phf_01_M 的 `[R_grasp | (0, +0.0068, 0)]`、`a_H = (0, −1, 0)`、`T_EG = I`）產生 s = 0 與 s = 0.03 m 兩個候選；
逐筆保存：群組、影格、距離、路徑（G0／L1）、法向方式（N-obs／N-prior）、觀測時刻、`prior_used`、拒絕原因、候選矩陣。
**不呼叫 GC1／GC2**：其介面由資產真值重建把手位姿，接上會讓真值取代視覺估計；幾何契約對視覺目標的核對列為後續。

## 3　評估（真值只用於此）

- 真值把手位姿：`T_WH* = [I | c*]`，c*＝meta 的 `handle_center_world_at_capture`（缺則依 `dl0_autolabel` 既有規則用靜態 prim，並逐筆標 `c_source`）；旋轉＝資產約定（櫃體 yaw 0，與 `dl0_autolabel.AXIS_W` 一致）。真值目標＝`object_target(T_WH*, …)`。
- 報告（分群組、分距離箱 0.1–0.4／0.4–1.0／1.0–2.0／2.0–3.0 m、分路徑與法向方式）：
  1. **候選產生率**（分母＝全部正樣本影格）與拒絕原因分佈（覆蓋）；
  2. 把手位姿誤差：中心位置（mm）、旋轉測地距離（°），另拆「軸向角誤差」與「繞軸滾轉誤差」；
  3. 目標誤差：s = 0、s = 0.03 的 TCP 位置（mm）與旋轉（°）；
  4. 軸號選擇與真值一致的比例（以真值軸號評估，不回饋到選擇）。
- 不設通過門檻；描述性結果。不宣稱可執行、可達、泛化或閉迴路。

## 4　測試（離線）
1. 合成觀測（由真值位姿加已知擾動）⇒ 位姿與目標誤差與擾動一致；法向缺、中心缺、ambiguous、過期各自拒絕且原因正確。
2. N-obs 在合成平面點雲上回到已知法向；朝相機側判定正確。
3. 軸號兩候選經參考選擇可重現；參考旋轉不合法 ⇒ 拒絕（沿用 DL1 介面）。
4. 評估腳本不讀真值於估計端（以模組匯入檢查：估計函式不得讀 truth／meta 真值欄位）。
5. 重用的既有逐格偵測與重算一致。

## 5　待審
1. 法向：N-obs 為主、N-prior 為對照是否恰當；或只做 N-prior（更小）。
2. `max_age` 取值依據（候選：S4 實測的固定 0.2 s 渲染延遲＋處理 p95；或本包只保留介面、不訂值）。
3. 資料只用兩個開發群組是否足夠（dev_near 用於訓練不納入）。
4. 不呼叫 GC1／GC2 是否同意。

---

# draft-2（依 Codex DL2 規格審查三項必修與裁定；以本節為準）

**定位**：檢查既有把手觀測結合明示抓取與方向先驗，能否生成一致的物體局部座標目標候選；尚未驗收碰撞、可達性、新鮮度或閉迴路執行。

### 0　資料與重用
- 兩開發群組共 186 格**全部有去向**：正樣本＝候選率分母；負樣本與排除項另列（負樣本若偵測器仍給 L2，照樣跑估計並計「負樣本產生候選」數）。
- 偵測結果**不重算**：逐格 G0／L1 的 L1、L2、拒絕原因取自 `DL0_dev_eval_v3.json`（在 `freeze_dl0_model_v2.sha256` 內，載入時核對雜湊）。N-obs 只另讀深度、內參與擷取時相機位姿。
- 真值：186 格皆有逐格 `handle_center_world_at_capture`（`c_source = per_frame`），**不使用靜態回補**；若遇缺則標 `evidence_insufficient`。

### 1　估計端輸入白名單
估計函式只接受：`path`、`L1`（p0、axis）、`L2`（center）、`reject`、`reject_L2`、`depth`、`K`、`T_cam`、`t_obs`；其他鍵（例如任何真值欄位）⇒ 拒絕 `unexpected_input:<key>`。真值只交給評估端。

### 2　軸號（修正把手／工具座標混用）
兩個把手候選 `T_WH^(i)` 各產生工具目標 `T_WE^(i)(0) = T_WH^(i) · T_HG · T_EG⁻¹`，以 `choose_candidate` 比較**工具目標旋轉**與**工具參考旋轉**（設計抓取 `R_grasp`，phf_01_M 停車位姿＋`q_grasp` 的 FK）。
選定索引後，s = 0 與 s = 0.03 共用同一候選。`prior_used` 記 `design_grasp_reference`；**軸號一致率不能解讀成模型辨識出物體正反方向**。
測試加入反例：以把手旋轉對工具參考比較會選到反向軸號。

### 3　N-obs（主）規則——登錄後凍結
- 點：凍結 D1 `backproject`（stride 2、深度 [0.1, 3.0] m）＋4 mm 體素（D1 值）；
  **DL2 新登錄參數**：鄰域半徑 0.15 m（到觀測中心）；橫桿排除半徑 0.025 m（到**觀測**軸線段的距離，線段＝觀測中心 ± 0.11 m 沿觀測軸；不讀真值標籤或遮罩）。
- 平面搜尋：剩餘點上依序以凍結 D1 `ransac_plane` 取至多 6 個平面（D1 `max_planes`、`plane_tol` 6 mm），每平面內點 ≥ 400（D1 `plane_min`），RANSAC 種子每格固定 `default_rng(0)`。
- 合格條件：|n_z| ≤ 0.2（**重力／鉛直先驗**，與 D1 前板條件同）；法向定向為朝相機側；**觀測中心到已定向平面的有號距離** ∈ [0.020, 0.060] m（D1 `BAND`）。
- 合格 0 ⇒ `normal_unobservable`；合格 ≥ 2 ⇒ `normal_ambiguous`。**N-obs 失敗不改用 N-prior。**
- N-prior（對照）：n ∝ axis × ẑ、朝相機側；`prior_used` 記 `vertical_front_prior`（同屬重力／鉛直先驗）。
- 四條路徑（G0／L1 × N-obs／N-prior）分開報告。

### 4　新鮮度
本包不訂操作用 `max_age`。介面保留顯式 `max_age_s`；測試過期、未來時間（年齡 < 0）、非有限時間；未設定時輸出 `freshness: not_accepted`（新鮮度未驗收），不冒充通過。離線評估在同一擷取時刻生成，年齡為零。

### 5　輸出與誤差定義
- 每筆輸出保留 `geometry_contract: unchecked`，明示不可直接執行；不呼叫 GC1／GC2。
- 把手位姿誤差：中心 ‖ĉ − c*‖；完整旋轉測地距離 geodesic(R̂, R*)。
  **軸向角誤差** = arccos(|x̂ · x*|)（不分號）；**繞軸滾轉誤差**：令 ỹ＝ŷ 投影到 x* 的正交平面後正規化，滾轉誤差＝∠(ỹ, y*)（ŷ 為估計的內側法向軸，與軸號無關）；**軸號一致**＝x̂ · x* > 0。
- 目標誤差：s = 0、s = 0.03 的 TCP 位置差與旋轉測地距離（選定候選對真值目標）。
