# GC1：幾何契約子包（bar26＋split 手指）——小規格（draft-1，送審；核可前不實作）

2026-10-06。接續 DL1 離線封存（`freeze_dl1_offline.sha256`）。DL1 的 `object_target` 只回工具目標並標 `geometry_contract: unchecked`；
本包定義**目標放行前的幾何核對**，只限目前唯一實跑過的配置：**`drawer_unit_bar26.yaml`＋`--finger-collision split`**。
純離線幾何（靜態、開爪姿態），不接節點、不跑模擬、不改資產、不放寬任何門檻、不擴其他把手／夾爪。

## 0　現況與已知矛盾（契約要先處理的）

| 項目 | 來源 | 值 |
|---|---|---|
| 橫桿 | `drawer_unit_bar26.yaml` `handle.bar` | 圓柱、軸 x、半徑 0.013、長 0.200、中心 (0, −0.285, 0.550)（櫃體本地） |
| 支柱 | 同上 `handle.posts` | 兩方塊 0.016×0.040×0.024，中心 (±0.085, −0.265, 0.550) |
| 放置 | `drawer_asset.build_usd` | **只平移 (ox, oy)，yaw 固定 0**；抽屜沿本地 −y 滑動，USD 碰撞＝方塊與解析圓柱 |
| 手指 | URDF `finger_joint1/2` | 原點 (0, 0, 0.0543)（夾爪連桿）、軸 ±y、0–0.0089 m、mimic 1:1；TCP 在夾爪連桿 z 0.0836 ⇒ 手指座標 z 29.3 mm |
| 手指碰撞 | `finger_collision.split_hulls`（Z_SPLIT 0.008） | 根部塊（z ≤ 8 mm，內面 y≈0）＋指片（z 8–28 mm，內面 y = ±11.2 mm）；網格 md5 finger1 `aa287b10…`、finger2 `9ac3f6e8…`（xarm_ros2 v1.18.1） |
| 抓取深度 | 實錄 `task.json args.grasp_depth_m` | **0.0068**（桿心在手指座標 z 22.5 mm） |

**矛盾**：`drawer_unit_bar26.yaml` 的 `grasp_surface` 區塊（`tcp_offset_along_tool_z: 0.0147`、`finger_gap_open_m: 0.0178`）仍是 **10 mm 桿／凸包手指**的舊值，
與同檔檔頭（指片間距 22.4–40.2 mm、抓取深度 6.8 mm）及實錄參數不符。`drawer_task_node` 的來源字串也寫 12.26 mm。
本包**不修改資產檔**（既有趟次的前提），改由契約檔明示採用的數值與來源，並把這兩處列為「已知過時欄位，契約不讀」。

## 1　契約檔（新檔 `evaluation/geometry_contract_bar26_split.yaml`，凍結後只讀）

鎖住：資產檔 sha256、URDF sha256、手指 STL md5、`finger_collision.py` sha256 與 `Z_SPLIT_M`、`drawer_asset.py` sha256；
數值：桿半徑、指片內面 |y|（閉）、手指行程、根部／指片分界、抓取深度、把手座標約定（＝DL1 資產 prim 約定，R_WH = I 於 yaw 0）。
核對器載入時重算雜湊，**任一不符 ⇒ 拒絕 `contract_version_mismatch`**（不以「差很小」通融）。

## 2　核對項（純函式 `evaluation/geometry_contract.py`，輸入＝DL1 目標＋開度＋契約）

1. **開口相容**：桿徑 d 與指片間距 [g_closed, g_open]＝[22.4, 40.2] mm：須 g_closed < d（夾得住）且每側全開餘裕 (g_open − d)/2 ≥ 既有 `enclosure_margin_min_m` 0.002；
   接觸手指位置 q_c = (d − g_closed)/2 須在 (0, 0.0089)。bar26：q_c 1.8 mm、每側 7.1 mm。否則拒絕 `opening_incompatible`。
2. **抓取深度相容**：桿截面在手指座標的 z 範圍與接觸高度——接觸高度（桿心 z）須落在指片段 [8, 28] mm 內；桿最近點 z（＝桿心 z − r）須 > 根部塊上界 8 mm。
   bar26：桿心 22.5、最近點 9.5 mm（離根部 1.5 mm）。否則拒絕 `depth_incompatible`。**1.5 mm 只回報，不稱安全餘裕。**
3. **共同變換**：物體碰撞形狀（櫃體方塊、抽屜方塊、支柱、橫桿圓柱）以 `T_W_obj · Trans(q·u)` 變換；手指兩凸塊以目標 `T_WE`＋手指位置（全開 0.0089）經 URDF 鏈變換。
   兩者都從**同一個**物體位姿與 DL1 目標導出，不另取世界常數。
4. **目標處碰撞**（開爪、接觸前目標 s ≥ 0 與 s = 0）：手指凸塊對**非設計接觸**形狀（櫃體、抽屜面板、支柱）的最小距離 ≥ 0（不穿透）⇒ 否則拒絕 `penetration_at_target`；
   對橫桿（設計接觸對）只要求開爪時不穿透，距離只回報。距離用**精確凸體距離**（GJK 或凸二次規劃；圓柱用解析或凸約束），**不用點雲取樣**（已知會高報餘裕）。
   本包**不新增**一般環境 5 mm 門檻到這些配對；只判穿透與回報數值。
5. **座標約定**：輸入的把手座標須為 DL1 約定（含 `reparam_grasp` 換算後）；與資產 prim 位姿（由契約重建）旋轉差 > 1e-9 rad 或平移差 > 1e-9 m ⇒ 拒絕 `handle_frame_mismatch`。
6. 輸出：`ok`／`why`、各項數值、`contract_id`；通過時標 `geometry_contract: 'checked_bar26_split_static'`——**不是**碰撞安全、可達或夾持成功。

## 3　離線測試（不跑模擬）

1. 基準：phf_01_M 的 ENGAGE 目標（DL1 已核對）⇒ 通過；回報值重現檔頭／`test_finger_collision.py` 的 1.8 mm、7.1 mm、1.5 mm（容差寫明）。
2. 等變：物體與目標同乘任意剛體 G（含 yaw，**僅離線幾何**；USD 建置目前不支援 yaw）⇒ 所有距離不變（≤ 1e-9 m）。
3. 反例：桿徑 20／44 mm ⇒ `opening_incompatible`；抓取深度使桿心 z 落在 6 或 30 mm ⇒ `depth_incompatible`；目標推入面板 ⇒ `penetration_at_target`；
   雜湊改 1 byte ⇒ `contract_version_mismatch`；未換算的 H' 座標 ⇒ `handle_frame_mismatch`。
4. 距離方法：以解析可解的方塊／圓柱配對核對精確距離（≤ 1e-6 m）；並與既知「點雲高報」案例對照，確認不高報。
5. 非法輸入回 dict 不拋例外（沿用 DL1 模糊測試形式）。

## 4　保留未解（不在本包處理、不得因本包宣稱已解）

- 收攏姿態：相機框水平半徑超出 costmap 5.1 mm（使用者裁定擱置）；收攏自碰餘裕 2.64 mm 未以精確三角測試驗證（`wgmpc_drawer_open_close.yaml` stow_posture.open_items）。
- 手指行程 17.8 mm vs 廠商 16 mm、實體閉合指片間距未量——契約只代表**官方模型**幾何。
- 夾爪殼（`uflite_gripper_link` 凸包）與手臂連桿對物體的碰撞、展開／接近軌跡逐點碰撞、閉爪後接觸力學、旋轉擺放的 USD 建置與實跑。

## 5　待審

1. 範圍是否足夠（只做開口、深度、共同變換、目標處穿透、座標約定；夾爪殼與軌跡不納入）。
2. 資產檔 `grasp_surface` 過時欄位：契約不讀並列名即可，或須另立資產版本（schema 升版）處理。
3. 根部塊 1.5 mm：只回報，或須訂拒絕門檻（若須訂，門檻來源？本包傾向不新訂）。
4. 精確距離實作選擇（自寫 GJK vs scipy 凸 QP；不新增系統套件）。

---

# draft-2（依 Codex GC1 規格審查四項必修與 §5 裁定；名稱改為「手指—抽屜靜態局部幾何契約」）

範圍確認：只做手指（split 兩凸塊）對抽屜櫃的靜態局部幾何；不是完整目標放行驗收。夾爪殼、手臂、軌跡維持未驗收。

### 1　座標與共同變換（取代 draft-1 §2.3）
- `O`＝櫃體座標（資產本地框），`T_WO` 為其世界位姿；`q`＝抽屜開度（公尺，`[lower, upper]`），`u_O`＝資產 `joint.axis`。
- 櫃體方塊：`T_WO · T_OS`；抽屜本體、支柱、橫桿：`T_WO · Trans(q·u_O) · T_OS`。
- 把手座標 **由契約重建**：`T_WH = T_WO · Trans(q·u_O) · Trans(c_bar)`（DL1 資產 prim 約定，旋轉＝R_WO）。核對器不接受「已含開度」的外部把手位姿再平移。
- 手指：同一版 `finger_collision.split_hulls`（Z_SPLIT 0.008）作用於契約鎖定的 STL 頂點；連桿變換由契約鎖定的 URDF 讀出
  （`joint_tcp` 0.0836、`finger_joint1/2` 原點 0.0543、軸 ±y、mimic）：`T_W,f_i = T_WE · T_tcp,glink · T_glink,f_i(q_f)`，不另畫替代形狀。

### 2　抓取相容與接觸前目標分開（取代 draft-1 §2.1、2.2、2.4）
- 輸入保留 **`T_HG`、`a_H`、`s`、`T_EG`**（不只一個工具矩陣）；目標由 DL1 `object_target` 產生。
- **抓取相容只在預定抓取姿態 s = 0 核對**：開口相容（指片閉合間距取自 split 凸塊在接觸高度的內面、接觸手指位置 q_c ∈ (0, 0.0089)、每側全開餘裕 ≥ 資產 `enclosure_margin_min_m` 0.002）；
  深度相容（桿心在手指座標 z ∈ 指片段 z 範圍；桿最近點 z > 根部塊 z 上界；**根部距離只回報**）；橫桿軸與手指閉合軸垂直（|cos| ≤ 1e-6）。
- **s > 0 的接觸前目標只核當下碰撞**，不要求桿進入指片（實錄旋轉下 s = 30 mm 時桿心在手指 z ≈ 52.5 mm 屬正常）。

### 3　凸模型的數值距離與交疊核對（取代 draft-1「最小距離 ≥ 0」）
- 每一配對分記 **`separated／contact／penetrating／indeterminate`**：
  - 交疊：兩凸多面體半空間式的 Chebyshev LP（`scipy.optimize.linprog`, HiGHS）求交集內切半徑 t*；t* > 1e-6 m ⇒ penetrating。
  - 距離：凸 QP（SLSQP）求最近點，**不憑 success 放行**：以最近點差向量 n 做分離平面驗證，下界＝min_A n·a − max_B n·b、上界＝‖p_A − p_B‖（可行點），
    間隙 > 1e-7 m ⇒ indeterminate。
  - 有限圓柱：以內接／外切 256 邊稜柱夾擠（半徑差 ≈ 1 µm）——外切稜柱分離 ⇒ separated（距離區間）；內接稜柱交疊 ⇒ penetrating；其餘依區間判定，無法判定 ⇒ indeterminate。
  - 判定：距離下界 > 1e-6 ⇒ separated；距離上界 ≤ 1e-6 且 t* ≤ 1e-6 ⇒ contact；非有限、LP 失敗、證書不閉合 ⇒ indeterminate。
- 放行：非設計接觸形狀（櫃體、抽屜本體、支柱）出現 penetrating 或 indeterminate ⇒ 拒絕；contact 回報不拒絕（不新訂門檻）。
  橫桿（設計接觸對）開爪時 penetrating 或 indeterminate ⇒ 拒絕；距離只回報。

### 4　座標約定與真值一致性分開（取代 draft-1 §2.5）
- `handle_frame_mismatch`：只針對**宣告的座標約定**（輸入須標 `frame='asset_prim'`；其他標記須先經 `reparam_grasp` 換算）與變換鏈。
- `offline_input_inconsistent`：僅離線使用——若另給外部 `T_WH`，與契約重建值比對（旋轉 1e-9 rad、平移 1e-9 m）；**不是**未來感知估計的真值放行閘門。
- 未換算的 H′ 參數另由 §2 的幾何相容（深度／垂直）攔下。

### 5　過時欄位（不改資產、不升 schema；精確列名）
- 不採用：`drawer_unit_bar26.yaml` `grasp_surface.tcp_offset_along_tool_z`（0.0147）、`grasp_surface.finger_gap_open_m`（0.0178）及其註解；
  `drawer_task_node.py` `--grasp-depth-m` 預設 0.01226 與 L462、L469 的「12.26 mm」來源字串（實錄以 `task.json args.grasp_depth_m` 為準）。
- 仍採用：`grasp_surface.finger_joint_open`（0.0089，與 URDF 上限一致）、`grasp_surface.enclosure_margin_min_m`（0.002）、`handle.bar`、`handle.posts`、`cabinet`、`drawer.body`、`drawer.joint`。

### 6　測試補充
基準 s = 0 與 s = 0.03 通過；回報值重現 1.8／7.1／1.5 mm（容差 0.05 mm，網格頂點為 float32）；含 yaw 的剛體等變（距離差 ≤ 2e-7 m，受證書容差限制）；
反例：桿徑 20／44 mm、桿心 z 6／30 mm、抓在支柱上（穿透支柱）、雜湊改 1 byte、`frame` 標記錯、外部 T_WH 不一致、櫃體不隨開度動的核對（q 變動時櫃體形狀不動）；
距離方法對解析方塊／方塊、點／圓柱案例核對（≤ 1e-6 m），交疊案例回 penetrating 而非距離 0。

通過時正式說法：**bar26＋split 手指在指定開爪目標下，通過局部靜態幾何契約核對；不代表整機碰撞安全、軌跡可達或物理夾持成功。**

---

# draft-3（依 Codex GC1 實作審查三項必修；**以本節為準，取代前文衝突處**）

### 取代／刪除前文條目
- draft-2 §2「橫桿軸與手指閉合軸垂直（|cos| ≤ 1e-6）」**刪除**：改為逐手指沿閉合方向二分到首次接觸（內接／外切稜柱各一次）。
  實錄抓取（IK 殘差）的橫桿軸與閉合方向 |cos| = 1.14×10⁻⁵（舊條件會誤拒）；相對夾爪 x 軸的傾角約 5.38×10⁻⁵ rad（0.0031°），只回報。每側餘裕的意義是**沿閉合方向的剩餘行程**（q_open − q_contact），不是一般法向淨空。
- draft-1 §3.1／draft-2 §6「回報值重現 1.5 mm」**刪除**：1.5 mm 是資產檔頭以名目分界 8 mm 推得；模型值（根部凸塊頂 6.8 mm）沿接近軸 2.7 mm，只回報。
- draft-2 §6「抓在支柱上（穿透支柱）」反例**刪除**：支柱高 24 mm < 桿徑 26 mm，靜態不碰；穿透反例改為向內退讓 50 mm。
- draft-2 §4「未換算的 H′ 參數另由 §2 的幾何相容攔下」**刪除**：未換算參數在 s = 0 等於繞橫桿軸翻 180° 的抓取，只含手指的契約判為相容；
  只有退讓變向內時被攔下。**夾爪殼未核對，本包不能排除此碰撞**（GC2 處理）。

### 狀態語意與數值容差
- `contact`＝**容差內接觸或微小交疊**：距離上界 ≤ 1e-6 m 且 Chebyshev 交集內切半徑 t* ≤ 1e-6 m；不是嚴格零穿透（方塊交疊 1 µm 仍判 contact，已列測試）。t* 不等於穿透深度。
- `penetrating`：t* > 1e-6 m。`separated`：證書距離下界 > 1e-6 m。其餘（LP 失敗、約束殘差 > 1e-9、證書間隙 > 1e-7 m、介於兩容差之間、圓柱夾擠不閉合）＝`indeterminate`。
- 圓柱以 256 邊內接／外切稜柱夾擠，半徑差約 1 µm：真圓柱的微小交疊在此量級內可能判為 contact。

### gap_guard（離線保守補查，納入放行）
- split 兩凸塊**沒有完整包含手指網格**：(a) 分界帶 z 6.8–8.3 mm 的跨界三角形不屬任一塊；(b) 長三角形在 z 6.8／8.3 截面上的部分落在兩凸塊外，最多 2.65 mm（根部側）／2.04 mm（指片側），位置在手指外側（y 11.3–15.3 mm、倒角處），不在夾持內面。
- 做法：所有三角形以 Sutherland–Hodgman 裁成三段（z ≤ z0、[z0, z1]、z ≥ z1；原頂點＋邊界交點），各段取凸包 ⇒ 三段聯集包含整個網格。載入時核對每段完整包含其裁切幾何，否則拒絕載入；測試另核三段面積和（扣回恰在分界平面上的三角形）＝網格面積。
- 結果分存 `pairs`（split_model，模擬所用形狀）與 `gap_guard`；任一出現 penetrating／indeterminate 即不給整體通過。不修改 `finger_collision`、不重跑歷史趟次；這不是宣稱模擬已有此碰撞形狀。
- draft-2 的舊建法（只取兩個相鄰頂點層的凸包）不保守：測試保留為反例（帶內裁切點落在其外 > 1 mm）。

### 契約完整性
- 必要欄位與型別：`schema`、`contract_id`（str）、`handle_frame`（str）、`z_split_m`（float，須等於鎖定 `finger_collision.Z_SPLIT_M`）、`files`（dict，**恰為七項**、每項 `path` 須等於預定路徑、`sha256` 為 64 字元）。缺項、多項、路徑替換 ⇒ `contract_incomplete:*`；雜湊不符 ⇒ `contract_version_mismatch:*`。
- `check()` 只接受 `load_contract` 產出的完整結構；手造 `{'ok': True}` 或缺鍵 ⇒ `contract_incomplete:*`，不拋例外。
