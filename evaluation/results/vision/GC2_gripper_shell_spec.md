# GC2：最小夾爪殼靜態檢查——小規格（draft-1，送審；核可前不實作）

2026-10-06。接續 GC1 封存（`freeze_gc1_offline.sha256`）。GC1 只核手指；未換算 H′ 的翻轉抓取在 s = 0 對手指相容，**夾爪殼未核對，不能排除碰撞**。
本包只補「夾爪殼對抽屜櫃」的靜態幾何，沿用 GC1 的契約、共同變換、凸體核對與四類狀態；不擴手臂、軌跡、模擬，不改資產或 GC1 凍結檔。

## 0　現況核對

| 項目 | 來源 | 值 |
|---|---|---|
| 夾爪殼 | URDF `uflite_gripper_link` collision＝`gripper/lite/visual/shell.stl`，原點與 `link_eef` 重合（`gripper_fix` 零位移） | 4406 頂點；夾爪連桿座標 x、y ∈ [−38.8, 38.8] mm、z ∈ [0, 56.7] mm |
| 相機支架 | URDF `link_eef` collision＝`camera/realsense/collision/d435_with_cam_stand.stl`（同一剛體） | 96 頂點；x [−37.7, 80.4]、y [−45.5, 45.9]、z [0, 28.0] mm |
| 模擬近似 | 實錄 `grasp_friction_readback` 只讀回手指（convexHull）與橫桿；**殼與相機支架未讀回** | 推定與手指同為匯入器預設 convexHull（未證實） |
| 基準幾何 | 手指原點在夾爪連桿 z 54.3 mm；基準桿心 z 76.8 mm、桿最近點 63.8 mm | 殼頂 56.7 mm ⇒ 沿接近軸距桿 7.1 mm（只回報） |

## 1　形狀與核對（新檔 `evaluation/gripper_shell_check.py`，呼叫 GC1 凍結函式，不改 GC1）

- 殼以**整個網格的凸包**核對：凸包 ⊇ 網格，故 separated 對網格保守成立；且與推定的模擬近似（convexHull）相同。
- 變換：`T_W,gl = T_WE · T_tcp,gl`（GC1 同一 URDF 鏈）；物體形狀沿用 GC1 `object_shapes`（櫃體不隨開度、抽屜隨開度）。
- 對 GC1 的 14 個物體形狀逐一分類（separated／contact／penetrating／indeterminate），penetrating 或 indeterminate ⇒ 拒絕 `shell_penetration_at_target:<tag>:<shape>`。
  殼對橫桿也一樣（殼不是設計接觸面）。contact 回報不拒絕（沿用 GC1 語意：容差內接觸或微小交疊）。
- 契約：新契約檔 `geometry_contract_gc2_shell.yaml` 另鎖 `shell.stl`（及第 5 節若納入的相機支架網格）sha256，並鎖 GC1 凍結清單本身的 sha256；GC1 契約檔不改。

## 2　目標（只三類，與 Codex 指定一致）

| 標記 | 定義 | 預期 |
|---|---|---|
| `base` | phf_01_M 抓取，s = 0 | 殼在桿外側（遠離面板），預期 separated |
| `retreat30` | 同上，s = 0.03 | 預期 separated |
| `flipped` | 未換算 H′ 參數（R_HH′ = diag(1,−1,−1)）當成資產約定使用，s = 0 | 殼位於桿與面板之間：殼沿接近軸佔桿心後方約 20–77 mm，面板距桿心 40 mm ⇒ **預期 penetrating（面板）** ⇒ 拒絕 |

GC1 手指核對照常執行；GC2 輸出與 GC1 輸出分開保存（`fingers`／`shell`），兩者都通過才算整體通過。

## 3　離線測試
1. 三類目標的預期結果；`flipped` 的拒絕原因須指名殼與面板。
2. 殼凸包包含網格全部頂點（殘差 ≤ 1e-9）。
3. 含 yaw 的剛體等變（同 GC1，距離差 ≤ 2e-7 m）。
4. 契約：殼網格改 1 byte（暫存複本）⇒ 版本不符；GC1 凍結清單不符 ⇒ 拒絕。
5. 非法輸入回 dict 不拋例外。

## 4　不做／不宣稱
不含手臂連桿、展開／接近軌跡、閉爪力學、旋轉擺放的 USD 建置；殼的模擬近似未讀回，結論只對「網格凸包」成立。
通過時說法：**基準與 30 mm 退讓目標下，夾爪殼凸包與抽屜櫃無交疊；翻轉抓取被殼核對攔下。不代表整機碰撞安全、軌跡可達或物理夾持成功。**

## 5　待審
1. 相機支架（同一剛體、x 伸到 80.4 mm）是否納入本包，或維持範圍外只揭露。我傾向納入（同樣是凸包、同一變換，成本低；翻轉抓取時也在面板側）。
2. 殼的模擬近似未讀回：本包只對凸包下結論是否足夠，或須另做一次讀回（需開模擬器，本包不做）。
3. `flipped` 只取 R_HH′ = diag(1,−1,−1) 一例是否足夠。

---

# draft-2（依 Codex GC2 規格審查三項；以本節為準）

1. **範圍改稱「夾爪殼＋腕部相機／支架」**：納入 URDF `link_eef` collision 的整個 `d435_with_cam_stand.stl`。各形狀的變換由各自連桿、固定關節
   （`gripper_fix`）與 collision origin／scale 從 URDF 導出，不用相機光學座標。結果分存 `fingers`（GC1）／`shell`／`camera_assembly`，三者全部通過才整體通過。
2. **刪除「與推定的模擬近似相同」**：殼與相機的模擬近似未讀回；本包只對網格凸包下結論。凸包分離 ⇒ 指定姿態下網格也分離；
   **凸包交疊只代表保守核對拒絕，不證明原網格或實際模擬一定碰撞**。三類目標的「預期」都是待驗證假設，結果依實測報告。
3. **相依驗證**：載入時除 GC1 凍結清單本身的雜湊外，**逐項重算清單內每個檔案**；任一不符 ⇒ `gc1_freeze_entry_mismatch:<path>`。
   拒絕原因分開：`penetration_at_target:<目標>:<元件>|<物體>` 與 `indeterminate_at_target:<目標>:<元件>|<物體>`。
4. 只做一個翻轉案例（R_HH′ = diag(1,−1,−1)），不宣稱已驗證所有翻轉或對稱抓取。

驗收用語：在指定基準與退讓目標下，夾爪殼及腕部相機／支架通過凸包靜態核對；指定翻轉案例是否被拒絕，依實測核對結果報告。不代表整機碰撞安全、軌跡可達或物理夾持成功。
